"""Tests for the self-update engine (``/update``) and the control channel.

The Git command is injected, so the whole flow - a clean fast-forward, a dirty
tree, an unreachable remote, a diverged branch, a hard reset - is exercised
without a network or a repository. The control channel is exercised against a
temp runtime directory with no running process.
"""

import os
import tempfile
import unittest

from astra import control, paths, update, version


class _FakeGit:
    """A scriptable stand-in for the git subprocess runner."""

    def __init__(self, *, head="aaa111", remote="aaa111", dirty=(),
                 fetch_ok=True, merge_ok=True, behind=0):
        self.head = head
        self.remote = remote
        self.dirty = list(dirty)
        self.fetch_ok = fetch_ok
        self.merge_ok = merge_ok
        self.behind = behind
        self.calls = []

    def __call__(self, args, timeout=60.0):
        self.calls.append(list(args))
        cmd = args[0]
        if cmd == "status":
            return (0, "\n".join(self.dirty), "")
        if cmd == "fetch":
            return (0 if self.fetch_ok else 1, "", "" if self.fetch_ok else "no net")
        if cmd == "rev-parse":
            ref = args[-1]
            if ref == "HEAD":
                return (0, self.head, "")
            if ref == "origin/main":
                return (0, self.remote, "")
            return (1, "", "unknown ref")
        if cmd == "rev-list":
            return (0, str(self.behind), "")
        if cmd in ("merge", "reset"):
            if not self.merge_ok:
                return (1, "", "merge conflict")
            self.head = self.remote
            return (0, "", "")
        return (1, "", "unexpected: " + " ".join(args))


class TestUpdateCheck(unittest.TestCase):
    def test_up_to_date_is_reported(self):
        engine = update.UpdateEngine(git_runner=_FakeGit(head="a", remote="a"))
        status = engine.check()
        self.assertTrue(status["up_to_date"])
        self.assertFalse(status["can_update"])

    def test_behind_is_reported_with_a_count(self):
        engine = update.UpdateEngine(
            git_runner=_FakeGit(head="a", remote="b", behind=3))
        status = engine.check()
        self.assertFalse(status["up_to_date"])
        self.assertTrue(status["can_update"])
        self.assertEqual(status["behind"], 3)

    def test_dirty_tree_blocks_update(self):
        engine = update.UpdateEngine(
            git_runner=_FakeGit(head="a", remote="b", dirty=[" M main.py"]))
        status = engine.check()
        self.assertFalse(status["can_update"])
        self.assertEqual(status["dirty"], ["M main.py"])

    def test_unreachable_remote_is_honest(self):
        engine = update.UpdateEngine(
            git_runner=_FakeGit(head="a", remote="b", fetch_ok=False))
        status = engine.check()
        self.assertFalse(status["fetched"])
        self.assertFalse(status["can_update"])

    def test_report_text_reflects_status(self):
        engine_report = update.update_report(
            git_runner=_FakeGit(head="a", remote="b", behind=2))
        self.assertIn("CODE UPDATE", engine_report)
        self.assertIn("2 commit(s) behind", engine_report)


class TestUpdatePull(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _engine(self, **kwargs):
        return update.UpdateEngine(git_runner=_FakeGit(**kwargs),
                                   runtime_dir=self.tmp)

    def test_fast_forward_updates_and_requests_restart(self):
        engine = self._engine(head="a", remote="b")
        result = engine.pull(restart=False, wait=False)
        self.assertTrue(result["updated"])
        self.assertEqual(result["mode"], "ff-only")
        self.assertEqual(result["previous_commit"], "a")
        self.assertEqual(result["new_commit"], "b")
        self.assertTrue(result["restart_requested"])
        # The restart request is the shared control file.
        self.assertTrue(os.path.exists(
            os.path.join(paths.control_dir(self.tmp), "restart.request")))

    def test_dirty_tree_refuses_and_changes_nothing(self):
        fake = _FakeGit(head="a", remote="b", dirty=[" M astra/machine.py"])
        engine = update.UpdateEngine(git_runner=fake, runtime_dir=self.tmp)
        result = engine.pull(restart=False, wait=False)
        self.assertFalse(result["updated"])
        self.assertEqual(result["reason"], "dirty-tree")
        # No fetch/merge was attempted once the tree was seen dirty.
        self.assertFalse(any(c and c[0] == "merge" for c in fake.calls))

    def test_failed_fetch_is_non_destructive(self):
        fake = _FakeGit(head="a", remote="b", fetch_ok=False)
        engine = update.UpdateEngine(git_runner=fake, runtime_dir=self.tmp)
        result = engine.pull(restart=False, wait=False)
        self.assertFalse(result["updated"])
        self.assertEqual(result["reason"], "fetch-failed")
        self.assertEqual(fake.head, "a")  # nothing moved

    def test_up_to_date_pull_is_a_noop(self):
        engine = self._engine(head="a", remote="a")
        result = engine.pull(restart=False, wait=False)
        self.assertFalse(result["updated"])
        self.assertEqual(result["reason"], "up-to-date")

    def test_diverged_branch_is_refused_ff_only(self):
        fake = _FakeGit(head="a", remote="b", merge_ok=False)
        engine = update.UpdateEngine(git_runner=fake, runtime_dir=self.tmp)
        result = engine.pull(mode="ff-only", restart=False, wait=False)
        self.assertFalse(result["updated"])
        self.assertEqual(result["reason"], "merge-failed")
        self.assertEqual(fake.head, "a")  # HEAD unchanged

    def test_reset_mode_hard_resets_to_remote(self):
        fake = _FakeGit(head="a", remote="b", merge_ok=True)
        engine = update.UpdateEngine(git_runner=fake, runtime_dir=self.tmp)
        result = engine.pull(mode="reset", restart=False, wait=False)
        self.assertTrue(result["updated"])
        self.assertEqual(result["mode"], "reset")
        self.assertTrue(any(c[:2] == ["reset", "--hard"] for c in fake.calls))

    def test_deploy_is_recorded(self):
        engine = self._engine(head="a", remote="b")
        engine.pull(restart=False, wait=False)
        status = version.RuntimeInfo(
            deploy_dir=os.path.join(self.tmp, "deploy")).status()
        self.assertIn("update.py", status.get("deploy_note", ""))


class TestUpdateCommandText(unittest.TestCase):
    def test_status_word_reports(self):
        class _C:
            def update_report(self, *, fetch=True):
                return "REPORT"
            def apply_update_from_runtime(self, *, mode):
                return f"APPLIED {mode}"
        self.assertEqual(update.command_text(_C(), []), "REPORT")
        self.assertEqual(update.command_text(_C(), ["check"]), "REPORT")

    def test_apply_and_reset_route_to_runtime_local(self):
        class _C:
            def update_report(self, *, fetch=True):
                return "REPORT"
            def apply_update_from_runtime(self, *, mode):
                return f"APPLIED {mode}"
        self.assertEqual(update.command_text(_C(), ["apply"]), "APPLIED ff-only")
        self.assertEqual(update.command_text(_C(), ["reset"]), "APPLIED reset")

    def test_unknown_word_shows_usage(self):
        class _C:
            def update_report(self, *, fetch=True):
                return "REPORT"
        self.assertIn("Usage: /update", update.command_text(_C(), ["wat"]))


class TestControlChannel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_submit_and_take_pending_round_trip(self):
        rid = control.submit("/status", runtime_dir=self.tmp)
        pending = control.take_pending(runtime_dir=self.tmp)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["id"], rid)
        self.assertEqual(pending[0]["text"], "/status")
        # Consumed once.
        self.assertEqual(control.take_pending(runtime_dir=self.tmp), [])

    def test_result_round_trip_and_consume(self):
        rid = control.submit("/status", runtime_dir=self.tmp)
        control.write_result(rid, "Astra is active.", runtime_dir=self.tmp)
        result = control.poll(rid, runtime_dir=self.tmp)
        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "Astra is active.")
        # Consumed once.
        self.assertIsNone(control.poll(rid, runtime_dir=self.tmp))

    def test_wait_for_result_times_out_when_absent(self):
        rid = control.submit("/status", runtime_dir=self.tmp)
        self.assertIsNone(control.wait_for_result(
            rid, timeout=0.1, interval=0.02, runtime_dir=self.tmp))

    def test_wait_for_result_returns_when_present(self):
        rid = control.submit("/status", runtime_dir=self.tmp)
        control.write_result(rid, "ok", runtime_dir=self.tmp)
        result = control.wait_for_result(
            rid, timeout=1.0, interval=0.02, runtime_dir=self.tmp)
        self.assertEqual(result["text"], "ok")

    def test_runtime_listening_without_a_lock_is_false(self):
        self.assertFalse(control.runtime_listening(self.tmp))

    def test_runtime_listening_with_our_pid_is_true(self):
        with open(os.path.join(self.tmp, "runtime.lock"), "w",
                  encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        self.assertTrue(control.runtime_listening(self.tmp))


if __name__ == "__main__":
    unittest.main()
