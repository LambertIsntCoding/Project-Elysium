"""Tests for the runtime/code separation, version info, and supervisor.

These exercise the infrastructure against real files on disk and a real child
process (a tiny Python script), so nothing here is mocked at the level of the
code under test. The store, the deployment record, and the control files are all
real.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from astra import paths, version
from astra.supervisor import RuntimeSupervisor, SingleInstanceLock


class _RuntimeFixture(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.runtime = os.path.join(self.root, "runtime")
        os.environ[paths.RUNTIME_ENV] = self.runtime

    def tearDown(self):
        os.environ.pop(paths.RUNTIME_ENV, None)
        shutil.rmtree(self.root, ignore_errors=True)


class TestPathSeparation(_RuntimeFixture):
    def test_runtime_root_is_outside_source(self):
        self.assertNotEqual(
            os.path.commonpath([paths.runtime_root(), paths.source_root()]),
            paths.source_root())

    def test_storage_separate_from_source_storage(self):
        self.assertNotEqual(paths.storage_dir(), paths.legacy_storage_dir())

    def test_seed_never_overwrites_and_leaves_source(self):
        source = os.path.join(self.root, "old_storage")
        os.makedirs(source)
        with open(os.path.join(source, "self_model.json"), "w") as fh:
            fh.write('{"version": 1}')
        with open(os.path.join(source, "presence.json"), "w") as fh:
            fh.write('{"last_seen": "x"}')

        target = os.path.join(self.runtime, "storage")
        first = paths.seed_runtime_storage(target=target, source=source)
        self.assertIn("self_model.json", first)
        # Simulate accumulated state, then re-seed: it must NOT be overwritten.
        with open(os.path.join(target, "self_model.json"), "w") as fh:
            fh.write('{"version": 99, "grew": true}')
        paths.seed_runtime_storage(target=target, source=source)
        with open(os.path.join(target, "self_model.json")) as fh:
            self.assertIn("grew", fh.read())
        # Source is untouched.
        self.assertTrue(os.path.exists(os.path.join(source, "self_model.json")))

    def test_library_is_a_tracked_fixture(self):
        self.assertTrue(paths.is_tracked_fixture("library/books/x.txt"))
        self.assertTrue(paths.is_tracked_fixture("library/anne.txt"))
        self.assertFalse(paths.is_tracked_fixture("roum_model.json"))
        self.assertFalse(paths.is_tracked_fixture("presence.json"))


class TestVersionInfo(_RuntimeFixture):
    def test_start_and_status_detect_change(self):
        info = version.RuntimeInfo()
        info.record_start(revision={"commit": "abc123", "branch": "main",
                                    "describe": "abc123"})
        status = info.status()
        self.assertEqual(status["running_commit"], "abc123")
        self.assertIn("started_at", status)

    def test_code_changed_flag(self):
        info = version.RuntimeInfo()
        info.record_start(revision={"commit": "old", "branch": "main",
                                    "describe": "old"})
        # Force the on-disk view to differ by asking status to use a stub file.
        original = version.git_revision
        try:
            version.git_revision = lambda: {"commit": "new", "branch": "main",
                                            "describe": "new"}
            status = info.status()
            self.assertTrue(status["code_changed_since_start"])
        finally:
            version.git_revision = original

    def test_restart_count_and_recent_flag(self):
        info = version.RuntimeInfo()
        info.record_start(revision={"commit": "a", "branch": "", "describe": ""})
        info.record_start(revision={"commit": "a", "branch": "", "describe": ""})
        status = info.status()
        self.assertEqual(status["restart_count"], 1)
        self.assertTrue(status["restarted_recently"])


class TestSingleInstanceLock(_RuntimeFixture):
    def test_second_acquire_fails_while_held(self):
        path = os.path.join(self.runtime, "lock")
        first = SingleInstanceLock(path)
        self.assertTrue(first.acquire())
        second = SingleInstanceLock(path)
        self.assertFalse(second.acquire())
        first.release()
        third = SingleInstanceLock(path)
        self.assertTrue(third.acquire())
        third.release()

    def test_stale_lock_from_dead_process_is_reclaimed(self):
        path = os.path.join(self.runtime, "lock")
        os.makedirs(self.runtime, exist_ok=True)
        with open(path, "w") as fh:
            fh.write("999999999")  # a PID that is not alive
        lock = SingleInstanceLock(path)
        self.assertTrue(lock.acquire())
        lock.release()


class TestSupervisor(_RuntimeFixture):
    def _script(self, body: str) -> str:
        path = os.path.join(self.root, "fake_runtime.py")
        with open(path, "w") as fh:
            fh.write(body)
        return path

    def test_graceful_restart_via_control_file(self):
        # A runtime that runs until a restart file appears, then exits 0.
        control = paths.control_dir()
        script = self._script(
            "import os, time, sys\n"
            f"control = {control!r}\n"
            "while True:\n"
            "    if os.path.exists(os.path.join(control, 'restart.request')):\n"
            "        sys.exit(0)\n"
            "    time.sleep(0.05)\n"
        )
        sup = RuntimeSupervisor(command=[sys.executable, script],
                                runtime_dir=self.runtime, poll_seconds=0.05)
        # Request a restart after a moment, from a helper thread.
        import threading
        def _later():
            time.sleep(0.4)
            sup.request_restart()
            time.sleep(0.4)
            sup.request_stop()
        threading.Thread(target=_later, daemon=True).start()
        result = sup.run(max_restarts=5)
        self.assertTrue(result["started"])
        self.assertGreaterEqual(result["restarts"], 1)
        self.assertTrue(os.path.exists(sup.log_path))

    def test_crash_is_restarted_then_budget_stops_it(self):
        # A runtime that exits immediately (a crash-on-start loop).
        script = self._script("import sys; sys.exit(3)\n")
        sup = RuntimeSupervisor(command=[sys.executable, script],
                                runtime_dir=self.runtime, poll_seconds=0.05)
        # Keep backoff tiny for the test so it does not sleep for real.
        sup._backoff = 0.05
        result = sup.run(max_restarts=2)
        self.assertEqual(result["reason"], "restart-budget")
        self.assertGreaterEqual(result["restarts"], 2)

    def test_second_supervisor_refuses_to_start(self):
        script = self._script("import time\nwhile True: time.sleep(0.05)\n")
        sup = RuntimeSupervisor(command=[sys.executable, script],
                                runtime_dir=self.runtime, poll_seconds=0.05)
        lock = SingleInstanceLock(os.path.join(paths.runtime_root(), "supervisor.lock"))
        self.assertTrue(lock.acquire())
        try:
            result = sup.run()
            self.assertFalse(result["started"])
            self.assertEqual(result["reason"], "already-running")
        finally:
            lock.release()


if __name__ == "__main__":
    unittest.main(verbosity=2)
