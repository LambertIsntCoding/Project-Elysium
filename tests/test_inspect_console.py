"""Tests for the read-only inspection console and the live-screen refresh fix.

These exercise the real :class:`astra.inspect_console.Inspector` against a real
session/controller built by ``build_session`` (a temp store, no network). The
model backend is stubbed only where a test proves it is never reached.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import main
from astra import admin as admin_mod
from astra import inspect_console


class _InspectFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = os.path.join(self.tmp, "storage")
        os.makedirs(self.storage)
        self.config = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config")
        self.session = main.build_session(
            config_dir=self.config, storage_dir=self.storage,
            enable_reader=False)
        # A real model is never needed by inspection; make a call fail loudly so
        # any accidental model use is caught by a test rather than hitting a
        # network.
        self.session.orchestrator.query_gemma = (
            lambda *a, **k: self.fail("inspection touched the model"))
        self.inspector = inspect_console.Inspector(
            self.session, storage_dir=self.storage, config_dir=self.config)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _storage_snapshot(self):
        out = {}
        for root, _dirs, files in os.walk(self.storage):
            for name in files:
                path = os.path.join(root, name)
                try:
                    out[path] = os.path.getmtime(path)
                except OSError:
                    pass
        return out


class TestInspectSnapshot(_InspectFixture):
    def test_snapshot_has_expected_sections_and_disclaims_chat(self):
        text = self.inspector.snapshot()
        self.assertIn("READ-ONLY INSPECTION", text)
        self.assertIn("cannot talk to her or change anything", text)
        self.assertIn("Autonomy:", text)
        self.assertIn("Discord:", text)

    def test_diagnostics_structure_is_truthful(self):
        diag = self.inspector.diagnostics()
        self.assertIn("runtime", diag)
        self.assertIn("running", diag["runtime"])
        self.assertIn("autonomy", diag)
        self.assertIn("notifications", diag)
        self.assertIn("tasks", diag)

    def test_runtime_detection_reads_the_lock_file(self):
        # Our own pid is alive, so a lock naming it reads as running.
        lock = os.path.join(self.tmp, "runtime.lock")
        with open(lock, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        self.inspector.lock_path = lock
        self.assertTrue(self.inspector.runtime_status()["running"])
        # A pid that cannot exist reads as not running.
        with open(lock, "w", encoding="utf-8") as handle:
            handle.write("999999999")
        self.assertFalse(self.inspector.runtime_status()["running"])
        os.remove(lock)
        self.assertFalse(self.inspector.runtime_status()["running"])


class TestInspectIsReadOnly(_InspectFixture):
    def test_no_read_only_command_writes_to_storage(self):
        commands = [
            "/report", "/tasks", "/schedules", "/scheduler", "/autonomy",
            "/notifications", "/proactive", "/power", "/proposals", "/decisions",
            "/sleep-log", "/status", "/memories", "/affect", "/relationship",
            "/self", "/experiences", "/questions", "/reading", "/library",
            "/time", "/help",
        ]
        before = self._storage_snapshot()
        for command in commands:
            self.inspector.handle(command)  # must not raise
        after = self._storage_snapshot()
        changed = {p for p in set(before) | set(after)
                   if before.get(p) != after.get(p)}
        self.assertEqual(changed, set(), f"inspection wrote files: {changed}")

    def test_mutating_commands_are_refused(self):
        for command in ("/sleep", "/wake", "/autonomy pause", "/autonomy resume",
                        "/notifications mute", "/proactive unmute", "/read force",
                        "/approve abc", "/reconcile"):
            text = self.inspector.handle(command)
            self.assertIn("not available in the read-only console", text)

    def test_conversation_is_never_routed(self):
        # A plain sentence must not become a turn: the session's conversational
        # path and the model call both fail loudly if reached.
        self.session.handle = lambda *a, **k: self.fail("conversation was routed")
        text = self.inspector.handle("hello Astra, how are you?")
        self.assertIn("Unknown inspection command", text)

    def test_router_entry_point_is_not_used(self):
        # The router's handle() records a human message as a side effect; the
        # console must call the read-only reports directly instead.
        self.session.router.handle = lambda *a, **k: self.fail("router.handle used")
        self.assertIn("Autonomy:", self.inspector.handle("/autonomy"))
        self.assertIn("Scheduler:", self.inspector.handle("/scheduler"))

    def test_reading_pause_command_is_not_executed(self):
        reader = getattr(self.session, "reader", None)
        before = getattr(reader, "enabled", None)
        self.inspector.handle("/read pause")
        self.assertEqual(getattr(reader, "enabled", before), before)


class TestLiveScreenRefresh(unittest.TestCase):
    def test_poll_timeout_returns_none_and_redraws(self):
        frames = {"n": 0}

        def render():
            frames["n"] += 1
            return [f"frame {frames['n']}"]

        screen = admin_mod.LiveScreen("live", render=render,
                                      store=admin_mod.SnapshotStore(
                                          data_dir=tempfile.mkdtemp()),
                                      refresh_seconds=0.01)
        emitted = []
        clears = {"n": 0}
        # Two timeouts (auto-refresh), then leave.
        keys = iter([None, None, "q"])
        screen.run(poll=lambda t: next(keys, "q"), emit=emitted.append,
                   sleep=lambda t: None,
                   clear=lambda: clears.__setitem__("n", clears["n"] + 1))
        text = "\n".join(emitted)
        self.assertIn("frame 3", text)          # refreshed on its own cadence
        self.assertGreaterEqual(clears["n"], 2)  # redrew in place, not appended
        self.assertIn("Left live", text)

    def test_session_screen_poll_returns_none_without_a_tty(self):
        # Under the test runner stdin is not a tty; the poll must NOT block on
        # input() - it returns None so the screen refreshes on its own.
        session = main.ChatSession.__new__(main.ChatSession)
        with mock.patch("sys.stdin") as fake:
            fake.isatty.return_value = False
            result = main.ChatSession._screen_poll(session, 0.0)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
