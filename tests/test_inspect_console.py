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
        self.assertIn("INSPECTION CONSOLE", text)
        self.assertIn("read-only view of the state on disk", text)
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

    def test_mutating_commands_are_refused_without_a_runtime(self):
        # No runtime is listening in this fixture, so mutating commands are
        # refused rather than silently doing nothing.
        for command in ("/sleep", "/wake", "/autonomy pause", "/autonomy resume",
                        "/notifications mute", "/proactive unmute", "/read force",
                        "/approve abc", "/reconcile"):
            text = self.inspector.handle(command)
            self.assertIn("without a running runtime", text)

    def test_conversation_is_never_routed(self):
        # A plain sentence must not become a turn: the session's conversational
        # path and the model call both fail loudly if reached.
        self.session.handle = lambda *a, **k: self.fail("conversation was routed")
        text = self.inspector.handle("hello Astra, how are you?")
        self.assertIn("commands only", text)

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


class TestInspectShowsReadingAndLibrary(_InspectFixture):
    """The library is deterministic, so inspection must show it, not "not attached".

    The console used to build the session with ``enable_reader=False`` and then
    read the library only through the reader daemon, so both /reading and
    /library reported nothing. The library is now a first-class session resource
    independent of the daemon.
    """

    def test_library_is_attached_without_the_reader_daemon(self):
        self.assertIsNone(self.session.reader)
        self.assertIsNotNone(self.session.library)

    def test_reading_and_library_are_not_reported_missing(self):
        for command in ("/reading", "/library"):
            text = self.inspector.handle(command)
            self.assertNotIn("not attached", text)
            self.assertNotIn("No library attached", text)

    def test_reading_shows_the_saved_position_when_daemon_is_off(self):
        # Seed a work and position directly through the library.
        self.session.library.add_text("body " * 200, title="A Work", work_id="a-work")
        text = self.inspector.handle("/reading")
        self.assertIn("A Work", text)
        self.assertIn("disabled", text)          # the gate is honestly off
        self.assertIn("saved reading position", text)

    def test_library_lists_ingested_works(self):
        self.session.library.add_text("body " * 200, title="A Work", work_id="a-work")
        text = self.inspector.handle("/library")
        self.assertIn("LIBRARY (1 work(s))", text)
        self.assertIn("a-work", text)

    def test_resources_view_is_available_and_read_only(self):
        self.assertIn("/resources", self.inspector.available_commands())
        before = self._storage_snapshot()
        text = self.inspector.handle("/resources")
        self.assertIn("Resource history:", text)
        self.assertIn("idle (machine alone)", text)
        after = self._storage_snapshot()
        changed = {p for p in set(before) | set(after)
                   if before.get(p) != after.get(p)}
        self.assertEqual(changed, set(), f"inspection wrote files: {changed}")


class TestInspectForwardsToRuntime(_InspectFixture):
    """With a runtime running, the console is a remote control for every command.

    A tiny in-process "runtime" stands in for the live process: it holds the
    single-instance lock and serves the control channel through the session's own
    router, exactly as ``ChatSession._install_control_watcher`` does. That proves
    the console reaches the *real* command implementation rather than a copy.
    """

    def setUp(self):
        super().setUp()
        import threading
        import time

        from astra import control
        self.control = control
        # A lock naming our own pid makes ``runtime_listening`` read as running.
        self.lock = os.path.join(self.tmp, "runtime.lock")
        with open(self.lock, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        self.runtime_dir = self.tmp
        self._stop = threading.Event()

        def _serve():
            while not self._stop.is_set():
                for request in control.take_pending(runtime_dir=self.runtime_dir):
                    result = self.session.router.handle_control(
                        request["text"], source=request.get("source", "inspect"))
                    control.write_result(request["id"], result.get("text", ""),
                                         ok=True, runtime_dir=self.runtime_dir)
                time.sleep(0.02)

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()
        self.inspector = inspect_console.Inspector(
            self.session, storage_dir=self.storage, config_dir=self.config,
            runtime_dir=self.runtime_dir)

    def tearDown(self):
        self._stop.set()
        super().tearDown()

    def test_runtime_is_seen_as_listening(self):
        self.assertTrue(self.inspector.runtime_listening())

    def test_read_only_command_is_answered_by_the_runtime(self):
        text = self.inspector.handle("/status")
        self.assertIn("Astra is active", text)

    def test_mutating_command_is_executed_by_the_runtime(self):
        # /sleep is refused when no runtime runs, but with one running it is
        # forwarded and actually executed.
        text = self.inspector.handle("/sleep")
        self.assertIn("sleep", text.lower())
        self.assertTrue(self.session.controller.sleeping())

    def test_update_check_is_forwarded(self):
        text = self.inspector.handle("/update")
        self.assertIn("CODE UPDATE", text)

    def test_unknown_command_is_forwarded_and_reported(self):
        text = self.inspector.handle("/definitely-not-a-command")
        self.assertTrue(text.strip())

    def test_conversation_is_still_never_routed(self):
        text = self.inspector.handle("hello Astra")
        self.assertIn("commands only", text)

    def test_help_reflects_the_running_state(self):
        self.assertIn("commands run on the runtime", self.inspector.help_text())

    def test_update_is_listed_among_available_commands(self):
        self.assertIn("/update", self.inspector.available_commands())

    def test_bare_word_command_is_accepted_and_forwarded(self):
        # "status" (no slash) names a real command, so it is forwarded.
        self.assertIn("Astra is active", self.inspector.handle("status"))


class TestInspectLiveScreens(_InspectFixture):
    """The live observation screens are watchable read-only from inspection."""

    def test_live_renders_the_three_screens(self):
        text = self.inspector.handle("/live")
        self.assertIn("STATUS (LIVE)", text)
        self.assertIn("RELATIONSHIP (LIVE)", text)
        self.assertIn("AFFECT (LIVE)", text)
        self.assertIn("Read-only snapshot", text)

    def test_live_can_target_one_screen(self):
        text = self.inspector.handle("/live affect")
        self.assertIn("AFFECT (LIVE)", text)
        self.assertNotIn("RELATIONSHIP (LIVE)", text)

    def test_live_does_not_write_storage(self):
        before = self._storage_snapshot()
        for command in ("/live", "/live status", "/live relationship", "/live affect"):
            self.inspector.handle(command)
        after = self._storage_snapshot()
        changed = {p for p in set(before) | set(after)
                   if before.get(p) != after.get(p)}
        self.assertEqual(changed, set(), f"inspection wrote files: {changed}")

    def test_snapshot_capture_is_refused_read_only(self):
        wrapper = inspect_console._ReadOnlySnapshots(None)
        with self.assertRaises(RuntimeError):
            wrapper.capture("affect", {"values": {}})

    def test_live_is_listed_in_help_and_available(self):
        self.assertIn("/live", self.inspector.available_commands())
        self.assertIn("/live", self.inspector.help_text())


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
