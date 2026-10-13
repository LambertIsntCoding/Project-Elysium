"""Integration tests for the application runtime wired through build_session.

These exercise the real router/controller against a real (temporary) store and
orchestrator. The model backend is replaced with a local stub, so no network is
used; nothing about the routing authority under test is mocked.
"""

import os
import shutil
import tempfile
import unittest

import main
from astra import prepared_context as pc
from astra import runtime_state as rs


class _StubSession:
    """Minimal ChatSession protocol double for router tests."""

    def __init__(self):
        self.output_fn = lambda text: None
        self.handle_calls = []
        self.activity = 0

    def note_activity(self):
        self.activity += 1

    def handle(self, text):
        self.handle_calls.append(text)
        return f"echo:{text}"


class _StubController:
    def __init__(self, sleeping=False):
        self._sleeping = sleeping
        self.wakes = []
        self.proactive = _StubProactive()

    def sleeping(self):
        return self._sleeping

    def wake(self, *, reason=None):
        self._sleeping = False
        self.wakes.append(reason)
        return {}


class _StubProactive:
    def __init__(self):
        self.surfaced = set()

    def was_surfaced(self, ref_id):
        return ref_id in self.surfaced

    def note_surfaced(self, ref_id):
        if ref_id:
            self.surfaced.add(ref_id)


class TestRouterAuthority(unittest.TestCase):
    def test_conversation_is_routed_to_session(self):
        from astra.application import ApplicationRouter
        session = _StubSession()
        router = ApplicationRouter(session, _StubController())
        result = router.handle("hello", source="discord")
        self.assertEqual(result["text"], "echo:hello")
        self.assertEqual(session.handle_calls, ["hello"])

    def test_sleep_command_goes_to_controller_not_model(self):
        from astra.application import ApplicationRouter
        session = _StubSession()
        controller = _StubController()

        calls = []

        def fake_sleep():
            calls.append("sleep")
            return {}

        controller.sleep = fake_sleep
        # The router captures session output; we only assert the model was not
        # asked to answer the slash command.
        ApplicationRouter(session, controller).handle("/sleep")
        self.assertEqual(session.handle_calls, [])

    def test_conversation_while_sleeping_wakes_with_human_reason(self):
        from astra.application import ApplicationRouter
        session = _StubSession()
        controller = _StubController(sleeping=True)
        ApplicationRouter(session, controller).handle("are you there?")
        self.assertEqual(controller.wakes, [rs.REASON_HUMAN_MESSAGE])


class TestBuildSessionWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = os.path.join(self.tmp, "storage")
        os.makedirs(self.storage)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _session(self):
        session = main.build_session(config_dir=self.tmp, storage_dir=self.storage,
                                     enable_reader=False)
        # Replace the network call with a local stub; routing is what we test.
        session.orchestrator.query_gemma = lambda *a, **k: "stub reply"
        return session

    def test_runtime_is_attached(self):
        session = self._session()
        self.assertIsNotNone(session.controller)
        self.assertIsNotNone(session.router)
        self.assertIs(session.orchestrator.prepared_context, session.controller.prepared)

    def test_sleep_then_conversation_wakes(self):
        session = self._session()
        session.router.handle("/sleep")
        self.assertTrue(session.controller.sleeping())
        session.router.handle("hello again")
        self.assertFalse(session.controller.sleeping())

    def test_prompt_building_does_not_rewrite_prepared_context(self):
        session = self._session()
        controller = session.controller
        controller.prepared.replace([{"kind": "memory", "ref_id": "mem_x",
                                      "label": "the sailing lesson"}])
        path = controller.prepared.path
        with open(path, "rb") as handle:
            before = handle.read()
        # Build twice with prepared context attached.
        session.orchestrator.build_prompt_with_diagnostics("tell me", [], detailed=False)
        session.orchestrator.build_prompt_with_diagnostics("the sailing lesson", [], detailed=False)
        with open(path, "rb") as handle:
            after = handle.read()
        self.assertEqual(before, after)

    def test_status_reports_without_a_model_call(self):
        session = self._session()

        def explode(*a, **k):
            raise AssertionError("status must not call the model")

        session.orchestrator.query_gemma = explode
        out = session.router.handle("/status")["text"]
        self.assertIn("Astra", out)

    def test_task_runs_only_while_sleeping(self):
        session = self._session()
        controller = session.controller
        controller.seed()
        # Active: nothing runs.
        controller.scheduler.run_cycle()
        self.assertEqual(controller.scheduler.last_report["tasks"], 0)
        controller.sleep()
        controller.scheduler.run_cycle()
        self.assertGreaterEqual(controller.scheduler.last_report["tasks"], 1)


class TestAutonomyCommands(unittest.TestCase):
    """The visibility/control surface: reports reflect real records, no model."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = os.path.join(self.tmp, "storage")
        os.makedirs(self.storage)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _session(self):
        session = main.build_session(config_dir=self.tmp, storage_dir=self.storage,
                                     enable_reader=False)
        session.orchestrator.query_gemma = lambda *a, **k: "stub reply"
        return session

    def test_autonomy_and_discord_reports_do_not_call_the_model(self):
        session = self._session()

        def explode(*a, **k):
            raise AssertionError("status reports must not call the model")

        session.orchestrator.query_gemma = explode
        for command in ("/autonomy", "/notifications", "/discord", "/schedules",
                        "/reader", "/normalize-text", "/autonomy now"):
            out = session.router.handle(command)["text"]
            self.assertIsInstance(out, str)

    def test_schedule_report_is_derived_from_task_records(self):
        session = self._session()
        controller = session.controller
        spec = {"mode": "daily", "at": "03:00", "timezone": "UTC",
                "catch_up": "run", "dedupe_key": "x"}
        from astra.tasks import BackgroundTask, ORIGIN_SCHEDULE
        task = BackgroundTask(kind="memory_review", origin=ORIGIN_SCHEDULE,
                              schedule=spec, dedupe_key="x",
                              due_at="2099-01-01T03:00:00+00:00")
        controller.task_store.enqueue(task)
        out = session.router.handle("/schedules")["text"]
        self.assertIn("why:", out)
        self.assertIn(task.kind, out)

    def test_normalize_text_preview_does_not_rewrite(self):
        session = self._session()
        store = session.orchestrator.store
        store.add_memory("roum", "clean", "explicit_fact", "user_statement")
        store.memories["roum"][0]["content"] = "noisy &amp; text"
        out = session.router.handle("/normalize-text")["text"]
        self.assertIn("preview", out.lower())
        self.assertEqual(store.memories["roum"][0]["content"], "noisy &amp; text")

    def test_reader_health_reports_without_model(self):
        session = self._session()
        session.orchestrator.query_gemma = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("reader health must not call the model"))
        out = session.router.handle("/reader")["text"]
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip())

    def test_pause_activity_does_not_pause_notifications(self):
        session = self._session()
        session.router.handle("/autonomy pause")
        self.assertTrue(session.controller.autonomy.paused)
        # Notifications remain un-paused: the two controls are independent.
        self.assertFalse(session.controller.notifications.paused)

    def test_pause_notifications_does_not_pause_activity(self):
        session = self._session()
        session.router.handle("/notifications pause")
        self.assertTrue(session.controller.notifications.paused)
        self.assertFalse(session.controller.autonomy.paused)

    def test_headless_flag_wires_think_only(self):
        session = main.build_session(config_dir=self.tmp, storage_dir=self.storage,
                                     enable_reader=False, think_only=True)
        self.assertTrue(session.think_only)

    def test_discord_disabled_without_token_env(self):
        # No bot token in the environment -> no adapter, no network attempt.
        os.environ.pop("DISCORD_BOT_TOKEN", None)
        session = self._session()
        # The configured channel exists but no token, so nothing is built.
        self.assertIsNone(session.discord)
        self.assertIn("not configured", session.controller.discord_report(None).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
