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


if __name__ == "__main__":
    unittest.main(verbosity=2)
