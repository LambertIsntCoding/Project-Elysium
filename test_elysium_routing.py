"""Routing tests proving ELYSIUM is an application-level command interface.

The application decides, before Astra's generation path is entered, whether an
input invokes Elysium. These tests assert that decision on a real ChatSession
with a real store, and prove that Astra's generation function is genuinely
bypassed (not merely prompted differently).
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timezone

from elysium import ElysiumCommandHandler
from main import ChatSession
from memory_store import CommandStore, TripleMemoryStore
from orchestrator import CompanionOrchestrator

_FIXED_NOW = datetime(2026, 1, 2, 15, 4, tzinfo=timezone.utc)
_ASTRA_ROLEPLAY_MARKERS = (
    "admin mode", "final boss", "dramatic", "circuits",
    "chaotic protagonist", "bestie", "fam.", "elysium:",
)


class _GenerationSpy:
    """Records calls to Astra's generation path; returns a canary string."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return "ASTRA_GENERATED"


class TestElysiumRouting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.cmd_store = CommandStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=self.tmp)
        self.spy = _GenerationSpy()
        self.orch.query_gemma = self.spy

        # Deterministic clock so the Python time assertion is exact.
        self.handler = ElysiumCommandHandler(clock=lambda: _FIXED_NOW)
        self.session = ChatSession(
            self.orch, self.cmd_store, elysium_handler=self.handler,
            output_fn=lambda *_: None, input_fn=lambda _: "",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # TEST A ---------------------------------------------------------------
    def test_a_elysium_time_command_bypasses_astra(self):
        result = self.session.handle("Elysium what is the time")
        self.assertEqual(result, "Elysium > 3:04 PM UTC")
        self.assertEqual(self.spy.prompts, [], "Astra generation must not be called")
        self.assertEqual(self.session.turn_counter, 0)
        self.assertEqual(self.session.history, [])
        # The value came from the injected Python clock, not the model.
        self.assertIn("3:04 PM UTC", result)

    # TEST B ---------------------------------------------------------------
    def test_b_plain_time_question_uses_astra_with_clock_context(self):
        result = self.session.handle("What is the time?")
        self.assertEqual(result, "ASTRA_GENERATED")
        self.assertEqual(len(self.spy.prompts), 1)
        self.assertIn("CURRENT SYSTEM DATE AND TIME:", self.spy.prompts[0])
        self.assertIn(str(datetime.now().year), self.spy.prompts[0])
        self.assertEqual(self.session.turn_counter, 1)

    # TEST C ---------------------------------------------------------------
    def test_c_bare_elysium_invokes_handler_without_astra(self):
        result = self.session.handle("Elysium")
        self.assertTrue(result.startswith("Elysium > "))
        self.assertEqual(self.spy.prompts, [], "Astra must not generate for a bare invocation")
        self.assertEqual(self.session.history, [])

    # TEST D ---------------------------------------------------------------
    def test_d_plain_greeting_uses_astra_without_elysium(self):
        invoked = []
        original = self.handler.handle

        def spy_handler(text):
            invoked.append(text)
            return original(text)

        self.session.elysium_handler = spy_handler
        result = self.session.handle("Heyo!")
        self.assertEqual(result, "ASTRA_GENERATED")
        self.assertEqual(invoked, [], "Elysium handler must not be invoked")
        self.assertEqual(len(self.spy.prompts), 1)

    # TEST E ---------------------------------------------------------------
    def test_e_elysium_requests_cannot_trigger_astra_roleplay(self):
        # If Astra's generation were (wrongly) called it would emit the canary;
        # assert no Elysium invocation can ever surface roleplay narration.
        for text in ("Elysium", "Elysium what is the time", "Elysium do X"):
            output = self.session.handle(text).casefold()
            for marker in _ASTRA_ROLEPLAY_MARKERS:
                self.assertNotIn(marker, output, f"{marker!r} leaked for {text!r}")
        self.assertEqual(self.spy.prompts, [])

    # Extra guard: a sentence that merely mentions Elysium is normal Astra. --
    def test_mention_without_invocation_is_astra(self):
        result = self.session.handle("I read about the Elysium protocols today")
        self.assertEqual(result, "ASTRA_GENERATED")
        self.assertEqual(len(self.spy.prompts), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
