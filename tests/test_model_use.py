"""Tests for the model-use gate: when the language model may be invoked.

Astra is not the model; the model is her voice. These tests pin the boundary the
project owner asked for: the voice is only active when it is doing a job, and the
one job that runs while Roum is away (the background reader) is off by default.
They assert behaviour (which features are enabled, and that a disabled feature
really skips the model call), not prompt strings.
"""

import os
import tempfile
import unittest
from unittest import mock

from astra import model_use


class TestModelUseGate(unittest.TestCase):
    def test_defaults_turn_background_reading_off(self):
        gate = model_use.from_config({})
        self.assertTrue(gate.enabled(model_use.FEATURE_CONVERSATION))
        self.assertTrue(gate.enabled(model_use.FEATURE_CONSOLIDATION))
        self.assertTrue(gate.enabled(model_use.FEATURE_ELYSIUM))
        self.assertFalse(gate.enabled(model_use.FEATURE_READING))

    def test_missing_or_malformed_block_uses_defaults(self):
        self.assertFalse(model_use.from_config({"runtime": "nonsense"})
                         .enabled(model_use.FEATURE_READING))
        self.assertFalse(model_use.from_config(None).enabled(model_use.FEATURE_READING))

    def test_config_can_turn_reading_on(self):
        gate = model_use.from_config({"model_use": {"reading": True}})
        self.assertTrue(gate.enabled(model_use.FEATURE_READING))

    def test_config_can_turn_conversation_off(self):
        gate = model_use.from_config({"model_use": {"conversation": False}})
        self.assertFalse(gate.enabled(model_use.FEATURE_CONVERSATION))

    def test_unknown_feature_is_off(self):
        gate = model_use.from_config({"model_use": {"whatever": True}})
        self.assertFalse(gate.enabled("not_a_feature"))


class TestConsolidationGate(unittest.TestCase):
    """A disabled consolidation feature must skip the per-turn model call."""

    def setUp(self):
        import shutil
        from astra.memory import CommandStore, TripleMemoryStore
        from astra.orchestrator import CompanionOrchestrator
        from main import ChatSession

        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir="./config")
        self.orch.query_gemma = lambda *a, **k: "Sure."
        self._shutil = shutil
        self._Cmd = CommandStore
        self._Session = ChatSession

    def tearDown(self):
        self._shutil.rmtree(self.tmp, ignore_errors=True)

    def _session(self, gate):
        calls = []
        session = self._Session(
            self.orch, self._Cmd(data_dir=self.tmp),
            consolidator=lambda *a, **k: calls.append(a),
            model_use=gate, output_fn=lambda *_: None,
        )
        return session, calls

    def test_disabled_consolidation_skips_the_model_call(self):
        gate = model_use.from_config({"model_use": {"consolidation": False}})
        session, calls = self._session(gate)
        session.handle("hello there")
        self.assertEqual(calls, [])

    def test_enabled_consolidation_runs_the_consolidator(self):
        gate = model_use.from_config({"model_use": {"consolidation": True}})
        session, calls = self._session(gate)
        session.handle("hello there")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
