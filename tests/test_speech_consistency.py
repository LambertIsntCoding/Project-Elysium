"""Regression tests for affect-driven speech and behavioural consistency.

The audit's concern was that Astra's internal state (affect, experiences,
reading) could leak into her *speech* as something to report, a canned voice
that fires identically regardless of subject, or a claim of an experience she
did not have. These tests pin the contract that:

* state shapes *how* she engages and is never an instruction to announce a
  number or narrate the machinery;
* a neutral state adds nothing (no canned voice);
* mixed states stay mixed rather than collapsing to one mood;
* she is not pushed to manufacture affect, experience, or profundity;
* a character's feeling is not hers;
* state-driven modulation is derived, bounded, and read-only.

They use a real ``TripleMemoryStore`` + ``CompanionOrchestrator`` on disk (no
mocks) plus the pure affect/behavior modules.
"""

import os
import shutil
import tempfile
import unittest

from astra import affect, behavior, generation
from astra.memory import TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator

CONFIG_DIR = "./config"


class _PromptCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def prompt(self, message, history=None):
        return self.orch.build_prompt(message, history or [])


class TestGenerationContract(unittest.TestCase):
    def test_the_block_states_how_state_is_used(self):
        block = generation.response_generation_block(config=None)
        self.assertIsNotNone(block)
        self.assertIn("NOT A TOPIC", block)

    def test_the_contract_forbids_narrating_the_machinery(self):
        block = generation.response_generation_block(config=None)
        # The generated block is code-owned and must forbid describing internal
        # processing; that is the mechanism that stops a state being narrated.
        self.assertIn("do not explain how the reply was produced", block.lower())

    def test_the_contract_protects_against_invented_experience(self):
        block = generation.response_generation_block(config=None)
        low = block.lower()
        self.assertIn("manufacture", low)
        self.assertIn("did not actually have", low)

    def test_the_contract_keeps_a_character_s_feeling_out_of_her_voice(self):
        block = generation.response_generation_block(config=None)
        low = block.lower()
        self.assertIn("character", low)
        self.assertIn("not report the character's emotion as her own", low)

    def test_the_contract_keeps_her_limited_experience_honest(self):
        block = generation.response_generation_block(config=None)
        # "she has only a little accumulated experience" is a design fact; the
        # contract must let a plain reaction stay plain rather than force depth.
        self.assertIn("a plain reaction can stay plain",
                      generation.response_generation_block(config=None).lower())


class TestAffectSpeech(unittest.TestCase):
    def test_neutral_state_adds_no_block(self):
        # A flat state must not produce a canned "here is my mood" block.
        self.assertIsNone(affect.prompt_block(affect._blank_state()))

    def test_non_neutral_state_is_a_behavioural_cause_not_an_instruction(self):
        block = affect.prompt_block({"frustration": 0.8, "weary": 0.7})
        self.assertIsNotNone(block)
        low = block.lower()
        self.assertIn("tendencies", low)
        # It must explicitly say it is not an instruction and not to be named.
        self.assertIn("not an instruction", low)
        self.assertIn("does not have to announce it", low)

    def test_no_numeric_readout_reaches_the_affect_block(self):
        block = affect.prompt_block({"frustration": 0.83, "engagement": 0.42})
        # Astra never sees her own numbers: the values that produce the state are
        # implementation, not introspection.
        self.assertNotIn("0.83", block)
        self.assertNotIn("0.42", block)
        self.assertNotRegex(block, r"\b(?:frustration|engagement)\s*[:=]\s*\d")

    def test_mixed_states_stay_mixed(self):
        block = affect.prompt_block({"weary": 0.7, "curiosity": 0.6})
        low = block.lower()
        # The contract must say these interact rather than collapse to a mood.
        self.assertIn("mix", low)

    def test_moderate_state_is_not_inflated(self):
        block = affect.prompt_block({"frustration": 0.2, "weary": 0.2})
        low = block.lower()
        self.assertIn("moderate states stay moderate", low)
        self.assertIn("a little friction is not anger", low)

    def test_a_low_state_does_not_manufacture_a_feeling(self):
        # Tiredness is framed as tiredness, not sadness or a verdict.
        block = affect.prompt_block({"weary": 0.6})
        self.assertIn("tiredness is not sadness", block.lower())


class TestBehaviouralConsistency(unittest.TestCase):
    def test_behavior_is_derived_and_bounded(self):
        # A extreme state must not produce unbounded modulation.
        modulation = behavior.from_affect({"sad": 1.0, "weary": 1.0,
                                           "frustration": 1.0})
        for key in ("energy", "initiative", "pace", "brevity"):
            if key in modulation:
                value = modulation[key]
                if isinstance(value, (int, float)):
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_a_neutral_state_changes_nothing(self):
        # A flat state must not invent a modulation: the baseline is neutral.
        neutral = behavior.from_affect({})
        self.assertTrue(neutral)
        # The timing projection of a neutral state is the neutral scale.
        timing = behavior.timing_modifiers(neutral)
        self.assertAlmostEqual(float(timing.get("typing_scale", 1.0)), 1.0, places=2)

    def test_sadness_and_tiredness_move_her_the_same_way(self):
        # Two low states should lower drive, so behaviour is consistent, not
        # contradictory, across the affect dimensions.
        sad = behavior.from_affect({"sad": 0.8, "weary": 0.6, "engagement": 0.2})
        tired = behavior.from_affect({"weary": 0.7, "engagement": 0.2})
        self.assertLessEqual(float(sad.get("energy", 1.0)), 0.6)
        self.assertLessEqual(float(tired.get("energy", 1.0)), 0.6)

    def test_modulation_never_reaches_the_prompt_as_numbers(self):
        block = behavior.prompt_block(behavior.from_affect({"sad": 0.9}))
        if block is not None:
            self.assertNotRegex(block, r"\b\d\.\d\b")


class TestPromptIntegration(_PromptCase):
    def test_the_generation_block_is_in_the_prompt(self):
        prompt = self.prompt("hello")
        self.assertIn("HOW ASTRA USES HER INTERNAL STATE", prompt)

    def test_a_flat_store_does_not_invent_affect_in_the_prompt(self):
        # With no experiences recorded, the prompt must not claim a mood.
        prompt = self.prompt("hello")
        self.assertNotIn("CURRENT CONDITION", prompt)

    def test_the_prompt_never_exposes_component_names_as_numbers(self):
        prompt = self.prompt("what are you thinking about?")
        self.assertNotRegex(prompt, r"Happiness\s*=\s*\d")


if __name__ == "__main__":
    unittest.main()
