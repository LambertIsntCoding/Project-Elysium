"""Tests for Slice 1: the experience primitive and experiential affect.

Everything runs against a real ``TripleMemoryStore`` and a real
``CompanionOrchestrator`` on disk - no mocks. The tests pin the behaviours the
brief asked for:

* an experience is a first-class, structured self-model record, so it reuses the
  existing evidence / decay / dormancy / supersession machinery;
* it is *not* a self_fact or self_preference, so a single experience can never
  silently redefine Astra's personality;
* recording an experience moves a *temporary* affective state, which is
  separate from the durable relational state and decays toward neutral;
* the current condition changes *processing* (retrieval breadth), not just
  wording, and is only surfaced when it is not neutral;
* affect and experiences never reach the prompt mislabelled as facts about Roum
  or as tentative inferences about him;
* building a prompt stays read-only - it never writes the store.
"""

import copy
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from astra import affect
from astra import relational
from astra.memory import TripleMemoryStore, compact_record, _present
from astra.orchestrator import CompanionOrchestrator

CONFIG_DIR = "./config"


class _AffectCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def affect_state(self):
        return self.store.current_affect()


# ---------------------------------------------------------------------
# The experience primitive
# ---------------------------------------------------------------------
class TestExperienceRecord(_AffectCase):
    def test_experience_is_a_structured_self_record(self):
        mem = self.store.record_experience(
            "Read a chapter on tidal forces.", kind=affect.EXPERIENCE_READ,
            work_id="tides", intensity=0.7, significance=0.6,
        )
        self.assertEqual(mem["type"], "experience")
        self.assertEqual(mem["target_model"], "self")
        self.assertEqual(mem.get("experience_kind"), "read")
        self.assertEqual(mem.get("work_id"), "tides")
        self.assertAlmostEqual(mem.get("intensity"), 0.7, places=3)
        self.assertAlmostEqual(mem.get("significance"), 0.6, places=3)
        self.assertIn("work:tides", mem.get("tags"))

    def test_experience_is_not_a_durable_self_trait(self):
        """A single experience must never become a self_fact/self_preference."""
        self.store.record_experience("Found something odd in the ledger.", kind="discovered")
        types = {m["type"] for m in self.store.get_memories("self", status=None)}
        self.assertIn("experience", types)
        self.assertNotIn("self_fact", types)
        self.assertNotIn("self_preference", types)

    def test_experiences_are_queryable_by_work_and_kind(self):
        self.store.record_experience("Read ch.1", kind="read", work_id="tides")
        self.store.record_experience("Read ch.2", kind="read", work_id="tides")
        self.store.record_experience("Finished it", kind="completed", work_id="tides")
        self.store.record_experience("Unrelated discovery", kind="discovered")
        self.assertEqual(len(self.store.get_experiences()), 4)
        self.assertEqual(len(self.store.get_experiences(work_id="tides")), 3)
        self.assertEqual(len(self.store.get_experiences(kind="read")), 2)
        self.assertEqual(len(self.store.get_experiences(work_id="tides", kind="completed")), 1)

    def test_experience_survives_round_trip_compactly(self):
        self.store.record_experience(
            "Read a chapter.", kind="read", work_id="tides",
            intensity=0.8, significance=0.4,
        )
        # A bare experience (no work, default intensity/significance) is compact.
        plain = self.store.record_experience("Noticed the sky.", kind="discovered")
        stored = next(m for m in self.store.get_memories("self", status=None)
                      if m["id"] == plain["id"])
        # Neutral structured fields are dropped on disk but restored on read.
        self.assertNotIn("work_id", compact_record(copy.deepcopy(stored)) or {})
        self.assertIsNone(stored.get("work_id"))
        self.assertEqual(stored.get("intensity"), 0.5)
        # And the whole thing reloads intact from disk.
        reloaded = TripleMemoryStore(data_dir=self.tmp)
        self.assertEqual(len(reloaded.get_experiences()), 2)

    def test_experience_participates_in_decay(self):
        """Experiences are ordinary memories, so an old weak one can go dormant."""
        old = self.store.record_experience("A faint, unimportant impression.", kind="read",
                                           confidence=0.1)
        # Backdate it far into the past so the existing utility rules apply.
        past = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        self.store.update_memory("self", old["id"], timestamp=past)
        self.store.apply_decay()
        self.store.apply_utility_decay()
        mem = self.store.get_memory("self", old["id"])
        self.assertIn(mem["status"], ("dormant", "archived"))


# ---------------------------------------------------------------------
# Current experiential affect
# ---------------------------------------------------------------------
class TestAffectState(_AffectCase):
    def test_recording_an_experience_moves_affect(self):
        self.assertTrue(affect.is_neutral(self.affect_state()))
        self.store.record_experience("Read a lot of the book.", kind="read",
                                     intensity=0.8, significance=0.7)
        state = self.affect_state()
        self.assertFalse(affect.is_neutral(state))
        self.assertGreater(state["engagement"], 0.0)
        self.assertGreater(state["curiosity"], 0.0)

    def test_unknown_kind_is_a_noop(self):
        state = affect.record_event(None, "not-a-real-kind", text="?")
        self.assertTrue(affect.is_neutral(state))
        self.assertEqual(state["observations"], 0)

    def test_completion_relieves_frustration(self):
        state = None
        for _ in range(3):
            state = affect.record_event(state, affect.EXPERIENCE_FRUSTRATION, intensity=0.8)
        before = state["frustration"]
        self.assertGreater(before, 0.0)
        state = affect.record_event(state, affect.EXPERIENCE_COMPLETED,
                                    intensity=0.9, significance=0.9)
        self.assertLess(state["frustration"], before)

    def test_affect_decays_toward_neutral(self):
        state = affect.record_event(None, affect.EXPERIENCE_READ, intensity=0.9)
        self.assertGreater(state["engagement"], 0.0)
        old = copy.deepcopy(state)
        old["last_updated"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        relaxed = affect.record_event(old, "noop-kind")  # unknown kind still decays first
        self.assertLess(relaxed["engagement"], state["engagement"])

    def test_affect_is_separate_from_relational_state(self):
        """Experiential affect must not touch the Roum-specific preference."""
        self.store.record_experience("Read a whole book.", kind="read", intensity=0.9)
        affinity = relational.load_state_from_memories(
            self.store.get_memories("relationship", status=None))
        self.assertEqual(affinity["observations"], 0)
        self.assertEqual(affinity["command_affinity"], 0.0)
        # ...and the affect accumulator is a self record, not a relationship one.
        self.assertEqual(self.store.get_memories("relationship", status=None), [])

    def test_state_persists_and_reloads(self):
        self.store.record_experience("Finished a long game.", kind="completed",
                                     intensity=0.9, significance=0.9)
        before = self.affect_state()
        reloaded = TripleMemoryStore(data_dir=self.tmp).current_affect()
        self.assertEqual(round(reloaded["engagement"], 4), round(before["engagement"], 4))
        self.assertEqual(reloaded["event_counts"], before["event_counts"])


# ---------------------------------------------------------------------
# Affect changes processing, not just wording
# ---------------------------------------------------------------------
class TestAffectChangesProcessing(unittest.TestCase):
    def test_retrieval_breadth_widens_with_engagement_and_curiosity(self):
        self.assertEqual(affect.retrieval_breadth(None, 4), 4)
        state = None
        for _ in range(4):
            state = affect.record_event(state, affect.EXPERIENCE_READ, intensity=1.0,
                                        significance=1.0)
        self.assertGreater(affect.retrieval_breadth(state, 4), 4)


# ---------------------------------------------------------------------
# Prompt integration
# ---------------------------------------------------------------------
class TestAffectPrompt(_AffectCase):
    def test_neutral_affect_adds_no_block(self):
        prompt, diag = self.orch.build_prompt_with_diagnostics("hello", [])
        self.assertNotIn("CURRENT CONDITION", prompt)
        self.assertTrue(diag["experiential_affect"]["neutral"])

    def test_non_neutral_affect_is_surfaced_as_state_not_instruction(self):
        self.store.record_experience("Read a fascinating chapter.", kind="read",
                                     intensity=0.9, significance=0.8)
        prompt, diag = self.orch.build_prompt_with_diagnostics("hello", [])
        self.assertIn("CURRENT CONDITION", prompt)
        self.assertIn("TEMPORARY", prompt)
        self.assertIn("NOT AN INSTRUCTION", prompt)
        self.assertFalse(diag["experiential_affect"]["neutral"])
        # It is explicitly framed as distinct from how she feels about Roum.
        self.assertIn("distinct from how she feels about Roum", prompt)

    def test_experiences_reach_the_prompt_as_her_own_history(self):
        self.store.record_experience("Read about deep-sea currents.", kind="read",
                                     work_id="tides")
        prompt, diag = self.orch.build_prompt_with_diagnostics("what's up", [])
        self.assertIn("ASTRA'S RECENT EXPERIENCES", prompt)
        self.assertIn("deep-sea currents", prompt)
        self.assertEqual(len(diag["experience_ids"]), 1)
        # Not misfiled as an inference about Roum.
        idx = prompt.find("TENTATIVE INFERENCES")
        if idx >= 0:
            self.assertNotIn("deep-sea currents", prompt[idx:])

    def test_affect_accumulator_record_never_leaks_as_a_fact(self):
        self.store.record_experience("Read a chapter.", kind="read", intensity=0.9)
        prompt, _ = self.orch.build_prompt_with_diagnostics("hello", [])
        # The backing record's content is not injected as generic context.
        self.assertNotIn(affect.memory_content(), prompt)

    def test_prompt_build_is_read_only(self):
        """Building a prompt must never write the affect state (byte-stable)."""
        self.store.record_experience("Read a chapter.", kind="read", intensity=0.9)
        before = self.affect_state()
        for _ in range(3):
            self.orch.build_prompt("tell me something", [])
        after = self.affect_state()
        self.assertEqual(after, before)


# ---------------------------------------------------------------------
# Emotion tied into memory (colour + salience)
# ---------------------------------------------------------------------
class TestEmotionColour(_AffectCase):
    def test_experience_carries_a_colour_fact_does_not(self):
        exp = self.store.record_experience("Felt a rush of joy.", kind=affect.EXPERIENCE_HAPPY)
        self.assertEqual(affect.memory_colour(exp), "glad")
        self.assertEqual(affect.memory_colour({"type": "explicit_fact", "content": "x"}), "")
        # A read is coloured too, so an analytical memory is not colourless.
        read = self.store.record_experience("Read a chapter.", kind=affect.EXPERIENCE_READ)
        self.assertEqual(affect.memory_colour(read), "absorbed")

    def test_colour_falls_back_to_formative_kind(self):
        # A strong experience recorded without a specific kind still reads as hard.
        mem = {"type": "experience", "experience_kind": "", "formative_kind": "traumatic"}
        self.assertEqual(affect.memory_colour(mem), "painful")

    def test_felt_salience_ranks_valence_above_analysis(self):
        charged = {"type": "experience", "experience_kind": affect.EXPERIENCE_SAD}
        plain = {"type": "experience", "experience_kind": affect.EXPERIENCE_READ}
        self.assertGreater(affect.felt_salience(charged), affect.felt_salience(plain))
        self.assertEqual(affect.felt_salience(plain), 0.0)

    def test_alias_kind_is_coloured_and_counted(self):
        # A record stored under a reading alias still resolves to its emotion.
        mem = {"type": "experience", "experience_kind": "joy"}
        self.assertEqual(affect.resolve_kind("joy"), affect.EXPERIENCE_HAPPY)
        self.assertEqual(affect.memory_colour(mem), "glad")
        self.assertGreater(affect.felt_salience(mem), 0.0)

    def test_a_felt_memory_outranks_a_newer_analytical_one(self):
        """Emotion ties into memory: a charged moment stays surfaced."""
        charged = self.store.record_experience(
            "Felt a wave of warmth at the ending.", kind=affect.EXPERIENCE_TENDER)
        # Backdate it so every later experience is strictly newer.
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        self.store.update_memory("self", charged["id"], timestamp=past)
        for i in range(4):
            self.store.record_experience(f"Read analytical passage {i}.", kind="read")
        prompt = self.orch.build_prompt("what have you been doing?", [])
        # The warm memory survives the 4-item cap despite being the oldest.
        self.assertIn("wave of warmth", prompt)
        self.assertIn("(felt warm)", prompt)

    def test_formative_block_colours_a_felt_memory(self):
        self.store.record_experience(
            "It hurt to finish that book.", kind=affect.EXPERIENCE_SAD,
            intensity=0.8, significance=0.9)
        prompt = self.orch.build_prompt("how's it going?", [])
        self.assertIn("EXPERIENCES THAT STAYED WITH ASTRA", prompt)
        self.assertIn("(felt sad)", prompt)

    def test_colouring_the_prompt_writes_nothing(self):
        self.store.record_experience("Felt glad about finishing.", kind=affect.EXPERIENCE_HAPPY,
                                     intensity=0.8, significance=0.7)
        before = self.affect_state()
        self.orch.build_prompt("tell me something", [])
        self.assertEqual(self.affect_state(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
