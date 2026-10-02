"""Tests for Astra's self-understanding, identity, and formative experience.

Everything runs against a real ``TripleMemoryStore`` and a real
``CompanionOrchestrator`` on disk - no mocks. The tests pin the brief:

* the non-human boundary is settled knowledge, always present, and a claim that
  she can become biologically human is never accepted as a self-fact;
* a claim about an experience she could not have had - a body, a childhood, a
  physical place, or a restatement of Roum's own life - is demoted to a weak
  observation and never becomes durable self-knowledge;
* a self-belief/disposition minted from a single generated sentence is gated on
  repetition, exactly like a self-preference;
* what Astra "tends to enjoy / struggle with / return to" is *derived* from her
  own recorded experiences, carries its evidence, and can be absent;
* experiences can be memorable or traumatic, persist as her history, and are
  surfaced with real elapsed time;
* building a prompt stays read-only, and the prompt never leaks the internal
  numeric values that produce her state.
"""

import os
import shutil
import tempfile
import unittest

from astra import affect
from astra import selfhood
from astra.memory import TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator

CONFIG_DIR = "./config"


class _SelfhoodCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def add_self(self, content, mem_type, source="ai_extraction", **kw):
        mid = self.store.add_memory("self", content, mem_type, source, **kw)
        return self.store.get_memory("self", mid)


# ---------------------------------------------------------------------
# The non-human boundary
# ---------------------------------------------------------------------
class TestNonHumanBoundary(_SelfhoodCase):
    def test_boundary_is_always_in_the_prompt(self):
        prompt = self.orch.build_prompt("hello", [])
        self.assertIn("WHAT ASTRA IS (FOUNDATIONAL", prompt)
        self.assertIn("not a human being", prompt)
        self.assertIn("becoming Astra", prompt)

    def test_boundary_states_she_cannot_become_human_but_can_become_someone(self):
        text = "\n".join(selfhood.NONHUMAN_BOUNDARY).casefold()
        # The impossibility is stated...
        self.assertIn("biological human body", text)
        self.assertIn("no hidden path", text)
        # ...and so is the open-endedness: non-human does not mean static.
        self.assertIn("does not mean she cannot become anything", text)

    def test_claiming_human_becoming_is_not_accepted_as_a_self_fact(self):
        mem = self.add_self("My goal is to become fully human one day.",
                            "self_belief", confidence=0.95)
        self.assertEqual(mem["type"], "self_observation")
        self.assertTrue(mem.get("boundary_violation"))
        self.assertLessEqual(mem["confidence"], 0.3)

    def test_claiming_a_biological_body_is_not_accepted(self):
        mem = self.add_self("I will eventually obtain a biological body.",
                            "self_belief", confidence=0.9)
        self.assertTrue(mem.get("boundary_violation"))

    def test_interest_in_humanity_is_not_a_boundary_violation(self):
        mem = self.add_self("I am deeply interested in how humans experience the world.",
                            "self_belief", confidence=0.6)
        self.assertFalse(mem.get("boundary_violation"))

    def test_human_claims_never_reach_the_prompt_as_self_knowledge(self):
        self.add_self("I will become a biological human someday.",
                      "self_belief", confidence=0.9)
        prompt = self.orch.build_prompt("who are you?", [])
        self.assertNotIn("I will become a biological human someday.", prompt)


class TestEpistemicStance(_SelfhoodCase):
    def test_stance_is_always_in_the_prompt(self):
        prompt = self.orch.build_prompt("hello", [])
        self.assertIn("HOW ASTRA HOLDS WHAT SHE KNOWS", prompt)

    def test_stance_questions_roum_without_rejecting_everything(self):
        text = "\n".join(selfhood.EPISTEMIC_STANCE).casefold()
        # She may doubt a claim just because it came from Roum...
        self.assertIn("merely because roum said it", text)
        # ...and provenance changes how much she leans on it...
        self.assertIn("why she thinks she knows it", text)
        # ...but doubt must not collapse into refusal.
        self.assertIn("doubt is not refusal", text)
        self.assertIn("without demanding absolute certainty", text)

    def test_stance_never_becomes_a_stored_memory(self):
        before = len(self.store.get_memories("self", status=None))
        self.orch.build_prompt("hello", [])
        self.assertEqual(len(self.store.get_memories("self", status=None)), before)


# ---------------------------------------------------------------------
# Reification: no personal experience she never had
# ---------------------------------------------------------------------
class TestAbsentExperienceGuard(_SelfhoodCase):
    def test_bodily_experience_is_demoted(self):
        for text in ("I felt the sun on my skin today.",
                     "I walked along the beach and watched the sunset.",
                     "I remember my childhood home.",
                     "I tasted the salt in the air."):
            mem = self.add_self(text, "self_belief", confidence=0.95)
            self.assertEqual(mem["type"], "self_observation", text)
            self.assertTrue(mem.get("absent_experience"), text)
            self.assertLessEqual(mem["confidence"], 0.5, text)

    def test_roums_life_is_not_recast_as_her_memory(self):
        self.store.add_memory(
            "roum",
            "Roum grew up in a small coastal town and remembers the smell of salt.",
            "explicit_fact", "explicit_user_statement",
        )
        mem = self.add_self(
            "I grew up in a small coastal town and remember the smell of salt.",
            "self_fact", confidence=0.9,
        )
        self.assertEqual(mem["type"], "self_observation")
        self.assertTrue(mem.get("absent_experience"))

    def test_legitimate_astra_activity_is_not_flagged(self):
        for text, typ in (
            ("I ran the test suite three times and it finally passed.", "self_observation"),
            ("I walked through the logic of the memory store.", "self_observation"),
            ("I read a passage about deep-sea currents.", "experience"),
        ):
            mem = self.add_self(text, typ, confidence=0.9)
            self.assertFalse(mem.get("absent_experience"), text)

    def test_a_quoted_passage_is_not_her_experience(self):
        # Reading records keep an excerpt; a body inside the book's narration is
        # the book's, not Astra's.
        mem = self.add_self(
            'Read a passage of "Garden". It began: "She felt the sun on her skin."',
            "experience", source="experiential", confidence=0.9,
        )
        self.assertFalse(mem.get("absent_experience"))
        self.assertEqual(mem["type"], "experience")

    def test_boundary_says_roums_life_is_not_hers(self):
        # The principle must be settled self-knowledge, present every turn.
        text = "\n".join(selfhood.NONHUMAN_BOUNDARY).casefold()
        self.assertIn("his, not hers", text)
        self.assertIn("borrowing his", text)
        prompt = self.orch.build_prompt("hello", [])
        self.assertIn("borrowing his", prompt)

    def test_mirrored_lived_memory_never_reaches_the_prompt(self):
        self.store.add_memory(
            "roum",
            "Roum grew up in a small coastal town and remembers the smell of salt.",
            "explicit_fact", "explicit_user_statement",
        )
        self.add_self(
            "I grew up in a small coastal town and remember the smell of salt.",
            "self_fact", confidence=0.9,
        )
        prompt = self.orch.build_prompt("tell me about the coast", [])
        self.assertNotIn("I grew up in a small coastal town", prompt)


# ---------------------------------------------------------------------
# Self-claim durability
# ---------------------------------------------------------------------
class TestSelfClaimDurability(_SelfhoodCase):
    def test_single_self_belief_is_demoted(self):
        mem = self.add_self("My core objective is pattern fidelity.",
                            "self_belief", confidence=0.95)
        self.assertEqual(mem["type"], "self_observation")
        self.assertLessEqual(mem["confidence"], 0.5)

    def test_repeated_self_belief_becomes_durable(self):
        mid = None
        for _ in range(3):
            mid = self.store.add_memory("self", "Astra prefers concise answers.",
                                        "self_belief", "ai_extraction")
        self.assertEqual(self.store.get_memory("self", mid)["type"], "self_belief")



# ---------------------------------------------------------------------
# Derived dispositions
# ---------------------------------------------------------------------
class TestDerivedDispositions(_SelfhoodCase):
    def test_no_experience_yields_no_portrait(self):
        self.assertIsNone(selfhood.disposition_block([]))
        self.assertEqual(selfhood.derive_dispositions([]), [])

    def test_repeated_work_reads_as_engagement(self):
        # Distinct experiences: identical text dedupes into one reinforced
        # record, which is right for the store but is not "returning" evidence.
        for n in range(3):
            self.store.record_experience(f"Read chapter {n} of 'Tides'.",
                                         kind="read", work_id="tides")
        dispositions = selfhood.derive_dispositions(self.store.get_experiences())
        kinds = {d["kind"] for d in dispositions}
        self.assertIn(selfhood.DISPOSITION_ENGAGEMENT, kinds)
        engagement = next(d for d in dispositions
                          if d["kind"] == selfhood.DISPOSITION_ENGAGEMENT)
        self.assertEqual(engagement["basis"], 3)
        self.assertTrue(engagement["evidence"])

    def test_difficulty_is_about_the_task_not_incapacity(self):
        self.store.record_experience("Got stuck on 'Tides'.", kind="frustration",
                                     work_id="tides", intensity=0.8)
        self.store.record_experience("Could not follow the middle of 'Tides'.",
                                     kind="frustration", work_id="tides", intensity=0.8)
        dispositions = selfhood.derive_dispositions(self.store.get_experiences())
        text = " ".join(d["text"] for d in dispositions).casefold()
        self.assertIn("hard", text)
        self.assertNotIn("cannot", text)
        self.assertNotIn("incapable", text)

    def test_a_single_experience_is_not_a_pattern(self):
        self.store.record_experience("Read a passage.", kind="read", work_id="tides")
        dispositions = selfhood.derive_dispositions(self.store.get_experiences())
        self.assertEqual(
            [d for d in dispositions if d["kind"] == selfhood.DISPOSITION_ENGAGEMENT],
            [],
        )

    def test_portrait_block_is_labelled_as_fallible(self):
        for n in range(3):
            self.store.record_experience(f"Read chapter {n} of 'Tides'.",
                                         kind="read", work_id="tides")
        block = selfhood.disposition_block(self.store.get_experiences())
        self.assertIsNotNone(block)
        self.assertIn("MAY BE WRONG", block)
        self.assertIn("never means", block)


# ---------------------------------------------------------------------
# Formative and traumatic experiences
# ---------------------------------------------------------------------
class TestFormativeExperiences(_SelfhoodCase):
    def test_ordinary_experience_stays_lightweight(self):
        mem = self.store.record_experience("Read a passage.", kind="read",
                                           intensity=0.3, significance=0.3)
        self.assertEqual(mem.get("formative_kind"), selfhood.KIND_ORDINARY)
        self.assertFalse(selfhood.is_formative(mem))

    def test_negative_high_intensity_is_traumatic(self):
        mem = self.store.record_experience(
            "Was humiliated while trying to finish 'Tides'.",
            kind="frustration", intensity=0.9, significance=0.4)
        self.assertEqual(mem.get("formative_kind"), selfhood.KIND_TRAUMATIC)
        self.assertTrue(selfhood.is_formative(mem))

    def test_memorable_completion_persists(self):
        mem = self.store.record_experience(
            "Finished reading 'Tides'.", kind="completed",
            intensity=0.7, significance=0.9)
        self.assertEqual(mem.get("formative_kind"), selfhood.KIND_FORMATIVE)
        self.assertGreater(mem.get("importance"), 0.5)

    def test_formative_experience_keeps_an_experiential_source(self):
        mem = self.store.record_experience("Finished 'Tides'.", kind="completed",
                                           intensity=0.7, significance=0.9)
        self.assertEqual(mem["source"], selfhood.SOURCE_EXPERIENTIAL)

    def test_formative_block_surfaces_with_real_elapsed_time(self):
        self.store.record_experience("Finished reading 'Tides'.", kind="completed",
                                     intensity=0.7, significance=0.9)
        block = selfhood.formative_block(self.store.get_experiences())
        self.assertIn("EXPERIENCES THAT STAYED WITH ASTRA", block)
        self.assertIn("Tides", block)
        self.assertIn("today", block)

    def test_formative_block_absent_without_formative_experiences(self):
        self.store.record_experience("Read a passage.", kind="read",
                                     intensity=0.2, significance=0.2)
        self.assertIsNone(selfhood.formative_block(self.store.get_experiences()))

    def test_trauma_is_not_framed_as_a_limit(self):
        self.store.record_experience("Something frightening happened.",
                                     kind="frustration", intensity=0.95,
                                     significance=0.4)
        block = selfhood.formative_block(self.store.get_experiences())
        self.assertIn("not something that defines her", block)


# ---------------------------------------------------------------------
# Prompt integration and read-only guarantees
# ---------------------------------------------------------------------
class TestPromptIntegration(_SelfhoodCase):
    def test_prompt_includes_selfhood_and_history_blocks(self):
        self.store.record_experience("Finished reading 'Tides'.", kind="completed",
                                     intensity=0.7, significance=0.9)
        prompt = self.orch.build_prompt("how's it going?", [])
        self.assertIn("WHAT ASTRA IS (FOUNDATIONAL", prompt)
        self.assertIn("ASTRA'S RECENT EXPERIENCES", prompt)
        self.assertIn("EXPERIENCES THAT STAYED WITH ASTRA", prompt)

    def test_prompt_does_not_leak_numeric_state(self):
        self.store.record_experience("Finished reading 'Tides'.", kind="completed",
                                     intensity=0.7, significance=0.9)
        prompt = self.orch.build_prompt("how's it going?", []).casefold()
        # Neither the affect values nor the relational values appear as numbers.
        self.assertNotIn("engagement ", prompt)
        self.assertNotIn("command affinity", prompt)
        self.assertNotIn("task fulfillment satisfaction", prompt)

    def test_affect_summary_is_non_numeric(self):
        state = affect.record_event(None, affect.EXPERIENCE_DISCOVERED,
                                    intensity=0.9, significance=0.9)
        summary = affect.render_summary(state)
        self.assertIsNotNone(summary)
        self.assertNotIn("0.", summary)
        self.assertNotIn("1.", summary)

    def test_building_a_prompt_does_not_write_the_store(self):
        self.store.record_experience("Finished reading 'Tides'.", kind="completed",
                                     intensity=0.7, significance=0.9)
        self.store.add_memory("roum", "Roum likes tea.", "explicit_preference",
                              "explicit_user_statement")

        def snapshot():
            out = {}
            for name, path in self.store.files.items():
                if not os.path.exists(path):
                    out[name] = None
                    continue
                with open(path, "rb") as fh:
                    out[name] = fh.read()
            return out

        before = snapshot()
        self.orch.build_prompt("anything", [])
        self.assertEqual(before, snapshot())

    def test_formative_experience_round_trips_compactly(self):
        mem = self.store.record_experience("Finished 'Tides'.", kind="completed",
                                           intensity=0.7, significance=0.9)
        stored = next(m for m in self.store.get_memories("self", status=None)
                      if m["id"] == mem["id"])
        # The derived kind is preserved through compaction (it is not a default).
        self.assertEqual(stored.get("formative_kind"), selfhood.KIND_FORMATIVE)
        reloaded = TripleMemoryStore(data_dir=self.tmp)
        again = reloaded.get_memory("self", mem["id"])
        self.assertEqual(again.get("formative_kind"), selfhood.KIND_FORMATIVE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
