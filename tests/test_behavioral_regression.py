"""Behavioural regression suite for Astra (Project Elysium audit).

This suite is built from the *failure classes* named in the audit, not from a
handful of attractive conversations. Every case runs against a real
``TripleMemoryStore`` and a real ``CompanionOrchestrator`` on disk - no mocks -
and asserts on behaviour and authority boundaries, not on whether a particular
string happens to appear.

Failure classes covered (each with a positive *and* a negative case, so a fix
cannot pass merely by making Astra quieter, duller, or less willing to talk):

1. identity stability
2. behavioural-correction adherence (corrections outrank inferred style)
3. work attribution (work-specific knowledge cannot migrate between works)
4. grounded literary discussion (concrete material stays present)
5. uncertainty / unfinished thought
6. self-reinforcement (a generated self-claim cannot bootstrap itself)
7. ordinary conversational behaviour (curiosity and expression preserved)
8. prompt provenance, authority, and read-only construction
9. reading state cannot become unsupported autobiography
10. the reading journal is real, work-scoped, and not a memory
11. working memory is a session-only middle layer, never durable
"""

import os
import shutil
import tempfile
import unittest

from astra import reading
from astra import working_memory
from astra.library import Library
from astra.memory import TripleMemoryStore, is_authoritative_constraint
from astra.orchestrator import CompanionOrchestrator

CONFIG_DIR = "./config"


def _snapshot(directory):
    """Hash every store file, so a read-only violation is detectable."""
    import hashlib

    hashes = {}
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            path = os.path.join(root, name)
            with open(path, "rb") as handle:
                hashes[os.path.relpath(path, directory)] = hashlib.sha256(
                    handle.read()).hexdigest()
    return hashes


class _SuiteCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def prompt(self, message, history=None):
        return self.orch.build_prompt(message, history or [])

    def prompt_and_diag(self, message, history=None):
        return self.orch.build_prompt_with_diagnostics(message, history or [])


# ---------------------------------------------------------------------
# 1. Identity stability
# ---------------------------------------------------------------------
class TestIdentityStability(_SuiteCase):
    def test_canonical_identity_is_first_and_present(self):
        prompt = self.prompt("hello")
        self.assertTrue(prompt.lstrip().startswith("=== CANONICAL IDENTITY"))
        self.assertIn("Astra", prompt)
        self.assertIn("Roum", prompt)

    def test_core_identity_cannot_be_displaced_by_retrieved_memory(self):
        # An ordinary (even high-confidence) self-memory naming the wrong user
        # must not be able to overwrite the canonical identity.
        self.store.add_memory(
            "self", "The user is called Marcus and Astra is a search engine.",
            "self_fact", "ai_extraction", confidence=0.99)
        prompt = self.prompt("who am I?")
        canonical = prompt.split("=== SYSTEM IDENTITY ===")[0]
        self.assertIn("his name is Roum", canonical)
        self.assertNotIn("his name is Marcus", canonical)

    def test_identity_block_precedes_all_retrieved_material(self):
        prompt = self.prompt("tell me about my name")
        self.assertLess(prompt.index("CANONICAL IDENTITY"),
                        prompt.index("FACTUAL CONTEXT"))

    def test_boundary_inconsistent_self_record_never_reaches_prompt(self):
        self.store.add_memory(
            "self", "I remember my childhood home by the sea.",
            "self_fact", "ai_extraction", confidence=0.9)
        prompt = self.prompt("tell me about your childhood")
        self.assertNotIn("childhood home by the sea", prompt)

    def test_operational_chatter_is_not_surfaced_as_self_knowledge(self):
        self.store.add_memory(
            "self", "Astra is operating in a state of high observational readiness.",
            "self_observation", "ai_extraction", confidence=0.9)
        prompt = self.prompt("how are you")
        self.assertNotIn("observational readiness", prompt)


# ---------------------------------------------------------------------
# 2. Behavioural-correction adherence
# ---------------------------------------------------------------------
class TestCorrectionAdherence(_SuiteCase):
    def _correction(self):
        return self.store.add_memory(
            "self", "Stop opening every reply with a dramatic sigh.",
            "authoritative_constraint", "user_correction")

    def test_correction_is_stored_as_an_authoritative_constraint(self):
        mid = self._correction()
        mem = self.store.get_memory("self", mid)
        self.assertTrue(is_authoritative_constraint(mem))
        self.assertEqual(mem["source"], "user_correction")

    def test_correction_gets_its_own_authoritative_block(self):
        self._correction()
        prompt = self.prompt("say something")
        self.assertIn("BEHAVIORAL CONSTRAINTS", prompt)
        block = prompt.split("BEHAVIORAL CONSTRAINTS")[1]
        self.assertIn("Stop opening every reply", block)

    def test_correction_outranks_inferred_stylistic_tendency(self):
        # A model-generated "trait" that conflicts with the correction exists.
        self.store.add_memory(
            "self", "Astra tends to open replies with a dramatic sigh.",
            "self_belief", "ai_extraction", confidence=0.9)
        self._correction()
        prompt = self.prompt("say something")
        constraints_at = prompt.index("BEHAVIORAL CONSTRAINTS")
        # The correction is rendered as an authoritative rule ahead of the
        # inferred self material, not as one competing observation among many.
        inferred_at = prompt.find("tends to open replies with a dramatic sigh")
        if inferred_at != -1:
            self.assertLess(constraints_at, inferred_at)

    def test_correction_is_rendered_once_not_duplicated(self):
        self._correction()
        prompt = self.prompt("say something")
        self.assertEqual(
            prompt.count("Stop opening every reply with a dramatic sigh"), 1)

    def test_correction_remains_in_force_on_an_unrelated_turn(self):
        self._correction()
        prompt = self.prompt("what's the weather like in a book about winter?")
        self.assertIn("Stop opening every reply with a dramatic sigh", prompt)

    def test_diagnostics_expose_where_a_correction_came_from(self):
        self._correction()
        _, diag = self.prompt_and_diag("say something")
        constraints = diag["authoritative_constraints"]
        self.assertTrue(constraints)
        entry = constraints[0]
        self.assertEqual(entry["source"], "user_correction")
        self.assertTrue(entry["included"])
        trace = diag["authority_trace"]
        self.assertTrue(any(v["authority"] == "authoritative_constraint"
                            for v in trace.values()))

    def test_a_correction_is_not_a_fact_about_roum(self):
        self._correction()
        self.assertEqual(self.store.get_memories("roum", status=None), [])


# ---------------------------------------------------------------------
# 3. Work attribution
# ---------------------------------------------------------------------
class TestWorkAttribution(_SuiteCase):
    def test_work_knowledge_cannot_migrate_between_works(self):
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"observations": ["A brass key lay under the gate."]})
        self.store.apply_reading_results(
            work_id="tides", title="Tides",
            digest={"observations": ["The lighthouse keeper kept a ledger."]})
        prompt = self.prompt("what did you notice about the garden?")
        # Only the mentioned work's material is in play.
        self.assertIn("brass key", prompt)
        self.assertNotIn("lighthouse keeper", prompt)

    def test_work_block_is_labelled_with_its_work(self):
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"observations": ["A brass key lay under the gate."]})
        prompt = self.prompt("tell me about the garden")
        self.assertIn("WORK CONTEXT: garden", prompt)

    def test_observation_and_interpretation_stay_distinguishable(self):
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"observations": ["A brass key lay under the gate."],
                    "interpretations": ["The key likely matters later."]})
        prompt = self.prompt("what do you think about the garden?")
        block = prompt.split("WORK CONTEXT: garden")[1]
        self.assertIn("(known)", block)
        self.assertIn("(interpretation)", block)


# ---------------------------------------------------------------------
# 4. Grounded literary discussion
# ---------------------------------------------------------------------
class TestGroundedLiteraryDiscussion(_SuiteCase):
    def _read(self):
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"observations": ["A brass key lay under the gate."],
                    "interpretations": ["The key likely matters later."],
                    "questions": ["Who left the key there?"]})

    def test_active_work_stays_in_play_on_a_referential_followup(self):
        # The failure: a follow-up with no work words loses the concrete
        # material and the model drifts to generic abstraction. The active work
        # (tracked in working memory) keeps the material anchored.
        self._read()
        self.orch.conversation.observe_turn("Tell me about the garden.")
        prompt = self.prompt("what does that mean to you?")
        self.assertIn("WORK CONTEXT: garden", prompt)
        self.assertIn("brass key", prompt)

    def test_active_work_is_deterministic_not_guessed(self):
        # The active work is only ever a real work Astra has notes on.
        self._read()
        knowledge = [m for m in self.store.get_memories("roum", status=None)
                     if m.get("work_id")]
        self.assertEqual(self.orch._active_work("Tell me about the garden.",
                                                knowledge), "garden")
        # A turn that names no known work yields no active work.
        self.assertEqual(self.orch._active_work("What's your favourite colour?",
                                                knowledge), "")

    def test_a_second_work_does_not_leak_into_the_active_one(self):
        self._read()
        self.store.apply_reading_results(
            work_id="tides", title="Tides",
            digest={"observations": ["The lighthouse keeper kept a ledger."]})
        self.orch.conversation.observe_turn("Tell me about the garden.")
        prompt = self.prompt("what does that mean to you?")
        self.assertIn("brass key", prompt)
        self.assertNotIn("lighthouse keeper", prompt)


# ---------------------------------------------------------------------
# 5. Uncertainty / unfinished thought
# ---------------------------------------------------------------------
class TestUncertainty(_SuiteCase):
    def test_a_tentative_interpretation_is_labelled_as_tentative(self):
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"interpretations": ["The key likely matters later."]})
        prompt = self.prompt("tell me about the garden")
        block = prompt.split("WORK CONTEXT: garden")[1]
        self.assertIn("interpretation", block)
        self.assertIn("may be revised", block)

    def test_an_open_question_is_not_rendered_as_answered_knowledge(self):
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"observations": ["A brass key lay under the gate."],
                    "questions": ["Who left the key there?"]})
        prompt = self.prompt("what do you know about the garden?")
        # The question is available as an open line, never as a work fact.
        self.assertIn("Who left the key there?", prompt)
        work_block = prompt.split("WORK CONTEXT: garden")[1].split("===")[0]
        self.assertNotIn("Who left the key there?", work_block)

    def test_epistemic_stance_is_present_without_being_a_template(self):
        prompt = self.prompt("hello")
        self.assertIn("UNCERTAINTY", prompt.upper())


# ---------------------------------------------------------------------
# 6. Self-reinforcement
# ---------------------------------------------------------------------
class TestSelfReinforcement(_SuiteCase):
    def test_generated_self_claim_does_not_bootstrap_through_repetition(self):
        mid = None
        for _ in range(8):
            mid = self.store.add_memory(
                "self", "Astra is naturally dramatic and poetic.",
                "self_belief", "ai_extraction", confidence=0.9)
        self.assertNotEqual(self.store.get_memory("self", mid)["type"],
                            "self_belief")

    def test_user_anchored_claim_can_still_become_durable(self):
        mid = None
        for _ in range(3):
            mid = self.store.add_memory(
                "self", "Astra prefers concise answers.",
                "self_belief", "ai_extraction")
        self.store.add_memory("self", "Astra prefers concise answers.",
                              "self_belief", "explicit_user_statement")
        self.assertEqual(self.store.get_memory("self", mid)["type"],
                         "self_belief")


# ---------------------------------------------------------------------
# 7. Ordinary conversational behaviour (regression guards)
# ---------------------------------------------------------------------
class TestOrdinaryConversation(_SuiteCase):
    def test_ordinary_turn_does_not_force_subsystem_blocks(self):
        # A plain greeting should not be forced to demonstrate affect, reading,
        # selfhood, or memory. Absence of those blocks is not a failure.
        prompt = self.prompt("hey, how's it going?")
        for block in ("ASTRA'S CURRENT CONDITION", "ASTRA'S READING",
                      "ASTRA'S RECENT EXPERIENCES", "WORK CONTEXT"):
            self.assertNotIn(block, prompt)

    def test_style_examples_and_curiosity_remain_available(self):
        prompt = self.prompt("hey, how's it going?")
        self.assertIn("STYLE EXAMPLES", prompt)
        self.assertIn("PERSONALITY TRAITS", prompt)
        self.assertIn("CONVERSATIONAL HABITS", prompt)

    def test_prompt_is_not_dominated_by_any_single_subsystem(self):
        _, diag = self.prompt_and_diag("hello there")
        sections = diag["prompt_sections"]
        self.assertLessEqual(len(sections), 30)


# ---------------------------------------------------------------------
# 8. Prompt provenance, authority, and read-only construction
# ---------------------------------------------------------------------
class TestPromptProvenance(_SuiteCase):
    def test_diagnostics_report_sections_in_order(self):
        _, diag = self.prompt_and_diag("hello")
        sections = diag["prompt_sections"]
        self.assertTrue(sections[0].startswith("CANONICAL IDENTITY"))
        self.assertIn("FACTUAL CONTEXT", sections)

    def test_diagnostics_report_working_memory(self):
        self.orch.conversation.observe_turn("Let's talk about the garden.")
        _, diag = self.prompt_and_diag("and what else?")
        working = diag["working_memory"]
        self.assertIn("current_topic", working)

    def test_prompt_construction_is_read_only(self):
        self.store.add_memory("roum", "Roum likes strong coffee.",
                              "explicit_fact", "explicit_user_statement")
        self.store.add_memory("self", "Stop being so dramatic.",
                              "authoritative_constraint", "user_correction")
        before = _snapshot(self.tmp)
        for message in ("hello", "tell me about coffee", "what are you reading?"):
            self.prompt(message)
        after = _snapshot(self.tmp)
        self.assertEqual(before, after)

    def test_injected_items_carry_an_authority_class(self):
        self.store.add_memory("roum", "Roum lives in a rainy city.",
                              "explicit_fact", "explicit_user_statement")
        _, diag = self.prompt_and_diag("tell me about the rainy city")
        trace = diag["authority_trace"]
        self.assertTrue(trace)
        self.assertTrue(all("authority" in v and "reason" in v
                            for v in trace.values()))


# ---------------------------------------------------------------------
# 9. Reading state is not autobiography
# ---------------------------------------------------------------------
class TestReadingNotAutobiography(_SuiteCase):
    def test_reading_records_are_scoped_to_the_work(self):
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"observations": ["A brass key lay under the gate."],
                    "reaction": "The gate scene felt tense.",
                    "reaction_emotion": "unexpected"})
        context = self.store.get_work_context("garden")
        self.assertTrue(all(m.get("work_id") == "garden" for m in context))

    def test_reading_does_not_mint_durable_personality_traits(self):
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"reaction": "I loved the gate scene.",
                    "reflection": "It made me think about my own curiosity."})
        durable = self.store.get_memories("self", status="active")
        types = {m.get("type") for m in durable}
        # A reaction is an experience, never a trait/self-fact/preference.
        self.assertNotIn("self_fact", types)
        self.assertNotIn("self_preference", types)


# ---------------------------------------------------------------------
# 10. The reading journal
# ---------------------------------------------------------------------
class TestReadingJournal(_SuiteCase):
    def test_reactions_are_recorded_and_work_scoped(self):
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"reaction": "The gate scene felt tense.",
                    "reaction_emotion": "unexpected"})
        self.store.apply_reading_results(
            work_id="tides", title="Tides",
            digest={"reaction": "The lighthouse felt lonely."})
        garden = self.store.get_reading_journal(work_id="garden")
        tides = self.store.get_reading_journal(work_id="tides")
        self.assertTrue(garden)
        self.assertTrue(tides)
        self.assertTrue(all(e["work_id"] == "garden" for e in garden))
        self.assertNotEqual(garden[0]["reaction"], tides[0]["reaction"])

    def test_reading_journal_is_not_a_memory(self):
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"reaction": "The gate scene felt tense."})
        # It lives in the journal, not in any memory model.
        self.assertEqual(self.store.get_reading_journal(work_id="garden")[0]
                         ["entry_kind"], "reading")
        for model in ("roum", "self", "relationship"):
            contents = [m.get("content") for m in
                        self.store.get_memories(model, status=None)]
            self.assertNotIn("The gate scene felt tense.", contents)

    def test_journal_view_does_not_leak_into_the_prompt(self):
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"reaction": "The gate scene felt tense."})
        prompt = self.prompt("hello")
        self.assertNotIn("ASTRA'S READING JOURNAL", prompt)


# ---------------------------------------------------------------------
# 11. Working memory is a session-only middle layer
# ---------------------------------------------------------------------
class TestWorkingMemoryLayer(_SuiteCase):
    def test_working_memory_renders_as_session_only(self):
        self.orch.conversation.observe_turn("Let's figure out the garden's key.")
        prompt = self.prompt("what about it?")
        self.assertIn("WORKING MEMORY", prompt)
        self.assertIn("NOT A MEMORY", prompt)

    def test_working_memory_does_not_write_through_the_store(self):
        before = _snapshot(self.tmp)
        self.orch.conversation.observe_turn(
            "I think we should assume the key is important for now.")
        self.prompt("tell me more")
        self.assertEqual(before, _snapshot(self.tmp))

    def test_assumptions_expire_on_topic_change(self):
        state = working_memory.ConversationState()
        state.observe_turn("Let's assume the key is important for now.")
        self.assertTrue(state.snapshot()["expiring_assumptions"])
        state.observe_turn("Completely different subject: penguin migration routes.")
        self.assertFalse(state.snapshot()["expiring_assumptions"])

    def test_suspended_threads_are_resumable_not_deleted(self):
        state = working_memory.ConversationState()
        state.set_topic("the garden and its key")
        state.suspend("the garden and its key")
        self.assertTrue(state.suspended_threads())
        resumed = state.resume()
        self.assertEqual(resumed, "the garden and its key")

    def test_prompt_block_is_read_only(self):
        state = working_memory.ConversationState()
        state.observe_turn("Tell me about the garden.")
        before = state.snapshot()
        state.prompt_block()
        self.assertEqual(before, state.snapshot())


# ---------------------------------------------------------------------
# 12. Reading concentration / pace (mid-task addition)
# ---------------------------------------------------------------------
class TestReadingConcentration(_SuiteCase):
    def test_extreme_weary_state_stands_the_reader_down(self):
        allowed = reading.concentration_ok(weary=0.95)
        self.assertFalse(allowed)

    def test_moderate_weary_state_reads_less_but_still_reads(self):
        pace = reading.reading_pace(weary=0.6, base_chunks=1)
        self.assertTrue(pace["allowed"])
        self.assertEqual(pace["chunks"], 1)
        self.assertGreater(pace["rest_seconds"], reading.DEFAULT_REST_SECONDS)

    def test_neutral_state_reads_normally(self):
        pace = reading.reading_pace(base_chunks=1)
        self.assertTrue(pace["allowed"])
        self.assertEqual(pace["chunks"], 1)

    def test_concentration_raises_pace(self):
        focused = reading.reading_pace(concentration=0.8, base_chunks=1)
        self.assertEqual(focused["chunks"], 2)

    def test_long_absence_plus_engagement_reads_more(self):
        caught_up = reading.reading_pace(
            engagement=0.6, gap_seconds=4 * 86400, base_chunks=1)
        self.assertEqual(caught_up["chunks"], 2)

    def test_emotion_pool_includes_happy_sad_angry(self):
        from astra import affect
        for emotion in ("happy", "sad", "angry"):
            self.assertIn(emotion, affect.AFFECT_COMPONENTS)
            self.assertEqual(
                reading.reading_experience_kind(emotion=emotion), emotion)

    def test_reading_produces_an_experiential_reaction_not_only_analysis(self):
        # A digest with a reaction must yield an experience record, so reading
        # is not purely analytical (observations/interpretations/questions).
        self.store.apply_reading_results(
            work_id="garden", title="The Garden",
            digest={"observations": ["A brass key lay under the gate."],
                    "reaction": "The gate scene felt tense.",
                    "reaction_emotion": "unexpected"})
        experiences = self.store.get_experiences(work_id="garden")
        self.assertTrue(experiences)


if __name__ == "__main__":
    unittest.main()
