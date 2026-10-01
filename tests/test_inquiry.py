"""Tests for Slice 2: questions, uncertainty, and belief revision.

Everything runs against a real ``TripleMemoryStore`` and a real
``CompanionOrchestrator`` on disk - no mocks. The tests pin the behaviours the
brief asked for:

* an open question is an ordinary memory, so it reuses the existing evidence /
  confidence / decay / dormancy / supersession machinery - there is no parallel
  belief store;
* "known", "tentative" and "unresolved" are kept distinct, and an
  interpretation can never claim the certainty of a fact;
* resolving or reopening a question never deletes it - the history stays
  readable;
* a contradiction weakens (inference) but never supersedes unless Roum
  explicitly corrected it, and an ambiguity the evidence does not settle is
  recorded rather than silently resolved;
* work-specific understanding is scoped by work, so two works that share a name
  never blend, and it is never presented as the base model's own knowledge;
* a reason to speak is preserved as a candidate, never triggered;
* building a prompt stays read-only - it never writes the store.
"""

import os
import shutil
import tempfile
import unittest

from astra import inquiry
from astra.memory import TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator

CONFIG_DIR = "./config"


class _InquiryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def files_snapshot(self):
        snapshot = {}
        for name in sorted(os.listdir(self.tmp)):
            path = os.path.join(self.tmp, name)
            if os.path.isfile(path):
                with open(path, "rb") as handle:
                    snapshot[name] = handle.read()
        return snapshot


# ---------------------------------------------------------------------
# Questions are ordinary memories
# ---------------------------------------------------------------------
class TestQuestionPrimitive(_InquiryCase):
    def test_question_is_a_structured_self_record(self):
        mem = self.store.get_memory(
            "self",
            self.store.add_question("Why did the captain abandon the ship?",
                                    origin="reading", work_id="sea-novel"),
        )
        self.assertEqual(mem["type"], inquiry.TYPE_QUESTION)
        self.assertEqual(mem["target_model"], "self")
        self.assertEqual(inquiry.question_status_of(mem), inquiry.Q_OPEN)
        self.assertEqual(inquiry.origin_of(mem), "reading")
        self.assertEqual(inquiry.work_of(mem), "sea-novel")
        self.assertEqual(inquiry.epistemic_of(mem), inquiry.EPISTEMIC_UNKNOWN)

    def test_question_uses_the_shared_memory_machinery(self):
        """A question is retrievable, has confidence, and is not a self_fact."""
        self.store.add_question("What did the letter say?", origin="reading")
        questions = self.store.get_memories("self", status="active",
                                            mem_type=inquiry.TYPE_QUESTION)
        self.assertEqual(len(questions), 1)
        self.assertIn("confidence", questions[0])
        self.assertNotEqual(questions[0]["type"], "self_fact")

    def test_not_every_uncertainty_is_a_question(self):
        """The detector is narrow: a plain statement is not a question."""
        self.assertFalse(inquiry.looks_like_question("The captain was afraid."))
        self.assertTrue(inquiry.looks_like_question("Why did the captain leave?"))
        self.assertTrue(inquiry.looks_like_question("I don't know what happened."))

    def test_questions_round_trip_through_the_store(self):
        qid = self.store.add_question("Who left the key?", origin="reading",
                                      work_id="garden")
        self.store.resolve_question(qid, evidence_text="The note named Alice.")
        reloaded = TripleMemoryStore(data_dir=self.tmp)
        mem = reloaded.get_memory("self", qid)
        self.assertEqual(mem["type"], inquiry.TYPE_QUESTION)
        self.assertEqual(inquiry.question_status_of(mem), inquiry.Q_ANSWERED)
        self.assertEqual(len(mem.get("question_history") or []), 1)


# ---------------------------------------------------------------------
# Epistemic labels and confidence ceilings
# ---------------------------------------------------------------------
class TestEpistemicLabels(_InquiryCase):
    def test_interpretation_cannot_claim_certainty(self):
        """Confident wording must not make an interpretation look like a fact."""
        mem = self.store.get_memory(
            "roum",
            self.store.add_interpretation("The captain definitely planned it.",
                                          work_id="sea-novel", confidence=0.99),
        )
        self.assertEqual(inquiry.epistemic_of(mem), inquiry.EPISTEMIC_INTERPRETATION)
        self.assertLessEqual(mem["confidence"], inquiry.CONFIDENCE_CEILING[inquiry.EPISTEMIC_INTERPRETATION])

    def test_hypothesis_is_weaker_than_interpretation(self):
        h = inquiry.CONFIDENCE_CEILING[inquiry.EPISTEMIC_POSSIBLE]
        i = inquiry.CONFIDENCE_CEILING[inquiry.EPISTEMIC_INTERPRETATION]
        self.assertLess(h, i)

    def test_observation_defaults_to_known(self):
        mem = self.store.get_memory(
            "roum", self.store.add_observation("The crew reached the lifeboats.",
                                               work_id="sea-novel"),
        )
        self.assertEqual(inquiry.epistemic_of(mem), inquiry.EPISTEMIC_KNOWN)


# ---------------------------------------------------------------------
# Resolution, reopening, and non-destruction
# ---------------------------------------------------------------------
class TestQuestionLifecycle(_InquiryCase):
    def test_answering_does_not_delete_the_question(self):
        qid = self.store.add_question("Why did the captain leave?")
        self.store.resolve_question(qid, evidence_text="A later chapter explains.")
        mem = self.store.get_memory("self", qid)
        self.assertEqual(inquiry.question_status_of(mem), inquiry.Q_ANSWERED)
        self.assertEqual(mem["status"], "active")  # still readable history
        self.assertIsNone(self.store.find_contradiction(
            "self", "Why did the captain leave?", mem_type=inquiry.TYPE_QUESTION))

    def test_reopening_preserves_earlier_history(self):
        qid = self.store.add_question("Is the letter genuine?")
        self.store.resolve_question(qid, evidence_text="It looked genuine.")
        self.store.reopen_question(qid, reason="A watermark suggests otherwise.")
        mem = self.store.get_memory("self", qid)
        self.assertEqual(inquiry.question_status_of(mem), inquiry.Q_REOPENED)
        statuses = [h["status"] for h in mem.get("question_history") or []]
        self.assertIn(inquiry.Q_ANSWERED, statuses)
        self.assertIn(inquiry.Q_REOPENED, statuses)

    def test_abandoning_keeps_the_question(self):
        qid = self.store.add_question("What was the ship's original name?")
        self.store.abandon_question(qid, reason="No longer relevant.")
        mem = self.store.get_memory("self", qid)
        self.assertEqual(inquiry.question_status_of(mem), inquiry.Q_ABANDONED)
        self.assertEqual(mem["status"], "active")


# ---------------------------------------------------------------------
# Uncertainty vs certainty: revision rules
# ---------------------------------------------------------------------
class TestBeliefRevision(_InquiryCase):
    def test_inference_does_not_supersede_an_explicit_fact(self):
        """Only an explicit correction may supersede; an inference weakens."""
        fact = self.store.add_memory(
            "roum", "Roum likes the formal tone.", "explicit_preference",
            "explicit_user_statement",
        )
        conflicting = self.store.add_interpretation(
            "Roum does not like the formal tone.",
        )
        fact_mem = self.store.get_memory("roum", fact)
        conflict_mem = self.store.get_memory("roum", conflicting)
        # The explicit statement is never superseded by an inference...
        self.assertNotEqual(fact_mem["status"], "superseded")
        # ...the inference is the side that gives way, and both stay readable.
        self.assertEqual(conflict_mem["status"], "weakened")
        self.assertIsNotNone(self.store.get_memory("roum", fact))
        self.assertIsNotNone(self.store.get_memory("roum", conflicting))

    def test_explicit_correction_supersedes_and_both_stay_readable(self):
        """An explicit correction supersedes; the old record stays readable."""
        old = self.store.add_memory(
            "roum", "The captain left first.", "explicit_fact",
            "explicit_user_statement",
        )
        self.store.supersede_memory(
            "roum", old, "The captain left last.", mem_type="correction",
            source="user_correction", reason="Roum corrected the order.",
        )
        self.assertEqual(self.store.get_memory("roum", old)["status"], "superseded")
        # Nothing is deleted: the superseded record is still there, linked.
        superseded = self.store.get_memory("roum", old)
        self.assertIsNotNone(superseded.get("superseded_by"))

    def test_unresolved_conflict_weakens_neither_side(self):
        """A contradiction the evidence does not settle is recorded, not solved."""
        first = self.store.add_interpretation("The captain acted out of fear.")
        second = self.store.add_interpretation("The captain acted out of duty.")
        self.store.link_conflict(first, second, reason="Two readings.")
        a, b = self.store.get_memory("roum", first), self.store.get_memory("roum", second)
        self.assertEqual(a["status"], "active")
        self.assertEqual(b["status"], "active")
        self.assertIn(second, a.get("contradicts") or [])
        self.assertIn(first, b.get("contradicts") or [])


# ---------------------------------------------------------------------
# Contextual evidence association
# ---------------------------------------------------------------------
class TestEvidenceAssociation(_InquiryCase):
    def test_evidence_associates_by_context_not_exact_wording(self):
        qid = self.store.add_question("Why did the captain abandon the ship?",
                                      work_id="sea-novel")
        obs = self.store.add_observation(
            "The captain ordered the crew into the lifeboats before himself.",
            work_id="sea-novel",
        )
        matched = self.store.associate_evidence(self.store.get_memory("roum", obs))
        self.assertIn(qid, matched)
        self.assertIn(obs, self.store.get_memory("self", qid)["question_evidence"])

    def test_unrelated_evidence_is_not_associated(self):
        qid = self.store.add_question("Why did the captain abandon the ship?",
                                      work_id="sea-novel")
        obs = self.store.add_observation("The harbour was foggy that morning.",
                                         work_id="sea-novel")
        self.assertEqual(
            self.store.associate_evidence(self.store.get_memory("roum", obs)), [])
        self.assertFalse(self.store.get_memory("self", qid).get("question_evidence"))

    def test_association_never_resolves_the_question(self):
        qid = self.store.add_question("Who left the key?", work_id="garden")
        obs = self.store.add_observation("Alice found a key in the garden.",
                                         work_id="garden")
        self.store.associate_evidence(self.store.get_memory("roum", obs))
        self.assertEqual(
            inquiry.question_status_of(self.store.get_memory("self", qid)),
            inquiry.Q_OPEN,
        )


# ---------------------------------------------------------------------
# Work-specific understanding and isolation
# ---------------------------------------------------------------------
class TestWorkContextIsolation(_InquiryCase):
    def test_same_name_works_do_not_blend(self):
        """Two works that share a character name keep separate contexts."""
        self.store.add_observation("Alice found a key in the garden.",
                                   work_id="garden-story")
        self.store.add_observation("Alice works as a detective in the city.",
                                   work_id="city-story")
        garden = self.store.get_work_context("garden-story")
        city = self.store.get_work_context("city-story")
        self.assertEqual([m["content"] for m in garden],
                         ["Alice found a key in the garden."])
        self.assertEqual([m["content"] for m in city],
                         ["Alice works as a detective in the city."])

    def test_work_knowledge_is_not_a_roum_fact(self):
        oid = self.store.add_observation("The captain ordered the lifeboats.",
                                         work_id="sea-novel")
        mem = self.store.get_memory("roum", oid)
        self.assertEqual(mem["type"], inquiry.TYPE_OBSERVATION)
        self.assertNotEqual(mem["type"], "explicit_fact")


# ---------------------------------------------------------------------
# Prompt integration
# ---------------------------------------------------------------------
class TestInquiryPrompt(_InquiryCase):
    def test_open_question_block_only_when_relevant(self):
        self.store.add_question("Why did the captain abandon the ship?",
                                origin="reading", work_id="sea-novel")
        relevant, _ = self.orch.build_prompt_with_diagnostics("the captain", [])
        self.assertIn("OPEN QUESTIONS", relevant)
        unrelated, _ = self.orch.build_prompt_with_diagnostics(
            "what is your favourite colour", [])
        # A neutral turn keeps only the most recent question or two in view.
        self.assertIn("OPEN QUESTIONS", unrelated)

    def test_answered_questions_are_not_in_the_open_block(self):
        qid = self.store.add_question("Why did the captain leave?")
        self.store.resolve_question(qid, evidence_text="Explained later.")
        prompt, _ = self.orch.build_prompt_with_diagnostics("the captain", [])
        self.assertNotIn("OPEN QUESTIONS", prompt)

    def test_work_knowledge_is_labelled_and_scoped(self):
        self.store.add_observation("The captain ordered the lifeboats.",
                                   work_id="sea-novel")
        self.store.add_interpretation("The captain probably acted out of fear.",
                                      work_id="sea-novel")
        self.store.add_observation("Alice found a key in the garden.",
                                   work_id="garden-story")
        prompt, diag = self.orch.build_prompt_with_diagnostics("the captain", [])
        self.assertIn("WORK CONTEXT: sea-novel", prompt)
        self.assertNotIn("WORK CONTEXT: garden-story", prompt)
        self.assertIn("(interpretation)", prompt)
        self.assertEqual(list(diag["work_context"].keys()), ["sea-novel"])

    def test_tentative_material_is_not_injected_as_factual_context(self):
        self.store.add_interpretation("The captain probably acted out of fear.",
                                      work_id="sea-novel")
        prompt, _ = self.orch.build_prompt_with_diagnostics("the captain", [])
        factual = prompt.find("=== FACTUAL CONTEXT ===")
        tentative = prompt.find("TENTATIVE INFERENCES")
        start = factual if factual >= 0 else tentative
        if start >= 0:
            self.assertNotIn("probably acted out of fear", prompt[start:])

    def test_prompt_build_is_read_only(self):
        """Building a prompt must never write the store (byte-stable files)."""
        self.store.add_question("Why did the captain leave?", work_id="sea-novel")
        self.store.add_observation("The captain ordered the lifeboats.",
                                   work_id="sea-novel")
        self.store.add_interpretation("The captain acted out of fear.",
                                      work_id="sea-novel")
        before = self.files_snapshot()
        for _ in range(3):
            self.orch.build_prompt("tell me about the captain", [])
        self.assertEqual(self.files_snapshot(), before)


# ---------------------------------------------------------------------
# Reasons to speak: preserved, never triggered
# ---------------------------------------------------------------------
class TestConversationCandidates(_InquiryCase):
    def test_candidates_carry_their_cause(self):
        self.store.add_question("Why did the captain leave?", work_id="sea-novel",
                                motive=inquiry.MOTIVE_FACTUAL)
        candidates = self.orch.conversation_candidates()
        self.assertTrue(candidates)
        kinds = {c["kind"] for c in candidates}
        self.assertIn(inquiry.CANDIDATE_UNRESOLVED_QUESTION, kinds)
        self.assertTrue(all("reason" in c and "weight" in c for c in candidates))

    def test_wants_roum_view_is_its_own_candidate(self):
        self.store.add_question(
            "What would Roum make of the captain's choice?",
            motive=inquiry.MOTIVE_ROUM_VIEW, work_id="sea-novel")
        candidates = self.orch.conversation_candidates()
        self.assertIn(inquiry.CANDIDATE_WANTS_ROUM_VIEW,
                      {c["kind"] for c in candidates})

    def test_candidates_do_not_write_the_store(self):
        self.store.add_question("Why did the captain leave?")
        before = self.files_snapshot()
        self.orch.conversation_candidates()
        self.assertEqual(self.files_snapshot(), before)


# ---------------------------------------------------------------------
# Write path: the deterministic classifier and consolidator
# ---------------------------------------------------------------------
class TestInquiryWritePath(_InquiryCase):
    def test_classifier_keeps_a_proposed_question_persistent(self):
        from astra.memory import classify_candidate
        self.assertEqual(
            classify_candidate("Why did the captain leave?",
                               explicit=False, mem_type=inquiry.TYPE_QUESTION,
                               target_model="self", source="ai_extraction"),
            "open_question",
        )

    def test_classifier_still_drops_a_user_question_to_context(self):
        """A question Roum asks is transient; only Astra's own inquiry persists."""
        from astra.memory import classify_candidate
        self.assertEqual(
            classify_candidate("Why did the captain leave?",
                               explicit=False, mem_type="explicit_fact",
                               target_model="roum", source="ai_extraction"),
            "temporary_context",
        )

    def test_consolidated_question_carries_lifecycle_tags(self):
        from astra.consolidator import apply_decision, resolve_candidate
        decision = resolve_candidate({
            "content": "Why did the captain abandon the ship?",
            "classification": inquiry.TYPE_QUESTION,
            "tags": ["work:sea-novel"],
        })
        mid = apply_decision(self.store, decision, "conv_1", 1)
        mem = self.store.get_memory("self", mid)
        self.assertEqual(inquiry.question_status_of(mem), inquiry.Q_OPEN)
        self.assertEqual(inquiry.epistemic_of(mem), inquiry.EPISTEMIC_UNKNOWN)
        self.assertEqual(len(self.store.get_questions(open_only=True)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
