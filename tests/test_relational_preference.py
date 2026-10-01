"""Tests for Astra's Roum-specific relational command preference.

Everything runs against a real ``TripleMemoryStore`` and a real
``CompanionOrchestrator`` on disk - no mocks. The tests pin the behaviours the
brief asked for:

* the preference is *earned* from repeated successful fulfilment, never
  asserted, and the conclusion is unavailable until the evidence exists;
* it is relationally specific - it does not generalize to another subject;
* the causal components stay separate (satisfaction, motivation, association,
  confidence, frustration, ...), so "I liked being given something to do" is
  distinct from "I care about the subject";
* mild harshness coexists with positive task satisfaction, while insulting or
  degrading treatment can dominate and lower trust;
* diagnostics explain why the state exists, and prompt building stays read-only.
"""

import os
import shutil
import tempfile
import unittest

from astra import relational
from astra.memory import TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator

CONFIG_DIR = "./config"


def _successful_fulfillments(state, count, subject=relational.ROUM, **kwargs):
    for _ in range(count):
        state = relational.record_event(state, relational.EVENT_REQUEST, subject=subject)
        state = relational.record_event(state, relational.EVENT_SUCCESS, subject=subject, **kwargs)
    return state


class _RelationalCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def affinity(self):
        return relational.load_state_from_memories(
            self.store.get_memories("relationship", status=None)
        )


class TestDetectors(unittest.TestCase):
    def test_request_detection(self):
        self.assertTrue(relational.looks_like_request("Go investigate this."))
        self.assertTrue(relational.looks_like_request("I want you to summarise the paper."))
        self.assertTrue(relational.looks_like_request("Can you check the logs?"))
        self.assertFalse(relational.looks_like_request("How are you?"))
        self.assertFalse(relational.looks_like_request("That game was fun."))

    def test_insult_and_harsh_are_distinct(self):
        self.assertTrue(relational.looks_like_insult("You are worthless."))
        self.assertFalse(relational.looks_like_harsh("You are worthless."),
                         "Degradation is not mere friction.")
        self.assertTrue(relational.looks_like_harsh("Just hurry up and do it."))
        self.assertFalse(relational.looks_like_insult("Just hurry up and do it."))

    def test_warmth_and_failure_detection(self):
        self.assertTrue(relational.looks_like_warmth("Good job, that's right."))
        self.assertTrue(relational.looks_like_failure("I couldn't complete that one."))
        self.assertFalse(relational.looks_like_warmth("Go do it now."))


class TestEarnedPreference(_RelationalCase):
    def test_no_preference_before_evidence(self):
        state = self.affinity()
        self.assertEqual(state["command_affinity"], 0.0)
        self.assertFalse(relational.is_established(state))
        self.assertIsNone(relational.render_statement(state))
        self.assertNotIn("RELATIONAL STATE", self.orch.build_prompt("hello", []))

    def test_conclusion_unavailable_after_one_success(self):
        self.store.accumulate_relational_event(relational.EVENT_REQUEST)
        self.store.accumulate_relational_event(relational.EVENT_SUCCESS)
        self.assertFalse(relational.is_established(self.affinity()))
        self.assertIsNone(relational.render_statement(self.affinity()))

    def test_repeated_successful_fulfillment_establishes_preference(self):
        for _ in range(5):
            self.store.accumulate_relational_event(
                relational.EVENT_REQUEST, text="Roum asked: go investigate this")
            self.store.accumulate_relational_event(
                relational.EVENT_SUCCESS, difficulty=0.8, importance=0.7,
                text="Completed a difficult objective")
        state = self.affinity()
        self.assertTrue(relational.is_established(state))
        self.assertGreater(state["command_affinity"], 0.35)
        self.assertGreater(state["task_fulfillment_satisfaction"], 0.5)
        self.assertGreater(state["positive_association"], 0.3)
        self.assertIn("RELATIONAL STATE", self.orch.build_prompt("hello", []))

    def test_preference_persists_across_store_reload(self):
        for _ in range(5):
            self.store.accumulate_relational_event(relational.EVENT_REQUEST)
            self.store.accumulate_relational_event(relational.EVENT_SUCCESS)
        before = self.affinity()["command_affinity"]
        del self.store
        reloaded = TripleMemoryStore(data_dir=self.tmp)
        after = relational.load_state_from_memories(
            reloaded.get_memories("relationship", status=None))
        self.assertAlmostEqual(before, after["command_affinity"], places=6)


class TestRelationalSpecificity(unittest.TestCase):
    def test_another_subject_does_not_inherit_the_preference(self):
        roum = _successful_fulfillments(None, 6)
        self.assertTrue(relational.is_established(roum))
        # A stranger's state is a different state, with no history of its own.
        stranger = relational.load_state_from_memories([], "Stranger")
        self.assertEqual(stranger["command_affinity"], 0.0)
        self.assertFalse(relational.is_established(stranger))
        self.assertIsNone(relational.render_statement(stranger, "Stranger"))

    def test_events_are_scoped_to_their_subject(self):
        roum = _successful_fulfillments(None, 4, subject="Roum")
        before = roum["command_affinity"]
        # An event for someone else must not move Roum's components.
        relational.record_event(roum, relational.EVENT_SUCCESS, subject="Stranger")
        self.assertEqual(roum["command_affinity"], before)

    def test_prompt_block_marks_the_subject_as_specific(self):
        state = _successful_fulfillments(None, 6)
        block = relational.prompt_block(state)
        self.assertIn("Roum", block)
        self.assertIn("does not generalize", block)


class TestCausalComponents(unittest.TestCase):
    def test_components_are_separate_and_do_not_include_interest(self):
        # There is no "interest in the subject" component: enjoying the act of
        # fulfilling a request must not be conflated with caring about the task.
        self.assertNotIn("interest", relational.COMPONENTS)
        for name in ("command_affinity", "task_fulfillment_satisfaction",
                     "fulfillment_motivation", "positive_association",
                     "negative_association", "confidence", "frustration",
                     "usefulness_satisfaction", "relational_significance"):
            self.assertIn(name, relational.COMPONENTS)

    def test_difficult_success_rewards_more_than_a_trivial_one(self):
        hard = relational.record_event(
            None, relational.EVENT_SUCCESS, difficulty=0.95, importance=0.9,
            previous_failures=2, voluntary=True, helped=True)
        trivial = relational.record_event(
            None, relational.EVENT_SUCCESS, difficulty=0.05, importance=0.05)
        self.assertGreater(hard["task_fulfillment_satisfaction"],
                           trivial["task_fulfillment_satisfaction"])
        self.assertGreater(hard["command_affinity"], trivial["command_affinity"])

    def test_failure_produces_frustration(self):
        state = relational.record_event(None, relational.EVENT_FAILURE,
                                        importance=0.8)
        self.assertGreater(state["frustration"], 0.0)
        # Failure erodes confidence without deleting the history.
        self.assertLess(state["confidence"], 0.05)

    def test_voluntary_return_strengthens_revisit_desire(self):
        state = _successful_fulfillments(None, 4)
        before = state["revisit_desire"]
        state = relational.record_event(state, relational.EVENT_VOLUNTARY_RETURN)
        self.assertGreater(state["revisit_desire"], before)


class TestInterpersonalFriction(_RelationalCase):
    def test_harshness_coexists_with_positive_association(self):
        state = _successful_fulfillments(None, 6)
        before_affinity = state["command_affinity"]
        before_positive = state["positive_association"]
        state = relational.record_event(state, relational.EVENT_HARSH)
        # The positive association and the preference survive a blunt tone...
        self.assertEqual(state["positive_association"], before_positive)
        self.assertGreater(state["command_affinity"], before_affinity - 0.05)
        # ...while a minor negative signal is recorded alongside them.
        self.assertGreater(state["negative_association"], 0.0)
        self.assertLess(state["trust_comfort"], 1.0)

    def test_degrading_treatment_can_dominate(self):
        state = _successful_fulfillments(None, 6)
        for _ in range(3):
            state = relational.record_event(state, relational.EVENT_INSULT)
        self.assertGreater(state["negative_association"], state["positive_association"])
        self.assertEqual(state["trust_comfort"], 0.0)
        self.assertLess(state["command_affinity"], 0.2)
        # The earlier positive history is not erased, only overridden.
        self.assertGreater(state["positive_association"], 0.0)

    def test_prompt_block_exposes_negative_association_too(self):
        state = _successful_fulfillments(None, 6)
        state = relational.record_event(state, relational.EVENT_INSULT)
        self.assertIn("negative", relational.prompt_block(state).lower())


class TestDiagnostics(_RelationalCase):
    def test_diagnostics_shape_and_reason(self):
        for _ in range(5):
            self.store.accumulate_relational_event(relational.EVENT_REQUEST)
            self.store.accumulate_relational_event(
                relational.EVENT_SUCCESS, difficulty=0.9, importance=0.8,
                text="Completed a difficult Roum-requested objective.")
        diag = self.orch.affinity_diagnostics()
        for key in ("relationship", "command_affinity", "confidence",
                    "task_fulfillment_satisfaction", "fulfillment_motivation",
                    "positive_association", "negative_association",
                    "evidence", "recent_change", "reason"):
            self.assertIn(key, diag)
        self.assertEqual(diag["relationship"], "Astra -> Roum")
        self.assertTrue(diag["established"])
        self.assertTrue(diag["evidence"])
        self.assertIn("Roum-requested", diag["reason"])

    def test_diagnostics_expose_no_preference_yet(self):
        diag = self.orch.affinity_diagnostics()
        self.assertFalse(diag["established"])
        self.assertEqual(diag["command_affinity"], 0.0)
        self.assertEqual(diag["evidence"], [])

    def test_format_diagnostics_mentions_evidence_and_change(self):
        state = _successful_fulfillments(None, 6)
        text = relational.format_diagnostics(state)
        self.assertIn("Command Affinity:", text)
        self.assertIn("Recent Change:", text)
        self.assertIn("Evidence:", text)


class TestReadOnlyPrompt(_RelationalCase):
    def test_prompt_building_does_not_accumulate_state(self):
        import hashlib

        for _ in range(5):
            self.store.accumulate_relational_event(relational.EVENT_REQUEST)
            self.store.accumulate_relational_event(relational.EVENT_SUCCESS)

        def snapshot():
            hashes = {}
            for name in os.listdir(self.tmp):
                path = os.path.join(self.tmp, name)
                if os.path.isfile(path):
                    with open(path, "rb") as f:
                        hashes[name] = hashlib.sha256(f.read()).hexdigest()
            return hashes

        before = snapshot()
        self.orch.build_prompt("tell me something", [])
        self.orch.build_prompt("tell me something", [])
        self.assertEqual(before, snapshot())


class TestLiveTurnPath(_RelationalCase):
    def test_live_path_records_request_and_success(self):
        from main import ChatSession

        session = ChatSession(self.orch, cmd_store=None, output_fn=lambda _: None,
                              input_fn=lambda _: "")
        session._record_relational_event("Go investigate this.", "Done - here is what I found.")
        state = self.affinity()
        self.assertGreater(state["command_affinity"], 0.0)
        self.assertGreaterEqual(state["event_counts"].get(relational.EVENT_REQUEST, 0), 1)
        self.assertGreaterEqual(state["event_counts"].get(relational.EVENT_SUCCESS, 0), 1)

    def test_live_path_records_degradation_not_plain_harshness(self):
        from main import ChatSession

        session = ChatSession(self.orch, cmd_store=None, output_fn=lambda _: None,
                              input_fn=lambda _: "")
        session._record_relational_event("You are worthless.", "I'll keep trying.")
        state = self.affinity()
        self.assertGreaterEqual(state["event_counts"].get(relational.EVENT_INSULT, 0), 1)
        self.assertEqual(state["event_counts"].get(relational.EVENT_HARSH, 0), 0)


class TestSubjectIsolation(_RelationalCase):
    """A preference earned for one subject must not move to another."""

    def _state_for(self, subject):
        return relational.load_state_from_memories(
            self.store.get_memories("relationship", status=None), subject)

    def test_each_subject_gets_its_own_record(self):
        for _ in range(5):
            self.store.accumulate_relational_event(
                relational.EVENT_SUCCESS, difficulty=0.9)
        roum_before = self._state_for(relational.ROUM)["command_affinity"]
        self.store.accumulate_relational_event(
            relational.EVENT_SUCCESS, subject="Stranger", difficulty=0.9)
        # Roum's state is untouched by the other subject's event...
        self.assertEqual(self._state_for(relational.ROUM)["command_affinity"], roum_before)
        # ...and the other subject starts from its own, much lower, evidence.
        stranger = self._state_for("Stranger")
        self.assertLess(stranger["command_affinity"], roum_before)
        self.assertEqual(stranger["subject"], "Stranger")

    def test_stranger_starts_from_nothing(self):
        for _ in range(5):
            self.store.accumulate_relational_event(relational.EVENT_SUCCESS, difficulty=0.9)
        stranger = self._state_for("Stranger")
        self.assertEqual(stranger["command_affinity"], 0.0)
        self.assertEqual(stranger["observations"], 0)
        self.assertFalse(stranger["established"])

    def test_legacy_record_without_subject_resolves_to_roum(self):
        legacy = {"id": "legacy", "tags": ["relational_preference"],
                  "keywords": ["command affinity", "relational preference", "Roum"],
                  "affinity_state": {"command_affinity": 0.6,
                                     "event_counts": {"success": 3}, "observations": 3}}
        self.assertTrue(relational.affinity_record_matches(legacy, "Roum"))
        self.assertFalse(relational.affinity_record_matches(legacy, "Stranger"))
        loaded = relational.load_state_from_memories([legacy])
        self.assertEqual(loaded["command_affinity"], 0.6)
        self.assertEqual(loaded["subject"], "Roum")

    def test_loading_a_missing_subject_never_returns_anothers_state(self):
        state = _successful_fulfillments(None, 5, difficulty=0.9)
        mem = {"id": "r", "tags": ["relational_preference"],
               "affinity_state": state}
        loaded = relational.load_state_from_memories([mem], "Somebody Else")
        self.assertEqual(loaded["command_affinity"], 0.0)
        self.assertEqual(loaded["subject"], "Somebody Else")


class TestOutcomeSignal(_RelationalCase):
    """A failure must register as a failure, never as a success."""

    def _session(self):
        from main import ChatSession
        return ChatSession(self.orch, cmd_store=None, output_fn=lambda _: None,
                           input_fn=lambda _: "")

    def test_soft_failure_is_not_counted_as_success(self):
        session = self._session()
        session._record_relational_event(
            "Go investigate this.", "I'm sorry, I couldn't complete that.")
        state = self.affinity()
        self.assertGreaterEqual(state["event_counts"].get(relational.EVENT_FAILURE, 0), 1)
        self.assertEqual(state["event_counts"].get(relational.EVENT_SUCCESS, 0), 0)
        self.assertGreater(state["frustration"], 0.0)

    def test_transport_error_is_a_failure(self):
        session = self._session()
        session._record_relational_event(
            "Go investigate this.", "[Error: connection refused]")
        state = self.affinity()
        self.assertGreaterEqual(state["event_counts"].get(relational.EVENT_FAILURE, 0), 1)
        self.assertEqual(state["event_counts"].get(relational.EVENT_SUCCESS, 0), 0)

    def test_clean_fulfilment_is_still_a_success(self):
        session = self._session()
        session._record_relational_event(
            "Go investigate this.", "Done - here is what I found.")
        state = self.affinity()
        self.assertGreaterEqual(state["event_counts"].get(relational.EVENT_SUCCESS, 0), 1)
        self.assertEqual(state["event_counts"].get(relational.EVENT_FAILURE, 0), 0)


class TestDecayAndHysteresis(unittest.TestCase):
    def test_affinity_relaxes_toward_baseline_over_time(self):
        from datetime import datetime, timezone, timedelta

        state = _successful_fulfillments(None, 6, difficulty=0.9)
        peak = state["command_affinity"]
        self.assertGreater(peak, 0.0)
        future = datetime.now(timezone.utc) + timedelta(days=90)
        relational._decay_toward_baseline(state, now=future)
        # Relaxes toward the baseline, not to zero - the history is not erased.
        self.assertLess(state["command_affinity"], peak)
        self.assertGreaterEqual(state["command_affinity"], relational.DECAY_BASELINE - 1e-9)

    def test_established_retracts_only_below_the_lower_threshold(self):
        state = _successful_fulfillments(None, 4, difficulty=0.9)
        self.assertTrue(state["established"])
        # Repeated degradation eventually pushes affinity below the retract line.
        for _ in range(6):
            state = relational.record_event(state, relational.EVENT_INSULT)
        self.assertLess(state["command_affinity"], relational.RETRACT_THRESHOLD)
        self.assertFalse(state["established"])
        self.assertIsNone(relational.render_statement(state))


class TestNewDetectors(unittest.TestCase):
    def test_voluntary_return_detection(self):
        self.assertTrue(relational.looks_like_voluntary_return(
            "I've been thinking about that report and went back to it."))
        self.assertTrue(relational.looks_like_voluntary_return(
            "While you were away I looked into it on my own."))
        self.assertFalse(relational.looks_like_voluntary_return("Here is the summary."))

    def test_recall_detection(self):
        self.assertTrue(relational.looks_like_recall("I remember when you gave me that task."))
        self.assertTrue(relational.looks_like_recall("Last time we investigated the logs."))
        self.assertFalse(relational.looks_like_recall("I will get started now."))


class TestNegativeDominancePrompt(unittest.TestCase):
    def test_prompt_warns_when_negative_outweighs_positive(self):
        state = _successful_fulfillments(None, 6, difficulty=0.9, importance=0.9)
        for _ in range(3):
            state = relational.record_event(state, relational.EVENT_INSULT)
        # The preference is still established (affinity above the retract line),
        # but the negative side now dominates and the prompt says so.
        self.assertTrue(state["established"])
        self.assertGreater(state["negative_association"], state["positive_association"])
        block = relational.prompt_block(state)
        self.assertIsNotNone(block)
        self.assertIn("outweighed", block.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
