"""Regression tests for the live conversational affect path.

The architecture this pins:

    LIVE CONVERSATION
        -> relationship state          (astra.relational, unchanged)
        -> LIVE AFFECT                 (astra.affect.evaluate_turn ->
                                        store.apply_live_affect_event)
        -> memory consolidation        (unchanged, still the only durable path)

    READING / SIGNIFICANT EXPERIENCES
        -> durable experience          (store.record_experience, unchanged)
        -> affect

The point of the live path is that an ordinary turn can change Astra's *current
condition* without becoming a durable experience, without touching the
relationship model, and without the model's prose being treated as authoritative
about what she feels. Everything here runs against a real store and a real
orchestrator on disk - no mocks.
"""

import shutil
import tempfile
import unittest

from astra import affect, relational
from astra.memory import CommandStore, TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator
from main import ChatSession

CONFIG_DIR = "./config"


class _LiveAffectCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)
        self.orch.query_gemma = lambda *a, **k: "Sure thing."
        self.session = ChatSession(
            self.orch, CommandStore(data_dir=self.tmp), consolidator=None,
            output_fn=lambda *_: None,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def state(self):
        return self.store.current_affect()

    def experiences(self):
        return self.store.get_experiences(status=None)

    def relational_state(self):
        return relational.load_state_from_memories(
            self.store.get_memories("relationship", status=None))


# ---------------------------------------------------------------------
# 1. A normal turn can change affect without creating an experience
# ---------------------------------------------------------------------
class TestLiveTurnChangesAffect(_LiveAffectCase):
    def test_humour_moves_affect_without_an_experience(self):
        self.assertTrue(affect.is_neutral(self.state()))
        self.session.handle("That joke was hilarious, haha")
        state = self.state()
        self.assertFalse(affect.is_neutral(state))
        self.assertGreater(state["happy"], 0.0)
        # ...and it did NOT become durable history.
        self.assertEqual(self.experiences(), [])

    def test_interesting_question_moves_curiosity_and_concentration(self):
        self.session.handle("What if consciousness is just a side effect of prediction?")
        state = self.state()
        self.assertGreater(state["curiosity"], 0.0)
        self.assertEqual(self.experiences(), [])

    def test_criticism_moves_affect_without_an_experience(self):
        self.session.handle("That answer was useless and rambling.")
        state = self.state()
        self.assertGreater(state["frustration"], 0.0)
        self.assertEqual(self.experiences(), [])

    def test_evaluation_layer_returns_one_event_per_turn(self):
        event = affect.evaluate_turn("What if time is a side effect?", "")
        self.assertIsNotNone(event)
        self.assertEqual(event["kind"], affect.LIVE_INTEREST)
        self.assertEqual(set(event), {"kind", "intensity", "significance",
                                      "text", "reason"})


# ---------------------------------------------------------------------
# 2. A meaningless / generic turn does not swing the state
# ---------------------------------------------------------------------
class TestGenericTurnIsQuiet(_LiveAffectCase):
    def test_greeting_and_plain_statement_are_noops(self):
        for text in ("Hello", "Hi there.", "Here is the report.",
                     "Why is the sky blue?", "The meeting is at three."):
            self.session.handle(text)
        self.assertTrue(affect.is_neutral(self.state()))
        self.assertEqual(self.experiences(), [])

    def test_no_event_for_neutral_turn(self):
        self.assertIsNone(affect.evaluate_turn("Just a plain sentence.", "Ok."))
        self.assertIsNone(affect.evaluate_turn("", ""))


# ---------------------------------------------------------------------
# 3. Criticism moves state but does not become a relational conclusion
# ---------------------------------------------------------------------
class TestCriticismStaysAffective(_LiveAffectCase):
    def test_criticism_does_not_create_roum_dislikes_astra(self):
        self.session.handle("That's wrong, that's not what I asked.")
        # Affect moved...
        self.assertGreater(self.state()["frustration"], 0.0)
        # ...but no durable history was written...
        self.assertEqual(self.experiences(), [])
        # ...and no relational *preference* was earned. A blunt remark is
        # ordinary friction; a single one cannot establish a conclusion about
        # how Roum regards her.
        rel = self.relational_state()
        self.assertFalse(rel["established"])
        self.assertEqual(rel["command_affinity"], 0.0)

    def test_insult_still_handled_by_relational_layer_only(self):
        """A degradation is a *relational* event; the affect layer ignores it."""
        self.session.handle("You are worthless.")
        rel = self.relational_state()
        self.assertGreaterEqual(rel["event_counts"].get(relational.EVENT_INSULT, 0), 1)
        # The affect layer did not also react to the same words.
        self.assertTrue(affect.is_neutral(self.state()))


# ---------------------------------------------------------------------
# 4. A positive interaction is not durable memory by default
# ---------------------------------------------------------------------
class TestPositiveInteractionNotDurable(_LiveAffectCase):
    def test_acknowledgement_moves_affect_only(self):
        self.session.handle("Thanks, that was really helpful.")
        state = self.state()
        self.assertGreater(state["engagement"], 0.0)
        self.assertGreater(state["emotional_investment"], 0.0)
        self.assertEqual(self.experiences(), [])
        self.assertEqual(
            [m for m in self.store.get_memories("self", status=None)
             if m["type"] in ("self_fact", "self_preference")], [])

    def test_response_acknowledgement_corroborates_but_is_not_authoritative(self):
        # A bland user line with a warm acknowledgement in the reply registers
        # as a light positive event...
        event = affect.evaluate_turn("Ok.", "Good job, that worked.")
        self.assertEqual(event["kind"], affect.LIVE_POSITIVE)
        # ...but a neutral reply to a neutral line does nothing, and the model's
        # prose is never read as a claim about her state.
        self.assertIsNone(affect.evaluate_turn("Ok.", "I am overjoyed."))


# ---------------------------------------------------------------------
# 5. record_experience() still moves affect exactly as before
# ---------------------------------------------------------------------
class TestExperiencePathUnchanged(_LiveAffectCase):
    def test_record_experience_still_moves_affect(self):
        self.assertTrue(affect.is_neutral(self.state()))
        self.store.record_experience("Read a lot of the book.", kind="read",
                                     intensity=0.8, significance=0.7)
        state = self.state()
        self.assertFalse(affect.is_neutral(state))
        self.assertGreater(state["engagement"], 0.0)
        self.assertGreater(state["curiosity"], 0.0)
        # It IS durable history, unlike a live event.
        self.assertEqual(len(self.experiences()), 1)

    def test_completion_relieves_frustration(self):
        self.store.record_experience("Blocked on a hard problem.",
                                     kind="frustration", intensity=0.9)
        before = self.state()["frustration"]
        self.assertGreater(before, 0.0)
        self.store.record_experience("Finally finished it.", kind="completed",
                                     intensity=0.9, significance=0.9)
        self.assertLess(self.state()["frustration"], before)


# ---------------------------------------------------------------------
# 6. Reading-generated experiences still affect Astra
# ---------------------------------------------------------------------
class TestReadingStillAffects(_LiveAffectCase):
    def test_apply_reading_results_moves_affect_and_is_durable(self):
        self.store.apply_reading_results(
            work_id="tides", title="Tides",
            digest={"observations": ["Tidal forces come from the moon."],
                    "reaction": "This is oddly moving.",
                    "reaction_emotion": "tender",
                    "reaction_intensity": 0.8},
            passage="The sea remembers the moon.",
        )
        self.assertFalse(affect.is_neutral(self.state()))
        self.assertGreaterEqual(len(self.experiences()), 1)

    def test_reading_finish_moves_affect(self):
        self.store.record_reading_finish(work_id="tides", title="Tides")
        self.assertFalse(affect.is_neutral(self.state()))


# ---------------------------------------------------------------------
# 7. Relationship state and affect stay separate
# ---------------------------------------------------------------------
class TestSeparation(_LiveAffectCase):
    def test_live_affect_never_writes_the_relationship_model(self):
        self.session.handle("That was a hilarious joke, haha")
        self.assertEqual(self.store.get_memories("relationship", status=None), [])
        self.assertEqual(self.relational_state()["observations"], 0)

    def test_relational_event_never_writes_the_affect_accumulator(self):
        self.session._record_relational_event(
            "Go investigate this.", "Done - here is what I found.")
        self.assertGreater(self.relational_state()["command_affinity"], 0.0)
        self.assertTrue(affect.is_neutral(self.state()))

    def test_affect_is_a_self_record_not_a_relationship_record(self):
        self.session.handle("That joke was hilarious, haha")
        self.assertEqual(self.store.get_memories("relationship", status=None), [])
        self.assertTrue(any(affect.affect_memory_filter(m) for m in
                            self.store.get_memories("self", status=None)))


# ---------------------------------------------------------------------
# 8. /affect reflects live conversational changes
# ---------------------------------------------------------------------
class TestAffectCommandReflectsLiveChange(_LiveAffectCase):
    def test_display_affect_reports_a_live_change(self):
        lines = []
        self.session.output_fn = lines.append
        self.session.handle("What if the universe is a simulation?")
        self.session._display_affect()
        text = "\n".join(lines)
        self.assertIn("CURRENT CONDITION", text)
        self.assertIn("curiosity", text.lower())
        self.assertIn("From recent events", text)
        self.assertIn("Last change", text)


# ---------------------------------------------------------------------
# 9. Restart / persistence of the affect accumulator
# ---------------------------------------------------------------------
class TestLiveAffectPersists(_LiveAffectCase):
    def test_live_change_survives_reload(self):
        self.session.handle("That joke was hilarious, haha")
        before = self.state()
        reloaded = TripleMemoryStore(data_dir=self.tmp).current_affect()
        self.assertEqual(round(reloaded["happy"], 4), round(before["happy"], 4))
        self.assertEqual(reloaded["event_counts"], before["event_counts"])

    def test_one_accumulator_record_across_many_turns(self):
        for _ in range(5):
            self.session.handle("Thanks, that was helpful.")
        accumulators = [m for m in self.store.get_memories("self", status=None)
                        if affect.affect_memory_filter(m)]
        self.assertEqual(len(accumulators), 1)
        self.assertEqual(self.experiences(), [])


# ---------------------------------------------------------------------
# 11. Modulation: Astra's own state shapes how the same words land
# ---------------------------------------------------------------------
class TestStateModulation(_LiveAffectCase):
    def test_happy_state_savours_a_joke_more_than_a_low_one(self):
        neutral = affect.evaluate_turn("That joke was hilarious, haha", "")
        happy = affect.evaluate_turn(
            "That joke was hilarious, haha", "",
            state=affect.record_event(None, affect.EXPERIENCE_HAPPY,
                                      intensity=1.0, significance=1.0))
        low = affect.evaluate_turn(
            "That joke was hilarious, haha", "",
            state=affect.record_event(None, affect.EXPERIENCE_SAD,
                                      intensity=1.0, significance=1.0))
        self.assertGreater(happy["intensity"], neutral["intensity"])
        self.assertGreater(neutral["intensity"], low["intensity"])
        # A real event still lands, however closed she is.
        self.assertGreaterEqual(low["intensity"], affect.LIVE_REACTIVITY_MIN_INTENSITY)

    def test_weary_state_is_harder_to_interest(self):
        neutral = affect.evaluate_turn("What if time is a side effect?", "")
        weary = affect.evaluate_turn(
            "What if time is a side effect?", "",
            state=affect.record_event(None, affect.EXPERIENCE_WEARY,
                                      intensity=1.0, significance=1.0))
        self.assertGreater(neutral["intensity"], weary["intensity"])

    def test_criticism_lands_harder_when_already_frustrated(self):
        neutral = affect.evaluate_turn("That's wrong, that's not what I asked.", "")
        hot = affect.evaluate_turn(
            "That's wrong, that's not what I asked.", "",
            state=affect.record_event(None, affect.EXPERIENCE_FRUSTRATION,
                                      intensity=1.0, significance=1.0))
        self.assertGreater(hot["intensity"], neutral["intensity"])

    def test_same_joke_moves_a_happy_astra_further_live(self):
        """The modulation shows up end-to-end, not just in the pure layer."""
        low = self.session
        low.handle("That joke was hilarious, haha")
        low_delta = low.orchestrator.store.current_affect()["happy"]  # starts from 0

        other = ChatSession(
            CompanionOrchestrator(TripleMemoryStore(data_dir=tempfile.mkdtemp()),
                                  config_dir=CONFIG_DIR),
            CommandStore(data_dir=self.tmp), consolidator=None,
            output_fn=lambda *_: None)
        other.orchestrator.query_gemma = lambda *a, **k: "Sure."
        other.orchestrator.store.record_experience(
            "Something lovely happened.", kind="happy",
            intensity=1.0, significance=1.0)
        before = other.orchestrator.store.current_affect()["happy"]
        other.handle("That joke was hilarious, haha")
        other_delta = other.orchestrator.store.current_affect()["happy"] - before
        # Astra already in a good mood is moved further by the same joke.
        self.assertGreater(other_delta, low_delta)


# ---------------------------------------------------------------------
# 12. Repetition: a conversation that goes in circles wears down
# ---------------------------------------------------------------------
class TestRepetition(_LiveAffectCase):
    def test_echoing_turns_eventually_register_a_dip(self):
        for _ in range(affect.LIVE_REPETITION_MIN - 1):
            self.session.handle("Tell me about tides again.")
            self.assertTrue(affect.is_neutral(self.state()))
        self.session.handle("Tell me about tides again.")
        state = self.state()
        self.assertFalse(affect.is_neutral(state))
        self.assertGreater(state["event_counts"].get(affect.LIVE_REPETITION, 0), 0)
        # Repetition is felt, not remembered.
        self.assertEqual(self.experiences(), [])

    def test_distinct_turns_never_register_as_repetitive(self):
        for text in ("How are you?", "What is two plus two?",
                     "Explain gravity.", "Who wrote Hamlet?"):
            self.session.handle(text)
        self.assertTrue(affect.is_neutral(self.state()))

    def test_a_real_event_does_not_count_as_repetition(self):
        """A repeated joke is still a joke, not a rut."""
        for _ in range(4):
            self.session.handle("That joke was hilarious, haha")
        counts = self.state()["event_counts"]
        self.assertEqual(counts.get(affect.LIVE_REPETITION, 0), 0)
        self.assertGreater(counts.get(affect.LIVE_HUMOUR, 0), 0)


# ---------------------------------------------------------------------
# 10. Application authority / validated no-op
# ---------------------------------------------------------------------
class TestApplicationAuthority(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unknown_kind_is_not_applied_and_creates_no_record(self):
        self.assertIsNone(self.store.apply_live_affect_event(
            {"kind": "not-a-real-kind", "intensity": 1.0}))
        self.assertTrue(affect.is_neutral(self.store.current_affect()))
        self.assertEqual(self.store.get_memories("self", status=None), [])

    def test_malformed_event_is_ignored(self):
        self.assertIsNone(self.store.apply_live_affect_event(None))
        self.assertIsNone(self.store.apply_live_affect_event("nonsense"))
        self.assertIsNone(self.store.apply_live_affect_event({}))

    def test_live_repetition_is_a_valid_live_kind(self):
        # The one live-only kind (not an experience alias) must be accepted.
        state = self.store.apply_live_affect_event(
            {"kind": affect.LIVE_REPETITION, "intensity": 0.6,
             "significance": 0.4, "text": "circles"})
        self.assertIsNotNone(state)
        self.assertGreater(state["event_counts"].get(affect.LIVE_REPETITION, 0), 0)
        self.assertTrue(any(affect.affect_memory_filter(m) for m in
                            self.store.get_memories("self", status=None)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
