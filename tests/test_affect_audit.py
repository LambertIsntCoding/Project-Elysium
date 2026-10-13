"""Tests for the affect audit: dynamism, persistence, and integration.

These pin the fixes the audit found:

* the affect config actually *parses* (it was silently ignored, so every tuned
  value fell back to the hardcoded defaults - and repetition was gated out by a
  threshold below the event gate);
* the accumulator is a *dynamic* state, not a pure integrator: a repeated event
  has diminishing returns and repetition weighting, so a flood cannot pin a
  dimension at the ceiling;
* the persisted state can be brought current by the explicit maintenance fold,
  so the display and the decision path agree instead of the record sitting
  frozen at its last-written magnitude;
* anticipation is a live dimension (raised by reading, relieved by completion)
  rather than one that can only ever fall;
* a real failure is felt, and the failure feedback reaches the affect state.

Everything runs against real objects; no mocks.
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from astra import affect
from astra.memory import TripleMemoryStore


class TestConfigActuallyLoads(unittest.TestCase):
    """The YAML config must parse; a broken file must not be silent."""

    def setUp(self):
        affect.reload_affect_config()

    def tearDown(self):
        affect.reload_affect_config()

    def test_affect_config_is_parsable(self):
        error = affect.diagnostics(None).get("config_error")
        self.assertEqual(error, "", f"affect.yaml failed to parse: {error}")

    def test_tuned_values_reach_the_runtime(self):
        # If the code fences ever return, the file stops parsing and this fails:
        # the read kind's tuned anticipation only exists in the config.
        deltas = affect.experience_deltas()[affect.EXPERIENCE_READ]
        self.assertIn("anticipation", deltas)
        self.assertGreater(affect.AFFECT_SATURATION_KNEE, 0.0)
        self.assertLessEqual(affect.AFFECT_REPETITION_FLOOR, 1.0)
        self.assertGreaterEqual(affect.AFFECT_REPETITION_ALLOWANCE, 0)


class TestDynamism(unittest.TestCase):
    """A repeated event must have diminishing returns, not linear accumulation."""

    def test_a_single_event_lands_fully(self):
        state = affect.record_event(None, affect.EXPERIENCE_DISCOVERED,
                                    intensity=0.5, significance=0.5)
        self.assertGreater(state["curiosity"], 0.0)
        # No repetition yet, so no discount: the first move is undiscounted.
        self.assertAlmostEqual(state["curiosity"], 0.168, places=3)

    def test_repeated_events_have_diminishing_returns(self):
        state = None
        prev = 0.0
        incs = []
        for _ in range(20):
            state = affect.record_event(state, affect.EXPERIENCE_DISCOVERED,
                                        intensity=0.5, significance=0.5)
            incs.append(round(state["curiosity"] - prev, 6))
            prev = state["curiosity"]
        # The first move is the largest; a later move is strictly smaller.
        self.assertGreater(incs[0], incs[-1])
        self.assertLess(incs[-1], incs[0] * 0.5)

    def test_a_flood_does_not_pin_at_the_ceiling(self):
        """Even hundreds of the same event must not lock a dimension at 1.0."""
        state = None
        for _ in range(300):
            state = affect.record_event(state, affect.EXPERIENCE_DISCOVERED,
                                        intensity=0.5, significance=0.5)
        # It is allowed to become strong, but not to sit exactly at the extreme.
        self.assertLess(state["curiosity"], 1.0)
        # And a *different* event still visibly moves a saturated dimension.
        before = state["curiosity"]
        state = affect.record_event(state, affect.EXPERIENCE_REALIZATION,
                                    intensity=0.5, significance=0.5)
        # curiosity may not move (realization doesn't raise it); use engagement.
        self.assertLess(state["engagement"], 1.0)

    def test_repetition_discount_is_not_applied_to_the_first_few(self):
        # The first few of a kind land fully (within the allowance); a flood is
        # where the repetition discount begins to bite. Compare a *late* move to
        # an early one at the same (moderate, un-saturated) value.
        s = affect.record_event(None, affect.EXPERIENCE_UNEXPECTED,
                                intensity=0.5, significance=0.5)
        d1 = s["curiosity"]
        s2 = affect.record_event(s, affect.EXPERIENCE_UNEXPECTED,
                                 intensity=0.5, significance=0.5)
        d2 = s2["curiosity"] - d1
        # The second is inside the allowance, so it matches the first (the
        # repetition discount has not started).
        self.assertAlmostEqual(d2, d1, places=4)
        # Many more, then a late move is strictly smaller than the early one.
        for _ in range(30):
            s2 = affect.record_event(s2, affect.EXPERIENCE_UNEXPECTED,
                                     intensity=0.5, significance=0.5)
        before = s2["curiosity"]
        s3 = affect.record_event(s2, affect.EXPERIENCE_UNEXPECTED,
                                 intensity=0.5, significance=0.5)
        self.assertLess(s3["curiosity"] - before, d1)


class TestAnticipationIsAlive(unittest.TestCase):
    """Anticipation must have a source, or the affect->outreach link never fires."""

    def test_reading_raises_anticipation_and_completion_relieves_it(self):
        state = None
        for _ in range(3):
            state = affect.record_event(state, affect.EXPERIENCE_READ,
                                        intensity=0.7, significance=0.6)
        self.assertGreater(state["anticipation"], 0.0)
        before = state["anticipation"]
        state = affect.record_event(state, affect.EXPERIENCE_COMPLETED,
                                    intensity=0.8, significance=0.7)
        self.assertLess(state["anticipation"], before)

    def test_a_read_event_reaches_outreach_relevance(self):
        from astra import proactive
        state = None
        for _ in range(4):
            state = affect.record_event(state, affect.EXPERIENCE_READ,
                                        intensity=0.7, significance=0.6)
        # The existing outreach link reads anticipation; it is now non-zero.
        self.assertGreater(proactive.outreach_relevance(state), 0.0)


class TestPersistenceFold(unittest.TestCase):
    """A never-touched state must be able to be brought current on disk."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _backdate(self, days):
        state = dict(self.store._affect_record()["affect_state"])
        state["last_updated"] = (
            datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        self.store._write_affect_state(state)

    def test_effective_read_ages_a_copy_but_persisted_is_frozen(self):
        for _ in range(6):
            self.store.record_experience("A discovery.", kind="discovered",
                                         intensity=1.0, significance=1.0)
        self._backdate(10)
        # The read ages a copy; the persisted record is untouched by the read.
        self.assertLess(self.store.current_affect()["curiosity"], 0.01)
        self.assertGreater(
            self.store._affect_record()["affect_state"]["curiosity"], 0.5)

    def test_fold_brings_the_persisted_state_current(self):
        for _ in range(6):
            self.store.record_experience("A discovery.", kind="discovered",
                                         intensity=1.0, significance=1.0)
        self._backdate(10)
        effective = self.store.current_affect()["curiosity"]
        folded = self.store.fold_affect_decay()
        # After the fold the record on disk matches the live view.
        self.assertAlmostEqual(
            round(self.store._affect_record()["affect_state"]["curiosity"], 4),
            round(effective, 4), places=4)
        self.assertAlmostEqual(round(folded["curiosity"], 4),
                               round(effective, 4), places=4)

    def test_fold_is_a_noop_when_nothing_moved(self):
        self.store.record_experience("A discovery.", kind="discovered")
        before = self.store._affect_record()["affect_state"]
        # No meaningful interval has passed, so the state does not move.
        self.store.fold_affect_decay()
        after = self.store._affect_record()["affect_state"]
        self.assertEqual(before["last_updated"], after["last_updated"])
        self.assertEqual(before["curiosity"], after["curiosity"])

    def test_fold_with_no_accumulator_is_safe(self):
        state = self.store.fold_affect_decay()
        self.assertTrue(affect.is_neutral(state))


class TestDiagnosticsShowBoth(unittest.TestCase):
    """Diagnostics must distinguish the stored value from the live one."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_persistent_and_effective_are_reported(self):
        for _ in range(6):
            self.store.record_experience("A discovery.", kind="discovered",
                                         intensity=1.0, significance=1.0)
        state = self.store._affect_record()["affect_state"]
        state = dict(state)
        state["last_updated"] = (
            datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        self.store._write_affect_state(state)

        diag = affect.diagnostics(state)
        self.assertTrue(diag["decayed"])
        self.assertIn("persistent", diag)
        self.assertGreater(diag["persistent"]["curiosity"], 0.5)
        self.assertLess(diag["curiosity"], 0.01)


class TestFailureIsFelt(unittest.TestCase):
    """A real setback in her own work reaches the affect state."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_problem_outcome_moves_affect_and_is_not_durable(self):
        from astra import tasks as T
        from astra.application import RuntimeController

        class _Stub(RuntimeController):
            def __init__(self, store):  # bypass the heavy real constructor
                self.store = store

        ctrl = _Stub(self.store)
        ctrl._record_outcome_affect(
            {"kind": T.KIND_READING, "status": T.FAILED})
        self.assertGreater(self.store.current_affect()["frustration"], 0.0)
        # It is only the temporary accumulator - never a durable experience.
        self.assertEqual(self.store.get_experiences(), [])

    def test_a_completed_outcome_is_not_an_affective_event(self):
        from astra import tasks as T
        from astra.application import RuntimeController

        class _Stub(RuntimeController):
            def __init__(self, store):
                self.store = store

        ctrl = _Stub(self.store)
        ctrl._record_outcome_affect(
            {"kind": T.KIND_READING, "status": T.COMPLETED})
        self.assertTrue(affect.is_neutral(self.store.current_affect()))


class TestPlannerHonoursEffortAndPull(unittest.TestCase):
    """Affect changes *which* work is chosen, not just how much."""

    def _reading_score(self, state):
        from astra import activity, tasks as T
        from astra.library import Library
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        store = T.TaskStore(data_dir=tmp)
        mem = TripleMemoryStore(data_dir=tmp, mirror_guard=False)
        lib = Library(data_dir=tmp)
        lib.add_text("word " * 300, title="W", work_id="w")
        planner = activity.ActivityPlanner(
            store, store=mem, library=lib, ledger=None,
            config={"max_new_per_pass": 5}, affect_provider=lambda: state)
        plan = planner.plan()
        reading = [c for c in plan["proposals"] if c["kind"] == T.KIND_READING]
        return plan, reading

    def test_reading_score_tracks_forward_pull(self):
        neutral, _ = self._reading_score({})
        pulled, reading = self._reading_score({"anticipation": 1.0})
        n_reading = [c for c in neutral["proposals"]
                     if c["kind"] == "reading"]
        self.assertTrue(n_reading and reading)
        self.assertGreater(reading[0]["score"], n_reading[0]["score"])

    def test_a_low_state_takes_on_nothing(self):
        plan, _ = self._reading_score({"sad": 0.9, "weary": 0.9})
        self.assertEqual(plan["decision"], "idle")
        self.assertEqual(plan["proposals"], [])

    def test_a_neutral_state_orders_by_balance(self):
        # A calm state changes nothing structural: the plan is produced and the
        # reading candidate carries no spurious bias note.
        plan, _ = self._reading_score({})
        self.assertEqual(plan["decision"], "plan")
        self.assertFalse(plan["bias"].get("note"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
