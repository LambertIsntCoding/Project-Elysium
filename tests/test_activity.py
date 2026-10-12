"""Tests for the bounded activity planner.

The planner proposes work from Astra's *real* state; it must never invent a
capability, must respect its budgets, and must balance across lanes. These tests
use a real TaskStore and a real memory store.
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timezone

from astra import activity, tasks as T
from astra.memory import TripleMemoryStore


def _utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


class TestCandidates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = T.TaskStore(data_dir=self.tmp)
        self.memory = TripleMemoryStore(data_dir=self.tmp, mirror_guard=False)
        self.ledger = T.TaskLedger(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _planner(self, **cfg):
        return activity.ActivityPlanner(
            self.store, store=self.memory, ledger=self.ledger, config=cfg,
            enabled_kinds=[T.KIND_MEMORY_REVIEW, T.KIND_SELF_REVIEW,
                           T.KIND_QUESTION_REVIEW, T.KIND_EXPERIENCE_REVIEW,
                           T.KIND_WORK_REVIEW, T.KIND_MEMORY_MAINTENANCE])

    def test_maintenance_candidates_always_available(self):
        cands = self._planner().candidates()
        kinds = {c["kind"] for c in cands}
        self.assertIn(T.KIND_MEMORY_MAINTENANCE, kinds)

    def test_open_question_adds_review_candidate(self):
        self.memory.add_question("Why does the watcher restart twice?")
        kinds = {c["kind"] for c in self._planner().candidates()}
        self.assertIn(T.KIND_QUESTION_REVIEW, kinds)

    def test_every_candidate_names_a_real_kind_and_reason(self):
        for cand in self._planner().candidates():
            self.assertIn(cand["kind"], T.TASK_KINDS)
            self.assertTrue(cand["reason"])
            self.assertIn(cand["lane"], (activity.LANE_MAINTENANCE,
                                         activity.LANE_PROJECT,
                                         activity.LANE_LEARNING,
                                         activity.LANE_EVALUATION,
                                         activity.LANE_EXPLORATION))


class TestPlanBounds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = T.TaskStore(data_dir=self.tmp)
        self.ledger = T.TaskLedger(data_dir=self.tmp)
        self.memory = TripleMemoryStore(data_dir=self.tmp, mirror_guard=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _planner(self, **cfg):
        return activity.ActivityPlanner(
            self.store, store=self.memory, ledger=self.ledger, config=cfg,
            enabled_kinds=[T.KIND_MEMORY_REVIEW, T.KIND_SELF_REVIEW,
                           T.KIND_MEMORY_MAINTENANCE])

    def test_max_new_bounds_proposals(self):
        result = self._planner(max_new_per_pass=1).plan(now=_utc("2026-06-01T09:00:00"))
        self.assertEqual(result["decision"], "plan")
        self.assertLessEqual(len(result["proposals"]), 1)

    def test_queue_capacity_is_bounded(self):
        for i in range(30):
            self.store.enqueue(T.BackgroundTask(kind=T.KIND_SELF_REVIEW,
                                                dedupe_key=f"self_{i}"))
        result = self._planner(max_live_tasks=5).plan(now=_utc("2026-06-01T09:00:00"))
        self.assertEqual(result["decision"], "bounded")
        self.assertEqual(result["proposals"], [])

    def test_enabled_kinds_are_respected(self):
        planner = activity.ActivityPlanner(
            self.store, store=self.memory, ledger=self.ledger,
            config={}, enabled_kinds=[T.KIND_MEMORY_MAINTENANCE])
        result = planner.plan(now=_utc("2026-06-01T09:00:00"))
        for cand in result["proposals"]:
            self.assertEqual(cand["kind"], T.KIND_MEMORY_MAINTENANCE)

    def test_per_kind_limit_skips_saturated_kind(self):
        self.store.enqueue(T.BackgroundTask(kind=T.KIND_MEMORY_MAINTENANCE,
                                            dedupe_key="a"))
        self.store.enqueue(T.BackgroundTask(kind=T.KIND_MEMORY_MAINTENANCE,
                                            dedupe_key="b"))
        result = self._planner(per_kind_limit=2, max_new_per_pass=5).plan(
            now=_utc("2026-06-01T09:00:00"))
        kinds = [c["kind"] for c in result["proposals"]]
        self.assertNotIn(T.KIND_MEMORY_MAINTENANCE, kinds)


class TestDecide(unittest.TestCase):
    def test_failed_with_attempts_left_retries(self):
        task = T.BackgroundTask(kind=T.KIND_SELF_REVIEW)
        task.status = T.FAILED
        task.attempts = 1
        self.assertEqual(activity.ActivityPlanner.decide(task), activity.RETRY)

    def test_exhausted_attempts_abandons(self):
        task = T.BackgroundTask(kind=T.KIND_SELF_REVIEW)
        task.status = T.FAILED
        task.attempts = task.max_attempts
        self.assertEqual(activity.ActivityPlanner.decide(task), activity.ABANDON)

    def test_confirmation_gated_defers(self):
        task = T.BackgroundTask(kind=T.KIND_SELF_REVIEW, requires_confirmation=True)
        self.assertEqual(activity.ActivityPlanner.decide(task), activity.DEFER)

    def test_completed_stops(self):
        task = T.BackgroundTask(kind=T.KIND_SELF_REVIEW)
        task.status = T.COMPLETED
        self.assertEqual(activity.ActivityPlanner.decide(task), activity.STOP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
