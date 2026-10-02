"""Tests for the background task model, store, dedupe and ledger."""

import os
import shutil
import tempfile
import unittest

from astra import tasks as T


class TestTaskModel(unittest.TestCase):
    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            T.BackgroundTask(kind="not_a_kind")

    def test_confirmation_task_starts_awaiting_human(self):
        task = T.BackgroundTask(kind=T.KIND_FILE_INSPECTION,
                                requires_confirmation=True)
        self.assertEqual(task.status, T.AWAITING_HUMAN)

    def test_default_priority_from_kind(self):
        task = T.BackgroundTask(kind=T.KIND_MEMORY_MAINTENANCE)
        self.assertEqual(task.priority, T.DEFAULT_PRIORITY[T.KIND_MEMORY_MAINTENANCE])


class TestTaskStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = T.TaskStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dedupe_refuses_duplicate_live_task(self):
        first = self.store.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="w1"))
        second = self.store.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="w1"))
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.store.all()), 1)

    def test_completed_task_allows_a_new_one(self):
        first = self.store.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="w1"))
        self.store.mark(first.id, T.COMPLETED)
        second = self.store.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="w1"))
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(len(self.store.all()), 2)

    def test_eligible_orders_by_priority_then_age(self):
        low = self.store.enqueue(T.BackgroundTask(kind=T.KIND_FILE_INSPECTION, dedupe_key="a", priority=10))
        high = self.store.enqueue(T.BackgroundTask(kind=T.KIND_MEMORY_MAINTENANCE, dedupe_key="b", priority=90))
        order = [t.id for t in self.store.eligible()]
        self.assertEqual(order[0], high.id)
        self.assertIn(low.id, order)

    def test_failed_tasks_are_retryable_until_max_attempts(self):
        task = self.store.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="x", max_attempts=2))
        self.store.mark(task.id, T.FAILED, attempts=1)
        self.assertIn(task.id, [t.id for t in self.store.eligible()])
        self.store.mark(task.id, T.FAILED, attempts=2)
        self.assertNotIn(task.id, [t.id for t in self.store.eligible()])

    def test_persistence_across_reload(self):
        self.store.enqueue(T.BackgroundTask(kind=T.KIND_CALCULATION, dedupe_key="c"))
        reloaded = T.TaskStore(data_dir=self.tmp)
        self.assertEqual(len(reloaded.all()), 1)
        self.assertEqual(reloaded.all()[0].kind, T.KIND_CALCULATION)

    def test_status_counts(self):
        self.store.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="r"))
        counts = self.store.counts()
        self.assertEqual(counts[T.PENDING], 1)


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ledger = T.TaskLedger(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_records_and_persists(self):
        self.ledger.record(task_id="t1", kind=T.KIND_READING,
                           status=T.COMPLETED, detail="read 1 chunk",
                           result={"count": 1, "summary": "ok", "extra": "dropped"})
        reloaded = T.TaskLedger(data_dir=self.tmp)
        entries = reloaded.entries()
        self.assertEqual(len(entries), 1)
        # The ledger stays small: only whitelisted result keys are kept.
        self.assertIn("count", entries[0]["result"])
        self.assertNotIn("extra", entries[0]["result"])

    def test_render_is_concise(self):
        self.ledger.record(task_id="t1", kind=T.KIND_READING, status=T.COMPLETED)
        self.assertIn("reading", self.ledger.render())


if __name__ == "__main__":
    unittest.main(verbosity=2)
