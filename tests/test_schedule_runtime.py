"""Tests for the schedule driver: materialisation, refresh, re-arming, honesty.

These use a real TaskStore and a real ledger, with an injected clock so
"when is it due" is deterministic and no sleeping is involved.
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timezone

from astra import schedule_runtime, schedules as sched, tasks as T


def _utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def _daily(key="s", kind=T.KIND_SELF_REVIEW, **extra):
    spec = {"mode": "daily", "at": "09:00", "timezone": "UTC",
            "kind": kind, "dedupe_key": key}
    spec.update(extra)
    return spec


class TestMaterialize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = T.TaskStore(data_dir=self.tmp)
        self.ledger = T.TaskLedger(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _driver(self, now_text="2026-06-01T08:00:00"):
        return schedule_runtime.ScheduleDriver(
            self.store, ledger=self.ledger,
            now_fn=lambda: _utc(now_text))

    def test_materialize_creates_a_scheduled_task(self):
        ids = self._driver().materialize([_daily()])
        self.assertEqual(len(ids), 1)
        task = self.store.get(ids[0])
        self.assertEqual(task.status, T.SCHEDULED)
        self.assertEqual(task.origin, T.ORIGIN_SCHEDULE)
        self.assertEqual(task.due_at, _utc("2026-06-01T09:00:00").isoformat())

    def test_materialize_is_idempotent(self):
        driver = self._driver()
        first = driver.materialize([_daily()])
        second = driver.materialize([_daily()])
        # Re-running across restarts does not double the queue.
        self.assertEqual(first, second)
        self.assertEqual(len(self.store.all()), 1)

    def test_invalid_schedule_is_skipped_not_guessed(self):
        ids = self._driver().materialize([{"mode": "daily"}])  # missing time
        self.assertEqual(ids, [])

    def test_unknown_kind_is_skipped(self):
        ids = self._driver().materialize([_daily(kind="not_a_kind")])
        self.assertEqual(ids, [])


class TestRefresh(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = T.TaskStore(data_dir=self.tmp)
        self.ledger = T.TaskLedger(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_due_task_becomes_runnable(self):
        now = {"t": _utc("2026-06-01T08:30:00")}
        driver = schedule_runtime.ScheduleDriver(
            self.store, ledger=self.ledger, now_fn=lambda: now["t"])
        driver.materialize([_daily()])
        now["t"] = _utc("2026-06-01T09:00:00")
        report = driver.refresh()
        self.assertEqual(len(report["due"]), 1)
        task = self.store.get(report["due"][0])
        self.assertEqual(task.status, T.PENDING)

    def test_not_yet_due_waits(self):
        now = {"t": _utc("2026-06-01T08:00:00")}
        driver = schedule_runtime.ScheduleDriver(
            self.store, ledger=self.ledger, now_fn=lambda: now["t"])
        driver.materialize([_daily()])
        report = driver.refresh()
        self.assertEqual(report["due"], [])
        self.assertEqual(report["waiting"], 1)

    def test_missed_window_is_recorded_not_run(self):
        now = {"t": _utc("2026-06-01T08:00:00")}
        driver = schedule_runtime.ScheduleDriver(
            self.store, ledger=self.ledger, now_fn=lambda: now["t"])
        # A short window: 08:00-10:00; the runtime comes back at 11:00.
        driver.materialize([_daily(start="08:00", end="10:00", catch_up="skip")])
        now["t"] = _utc("2026-06-01T11:00:00")
        report = driver.refresh()
        self.assertTrue(report["missed"])
        task = self.store.get(report["missed"][0])
        # It is now the *next* occurrence (re-armed) or missed; the record is
        # honest and it was not run.
        self.assertIn(task.status, (T.MISSED, T.SCHEDULED))
        statuses = [e["status"] for e in self.ledger.entries()]
        self.assertIn(T.MISSED, statuses)

    def test_expired_never_runs(self):
        now = {"t": _utc("2026-06-01T08:00:00")}
        driver = schedule_runtime.ScheduleDriver(
            self.store, ledger=self.ledger, now_fn=lambda: now["t"])
        driver.materialize([_daily(expires="2026-06-01T08:30:00")])
        now["t"] = _utc("2026-06-01T09:00:00")
        report = driver.refresh()
        self.assertTrue(report["expired"])
        task = self.store.get(report["expired"][0])
        self.assertEqual(task.status, T.EXPIRED)

    def test_catch_up_run_publishes_a_late_run(self):
        now = {"t": _utc("2026-06-01T08:00:00")}
        driver = schedule_runtime.ScheduleDriver(
            self.store, ledger=self.ledger, now_fn=lambda: now["t"])
        driver.materialize([_daily(catch_up="run")])
        now["t"] = _utc("2026-06-01T12:00:00")  # late, same day
        report = driver.refresh()
        self.assertEqual(len(report["due"]), 1)


class TestRearm(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = T.TaskStore(data_dir=self.tmp)
        self.ledger = T.TaskLedger(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_recurring_rearms_to_next_occurrence(self):
        now = {"t": _utc("2026-06-01T09:00:00")}
        driver = schedule_runtime.ScheduleDriver(
            self.store, ledger=self.ledger, now_fn=lambda: now["t"])
        ids = driver.materialize([_daily()])
        task = self.store.get(ids[0])
        self.store.update(task.id, status=T.PENDING)
        next_due = driver.rearm_after_run(task.id)
        self.assertEqual(next_due, _utc("2026-06-02T09:00:00").isoformat())
        task = self.store.get(ids[0])
        self.assertEqual(task.status, T.SCHEDULED)
        self.assertEqual(task.attempts, 0)

    def test_once_schedule_is_exhausted_after_run(self):
        now = {"t": _utc("2026-06-01T09:00:00")}
        driver = schedule_runtime.ScheduleDriver(
            self.store, ledger=self.ledger, now_fn=lambda: now["t"])
        spec = {"mode": "once", "at_time": "2026-06-01T09:00:00", "timezone": "UTC",
                "kind": T.KIND_SELF_REVIEW, "dedupe_key": "one"}
        ids = driver.materialize([spec])
        task = self.store.get(ids[0])
        self.store.update(task.id, status=T.PENDING)
        self.assertIsNone(driver.rearm_after_run(task.id))


if __name__ == "__main__":
    unittest.main(verbosity=2)
