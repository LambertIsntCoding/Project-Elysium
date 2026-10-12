"""Integration tests for the autonomy driver and proactive messaging lifecycle.

These wire a real TaskStore, real ledger, real schedule driver, real
notification policy and a real ActivityPlanner, and drive them with an injected
clock. The Discord transport is *not* mocked at the policy level: a tiny
in-process fake stands in for the HTTP send, so the whole path
(tool -> ledger -> decision -> notification policy -> delivery -> dedupe/record)
is exercised for real.
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from astra import (activity, autonomy, notifications as N, schedule_runtime,
                   schedules as sched, tasks as T)
from astra.memory import TripleMemoryStore


def _utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


class FakeNotifier:
    """A delivery sink that records calls, can fail, and honours dedupe id."""

    def __init__(self):
        self.sent = []
        self.fail = False
        self.delivered_ids = set()

    def __call__(self, text, ref_id, outcome):
        if self.fail:
            return {"ok": False, "error": "http 500"}
        if ref_id and ref_id in self.delivered_ids:
            return {"ok": True, "sent": False, "suppressed": "duplicate"}
        if ref_id:
            self.delivered_ids.add(ref_id)
        self.sent.append({"text": text, "ref_id": ref_id, "outcome": outcome})
        return {"ok": True, "sent": True, "message_id": "m1"}


class FakeScheduler:
    """A scheduler double that runs whatever is due and records outcomes."""

    def __init__(self, store, ledger, now_fn=None):
        self.store = store
        self.ledger = ledger
        self.ran = []
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def run_cycle(self):
        did = 0
        for task in self.store.eligible(now=self._now_fn()):
            self.store.mark(task.id, T.COMPLETED)
            self.ledger.record(task_id=task.id, kind=task.kind,
                               status=T.COMPLETED, detail=f"ran {task.kind}",
                               result={"importance": 0.9})
            self.ran.append(task.kind)
            did += 1
        return {"tasks": did}


class TestAutonomy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = T.TaskStore(data_dir=self.tmp)
        self.ledger = T.TaskLedger(data_dir=self.tmp)
        self.memory = TripleMemoryStore(data_dir=self.tmp, mirror_guard=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _driver(self, *, notifier=None, config=None, notif_config=None,
                start="2026-06-01T11:00:00"):
        # A mutable clock: materialise before the due instant, then advance.
        holder = {"now": _utc(start)}

        def clock():
            return holder["now"]

        self.holder = holder
        self.clock = clock
        driver = schedule_runtime.ScheduleDriver(self.store, ledger=self.ledger,
                                                 now_fn=clock)
        planner = activity.ActivityPlanner(
            self.store, store=self.memory, ledger=self.ledger,
            config={"planning_enabled": True, "max_new_per_pass": 1},
            enabled_kinds=[T.KIND_MEMORY_MAINTENANCE, T.KIND_SELF_REVIEW])
        policy = N.NotificationPolicy(data_dir=self.tmp, config=notif_config or {
            "timezone": "UTC", "interval_seconds": 0, "max_per_interval": 5,
            "min_gap_seconds": 0, "max_per_day": 10, "min_importance": 0.5,
            "allow_high_priority_bypass": False, "digest_enabled": True})
        sched_double = FakeScheduler(self.store, self.ledger, now_fn=clock)
        auto = autonomy.AutonomyDriver(
            schedule_driver=driver, planner=planner, notifications=policy,
            scheduled_scheduler=sched_double, ledger=self.ledger,
            notifier=notifier, config=config or {}, now_fn=clock)
        return auto, driver, sched_double

    def _at(self, text):
        self.holder["now"] = _utc(text)

    def _due_daily(self, driver):
        driver.materialize([{"mode": "daily", "at": "12:00", "timezone": "UTC",
                             "kind": T.KIND_SELF_REVIEW, "dedupe_key": "s"}])

    def test_wake_runs_due_scheduled_work_without_a_conversation(self):
        notifier = FakeNotifier()
        auto, driver, sched_double = self._driver(notifier=notifier)
        self._due_daily(driver)
        self._at("2026-06-01T12:00:00")
        report = auto.wake(now=self.clock())
        self.assertIn(T.KIND_SELF_REVIEW, sched_double.ran)
        self.assertEqual(report["scheduled_cycle"]["tasks"], 1)

    def test_outcome_triggers_proactive_message(self):
        notifier = FakeNotifier()
        auto, driver, _ = self._driver(notifier=notifier)
        self._due_daily(driver)
        self._at("2026-06-01T12:00:00")
        auto.wake(now=self.clock())
        self.assertEqual(len(notifier.sent), 1)
        self.assertEqual(auto.last_report["notifications"]["delivered"], 1)

    def test_no_outcome_no_message(self):
        notifier = FakeNotifier()
        auto, _, _ = self._driver(notifier=notifier)
        report = auto.wake(now=self.clock())
        self.assertEqual(len(notifier.sent), 0)

    def test_user_active_queues_instead_of_interrupting(self):
        notifier = FakeNotifier()
        auto, driver, _ = self._driver(notifier=notifier)
        self._due_daily(driver)
        self._at("2026-06-01T12:00:00")
        report = auto.wake(user_active=True, now=self.clock())
        self.assertEqual(len(notifier.sent), 0)
        self.assertGreaterEqual(report["notifications"]["queued"], 1)

    def test_delivery_failure_is_not_counted_as_received_and_is_queued(self):
        notifier = FakeNotifier()
        notifier.fail = True
        auto, driver, _ = self._driver(notifier=notifier)
        self._due_daily(driver)
        self._at("2026-06-01T12:00:00")
        report = auto.wake(now=self.clock())
        self.assertEqual(report["notifications"]["delivered"], 0)
        self.assertGreaterEqual(report["notifications"]["queued"], 1)
        # It survives for a later digest rather than being reported as received.
        self.assertGreater(auto.notifications.pending_digest_count(), 0)

    def test_no_duplicate_message_on_second_wake(self):
        notifier = FakeNotifier()
        # Planning off so the only outcome is the scheduled one; the second wake
        # must not re-send the *same* outcome.
        auto, driver, _ = self._driver(notifier=notifier,
                                       config={"planning_enabled": False})
        self._due_daily(driver)
        self._at("2026-06-01T12:00:00")
        auto.wake(now=self.clock())
        first = len(notifier.sent)
        self.assertEqual(first, 1)
        auto.wake(now=self.clock() + timedelta(minutes=1))
        self.assertEqual(len(notifier.sent), first)

    def test_distinct_outcomes_are_distinct_messages(self):
        # A second genuinely-new outcome is a second message (never suppressed as
        # a duplicate of an unrelated one).
        notifier = FakeNotifier()
        auto, driver, sched_double = self._driver(
            notifier=notifier, config={"planning_enabled": False})
        self._due_daily(driver)
        self._at("2026-06-01T12:00:00")
        auto.wake(now=self.clock())
        # Force a second, distinct recorded outcome.
        self.ledger.record(task_id="t2", kind=T.KIND_SELF_REVIEW,
                           status=T.COMPLETED, detail="second thing",
                           result={"importance": 0.9})
        self._at("2026-06-01T13:00:00")
        auto.wake(now=self.clock())
        self.assertEqual(len(notifier.sent), 2)

    def test_pause_activity_keeps_notifications_independent(self):
        notifier = FakeNotifier()
        auto, driver, _ = self._driver(notifier=notifier)
        auto.set_paused(True)
        self._due_daily(driver)
        self._at("2026-06-01T12:00:00")
        report = auto.wake(now=self.clock())
        # Paused: the scheduled cycle does not run...
        self.assertEqual(report["scheduled_cycle"]["reason"], "paused")
        # ...but the notification policy is *not* paused (no cross-talk).
        self.assertFalse(auto.notifications.paused)

    def test_disabled_driver_does_nothing(self):
        notifier = FakeNotifier()
        auto, driver, _ = self._driver(notifier=notifier, config={"enabled": False})
        self._due_daily(driver)
        self._at("2026-06-01T12:00:00")
        report = auto.wake(now=self.clock())
        self.assertEqual(report.get("skipped"), "disabled")
        self.assertEqual(len(notifier.sent), 0)

    def test_planner_enqueues_bounded_activity(self):
        notifier = FakeNotifier()
        auto, driver, _ = self._driver(notifier=notifier)
        auto.wake(now=self.clock())
        live = [t for t in self.store.all() if t.origin == T.ORIGIN_ACTIVITY]
        self.assertLessEqual(len(live), 1)

    def test_missing_notifier_does_not_crash_or_claim_success(self):
        auto, driver, _ = self._driver(notifier=None)
        self._due_daily(driver)
        self._at("2026-06-01T12:00:00")
        report = auto.wake(now=self.clock())
        # With no transport the outcome is held, never reported as delivered.
        self.assertEqual(report["notifications"]["delivered"], 0)


class TestRestartRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missed_schedule_is_honest_across_restart(self):
        # First run: materialise at 08:00 for a 09:00 daily.
        store = T.TaskStore(data_dir=self.tmp)
        ledger = T.TaskLedger(data_dir=self.tmp)
        d1 = schedule_runtime.ScheduleDriver(
            store, ledger=ledger, now_fn=lambda: _utc("2026-06-01T08:00:00"))
        d1.materialize([{"mode": "daily", "at": "09:00", "timezone": "UTC",
                         "kind": T.KIND_SELF_REVIEW, "dedupe_key": "s",
                         "start": "08:00", "end": "10:00", "catch_up": "skip"}])
        # Simulate downtime, then a fresh driver at 11:00 (window missed).
        store2 = T.TaskStore(data_dir=self.tmp)
        ledger2 = T.TaskLedger(data_dir=self.tmp)
        d2 = schedule_runtime.ScheduleDriver(
            store2, ledger=ledger2, now_fn=lambda: _utc("2026-06-01T11:00:00"))
        report = d2.refresh()
        self.assertTrue(report["missed"])
        # The missed occurrence is recorded, never pretended as a run.
        self.assertIn(T.MISSED, [e["status"] for e in ledger2.entries()])

    def test_schedule_survives_restart_without_duplication(self):
        store = T.TaskStore(data_dir=self.tmp)
        d1 = schedule_runtime.ScheduleDriver(
            store, now_fn=lambda: _utc("2026-06-01T08:00:00"))
        ids = d1.materialize([{"mode": "daily", "at": "09:00", "timezone": "UTC",
                               "kind": T.KIND_SELF_REVIEW, "dedupe_key": "s"}])
        store2 = T.TaskStore(data_dir=self.tmp)
        d2 = schedule_runtime.ScheduleDriver(
            store2, now_fn=lambda: _utc("2026-06-01T08:05:00"))
        ids2 = d2.materialize([{"mode": "daily", "at": "09:00", "timezone": "UTC",
                                "kind": T.KIND_SELF_REVIEW, "dedupe_key": "s"}])
        self.assertEqual(ids, ids2)
        self.assertEqual(len(store2.all()), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
