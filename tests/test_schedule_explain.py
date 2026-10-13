"""Schedule explainability: the admin's why/when/next view is derived, honest.

The brief (sections 2, 3, 8) asks that Astra's status reflect actual task
records rather than optimistic assumptions. ``astra.schedule_explain`` is the
single read-only bridge from a task to that explanation. These tests pin the
honesty properties: an overdue occurrence is stated per its catch-up policy,
a blocked/failed task says so, and the explanation is derived - never stored,
never guessed.
"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from astra import schedule_explain as se
from astra import schedules, tasks as task_mod
from astra.tasks import BackgroundTask, TaskStore

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _store():
    return TaskStore(data_dir=tempfile.mkdtemp())


def _scheduled(store, spec, *, due, status=None, attempts=0, max_attempts=3,
               origin=task_mod.ORIGIN_SCHEDULE, last_error=None, result=None,
               blocked_reason=None, dedupe_key=None):
    task = BackgroundTask(
        kind="memory_maintenance", origin=origin, schedule=spec,
        dedupe_key=dedupe_key or f"k_{due}",
        due_at=due, attempts=attempts, max_attempts=max_attempts,
        last_error=last_error, result=result, blocked_reason=blocked_reason)
    if status is not None:
        task.status = status
    store.enqueue(task)
    return task


class TestExplainDerivation(unittest.TestCase):
    def test_future_occurrence_says_when_it_will_run(self):
        store = _store()
        spec = {"mode": "daily", "at": "03:00", "timezone": "UTC"}
        due = (NOW + timedelta(hours=5)).isoformat()
        _scheduled(store, spec, due=due)
        row = se.explain_all(store, now=NOW)[0]
        self.assertIn("will be attempted", row["next_step"])
        self.assertIsNone(row["missed_note"])

    def test_overdue_skip_policy_does_not_promise_a_run(self):
        store = _store()
        spec = {"mode": "daily", "at": "03:00", "timezone": "UTC",
                "catch_up": "skip", "deadline": "04:00"}
        due = (NOW - timedelta(hours=20)).isoformat()
        _scheduled(store, spec, due=due)
        row = se.explain_all(store, now=NOW)[0]
        self.assertEqual(schedules.resolve_missed(spec, datetime.fromisoformat(due),
                                                  now=NOW), schedules.CATCH_UP_SKIP)
        self.assertIn("will not run", row["next_step"])
        self.assertIn("skip", row["missed_note"])

    def test_overdue_run_policy_says_it_will_still_be_attempted(self):
        store = _store()
        spec = {"mode": "interval", "every_hours": 6, "catch_up": "run"}
        due = (NOW - timedelta(hours=3)).isoformat()
        _scheduled(store, spec, due=due)
        row = se.explain_all(store, now=NOW)[0]
        self.assertIn("attempt", row["next_step"].lower())

    def test_failed_task_reports_retry_eligibility_from_attempts(self):
        store = _store()
        spec = {"mode": "daily", "at": "03:00"}
        _scheduled(store, spec, due=(NOW - timedelta(hours=1)).isoformat(),
                   status=task_mod.FAILED, attempts=1, max_attempts=3,
                   last_error="model unreachable")
        row = se.explain_all(store, now=NOW)[0]
        self.assertIn("retry", row["next_step"])
        self.assertEqual(row["last_error"], "model unreachable")

    def test_failed_task_with_exhausted_retries_says_so(self):
        store = _store()
        spec = {"mode": "daily", "at": "03:00"}
        _scheduled(store, spec, due=(NOW - timedelta(hours=1)).isoformat(),
                   status=task_mod.FAILED, attempts=3, max_attempts=3)
        row = se.explain_all(store, now=NOW)[0]
        self.assertIn("exhausted", row["next_step"])

    def test_blocked_task_surfaces_the_reason(self):
        store = _store()
        spec = {"mode": "daily", "at": "03:00"}
        _scheduled(store, spec, due=(NOW - timedelta(minutes=5)).isoformat(),
                   status=task_mod.BLOCKED, blocked_reason="permission denied")
        row = se.explain_all(store, now=NOW)[0]
        self.assertIn("permission denied", row["next_step"])

    def test_awaiting_human_is_not_presented_as_running(self):
        store = _store()
        spec = {"mode": "once", "at": "03:00"}
        _scheduled(store, spec, due=(NOW - timedelta(minutes=5)).isoformat(),
                   status=task_mod.AWAITING_HUMAN)
        row = se.explain_all(store, now=NOW)[0]
        self.assertIn("approval", row["next_step"])

    def test_origin_is_explained_not_raw(self):
        store = _store()
        spec = {"mode": "daily", "at": "03:00"}
        _scheduled(store, spec, due=(NOW + timedelta(hours=1)).isoformat(),
                   origin=task_mod.ORIGIN_ACTIVITY)
        row = se.explain_all(store, now=NOW)[0]
        self.assertIn("planner", row["reason"])


class TestExplainOrderingAndRender(unittest.TestCase):
    def test_soonest_due_leads_and_higher_priority_breaks_ties(self):
        store = _store()
        late = _scheduled(store, {"mode": "daily", "at": "03:00"},
                          due=(NOW + timedelta(hours=9)).isoformat(),
                          dedupe_key="late")
        late.priority = 99
        soon_hi = _scheduled(store, {"mode": "daily", "at": "03:00"},
                             due=(NOW + timedelta(hours=1)).isoformat(),
                             dedupe_key="soon_hi")
        soon_hi.priority = 80
        soon_lo = _scheduled(store, {"mode": "daily", "at": "03:00"},
                             due=(NOW + timedelta(hours=1)).isoformat(),
                             dedupe_key="soon_lo")
        soon_lo.priority = 10
        rows = se.explain_all(store, now=NOW)
        self.assertEqual([r["id"] for r in rows][:3],
                         [soon_hi.id, soon_lo.id, late.id])

    def test_render_shows_why_when_and_next(self):
        store = _store()
        _scheduled(store, {"mode": "daily", "at": "03:00", "timezone": "UTC",
                           "catch_up": "run"},
                   due=(NOW + timedelta(hours=2)).isoformat())
        text = se.render(se.explain_all(store, now=NOW))
        self.assertIn("why:", text)
        self.assertIn("when:", text)
        self.assertIn("next:", text)

    def test_empty_store_renders_honestly(self):
        store = _store()
        rows = se.explain_all(store, now=NOW)
        self.assertEqual(rows, [])
        self.assertIn("no scheduled", se.render(rows).lower())

    def test_dependency_state_only_reported_when_lookup_is_provided(self):
        store = _store()
        spec = {"mode": "daily", "at": "03:00", "depends_on": ["task_missing"]}
        task = _scheduled(store, spec, due=(NOW + timedelta(hours=1)).isoformat())

        def lookup(ident):
            getter = getattr(store, "get", None)
            return getter(ident) if callable(getter) else None

        row = se.explain(task, now=NOW, dependency_lookup=lookup)
        self.assertFalse(row["dependencies_met"])


class TestExplainIsReadOnly(unittest.TestCase):
    def test_explaining_does_not_change_the_store(self):
        store = _store()
        _scheduled(store, {"mode": "daily", "at": "03:00"},
                   due=(NOW + timedelta(hours=1)).isoformat())
        before = store.all()
        se.explain_all(store, now=NOW)
        self.assertEqual([t.to_dict() for t in store.all()],
                         [t.to_dict() for t in before])


if __name__ == "__main__":
    unittest.main()
