"""The schedule driver: turning configured schedules into durable task state.

:mod:`astra.schedules` is pure time arithmetic; :mod:`astra.tasks` is the
durable queue. This module is the small piece between them - it *materialises* a
configured schedule as a task, keeps that task's due instant current, marks
expired work honestly, and re-arms a recurring occurrence after it runs.

It is deliberately separate from the scheduler (:mod:`astra.scheduler`), which
decides *what may run* under resource and sleep policy. The driver only answers
*when* a task is next due, and records nothing conversational.

Two guarantees:

* **No duplicate execution.** One schedule maps to at most one live task,
  found by ``(kind, dedupe_key)``, so re-running the driver across restarts
  never doubles the queue.
* **No replay of missed history.** A recurring schedule whose window was missed
  advances to its next future occurrence rather than firing once per missed
  period; a missed occurrence is recorded as ``missed`` (not run, not pretended).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from . import schedules as sched
from . import tasks as task_mod

# Out-comes the driver reports per scheduled task during a refresh.
REFRESH_DUE = "due"
REFRESH_WAITING = "waiting"
REFRESH_MISSED = "missed"
REFRESH_EXPIRED = "expired"
REFRESH_REARMED = "rearmed"


class ScheduleDriver:
    """Keeps scheduled tasks' due instants current, honestly and idempotently."""

    def __init__(self, task_store: task_mod.TaskStore, *, ledger: Any = None,
                 now_fn: Optional[Callable[[], datetime]] = None) -> None:
        self.task_store = task_store
        self.ledger = ledger
        self._now = now_fn or (lambda: datetime.now(timezone.utc))

    # -- materialise configured schedules --------------------------------
    def materialize(self, configured: Any) -> List[str]:
        """Create/refresh one durable task per configured schedule.

        ``configured`` is a list of schedule dicts (see :mod:`astra.schedules`),
        each naming a ``kind`` and a ``dedupe_key``. Returns the task ids touched.
        An invalid schedule is skipped - never guessed into existence.
        """
        touched: List[str] = []
        now = self._now()
        for spec in configured or []:
            norm = sched.normalize(spec)
            if norm is None:
                continue
            kind = str(norm.get("kind") or task_mod.KIND_APP_MAINTENANCE)
            if kind not in task_mod.TASK_KINDS:
                continue
            key = str(norm.get("dedupe_key") or f"sched-{kind}")
            existing = self.task_store.find_live_by_key(kind, key)
            if existing is not None:
                # Refresh the schedule/due on the live task without duplicating.
                due = sched.next_due(norm, last_run=_parse(existing.last_run_at), now=now)
                self.task_store.update(
                    existing.id, schedule=norm,
                    due_at=_iso(due), next_due_at=_iso(due))
                touched.append(existing.id)
                continue
            due = sched.next_due(norm, now=now)
            task = task_mod.BackgroundTask(
                kind=kind, origin=task_mod.ORIGIN_SCHEDULE,
                priority=int(norm.get("priority") or task_mod.DEFAULT_PRIORITY.get(kind, 25)),
                dedupe_key=key, schedule=norm, payload=dict(norm.get("payload") or {}),
                due_at=_iso(due), next_due_at=_iso(due),
            )
            stored = self.task_store.enqueue(task)
            if stored is not None:
                touched.append(stored.id)
        return touched

    # -- keep due instants current ---------------------------------------
    def refresh(self) -> Dict[str, Any]:
        """Advance scheduled tasks: publish due ones, miss/expire honestly.

        Called on a bounded cadence and once at startup, so a schedule that came
        due while the runtime was down is handled on return rather than lost.
        """
        now = self._now()
        report: Dict[str, Any] = {"due": [], "waiting": 0, "missed": [],
                                  "expired": [], "rearmed": []}

        # Expiry first: past `expires` never runs again, on any policy.
        for task in self.task_store.expired(now=now):
            self._terminal(task, task_mod.EXPIRED, detail="expired",
                           now=now, report=report, bucket="expired")

        for task in self.task_store.scheduled():
            spec = task.schedule or {}
            due = _parse(task.due_at or task.next_due_at)
            if due is None:
                # A schedule with no due instant yet: compute one.
                due = sched.next_due(spec, last_run=_parse(task.last_run_at), now=now)
                self.task_store.update(task.id, due_at=_iso(due), next_due_at=_iso(due))
                if due is None:
                    continue
            if task.status not in (task_mod.SCHEDULED, task_mod.PENDING,
                                   task_mod.FAILED, task_mod.DEFERRED):
                continue
            if due <= now:
                # It is (at least) due. The catch-up policy decides honestly.
                policy = sched.resolve_missed(spec, due, now=now)
                if policy == sched.CATCH_UP_SKIP and due < now:
                    self._terminal(task, task_mod.MISSED,
                                   detail=f"missed window @ {task.due_at}",
                                   now=now, report=report, bucket="missed")
                    self._rearm(task, now, report)
                    continue
                # run (or coalesce): publish as runnable now.
                self.task_store.update(task.id, status=task_mod.PENDING)
                report["due"].append(task.id)
            else:
                report["waiting"] += 1
        return report

    # -- re-arm after an occurrence runs ---------------------------------
    def rearm_after_run(self, task_id: str) -> Optional[str]:
        """After a scheduled occurrence finishes, arm the next one.

        Returns the next due instant (ISO) or ``None`` when the schedule is
        exhausted. The completed occurrence stays in the ledger; the task record
        is reused so no duplicate accrues.
        """
        task = self.task_store.get(task_id)
        if task is None or not task.schedule:
            return None
        now = self._now()
        report: Dict[str, Any] = {"rearmed": []}
        self._rearm(task, now, report)
        if not report["rearmed"]:
            return None
        updated = self.task_store.get(task_id)
        return (updated.due_at or updated.next_due_at) if updated else None

    def _rearm(self, task: task_mod.BackgroundTask, now: datetime,
               report: Dict[str, Any]) -> None:
        spec = task.schedule or {}
        nxt = sched.next_due(spec, last_run=_parse(task.last_run_at) or now, now=now)
        if nxt is None:
            # Exhausted (once, or expired): leave it terminal.
            return
        # Reset per-occurrence state so a recurring schedule runs again cleanly.
        self.task_store.update(
            task.id, status=task_mod.SCHEDULED, attempts=0, last_error=None,
            due_at=_iso(nxt), next_due_at=_iso(nxt))
        report.setdefault("rearmed", []).append(task.id)

    # -- helpers ---------------------------------------------------------
    def _terminal(self, task: task_mod.BackgroundTask, status: str, *,
                  detail: str, now: datetime, report: Dict[str, Any],
                  bucket: str) -> None:
        self.task_store.mark(task.id, status, last_error=detail)
        if self.ledger is not None:
            self.ledger.record(task_id=task.id, kind=task.kind, status=status,
                               detail=detail)
        report.setdefault(bucket, []).append(task.id)


def _parse(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(timezone.utc).isoformat() if value else None


__all__ = ["ScheduleDriver", "REFRESH_DUE", "REFRESH_WAITING",
           "REFRESH_MISSED", "REFRESH_EXPIRED", "REFRESH_REARMED"]
