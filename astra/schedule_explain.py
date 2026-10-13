"""Explain a scheduled task from its own records - nothing assumed.

The admin views need to answer, without optimistic guessing: *why* is this
activity here, *when* is it meant to run, *what* happened last time, and *what
happens next*. Every field this module reads already exists on the task (kind,
origin, priority, schedule, due/next-due, attempts/max-attempts, status,
last_error, result) and on its schedule spec (window, deadline, catch-up,
dependencies). So the explanation is derived, not stored: it cannot drift from
the task record, and it is never a second source of truth.

It is pure - no clock beyond the ``now`` passed in, no store, no I/O - so it is
cheap enough to call for every task in a listing and deterministic under test.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import schedules, tasks as task_mod


def _parse(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, tz_name: Any = None) -> str:
    """A due instant in the schedule's own timezone, or 'unknown'."""
    parsed = _parse(value)
    if parsed is None:
        return "unknown"
    tz = schedules.resolve_tz(tz_name)
    local = parsed.astimezone(tz)
    return local.strftime("%Y-%m-%d %H:%M %Z")


def _reason(task: Any) -> str:
    """Why this activity is in play, in plain words."""
    origin = str(getattr(task, "origin", "") or "")
    if origin == task_mod.ORIGIN_SCHEDULE:
        return "scheduled occurrence (a recurring or dated plan)"
    if origin in (task_mod.ORIGIN_ACTIVITY, task_mod.ORIGIN_SCHEDULER):
        return "selected by the activity planner from approved work"
    if origin == task_mod.ORIGIN_HUMAN:
        return "requested explicitly"
    if origin == task_mod.ORIGIN_MODEL:
        return "proposed by the model and validated before running"
    if origin == task_mod.ORIGIN_LIFECYCLE:
        return "lifecycle work (created at startup or shutdown)"
    return origin or "unknown origin"


def _retry_eligible(task: Any) -> bool:
    return int(getattr(task, "attempts", 0) or 0) < int(getattr(task, "max_attempts", 0) or 0)


def _next_step(task: Any, *, now: datetime) -> str:
    """What should happen next, honestly, from the task's own state."""
    status = str(getattr(task, "status", "") or "")
    if status == task_mod.COMPLETED:
        return "done; a recurring task will schedule its next occurrence"
    if status == task_mod.CANCELLED:
        return "cancelled; no further action"
    if status == task_mod.EXPIRED:
        return "expired and will not run"
    if status == task_mod.FAILED:
        if _retry_eligible(task):
            return (f"eligible to retry (attempt {getattr(task, 'attempts', 0) + 1}"
                    f"/{getattr(task, 'max_attempts', 3)})")
        return "retries exhausted; will remain failed unless rescheduled"
    if status == task_mod.AWAITING_HUMAN:
        return "waiting for your approval before it may run"
    if status == task_mod.BLOCKED:
        return f"blocked: {getattr(task, 'blocked_reason', None) or 'unstated'}"
    if status == task_mod.SCHEDULED:
        due = _parse(getattr(task, "due_at", None)
                     or getattr(task, "next_due_at", None))
        spec = getattr(task, "schedule", None) or {}
        if due and due > now:
            return f"will be attempted at {_fmt(due, spec.get('timezone'))}"
        # Overdue: the catch-up policy decides, and says so honestly rather than
        # implying a run that the policy will refuse.
        policy = schedules.resolve_missed(spec, due, now=now) if due else schedules.CATCH_UP_RUN
        if policy == schedules.CATCH_UP_SKIP:
            return "missed window; will not run (catch-up=skip)"
        if policy == schedules.CATCH_UP_COALESCE:
            return "will attempt one coalesced run for the missed window"
        return "due now (or overdue); the scheduler will attempt it"
    if status == task_mod.PENDING:
        return "ready; the next scheduler pass may pick it up"
    if status == task_mod.RUNNING:
        return "running now"
    return f"status {status or 'unknown'}"


def _missed_note(task: Any, *, now: datetime) -> Optional[str]:
    """A truthful note when a past-due occurrence was missed, or ``None``."""
    spec = getattr(task, "schedule", None)
    if not isinstance(spec, dict):
        return None
    due = _parse(getattr(task, "due_at", None) or getattr(task, "next_due_at", None))
    if due is None or due >= now:
        return None
    policy = schedules.resolve_missed(spec, due, now=now)
    if policy == schedules.CATCH_UP_RUN:
        return "overdue; catch-up policy is 'run' so it will still be attempted"
    if policy == schedules.CATCH_UP_SKIP:
        return "missed (past its window/deadline); catch-up policy is 'skip'"
    if policy == schedules.CATCH_UP_COALESCE:
        return "missed; catch-up policy is 'coalesce' (one run, not a backlog)"
    return None


def explain(task: Any, *, now: Optional[datetime] = None,
            dependency_lookup: Any = None) -> Dict[str, Any]:
    """A read-only, derived explanation of one task.

    Returns a dict, not prose, so callers can render it however their surface
    needs. ``dependencies_met`` is only reported when a ``dependency_lookup``
    is supplied, because omitted the question cannot be answered truthfully.
    """
    now = now or datetime.now(timezone.utc)
    spec = getattr(task, "schedule", None)
    spec = spec if isinstance(spec, dict) else {}
    result = getattr(task, "result", None)
    outcome = None
    if isinstance(result, dict):
        outcome = result.get("outcome") or result.get("summary") or result.get("status")
    return {
        "id": getattr(task, "id", None),
        "kind": getattr(task, "kind", None),
        "status": getattr(task, "status", None),
        "priority": getattr(task, "priority", None),
        "origin": getattr(task, "origin", None),
        "reason": _reason(task),
        "schedule": schedules.describe(spec) if spec else "(not scheduled)",
        "due_at": getattr(task, "due_at", None),
        "next_due_at": getattr(task, "next_due_at", None),
        "due_local": _fmt(getattr(task, "due_at", None) or getattr(task, "next_due_at", None),
                          spec.get("timezone")),
        "attempts": getattr(task, "attempts", 0),
        "max_attempts": getattr(task, "max_attempts", 0),
        "last_error": getattr(task, "last_error", None),
        "last_outcome": outcome,
        "dependencies_met": (schedules.dependencies_met(spec, dependency_lookup)
                             if dependency_lookup is not None else None),
        "missed_note": _missed_note(task, now=now),
        "next_step": _next_step(task, now=now),
    }


def render(rows: List[Dict[str, Any]], *, limit: Optional[int] = None) -> str:
    """Render ``explain`` rows as a compact, readable block."""
    if not rows:
        return "  (no scheduled or planned activities)"
    shown = rows if limit is None else rows[:limit]
    lines: List[str] = []
    for row in shown:
        lines.append(f"  [{row['id']}] {row['kind']} — {row['status']}")
        lines.append(f"      why:   {row['reason']}")
        lines.append(f"      when:  {row['schedule']}  (due {row['due_local']})")
        lines.append(f"      tries: {row['attempts']}/{row['max_attempts']}"
                     + (f"  last error: {row['last_error']}" if row.get("last_error") else ""))
        if row.get("dependencies_met") is False:
            lines.append("      deps:  unmet (waiting on a dependency)")
        if row.get("missed_note"):
            lines.append(f"      note:  {row['missed_note']}")
        lines.append(f"      next:  {row['next_step']}")
    if limit is not None and len(rows) > limit:
        lines.append(f"  ... and {len(rows) - limit} more")
    return "\n".join(lines)


def explain_all(task_store: Any, *, now: Optional[datetime] = None,
                limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Explain every scheduled/planned task, most urgent first.

    Ordering is by due instant (soonest first), then priority (higher first), so
    what is about to happen leads the list. Read-only.
    """
    now = now or datetime.now(timezone.utc)

    def lookup(ident: str) -> Any:
        getter = getattr(task_store, "get", None)
        return getter(ident) if callable(getter) else None

    scheduled = task_store.scheduled() if hasattr(task_store, "scheduled") else []
    rows = [explain(t, now=now, dependency_lookup=lookup) for t in scheduled]

    def sort_key(row: Dict[str, Any]):
        due = _parse(row.get("due_at") or row.get("next_due_at"))
        return (due is None, due or now, -(row.get("priority") or 0))

    rows.sort(key=sort_key)
    return rows if limit is None else rows[:limit]


__all__ = ["explain", "explain_all", "render"]
