"""Time-aware scheduling: when a durable task is *due*.

This module is the time authority the task store deliberately lacks. The task
store (:mod:`astra.tasks`) owns the durable queue and its lifecycle; this module
owns the pure arithmetic of *when* a task may run - recurrence, local times, time
windows, quiet periods, deadlines, dependencies, retry backoff and expiry - so
that arithmetic is deterministic, testable without I/O, and shared by every
caller.

Two design rules matter more than the features:

* **Time zones are explicit and DST-correct.** A schedule names a zone
  (``America/New_York``, ``Europe/London``, ``UTC``, ...). ``zoneinfo`` resolves
  it, so "09:00 local" stays 09:00 local across a daylight-saving change instead
  of drifting by an hour. When ``zoneinfo`` data is unavailable the schedule
  falls back to a fixed UTC offset, which is honest about the loss of DST rather
  than pretending.
* **Nothing here polls.** A schedule computes its *next* due instant; the runtime
  wakes on a bounded cadence and asks "is anything due?", which is a cheap read,
  not a busy loop.

A schedule is a plain dict so it round-trips through JSON without a schema
migration:

    {
      "mode": "daily" | "interval" | "once",
      "at": "09:00",            # daily: local HH:MM
      "every_seconds": 3600,    # interval: spacing (or "every_hours"/"every_minutes")
      "at_time": "2026-11-01T09:00:00",   # once: local datetime
      "start": "08:00",         # window: run no earlier than this local time
      "end": "22:00",           # window: run no later than this local time
      "deadline": "23:00",      # window-end deadline; miss => missed (not pretended)
      "expires": "2026-12-01T00:00:00",   # no run at/after this instant, ever
      "timezone": "UTC",
      "days": ["mon", "tue"],   # daily: restrict to these weekdays (optional)
      "catch_up": "run" | "skip" | "coalesce",  # missed-run policy
      "priority": 50,
      "kind": "reading",
      "dedupe_key": "nightly",
      "payload": {},
      "depends_on": ["task_abc"],  # ids that must be terminal first
    }
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Dict, List, Optional, Tuple

# -- schedule modes ----------------------------------------------------------
MODE_DAILY = "daily"
MODE_INTERVAL = "interval"
MODE_ONCE = "once"
MODES = (MODE_DAILY, MODE_INTERVAL, MODE_ONCE)

# -- missed-run (catch-up) policies -----------------------------------------
CATCH_UP_RUN = "run"          # it is late but still runs, once, when the runtime returns
CATCH_UP_SKIP = "skip"        # it is past its window: record it as missed, do not run
CATCH_UP_COALESCE = "coalesce"  # like skip, but collapse many missed runs into one
CATCH_UP_POLICIES = (CATCH_UP_RUN, CATCH_UP_SKIP, CATCH_UP_COALESCE)

_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

# A hard cap so a pathological interval cannot make the planner busy.
MIN_INTERVAL_SECONDS = 60


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _parse_hhmm(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _fallback_tz(name: str) -> tzinfo:
    """A fixed-offset zone for names zoneinfo cannot resolve.

    ``UTC+HH:MM`` (or ``UTC-HH:MM``) is honoured; anything else degrades to UTC.
    A fixed offset cannot follow DST, but it is the honest best effort and it
    keeps a schedule usable when the tz database is missing.
    """
    if name in ("UTC", "Z", "GMT"):
        return timezone.utc
    match = re.fullmatch(r"(?:UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?", name)
    if match:
        sign = 1 if match.group(1) == "+" else -1
        hours = int(match.group(2))
        minutes = int(match.group(3) or 0)
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    return timezone.utc


def resolve_tz(name: Any) -> tzinfo:
    """Resolve a zone name via zoneinfo, with an honest fixed-offset fallback."""
    text = str(name or "UTC").strip() or "UTC"
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return _fallback_tz(text)
    try:
        return ZoneInfo(text)
    except Exception:
        return _fallback_tz(text)


def day_key(now: datetime) -> str:
    """The local calendar day (YYYY-MM-DD) of an aware instant."""
    return now.strftime("%Y-%m-%d")


def in_quiet_hours(now: datetime, quiet: Any) -> bool:
    """Whether ``now`` falls inside quiet hours.

    ``quiet`` is ``{"start": "23:00", "end": "07:00"}`` (a window may wrap
    midnight). Absent, empty or unparseable values mean *never quiet* - a
    misconfiguration must never mute Astra by accident.
    """
    if not isinstance(quiet, dict):
        return False
    start = _parse_hhmm(quiet.get("start"))
    end = _parse_hhmm(quiet.get("end"))
    if start is None or end is None:
        return False
    if start == end:
        return False  # zero-width window: not quiet
    minutes = now.hour * 60 + now.minute
    s = start[0] * 60 + start[1]
    e = end[0] * 60 + end[1]
    if s < e:
        return s <= minutes < e
    # Wraps midnight, e.g. 23:00 -> 07:00.
    return minutes >= s or minutes < e


def _in_days(spec: Dict[str, Any], now: datetime) -> bool:
    days = spec.get("days")
    if not days:
        return True
    wanted = {str(d).strip().lower()[:3] for d in days}
    return _WEEKDAYS[now.weekday()] in wanted


def _daily_candidate(spec: Dict[str, Any], local_now: datetime) -> Optional[datetime]:
    at = _parse_hhmm(spec.get("at"))
    if at is None:
        return None
    candidate = local_now.replace(hour=at[0], minute=at[1], second=0, microsecond=0)
    if not _in_days(spec, candidate):
        # Roll forward to the next allowed weekday.
        for _ in range(7):
            candidate += timedelta(days=1)
            if _in_days(spec, candidate):
                return candidate
        return None
    if candidate <= local_now:
        candidate += timedelta(days=1)
        for _ in range(7):
            if _in_days(spec, candidate):
                return candidate
            candidate += timedelta(days=1)
        return None
    return candidate


def _interval_seconds(spec: Dict[str, Any]) -> Optional[int]:
    for key, mult in (("every_seconds", 1), ("every_minutes", 60),
                      ("every_hours", 3600)):
        if spec.get(key) is not None:
            try:
                return max(MIN_INTERVAL_SECONDS, int(float(spec[key]) * mult))
            except (TypeError, ValueError):
                return None
    return None


def _apply_window(spec: Dict[str, Any], candidate: datetime,
                  tz: tzinfo) -> Optional[datetime]:
    start = _parse_hhmm(spec.get("start"))
    end = _parse_hhmm(spec.get("end"))
    if start is None and end is None:
        return candidate
    local = candidate.astimezone(tz)
    if start is not None:
        floor = local.replace(hour=start[0], minute=start[1], second=0, microsecond=0)
        if local < floor:
            local = floor
    if end is not None:
        ceil = local.replace(hour=end[0], minute=end[1], second=0, microsecond=0)
        if start is not None and (end[0] * 60 + end[1]) <= (start[0] * 60 + start[1]):
            # Window wraps midnight (e.g. 22:00 -> 02:00): end is tomorrow.
            ceil += timedelta(days=1)
        if local >= ceil:
            # Past the window's end: no runnable instant today. The caller's
            # catch-up policy decides whether a late run is honest.
            return None
    return local


def next_due(spec: Dict[str, Any], *, last_run: Optional[datetime] = None,
             now: Optional[datetime] = None) -> Optional[datetime]:
    """The next instant this schedule is due, as an aware UTC datetime.

    Pure. ``None`` means the schedule is exhausted (a ``once`` that has passed,
    or one that has expired). The result is compared with ``now`` by the caller;
    this function never sleeps.
    """
    if not isinstance(spec, dict):
        return None
    now = now or _utcnow()
    tz = resolve_tz(spec.get("timezone"))
    local_now = now.astimezone(tz)
    mode = str(spec.get("mode") or MODE_INTERVAL).lower()

    candidate: Optional[datetime] = None
    if mode == MODE_ONCE:
        at = _parse_dt(spec.get("at_time"))
        if at is not None:
            if at.tzinfo is None:
                at = at.replace(tzinfo=tz)
            candidate = at
        else:
            # A wall-clock "once" is today's (or the next) occurrence of `at`.
            at_hhmm = _parse_hhmm(spec.get("at"))
            if at_hhmm is None:
                return None
            candidate = local_now.replace(hour=at_hhmm[0], minute=at_hhmm[1],
                                          second=0, microsecond=0)
    elif mode == MODE_INTERVAL:
        seconds = _interval_seconds(spec)
        if seconds is None:
            return None
        # With no prior run, the first occurrence is one interval from now: a
        # freshly materialised schedule does not all fire at once on startup.
        base = last_run.astimezone(tz) if last_run else local_now
        candidate = base + timedelta(seconds=seconds)
        if candidate <= local_now:
            # A stale last_run (e.g. after downtime) is due now, not skipped.
            candidate = local_now
    elif mode == MODE_DAILY:
        candidate = _daily_candidate(spec, local_now)
    else:
        return None

    if candidate is None:
        return None

    # A ``once`` occurrence that has already run (or whose instant is not after a
    # prior run) is exhausted: it must never be re-armed to run again.
    if mode == MODE_ONCE and last_run is not None \
            and candidate.astimezone(timezone.utc) <= last_run.astimezone(timezone.utc):
        return None

    # Expiry: at/after this instant the schedule never fires again.
    expires = _parse_dt(spec.get("expires"))
    if expires is not None and candidate.astimezone(timezone.utc) >= expires:
        return None

    # Window: clamp the candidate to the window start (do not fire before it),
    # and if it would fall past the window end, there is no instant today.
    candidate = _apply_window(spec, candidate, tz)
    if candidate is None:
        return None
    if expires is not None and candidate.astimezone(timezone.utc) >= expires:
        return None
    return candidate.astimezone(timezone.utc)


def is_late(spec: Dict[str, Any], due: datetime, *, now: Optional[datetime] = None) -> bool:
    """Whether ``due`` is already in the past (a missed or delayed window)."""
    now = now or _utcnow()
    return due <= now


def resolve_missed(spec: Dict[str, Any], due: datetime, *,
                   now: Optional[datetime] = None) -> str:
    """How a past-due schedule should be handled: run / skip / coalesce.

    A deadline or a window end already in the past makes an overdue run
    *missed* - it is recorded honestly rather than pretended. Within the window,
    the configured catch-up policy decides.
    """
    now = now or _utcnow()
    if due > now:
        return CATCH_UP_RUN
    tz = resolve_tz(spec.get("timezone"))
    local_due = due.astimezone(tz)
    local_now = now.astimezone(tz)
    deadline = _parse_hhmm(spec.get("deadline"))
    end = _parse_hhmm(spec.get("end"))
    limit = deadline or end
    if limit is not None:
        limit_at = local_due.replace(hour=limit[0], minute=limit[1], second=0,
                                     microsecond=0)
        if day_key(local_now) == day_key(local_due) and local_now > limit_at:
            return CATCH_UP_SKIP
        if day_key(local_now) != day_key(local_due):
            return CATCH_UP_SKIP
    policy = str(spec.get("catch_up") or CATCH_UP_RUN).lower()
    return policy if policy in CATCH_UP_POLICIES else CATCH_UP_RUN


def dependencies_met(spec: Dict[str, Any], lookup: Any) -> bool:
    """Whether every dependency id exists and is terminal (completed/cancelled).

    ``lookup`` is a callable ``id -> task-or-None``. A missing dependency is a
    *not met* condition (it never silently resolves), and a cancelled dependency
    counts as satisfied so a queue cannot deadlock.
    """
    deps = spec.get("depends_on") or []
    for dep in deps:
        task = lookup(dep) if callable(lookup) else None
        if task is None:
            return False
        status = getattr(task, "status", None)
        if status in ("completed", "cancelled"):
            continue
        return False
    return True


def normalize(spec: Any) -> Optional[Dict[str, Any]]:
    """Validate and normalise a schedule dict; ``None`` if unusable.

    Normalisation is conservative: an invalid schedule is refused rather than
    silently repaired into something that fires at the wrong time.
    """
    if not isinstance(spec, dict):
        return None
    mode = str(spec.get("mode") or "").lower()
    if mode not in MODES:
        return None
    out: Dict[str, Any] = {"mode": mode}
    if mode == MODE_DAILY:
        if _parse_hhmm(spec.get("at")) is None:
            return None
        out["at"] = str(spec.get("at"))
    elif mode == MODE_INTERVAL:
        if _interval_seconds(spec) is None:
            return None
        for key in ("every_seconds", "every_minutes", "every_hours"):
            if spec.get(key) is not None:
                out[key] = spec[key]
    elif mode == MODE_ONCE:
        if _parse_dt(spec.get("at_time")) is None and _parse_hhmm(spec.get("at")) is None:
            return None
        if spec.get("at_time") is not None:
            out["at_time"] = str(spec.get("at_time"))
        else:
            out["at"] = str(spec.get("at"))
    for key in ("start", "end", "deadline", "expires", "timezone",
                "catch_up", "days", "priority", "kind", "dedupe_key", "payload",
                "depends_on", "description"):
        if spec.get(key) is not None:
            out[key] = spec.get(key)
    if out.get("catch_up") is not None:
        policy = str(out["catch_up"]).lower()
        if policy not in CATCH_UP_POLICIES:
            return None
        out["catch_up"] = policy
    return out


def describe(spec: Dict[str, Any]) -> str:
    """A short human-readable summary for the admin views."""
    if not isinstance(spec, dict):
        return "(invalid schedule)"
    mode = str(spec.get("mode") or "?").lower()
    tz = str(spec.get("timezone") or "UTC")
    if mode == MODE_DAILY:
        base = f"daily at {spec.get('at')} {tz}"
    elif mode == MODE_INTERVAL:
        seconds = _interval_seconds(spec) or 0
        base = f"every {seconds}s"
    elif mode == MODE_ONCE:
        base = f"once at {spec.get('at_time') or spec.get('at')} {tz}"
    else:
        base = "unknown schedule"
    bits = [base]
    if spec.get("start") or spec.get("end"):
        bits.append(f"window {spec.get('start')}-{spec.get('end')}")
    if spec.get("catch_up"):
        bits.append(f"catch-up={spec.get('catch_up')}")
    return ", ".join(bits)


__all__ = [
    "MODE_DAILY", "MODE_INTERVAL", "MODE_ONCE", "MODES",
    "CATCH_UP_RUN", "CATCH_UP_SKIP", "CATCH_UP_COALESCE", "CATCH_UP_POLICIES",
    "resolve_tz", "in_quiet_hours", "day_key", "next_due", "is_late",
    "resolve_missed", "dependencies_met", "normalize", "describe",
]
