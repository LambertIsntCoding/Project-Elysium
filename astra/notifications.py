"""Proactive notification policy: whether an outcome may become a message.

This is the *decision layer* the brief asks for, and it is deliberately separate
from both the reasons to speak (:mod:`astra.proactive`) and the transport
(:mod:`astra.discord_adapter`). It answers one question about an already-formed
outcome - an activity result, a discovery, an obstacle - with one of three
dispositions:

* ``deliver``  - send a standalone message now;
* ``digest``   - hold it for the next scheduled summary;
* ``drop``     - do not communicate it (it is still recorded in the ledger).

Nothing here generates a reason to speak. With no concrete outcome it returns
``drop``; it never manufactures a message to prove Astra is active.

The limits the brief requires live here, all configurable, never hardcoded at
the call sites:

* a minimum gap between unsolicited standalone messages;
* a cap per rolling interval and per local calendar day;
* quiet hours;
* an importance/relevance threshold;
* a higher-priority class that may bypass *ordinary* limits only (never quiet
  hours, a pause, or a mute - those are the user's standing decisions).

Time is only ever used to *hold back*, never to prompt.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from . import schedules as sched
from .memory import atomic_save, load_json

STATE_VERSION = 1

# Dispositions.
DELIVER = "deliver"
DIGEST = "digest"
DROP = "drop"

# Why a delivery was held back (mirrors the proactive suppression vocabulary).
HOLD_MUTED = "muted"
HOLD_PAUSED = "paused"
HOLD_QUIET = "quiet_hours"
HOLD_MIN_GAP = "min_gap"
HOLD_INTERVAL_CAP = "interval_cap"
HOLD_DAILY_CAP = "daily_cap"
HOLD_LOW_IMPORTANCE = "low_importance"
HOLD_DUPLICATE = "duplicate"
HOLD_NO_CONTENT = "no_content"

# Conservative, configurable defaults from the brief: at most one unsolicited
# standalone message per eight hours, and at most two proactive messages per
# local day (a digest counts).
DEFAULT_MIN_GAP_SECONDS = 8 * 3600
DEFAULT_INTERVAL_SECONDS = 8 * 3600
DEFAULT_MAX_PER_INTERVAL = 1
DEFAULT_MAX_PER_DAY = 2
DEFAULT_MIN_IMPORTANCE = 0.55
DEFAULT_HIGH_PRIORITY_IMPORTANCE = 0.85
# How many recent delivered ids / timestamps to keep for dedupe and limits.
RECENT_LIMIT = 100


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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


class NotificationPolicy:
    """Persistent, configurable limits + the deliver/digest/drop decision."""

    def __init__(
        self,
        data_dir: str = "./storage",
        *,
        config: Optional[Dict[str, Any]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "notifications_state.json")
        cfg = config if isinstance(config, dict) else {}
        self.timezone = str(cfg.get("timezone") or "UTC")
        self.min_gap_seconds = float(cfg.get("min_gap_seconds", DEFAULT_MIN_GAP_SECONDS))
        self.interval_seconds = float(cfg.get("interval_seconds", DEFAULT_INTERVAL_SECONDS))
        self.max_per_interval = int(cfg.get("max_per_interval", DEFAULT_MAX_PER_INTERVAL))
        self.max_per_day = int(cfg.get("max_per_day", DEFAULT_MAX_PER_DAY))
        self.min_importance = float(cfg.get("min_importance", DEFAULT_MIN_IMPORTANCE))
        self.high_priority_importance = float(cfg.get(
            "high_priority_importance", DEFAULT_HIGH_PRIORITY_IMPORTANCE))
        self.allow_high_priority_bypass = bool(cfg.get(
            "allow_high_priority_bypass", True))
        self.quiet = cfg.get("quiet_hours") if isinstance(cfg.get("quiet_hours"), dict) else {}
        self.digest_enabled = bool(cfg.get("digest_enabled", True))
        self.digest_at = cfg.get("digest_at")  # local HH:MM
        self._muted = bool(cfg.get("muted", False))
        self._now = now_fn or _utcnow
        # Optional: a callable returning an extra threshold (0..N) driven by
        # adaptation to non-response. It can only *raise* the bar; it never
        # relaxes a limit, quiet hours, or a pause. Injected after construction
        # so the dependency stays one-way (policy does not import engagement).
        self.engagement_provider: Optional[Callable[[], float]] = None
        # Optional: a callable invoked after a *standalone* message is delivered,
        # so the engagement tracker can open an obligation. It receives the ref
        # id and a ``now`` keyword. Best-effort; a failure never affects delivery.
        self.note_sent_hook: Optional[Callable[..., None]] = None
        self._lock = threading.RLock()
        self._state = self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        raw = load_json(self.path, dict, dict)
        raw = raw if isinstance(raw, dict) else {}
        return {
            "version": STATE_VERSION,
            "paused": bool(raw.get("paused", False)),
            "delivered": [t for t in (raw.get("delivered") or []) if isinstance(t, str)][-RECENT_LIMIT:],
            "delivered_ids": [str(i) for i in (raw.get("delivered_ids") or [])][-RECENT_LIMIT:],
            "pending_digest": [e for e in (raw.get("pending_digest") or []) if isinstance(e, dict)][-50:],
            "last_digest_at": raw.get("last_digest_at"),
        }

    def reload(self) -> None:
        """Re-read the persisted state, for a separate inspection process."""
        with self._lock:
            self._state = self._load()

    def _persist(self) -> None:
        with self._lock:
            atomic_save(self.path, self._state)

    # -- pause / mute ----------------------------------------------------
    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)

    @property
    def paused(self) -> bool:
        return bool(self._state.get("paused"))

    def set_paused(self, paused: bool, *, persist: bool = True) -> None:
        with self._lock:
            self._state["paused"] = bool(paused)
            if persist:
                self._persist()

    # -- limit accounting ------------------------------------------------
    def recent_deliveries(self) -> List[datetime]:
        out = [_parse(t) for t in (self._state.get("delivered") or [])]
        return [t for t in out if t is not None]

    def delivered_in_interval(self, now: Optional[datetime] = None) -> int:
        now = now or self._now()
        cutoff = now - timedelta(seconds=self.interval_seconds)
        return sum(1 for t in self.recent_deliveries() if t >= cutoff)

    def delivered_on_day(self, now: Optional[datetime] = None) -> int:
        now = now or self._now()
        tz = sched.resolve_tz(self.timezone)
        key = sched.day_key(now.astimezone(tz))
        count = 0
        for stamp in self.recent_deliveries():
            if sched.day_key(stamp.astimezone(tz)) == key:
                count += 1
        return count

    def last_delivery(self) -> Optional[datetime]:
        stamps = self.recent_deliveries()
        return max(stamps) if stamps else None

    def in_quiet_hours(self, now: Optional[datetime] = None) -> bool:
        now = now or self._now()
        tz = sched.resolve_tz(self.timezone)
        return sched.in_quiet_hours(now.astimezone(tz), self.quiet)

    def is_duplicate(self, ref_id: Any) -> bool:
        if not ref_id:
            return False
        with self._lock:
            ids = self._state.get("delivered_ids") or []
            pending = [e.get("ref_id") for e in (self._state.get("pending_digest") or [])]
        return str(ref_id) in ids or str(ref_id) in pending

    def effective_min_importance(self) -> float:
        """The importance threshold, raised by adaptation to non-response.

        The base threshold is the configured ``min_importance``. When proactive
        messages have gone unanswered, the injected provider adds a bounded
        boost (capped at 1.0). This only ever *raises* the bar - no limit, quiet
        hour, or pause is relaxed by it, and a genuinely high-priority outcome
        can still bypass ordinary limits exactly as before.
        """
        if self.engagement_provider is None:
            return self.min_importance
        try:
            boost = float(self.engagement_provider() or 0.0)
        except Exception:
            boost = 0.0
        return round(min(1.0, self.min_importance + max(0.0, boost)), 4)

    # -- the decision ----------------------------------------------------
    def decide(self, outcome: Dict[str, Any], *, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Decide the disposition of one concrete outcome.

        ``outcome`` must carry a ``ref_id`` and an ``importance`` plus a short
        ``summary``. A malformed or empty outcome is dropped - there is no path
        from "nothing happened" to a message.
        """
        now = now or self._now()
        if not isinstance(outcome, dict):
            return self._verdict(DROP, HOLD_NO_CONTENT, now)
        ref_id = outcome.get("ref_id")
        summary = str(outcome.get("summary") or "").strip()
        if not ref_id or not summary:
            return self._verdict(DROP, HOLD_NO_CONTENT, now)

        importance = _clamp(outcome.get("importance"))
        high = importance >= self.high_priority_importance

        if self._muted:
            return self._verdict(DIGEST, HOLD_MUTED, now, importance=importance)
        if self.paused:
            return self._verdict(DIGEST, HOLD_PAUSED, now, importance=importance)
        if self.is_duplicate(ref_id):
            return self._verdict(DROP, HOLD_DUPLICATE, now, importance=importance)
        if importance < self.effective_min_importance():
            # Not worth an interruption; keep it for a digest only if it has
            # some substance.
            return self._verdict(DIGEST, HOLD_LOW_IMPORTANCE, now,
                                 importance=importance)
        # Quiet hours are the user's standing decision: never bypassed.
        if self.in_quiet_hours(now):
            return self._verdict(DIGEST, HOLD_QUIET, now, importance=importance)

        bypass = bool(high and self.allow_high_priority_bypass)
        # Ordinary limits (bypassable only by a genuinely high-priority outcome).
        last = self.last_delivery()
        if last is not None and not bypass:
            if (now - last).total_seconds() < self.min_gap_seconds:
                return self._verdict(DIGEST, HOLD_MIN_GAP, now, importance=importance)
        if not bypass and self.delivered_in_interval(now) >= self.max_per_interval:
            return self._verdict(DIGEST, HOLD_INTERVAL_CAP, now, importance=importance)
        if not bypass and self.delivered_on_day(now) >= self.max_per_day:
            return self._verdict(DIGEST, HOLD_DAILY_CAP, now, importance=importance)

        return self._verdict(DELIVER, None, now, importance=importance,
                             bypassed=bypass and (self.delivered_in_interval(now) > 0
                                                  or self.delivered_on_day(now) > 0))

    def _verdict(self, disposition: str, suppressed_by: Optional[str],
                 now: datetime, *, importance: float = 0.0,
                 bypassed: bool = False) -> Dict[str, Any]:
        return {
            "disposition": disposition,
            "suppressed_by": suppressed_by,
            "importance": round(importance, 4),
            "bypassed": bool(bypassed),
            "at": now.isoformat(),
        }

    # -- digest ----------------------------------------------------------
    def queue_digest(self, entry: Dict[str, Any]) -> None:
        """Hold a lower-priority outcome for the next summary."""
        if not isinstance(entry, dict) or not entry.get("ref_id"):
            return
        with self._lock:
            pending = self._state.setdefault("pending_digest", [])
            if any(e.get("ref_id") == entry.get("ref_id") for e in pending):
                return
            pending.append({
                "ref_id": str(entry.get("ref_id")),
                "kind": str(entry.get("kind") or "note"),
                "summary": _clean(entry.get("summary")),
                "importance": round(_clamp(entry.get("importance")), 4),
                "at": (entry.get("at") or _utcnow()).isoformat()
                if isinstance(entry.get("at"), datetime) else str(entry.get("at") or _utcnow().isoformat()),
            })
            self._state["pending_digest"] = pending[-50:]
            self._persist()

    def digest_due(self, now: Optional[datetime] = None) -> bool:
        """Whether a scheduled digest should be produced now."""
        if not self.digest_enabled or self._muted or self.paused:
            return False
        if not (self._state.get("pending_digest")):
            return False
        now = now or self._now()
        if self.in_quiet_hours(now):
            return False
        if self.digest_at:
            tz = sched.resolve_tz(self.timezone)
            local = now.astimezone(tz)
            at = _parse_hhmm(self.digest_at)
            last = _parse(self._state.get("last_digest_at"))
            if at is not None:
                today = local.replace(hour=at[0], minute=at[1], second=0, microsecond=0)
                if local < today:
                    return False
                if last is not None and last.astimezone(tz) >= today:
                    return False
        # A digest counts toward the daily cap, like a standalone message.
        if self.delivered_on_day(now) >= self.max_per_day:
            return False
        return True

    def pending_digest_count(self) -> int:
        return len(self._state.get("pending_digest") or [])

    def take_digest(self) -> List[Dict[str, Any]]:
        """Consume and return the queued digest entries."""
        with self._lock:
            entries = list(self._state.get("pending_digest") or [])
            self._state["pending_digest"] = []
            self._persist()
        return entries

    # -- recording delivery ----------------------------------------------
    def note_delivered(self, ref_id: Optional[str], *, now: Optional[datetime] = None,
                       is_digest: bool = False) -> None:
        """Record a *successful* delivery. Called only after a real send.

        Also opens an engagement obligation (via the injected tracker, if any):
        a delivered message counts as unanswered until a reply is reliably
        associated with it. A digest does not open an obligation - it is a
        summary, not a message awaiting a reply.
        """
        now = now or self._now()
        with self._lock:
            stamps = list(self._state.get("delivered") or [])
            stamps.append(now.isoformat())
            self._state["delivered"] = stamps[-RECENT_LIMIT:]
            if ref_id:
                ids = list(self._state.get("delivered_ids") or [])
                ids.append(str(ref_id))
                self._state["delivered_ids"] = ids[-RECENT_LIMIT:]
            if is_digest:
                self._state["last_digest_at"] = now.isoformat()
            self._persist()
        if not is_digest and self.note_sent_hook is not None:
            try:
                self.note_sent_hook(ref_id, now=now)
            except Exception:
                pass

    def snapshot(self) -> Dict[str, Any]:
        now = self._now()
        return {
            "muted": self._muted,
            "paused": self.paused,
            "delivered_in_interval": self.delivered_in_interval(now),
            "max_per_interval": self.max_per_interval,
            "delivered_today": self.delivered_on_day(now),
            "max_per_day": self.max_per_day,
            "min_gap_seconds": self.min_gap_seconds,
            "quiet_hours": dict(self.quiet),
            "in_quiet_hours": self.in_quiet_hours(now),
            "pending_digest": self.pending_digest_count(),
            "digest_enabled": self.digest_enabled,
        }


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _clean(text: Any, limit: int = 200) -> str:
    return " ".join(str(text or "").split())[:limit]


def _parse_hhmm(value: Any):
    return sched._parse_hhmm(value)


__all__ = [
    "NotificationPolicy", "DELIVER", "DIGEST", "DROP",
    "HOLD_MUTED", "HOLD_PAUSED", "HOLD_QUIET", "HOLD_MIN_GAP",
    "HOLD_INTERVAL_CAP", "HOLD_DAILY_CAP", "HOLD_LOW_IMPORTANCE",
    "HOLD_DUPLICATE", "HOLD_NO_CONTENT",
    "DEFAULT_MIN_GAP_SECONDS", "DEFAULT_MAX_PER_DAY", "DEFAULT_MIN_IMPORTANCE",
]
