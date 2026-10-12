"""Adaptation to repeated non-response to proactive messages.

Astra can initiate grounded, policy-approved messages (:mod:`astra.notifications`,
:mod:`astra.autonomy`). This module answers a *different* question: has she been
speaking into silence, and should that make her more reluctant to speak again?

It is deliberately minimal and separate from every existing system:

* It does **not** replace the notification policy. Rate limits, quiet hours,
  dedupe and delivery accounting stay in :class:`astra.notifications.NotificationPolicy`.
  Engagement only *raises the bar* - it can tighten the threshold a message must
  clear, never relax a limit or manufacture a message.
* It does **not** store conversational content. It keeps two counters and a
  little bookkeeping: how many consecutive proactive messages went unanswered,
  and when the last signal (a send, or a reply) happened.
* It is **not** a memory. It never writes a memory record, so it can never be
  retrieved, narrated, or mistaken for knowledge about Roum or about herself.

Model of the world:

* A *delivered* proactive message is a fact about transport, not engagement. A
  send is recorded as a *pending* message; the same send is never counted twice.
* When a human message actually arrives, it is associated with the most recent
  undischarged pending send **only if** it arrives within a reliable window
  (``reply_window_seconds``). Outside that window the message is treated as
  unrelated - Astra was asleep, or Roum was away - and at most one old pending
  send is retired as ambiguous rather than counted as "answered" or "ignored".
* An unanswered message is *weak* evidence. Silence is never rejection,
  loneliness, or a relationship event; the state only counts, and the pull is
  bounded and recovers on its own as time passes.

All clock access is injected, so a test drives it deterministically.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from .memory import atomic_save, load_json

STATE_VERSION = 1

# How long a delivered message stays open to being counted as answered. A reply
# after this is treated as unrelated (Roum was away), not as engagement with it.
DEFAULT_REPLY_WINDOW_SECONDS = 6 * 3600

# Consecutive unanswered messages before adaptation starts to bite.
DEFAULT_SILENCE_AFTER = 2
# Extra threshold a fully-adapted state adds (added to the policy's own min).
DEFAULT_MAX_THRESHOLD_BOOST = 0.25
# Extra threshold per consecutive unanswered message beyond the first counted.
DEFAULT_THRESHOLD_STEP = 0.08
# Elapsed time over which the pull decays back toward an unanswered baseline.
DEFAULT_DECAY_DAYS = 3.0

_PENDING_LIMIT = 20


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


class EngagementTracker:
    """Persistent, bounded adaptation to unanswered proactive messages."""

    def __init__(
        self,
        data_dir: str = "./storage",
        *,
        config: Optional[Dict[str, Any]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "engagement_state.json")
        cfg = config if isinstance(config, dict) else {}
        self.enabled = bool(cfg.get("enabled", True))
        self.reply_window_seconds = float(cfg.get(
            "reply_window_seconds", DEFAULT_REPLY_WINDOW_SECONDS))
        self.silence_after = max(1, int(cfg.get("silence_after", DEFAULT_SILENCE_AFTER)))
        self.max_threshold_boost = float(cfg.get(
            "max_threshold_boost", DEFAULT_MAX_THRESHOLD_BOOST))
        self.threshold_step = float(cfg.get(
            "threshold_step", DEFAULT_THRESHOLD_STEP))
        self.decay_days = max(0.01, float(cfg.get("decay_days", DEFAULT_DECAY_DAYS)))
        self._now = now_fn or _utcnow
        self._lock = threading.RLock()
        self._state = self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        raw = load_json(self.path, dict, dict)
        raw = raw if isinstance(raw, dict) else {}
        return {
            "version": STATE_VERSION,
            "pending": [p for p in (raw.get("pending") or []) if isinstance(p, dict)][-_PENDING_LIMIT:],
            "last_send_at": raw.get("last_send_at"),
            "last_reply_at": raw.get("last_reply_at"),
            "last_signal_at": raw.get("last_signal_at"),
        }

    def _persist(self) -> None:
        with self._lock:
            atomic_save(self.path, self._state)

    # -- recording a send -------------------------------------------------
    def note_sent(self, ref_id: Any = None, *, channel_id: Any = None,
                  now: Optional[datetime] = None) -> None:
        """Record that a proactive message was *delivered* (a transport fact).

        A delivered message becomes an open obligation: it is counted as
        unanswered until a reply is reliably associated with it. Deduped by
        ``ref_id`` so a re-send of the same outcome is never double-counted, and
        so a restart cannot inflate the count.
        """
        if not self.enabled:
            return
        now = now or self._now()
        key = str(ref_id) if ref_id else ""
        with self._lock:
            pending = self._state.setdefault("pending", [])
            if key and any(str(p.get("ref_id")) == key for p in pending):
                return  # already an open obligation for this outcome
            pending.append({"ref_id": key, "channel_id": str(channel_id or ""),
                            "sent_at": now.isoformat()})
            self._state["pending"] = pending[-_PENDING_LIMIT:]
            self._state["last_send_at"] = now.isoformat()
            self._state["last_signal_at"] = now.isoformat()
            self._persist()

    # -- recording a reply ------------------------------------------------
    def note_reply(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Record that a human message *actually arrived* (from any transport).

        Associates it with the most recent undischarged send only when that send
        is within the reliable reply window. Otherwise the message is unrelated:
        the oldest stale obligation is retired as *ambiguous* (neither answered
        nor counted as ignored), so an away period does not accrue as silence.
        """
        if not self.enabled:
            return {"associated": False, "reason": "disabled"}
        now = now or self._now()
        with self._lock:
            self._state["last_reply_at"] = now.isoformat()
            self._state["last_signal_at"] = now.isoformat()
            pending = self._state.get("pending") or []
            if not pending:
                self._persist()
                return {"associated": False, "reason": "no_pending"}
            newest = pending[-1]
            sent = _parse(newest.get("sent_at"))
            within = (sent is not None
                      and (now - sent).total_seconds() <= self.reply_window_seconds)
            if within:
                # Real engagement: the reply answers the newest message, and the
                # whole silence chain is discharged. Nothing is counted.
                self._state["pending"] = []
                self._persist()
                return {"associated": True, "ref_id": newest.get("ref_id")}
            # Too old to attribute reliably: retire one (the oldest) as
            # ambiguous, and do not treat the reply as engagement with it.
            pending.pop(0)
            self._state["pending"] = pending
            self._persist()
            return {"associated": False, "reason": "out_of_window"}

    # -- adaptation -------------------------------------------------------
    def unanswered_count(self, now: Optional[datetime] = None) -> int:
        """How many delivered messages have gone unanswered.

        A message only counts once it is older than the reply window: a freshly
        sent message is *awaiting*, not unanswered, so a normal reply is never
        penalised. This is derived from the pending list, never incremented
        separately, so it cannot drift from what actually happened.
        """
        now = now or self._now()
        with self._lock:
            pending = self._state.get("pending") or []
            count = 0
            for p in pending:
                sent = _parse(p.get("sent_at"))
                if sent is not None and (now - sent).total_seconds() >= self.reply_window_seconds:
                    count += 1
            return count

    def _decayed_count(self, now: datetime) -> float:
        """The counted silence, decayed toward an unanswered baseline with time.

        Silence is weak evidence, so its *effect* fades: with no new sends, the
        pull slides from the recorded count toward ``1`` across ``decay_days``,
        which is at or below ``silence_after``, so time alone recovers her.
        """
        raw = float(self.unanswered_count(now))
        if raw <= 0:
            return 0.0
        if raw <= 1:
            return raw
        last_send = _parse(self._state.get("last_send_at"))
        if last_send is None:
            return raw
        days = max(0.0, (now - last_send).total_seconds() / 86400.0)
        frac = min(1.0, days / self.decay_days)
        return max(1.0, raw - (raw - 1.0) * frac)

    def effective_threshold_boost(self, now: Optional[datetime] = None) -> float:
        """The extra threshold her current silence adds (0..``max_threshold_boost``).

        Zero until ``silence_after`` messages have gone unanswered, then rising
        by ``threshold_step`` per further message, capped, and scaled by the
        decayed count. Raising the bar is the *only* effect: the notification
        policy still applies every limit and quiet hours.
        """
        if not self.enabled:
            return 0.0
        now = now or self._now()
        with self._lock:
            count = self._decayed_count(now)
        if count <= self.silence_after:
            return 0.0
        steps = count - self.silence_after
        boost = min(self.max_threshold_boost, steps * self.threshold_step)
        return round(max(0.0, boost), 4)

    def threshold_scale(self, now: Optional[datetime] = None) -> float:
        """A multiplier for the importance threshold (>= 1), for callers that
        want a proportional form rather than an absolute addend."""
        base = 0.55
        boost = self.effective_threshold_boost(now)
        return round(min(2.0, (base + boost) / base), 4)

    def consecutive_unanswered(self, now: Optional[datetime] = None) -> int:
        now = now or self._now()
        with self._lock:
            return int(round(self._decayed_count(now)))

    def suppressed(self, now: Optional[datetime] = None) -> bool:
        """Reserved: adaptation raises the bar, it does not suppress outright.

        Kept as an explicit, always-False method (rather than a comment) so the
        contract is visible: silence never mutes her.
        """
        return False

    def snapshot(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        now = now or self._now()
        with self._lock:
            return {
                "enabled": self.enabled,
                "consecutive_unanswered": self.consecutive_unanswered(now),
                "pending": len(self._state.get("pending") or []),
                "threshold_boost": self.effective_threshold_boost(now),
                "last_send_at": self._state.get("last_send_at"),
                "last_reply_at": self._state.get("last_reply_at"),
                "suppressed": self.suppressed(now),
            }


__all__ = [
    "EngagementTracker",
    "DEFAULT_REPLY_WINDOW_SECONDS",
    "DEFAULT_SILENCE_AFTER",
    "DEFAULT_MAX_THRESHOLD_BOOST",
]
