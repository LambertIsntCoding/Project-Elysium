"""The autonomy driver: one bounded "wake" that advances Astra off the console.

This is the coordinator the runtime calls on a bounded cadence - the piece that
makes scheduled work, bounded self-selected activity and proactive communication
happen without an interactive conversation. It owns no store of its own and
introduces no second source of truth: it reads and writes the durable state that
already exists (the task store, the ledger, the notification policy) through
those owners.

One wake does exactly five bounded things, in order:

1. **Advance schedules** (:class:`astra.schedule_runtime.ScheduleDriver`), so a
   schedule that came due while the runtime was down is handled on return.
2. **Run scheduled work** through a scheduler built with ``require_sleep=False``
   - configured activities must run even while a human is engaged, which is what
   distinguishes them from the opportunistic sleep-only work.
3. **Plan** a small number of new activities from real state
   (:class:`astra.activity.ActivityPlanner`), unless autonomy is paused or the
   queue is at its bound. Idle is a legitimate result and is reported as such.
4. **Consider outcomes** from the ledger since the last wake and ask the
   notification policy whether any deserves a message.
5. **Deliver** a digest when one is due.

It never fabricates a notification for a task that did not run, and a delivery
failure is recorded as a failure - it is never reported as a message the user
received. All clock access is injected so a test drives it deterministically.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from . import notifications as notify_mod
from . import proactive as proactive_mod
from . import tasks as task_mod

# Default importance of a completed activity kind, when the handler does not
# state one. Ordinary maintenance is low; a discovery/obstacle raised by a
# handler overrides this via its result payload.
_KIND_IMPORTANCE = {
    task_mod.KIND_SELF_IMPROVEMENT: 0.6,
    task_mod.KIND_KNOWLEDGE_GAP: 0.55,
    task_mod.KIND_EXPERIENCE_REVIEW: 0.5,
    task_mod.KIND_SELF_REVIEW: 0.45,
    task_mod.KIND_MEMORY_REVIEW: 0.45,
    task_mod.KIND_WORK_REVIEW: 0.5,
    task_mod.KIND_READING: 0.5,
    task_mod.KIND_TEST_RUN: 0.5,
    task_mod.KIND_QUESTION_REVIEW: 0.4,
    task_mod.KIND_MEMORY_MAINTENANCE: 0.3,
    task_mod.KIND_APP_MAINTENANCE: 0.25,
    task_mod.KIND_CONTEXT_PREPARATION: 0.3,
}

# Statuses that are worth reporting as an obstacle rather than progress.
_PROBLEM_STATUSES = (task_mod.FAILED, task_mod.BLOCKED, task_mod.MISSED,
                     task_mod.EXPIRED)


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


class AutonomyDriver:
    """Bounded, time-driven coordination of Astra's autonomous work."""

    def __init__(
        self,
        *,
        schedule_driver: Any,
        planner: Any,
        notifications: notify_mod.NotificationPolicy,
        scheduled_scheduler: Any,
        ledger: Any,
        notifier: Optional[Callable[[str, Optional[str], Dict[str, Any]], Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        cfg = config if isinstance(config, dict) else {}
        self.schedule_driver = schedule_driver
        self.planner = planner
        self.notifications = notifications
        self.scheduled_scheduler = scheduled_scheduler
        self.ledger = ledger
        self.notifier = notifier
        self._now = now_fn or _utcnow
        self.wake_seconds = max(5.0, float(cfg.get("wake_interval_seconds", 30.0)))
        self.plan_enabled = bool(cfg.get("planning_enabled", True))
        self.enabled = bool(cfg.get("enabled", True))
        self._lock = threading.RLock()
        # Activity pause is deliberately *separate* from the notification
        # policy's pause: pausing autonomous work must not silence messaging, and
        # silencing messaging must not stop autonomous work.
        self._activity_paused = False
        self.last_wake_at: Optional[str] = None
        self._last_outcome_seen: Optional[datetime] = None
        self.last_report: Dict[str, Any] = {}
        # Optional injectable readers (same pattern as the controller's
        # affect_provider): grounded reasons that are not task outcomes, and a
        # composer that may phrase an already-permitted message. Neither can
        # change *whether* a message is sent - the notification policy owns that.
        self.candidate_provider: Optional[Callable[[], List[Dict[str, Any]]]] = None
        self.relevance_provider: Optional[Callable[[], float]] = None
        self.composer: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None
        # Optional hook that folds a *problem* outcome (a failure/block/miss) into
        # the affect state. Injected by the application so the driver itself never
        # imports affect or the store: this is a one-directional signal, and the
        # store stays the single write authority. It must never affect whether a
        # notification is sent.
        self.outcome_affect_hook: Optional[Callable[[Dict[str, Any]], None]] = None

    # -- pause/unpause (independent of notifications) --------------------
    @property
    def paused(self) -> bool:
        return self._activity_paused

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._activity_paused = bool(paused)

    # -- one bounded wake ------------------------------------------------
    def wake(self, *, user_active: bool = False,
             now: Optional[datetime] = None) -> Dict[str, Any]:
        """Advance everything once. Never raises; returns an honest report."""
        now = now or self._now()
        report: Dict[str, Any] = {"at": now.isoformat(), "user_active": user_active}
        with self._lock:
            if not self.enabled:
                report["skipped"] = "disabled"
                self.last_report = report
                return report

            # 1 + 2. Schedules and the work they publish.
            if self.schedule_driver is not None:
                try:
                    report["schedules"] = self.schedule_driver.refresh()
                except Exception as exc:
                    report["schedules_error"] = str(exc)
            if not self.paused and self.scheduled_scheduler is not None:
                try:
                    report["scheduled_cycle"] = self.scheduled_scheduler.run_cycle()
                except Exception as exc:
                    report["scheduled_error"] = str(exc)
            else:
                report["scheduled_cycle"] = {"reason": "paused" if self.paused else "disabled"}

            # 3. Planning (only when not paused and not interacting).
            if self.plan_enabled and not self.paused and not user_active and self.planner is not None:
                try:
                    plan = self.planner.plan(now=now)
                    report["plan"] = {"decision": plan.get("decision"),
                                      "count": len(plan.get("proposals") or [])}
                    self._enqueue_plan(plan.get("proposals") or [], now)
                except Exception as exc:
                    report["plan_error"] = str(exc)
            else:
                report["plan"] = {"decision": "skipped"}

            # 4. Outcomes -> notifications.
            report["notifications"] = self._dispatch_outcomes(now, user_active=user_active)

            # 4b. Grounded non-task reasons -> the *same* notification policy.
            report["candidate_notifications"] = self._dispatch_candidates(
                now, user_active=user_active)

            # 5. Digest.
            report["digest"] = self._maybe_digest(now, user_active=user_active)

        self.last_wake_at = now.isoformat()
        self.last_report = report
        return report

    # -- planning --------------------------------------------------------
    def _enqueue_plan(self, proposals: List[Dict[str, Any]],
                      now: datetime) -> List[str]:
        """Enqueue at most the proposals the planner evidenced, deduped."""
        ids: List[str] = []
        for prop in proposals:
            kind = prop.get("kind")
            if kind not in task_mod.TASK_KINDS:
                continue
            task = task_mod.BackgroundTask(
                kind=kind, origin=task_mod.ORIGIN_ACTIVITY,
                priority=int(prop.get("priority") or 25),
                dedupe_key=str(prop.get("dedupe_key") or kind),
                payload=dict(prop.get("payload") or {}),
                progress={"why": prop.get("reason"), "lane": prop.get("lane"),
                          "scope": prop.get("scope"), "selected_at": now.isoformat()},
            )
            stored = self.planner.task_store.enqueue(task) if self.planner else None
            if stored is not None:
                ids.append(stored.id)
                if self.ledger is not None:
                    self.ledger.record(task_id=stored.id, kind=kind,
                                       status=task_mod.PENDING,
                                       detail=f"selected: {prop.get('reason')}")
        return ids

    # -- notifications ---------------------------------------------------
    def _dispatch_outcomes(self, now: datetime,
                           *, user_active: bool) -> Dict[str, Any]:
        """Turn newly recorded ledger outcomes into (at most) a delivery."""
        if self.ledger is None:
            return {"considered": 0, "delivered": 0}
        since = self._last_outcome_seen
        try:
            entries = self.ledger.entries()
        except Exception:
            entries = []
        considered = delivered = queued = 0
        for entry in entries:
            stamp = _parse(entry.get("at"))
            if stamp is None:
                continue
            if since is not None and stamp <= since:
                continue
            if entry.get("status") in (task_mod.PENDING, task_mod.SKIPPED):
                # A selection note, or a deferred/non-event (a reading hold, a
                # window that passed). Neither is something to report; the
                # ledger still records it for inspection.
                continue
            considered += 1
            # A genuine setback is felt before any notification decision, so her
            # state reflects the outcome regardless of whether it is worth a
            # message. The hook is affect-only and cannot change delivery. It is
            # gated on the watermark being established so the first pass after a
            # restart does not replay the whole ledger's history as fresh
            # frustration - only outcomes that happen *now* are felt.
            if self.outcome_affect_hook is not None and since is not None:
                try:
                    self.outcome_affect_hook(entry)
                except Exception:
                    pass
            outcome = self._outcome_from_entry(entry)
            if outcome is None:
                continue
            # Being present is a standing reason not to interrupt.
            if user_active:
                decision = self.notifications.decide(outcome, now=now)
                if decision["disposition"] != notify_mod.DROP:
                    self.notifications.queue_digest(outcome)
                    queued += 1
                continue
            decision = self.notifications.decide(outcome, now=now)
            disp = decision["disposition"]
            if disp == notify_mod.DELIVER:
                if self._deliver(outcome, now):
                    delivered += 1
                else:
                    # Delivery failed: hold it for a digest so it is not lost and
                    # is never counted as received.
                    self.notifications.queue_digest(outcome)
                    queued += 1
            elif disp == notify_mod.DIGEST:
                self.notifications.queue_digest(outcome)
                queued += 1
        if entries:
            newest = max((_parse(e.get("at")) for e in entries if _parse(e.get("at"))),
                         default=None)
            if newest is not None:
                self._last_outcome_seen = newest
        return {"considered": considered, "delivered": delivered, "queued": queued}

    def _dispatch_candidates(self, now: datetime,
                             *, user_active: bool) -> Dict[str, Any]:
        """Grounded non-task reasons -> the same notification policy as outcomes.

        Conversation candidates (unresolved questions, realisations, recorded
        conflicts, discoveries) are mapped to notification outcomes and run
        through the *existing* policy, so rate limits, quiet hours, dedupe and
        message-counting are shared with the task-outcome path. A candidate
        whose weight does not clear ``min_importance`` is not delivered - it is
        at most queued for a digest, exactly like a low-importance outcome.
        """
        if self.candidate_provider is None or self.notifier is None:
            return {"considered": 0, "delivered": 0, "queued": 0}
        try:
            candidates = self.candidate_provider() or []
        except Exception as exc:
            return {"considered": 0, "delivered": 0, "queued": 0,
                    "error": str(exc)}
        relevance = 0.0
        if self.relevance_provider is not None:
            try:
                relevance = float(self.relevance_provider() or 0.0)
            except Exception:
                relevance = 0.0

        considered = delivered = queued = 0
        for candidate in candidates:
            outcome = proactive_mod.candidate_outcome(candidate)
            if outcome is None:
                continue
            # Fold her affect-derived relevance in *before* the decision, so a
            # forward pull can lift a borderline grounded reason over the bar.
            # It can only raise importance and is clamped - it never invents a
            # reason and never lifts an arbitrary candidate into a message.
            outcome = self._effective_outcome(outcome, relevance)
            considered += 1
            # A candidate that would be an *interruption* when a human is present
            # is still, in reality, not an interruption: they are here. Route it
            # to the digest rather than speaking over them.
            if user_active:
                decision = self.notifications.decide(outcome, now=now)
                if decision["disposition"] != notify_mod.DROP:
                    self.notifications.queue_digest(outcome)
                    queued += 1
                continue
            decision = self.notifications.decide(outcome, now=now)
            disp = decision["disposition"]
            if disp == notify_mod.DELIVER:
                if self._deliver(outcome, now):
                    delivered += 1
                else:
                    self.notifications.queue_digest(outcome)
                    queued += 1
            elif disp == notify_mod.DIGEST:
                self.notifications.queue_digest(outcome)
                queued += 1
        return {"considered": considered, "delivered": delivered, "queued": queued}

    def _outcome_from_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        status = str(entry.get("status") or "")
        kind = str(entry.get("kind") or "")
        task_id = entry.get("task_id")
        detail = str(entry.get("detail") or "").strip()
        problem = status in _PROBLEM_STATUSES
        # A recorded *problem* is worth a message even if detail is terse.
        if not problem and not detail:
            return None
        importance = _KIND_IMPORTANCE.get(kind, 0.4)
        if problem:
            importance = max(importance, 0.8)
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        if result.get("importance") is not None:
            try:
                importance = float(result["importance"])
            except (TypeError, ValueError):
                pass
        summary = result.get("summary") or detail or f"{kind} {status}"
        ref_id = f"outcome:{task_id}:{entry.get('at')}"
        return {
            "ref_id": ref_id,
            "kind": kind,
            "status": status,
            "problem": problem,
            "reason": f"{summary}",
            "summary": f"{summary}",
            "importance": importance,
            "at": entry.get("at"),
        }

    def _effective_outcome(self, outcome: Dict[str, Any], relevance: float) -> Dict[str, Any]:
        """Fold affect-derived relevance into an outcome's importance (copy).

        ``relevance`` is a 0..1 boost from the *candidate's* forward pull (her
        anticipation). It can only *raise* importance, is clamped to 1.0, and
        never lowers it, so a real reason is never suppressed by mood and an
        unimportant one cannot be inflated into an interruption.
        """
        if not relevance:
            return outcome
        folded = dict(outcome)
        try:
            base = float(folded.get("importance") or 0.0)
        except (TypeError, ValueError):
            base = 0.0
        folded["importance"] = round(max(0.0, min(1.0, base + float(relevance))), 4)
        return folded

    def _compose(self, outcome: Dict[str, Any]) -> Optional[str]:
        """Optionally phrase a permitted outcome in Astra's voice, or ``None``.

        Runs only after the policy has already decided to deliver, so it can
        never originate a reason or change the disposition. Any failure returns
        ``None`` and the deterministic summary is used instead.
        """
        if self.composer is None:
            return None
        try:
            text = self.composer(outcome)
        except Exception:
            return None
        text = str(text or "").strip()
        return text or None

    def _deliver(self, outcome: Dict[str, Any], now: datetime,
                 *, relevance: float = 0.0) -> bool:
        if self.notifier is None:
            return False
        outcome = self._effective_outcome(outcome, relevance)
        decision = self.notifications.decide(outcome, now=now)
        if decision["disposition"] != notify_mod.DELIVER:
            return False
        message = self._compose(outcome) or outcome.get("summary") or ""
        try:
            result = self.notifier(message, outcome.get("ref_id"), outcome)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        if result and result.get("ok"):
            self.notifications.note_delivered(outcome.get("ref_id"), now=now)
            return True
        return False

    def _maybe_digest(self, now: datetime, *, user_active: bool) -> Dict[str, Any]:
        if user_active or self.paused or self.notifier is None:
            return {"due": False, "reason": "suppressed"}
        if not self.notifications.digest_due(now=now):
            return {"due": False}
        entries = self.notifications.take_digest()
        if not entries:
            return {"due": False}
        text = self._render_digest(entries)
        try:
            result = self.notifier(text, f"digest:{now.isoformat()}", {"kind": "digest"})
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        if result and result.get("ok"):
            self.notifications.note_delivered(f"digest:{now.isoformat()}", now=now,
                                              is_digest=True)
            return {"due": True, "delivered": True, "count": len(entries)}
        # Failed: put the entries back so they are not lost.
        for entry in entries:
            self.notifications.queue_digest(entry)
        return {"due": True, "delivered": False, "count": len(entries)}

    @staticmethod
    def _render_digest(entries: List[Dict[str, Any]]) -> str:
        lines = ["A few things from the last stretch:"]
        for entry in entries[:8]:
            summary = str(entry.get("summary") or "").strip()
            if summary:
                lines.append(f"- {summary}")
        return "\n".join(lines)

    # -- diagnostics -----------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "planning_enabled": self.plan_enabled,
            "paused": self.paused,
            "wake_interval_seconds": self.wake_seconds,
            "last_wake_at": self.last_wake_at,
            "last_report": self.last_report,
        }


__all__ = ["AutonomyDriver"]
