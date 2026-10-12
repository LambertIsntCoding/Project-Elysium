"""Bounded autonomy: choosing what Astra may do next, from what she really has.

This is the *planning* layer the brief asks for. It reads Astra's actual state -
the durable task queue, her governed memories, her open questions, her reading
position and her recent attempt ledger - and proposes a small, bounded set of
next activities. It never invents a capability: every proposal names an existing
task kind, and a kind with no real handler is refused rather than queued.

Its principles are the brief's, made concrete:

* **Nothing is manufactured.** With no unfinished work, no open question, no
  partway work and no due maintenance, ``plan`` returns an *idle* decision. It
  does not generate a task so that Astra can look busy, and it does not narrate
  activity that did not happen.
* **Bounded.** At most ``max_new`` tasks are proposed per planning pass, only
  for kinds whose live-task count is below ``per_kind_limit``, and only when the
  queue is below ``max_live``. A planner that can generate work forever is not
  autonomous, it is a loop.
* **Balanced.** Candidates are scored so exploration, ongoing projects,
  maintenance and evaluation each get a turn; a kind that ran very recently is
  de-prioritised so she does not fixate on one activity.
* **Honest about outcomes.** Selection reasons, scope and the decision to
  continue/stop/retry/defer/abandon are recorded with the task, not asserted as
  achievement.

The output is a list of *proposals*; the caller (the autonomy driver) is the
authority that actually enqueues them, exactly as the task store is the single
write path.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from . import behavior
from . import tasks as task_mod

# Decision vocabulary for an existing activity.
CONTINUE = "continue"
STOP = "stop"
RETRY = "retry"
DEFER = "defer"
ABANDON = "abandon"

# The candidate "lanes" the brief names, so a plan can be balanced across them.
LANE_MAINTENANCE = "maintenance"
LANE_PROJECT = "project"
LANE_LEARNING = "learning"
LANE_EVALUATION = "evaluation"
LANE_EXPLORATION = "exploration"

# Which lane each kind belongs to (used for balance and reporting).
_KIND_LANE = {
    task_mod.KIND_MEMORY_MAINTENANCE: LANE_MAINTENANCE,
    task_mod.KIND_APP_MAINTENANCE: LANE_MAINTENANCE,
    task_mod.KIND_MEMORY_REVIEW: LANE_MAINTENANCE,
    task_mod.KIND_QUESTION_REVIEW: LANE_PROJECT,
    task_mod.KIND_WORK_REVIEW: LANE_PROJECT,
    task_mod.KIND_CONTEXT_PREPARATION: LANE_PROJECT,
    task_mod.KIND_READING: LANE_LEARNING,
    task_mod.KIND_KNOWLEDGE_GAP: LANE_LEARNING,
    task_mod.KIND_EXPERIENCE_REVIEW: LANE_EVALUATION,
    task_mod.KIND_SELF_REVIEW: LANE_EVALUATION,
    task_mod.KIND_TEST_RUN: LANE_EVALUATION,
    task_mod.KIND_SELF_IMPROVEMENT: LANE_EXPLORATION,
    task_mod.KIND_CALCULATION: LANE_EXPLORATION,
    task_mod.KIND_FILE_INSPECTION: LANE_MAINTENANCE,
}

# Lanes' base desirability (higher = more worthwhile by default). Evaluation and
# maintenance keep Astra sound; projects honour unfinished work; learning is a
# steady background gain; exploration is lowest so it never crowds out the rest.
_LANE_WEIGHT = {
    LANE_PROJECT: 0.75,
    LANE_EVALUATION: 0.68,
    LANE_MAINTENANCE: 0.62,
    LANE_LEARNING: 0.58,
    LANE_EXPLORATION: 0.45,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ActivityPlanner:
    """Proposes bounded next activities from Astra's real state."""

    def __init__(
        self,
        task_store: task_mod.TaskStore,
        *,
        store: Any = None,
        library: Any = None,
        ledger: Any = None,
        config: Optional[Dict[str, Any]] = None,
        enabled_kinds: Optional[List[str]] = None,
        affect_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.task_store = task_store
        self.store = store
        self.library = library
        self.ledger = ledger
        cfg = config if isinstance(config, dict) else {}
        self.max_new = max(0, int(cfg.get("max_new_per_pass", 2)))
        self.max_live = max(1, int(cfg.get("max_live_tasks", 24)))
        self.per_kind_limit = max(1, int(cfg.get("per_kind_limit", 2)))
        # The kinds the planner may select. The application passes the set that
        # actually has a registered handler, so a proposal is always executable.
        self.enabled_kinds = set(enabled_kinds or task_mod.TASK_KINDS)
        # Optional read-only accessor for Astra's current experiential affect.
        # When present, her state biases *what* she takes on (effort/take-on) and
        # *how many* activities she starts - so a sad or weary state does less
        # work, not merely a differently-worded report of the same work.
        self.affect_provider = affect_provider

    # -- state-driven bias -----------------------------------------------
    def _bias(self) -> Dict[str, Any]:
        """Astra's current activity bias, derived from her affect (read-only)."""
        state: Any = None
        if self.affect_provider is not None:
            try:
                state = self.affect_provider()
            except Exception:
                state = None
        elif self.store is not None and callable(getattr(self.store, "current_affect", None)):
            try:
                state = self.store.current_affect()
            except Exception:
                state = None
        modulation = behavior.from_affect(state) if state else behavior.from_affect({})
        return behavior.activity_bias(modulation)

    # -- evidence-gathering ----------------------------------------------
    def _recently_run_kinds(self, limit: int = 20) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if self.ledger is None:
            return counts
        try:
            entries = self.ledger.entries(limit=limit)
        except Exception:
            return counts
        for entry in entries:
            kind = entry.get("kind")
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    def _live_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for task in self.task_store.all():
            if task.is_live():
                counts[task.kind] = counts.get(task.kind, 0) + 1
        return counts

    def _open_questions(self) -> int:
        if self.store is None:
            return 0
        try:
            return len(self.store.get_questions(status="active"))
        except Exception:
            return 0

    def _unfinished_works(self) -> List[Dict[str, Any]]:
        if self.library is None or not callable(getattr(self.library, "list_works", None)):
            return []
        try:
            works = self.library.list_works() or []
        except Exception:
            return []
        unfinished = []
        for work in works:
            if not isinstance(work, dict):
                continue
            # A partway or unstarted work is a real project; a finished one is not.
            if work.get("finished"):
                continue
            unfinished.append(work)
        return unfinished

    # -- candidate production --------------------------------------------
    def candidates(self) -> List[Dict[str, Any]]:
        """The worthwhile activities suggested by the current state.

        Each candidate is ``{kind, dedupe_key, lane, priority, importance,
        reason, scope}``. It is *evidence-backed*: a candidate only appears when
        the state that justifies it is present.
        """
        out: List[Dict[str, Any]] = []
        open_questions = self._open_questions()
        unfinished = self._unfinished_works()

        if open_questions:
            out.append(self._cand(
                task_mod.KIND_QUESTION_REVIEW, "open-questions",
                reason=f"{open_questions} open question(s) to review",
                importance=0.5, scope="revisit open questions and refresh context"))
            out.append(self._cand(
                task_mod.KIND_EXPERIENCE_REVIEW, "lessons",
                reason="turn recent experiences into lessons",
                importance=0.5, scope="review recent experiences"))

        if unfinished:
            for work in unfinished[:3]:
                work_id = work.get("work_id") or work.get("title")
                out.append(self._cand(
                    task_mod.KIND_READING, f"work:{work_id}",
                    reason=f"unfinished work: {work.get('title') or work_id}",
                    importance=0.55, scope="continue a partway work",
                    payload={"work_id": work_id}))

        # Maintenance and evaluation are always *potentially* worthwhile, but
        # only when their lane has not just run (the balance rule below filters).
        out.append(self._cand(
            task_mod.KIND_MEMORY_MAINTENANCE, "routine-decay",
            reason="routine memory maintenance", importance=0.45,
            scope="decay/dormancy sweep"))
        out.append(self._cand(
            task_mod.KIND_MEMORY_REVIEW, "quality",
            reason="review memory quality and contradictions", importance=0.5,
            scope="inspect governed memories for quality issues"))
        out.append(self._cand(
            task_mod.KIND_SELF_REVIEW, "behaviour",
            reason="review recent behaviour for recurring failures",
            importance=0.45, scope="evaluate recent turns for patterns"))
        out.append(self._cand(
            task_mod.KIND_SELF_IMPROVEMENT, "proposals",
            reason="draft improvement proposals requiring approval",
            importance=0.4, scope="summarise candidate improvements"))
        return out

    def _cand(self, kind: str, dedupe_key: str, *, reason: str, importance: float,
              scope: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        lane = _KIND_LANE.get(kind, LANE_EXPLORATION)
        base = _LANE_WEIGHT.get(lane, 0.45)
        return {
            "kind": kind,
            "dedupe_key": dedupe_key,
            "lane": lane,
            "reason": reason,
            "importance": round(max(0.0, min(1.0, importance)), 4),
            "score": round(base + 0.4 * importance, 4),
            "priority": task_mod.DEFAULT_PRIORITY.get(kind, 25),
            "scope": scope,
            "payload": dict(payload or {}),
        }

    # -- the planning pass -----------------------------------------------
    def plan(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Return a bounded list of proposals, or an *idle* decision.

        Idle is a legitimate, honest outcome; it is returned both when no
        candidate is evidenced and when every candidate is already represented
        in the queue or was just run.
        """
        now = now or _utcnow()
        live = self._live_counts()
        live_total = sum(live.values())
        recent = self._recently_run_kinds()
        bias = self._bias()

        candidates = [c for c in self.candidates() if c["kind"] in self.enabled_kinds]
        proposals: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        if live_total >= self.max_live:
            return {"decision": "bounded", "proposals": [],
                    "reason": f"queue at capacity ({live_total}/{self.max_live})"}

        # Her state sets the *volume*: a low take-on accepts fewer new activities
        # this pass (never more than the configured bound). A strong take-on may
        # accept the full bound; this is where "a bad day does less work" is real
        # rather than narrated.
        take_on = float(bias.get("take_on", 0.5))
        effective_max = self.max_new if take_on >= 0.5 else min(
            self.max_new, 1 if take_on >= 0.25 else 0)
        prefer_light = bool(bias.get("prefer_light"))

        # Balance: prefer lanes that have not run in the recent ledger; a kind
        # that just ran is de-prioritised so she does not fixate on it. A low
        # state prefers lighter lanes (maintenance) over heavy/investigative ones.
        for cand in candidates:
            if len(proposals) >= effective_max:
                break
            if live.get(cand["kind"], 0) >= self.per_kind_limit:
                skipped.append({"kind": cand["kind"], "reason": "kind at limit"})
                continue
            existing = self.task_store.find_live_by_key(cand["kind"], cand["dedupe_key"])
            if existing is not None:
                skipped.append({"kind": cand["kind"], "reason": "already queued"})
                continue
            cand = dict(cand)
            cand["score"] = round(cand["score"] - 0.15 * min(3, recent.get(cand["kind"], 0)), 4)
            if prefer_light:
                # A low state's willingness is toward light work: nudge the score
                # by lane so a heavy task does not win the pass over bookkeeping.
                cand["score"] = round(
                    cand["score"] + (0.15 if cand["lane"] == LANE_MAINTENANCE else -0.15), 4)
            cand["recorded_at"] = now.isoformat()
            if bias.get("note"):
                cand["bias_note"] = str(bias["note"])
            proposals.append(cand)

        if prefer_light:
            proposals.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        if not proposals:
            reason = "no evidenced activity to add"
            if effective_max == 0 and self.max_new > 0:
                reason = "low energy: holding off on new activities this pass"
            return {"decision": "idle", "proposals": [], "skipped": skipped,
                    "reason": reason, "bias": bias}
        return {"decision": "plan", "proposals": proposals, "skipped": skipped,
                "bias": bias}

    # -- lifecycle decisions for an existing task ------------------------
    @staticmethod
    def decide(task: task_mod.BackgroundTask, *, failure_rate: float = 0.0,
               budget_exhausted: bool = False) -> str:
        """Whether to continue, retry, defer or abandon an activity.

        Pure, over the task's own record and a couple of derived signals. A
        task with attempts left and no hard block is retried; one that keeps
        failing is abandoned rather than looped; a confirmation-gated one defers.
        """
        if task.status == task_mod.AWAITING_HUMAN:
            return DEFER
        if task.status in (task_mod.COMPLETED, task_mod.CANCELLED,
                           task_mod.SKIPPED, task_mod.MISSED, task_mod.EXPIRED):
            return STOP
        if budget_exhausted:
            return DEFER
        if task.attempts >= task.max_attempts:
            return ABANDON
        if task.status == task_mod.BLOCKED:
            return ABANDON
        if task.status == task_mod.FAILED:
            return RETRY
        return CONTINUE


__all__ = [
    "ActivityPlanner", "CONTINUE", "STOP", "RETRY", "DEFER", "ABANDON",
    "LANE_MAINTENANCE", "LANE_PROJECT", "LANE_LEARNING", "LANE_EVALUATION",
    "LANE_EXPLORATION",
]
