"""The background-work scheduler: an application-level authority.

The scheduler decides *what work is legal and when it runs*. The LLM may propose
work (a task payload) or classify material, but it never decides eligibility.
This module is deterministic and pure with respect to policy: given the same
task queue, runtime state and process list it always makes the same decision.

Guarantees (from the spec):

* **Sleep-only.** A cycle runs only while the runtime state is SLEEP; the moment
  ACTIVE begins, the cycle stops between tasks. A background task can therefore
  never continue unrestricted work once a human is engaged.
* **Bounded.** Each cycle has a wall-clock budget and a per-task budget, and
  runs at most ``max_tasks_per_cycle`` tasks. A cycle may legitimately run zero
  tasks.
* **Policy-respecting.** It consults the shared :class:`reading.ProcessPolicy`
  for every cycle, so a running game or an explicitly busy workload stands down
  the machine-intensive work exactly as it does the reader.
* **No duplicate execution.** Deduplication lives in the task store; the
  scheduler only ever runs ``eligible`` tasks.
* **Accurate outcomes.** A task is marked by what its handler reports, and a
  failure is never recorded as success. A crash mid-task leaves it ``running``;
  recovery converts that to ``failed`` (see the worker).
* **No conversation.** A cycle never produces chat output, however many tasks it
  runs. Results go to the ledger and (optionally) prepared context.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import reading, tasks as task_mod

Handler = Callable[[task_mod.BackgroundTask], Dict[str, Any]]


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

# Task kinds that are machine-intensive (GPU/CPU/disk). These stand down on a
# game or a busy workload. Light kinds (bookkeeping) may still run when only a
# configured busy task is active, mirroring the reader's use of the policy.
INTENSIVE_KINDS = (
    task_mod.KIND_READING,
    task_mod.KIND_FILE_INSPECTION,
    task_mod.KIND_CALCULATION,
    task_mod.KIND_CONTEXT_PREPARATION,
    task_mod.KIND_WORK_REVIEW,
    task_mod.KIND_TEST_RUN,
)
LIGHT_KINDS = (
    task_mod.KIND_MEMORY_MAINTENANCE,
    task_mod.KIND_QUESTION_REVIEW,
    task_mod.KIND_APP_MAINTENANCE,
    task_mod.KIND_MEMORY_REVIEW,
    task_mod.KIND_EXPERIENCE_REVIEW,
    task_mod.KIND_SELF_REVIEW,
    task_mod.KIND_KNOWLEDGE_GAP,
    task_mod.KIND_SELF_IMPROVEMENT,
)

# Why a cycle did not run a task (mirrors the reader's explainable reasons).
REASON_OK = "ok"
REASON_ACTIVE = "active"
REASON_GAME = "game"
REASON_BUSY = "busy"
REASON_NO_WORK = "no_work"
REASON_BUDGET = "budget"
REASON_STOPPED = "stopped"
REASON_DISABLED = "disabled"
REASON_LOW_ENERGY = "low_energy"
REASON_MACHINE_COLD = "machine_cold"

# Below this "how much she can bring right now" (0..1), only light work runs.
LOW_ENERGY_THRESHOLD = 0.4

# A task completion hook: (task, final_status) -> None. The scheduled runner uses
# it to re-arm a recurring occurrence; the sleep scheduler passes none. A hook
# failure is swallowed so a bookkeeping error can never fail a task.
CompletionHook = Callable[[task_mod.BackgroundTask, str], None]


class Scheduler:
    """Deterministic, bounded background-work scheduler."""

    def __init__(
        self,
        *,
        task_store: task_mod.TaskStore,
        ledger: task_mod.TaskLedger,
        runtime_state: Any,
        process_policy: Optional[reading.ProcessPolicy] = None,
        handlers: Optional[Dict[str, Handler]] = None,
        max_tasks_per_cycle: int = 3,
        cycle_budget_seconds: float = 5.0,
        task_budget_seconds: float = 3.0,
        process_names_fn: Optional[Callable[[], Optional[List[str]]]] = None,
        cpu_load_fn: Optional[Callable[[], Optional[float]]] = None,
        load_ceiling: float = reading.DEFAULT_LOAD_CEILING,
        enabled: bool = True,
        require_sleep: bool = True,
        intensity_provider: Optional[Callable[[], float]] = None,
        machine_ready_provider: Optional[Callable[[], bool]] = None,
        completion_hook: Optional[CompletionHook] = None,
        clock: Callable[[], float] = time.monotonic,
        now_fn: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.task_store = task_store
        self.ledger = ledger
        self.runtime_state = runtime_state
        self.process_policy = process_policy or reading.ProcessPolicy()
        self.max_tasks_per_cycle = max(0, int(max_tasks_per_cycle))
        self.cycle_budget_seconds = max(0.0, float(cycle_budget_seconds))
        self.task_budget_seconds = max(0.0, float(task_budget_seconds))
        self.process_names_fn = process_names_fn
        self.cpu_load_fn = cpu_load_fn
        self.load_ceiling = float(load_ceiling)
        self.enabled = bool(enabled)
        # Sleep-only (the opportunistic work) is the default. The scheduled
        # runner builds a second scheduler with require_sleep=False, because
        # configured scheduled work must run even while a human is engaged.
        self.require_sleep = bool(require_sleep)
        # Optional: a pure 0..1 provider of how much Astra can bring to a task
        # right now (the behaviour modulation's ``effort``). When it is low, only
        # light work runs - so a sad or weary state genuinely does less, rather
        # than merely being described. ``None`` means "no modulation", and then
        # nothing changes.
        self.intensity_provider = intensity_provider
        # Optional: a pure "is the machine warm and calm enough for heavy work?"
        # provider. On a cold or busy machine Astra stays in efficiency mode, so
        # only light work runs until the machine has been up a while and is
        # measured calm. ``None`` means the gate is not applied.
        self.machine_ready_provider = machine_ready_provider
        self.completion_hook = completion_hook
        self._clock = clock
        self._now_fn = now_fn
        self._handlers: Dict[str, Handler] = dict(handlers or {})
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self.last_report: Dict[str, Any] = {}
        self.last_reason: str = REASON_NO_WORK

    # -- registration ----------------------------------------------------
    def register(self, kind: str, handler: Handler) -> None:
        with self._lock:
            self._handlers[kind] = handler

    def has_handler(self, kind: str) -> bool:
        return kind in self._handlers

    def _now(self) -> Any:
        if self._now_fn is not None:
            return self._now_fn()
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)

    def stop(self) -> None:
        self._stop.set()

    def clear_stop(self) -> None:
        self._stop.clear()

    def stopped(self) -> bool:
        return self._stop.is_set()

    # -- policy ----------------------------------------------------------
    def intensity(self) -> Optional[float]:
        """How much Astra can bring right now (0..1), or ``None`` when unknown.

        Pure and read-only: it asks the injected provider (the behaviour
        modulation's ``effort``). ``None`` means "no modulation is in play", and
        then her state never changes what may run.
        """
        if self.intensity_provider is None:
            return None
        try:
            return max(0.0, min(1.0, float(self.intensity_provider())))
        except (TypeError, ValueError):
            return None
        except Exception:
            return None

    def machine_ready(self) -> bool:
        """Whether the machine is warm and calm enough for heavy work.

        ``None`` provider means "not gated" (always ready). A provider error is
        treated as ready, so an unmeasurable machine never gets stuck in
        efficiency mode.
        """
        if self.machine_ready_provider is None:
            return True
        try:
            return bool(self.machine_ready_provider())
        except Exception:
            return True

    def machine_state(self) -> Dict[str, Any]:
        """The shared process/load answer for this cycle (never invents one)."""
        game = busy = None
        if self.process_names_fn is not None:
            try:
                game, busy = self.process_policy.classify(self.process_names_fn() or [])
            except Exception:
                game, busy = None, None
        load = None
        if self.cpu_load_fn is not None:
            try:
                load = self.cpu_load_fn()
            except Exception:
                load = None
        return {"game": game, "busy": busy, "cpu_load": load,
                "intensity": self.intensity(), "machine_ready": self.machine_ready()}

    def _kind_allowed(self, kind: str, machine: Dict[str, Any]) -> bool:
        if machine.get("game"):
            # A running game claims the GPU: only light bookkeeping continues.
            return kind in LIGHT_KINDS
        if machine.get("busy"):
            return kind in LIGHT_KINDS
        load = machine.get("cpu_load")
        if load is not None:
            try:
                if float(load) >= self.load_ceiling and kind in INTENSIVE_KINDS:
                    return False
            except (TypeError, ValueError):
                pass
        intensity = machine.get("intensity")
        if intensity is not None and float(intensity) < LOW_ENERGY_THRESHOLD:
            # She cannot bring much right now: only light work, so a flat or sad
            # state does less rather than being described as if it did.
            return kind in LIGHT_KINDS
        if not machine.get("machine_ready", True):
            # Cold or busy machine (e.g. just after boot): efficiency mode, so
            # only light work runs until the machine has warmed and settled.
            return kind in LIGHT_KINDS
        return True

    # -- one cycle -------------------------------------------------------
    def run_cycle(self) -> Dict[str, Any]:
        """Run at most a bounded number of eligible tasks while SLEEP.

        Returns a report. Never raises for an ordinary failure and never emits
        conversational text.
        """
        started = self._clock()
        report: Dict[str, Any] = {
            "ran": [], "skipped": [], "reason": REASON_OK,
            "tasks": 0, "stopped": False,
        }
        if not self.enabled:
            report["reason"] = REASON_DISABLED
            self.last_report = report
            self.last_reason = REASON_DISABLED
            return report
        if self._stop.is_set():
            report["reason"] = REASON_STOPPED
            report["stopped"] = True
            self.last_report = report
            self.last_reason = REASON_STOPPED
            return report
        if self.require_sleep and not self.runtime_state.is_sleeping():
            # ACTIVE: no opportunistic background work at all.
            report["reason"] = REASON_ACTIVE
            self.last_report = report
            self.last_reason = REASON_ACTIVE
            return report

        eligible = self.task_store.eligible(now=self._now())
        if not eligible:
            report["reason"] = REASON_NO_WORK
            self.last_report = report
            self.last_reason = REASON_NO_WORK
            return report

        machine = self.machine_state()
        ran = 0
        for task in eligible:
            if ran >= self.max_tasks_per_cycle:
                report["reason"] = REASON_BUDGET
                break
            if self._stop.is_set():
                report["stopped"] = True
                report["reason"] = REASON_STOPPED
                break
            # Re-check the runtime state between tasks: interaction wins (only
            # for the sleep-only opportunistic scheduler).
            if self.require_sleep and not self.runtime_state.is_sleeping():
                report["reason"] = REASON_ACTIVE
                break
            if self._clock() - started >= self.cycle_budget_seconds:
                report["reason"] = REASON_BUDGET
                break
            if not self._kind_allowed(task.kind, machine):
                if machine.get("game"):
                    reason = REASON_GAME
                elif machine.get("busy"):
                    reason = REASON_BUSY
                elif not machine.get("machine_ready", True):
                    reason = REASON_MACHINE_COLD
                else:
                    reason = REASON_LOW_ENERGY
                report["skipped"].append({
                    "id": task.id, "kind": task.kind, "reason": reason,
                })
                continue
            self._run_task(task)
            ran += 1
            # A handler may request a stop (e.g. shutdown) mid-cycle; honour it
            # immediately and report honestly.
            if self._stop.is_set():
                report["stopped"] = True
                report["reason"] = REASON_STOPPED
                break

        report["tasks"] = ran
        self.last_report = report
        self.last_reason = report["reason"]
        return report

    def _run_task(self, task: task_mod.BackgroundTask) -> None:
        handler = self._handlers.get(task.kind)
        # Persist RUNNING before the side effect: a crash leaves it visible, and
        # recovery turns a stuck RUNNING into FAILED rather than success.
        self.task_store.mark(task.id, task_mod.RUNNING)
        if handler is None:
            self.task_store.mark(
                task.id, task_mod.BLOCKED,
                blocked_reason="no handler registered",
                attempts=task.attempts + 1,
            )
            self.ledger.record(task_id=task.id, kind=task.kind,
                               status=task_mod.BLOCKED, detail="no handler")
            return
        deadline = self._clock() + self.task_budget_seconds
        try:
            outcome = handler(task) or {}
        except Exception as exc:  # a handler bug must not kill the cycle
            outcome = {"status": task_mod.FAILED, "detail": f"handler error: {exc}"}

        if self._clock() > deadline:
            # Over budget: the handler finished but slowly. Record it honestly.
            outcome.setdefault("detail", "")
            outcome["detail"] = (str(outcome.get("detail") or "") + " [over budget]").strip()

        status = str(outcome.get("status") or task_mod.COMPLETED)
        if status not in task_mod.TASK_STATUSES:
            status = task_mod.FAILED
        attempts = task.attempts + 1
        fields: Dict[str, Any] = {
            "attempts": attempts,
            "result": outcome.get("result") if isinstance(outcome.get("result"), dict) else None,
        }
        if status == task_mod.FAILED:
            fields["last_error"] = str(outcome.get("detail") or "failed")
            if attempts >= task.max_attempts:
                # Exhausted: stop retrying rather than looping forever.
                status = task_mod.BLOCKED
                fields["blocked_reason"] = "max attempts reached"
        if task.schedule:
            # Stamp the occurrence so the driver can compute the next due time.
            fields["last_run_at"] = _now_iso()
        self.task_store.mark(task.id, status, **fields)
        self.ledger.record(
            task_id=task.id, kind=task.kind, status=status,
            detail=str(outcome.get("detail") or ""),
            result=fields.get("result"),
        )
        if self.completion_hook is not None:
            # Re-arm a recurring schedule (or clean up). Never let bookkeeping
            # turn a real completion into a failure.
            try:
                self.completion_hook(task, status)
            except Exception:
                pass

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "last_reason": self.last_reason,
            "last_report": self.last_report,
            "task_counts": self.task_store.counts(),
            "machine": self.machine_state(),
        }
