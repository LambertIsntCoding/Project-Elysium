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

# Task kinds that are machine-intensive (GPU/CPU/disk). These stand down on a
# game or a busy workload. Light kinds (bookkeeping) may still run when only a
# configured busy task is active, mirroring the reader's use of the policy.
INTENSIVE_KINDS = (
    task_mod.KIND_READING,
    task_mod.KIND_FILE_INSPECTION,
    task_mod.KIND_CALCULATION,
    task_mod.KIND_CONTEXT_PREPARATION,
    task_mod.KIND_WORK_REVIEW,
)
LIGHT_KINDS = (
    task_mod.KIND_MEMORY_MAINTENANCE,
    task_mod.KIND_QUESTION_REVIEW,
    task_mod.KIND_APP_MAINTENANCE,
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
        clock: Callable[[], float] = time.monotonic,
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
        self._clock = clock
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

    def stop(self) -> None:
        self._stop.set()

    def clear_stop(self) -> None:
        self._stop.clear()

    def stopped(self) -> bool:
        return self._stop.is_set()

    # -- policy ----------------------------------------------------------
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
        return {"game": game, "busy": busy, "cpu_load": load}

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
        if not self.runtime_state.is_sleeping():
            # ACTIVE: no background work at all.
            report["reason"] = REASON_ACTIVE
            self.last_report = report
            self.last_reason = REASON_ACTIVE
            return report

        eligible = self.task_store.eligible()
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
            # Re-check the runtime state between tasks: interaction wins.
            if not self.runtime_state.is_sleeping():
                report["reason"] = REASON_ACTIVE
                break
            if self._clock() - started >= self.cycle_budget_seconds:
                report["reason"] = REASON_BUDGET
                break
            if not self._kind_allowed(task.kind, machine):
                report["skipped"].append({
                    "id": task.id, "kind": task.kind,
                    "reason": REASON_GAME if machine.get("game") else REASON_BUSY,
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
        self.task_store.mark(task.id, status, **fields)
        self.ledger.record(
            task_id=task.id, kind=task.kind, status=status,
            detail=str(outcome.get("detail") or ""),
            result=fields.get("result"),
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "last_reason": self.last_reason,
            "last_report": self.last_report,
            "task_counts": self.task_store.counts(),
            "machine": self.machine_state(),
        }
