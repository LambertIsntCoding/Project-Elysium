"""The application layer: the shared runtime both frontends go through.

Two objects live here, and together they are the "one Astra runtime" the spec
asks for:

* :class:`RuntimeController` - owns the persistent background life: the runtime
  state (ACTIVE/SLEEP), the task queue and ledger, the scheduler, the file
  organizer and its decisions, proactive policy and prepared context. Closing a
  UI does not stop it; its state is on disk. It owns the background threads and
  stops them cleanly on shutdown.
* :class:`ApplicationRouter` - the single application-level authority for a
  *message*, whatever transport it arrived on. Discord and the CLI call the same
  ``handle``; neither gets its own orchestrator, memory, or scheduling. Commands
  about runtime state, tasks and file decisions are handled here; everything
  else is delegated to the existing :class:`main.ChatSession` (which already owns
  Elysium routing and conversation).

No transport-specific logic here, and no duplicate conversational state: the
router reuses the session's own routing by temporarily capturing its output.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from . import (file_organizer, filesystem, prepared_context, proactive,
               reading, runtime_state, tasks as task_mod)
from .scheduler import Scheduler
from .tasks import (BackgroundTask, KIND_APP_MAINTENANCE,
                    KIND_CONTEXT_PREPARATION, KIND_FILE_INSPECTION,
                    KIND_MEMORY_MAINTENANCE, KIND_QUESTION_REVIEW,
                    KIND_READING)

DEFAULT_POLL_SECONDS = 5.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_ref(candidate: Dict[str, Any], idx: int) -> str:
    """A stable identity for a conversation candidate.

    Derived from its kind and the referenced question (or its text), so the same
    preserved candidate keeps the same id across turns and is not re-surfaced;
    the index is only a last-resort tiebreaker.
    """
    qid = candidate.get("question_id")
    if qid:
        return f"q:{qid}"
    kind = str(candidate.get("kind") or "cand")
    return f"{kind}:{idx}"


class RuntimeController:
    """Owns the persistent background runtime and its worker thread."""

    def __init__(
        self,
        *,
        session: Any,
        store: Any,
        data_dir: str = "./storage",
        config_dir: str = "./config",
        config: Optional[Dict[str, Any]] = None,
        reader: Any = None,
        library: Any = None,
        process_names_fn: Optional[Callable[[], Optional[List[str]]]] = None,
        cpu_load_fn: Optional[Callable[[], Optional[float]]] = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self.session = session
        self.store = store
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.config = config if isinstance(config, dict) else {}
        self.reader = reader
        self.library = library

        runtime_cfg = self.config.get("runtime") if isinstance(self.config.get("runtime"), dict) else {}
        sched_cfg = runtime_cfg.get("scheduler") if isinstance(runtime_cfg.get("scheduler"), dict) else {}
        fs_cfg = self.config.get("filesystem") if isinstance(self.config.get("filesystem"), dict) else {}

        # Persistent application state (each with its own serialization).
        self.state = runtime_state.RuntimeStateStore(data_dir=data_dir)
        self.task_store = task_mod.TaskStore(data_dir=data_dir)
        self.ledger = task_mod.TaskLedger(data_dir=data_dir)
        self.decisions = file_organizer.DecisionStore(data_dir=data_dir)
        self.prepared = prepared_context.PreparedContext(data_dir=data_dir)
        self.proactive = proactive.ProactivePolicy(
            data_dir=data_dir,
            interrupt_cooldown=float(self.config.get("proactive", {}).get(
                "interrupt_cooldown_seconds", proactive.DEFAULT_INTERRUPT_COOLDOWN)
                if isinstance(self.config.get("proactive"), dict) else proactive.DEFAULT_INTERRUPT_COOLDOWN),
            muted=bool(self.config.get("proactive", {}).get("muted", False)
                       if isinstance(self.config.get("proactive"), dict) else False),
        )

        # Filesystem policy. Astra's own paths are protected; hard safety
        # boundaries below cannot be bypassed by config accidentally.
        runtime_paths = file_organizer.default_runtime_paths(
            config_dir=config_dir, storage_dir=data_dir,
            extra=fs_cfg.get("runtime_paths") or (),
        )
        self.fs_policy = filesystem.FsPolicy(
            roots=fs_cfg.get("allowed_roots") or (),
            protected=fs_cfg.get("protected_paths") or (),
            runtime_paths=runtime_paths,
            allow_delete=bool(fs_cfg.get("allow_delete", False)),
            allow_overwrite=bool(fs_cfg.get("allow_overwrite", False)),
        )
        self.file_ops = filesystem.FileOps(self.fs_policy, data_dir=data_dir)
        self.organizer = file_organizer.FileOrganizer(
            self.fs_policy, self.file_ops, self.decisions,
            user_roots=self.fs_policy.roots,
            max_files_per_cycle=int(fs_cfg.get("max_files_per_cycle", 40)),
            dry_run=bool(fs_cfg.get("dry_run", False)),
        )
        # Crash recovery: no interrupted operation is ever reported as success.
        self.organizer.recover()

        # The scheduler, with one handler per task kind.
        self.scheduler = Scheduler(
            task_store=self.task_store, ledger=self.ledger,
            runtime_state=self.state,
            process_policy=getattr(reader, "process_policy", None) or reading.ProcessPolicy(),
            process_names_fn=process_names_fn,
            cpu_load_fn=cpu_load_fn,
            enabled=bool(sched_cfg.get("enabled", True)),
            max_tasks_per_cycle=int(sched_cfg.get("max_tasks_per_cycle", 3)),
            cycle_budget_seconds=float(sched_cfg.get("cycle_budget_seconds", 5.0)),
            task_budget_seconds=float(sched_cfg.get("task_budget_seconds", 3.0)),
        )
        self._register_handlers()

        self.poll_seconds = max(1.0, float(poll_seconds))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        # In-memory marker of the last human interaction (see ``touch``).
        self.last_interaction: Optional[str] = None

    # -- handlers --------------------------------------------------------
    def _register_handlers(self) -> None:
        self.scheduler.register(KIND_READING, self._handle_reading)
        self.scheduler.register(KIND_FILE_INSPECTION, self._handle_files)
        self.scheduler.register(KIND_MEMORY_MAINTENANCE, self._handle_memory)
        self.scheduler.register(KIND_QUESTION_REVIEW, self._handle_questions)
        self.scheduler.register(KIND_CONTEXT_PREPARATION, self._handle_prepare)
        self.scheduler.register(KIND_APP_MAINTENANCE, self._handle_maintenance)

    def _handle_reading(self, task: BackgroundTask) -> Dict[str, Any]:
        reader = self.reader
        if reader is None:
            return {"status": task_mod.COMPLETED, "detail": "no reader"}
        report = reader.run_once()
        return {"status": task_mod.COMPLETED,
                "detail": f"read={report.get('read')} reason={report.get('reason')}",
                "result": {"summary": report.get("reason"),
                           "count": report.get("chunks")}}

    def _handle_files(self, task: BackgroundTask) -> Dict[str, Any]:
        report = self.organizer.run_cycle(library=self.library)
        self.ledger.record(task_id=task.id, kind=task.kind,
                           status=task_mod.COMPLETED,
                           detail=f"scanned={report['scanned']} "
                                  f"acted={len(report['autonomous'])} "
                                  f"asked={len(report['asked'])}")
        return {"status": task_mod.COMPLETED,
                "detail": f"scanned={report['scanned']}",
                "result": {"count": report["scanned"]}}

    def _handle_memory(self, task: BackgroundTask) -> Dict[str, Any]:
        # Deterministic maintenance only: decay, dormancy, archival. It never
        # supersedes an explicit memory and never creates durable claims.
        applied = {}
        try:
            applied = self.store.apply_decay()
        except Exception as exc:
            return {"status": task_mod.FAILED, "detail": f"decay failed: {exc}"}
        try:
            self.store.maybe_maintain(force=True)
        except Exception:
            pass
        return {"status": task_mod.COMPLETED, "detail": f"decayed={sum(applied.values())}"}

    def _handle_questions(self, task: BackgroundTask) -> Dict[str, Any]:
        # Review open questions and refresh candidate references. Never answers
        # a question, never promotes a hypothesis to a fact.
        entries = prepared_context.build(self.store, library=self.library)
        self.prepared.replace(entries)
        return {"status": task_mod.COMPLETED, "detail": f"{len(entries)} refs"}

    def _handle_prepare(self, task: BackgroundTask) -> Dict[str, Any]:
        entries = prepared_context.build(self.store, library=self.library)
        self.prepared.replace(entries)
        return {"status": task_mod.COMPLETED, "detail": f"{len(entries)} refs"}

    def _handle_maintenance(self, task: BackgroundTask) -> Dict[str, Any]:
        removed = self.task_store.clear_terminal()
        return {"status": task_mod.COMPLETED, "detail": f"cleared={removed}"}

    # -- seeding recurring work -----------------------------------------
    def seed(self) -> None:
        """Enqueue the routine background work, deduped by the store.

        Deduplication means re-seeding never doubles the queue and only writes
        when a kind is genuinely missing, so this is safe and cheap to call
        every cycle across restarts.
        """
        seeds = [
            (KIND_FILE_INSPECTION, "allowed-roots", 20),
            (KIND_MEMORY_MAINTENANCE, "routine-decay", 60),
            (KIND_CONTEXT_PREPARATION, "nightly", 50),
            (KIND_READING, "next-work", 30),
            (KIND_APP_MAINTENANCE, "queue-sweep", 45),
        ]
        with self._lock:
            for kind, key, priority in seeds:
                self.task_store.enqueue(BackgroundTask(
                    kind=kind, origin=task_mod.ORIGIN_SCHEDULER,
                    priority=priority, dedupe_key=key))

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        """Start the background worker thread (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="astra-runtime",
                                            daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        self.seed()
        while not self._stop.is_set():
            try:
                if self.state.is_sleeping():
                    # Re-seed so a drained queue refills while asleep. Cheap: the
                    # task store's dedupe means only genuinely missing kinds are
                    # added, so this does not churn the file every cycle.
                    self.seed()
                    self.scheduler.run_cycle()
            except Exception:
                pass  # a worker must never take down the runtime
            self._stop.wait(self.poll_seconds)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the worker cleanly and flush. Safe to call repeatedly."""
        self._stop.set()
        self.scheduler.stop()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    # -- state control ---------------------------------------------------
    def sleep(self, *, reason: str = runtime_state.REASON_HUMAN_COMMAND) -> Dict[str, Any]:
        return self.state.set_state(runtime_state.SLEEP, reason=reason)

    def wake(self, *, reason: str = runtime_state.REASON_HUMAN_WAKE) -> Dict[str, Any]:
        # A human waking Astra is a human-caused wake event.
        return self.state.set_state(runtime_state.ACTIVE, reason=reason)

    def sleeping(self) -> bool:
        return self.state.is_sleeping()

    def touch(self) -> None:
        """Mark that a human is present and interacting, without a state flip.

        Used on the live turn path for any human input, so the runtime knows a
        person is here even when the input is a slash command (which the router
        does not route). In-memory only: it never sleeps, wakes, or writes a
        file, so a busy session does not churn runtime state.
        """
        self.last_interaction = _now()

    # -- status rendering ------------------------------------------------
    def status_report(self) -> str:
        lines = [self.state.render()]
        counts = self.task_store.counts()
        open_decisions = len([d for d in self.decisions.open() if d.status == file_organizer.D_PENDING])
        lines.append(f"Background tasks: " + ", ".join(
            f"{k}={v}" for k, v in sorted(counts.items()) if v) or "none")
        lines.append(f"File decisions waiting: {open_decisions}")
        if self.proactive.muted:
            lines.append("Proactive messages: muted")
        report = self.scheduler.last_report or {}
        if report:
            lines.append(f"Last cycle: {self.scheduler.last_reason} "
                         f"({report.get('tasks', 0)} tasks)")
        return "\n".join(lines)

    def sleep_log(self, limit: int = 12) -> str:
        return self.ledger.render(limit)

    def pending_decisions_text(self, limit: int = 8) -> str:
        return self.decisions.render_open(limit)


class ApplicationRouter:
    """The single authority every frontend calls to handle a message."""

    def __init__(self, session: Any, controller: RuntimeController, *,
                 voice: Optional[Any] = None) -> None:
        self.session = session
        self.controller = controller
        self.voice = voice
        # Astra is one conversation. Serialize message handling across
        # transports (CLI and Discord threads) so the shared ChatSession state
        # (history, working memory, output capture) is never mutated
        # concurrently. This is what makes "the same logical interaction is
        # handled the same way" true regardless of transport.
        self._lock = threading.RLock()

    # -- helpers ---------------------------------------------------------
    def _capture(self, fn: Callable[[], Any]) -> str:
        """Run a session method that emits, and return what it emitted.

        This reuses the session's existing routing (Elysium, slash commands,
        conversation) without duplicating any of it for Discord. Callers hold
        ``self._lock``, so swapping ``output_fn`` cannot race another transport.
        """
        buffer: List[str] = []
        original = getattr(self.session, "output_fn", print)
        self.session.output_fn = buffer.append
        try:
            result = fn()
        finally:
            self.session.output_fn = original
        if isinstance(result, str) and result:
            buffer.append(result)
        return "\n".join(buffer)

    def _is_sleep_command(self, command: str) -> bool:
        return command in ("/sleep", "/sleep-mode")

    # -- the one entry point --------------------------------------------
    def handle(self, text: str, *, source: str = "cli") -> Dict[str, Any]:
        """Handle one message from any transport. Returns a structured result."""
        with self._lock:
            return self._handle_locked(text, source=source)

    def _handle_locked(self, text: str, *, source: str) -> Dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            return {"text": "", "handled": True, "source": source}

        # Any human message is a human-caused wake event.
        if not text.startswith("/") and self.controller.sleeping():
            self.controller.wake(reason=runtime_state.REASON_HUMAN_MESSAGE)

        if text.startswith("/"):
            return self._handle_command(text, source=source)

        # Conversation / Elysium: the session's own authority handles it.
        response = self.session.handle(text)
        return {"text": response, "handled": True, "source": source,
                "conversation": True}

    def _handle_command(self, text: str, *, source: str) -> Dict[str, Any]:
        parts = text.split()
        command = parts[0].lower()
        controller = self.controller

        if self._is_sleep_command(command):
            controller.sleep()
            self.session.note_activity()
            return {"text": "Going to sleep. I'll keep an eye on things quietly.",
                    "handled": True, "source": source}

        if command == "/wake":
            controller.wake()
            return {"text": "Awake.", "handled": True, "source": source}

        if command == "/status":
            return {"text": controller.status_report(), "handled": True, "source": source}

        if command == "/tasks":
            return {"text": self._tasks_text(), "handled": True, "source": source}

        if command in ("/decisions", "/files"):
            return {"text": controller.pending_decisions_text(),
                    "handled": True, "source": source}

        if command == "/sleep-log":
            return {"text": controller.sleep_log(), "handled": True, "source": source}

        if command in ("/approve", "/reject", "/defer"):
            if len(parts) < 2:
                return {"text": f"Usage: {command} <decision_id>",
                        "handled": True, "source": source}
            return {"text": self._resolve_decision(command, parts[1]),
                    "handled": True, "source": source}

        if command == "/proactive":
            return {"text": self._proactive_command(parts[1:]), "handled": True,
                    "source": source}

        if command == "/scheduler":
            diag = controller.scheduler.diagnostics()
            return {"text": f"Scheduler: enabled={diag['enabled']} "
                            f"last={diag['last_reason']} counts={diag['task_counts']}",
                    "handled": True, "source": source}

        # Anything else: the session's existing slash handling (captured so it
        # works identically from Discord, which cannot "print").
        return {"text": self._capture(lambda: self.session._handle_slash(text)),
                "handled": True, "source": source}

    def _tasks_text(self) -> str:
        counts = self.controller.task_store.counts()
        lines = ["Background tasks:"]
        lines.append("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()) if v)
                     if any(counts.values()) else "  (none)")
        for task in self.controller.task_store.all()[:10]:
            if task.status in (task_mod.PENDING, task_mod.RUNNING,
                               task_mod.AWAITING_HUMAN, task_mod.BLOCKED):
                lines.append(f"  [{task.id}] {task.kind} :: {task.status}")
        return "\n".join(lines)

    def _resolve_decision(self, command: str, decision_id: str) -> str:
        if command == "/defer":
            result = self.controller.organizer.defer(decision_id)
        else:
            result = self.controller.organizer.resolve(
                decision_id, approve=(command == "/approve"))
        if not result.get("ok") and result.get("status") != file_organizer.D_REJECTED:
            return f"Could not resolve {decision_id}: {result.get('error', result.get('outcome'))}"
        status = result.get("status")
        if status == file_organizer.D_REJECTED:
            return f"Rejected {decision_id}."
        if status == file_organizer.D_DEFERRED:
            return f"Deferred {decision_id}."
        return f"{decision_id}: {result.get('outcome', status)} ({result.get('detail', '')})"

    def _proactive_command(self, args: List[str]) -> str:
        if not args:
            return (f"Proactive: muted={self.controller.proactive.muted}. "
                    "Use /proactive mute or /proactive unmute.")
        if args[0].lower() == "mute":
            self.controller.proactive.set_muted(True)
            return "Proactive messages muted."
        if args[0].lower() == "unmute":
            self.controller.proactive.set_muted(False)
            return "Proactive messages unmuted."
        return "Usage: /proactive [mute|unmute]"

    # -- proactive surfacing --------------------------------------------
    def proactive_candidates(self, *, user_active: bool = False,
                             now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Decide which preserved candidates, if any, may be surfaced now.

        This is the explicit multi-gate path: candidate -> strength -> retain ->
        surface -> interrupt -> permitted. It never manufactures a candidate
        from elapsed time; with no concrete candidate it returns nothing. Each
        candidate carries a *stable* id (derived from its kind and text) so the
        same candidate is not offered again on a later turn.
        """
        candidates = self.session.orchestrator.conversation_candidates(limit=6)
        if not candidates:
            return []
        machine = self.controller.scheduler.machine_state()
        out: List[Dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            cand = dict(cand)
            cand["ref_id"] = _candidate_ref(cand, idx)
            strength = proactive.candidate_strength(cand)
            decision = self.controller.proactive.evaluate(
                cand, strength=strength, user_active=user_active,
                game=machine.get("game"), busy=machine.get("busy"), now=now)
            if decision["surface"]:
                out.append({"candidate": cand, "decision": decision})
        return out
