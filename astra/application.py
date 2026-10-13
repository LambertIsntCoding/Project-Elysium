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

from . import (activity, autonomy, engagement as engagement_mod,
               file_organizer, filesystem, intentions, machine as machine_mod,
               model_use, notifications as notify_mod, prepared_context, proactive,
               proactive_compose, proposals as proposal_mod,
               reading, resources as resources_mod, runtime_state,
               schedule_explain, schedules as sched,
               tasks as task_mod)
from . import schedule_runtime
from .scheduler import Scheduler
from .tasks import (BackgroundTask, KIND_APP_MAINTENANCE,
                    KIND_CONTEXT_PREPARATION, KIND_EXPERIENCE_REVIEW,
                    KIND_FILE_INSPECTION, KIND_KNOWLEDGE_GAP,
                    KIND_MEMORY_MAINTENANCE, KIND_MEMORY_REVIEW,
                    KIND_QUESTION_REVIEW, KIND_READING, KIND_SELF_IMPROVEMENT,
                    KIND_SELF_REVIEW, KIND_TEST_RUN, KIND_WORK_REVIEW)

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
        machine: Optional[Any] = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        think_only: bool = False,
    ) -> None:
        self.session = session
        self.store = store
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.config = config if isinstance(config, dict) else {}
        self.reader = reader
        self.library = library
        # Headless "think-only" mode: the runtime runs fully but never presents
        # an interactive prompt, and asks an idle policy to sleep so the
        # sleep-only opportunistic work still gets its turn.
        self.think_only = bool(think_only)

        runtime_cfg = self.config.get("runtime") if isinstance(self.config.get("runtime"), dict) else {}
        sched_cfg = runtime_cfg.get("scheduler") if isinstance(runtime_cfg.get("scheduler"), dict) else {}
        power_cfg = runtime_cfg.get("power") if isinstance(runtime_cfg.get("power"), dict) else {}
        fs_cfg = self.config.get("filesystem") if isinstance(self.config.get("filesystem"), dict) else {}
        auto_cfg = self.config.get("autonomy") if isinstance(self.config.get("autonomy"), dict) else {}
        notif_cfg = self.config.get("notifications") if isinstance(self.config.get("notifications"), dict) else {}

        # Real machine awareness: measured CPU/memory/uptime/idle, so startup can
        # be resource-efficient and a machine suspend can be recognised. It is a
        # pure sampler plus pure decisions; ``None`` (unmeasurable) changes
        # nothing. ``cpu_load_fn`` (if given) still feeds the scheduler/reader;
        # this object is the richer, numeric source for power decisions.
        self.machine = machine if machine is not None else machine_mod.build_default()
        # If no explicit CPU provider was supplied, use the machine's own
        # numeric reading so the scheduler stands heavy work down on real load.
        # (The reader is built earlier with its own optional provider; passing
        # the machine here keeps the scheduler's answer measured, not guessed.)
        if cpu_load_fn is None and hasattr(self.machine, "cpu_reading"):
            cpu_load_fn = self.machine.cpu_reading
        self.power_cfg = {
            "startup_efficiency": bool(power_cfg.get("startup_efficiency", True)),
            "sleep_with_computer": bool(power_cfg.get("sleep_with_computer", True)),
            "wake_on_resume": bool(power_cfg.get("wake_on_resume", False)),
        }
        self._last_suspend_seconds: Optional[float] = None
        # Durable resource history: the idle/active baselines and the learned
        # per-kind cost that ground the scheduler's "can this actually be done?"
        # estimate. It is application state (never a memory) and never reaches
        # the prompt.
        self.resource_history = resources_mod.ResourceHistory(data_dir=data_dir)

        # Persistent application state (each with its own serialization).
        self.state = runtime_state.RuntimeStateStore(data_dir=data_dir)
        self.task_store = task_mod.TaskStore(data_dir=data_dir)
        self.ledger = task_mod.TaskLedger(data_dir=data_dir)
        self.decisions = file_organizer.DecisionStore(data_dir=data_dir)
        # Durable improvement proposals: changes to identity, boundaries,
        # credentials, permissions or source code are recorded here and require
        # Roum's acknowledgement/approval; nothing here applies a change.
        self.proposals = proposal_mod.ProposalStore(data_dir=data_dir)
        self.prepared = prepared_context.PreparedContext(data_dir=data_dir)
        self.proactive = proactive.ProactivePolicy(
            data_dir=data_dir,
            interrupt_cooldown=float(self.config.get("proactive", {}).get(
                "interrupt_cooldown_seconds", proactive.DEFAULT_INTERRUPT_COOLDOWN)
                if isinstance(self.config.get("proactive"), dict) else proactive.DEFAULT_INTERRUPT_COOLDOWN),
            muted=bool(self.config.get("proactive", {}).get("muted", False)
                       if isinstance(self.config.get("proactive"), dict) else False),
        )
        # The notification policy: configurable limits, quiet hours, digest and
        # pause. It is a separate layer from the proactive candidate policy and
        # from the transport.
        self.notifications = notify_mod.NotificationPolicy(
            data_dir=data_dir, config=notif_cfg)

        # Adaptation to repeated non-response: a *separate*, content-free
        # bookkeeping layer. It never replaces the policy; it only raises the
        # bar when proactive messages have gone unanswered, and recovers on its
        # own with time or engagement. The threshold provider is injected into
        # the policy only after construction so the dependency stays one-way.
        eng_cfg = self.config.get("engagement") if isinstance(self.config.get("engagement"), dict) else {}
        self.engagement = engagement_mod.EngagementTracker(
            data_dir=data_dir, config=eng_cfg)
        self.notifications.engagement_provider = self._engagement_boost
        self.notifications.note_sent_hook = self._note_proactive_sent

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

        # The scheduler, with one handler per task kind. This is the sleep-only
        # opportunistic scheduler: it may only run while SLEEP.
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
            intensity_provider=self._intensity,
            machine_ready_provider=self._machine_ready_for_work,
            machine_sample_provider=self._machine_sample_now,
            resource_history=self.resource_history,
        )
        self._register_handlers()

        # Time-aware schedules: a driver that keeps due instants current, and a
        # second scheduler that runs *scheduled* work regardless of SLEEP/ACTIVE
        # (configured activities must run even while a human is engaged). It runs
        # the same registered handlers and re-arms recurring occurrences.
        self.schedule_driver = schedule_runtime.ScheduleDriver(
            self.task_store, ledger=self.ledger)
        self.scheduled_scheduler = Scheduler(
            task_store=self.task_store, ledger=self.ledger,
            runtime_state=self.state,
            process_policy=getattr(reader, "process_policy", None) or reading.ProcessPolicy(),
            process_names_fn=process_names_fn,
            cpu_load_fn=cpu_load_fn,
            enabled=bool(auto_cfg.get("enabled", True)),
            require_sleep=False,
            max_tasks_per_cycle=int(auto_cfg.get("max_scheduled_per_wake", 2)),
            cycle_budget_seconds=float(auto_cfg.get("cycle_budget_seconds", 30.0)),
            task_budget_seconds=float(auto_cfg.get("task_budget_seconds", 20.0)),
            completion_hook=self._after_scheduled_run,
            machine_ready_provider=self._machine_ready_for_work,
            machine_sample_provider=self._machine_sample_now,
            resource_history=self.resource_history,
        )
        for kind, handler in self.scheduler._handlers.items():
            self.scheduled_scheduler.register(kind, handler)

        # Bounded activity planner: proposes work from real state; the autonomy
        # driver (below) is the enqueue authority. Her affect biases *what* and
        # *how much* she takes on, so mood changes real selection, not wording.
        executable = tuple(self.scheduler._handlers.keys())
        enabled = [k for k in task_mod.TASK_KINDS if k in executable
                   or auto_cfg.get("allow_unhandled")]
        # Don't let the planner select reading when its model call is gated off:
        # a kind that cannot actually run should not be proposed as work.
        if not model_use.from_config(self.config).enabled(model_use.FEATURE_READING):
            enabled = [k for k in enabled if k != task_mod.KIND_READING]
        self.planner = activity.ActivityPlanner(
            self.task_store, store=store, library=library, ledger=self.ledger,
            config=auto_cfg,
            enabled_kinds=enabled,
            affect_provider=(store.current_affect
                             if callable(getattr(store, "current_affect", None)) else None),
        )

        # The autonomy driver coordinates schedules + planning + notifications on
        # one bounded cadence. The notifier is injected once a Discord transport
        # exists; until then it delivers nothing (and says so honestly).
        self.autonomy = autonomy.AutonomyDriver(
            schedule_driver=self.schedule_driver, planner=self.planner,
            notifications=self.notifications,
            scheduled_scheduler=self.scheduled_scheduler, ledger=self.ledger,
            notifier=None, config=auto_cfg)
        # Grounded reasons that are not task outcomes, and the affect-derived
        # relevance that tilts whether such a reason is worth saying. Both are
        # read-only views over the *existing* store; neither stores anything.
        self.autonomy.candidate_provider = self._grounded_candidates
        self.autonomy.relevance_provider = self._outreach_relevance
        self._notifier: Optional[Callable[..., Dict[str, Any]]] = None

        self.poll_seconds = max(1.0, float(poll_seconds))
        self._auto_wake_seconds = max(5.0, float(auto_cfg.get(
            "wake_interval_seconds", 30.0)))
        self.idle_sleep_seconds = max(0.0, float(runtime_cfg.get(
            "idle_sleep_seconds", 30.0)))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        # In-memory marker of the last human interaction (see ``touch``).
        self.last_interaction: Optional[str] = None
        self._last_auto_wake: Optional[datetime] = None

    # -- scheduled-work hook --------------------------------------------
    def _intensity(self) -> Optional[float]:
        """How much Astra can bring to work right now (0..1), or ``None``.

        A pure read of the behaviour modulation's ``effort`` derived from her
        current affect. The scheduler uses it to admit only light work when she
        is flat or sad, so her state changes what actually runs. Read-only: it
        never writes affect, memory or the store.
        """
        try:
            from . import behavior
            state = None
            if callable(getattr(self.store, "current_affect", None)):
                state = self.store.current_affect()
            mod = behavior.from_affect(state) if state else behavior.from_affect({})
            return float(behavior.activity_bias(mod).get("effort", 0.5))
        except Exception:
            return None

    def machine_snapshot(self) -> Dict[str, Any]:
        """A bounded, numeric view of the machine (for status/log, never memory)."""
        try:
            return self.machine.snapshot()
        except Exception:
            return {}

    def _machine_ready_for_work(self) -> bool:
        """Whether the computer is warm and settled enough to do heavy work.

        Conservative at startup (see :mod:`astra.machine`): a machine that just
        booted, or one that just resumed, keeps her in efficiency mode. This is
        the *energy* half of readiness - it deliberately does not refuse work
        because of a high CPU/memory number, because whether a specific task can
        be done is a finer, per-task measured question (``astra.resources``,
        wired through the scheduler's ``machine_sample_provider``). ``None`` /
        unmeasurable is treated as "no gate".
        """
        if not self.power_cfg.get("startup_efficiency", True):
            return True
        try:
            sample = self.machine.sample()
            settled = getattr(self.machine, "sustainably_calm", None)
            if callable(settled):
                return bool(settled(sample))
            return bool(self.machine.ready_for_work(sample))
        except Exception:
            return True

    def _machine_sample_now(self) -> Any:
        """A fresh machine reading for the scheduler's affordability estimate."""
        try:
            sample_now = getattr(self.machine, "sample_now", None)
            if callable(sample_now):
                return sample_now()
            return self.machine.sample()
        except Exception:
            return None

    def _after_scheduled_run(self, task: BackgroundTask, status: str) -> None:
        """Re-arm a recurring scheduled occurrence after it has run.

        Only a completed/cancelled occurrence advances the schedule; a failure
        keeps its due instant so the bounded retry policy (not the schedule)
        decides, and a blocked/missed/expired task is left as-is.
        """
        if not task.schedule:
            return
        if status not in (task_mod.COMPLETED, task_mod.CANCELLED,
                          task_mod.SKIPPED):
            return
        self.schedule_driver.rearm_after_run(task.id)

    def set_notifier(self, notifier: Callable[..., Dict[str, Any]]) -> None:
        """Attach the outbound delivery path (the Discord adapter, usually).

        The autonomy driver only *reads* this; wiring it here keeps the
        application layer free of transport knowledge.
        """
        with self._lock:
            self._notifier = notifier
            self.autonomy.notifier = notifier

    def set_composer(self, composer: Optional[Callable[[Dict[str, Any]], Optional[str]]]) -> None:
        """Attach the optional voice composer for permitted messages.

        The composer only phrases a message the policy already permitted; it can
        never change the delivery decision. Passing ``None`` keeps the
        deterministic summary.
        """
        with self._lock:
            self.autonomy.composer = composer

    # -- grounded reasons for outreach (read-only views) -----------------
    def _grounded_candidates(self) -> List[Dict[str, Any]]:
        """Conversation candidates: real, recorded reasons to speak.

        Reads the same source the surfacing path uses, filtered/ranked by the
        proactive policy. Returns only candidates that carry a concrete reason;
        an empty list is the honest answer when there is nothing grounded to
        say.
        """
        session = self.session
        orchestrator = getattr(session, "orchestrator", None) if session else None
        if orchestrator is None or not callable(
                getattr(orchestrator, "conversation_candidates", None)):
            return []
        try:
            return list(orchestrator.conversation_candidates(limit=6) or [])
        except Exception:
            return []

    def _outreach_relevance(self) -> float:
        """The affect-derived relevance her current state adds to outreach.

        A pure read of the existing affect state (via ``proactive``); returns
        ``0.0`` when the store has no affect accessor or she is flat, so a
        neutral state leaves the decision unchanged.
        """
        if self.store is None or not callable(getattr(self.store, "current_affect", None)):
            return 0.0
        try:
            return proactive.outreach_relevance(self.store.current_affect())
        except Exception:
            return 0.0

    def _engagement_boost(self) -> float:
        """The extra importance threshold her unanswered messages add.

        A pure read of the separate engagement tracker; ``0.0`` when disabled or
        when she has not been speaking into silence. Raising the bar is its only
        effect - it never touches the notification limits or quiet hours.
        """
        try:
            return float(self.engagement.effective_threshold_boost() or 0.0)
        except Exception:
            return 0.0

    def _note_proactive_sent(self, ref_id: Any, *, now: Optional[datetime] = None) -> None:
        """Open an engagement obligation for a delivered standalone message.

        Injected into the notification policy so a *successful send* (never the
        mere decision to send) is what starts the unanswered count. Best-effort:
        a tracker failure never affects delivery.
        """
        try:
            self.engagement.note_sent(ref_id, now=now)
        except Exception:
            pass

    def note_human_message(self) -> None:
        """Record that a human message actually arrived, on any transport.

        Called from the single message-handling choke point
        (``ApplicationRouter.handle``), so a CLI turn and a Discord turn are
        treated identically. This is what lets her recover from silence: a reply
        within the reliable window discharges the unanswered chain.
        """
        try:
            self.engagement.note_reply()
        except Exception:
            pass

    # -- handlers --------------------------------------------------------
    def _register_handlers(self) -> None:
        self.scheduler.register(KIND_READING, self._handle_reading)
        self.scheduler.register(KIND_FILE_INSPECTION, self._handle_files)
        self.scheduler.register(KIND_MEMORY_MAINTENANCE, self._handle_memory)
        self.scheduler.register(KIND_QUESTION_REVIEW, self._handle_questions)
        self.scheduler.register(KIND_CONTEXT_PREPARATION, self._handle_prepare)
        self.scheduler.register(KIND_APP_MAINTENANCE, self._handle_maintenance)
        self.scheduler.register(KIND_WORK_REVIEW, self._handle_work_review)
        self.scheduler.register(KIND_MEMORY_REVIEW, self._handle_memory_review)
        self.scheduler.register(KIND_EXPERIENCE_REVIEW, self._handle_experience_review)
        self.scheduler.register(KIND_SELF_REVIEW, self._handle_self_review)
        self.scheduler.register(KIND_KNOWLEDGE_GAP, self._handle_knowledge_gap)
        self.scheduler.register(KIND_TEST_RUN, self._handle_test_run)
        self.scheduler.register(KIND_SELF_IMPROVEMENT, self._handle_self_improvement)

    def _handle_reading(self, task: BackgroundTask) -> Dict[str, Any]:
        reader = self.reader
        if reader is None:
            # The background reader is gated off (astra/model_use): there is no
            # model call permitted for reading, so this is *not* a completed read.
            # Report it honestly and let the planner stop selecting it.
            return {"status": task_mod.BLOCKED,
                    "detail": "background reader disabled (model_use.reading)"}
        report = reader.run_once()
        read = bool(report.get("read"))
        reason = str(report.get("reason") or "")
        detail = (f"read={read} reason={reason} "
                  f"words={report.get('words')} chunks={report.get('chunks')}")
        summary = str(report.get("reason") or "").strip() or "reading cycle"
        result = {"summary": summary, "count": report.get("chunks")}
        if read:
            return {"status": task_mod.COMPLETED, "detail": detail, "result": result}
        # Nothing was read. Report *why*, honestly, so a genuine failure is not
        # recorded as a completed read and the bounded retry policy can act:
        #   * model unreachable -> FAILED (the scheduler retries a bounded number
        #     of times, then blocks; the passage is not skipped, the offset held)
        #   * no model permitted -> BLOCKED (a decision, not a fault, not work)
        #   * held by courtesy/state (Roum active, a game, load, no concentration)
        #     -> SKIPPED (deferred, not failed; a recurring occurrence re-arms)
        #   * nothing left to read -> COMPLETED with an explicit "no material"
        # A skipped/failed read never fabricates a discovery and never advances
        # the position, because the reader itself does not advance on a hold.
        if reason == reading.IDLE_MODEL_FAILED:
            return {"status": task_mod.FAILED, "detail": detail, "result": result}
        if reason == reading.IDLE_HELD:
            return {"status": task_mod.BLOCKED, "detail": detail, "result": result}
        if reason in (reading.IDLE_NO_WORK, reading.IDLE_WORK_FINISHED):
            return {"status": task_mod.COMPLETED, "detail": detail, "result": result}
        return {"status": task_mod.SKIPPED, "detail": detail, "result": result}

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

    # -- autonomous review/self-improvement handlers ---------------------
    # These are read-only reviews over Astra's real state. They record findings
    # as *outcomes/questions*, never as durable self-knowledge, and they never
    # promote a generated theory about her own implementation into authority.
    def _handle_work_review(self, task: BackgroundTask) -> Dict[str, Any]:
        work_id = str(task.payload.get("work_id") or "").strip() or None
        if self.store is None:
            return {"status": task_mod.COMPLETED, "detail": "no store"}
        try:
            records = self.store.get_work_context(work_id) if work_id else []
        except Exception as exc:
            return {"status": task_mod.FAILED, "detail": f"work review failed: {exc}"}
        return {"status": task_mod.COMPLETED,
                "detail": f"reviewed work {work_id or '(none)'}: {len(records)} record(s)",
                "result": {"count": len(records), "work_id": work_id}}

    def _handle_memory_review(self, task: BackgroundTask) -> Dict[str, Any]:
        if self.store is None:
            return {"status": task_mod.COMPLETED, "detail": "no store"}
        try:
            questions = self.store.get_questions(status="active")
            contradictions = 0
            for model in ("self", "roum"):
                for mem in self.store.get_active_memories(model):
                    if mem.get("contradiction_id") or mem.get("conflict"):
                        contradictions += 1
        except Exception as exc:
            return {"status": task_mod.FAILED, "detail": f"memory review failed: {exc}"}
        detail = f"{len(questions)} open question(s), {contradictions} flagged conflict(s)"
        return {"status": task_mod.COMPLETED, "detail": detail,
                "result": {"count": len(questions)},
                "importance": 0.45 if contradictions else 0.35}

    def _handle_experience_review(self, task: BackgroundTask) -> Dict[str, Any]:
        if self.store is None or not callable(getattr(self.store, "get_experiences", None)):
            return {"status": task_mod.COMPLETED, "detail": "no store"}
        try:
            experiences = self.store.get_experiences()
        except Exception as exc:
            return {"status": task_mod.FAILED, "detail": f"experience review failed: {exc}"}
        # Report the count as an observation, never as proof of learning.
        return {"status": task_mod.COMPLETED,
                "detail": f"reviewed {len(experiences)} recorded experience(s)",
                "result": {"count": len(experiences)},
                "importance": 0.4}

    def _handle_self_review(self, task: BackgroundTask) -> Dict[str, Any]:
        # Review recent behaviour for recurring failures using the kill-safe
        # ledger (not conversational guesses). Read-only; proposes nothing.
        recent = self.ledger.entries(limit=40) if self.ledger else []
        failures = [e for e in recent if e.get("status") in
                    (task_mod.FAILED, task_mod.BLOCKED, task_mod.MISSED)]
        return {"status": task_mod.COMPLETED,
                "detail": f"{len(failures)} recent failure(s) in the last {len(recent)} outcome(s)",
                "result": {"count": len(failures)},
                "importance": 0.5 if len(failures) >= 3 else 0.4}

    def _handle_knowledge_gap(self, task: BackgroundTask) -> Dict[str, Any]:
        # Identify gaps from real state: unresolved questions and unfinished
        # works. It never invents a knowledge source or fetches the network.
        gaps = []
        try:
            if self.store is not None:
                gaps = list(self.store.get_questions(status="active") or [])
        except Exception:
            gaps = []
        try:
            works = self.library.list_works() if self.library is not None else []
        except Exception:
            works = []
        unfinished = [w for w in works if isinstance(w, dict) and not w.get("finished")]
        detail = f"{len(gaps)} open question(s), {len(unfinished)} unfinished work(s)"
        return {"status": task_mod.COMPLETED, "detail": detail,
                "result": {"count": len(gaps)},
                "importance": 0.5 if gaps else 0.35}

    def _handle_test_run(self, task: BackgroundTask) -> Dict[str, Any]:
        """Run the project's own test suite as a bounded, sandboxed evaluation.

        This is an *evaluation* activity, not a change: it runs the code's own
        checks read-only and reports the result. It never modifies source and
        never claims improvement from a passing run.
        """
        import subprocess
        import sys
        from . import paths
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "tests"], cwd=paths.source_root(),
                capture_output=True, text=True,
                timeout=float(self.config.get("autonomy", {}).get("test_timeout_seconds", 300))
                if isinstance(self.config.get("autonomy"), dict) else 300.0)
        except Exception as exc:
            return {"status": task_mod.FAILED, "detail": f"test run failed: {exc}"}
        ok = proc.returncode == 0
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-1:] or [""]
        return {"status": task_mod.COMPLETED if ok else task_mod.FAILED,
                "detail": f"tests {'passed' if ok else 'FAILED'}: {tail[0][:120]}",
                "result": {"summary": "tests passed" if ok else "tests failed"},
                "importance": 0.7 if not ok else 0.4}

    def _handle_self_improvement(self, task: BackgroundTask) -> Dict[str, Any]:
        """Draft an improvement *proposal* for human approval; change nothing.

        The proposal is recorded durably (so it can be acknowledged or approved
        later) and as an outcome (and, if warranted, a message), never applied.
        Core identity, boundaries, credentials, permissions and source code are
        explicitly out of scope for autonomous change; a proposal that would
        touch them is recorded in a protected scope and flagged as requiring
        approval, so it can never proceed on its own.
        """
        proposals = []
        try:
            entries = self.ledger.entries(limit=40) if self.ledger else []
        except Exception:
            entries = []
        failures = [e for e in entries if e.get("status") in
                    (task_mod.FAILED, task_mod.BLOCKED)]
        if failures:
            proposals.append(f"Review {len(failures)} recurring background failure(s)")

        recorded = []
        for title in proposals:
            proposal = proposal_mod.ImprovementProposal(
                title=title,
                scope=proposal_mod.SCOPE_ROUTINE,
                evidence=f"{len(failures)} failed/blocked outcome(s) in the recent ledger",
                proposed_change="Surface the recurring failures for review; no code "
                                "or configuration is changed automatically.",
                source_task_id=task.id,
            )
            stored = self.proposals.add(proposal) if self.proposals else None
            if stored is not None:
                recorded.append(stored.id)

        summary = ("; ".join(proposals) if proposals
                   else "No improvement proposals at this time (nothing evidenced).")
        return {"status": task_mod.COMPLETED,
                "detail": summary,
                "result": {"summary": summary, "count": len(recorded),
                           "importance": 0.55 if proposals else 0.3},
                "importance": 0.55 if proposals else 0.3}

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
        # Materialize configured schedules once at startup so a schedule that
        # came due while the runtime was down is handled on return, honestly.
        self._materialize_schedules()
        # Sample the machine once so startup reads real numbers, not a guess.
        self._sample_machine()
        while not self._stop.is_set():
            try:
                # Recognise a machine suspend (or an imminent one) before doing
                # anything: sleeping alongside the computer is the honest
                # behaviour, and a resume must not be treated as active time.
                self._handle_power()
                # Autonomous cadence: schedules, planned work and (optionally)
                # proactive messages. This is independent of whether a console
                # is open and of the SLEEP/ACTIVE state.
                self._maybe_autonomous_wake()
                if self.state.is_sleeping():
                    # Re-seed so a drained queue refills while asleep. Cheap: the
                    # task store's dedupe means only genuinely missing kinds are
                    # added, so this does not churn the file every cycle.
                    self.seed()
                    self.scheduler.run_cycle()
            except Exception:
                pass  # a worker must never take down the runtime
            self._stop.wait(self.poll_seconds)

    def _sample_machine(self) -> None:
        """Sample the machine once and record it (bounded, read-only).

        The observation is tagged with whether Roum is currently engaged, so the
        history can distinguish the machine's idle baseline from the load his own
        applications produce - the active-vs-idle comparison the affordability
        estimate is grounded in.
        """
        try:
            sample = self.machine.sample()
        except Exception:
            return
        try:
            self.resource_history.observe(sample, active=self._human_active())
        except Exception:
            pass

    def _human_active(self) -> bool:
        """Whether Roum appears to be actively using the machine right now.

        Two read-only signals: a recent *conversation* turn (the in-memory
        ``last_interaction`` marker) or recent OS input (the machine's own idle
        time). Either means he is here, so the machine's current load is his
        doing rather than the runtime's. Unknown idle time is treated as *not*
        active, so a platform without the reading never mislabels a busy machine
        as his.
        """
        try:
            if self._user_present(datetime.now(timezone.utc)):
                return True
        except Exception:
            pass
        try:
            sample = getattr(self.machine, "_last", None)
            idle = getattr(sample, "idle_seconds", None)
            if idle is None:
                return False
            return float(idle) < reading.DEFAULT_IDLE_SECONDS
        except Exception:
            return False

    def _handle_power(self) -> None:
        """Sleep alongside the machine and re-orient on resume.

        Detects a suspend from the machine's own uptime vs. the process clock
        (see :mod:`astra.machine`): on a detected suspend Astra SLEEPs, so the
        gap is not mistaken for active time and she does not run scheduled work
        through a period the computer was off. On resume, work stays
        conservative until the machine is measured warm and calm again.
        """
        sample = None
        try:
            sample = self.machine.sample()
        except Exception:
            return
        if not self.power_cfg.get("sleep_with_computer", True):
            return
        suspended = getattr(sample, "suspended_seconds", None)
        if suspended:
            self._last_suspend_seconds = float(suspended)
            try:
                self.ledger.record(
                    task_id="power:suspend", kind="power",
                    status=task_mod.COMPLETED,
                    detail=f"machine suspended ~{int(suspended)}s; Astra slept alongside it")
            except Exception:
                pass
            if not self.state.is_sleeping():
                self.state.set_state(runtime_state.SLEEP,
                                     reason=runtime_state.REASON_POWER)
            if self.power_cfg.get("wake_on_resume", False):
                # Opt-in: re-arm as ACTIVE on resume. Default is to stay asleep
                # until a human speaks, which is the least presumptuous choice.
                self.wake(reason=runtime_state.REASON_LIFECYCLE)

    def _materialize_schedules(self) -> None:
        configured = []
        auto_cfg = self.config.get("autonomy")
        if isinstance(auto_cfg, dict):
            configured = auto_cfg.get("schedules") or []
        try:
            self.schedule_driver.materialize(configured)
        except Exception:
            pass

    def _maybe_autonomous_wake(self) -> None:
        """Run the autonomy driver on its cadence (bounded, never blocking)."""
        now = datetime.now(timezone.utc)
        last = self._last_auto_wake
        if last is not None and (now - last).total_seconds() < self._auto_wake_seconds:
            return
        self._last_auto_wake = now
        user_active = self._user_present(now)
        try:
            self.autonomy.wake(user_active=user_active, now=now)
        except Exception:
            pass
        # Headless idle policy: with no console and no recent interaction, a
        # *policy* (not background work) may put her to sleep so the sleep-only
        # opportunistic work still gets a turn. A user message always wakes her.
        if self.think_only and self.idle_sleep_seconds > 0 and not user_active:
            if not self.state.is_sleeping():
                self.state.set_state(runtime_state.SLEEP,
                                     reason=runtime_state.REASON_IDLE_POLICY)

    def _user_present(self, now: datetime) -> bool:
        """Whether a human has interacted recently (in-memory marker only)."""
        if not self.last_interaction:
            return False
        try:
            stamp = datetime.fromisoformat(self.last_interaction)
        except (TypeError, ValueError):
            return False
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return (now - stamp).total_seconds() < max(10.0, self.idle_sleep_seconds)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the worker cleanly and flush. Safe to call repeatedly."""
        self._stop.set()
        self.scheduler.stop()
        self.scheduled_scheduler.stop()
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
        pcounts = self.proposals.counts()
        lines.append(f"Improvement proposals open: "
                     f"{pcounts.get(proposal_mod.P_OPEN, 0)}"
                     f" (approved {pcounts.get(proposal_mod.P_APPROVED, 0)})")
        if self.proactive.muted:
            lines.append("Proactive messages: muted")
        gate = model_use.from_config(self.config)
        lines.append("Model use: " + ", ".join(gate.enabled_features)
                     + f" (reading={'on' if gate.enabled(model_use.FEATURE_READING) else 'off'})")
        report = self.scheduler.last_report or {}
        if report:
            lines.append(f"Last cycle: {self.scheduler.last_reason} "
                         f"({report.get('tasks', 0)} tasks)")
        snap = self.machine_snapshot()
        if snap:
            cpu = snap.get("cpu_load")
            mem = snap.get("memory_pressure")
            gpu = snap.get("gpu_util")
            lines.append(
                "Machine: "
                f"cpu={'?' if cpu is None else f'{cpu * 100:.0f}%'} "
                f"mem={'?' if mem is None else f'{mem * 100:.0f}%'} "
                f"gpu={'?' if gpu is None else f'{gpu * 100:.0f}%'} "
                f"{'efficiency-mode' if snap.get('efficiency_mode') else 'ready'}")
            if snap.get("suspended_seconds"):
                lines.append(f"  last suspend: ~{int(snap['suspended_seconds'])}s "
                             "(slept alongside the computer)")
        return "\n".join(lines)

    def snapshot_intentions(self) -> Dict[str, Any]:
        """Astra's own plans, derived read-only from the real records.

        This is the single bridge between the runtime's task/planner/ledger state
        and her conversational self-expression: the orchestrator renders this
        snapshot when asked about her plans, so what she says and what the
        systems record cannot drift. It reads only; nothing here mutates a task.
        """
        return intentions.build_snapshot(
            self.task_store, planner=self.planner, ledger=self.ledger,
            paused=self.autonomy.paused,
            notifications_paused=bool(self.notifications.snapshot().get("paused")),
        )

    def autonomy_report(self) -> str:
        """A truthful, runtime-and-record-based autonomous-status view."""
        lines = ["Autonomy:"]
        snap = self.autonomy.snapshot()
        lines.append(f"  activity: {'paused' if snap['paused'] else 'running'}"
                     f" (enabled={snap['enabled']}, planning={snap['planning_enabled']})")
        lines.append(f"  last wake: {snap.get('last_wake_at') or 'never'}")
        notif = self.notifications.snapshot()
        lines.append(
            "  notifications: "
            f"{'muted' if notif['muted'] else ('paused' if notif['paused'] else 'on')}"
            f", sent {notif['delivered_today']}/{notif['max_per_day']} today"
            f", {notif['delivered_in_interval']}/{notif['max_per_interval']} this interval")
        if notif.get("in_quiet_hours"):
            lines.append("  (quiet hours: active)")
        lines.append(f"  pending digest entries: {notif['pending_digest']}")
        eng = self.engagement.snapshot()
        if eng.get("enabled"):
            unanswered = eng.get("consecutive_unanswered", 0)
            boost = eng.get("threshold_boost", 0.0)
            lines.append(
                f"  engagement: {unanswered} unanswered proactive message(s)"
                + (f", threshold +{boost:.2f}" if boost else "")
                + f", last reply: {eng.get('last_reply_at') or 'never'}")
        scheduled = self.task_store.scheduled()
        if scheduled:
            lines.append("Scheduled activities:")
            for task in scheduled[:8]:
                lines.append(f"  [{task.id}] {task.kind} due {task.due_at or task.next_due_at} "
                             f"({sched.describe(task.schedule or {})})")
        else:
            lines.append("Scheduled activities: none")
        last = self._last_completed()
        lines.append(f"Last completed activity: {last or 'none recorded'}")
        return "\n".join(lines)

    def schedule_report(self, limit: int = 15) -> str:
        """Explain every scheduled/planned activity from its own records.

        This is the section-8/9 "what is scheduled, why, and what next" view.
        It is derived read-only from the task store, so it cannot drift from the
        queue, and it states an overdue occurrence according to its catch-up
        policy rather than implying a run the policy would refuse.
        """
        rows = schedule_explain.explain_all(self.task_store, limit=limit + 1)
        header = ["Scheduled & planned activities:"]
        body = schedule_explain.render(rows, limit=limit).splitlines()
        return "\n".join(header + body)

    def normalize_text_report(self, *, apply: bool = False,
                              limit: int = 8) -> str:
        """Preview or apply normalization of stored memory text (section 10).

        Preview is read-only. Apply rewrites *content only* (idempotent, audited,
        never writing emptiness) and reports the count. Kept behind an explicit
        call so a bulk rewrite of the corpus is never a side effect of viewing.
        """
        if apply:
            result = self.store.normalize_memory_texts()
            return ("Text normalization applied: "
                    f"{result.get('changed', 0)} record(s) rewritten; "
                    "audit at text_normalization_audit.json")
        view = self.store.preview_text_normalization(limit=limit)
        lines = ["Text normalization preview (read-only):"]
        lines.append(f"  records scanned: {view['total']}")
        lines.append(f"  would change: {view['changed']}  "
                     f"(would-be-emptied, skipped: {view['emptied']})")
        for row in view["rows"]:
            lines.append(f"  [{row['id']}] {row['model']}")
            lines.append(f"      before: {str(row['before'])[:120]}")
            lines.append(f"      after:  {str(row['after'])[:120]}")
        if view["changed"] > len(view["rows"]):
            lines.append(f"  ... and {view['changed'] - len(view['rows'])} more")
        lines.append("  Apply with: /normalize-text apply")
        return "\n".join(lines)

    def reader_health_report(self) -> str:
        """Why the background reader is (not) running, in plain language.

        Built from the reader's own diagnostics: the state, the machine reading,
        and whether a busy application is merely a workload (overridable on a
        calm machine) or an absolute exclusion.
        """
        reader = self.reader
        if reader is None:
            return "Reader: not attached in this session."
        fmt = getattr(reader, "format_diagnostics", None)
        if callable(fmt):
            return fmt()
        return "Reader: diagnostics unavailable."

    def discord_report(self, discord: Any) -> str:
        if discord is None:
            return "Discord: not configured"
        diag = discord.diagnostics()
        lines = ["Discord:"]
        lines.append(f"  connected: {diag['connected']}")
        lines.append(f"  last successful poll: {diag.get('last_fetch_ok_at') or 'never'}")
        lines.append(f"  last proactive send: {diag.get('last_proactive_at') or 'never'}")
        lines.append(f"  inbound delivered={diag['inbound_delivered']} "
                     f"failed={diag['inbound_failed']}")
        d = diag.get("deliveries") or {}
        lines.append(f"  proactive deliveries tracked={d.get('tracked', 0)} "
                     f"delivered={d.get('delivered', 0)} failed={d.get('failed', 0)}")
        if diag.get("presence"):
            lines.append(f"  presence: {diag['presence']} "
                         f"(exchange open={diag.get('exchange_open')}, "
                         f"patience={diag.get('patience_seconds')}s)")
        timing = diag.get("timing") or {}
        if timing:
            lines.append(f"  state-scaled timing: typing x{timing.get('typing_scale')} "
                         f"patience x{timing.get('patience_scale')}")
        if diag.get("last_error"):
            lines.append(f"  last error: {diag['last_error']}")
        return "\n".join(lines)

    def _last_completed(self) -> Optional[str]:
        for entry in reversed(self.ledger.entries(limit=50)):
            if entry.get("status") == task_mod.COMPLETED:
                return f"{entry.get('kind')} @ {entry.get('at')}"
        return None

    def pause_activity(self, paused: bool = True) -> str:
        self.autonomy.set_paused(paused)
        return "Autonomous activity paused." if paused else "Autonomous activity resumed."

    def pause_notifications(self, paused: bool = True) -> str:
        self.notifications.set_paused(paused)
        return ("Proactive notifications paused (activity continues)." if paused
                else "Proactive notifications resumed.")

    def sleep_log(self, limit: int = 12) -> str:
        return self.ledger.render(limit)

    def power_report(self) -> str:
        """A truthful, numeric view of machine resources and power behaviour."""
        snap = self.machine_snapshot()
        if not snap:
            return "Machine: not measurable on this platform."
        def pct(value: Any) -> str:
            return "?" if value is None else f"{float(value) * 100:.0f}%"
        lines = ["Machine and power:"]
        lines.append(f"  cpu: {pct(snap.get('cpu_load'))} "
                     f"memory: {pct(snap.get('memory_pressure'))}")
        gpu = snap.get("gpu_util")
        vram_used, vram_total = snap.get("vram_used"), snap.get("vram_total")
        if gpu is not None or vram_total is not None:
            vram = "?"
            if vram_used is not None and vram_total:
                vram = (f"{vram_used / (1024 ** 3):.1f}/"
                        f"{vram_total / (1024 ** 3):.1f} GB")
            lines.append(f"  gpu: {pct(gpu)} vram: {vram}")
        else:
            lines.append("  gpu: not measurable on this platform")
        up = snap.get("uptime_seconds")
        lines.append(f"  uptime: {'?' if up is None else f'{up / 3600:.1f}h'}")
        lines.append(f"  ready for heavy work: {snap.get('ready_for_work')} "
                     f"(efficiency mode: {snap.get('efficiency_mode')})")
        lines.append(f"  startup_efficiency={self.power_cfg['startup_efficiency']} "
                     f"sleep_with_computer={self.power_cfg['sleep_with_computer']} "
                     f"wake_on_resume={self.power_cfg['wake_on_resume']}")
        if self._last_suspend_seconds:
            lines.append(f"  last suspend: ~{int(self._last_suspend_seconds)}s "
                         "(Astra slept alongside the computer)")
        return "\n".join(lines)

    def resource_report(self) -> str:
        """The active-vs-idle comparison and the learned per-task costs.

        Read-only: it renders the durable resource history (what the machine
        looks like when Roum is engaged versus idle, and what each kind of task
        has actually cost) and the estimate for the next task of each kind. It
        never stores anything and never reaches Astra's prompt.
        """
        def pct(value: Any) -> str:
            return "?" if value is None else f"{float(value) * 100:.0f}%"

        def row(label: str, base: Dict[str, Any]) -> str:
            return (f"  {label}: cpu={pct(base.get('cpu'))} "
                    f"mem={pct(base.get('mem'))} gpu={pct(base.get('gpu'))} "
                    f"vram={pct(base.get('vram'))} "
                    f"({base.get('samples', 0)} sample(s))")

        lines = ["Resource history:"]
        try:
            snap = self.resource_history.snapshot()
        except Exception:
            return "Resource history: unavailable."
        lines.append(row("idle (machine alone)", snap.get("idle_baseline") or {}))
        lines.append(row("active (Roum's apps)", snap.get("active_baseline") or {}))
        lines.append("Estimated cost per task kind (learned, else prior):")
        for kind in sorted(task_mod.TASK_KINDS):
            cost = self.scheduler.task_cost(kind)
            used = resources_mod.KIND_RESOURCES.get(kind, ())
            parts = ", ".join(f"{r}={pct(getattr(cost, r))}" for r in used
                              if getattr(cost, r, 0.0) > 0.0)
            lines.append(f"  {kind}: {parts or 'negligible'}")
        return "\n".join(lines)

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

        # A human message actually arrived: this is the *only* signal that a
        # proactive message was engaged with (delivery alone never is). Slash
        # commands count - the person is present - so every transport and every
        # input kind recovers the unanswered chain identically.
        if self.controller is not None and callable(
                getattr(self.controller, "note_human_message", None)):
            self.controller.note_human_message()
        # Presence: any routed human message (CLI or Discord) marks Roum as here,
        # so the runtime knows he is around without a state flip. In-memory only.
        if self.controller is not None and callable(
                getattr(self.controller, "touch", None)):
            self.controller.touch()

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

        if command in ("/autonomy", "/autonomous"):
            return {"text": self._autonomy_command(parts[1:]), "handled": True,
                    "source": source}

        if command in ("/notifications", "/notify"):
            return {"text": self._notifications_command(parts[1:]), "handled": True,
                    "source": source}

        if command == "/discord":
            return {"text": controller.discord_report(self.session.discord),
                    "handled": True, "source": source}

        if command in ("/check-messages", "/check_messages", "/messages"):
            return {"text": self._check_messages_command(), "handled": True,
                    "source": source}

        if command == "/proposals":
            return {"text": self._proposals_command(parts[1:]), "handled": True,
                    "source": source}

        if command in ("/power", "/machine"):
            return {"text": controller.power_report(), "handled": True, "source": source}

        if command in ("/resources", "/resource-history"):
            return {"text": controller.resource_report(), "handled": True,
                    "source": source}

        if command == "/schedules":
            return {"text": self._schedules_text(), "handled": True, "source": source}

        if command in ("/normalize-text", "/normalize_text"):
            apply = any(a.lower() in ("apply", "--apply", "-a") for a in parts[1:])
            return {"text": controller.normalize_text_report(apply=apply),
                    "handled": True, "source": source}

        if command in ("/reader", "/reader-health"):
            return {"text": controller.reader_health_report(), "handled": True,
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

    def _check_messages_command(self) -> str:
        """The explicit "check my Discord messages" action.

        Fetches the channel and answers anything she has not handled, bypassing
        the mention gate (she is looking on purpose). Reports honestly whether
        Discord is even connected, so an empty result is not mistaken for silence.
        """
        adapter = self.session.discord
        if adapter is None:
            return "Discord is not configured, so there are no messages to check."
        report = adapter.poll_once(force=True)
        if not report.get("ok"):
            return f"Could not reach Discord: {report.get('error')}"
        handled = int(report.get("handled") or 0)
        accepted = int(report.get("accepted") or 0)
        if handled:
            return f"Checked messages: answered {handled}."
        if accepted:
            return f"Checked messages: {accepted} held for a moment (she is reading)."
        return "Checked messages: nothing new addressed to me."

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

    def _autonomy_command(self, args: List[str]) -> str:
        controller = self.controller
        if not args:
            return controller.autonomy_report()
        verb = args[0].lower()
        if verb in ("pause", "stop"):
            # Pausing activity does not touch notifications, and vice versa.
            return controller.pause_activity(True)
        if verb in ("resume", "start"):
            return controller.pause_activity(False)
        if verb == "now":
            # One bounded wake, on demand, for verification.
            report = controller.autonomy.wake(now=datetime.now(timezone.utc))
            return ("Autonomy wake:\n" +
                    "\n".join(f"  {k}: {v}" for k, v in report.items()))
        return ("Usage: /autonomy [pause|resume|now]\n\n" +
                controller.autonomy_report())

    def _notifications_command(self, args: List[str]) -> str:
        controller = self.controller
        if not args:
            return controller.autonomy_report()
        verb = args[0].lower()
        if verb in ("pause", "off", "hold"):
            return controller.pause_notifications(True)
        if verb in ("resume", "on"):
            return controller.pause_notifications(False)
        if verb == "mute":
            controller.notifications.set_muted(True)
            return "Proactive notifications muted."
        if verb == "unmute":
            controller.notifications.set_muted(False)
            return "Proactive notifications unmuted."
        return "Usage: /notifications [pause|resume|mute|unmute]"

    def _schedules_text(self) -> str:
        # The explanation is derived read-only from the task records, and states
        # an overdue occurrence according to its catch-up policy rather than
        # implying a run the policy would refuse (see astra.schedule_explain).
        return self.controller.schedule_report()

    def _proposals_command(self, args: List[str]) -> str:
        """Inspect or resolve durable improvement proposals.

        Listing and acknowledging change no code; approving only flips a status
        that a separate human-driven step would consume. Nothing here applies a
        proposal, and nothing here can touch identity, boundaries, credentials,
        permissions or source code.
        """
        store = self.controller.proposals
        if not args:
            return store.render_open()
        verb = args[0].lower()
        if verb in ("all", "history"):
            items = store.all()
            if not items:
                return "No improvement proposals have been recorded."
            return "\n".join(["Improvement proposals:"] +
                             [f"  {p.public_line()}" for p in items[-20:]])
        if verb in ("ack", "acknowledge", "approve", "decline", "reject"):
            if len(args) < 2:
                return f"Usage: /proposals {verb} <proposal_id>"
            status = {"ack": proposal_mod.P_ACKNOWLEDGED,
                      "acknowledge": proposal_mod.P_ACKNOWLEDGED,
                      "approve": proposal_mod.P_APPROVED,
                      "decline": proposal_mod.P_DECLINED,
                      "reject": proposal_mod.P_DECLINED}[verb]
            note = " ".join(args[2:]).strip()
            item = store.set_status(args[1], status, note=note)
            if item is None:
                return f"No proposal {args[1]}."
            if status == proposal_mod.P_APPROVED and item.requires_approval:
                return (f"Approved {item.id} (scope {item.scope}). This is a "
                        "recorded decision only; the change is applied by a "
                        "separate, deliberate step, never by the runtime.")
            return f"{item.public_line()}"
        return "Usage: /proposals [a|all|ack|approve|decline <proposal_id>]"

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
