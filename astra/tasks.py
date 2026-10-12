"""Background tasks: the application-level unit of work the scheduler runs.

A background task is an *application object*, not a memory and not a prompt
instruction. It has an explicit identity, kind, origin, priority and lifecycle,
and it is persisted with its own serialization contract. The LLM may propose a
task's payload or classify its material, but the scheduler - not the model -
decides whether a task is legal, eligible, and when it runs.

Two stores:

* :class:`TaskStore` - the durable task queue, one JSON file with a version.
* :class:`TaskLedger` - a bounded, append-only record of what was attempted and
  what happened, for reporting. The ledger is application state, never
  autobiography, and is never written into the memory models.

Lifecycle states are explicit and distinguish the outcomes the spec requires:
``completed``, ``failed``, ``blocked``, ``cancelled``, ``deferred`` and
``awaiting_human``.
"""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .memory import atomic_save, load_json

TASK_VERSION = 1
LEDGER_VERSION = 1

# -- lifecycle ---------------------------------------------------------------
PENDING = "pending"
SCHEDULED = "scheduled"          # planned for a future due time (not yet runnable)
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
BLOCKED = "blocked"
CANCELLED = "cancelled"
DEFERRED = "deferred"
SKIPPED = "skipped"              # window passed / catch-up=skip: not run, honestly
MISSED = "missed"               # its window arrived while the runtime was down
EXPIRED = "expired"             # past its `expires` instant without running
AWAITING_HUMAN = "awaiting_human"

TASK_STATUSES = (PENDING, SCHEDULED, RUNNING, COMPLETED, FAILED, BLOCKED,
                 CANCELLED, DEFERRED, SKIPPED, MISSED, EXPIRED, AWAITING_HUMAN)
# A task in one of these is finished; it will not run again unless re-queued.
# MISSED and SKIPPED are terminal *for that occurrence*: the schedule (if any)
# decides when the next occurrence is, rather than resurrecting this record.
TERMINAL_STATUSES = (COMPLETED, CANCELLED, SKIPPED, MISSED, EXPIRED)
# A task in one of these may still run later.
LIVE_STATUSES = (PENDING, SCHEDULED, FAILED, DEFERRED, BLOCKED, AWAITING_HUMAN)

# -- task kinds --------------------------------------------------------------
# The task classes the specification requires. Each is a distinct kind so the
# scheduler can budget and prioritise them independently; new kinds integrate
# through the same interface rather than inventing their own idle loop.
KIND_MEMORY_MAINTENANCE = "memory_maintenance"
KIND_QUESTION_REVIEW = "question_review"
KIND_WORK_REVIEW = "work_knowledge_review"
KIND_READING = "reading"
KIND_FILE_INSPECTION = "file_inspection"
KIND_CALCULATION = "calculation"
KIND_CONTEXT_PREPARATION = "context_preparation"
KIND_APP_MAINTENANCE = "app_maintenance"
# Autonomous/self-improvement kinds. These are bounded activities Astra may
# select for herself from her own state; each maps to a real capability or
# explicitly no-ops, so no kind invents an ability the runtime does not have.
KIND_MEMORY_REVIEW = "memory_review"
KIND_EXPERIENCE_REVIEW = "experience_review"
KIND_SELF_REVIEW = "self_review"
KIND_KNOWLEDGE_GAP = "knowledge_gap_review"
KIND_TEST_RUN = "test_run"
KIND_SELF_IMPROVEMENT = "self_improvement"

TASK_KINDS = (
    KIND_MEMORY_MAINTENANCE,
    KIND_QUESTION_REVIEW,
    KIND_WORK_REVIEW,
    KIND_READING,
    KIND_FILE_INSPECTION,
    KIND_CALCULATION,
    KIND_CONTEXT_PREPARATION,
    KIND_APP_MAINTENANCE,
    KIND_MEMORY_REVIEW,
    KIND_EXPERIENCE_REVIEW,
    KIND_SELF_REVIEW,
    KIND_KNOWLEDGE_GAP,
    KIND_TEST_RUN,
    KIND_SELF_IMPROVEMENT,
)

# -- origins -----------------------------------------------------------------
ORIGIN_SCHEDULER = "scheduler"     # enqueued deterministically by the app
ORIGIN_HUMAN = "human"             # a person asked for it
ORIGIN_LIFECYCLE = "lifecycle"     # created at startup/shutdown
ORIGIN_MODEL = "model"             # proposed by the model (still validated)
ORIGIN_SCHEDULE = "schedule"       # materialised from a configured schedule
ORIGIN_ACTIVITY = "activity"       # selected by the bounded activity planner

# Default priority per kind (higher runs first). These are policy defaults the
# scheduler may override per task; they are not a truth about the task.
DEFAULT_PRIORITY: Dict[str, int] = {
    KIND_MEMORY_MAINTENANCE: 60,
    KIND_QUESTION_REVIEW: 55,
    KIND_CONTEXT_PREPARATION: 50,
    KIND_APP_MAINTENANCE: 45,
    KIND_WORK_REVIEW: 40,
    KIND_CALCULATION: 35,
    KIND_READING: 30,
    KIND_FILE_INSPECTION: 20,
    # Autonomous/self-improvement defaults. Review kinds sit mid-range; test
    # running is heavier and lower; a self-improvement proposal is a summary.
    KIND_EXPERIENCE_REVIEW: 58,
    KIND_MEMORY_REVIEW: 54,
    KIND_KNOWLEDGE_GAP: 48,
    KIND_SELF_REVIEW: 42,
    KIND_SELF_IMPROVEMENT: 38,
    KIND_TEST_RUN: 28,
}
DEFAULT_KIND_PRIORITY = 25


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _is_due(task: "BackgroundTask", now_dt: datetime) -> bool:
    """Whether a task is runnable now. Scheduled tasks wait for their due time."""
    if not task.schedule:
        return True
    due = _parse_iso(task.due() or "")
    if due is None:
        # A scheduled task with no computed due instant is not yet runnable;
        # the schedule driver fills it in. Never run it "just in case".
        return False
    return due <= now_dt


@dataclass
class BackgroundTask:
    """One unit of background work with an explicit lifecycle.

    ``dedupe_key`` is the *logical* identity: two tasks with the same
    ``(kind, dedupe_key)`` are the same work, so the store refuses to enqueue a
    duplicate while one is still live. ``requires_confirmation`` marks work that
    may not run until a human authorises it (it parks in ``awaiting_human``).
    """

    kind: str
    origin: str = ORIGIN_SCHEDULER
    priority: Optional[int] = None
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    status: str = PENDING
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    dedupe_key: str = ""
    requires_confirmation: bool = False
    confirmation_id: Optional[str] = None
    blocked_reason: Optional[str] = None
    # -- schedule awareness -------------------------------------------------
    # A task with a `schedule` is *planned*: it runs only when its due instant
    # arrives (the scheduler enforces this), never merely because it is pending.
    schedule: Optional[Dict[str, Any]] = None
    due_at: Optional[str] = None        # this occurrence's due instant (UTC ISO)
    last_run_at: Optional[str] = None   # when it last actually ran
    next_due_at: Optional[str] = None   # the next computed occurrence (UTC ISO)

    def __post_init__(self) -> None:
        if self.kind not in TASK_KINDS:
            raise ValueError(f"unknown task kind: {self.kind!r}")
        if not self.dedupe_key:
            self.dedupe_key = self.kind
        if self.priority is None:
            self.priority = DEFAULT_PRIORITY.get(self.kind, DEFAULT_KIND_PRIORITY)
        if self.requires_confirmation and self.status == PENDING:
            # A confirmation-gated task never starts life as runnable.
            self.status = AWAITING_HUMAN
        # A scheduled task that is not yet runnable parks as SCHEDULED so it is
        # visible as *planned* rather than an immediately-eligible pending task.
        if self.schedule and self.status == PENDING and self.origin == ORIGIN_SCHEDULE:
            self.status = SCHEDULED

    def logical_key(self) -> str:
        return f"{self.kind}:{self.dedupe_key}"

    def is_live(self) -> bool:
        return self.status in LIVE_STATUSES or self.status == RUNNING

    def is_scheduled(self) -> bool:
        return bool(self.schedule)

    def due(self) -> Optional[str]:
        """The instant this occurrence is due, or None for an unscheduled task."""
        return self.due_at or self.next_due_at

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> Optional["BackgroundTask"]:
        if not isinstance(data, dict):
            return None
        try:
            task = cls(
                kind=str(data.get("kind") or ""),
                origin=str(data.get("origin") or ORIGIN_SCHEDULER),
                priority=int(data.get("priority") or 0),
                id=str(data.get("id") or f"task_{uuid.uuid4().hex[:12]}"),
                status=str(data.get("status") or PENDING),
                created_at=str(data.get("created_at") or _now()),
                updated_at=str(data.get("updated_at") or _now()),
                attempts=int(data.get("attempts") or 0),
                max_attempts=int(data.get("max_attempts") or 3),
                last_error=data.get("last_error"),
                result=data.get("result") if isinstance(data.get("result"), dict) else None,
                progress=dict(data.get("progress") or {}),
                payload=dict(data.get("payload") or {}),
                dedupe_key=str(data.get("dedupe_key") or ""),
                requires_confirmation=bool(data.get("requires_confirmation")),
                confirmation_id=data.get("confirmation_id"),
                blocked_reason=data.get("blocked_reason"),
                schedule=data.get("schedule") if isinstance(data.get("schedule"), dict) else None,
                due_at=data.get("due_at"),
                last_run_at=data.get("last_run_at"),
                next_due_at=data.get("next_due_at"),
            )
        except (TypeError, ValueError):
            return None
        if task.status not in TASK_STATUSES:
            task.status = PENDING
        return task


class TaskStore:
    """Durable task queue. Own file, own version, atomic writes."""

    def __init__(self, data_dir: str = "./storage") -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "background_tasks.json")
        self._lock = threading.RLock()
        self._tasks: List[BackgroundTask] = self._load()

    def _load(self) -> List[BackgroundTask]:
        raw = load_json(self.path, dict, dict)
        items = raw.get("tasks") if isinstance(raw, dict) else None
        out: List[BackgroundTask] = []
        for item in items or []:
            task = BackgroundTask.from_dict(item)
            if task is not None:
                out.append(task)
        return out

    def _persist(self) -> None:
        with self._lock:
            atomic_save(self.path, {
                "version": TASK_VERSION,
                "tasks": [t.to_dict() for t in self._tasks],
            })

    # -- enqueue ---------------------------------------------------------
    def enqueue(self, task: BackgroundTask) -> Optional[BackgroundTask]:
        """Add a task unless a live task with the same logical key exists.

        Returns the stored task (existing or new), or ``None`` if the argument
        was not a task. Duplicate prevention is here, at the single write path,
        so no caller can enqueue the same logical work twice.
        """
        if not isinstance(task, BackgroundTask):
            return None
        with self._lock:
            existing = self._find_live(task.logical_key())
            if existing is not None:
                return existing
            self._tasks.append(task)
            self._persist()
            return task

    def _find_live(self, logical_key: str) -> Optional[BackgroundTask]:
        for task in self._tasks:
            if task.logical_key() == logical_key and task.is_live():
                return task
        return None

    # -- read ------------------------------------------------------------
    def get(self, task_id: str) -> Optional[BackgroundTask]:
        with self._lock:
            for task in self._tasks:
                if task.id == task_id:
                    return task
        return None

    def find_live_by_key(self, kind: str, dedupe_key: str) -> Optional[BackgroundTask]:
        with self._lock:
            return self._find_live(f"{kind}:{dedupe_key}")

    def all(self) -> List[BackgroundTask]:
        with self._lock:
            return list(self._tasks)

    def eligible(self, *, now: Optional[datetime] = None) -> List[BackgroundTask]:
        """Live, runnable tasks, highest priority first then oldest first.

        A *scheduled* task is eligible only once its due instant has arrived;
        an unscheduled task is eligible immediately. ``now`` is injected (UTC)
        so a caller's clock is the single source of truth and tests are
        deterministic.
        """
        now_dt = now or datetime.now(timezone.utc)
        with self._lock:
            runnable = [
                t for t in self._tasks
                if t.status in (PENDING, FAILED, DEFERRED)
                and t.attempts < t.max_attempts
                and _is_due(t, now_dt)
            ]
        runnable.sort(key=lambda t: (-t.priority, t.created_at, t.id))
        return runnable

    def scheduled(self) -> List[BackgroundTask]:
        """Scheduled tasks (planned work), regardless of due time."""
        with self._lock:
            return [t for t in self._tasks if t.schedule and t.is_live()]

    def expired(self, *, now: Optional[datetime] = None) -> List[BackgroundTask]:
        """Scheduled tasks whose ``expires`` instant has passed while live."""
        now_dt = now or datetime.now(timezone.utc)
        out = []
        with self._lock:
            for t in self._tasks:
                if not t.schedule or t.status in TERMINAL_STATUSES:
                    continue
                expires = _parse_iso(t.schedule.get("expires"))
                if expires is not None and now_dt >= expires:
                    out.append(t)
        return out

    def awaiting_human(self) -> List[BackgroundTask]:
        with self._lock:
            return [t for t in self._tasks if t.status == AWAITING_HUMAN]

    # -- update ----------------------------------------------------------
    def update(self, task_id: str, **fields: Any) -> Optional[BackgroundTask]:
        with self._lock:
            task = self.get(task_id)
            if task is None:
                return None
            for key, value in fields.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = _now()
            self._persist()
            return task

    def mark(self, task_id: str, status: str, **fields: Any) -> Optional[BackgroundTask]:
        if status not in TASK_STATUSES:
            raise ValueError(f"unknown task status: {status!r}")
        return self.update(task_id, status=status, **fields)

    def clear_terminal(self) -> int:
        """Drop completed/cancelled tasks from the queue (history stays in the
        ledger). Returns how many were dropped."""
        with self._lock:
            before = len(self._tasks)
            self._tasks = [t for t in self._tasks if t.status not in TERMINAL_STATUSES]
            removed = before - len(self._tasks)
            if removed:
                self._persist()
            return removed

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {status: 0 for status in TASK_STATUSES}
        with self._lock:
            for task in self._tasks:
                out[task.status] = out.get(task.status, 0) + 1
        return out


class TaskLedger:
    """Bounded record of background attempts and their outcomes.

    Application state for reporting - never autobiography. Entries are small and
    capped, and the ledger is never converted into memories.
    """

    MAX_ENTRIES = 200

    def __init__(self, data_dir: str = "./storage") -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "sleep_ledger.json")
        self._lock = threading.RLock()
        self._entries: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        raw = load_json(self.path, dict, dict)
        items = raw.get("entries") if isinstance(raw, dict) else None
        return [dict(e) for e in items or [] if isinstance(e, dict)]

    def _persist(self) -> None:
        with self._lock:
            atomic_save(self.path, {
                "version": LEDGER_VERSION,
                "entries": self._entries[-self.MAX_ENTRIES:],
            })

    def record(self, *, task_id: str, kind: str, status: str,
               detail: str = "", result: Optional[Dict[str, Any]] = None,
               cycle: Optional[int] = None) -> Dict[str, Any]:
        entry = {
            "at": _now(),
            "task_id": task_id,
            "kind": kind,
            "status": status,
            "detail": str(detail or ""),
            "cycle": cycle,
        }
        if result:
            # Keep the ledger small: store a short summary, not the whole result.
            entry["result"] = {
                k: result[k] for k in ("summary", "count", "reason", "path",
                                       "importance", "work_id")
                if k in result
            }
        with self._lock:
            self._entries.append(entry)
            self._persist()
        return entry

    def entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._entries)
        if limit is not None:
            items = items[-limit:]
        return items

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._persist()

    def render(self, limit: int = 12) -> str:
        """A concise, human-readable recent-activity summary."""
        items = self.entries(limit)
        if not items:
            return "No background activity recorded yet."
        lines = ["Recent sleep activity:"]
        for entry in reversed(items):
            detail = f" - {entry['detail']}" if entry.get("detail") else ""
            lines.append(
                f"  [{entry['status']}] {entry['kind']}{detail}"
            )
        return "\n".join(lines)
