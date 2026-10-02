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
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
BLOCKED = "blocked"
CANCELLED = "cancelled"
DEFERRED = "deferred"
AWAITING_HUMAN = "awaiting_human"

TASK_STATUSES = (PENDING, RUNNING, COMPLETED, FAILED, BLOCKED, CANCELLED,
                 DEFERRED, AWAITING_HUMAN)
# A task in one of these is finished; it will not run again unless re-queued.
TERMINAL_STATUSES = (COMPLETED, CANCELLED)
# A task in one of these may still run later.
LIVE_STATUSES = (PENDING, FAILED, DEFERRED, BLOCKED, AWAITING_HUMAN)

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

TASK_KINDS = (
    KIND_MEMORY_MAINTENANCE,
    KIND_QUESTION_REVIEW,
    KIND_WORK_REVIEW,
    KIND_READING,
    KIND_FILE_INSPECTION,
    KIND_CALCULATION,
    KIND_CONTEXT_PREPARATION,
    KIND_APP_MAINTENANCE,
)

# -- origins -----------------------------------------------------------------
ORIGIN_SCHEDULER = "scheduler"     # enqueued deterministically by the app
ORIGIN_HUMAN = "human"             # a person asked for it
ORIGIN_LIFECYCLE = "lifecycle"     # created at startup/shutdown
ORIGIN_MODEL = "model"             # proposed by the model (still validated)

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
}
DEFAULT_KIND_PRIORITY = 25


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def logical_key(self) -> str:
        return f"{self.kind}:{self.dedupe_key}"

    def is_live(self) -> bool:
        return self.status in LIVE_STATUSES or self.status == RUNNING

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

    def all(self) -> List[BackgroundTask]:
        with self._lock:
            return list(self._tasks)

    def eligible(self) -> List[BackgroundTask]:
        """Live, runnable tasks (pending or retryable-failed), highest priority
        first then oldest first - a deterministic order."""
        with self._lock:
            runnable = [
                t for t in self._tasks
                if t.status in (PENDING, FAILED, DEFERRED)
                and t.attempts < t.max_attempts
            ]
        runnable.sort(key=lambda t: (-t.priority, t.created_at, t.id))
        return runnable

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
                k: result[k] for k in ("summary", "count", "reason", "path")
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
