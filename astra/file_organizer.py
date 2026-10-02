"""Autonomous background file organization: assessment, decisions, worker.

Astra can monitor and organize the *user's* filesystem as an ongoing background
responsibility, not only when prompted. This module holds the three pieces that
make that work:

* :class:`FileAssessor` - a deterministic, evidence-based judgement of how
  *important* a file is and how *confident* we are about a disposition. It uses
  context (directory, relationships, project usage, age, duplicates, type and
  optionally contents) rather than the filename, which is never treated as
  authoritative. Confidence and importance are separate: a file can be probably
  unimportant without that being certain.
* :class:`DecisionStore` - persistent pending decisions with their reasoning,
  proposed action, confidence, evidence and status. Owned application state,
  separate from conversational history, with an explicit status lifecycle
  (pending / approved / executing / completed / rejected / deferred / failed).
  Closing the UI does not lose them.
* :class:`FileOrganizer` - the bounded background worker. It scans, assesses,
  acts autonomously on routine high-confidence non-destructive work, and parks
  ambiguous or destructive work as a pending decision. Bulk mechanical work is
  delegated to an external utility when one is available.

Astra's own runtime/storage/config/log paths are protected and are never
inspected as user files nor treated as evidence of user importance.
"""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from . import filesystem
from .memory import atomic_save, load_json

DECISION_VERSION = 1

# Decision lifecycle states (spec). "executing" is the in-flight state.
D_PENDING = "pending"
D_APPROVED = "approved"
D_EXECUTING = "executing"
D_COMPLETED = "completed"
D_REJECTED = "rejected"
D_DEFERRED = "deferred"
D_FAILED = "failed"
DECISION_STATUSES = (D_PENDING, D_APPROVED, D_EXECUTING, D_COMPLETED,
                    D_REJECTED, D_DEFERRED, D_FAILED)
OPEN_DECISION_STATUSES = (D_PENDING, D_APPROVED, D_DEFERRED)

# Dispositions offered for a file.
DISPOSITION_KEEP = "keep"
DISPOSITION_ORGANIZE = "organize"
DISPOSITION_ARCHIVE = "archive"
DISPOSITION_TRASH = "trash"          # destructive: routed to trash, asked first
DISPOSITION_INSPECT = "inspect"      # need more evidence before deciding

# Thresholds. Autonomous action needs BOTH high confidence and a routine
# (non-destructive) disposition; anything ambiguous or destructive is asked.
AUTONOMOUS_CONFIDENCE = 0.75
ASK_CONFIDENCE = 0.55
IMPORTANT_THRESHOLD = 0.6

# Files whose contents are worth reading before deciding (small, textual).
_INSPECTABLE_EXTS = {".txt", ".md", ".markdown", ".log", ".csv", ".json",
                     ".yaml", ".yml", ".ini", ".cfg", ".py", ".js", ".html"}
_MAX_INSPECT_BYTES = 8192
# Directory names that usually mean "throwaway" (weak, contextual signals only).
_TEMP_DIR_NAMES = {"tmp", "temp", "cache", ".cache", "logs", "log", "build",
                   "dist", ".tmp", "downloads", "trash"}
# Markers that a directory is a real project the user works in.
_PROJECT_MARKERS = {".git", "package.json", "pyproject.toml", "requirements.txt",
                    "setup.py", "Cargo.toml", "go.mod", "pom.xml", "Makefile",
                    "composer.json", "Gemfile"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class Assessment:
    """The result of judging one file: separate importance and confidence."""

    importance: float = 0.5          # how much the file likely matters to the user
    confidence: float = 0.0          # how sure we are of the proposed disposition
    disposition: str = DISPOSITION_INSPECT
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    sensitive: bool = False          # privacy is separate from importance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def inspect_text(path: str, *, max_bytes: int = _MAX_INSPECT_BYTES) -> Optional[str]:
    """Read a bounded, best-effort text preview of a file (evidence, not action).

    Reuses the library's HTML/EPUB extraction for those types when available, so
    the existing file-reading capability is what inspects content; otherwise a
    plain bounded read is used. Returns ``None`` when the file is not readable
    as text - never raises.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".html", ".htm"):
            from . import library
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                return library.html_to_text(handle.read(max_bytes))[:max_bytes]
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(max_bytes)
    except OSError:
        return None


class FileAssessor:
    """Deterministic, evidence-based file assessment (no model, no filename truth)."""

    def __init__(self, policy: filesystem.FsPolicy, *,
                 sensitive_patterns: Iterable[str] = ()) -> None:
        self.policy = policy
        self.sensitive_patterns = tuple(str(p).lower() for p in sensitive_patterns or ())

    def _is_sensitive(self, path: str) -> bool:
        low = os.path.basename(path).lower()
        return any(pat and pat in low for pat in self.sensitive_patterns)

    def assess(self, path: str, *, entries: Iterable[Dict[str, Any]] = (),
               duplicate: bool = False, inspect: bool = True) -> Assessment:
        path = os.path.abspath(path)
        name = os.path.basename(path)
        ext = os.path.splitext(name)[1].lower()
        evidence: Dict[str, Any] = {"ext": ext}
        sensitive = self._is_sensitive(path)

        # A filename contributes only its extension as a *weak* signal; the base
        # name is deliberately not used to judge importance, because names are
        # not authoritative evidence.
        try:
            st = os.stat(path)
            age_days = (datetime.now(timezone.utc)
                        - datetime.fromtimestamp(st.st_mtime, timezone.utc)).total_seconds() / 86400.0
            evidence["age_days"] = round(age_days, 1)
            evidence["size"] = st.st_size
        except OSError:
            age_days = None

        parts = {p.lower() for p in path.split(os.sep)}
        in_temp = bool(parts & _TEMP_DIR_NAMES)
        evidence["in_temp_dir"] = in_temp

        # Project context: a directory carrying a project marker means the user
        # actively works here, which raises importance.
        parent = os.path.dirname(path)
        project_markers = self._project_markers(parent, entries)
        evidence["project_markers"] = sorted(project_markers)
        in_project = bool(project_markers)
        evidence["duplicate"] = bool(duplicate)

        # Content inspection (optional): a change in what we know, not a reason
        # to act on its own.
        content_evidence: Dict[str, Any] = {}
        if inspect and not sensitive and ext in _INSPECTABLE_EXTS:
            text = inspect_text(path)
            if text is not None:
                content_evidence["readable"] = True
                content_evidence["chars"] = len(text)
        evidence["content"] = content_evidence

        # --- importance -------------------------------------------------
        importance = 0.45
        if in_project:
            importance += 0.25
        if in_temp:
            importance -= 0.15
        if duplicate:
            importance -= 0.10
        if age_days is not None and age_days > 365:
            importance -= 0.10
        if ext in (".txt", ".md", ".markdown", ".pdf", ".doc", ".docx", ".odt"):
            importance += 0.05
        if ext in (".tmp", ".temp", ".bak", ".log", ".cache"):
            importance -= 0.10
        importance = _clamp01(importance, 0.5)

        # --- confidence in a disposition --------------------------------
        # Routine, well-understood situations are confident; ambiguity is not.
        if in_project:
            disposition = DISPOSITION_KEEP
            confidence = 0.8
            reason = "inside an active project directory"
        elif in_temp and (age_days is not None and age_days > 30):
            disposition = DISPOSITION_ARCHIVE
            confidence = 0.7
            reason = "old file in a temporary/cache directory"
        elif duplicate and age_days is not None and age_days > 90:
            disposition = DISPOSITION_TRASH
            confidence = 0.6  # destructive: asked, never autonomous
            reason = "duplicate content, older copy"
        elif age_days is not None and age_days > 730 and importance < 0.4:
            disposition = DISPOSITION_ARCHIVE
            confidence = 0.65
            reason = "long-unused, low-importance file"
        else:
            disposition = DISPOSITION_INSPECT
            confidence = 0.3
            reason = "not enough context to judge"

        if sensitive:
            # Privacy does not lower confidence or importance; it only means the
            # contents are not surfaced and destructive acts are extra care.
            evidence["sensitive"] = True

        return Assessment(importance=round(importance, 4),
                          confidence=round(confidence, 4),
                          disposition=disposition, reason=reason,
                          evidence=evidence, sensitive=sensitive)

    @staticmethod
    def _project_markers(parent: str, entries: Iterable[Dict[str, Any]]) -> set:
        markers = set()
        try:
            names = {n.lower() for n in os.listdir(parent)}
        except OSError:
            names = set()
        markers |= (names & _PROJECT_MARKERS)
        for entry in entries or []:
            ep = str(entry.get("path") or "")
            if os.path.dirname(ep) == parent and entry.get("name", "").lower() in _PROJECT_MARKERS:
                markers.add(entry["name"].lower())
        return markers


# ---------------------------------------------------------------------
# Persistent decisions
# ---------------------------------------------------------------------
@dataclass
class FileDecision:
    """A pending file decision that survives UI restarts and context turnover."""

    path: str
    proposed_action: str
    reasoning: str
    confidence: float
    importance: float
    disposition: str
    status: str = D_PENDING
    evidence: Dict[str, Any] = field(default_factory=dict)
    sensitive: bool = False
    id: str = field(default_factory=lambda: f"dec_{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    resolved_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> Optional["FileDecision"]:
        if not isinstance(data, dict):
            return None
        try:
            decision = cls(
                path=str(data.get("path") or ""),
                proposed_action=str(data.get("proposed_action") or ""),
                reasoning=str(data.get("reasoning") or ""),
                confidence=_clamp01(data.get("confidence")),
                importance=_clamp01(data.get("importance", 0.5), 0.5),
                disposition=str(data.get("disposition") or DISPOSITION_INSPECT),
                status=str(data.get("status") or D_PENDING),
                evidence=dict(data.get("evidence") or {}),
                sensitive=bool(data.get("sensitive")),
                id=str(data.get("id") or f"dec_{uuid.uuid4().hex[:12]}"),
                created_at=str(data.get("created_at") or _now()),
                updated_at=str(data.get("updated_at") or _now()),
                resolved_at=data.get("resolved_at"),
                result=data.get("result") if isinstance(data.get("result"), dict) else None,
                hash=data.get("hash"),
            )
        except (TypeError, ValueError):
            return None
        if decision.status not in DECISION_STATUSES:
            decision.status = D_PENDING
        return decision

    def public_label(self) -> str:
        """A label safe to show in conversation (sensitive files get less)."""
        if self.sensitive:
            return "(a private file)"
        return self.path


class DecisionStore:
    """Durable store of file decisions. Own file, own version, atomic writes."""

    def __init__(self, data_dir: str = "./storage") -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "file_decisions.json")
        self._lock = threading.RLock()
        self._decisions: List[FileDecision] = self._load()

    def _load(self) -> List[FileDecision]:
        raw = load_json(self.path, dict, dict)
        items = raw.get("decisions") if isinstance(raw, dict) else None
        out: List[FileDecision] = []
        for item in items or []:
            decision = FileDecision.from_dict(item)
            if decision is not None:
                out.append(decision)
        return out

    def _persist(self) -> None:
        with self._lock:
            atomic_save(self.path, {
                "version": DECISION_VERSION,
                "decisions": [d.to_dict() for d in self._decisions],
            })

    def add(self, decision: FileDecision) -> Optional[FileDecision]:
        """Add a decision unless an open one already exists for the same path."""
        if not isinstance(decision, FileDecision):
            return None
        with self._lock:
            for existing in self._decisions:
                if existing.path == decision.path and existing.status in OPEN_DECISION_STATUSES:
                    return existing
            self._decisions.append(decision)
            self._persist()
            return decision

    def all(self) -> List[FileDecision]:
        with self._lock:
            return list(self._decisions)

    def open(self) -> List[FileDecision]:
        with self._lock:
            return [d for d in self._decisions if d.status in OPEN_DECISION_STATUSES]

    def get(self, decision_id: str) -> Optional[FileDecision]:
        with self._lock:
            for d in self._decisions:
                if d.id == decision_id:
                    return d
        return None

    def update(self, decision_id: str, **fields: Any) -> Optional[FileDecision]:
        with self._lock:
            decision = self.get(decision_id)
            if decision is None:
                return None
            for key, value in fields.items():
                if hasattr(decision, key):
                    setattr(decision, key, value)
            decision.updated_at = _now()
            self._persist()
            return decision

    def set_status(self, decision_id: str, status: str, **fields: Any) -> Optional[FileDecision]:
        if status not in DECISION_STATUSES:
            raise ValueError(f"unknown decision status: {status!r}")
        if status in (D_COMPLETED, D_REJECTED, D_FAILED):
            fields.setdefault("resolved_at", _now())
        return self.update(decision_id, status=status, **fields)

    def counts(self) -> Dict[str, int]:
        out = {s: 0 for s in DECISION_STATUSES}
        with self._lock:
            for d in self._decisions:
                out[d.status] = out.get(d.status, 0) + 1
        return out

    def render_open(self, limit: int = 8) -> str:
        open_items = [d for d in self.open() if d.status == D_PENDING][:limit]
        if not open_items:
            return "No file decisions are waiting."
        lines = ["File decisions waiting for you:"]
        for d in open_items:
            action = d.proposed_action or d.disposition
            lines.append(
                f"  [{d.id}] {d.public_label()} -> {action} "
                f"(confidence {d.confidence:.2f}, importance {d.importance:.2f})"
            )
            if d.reasoning:
                lines.append(f"      why: {d.reasoning}")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# The background organizer
# ---------------------------------------------------------------------
class FileOrganizer:
    """One bounded cycle of assess-decide-act over the user's allowed roots."""

    def __init__(self, policy: filesystem.FsPolicy, ops: filesystem.FileOps,
                 decisions: DecisionStore, *, assessor: Optional[FileAssessor] = None,
                 user_roots: Iterable[str] = (), max_files_per_cycle: int = 40,
                 dry_run: bool = False) -> None:
        self.policy = policy
        self.ops = ops
        self.decisions = decisions
        self.assessor = assessor or FileAssessor(policy)
        self.user_roots = tuple(user_roots)
        self.max_files_per_cycle = max(1, int(max_files_per_cycle))
        self.dry_run = bool(dry_run)

    def recover(self) -> int:
        """On startup, resolve decisions left mid-flight (crash-safe).

        A decision stuck in ``executing`` is moved to ``failed`` (not silently
        completed), so an interrupted operation is never reported as success and
        is reconsidered rather than duplicated.
        """
        recovered = 0
        for decision in self.decisions.all():
            if decision.status == D_EXECUTING:
                self.decisions.set_status(
                    decision.id, D_FAILED,
                    result={"summary": "interrupted before completion"})
                recovered += 1
        return recovered

    def run_cycle(self, *, library: Any = None,
                  budget_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """One bounded, deterministic pass. Returns a small report."""
        scanned = 0
        autonomous: List[Dict[str, Any]] = []
        created: List[str] = []
        # Files already handled or awaiting a decision are not reconsidered, so
        # a routine file is not re-judged (and re-logged) every cycle.
        handled = self._handled_paths()
        for root in self.user_roots:
            if not os.path.isdir(root):
                continue
            result = self.ops.scan(root, max_entries=self.max_files_per_cycle)
            if result.get("outcome") != filesystem.OUTCOME_OK:
                continue
            entries = result.get("entries", [])
            dupes = self.ops.find_duplicates(entries)
            duplicate_paths = {p for group in dupes.get("duplicate_groups", [])
                               for p in group}
            for entry in entries:
                if scanned >= self.max_files_per_cycle:
                    break
                path = entry.get("path")
                if entry.get("is_dir") or not path or path in handled:
                    continue
                scanned += 1
                assessment = self.assessor.assess(
                    path, entries=entries,
                    duplicate=path in duplicate_paths)
                outcome = self._decide(path, assessment)
                handled.add(path)
                if outcome["action"] == "autonomous":
                    autonomous.append(outcome)
                elif outcome["action"] == "asked":
                    created.append(outcome["decision_id"])
            if scanned >= self.max_files_per_cycle:
                break
        return {
            "scanned": scanned,
            "autonomous": autonomous,
            "asked": created,
            "awaiting": len(self.decisions.open()),
        }

    def _handled_paths(self) -> set:
        """Paths already acted on or holding an open decision."""
        paths = set()
        for entry in self.ops.history():
            if entry.get("outcome") in (filesystem.OUTCOME_OK,
                                        filesystem.OUTCOME_DRY_RUN) and entry.get("path"):
                paths.add(entry["path"])
        for decision in self.decisions.all():
            if decision.path:
                paths.add(decision.path)
        return paths

    def _decide(self, path: str, assessment: Assessment) -> Dict[str, Any]:
        disposition = assessment.disposition
        # Destructive or ambiguous work is always a decision, never autonomous.
        destructive = disposition == DISPOSITION_TRASH
        routine = disposition in (DISPOSITION_KEEP, DISPOSITION_ARCHIVE,
                                  DISPOSITION_ORGANIZE)
        confident = assessment.confidence >= AUTONOMOUS_CONFIDENCE

        if routine and confident and not destructive:
            record = self._act(path, disposition, assessment)
            return {"action": "autonomous", "path": path,
                    "outcome": record.outcome, "operation": record.operation}

        # Everything else: park a persistent decision for the user.
        decision = FileDecision(
            path=path,
            proposed_action=disposition,
            reasoning=assessment.reason,
            confidence=assessment.confidence,
            importance=assessment.importance,
            disposition=disposition,
            evidence=assessment.evidence,
            sensitive=assessment.sensitive,
        )
        stored = self.decisions.add(decision)
        return {"action": "asked", "path": path,
                "decision_id": (stored or decision).id}

    def _act(self, path: str, disposition: str,
             assessment: Assessment) -> filesystem.OperationRecord:
        if disposition == DISPOSITION_KEEP:
            # "Keep" is a no-op; it is still recorded so the file is not
            # reconsidered forever.
            rec = filesystem.OperationRecord(
                filesystem.OP_ORGANIZE, filesystem.OUTCOME_OK, path=path,
                detail="kept (routine, high confidence)",
                reason=assessment.reason)
            self.ops._record(rec)
            return rec
        # Archive: move into an "Archive" sibling folder, non-destructively.
        dest = os.path.join(self._archive_root(path), os.path.basename(path))
        plan = filesystem.FilePlan(filesystem.OP_ARCHIVE, path, destination=dest,
                                   reason=assessment.reason,
                                   confidence=assessment.confidence,
                                   importance=assessment.importance)
        return self.ops.execute(plan, dry_run=self.dry_run)

    @staticmethod
    def _archive_root(path: str) -> str:
        return os.path.join(os.path.dirname(path), "Archive")

    # -- resolving decisions --------------------------------------------
    def resolve(self, decision_id: str, *, approve: bool = True,
                dry_run: Optional[bool] = None,
                authorized: bool = False) -> Dict[str, Any]:
        """Approve/reject a pending decision and carry out the operation.

        On approve, the decision moves pending -> executing -> completed/failed.
        The ``executing`` state is persisted *before* the side effect, so a crash
        mid-operation is recoverable (``recover`` turns it into ``failed``) and
        can never be mistaken for success.
        """
        decision = self.decisions.get(decision_id)
        if decision is None:
            return {"ok": False, "error": "unknown decision"}
        if not approve:
            self.decisions.set_status(decision_id, D_REJECTED)
            return {"ok": True, "status": D_REJECTED, "id": decision_id}

        self.decisions.set_status(decision_id, D_EXECUTING)
        dry = self.dry_run if dry_run is None else bool(dry_run)
        plan = filesystem.FilePlan(
            decision.proposed_action if decision.proposed_action in filesystem.OPERATION_KINDS
            else filesystem.OP_MOVE,
            decision.path,
            destination=decision.evidence.get("proposed_destination"),
            reason=decision.reasoning,
            confidence=decision.confidence,
            importance=decision.importance,
        )
        # A destructive proposed action needs the human authorization that the
        # approval itself represents, plus the policy flag.
        is_delete = decision.disposition == DISPOSITION_TRASH
        record = self.ops.execute(
            plan, dry_run=dry,
            authorized=bool(authorized or is_delete) and not dry,
        )
        status = {
            filesystem.OUTCOME_OK: D_COMPLETED,
            filesystem.OUTCOME_DRY_RUN: D_APPROVED,
            filesystem.OUTCOME_NEEDS_AUTHORIZATION: D_PENDING,
            filesystem.OUTCOME_BLOCKED: D_FAILED,
            filesystem.OUTCOME_FAILED: D_FAILED,
        }.get(record.outcome, D_FAILED)
        self.decisions.set_status(decision_id, status,
                                  result={"summary": record.summary(),
                                          "outcome": record.outcome,
                                          "operation": record.operation})
        return {"ok": record.outcome in (filesystem.OUTCOME_OK, filesystem.OUTCOME_DRY_RUN),
                "status": status, "id": decision_id,
                "outcome": record.outcome, "detail": record.detail}

    def defer(self, decision_id: str) -> Dict[str, Any]:
        decision = self.decisions.set_status(decision_id, D_DEFERRED)
        return {"ok": decision is not None, "id": decision_id}


def default_runtime_paths(*, config_dir: str = "./config",
                          storage_dir: str = "./storage",
                          extra: Iterable[str] = ()) -> List[str]:
    """Astra's own paths, derived from the actual runtime layout.

    These are protected from autonomous cleanup and are not treated as user
    files, so activity inside them cannot distort user-file importance.
    """
    paths = [
        os.path.abspath(config_dir),
        os.path.abspath(storage_dir),
        os.path.join(os.path.abspath(storage_dir), "library"),
        os.path.join(os.path.abspath(storage_dir), "books"),
    ]
    paths.extend(os.path.abspath(str(p)) for p in extra if str(p or "").strip())
    return paths
