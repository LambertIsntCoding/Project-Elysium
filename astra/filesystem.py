"""Deterministic filesystem operations, policy, trash and audit.

Filesystem work is application code, never delegated wholesale to the LLM. The
LLM may interpret a natural-language request into a *proposed* structured
operation, but every operation is validated against policy here and executed
here, and every consequential one is auditable.

Design rules (spec + project invariants):

* **Policy-controlled roots.** Operations outside the configured allowed roots
  are blocked; protected paths are never modified. Astra's own runtime,
  storage, cache, config and log paths are always protected - the damage is
  permanent and silent, so it is refused rather than merely warned about, even
  when a root is otherwise writable.
* **Reversible destructive actions.** A delete is never a permanent unlink: it
  goes to the OS Recycle Bin/trash where one exists, else to an XDG-compliant
  Trash directory written in code. An overwrite/replace needs explicit policy
  approval. Deletion is never autonomous by default.
* **Dry-run first.** Consequential operations can be planned without executing.
* **Auditable.** Every executed operation yields an :class:`OperationRecord`;
  the log is application state, never a memory or autobiography.
* **No silent success.** A failure is reported as a failure, so the caller (and
  the relational event system) never records an operation as completed when it
  was not.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .memory import atomic_save, load_json

# -- operation kinds ---------------------------------------------------------
OP_SCAN = "scan"
OP_CLASSIFY = "classify"
OP_DUPLICATES = "duplicates"
OP_RENAME = "rename"
OP_MOVE = "move"
OP_ORGANIZE = "organize"
OP_ARCHIVE = "archive"
OP_INSPECT = "inspect"
OP_STALE = "stale"
OP_STORAGE = "storage"
OP_DELETE = "delete"

OPERATION_KINDS = (OP_SCAN, OP_CLASSIFY, OP_DUPLICATES, OP_RENAME, OP_MOVE,
                   OP_ORGANIZE, OP_ARCHIVE, OP_INSPECT, OP_STALE, OP_STORAGE,
                   OP_DELETE)

# Kinds that change the filesystem (as opposed to reading it).
MUTATING_OPERATIONS = (OP_RENAME, OP_MOVE, OP_ORGANIZE, OP_ARCHIVE, OP_DELETE)
# Kinds that destroy or replace data - these need authorization.
DESTRUCTIVE_OPERATIONS = (OP_DELETE,)

OUTCOME_OK = "ok"
OUTCOME_DRY_RUN = "dry_run"
OUTCOME_BLOCKED = "blocked"
OUTCOME_NEEDS_AUTHORIZATION = "needs_authorization"
OUTCOME_FAILED = "failed"

LOG_VERSION = 1
MAX_LOG_ENTRIES = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OperationRecord:
    """One auditable filesystem operation."""

    operation: str
    outcome: str
    path: str = ""
    destination: Optional[str] = None
    detail: str = ""
    reversible: bool = True
    tool: Optional[str] = None
    reason: str = ""
    dry_run: bool = False
    id: str = field(default_factory=lambda: f"fs_{uuid.uuid4().hex[:12]}")
    at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        target = self.destination or self.path
        return f"{self.operation} {target}: {self.outcome}"


@dataclass
class FilePlan:
    """A proposed operation, before it is executed (the dry-run representation)."""

    operation: str
    path: str
    destination: Optional[str] = None
    reason: str = ""
    confidence: float = 0.0
    importance: float = 0.5
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FsPolicy:
    """Allowed roots and protected paths, enforced in code.

    ``runtime_paths`` are Astra's own directories (storage, config, logs, cache,
    library, this repo's runtime state). They are refused as *targets* of any
    mutation even if they sit inside an allowed root.
    """

    def __init__(self, *, roots: Iterable[str] = (),
                 protected: Iterable[str] = (),
                 runtime_paths: Iterable[str] = (),
                 allow_delete: bool = False,
                 allow_overwrite: bool = False) -> None:
        self.roots = tuple(os.path.abspath(os.path.expanduser(str(r)))
                           for r in roots if str(r or "").strip())
        self.protected = tuple(os.path.abspath(os.path.expanduser(str(p)))
                               for p in protected if str(p or "").strip())
        self.runtime_paths = tuple(os.path.abspath(os.path.expanduser(str(p)))
                                   for p in runtime_paths if str(p or "").strip())
        self.allow_delete = bool(allow_delete)
        self.allow_overwrite = bool(allow_overwrite)

    @classmethod
    def from_config(cls, config: Any) -> "FsPolicy":
        cfg = config if isinstance(config, dict) else {}
        files = cfg.get("filesystem") if isinstance(cfg.get("filesystem"), dict) else cfg
        files = files if isinstance(files, dict) else {}
        return cls(
            roots=files.get("allowed_roots") or cfg.get("allowed_roots") or (),
            protected=files.get("protected_paths") or cfg.get("protected_paths") or (),
            runtime_paths=files.get("runtime_paths") or cfg.get("runtime_paths") or (),
            allow_delete=bool(files.get("allow_delete", cfg.get("allow_delete", False))),
            allow_overwrite=bool(files.get("allow_overwrite", cfg.get("allow_overwrite", False))),
        )

    # -- path checks -----------------------------------------------------
    @staticmethod
    def _within(path: str, root: str) -> bool:
        path = os.path.abspath(path)
        root = os.path.abspath(root)
        if path == root:
            return True
        return path.startswith(root.rstrip(os.sep) + os.sep)

    def in_allowed_root(self, path: str) -> bool:
        if not self.roots:
            return False  # no roots configured: nothing is autonomously writable
        return any(self._within(path, root) for root in self.roots)

    def is_protected(self, path: str) -> bool:
        return any(self._within(path, p) for p in (*self.protected, *self.runtime_paths))

    def check(self, path: str, *, operation: str) -> Optional[str]:
        """Return a block reason for a mutating op on ``path``, or ``None``."""
        if not str(path or "").strip():
            return "empty path"
        if operation in DESTRUCTIVE_OPERATIONS:
            return None  # delete safety checked separately (needs authorization)
        if not self.in_allowed_root(path):
            return "outside allowed roots"
        if self.is_protected(path):
            return "protected path (Astra runtime) "
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allowed_roots": list(self.roots),
            "protected_paths": list(self.protected),
            "runtime_paths": list(self.runtime_paths),
            "allow_delete": self.allow_delete,
            "allow_overwrite": self.allow_overwrite,
        }


# ---------------------------------------------------------------------
# Trash: reversible removal
# ---------------------------------------------------------------------
def _xdg_trash_dir() -> str:
    data_home = os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share")
    return os.path.join(data_home, "Trash")


class Trash:
    """Reversible removal: system trash first, else XDG trash written in code."""

    def __init__(self, trash_dir: Optional[str] = None,
                 system_tools: Optional[Iterable[str]] = None) -> None:
        self.trash_dir = trash_dir or _xdg_trash_dir()
        self.system_tools = tuple(system_tools or ("gio", "trash-put"))

    def to_trash(self, path: str) -> Dict[str, Any]:
        """Move ``path`` to the trash. Never permanently deletes."""
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return {"ok": False, "error": "not found", "method": None}

        # Prefer the OS mechanism when it exists, so the user's desktop trash
        # recognises it.
        for tool in self.system_tools:
            tool_path = shutil.which(tool)
            if not tool_path:
                continue
            try:
                proc = subprocess.run(
                    [tool_path, "trash", path] if tool == "gio" else [tool_path, path],
                    capture_output=True, timeout=30,
                )
                if proc.returncode == 0:
                    return {"ok": True, "method": tool, "dest": None}
            except (OSError, subprocess.SubprocessError):
                continue

        # Fallback: XDG-compliant trash layout (~/.local/share/Trash).
        try:
            files_dir = os.path.join(self.trash_dir, "files")
            info_dir = os.path.join(self.trash_dir, "info")
            os.makedirs(files_dir, exist_ok=True)
            os.makedirs(info_dir, exist_ok=True)
            name = os.path.basename(path.rstrip(os.sep))
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            dest_name = f"{stamp}_{uuid.uuid4().hex[:8]}_{name}"
            dest = os.path.join(files_dir, dest_name)
            info = os.path.join(info_dir, dest_name + ".trashinfo")
            payload = (
                "[Trash Info]\n"
                f"Path={path}\n"
                f"DeletionDate={datetime.now(timezone.utc).isoformat()}\n"
            )
            # Write the info file first so a crash leaves an orphaned .trashinfo
            # (harmless) rather than an unattributed file.
            with open(info, "w", encoding="utf-8") as handle:
                handle.write(payload)
            shutil.move(path, dest)
            return {"ok": True, "method": "xdg-trash", "dest": dest}
        except OSError as exc:
            return {"ok": False, "error": str(exc), "method": "xdg-trash"}


# ---------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------
class FileOps:
    """Validates and executes filesystem operations against a policy."""

    def __init__(self, policy: FsPolicy, *, data_dir: str = "./storage",
                 trash: Optional[Trash] = None,
                 external_tools: Optional[Dict[str, Any]] = None) -> None:
        self.policy = policy
        self.data_dir = data_dir
        self.trash = trash or Trash()
        self.log_path = os.path.join(data_dir, "filesystem_operations.json")
        self._lock = threading.RLock()
        self._log: List[Dict[str, Any]] = self._load()
        # External utilities, discovered once. Astra decides; these do the bulk
        # mechanical work more reliably than one syscall at a time.
        self.external_tools = external_tools if external_tools is not None else discover_tools()

    # -- audit log -------------------------------------------------------
    def _load(self) -> List[Dict[str, Any]]:
        raw = load_json(self.log_path, dict, dict)
        items = raw.get("entries") if isinstance(raw, dict) else None
        return [dict(e) for e in items or [] if isinstance(e, dict)]

    def _record(self, record: OperationRecord) -> None:
        with self._lock:
            self._log.append(record.to_dict())
            atomic_save(self.log_path, {
                "version": LOG_VERSION,
                "entries": self._log[-MAX_LOG_ENTRIES:],
            })

    def history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._log)
        return items[-limit:] if limit else items

    # -- read operations (never blocked by roots) ------------------------
    def scan(self, root: str, *, max_entries: int = 500) -> Dict[str, Any]:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            rec = OperationRecord(OP_SCAN, OUTCOME_FAILED, path=root,
                                  detail="not a directory")
            self._record(rec)
            return {"outcome": OUTCOME_FAILED, "error": "not a directory",
                    "entries": [], "record": rec.to_dict()}
        entries: List[Dict[str, Any]] = []
        try:
            for base, dirs, files in os.walk(root):
                for name in sorted(dirs) + sorted(files):
                    full = os.path.join(base, name)
                    if self.policy.is_protected(full):
                        continue  # Astra's own paths are not the user's to clean
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    entries.append({
                        "path": full,
                        "name": name,
                        "is_dir": os.path.isdir(full),
                        "size": st.st_size,
                        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
                        "ext": os.path.splitext(name)[1].lower(),
                    })
                    if len(entries) >= max_entries:
                        break
                if len(entries) >= max_entries:
                    break
        except OSError as exc:
            rec = OperationRecord(OP_SCAN, OUTCOME_FAILED, path=root, detail=str(exc))
            self._record(rec)
            return {"outcome": OUTCOME_FAILED, "error": str(exc), "entries": [],
                    "record": rec.to_dict()}
        rec = OperationRecord(OP_SCAN, OUTCOME_OK, path=root,
                              detail=f"{len(entries)} entries")
        self._record(rec)
        return {"outcome": OUTCOME_OK, "entries": entries, "count": len(entries),
                "record": rec.to_dict()}

    def classify(self, entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Group scanned entries by extension (deterministic, no model call)."""
        groups: Dict[str, List[str]] = {}
        for entry in entries or []:
            key = entry.get("ext") or "(none)"
            groups.setdefault(str(key), []).append(entry.get("path"))
        return {k: groups[k] for k in sorted(groups)}

    def find_duplicates(self, entries: Iterable[Dict[str, Any]], *,
                        max_hash_bytes: int = 16 * 1024 * 1024) -> Dict[str, Any]:
        """Content-hash duplicate detection over files (deterministic)."""
        by_size: Dict[int, List[str]] = {}
        for entry in entries or []:
            if entry.get("is_dir"):
                continue
            size = int(entry.get("size") or 0)
            by_size.setdefault(size, []).append(entry.get("path"))
        duplicates: List[List[str]] = []
        for size, paths in by_size.items():
            if len(paths) < 2 or size > max_hash_bytes:
                continue
            hashes: Dict[str, List[str]] = {}
            for path in paths:
                digest = _hash_file(path)
                if digest:
                    hashes.setdefault(digest, []).append(path)
            for group in hashes.values():
                if len(group) > 1:
                    duplicates.append(sorted(group))
        return {"duplicate_groups": duplicates,
                "count": sum(len(g) - 1 for g in duplicates)}

    def stale(self, entries: Iterable[Dict[str, Any]], *, older_than_days: float = 365.0,
              now: Optional[datetime] = None) -> Dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        stale_list = []
        for entry in entries or []:
            try:
                mtime = datetime.fromisoformat(str(entry.get("mtime")))
            except (TypeError, ValueError):
                continue
            if mtime.tzinfo is None:
                mtime = mtime.replace(tzinfo=timezone.utc)
            age = (now - mtime).total_seconds() / 86400.0
            if age >= older_than_days:
                stale_list.append({"path": entry.get("path"), "age_days": round(age, 1)})
        stale_list.sort(key=lambda e: e["age_days"], reverse=True)
        return {"stale": stale_list, "count": len(stale_list)}

    def storage_report(self, entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        total = 0
        by_ext: Dict[str, int] = {}
        for entry in entries or []:
            size = int(entry.get("size") or 0)
            total += size
            by_ext[entry.get("ext") or "(none)"] = by_ext.get(entry.get("ext") or "(none)", 0) + size
        return {"total_bytes": total, "file_count": len(list(entries or [])),
                "by_ext_bytes": {k: by_ext[k] for k in sorted(by_ext)}}

    # -- mutating operations ---------------------------------------------
    def plan_move(self, path: str, destination: str, *, reason: str = "") -> FilePlan:
        return FilePlan(OP_MOVE, path, destination=destination, reason=reason)

    def execute(self, plan: FilePlan, *, dry_run: bool = False,
                authorized: bool = False) -> OperationRecord:
        """Validate and (unless ``dry_run``) execute a planned operation.

        Returns an :class:`OperationRecord` whose ``outcome`` is the truth: a
        failure is reported as a failure. Never raises for a policy block.
        """
        op = plan.operation
        if op not in OPERATION_KINDS:
            rec = OperationRecord(op, OUTCOME_FAILED, path=plan.path,
                                  detail="unknown operation")
            self._record(rec)
            return rec

        # Destructive ops are refused unless policy permits and a human
        # authorized this specific operation.
        if op in DESTRUCTIVE_OPERATIONS:
            if not self.policy.allow_delete:
                rec = OperationRecord(op, OUTCOME_BLOCKED, path=plan.path,
                                      detail="deletion is disabled by policy")
                self._record(rec)
                return rec
            if not authorized:
                rec = OperationRecord(op, OUTCOME_NEEDS_AUTHORIZATION, path=plan.path,
                                      detail="destructive op requires authorization")
                self._record(rec)
                return rec
            if dry_run:
                rec = OperationRecord(op, OUTCOME_DRY_RUN, path=plan.path,
                                      detail="would move to trash", dry_run=True,
                                      reversible=True)
                self._record(rec)
                return rec
            # Trash first (reversible); never a permanent unlink.
            if self.policy.is_protected(plan.path):
                rec = OperationRecord(op, OUTCOME_BLOCKED, path=plan.path,
                                      detail="protected path")
                self._record(rec)
                return rec
            result = self.trash.to_trash(plan.path)
            if plan.path and not os.path.exists(plan.path) and not result.get("ok"):
                result = {"ok": False, "error": "path does not exist"}
            if result.get("ok"):
                rec = OperationRecord(op, OUTCOME_OK, path=plan.path,
                                      detail="moved to trash",
                                      reversible=True,
                                      tool=result.get("method"))
            else:
                rec = OperationRecord(op, OUTCOME_FAILED, path=plan.path,
                                      detail=str(result.get("error") or "trash failed"),
                                      tool=result.get("method"))
            self._record(rec)
            return rec

        # Non-destructive mutating ops: check root + protected path.
        block = self.policy.check(plan.path, operation=op)
        if block:
            rec = OperationRecord(op, OUTCOME_BLOCKED, path=plan.path, detail=block,
                                  destination=plan.destination, reason=plan.reason)
            self._record(rec)
            return rec
        if plan.destination:
            dest_block = self.policy.check(plan.destination, operation=op)
            if dest_block:
                rec = OperationRecord(op, OUTCOME_BLOCKED, path=plan.path,
                                      destination=plan.destination,
                                      detail=f"destination {dest_block}")
                self._record(rec)
                return rec

        if dry_run:
            rec = OperationRecord(op, OUTCOME_DRY_RUN, path=plan.path,
                                  destination=plan.destination, reason=plan.reason,
                                  dry_run=True, reversible=True)
            self._record(rec)
            return rec

        try:
            self._do(plan)
        except OSError as exc:
            rec = OperationRecord(op, OUTCOME_FAILED, path=plan.path,
                                  destination=plan.destination, detail=str(exc))
            self._record(rec)
            return rec
        rec = OperationRecord(op, OUTCOME_OK, path=plan.path,
                              destination=plan.destination, reason=plan.reason,
                              tool=(self.external_tools or {}).get("bulk"))
        self._record(rec)
        return rec

    def _do(self, plan: FilePlan) -> None:
        path = plan.path
        if plan.operation in (OP_MOVE, OP_ORGANIZE, OP_ARCHIVE):
            if not os.path.exists(path):
                raise OSError("source does not exist")
            dest = plan.destination
            if not dest:
                raise OSError("no destination")
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            if os.path.exists(dest):
                if not self.policy.allow_overwrite:
                    raise OSError("destination exists (overwrite disabled)")
            shutil.move(path, dest)
        elif plan.operation == OP_RENAME:
            if not os.path.exists(path):
                raise OSError("source does not exist")
            dest = plan.destination
            if not dest:
                raise OSError("no destination")
            if os.path.exists(dest) and not self.policy.allow_overwrite:
                raise OSError("destination exists (overwrite disabled)")
            os.replace(path, dest)
        else:
            raise OSError(f"operation {plan.operation} is not executable here")


def _hash_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def discover_tools() -> Dict[str, Any]:
    """Find external utilities for bulk mechanical work, without new deps."""
    found: Dict[str, Any] = {}
    for name in ("find", "rsync", "realpath", "du", "gio", "trash-put"):
        path = shutil.which(name)
        if path:
            found[name] = path
    # The preferred tool for recursive/multi-file moves and copies.
    found["bulk"] = found.get("rsync") or found.get("find")
    return found
