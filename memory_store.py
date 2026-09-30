"""
Persistent storage for Astra: memories, journal, and conditional commands.

Design rules
------------
* On-disk format is unchanged: each store file is a plain JSON list of dicts.
  New fields are optional, so files written by older versions load fine.
* Reads never write. Loading a store, get_active_memories(), stats(), etc.
  leave every file byte-for-byte identical.
* Writes are atomic (temp file + fsync + replace) and keep one ".bak" of the
  previous good version. A corrupt file is quarantined and the backup is used.
* Every mutation is transactional: if saving fails, in-memory state is rolled
  back so RAM and disk never disagree.
* Mutations are guarded by a lock, so a background extraction thread can write
  while the chat loop reads.
"""
import contextlib
import copy
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("astra.memory")

VALID_TARGET_MODELS = {"roum", "self", "relationship"}
VALID_MEMORY_TYPES = {
    "explicit_fact", "explicit_preference", "explicit_project_information",
    "behavioral_pattern", "uncertain_inference", "self_fact",
    "self_observation", "self_belief", "self_preference",
    "relationship_event", "decision", "correction",
}
VALID_STATUSES = {"active", "archived", "superseded"}

MAX_COMMAND_HISTORY = 500
REINFORCE_STEP = 0.05          # confidence bump when a memory is re-learned
_STORE_MANAGED_FIELDS = {"id", "target_model", "content", "type", "source"}


class MemoryNotFoundError(LookupError):
    """Raised when a memory id does not exist in the requested model."""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    # Millisecond timestamp keeps ids roughly sortable; the random suffix
    # prevents collisions when several items are created in the same ms.
    ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return f"{prefix}_{ms}_{uuid.uuid4().hex[:6]}"


def _normalize(text: Any) -> str:
    return " ".join(str(text).lower().split())


def _clamp_confidence(value: Any, default: float = 1.0) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _clean_str_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out, seen = [], set()
    for v in values:
        text = " ".join(str(v).split())
        if text and text.lower() not in seen:
            out.append(text)
            seen.add(text.lower())
    return out


def _check_timestamp(value: Any) -> str:
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"timestamp must be ISO-8601, got {value!r}") from None
    return str(value)


# ---------------------------------------------------------------------
# Safe file IO
# ---------------------------------------------------------------------
def atomic_save(filepath: str, data: Any, backup: bool = True) -> None:
    """
    Save JSON atomically: write to a temp file, fsync, then replace.
    If `backup` is set, the previous version is kept as "<file>.bak".
    """
    filepath = os.path.abspath(filepath)
    dir_name = os.path.dirname(filepath)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=dir_name, prefix=os.path.basename(filepath) + ".", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        if backup and os.path.exists(filepath):
            shutil.copy2(filepath, filepath + ".bak")
        os.replace(tmp_path, filepath)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def _quarantine(path: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    target = f"{path}.corrupt-{stamp}"
    with contextlib.suppress(OSError):
        os.replace(path, target)
        logger.warning("Moved unreadable file %s to %s", path, target)


def load_json(path: str, expected_type: type, default) -> Any:
    """
    Load JSON from `path`, falling back to "<path>.bak" if the main file is
    corrupt. A corrupt main file is renamed aside (never deleted).
    Returns default() if nothing usable exists.
    """
    for candidate in (path, path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, expected_type):
                raise ValueError(
                    f"expected {expected_type.__name__}, got {type(data).__name__}"
                )
        except ValueError as exc:  # includes JSONDecodeError
            logger.warning("Corrupt data in %s: %s", candidate, exc)
            if candidate == path:
                _quarantine(path)
            continue
        except OSError as exc:
            logger.warning("Could not read %s: %s", candidate, exc)
            continue
        if candidate != path:
            logger.warning("Recovered %s from its backup.", path)
        return data
    return default()


def _load_dict_list(path: str) -> List[Dict[str, Any]]:
    data = load_json(path, list, list)
    valid = [item for item in data if isinstance(item, dict)]
    if len(valid) != len(data):
        logger.warning("Ignored %d non-object entries in %s", len(data) - len(valid), path)
    return valid


# ---------------------------------------------------------------------
# Conditional commands
# ---------------------------------------------------------------------
def _trigger_matches(trigger: str, text: str) -> bool:
    """Whole-word/phrase match, so trigger 'hi' does not fire on 'this'."""
    return re.search(rf"(?<!\w){re.escape(trigger)}(?!\w)", text) is not None


class CommandStore:
    """Independent structured storage for direct conditional commands and history."""

    def __init__(self, data_dir: str = "./storage"):
        self.commands_file = os.path.join(data_dir, "active_commands.json")
        self.history_file = os.path.join(data_dir, "command_history.json")
        self._lock = threading.RLock()

        self.active_commands = [
            c for c in _load_dict_list(self.commands_file)
            if isinstance(c.get("trigger"), str) and c.get("trigger")
            and isinstance(c.get("response"), str)
        ]
        for cmd in self.active_commands:
            cmd.setdefault("id", _new_id("cmd"))  # legacy commands; saved on next write
        self.history = _load_dict_list(self.history_file)

    def add_command(self, raw_text: str, trigger: str, response: str, once: bool = True) -> str:
        """Register a command and return its id (an identical active command is reused)."""
        trigger_n = _normalize(trigger)
        response_s = str(response).strip()
        if not trigger_n:
            raise ValueError("trigger must not be empty")
        if not response_s:
            raise ValueError("response must not be empty")

        with self._lock:
            for existing in self.active_commands:
                if (existing["trigger"] == trigger_n and existing["response"] == response_s
                        and bool(existing.get("once", True)) == bool(once)):
                    return existing["id"]

            cmd = {
                "id": _new_id("cmd"),
                "trigger": trigger_n,
                "response": response_s,
                "once": bool(once),
                "timestamp": _now(),
            }
            old_commands, old_history = list(self.active_commands), list(self.history)
            self.active_commands.append(cmd)
            self.history.append({
                "user_input": raw_text,
                "structured_command": cmd,
                "timestamp": cmd["timestamp"],
            })
            self.history = self.history[-MAX_COMMAND_HISTORY:]
            try:
                # Commands first: a command without a history row is harmless,
                # a history row for a command that never saved is not.
                atomic_save(self.commands_file, self.active_commands)
                atomic_save(self.history_file, self.history)
            except BaseException:
                self.active_commands, self.history = old_commands, old_history
                raise
            return cmd["id"]

    def check_trigger(self, text: str) -> Optional[str]:
        """Return the response of the most specific matching command, if any."""
        text_n = _normalize(text)
        if not text_n:
            return None
        with self._lock:
            best_idx = None
            for idx, cmd in enumerate(self.active_commands):
                if _trigger_matches(cmd["trigger"], text_n):
                    if best_idx is None or len(cmd["trigger"]) > len(self.active_commands[best_idx]["trigger"]):
                        best_idx = idx  # longest trigger wins ("good night" beats "night")
            if best_idx is None:
                return None

            cmd = self.active_commands[best_idx]
            if cmd.get("once", True):
                snapshot = list(self.active_commands)
                self.active_commands.pop(best_idx)
                try:
                    atomic_save(self.commands_file, self.active_commands)
                except OSError as exc:
                    # Still answer the user; the command simply stays armed.
                    logger.error("Could not persist command consumption: %s", exc)
                    self.active_commands = snapshot
            return cmd["response"]

    def list_commands(self) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self.active_commands)

    def remove_command(self, cmd_id: str) -> bool:
        with self._lock:
            remaining = [c for c in self.active_commands if c.get("id") != cmd_id]
            if len(remaining) == len(self.active_commands):
                return False
            snapshot, self.active_commands = self.active_commands, remaining
            try:
                atomic_save(self.commands_file, self.active_commands)
            except BaseException:
                self.active_commands = snapshot
                raise
            return True

    def get_last_commands_context(self, n: int = 3) -> List[Dict]:
        with self._lock:
            return copy.deepcopy(self.history[-n:]) if n > 0 else []


# ---------------------------------------------------------------------
# Triple memory store
# ---------------------------------------------------------------------
class TripleMemoryStore:
    def __init__(self, data_dir: str = "./storage"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.files = {
            "roum": os.path.join(data_dir, "roum_model.json"),
            "self": os.path.join(data_dir, "self_model.json"),
            "relationship": os.path.join(data_dir, "relationship_model.json"),
        }
        self.journal_file = os.path.join(data_dir, "ai_journal.json")
        self._lock = threading.RLock()

        self.memories = {name: self._load_file(path) for name, path in self.files.items()}
        self.journal = self._load_file(self.journal_file)

    # ---- internals ---------------------------------------------------
    def _load_file(self, path: str) -> List[Dict[str, Any]]:
        return _load_dict_list(path)

    @staticmethod
    def _check_target(target_model: str) -> None:
        if target_model not in VALID_TARGET_MODELS:
            raise ValueError(
                f"Unknown target_model {target_model!r}; expected one of {sorted(VALID_TARGET_MODELS)}"
            )

    @staticmethod
    def _check_type(mem_type: str) -> None:
        if mem_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"Unknown mem_type {mem_type!r}; expected one of {sorted(VALID_MEMORY_TYPES)}"
            )

    def _get_list(self, key: str) -> List[Dict[str, Any]]:
        return self.journal if key == "journal" else self.memories[key]

    def _set_list(self, key: str, value: List[Dict[str, Any]]) -> None:
        if key == "journal":
            self.journal = value
        else:
            self.memories[key] = value

    def _path(self, key: str) -> str:
        return self.journal_file if key == "journal" else self.files[key]

    def _persist(self, key: str) -> None:
        atomic_save(self._path(key), self._get_list(key))

    def _save_model(self, target_model: str) -> None:
        """Immediately persists a specific model upon change."""
        self._persist(target_model)

    @contextmanager
    def _transaction(self, key: str) -> Iterator[None]:
        """Mutate, then save; if anything fails, restore the previous state."""
        snapshot = copy.deepcopy(self._get_list(key))
        try:
            yield
            self._persist(key)
        except BaseException:
            self._set_list(key, snapshot)
            raise

    def _locate(self, target_model: str, mem_id: str):
        for idx, mem in enumerate(self.memories[target_model]):
            if mem.get("id") == mem_id:
                return idx, mem
        raise MemoryNotFoundError(f"No memory {mem_id!r} in model {target_model!r}")

    def _find_duplicate(self, target_model: str, mem_type: str, content: str):
        key = _normalize(content)
        for mem in self.memories[target_model]:
            if (mem.get("status") == "active" and mem.get("type") == mem_type
                    and _normalize(mem.get("content", "")) == key):
                return mem
        return None

    @staticmethod
    def _merge_list_field(mem: Dict[str, Any], field: str, values: List[str]) -> None:
        if values:
            mem[field] = _clean_str_list(list(mem.get(field) or []) + values)

    def _reinforce(self, mem: Dict[str, Any], keywords: List[str], tags: List[str]) -> None:
        mem["reinforcement_count"] = int(mem.get("reinforcement_count", 1)) + 1
        mem["last_reinforced"] = _now()
        mem["confidence"] = min(
            1.0, _clamp_confidence(mem.get("confidence", 1.0)) + REINFORCE_STEP
        )
        self._merge_list_field(mem, "keywords", keywords)
        self._merge_list_field(mem, "tags", tags)

    def _build_memory(self, target_model, content, mem_type, source,
                      keywords, tags, confidence, extra) -> Dict[str, Any]:
        memory: Dict[str, Any] = {
            "id": _new_id("mem"),
            "target_model": target_model,
            "content": content,
            "type": mem_type,
            "source": source,
            "status": "active",
            "timestamp": _now(),
            "confidence": _clamp_confidence(confidence),
        }
        if _clean_str_list(keywords):
            memory["keywords"] = _clean_str_list(keywords)
        if _clean_str_list(tags):
            memory["tags"] = _clean_str_list(tags)
        for field, value in extra.items():
            if field in _STORE_MANAGED_FIELDS:
                raise ValueError(f"'{field}' is managed by the store and cannot be overridden")
            if field == "status" and value not in VALID_STATUSES:
                raise ValueError(f"Invalid status {value!r}")
            if field == "timestamp":
                value = _check_timestamp(value)  # allows backdating imported memories
            memory[field] = value
        return memory

    @staticmethod
    def _clean_content(content: Any) -> str:
        text = str(content).strip() if content is not None else ""
        if not text:
            raise ValueError("content must not be empty")
        return text

    # ---- writing memories -------------------------------------------
    def add_memory(self, target_model: str, content: str, mem_type: str, source: str, *,
                   keywords: Optional[List[str]] = None, tags: Optional[List[str]] = None,
                   confidence: float = 1.0, dedupe: bool = True, **extra) -> str:
        """
        Store a memory and return its id.

        With dedupe=True (default), re-learning an identical active memory of
        the same type reinforces the existing one (higher confidence, count,
        merged keywords/tags) instead of adding a duplicate, and returns its id.
        """
        self._check_target(target_model)
        self._check_type(mem_type)
        content = self._clean_content(content)

        with self._lock:
            if dedupe:
                existing = self._find_duplicate(target_model, mem_type, content)
                if existing is not None:
                    with self._transaction(target_model):
                        self._reinforce(existing, _clean_str_list(keywords), _clean_str_list(tags))
                    return existing["id"]

            memory = self._build_memory(target_model, content, mem_type, source,
                                        keywords, tags, confidence, extra)
            with self._transaction(target_model):
                self.memories[target_model].append(memory)
            return memory["id"]

    def update_memory(self, target_model: str, mem_id: str, *, content: Optional[str] = None,
                      confidence: Optional[float] = None, keywords: Optional[List[str]] = None,
                      tags: Optional[List[str]] = None, mem_type: Optional[str] = None) -> Dict[str, Any]:
        """Edit selected fields of a memory; returns a copy of the updated memory."""
        self._check_target(target_model)
        if all(v is None for v in (content, confidence, keywords, tags, mem_type)):
            raise ValueError("Nothing to update")
        if mem_type is not None:
            self._check_type(mem_type)
        if content is not None:
            content = self._clean_content(content)

        with self._lock:
            _, mem = self._locate(target_model, mem_id)
            with self._transaction(target_model):
                if content is not None:
                    mem["content"] = content
                if confidence is not None:
                    mem["confidence"] = _clamp_confidence(confidence, mem.get("confidence", 1.0))
                if keywords is not None:
                    mem["keywords"] = _clean_str_list(keywords)
                if tags is not None:
                    mem["tags"] = _clean_str_list(tags)
                if mem_type is not None:
                    mem["type"] = mem_type
                mem["updated_at"] = _now()
            return copy.deepcopy(mem)

    def archive_memory(self, target_model: str, mem_id: str, reason: Optional[str] = None) -> None:
        """Retire a memory without deleting it. Archived memories are never injected into prompts."""
        self._check_target(target_model)
        with self._lock:
            _, mem = self._locate(target_model, mem_id)
            if mem.get("status") == "archived":
                return
            with self._transaction(target_model):
                mem["status"] = "archived"
                mem["archived_at"] = _now()
                if reason:
                    mem["archive_reason"] = str(reason).strip()

    def restore_memory(self, target_model: str, mem_id: str) -> None:
        """Re-activate an archived memory."""
        self._check_target(target_model)
        with self._lock:
            _, mem = self._locate(target_model, mem_id)
            if mem.get("status") == "active":
                return
            if mem.get("status") != "archived":
                raise ValueError("Only archived memories can be restored")
            with self._transaction(target_model):
                mem["status"] = "active"
                mem["restored_at"] = _now()

    def supersede_memory(self, target_model: str, old_id: str, content: str, *,
                         mem_type: Optional[str] = None, source: str = "supersession",
                         keywords: Optional[List[str]] = None, tags: Optional[List[str]] = None,
                         confidence: float = 1.0, **extra) -> str:
        """
        Replace an active memory with a corrected one while keeping the history:
        the old memory is marked "superseded" and linked to the new one.
        """
        self._check_target(target_model)
        content = self._clean_content(content)

        with self._lock:
            _, old = self._locate(target_model, old_id)
            if old.get("status") != "active":
                raise ValueError("Only active memories can be superseded")
            new_type = mem_type or old.get("type")
            self._check_type(new_type)

            new_mem = self._build_memory(target_model, content, new_type, source,
                                         keywords, tags, confidence, extra)
            new_mem["supersedes"] = old_id
            with self._transaction(target_model):
                old["status"] = "superseded"
                old["superseded_by"] = new_mem["id"]
                old["superseded_at"] = _now()
                self.memories[target_model].append(new_mem)
            return new_mem["id"]

    # ---- reading memories (never write) -------------------------------
    def get_memory(self, target_model: str, mem_id: str) -> Dict[str, Any]:
        self._check_target(target_model)
        with self._lock:
            return copy.deepcopy(self._locate(target_model, mem_id)[1])

    def get_memories(self, target_model: str, status: Optional[str] = "active",
                     mem_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Copies of memories filtered by status (None = all) and optionally type."""
        self._check_target(target_model)
        with self._lock:
            return [
                copy.deepcopy(m) for m in self.memories[target_model]
                if (status is None or m.get("status") == status)
                and (mem_type is None or m.get("type") == mem_type)
            ]

    def get_active_memories(self, target_model: str) -> List[Dict[str, Any]]:
        return self.get_memories(target_model, status="active")

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            result: Dict[str, Any] = {}
            for name in sorted(VALID_TARGET_MODELS):
                items = self.memories[name]
                active = [m for m in items if m.get("status") == "active"]
                result[name] = {
                    "total": len(items),
                    "active": len(active),
                    "by_type": dict(Counter(m.get("type", "unknown") for m in active)),
                }
            result["journal_entries"] = len(self.journal)
            return result

    # ---- journal ------------------------------------------------------
    def add_journal_entry(self, title: str, observation: str, trigger_conv: str) -> str:
        title_s = str(title).strip() if title is not None else ""
        obs_s = str(observation).strip() if observation is not None else ""
        if not title_s or not obs_s:
            raise ValueError("Journal entries need a title and an observation")
        entry = {
            "id": _new_id("jnl"),
            "title": title_s,
            "observation": obs_s,
            "trigger_conv": trigger_conv,
            "timestamp": _now(),
        }
        with self._lock:
            with self._transaction("journal"):
                self.journal.append(entry)
        return entry["id"]

    def get_journal(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Journal entries, oldest first; with `limit`, only the most recent N."""
        with self._lock:
            items = self.journal[-limit:] if limit else self.journal
            return copy.deepcopy(items)

    # ---- export -------------------------------------------------------
    def export_snapshot(self, path: str) -> None:
        """Write a single combined JSON snapshot of all memories and the journal."""
        with self._lock:
            payload = {
                "exported_at": _now(),
                "memories": copy.deepcopy(self.memories),
                "journal": copy.deepcopy(self.journal),
            }
        atomic_save(path, payload, backup=False)
