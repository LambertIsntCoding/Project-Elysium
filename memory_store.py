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
from typing import Any, Dict, Iterator, List, Optional, Tuple

import memory_authority as _authority

logger = logging.getLogger("astra.memory")

VALID_TARGET_MODELS = {"roum", "self", "relationship"}
VALID_MEMORY_TYPES = {
    "explicit_fact", "explicit_preference", "explicit_project_information",
    "behavioral_pattern", "uncertain_inference", "self_fact",
    "self_observation", "self_belief", "self_preference",
    "relationship_event", "relationship_observation", "relationship_boundary",
    "decision", "correction",
}
# "weakened" joins the existing statuses: a memory contradicted by a stronger
# one stays readable but is no longer authoritative.
VALID_STATUSES = {"active", "weakened", "archived", "superseded"}

MAX_COMMAND_HISTORY = 500
REINFORCE_STEP = 0.05          # confidence bump when a memory is re-learned
WEAKEN_FACTOR = 0.6            # multiplier applied to the weaker side of a contradiction
_STORE_MANAGED_FIELDS = {"id", "target_model", "content", "type", "source"}

# How many times one prompt build may record usage / apply decay before it
# starts throttling, so a tight loop can't rewrite the store every turn.
DEFAULT_MAINTENANCE_EVERY = 1
DEFAULT_DECAY_EVERY = 5


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


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _days_since(value: Any, now: Optional[datetime] = None) -> Optional[float]:
    ts = _parse_iso(value)
    if ts is None:
        return None
    return max(0.0, ((now or datetime.now(timezone.utc)) - ts).total_seconds() / 86400.0)


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
    def __init__(self, data_dir: str = "./storage", *,
                 maintenance_every: int = DEFAULT_MAINTENANCE_EVERY,
                 decay_every: int = DEFAULT_DECAY_EVERY):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.files = {
            "roum": os.path.join(data_dir, "roum_model.json"),
            "self": os.path.join(data_dir, "self_model.json"),
            "relationship": os.path.join(data_dir, "relationship_model.json"),
        }
        self.journal_file = os.path.join(data_dir, "ai_journal.json")
        self._lock = threading.RLock()
        # Maintenance throttles: usage recording / decay run on the prompt path
        # (a read that must sometimes write) but not on every single turn.
        self.maintenance_every = max(1, int(maintenance_every))
        self.decay_every = max(1, int(decay_every))
        self._maintenance_counter = 0

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
        now = _now()
        memory: Dict[str, Any] = {
            "id": _new_id("mem"),
            "target_model": target_model,
            "content": content,
            "type": mem_type,
            "source": source,
            "status": "active",
            "timestamp": now,
            "confidence": _clamp_confidence(confidence),
            # --- reliability metadata (sections 1, 4, 13) ----------------
            # All optional on disk: a memory written before this change simply
            # loads without them and the readers fall back to safe defaults.
            # ``importance`` is only stored when given; otherwise it is derived
            # from the memory type at read time.
            "created_at": now,
            "last_used": now,
            "use_count": 0,
            "contradiction_count": 0,
            "contradicts": [],
            "supersedes": None,
            "superseded_by": None,
            "source_type": _authority.source_tier(source),
            "reinforcement_count": 1,
            "last_reinforced": now,
        }
        importance = extra.pop("importance", None)
        if importance is not None:
            memory["importance"] = _clamp_confidence(importance, 0.5)
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
                # A freshly created memory's lifetime starts at its timestamp.
                memory["timestamp"] = value
                memory["created_at"] = value
                memory["last_used"] = value
                memory["last_reinforced"] = value
                continue
            memory[field] = value
        # Provenance guard (section 14): Astra's own inference can never be
        # stored as a confirmed user fact.
        if _authority.is_astra_source(source) and memory.get("type") in {
            "explicit_fact", "explicit_preference", "explicit_project_information",
        }:
            memory["type"] = "uncertain_inference"
        # Relationship guard (sections 8-10): ordinary affection must never
        # become a romantic instruction.
        _authority.apply_relationship_guard(memory)
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
                self._resolve_contradictions(target_model, memory)
            return memory["id"]

    # ---- contradiction resolution & decay (sections 3, 4, 5, 13) ------
    def _authority_key(self, mem: Dict[str, Any]) -> tuple:
        """Sort key: source tier first, then effective strength, then recency.

        Tier-first means an explicit user statement always beats an inference,
        even a fresh, confident one (section 2).
        """
        return (
            _authority.source_tier(mem.get("source")),
            _authority.effective_strength(mem),
            str(mem.get("last_used") or mem.get("timestamp") or ""),
        )

    def _resolve_contradictions(self, target_model: str, new_mem: Dict[str, Any]) -> List[str]:
        """Weaken older memories the new one contradicts. Never deletes them.

        Returns the ids of the memories that were weakened. The weaker side
        keeps its content (history is preserved) but loses authority, gains a
        ``contradiction_reason`` and a ``superseded_by`` pointer.
        """
        weakened: List[str] = []
        for old in self.memories[target_model]:
            if old is new_mem or old.get("status") in ("superseded", "archived"):
                continue
            if old.get("id") == new_mem.get("id"):
                continue
            reason = _authority.detect_contradiction(old, new_mem)
            if not reason:
                continue

            old_key, new_key = self._authority_key(old), self._authority_key(new_mem)
            loser, winner = (old, new_mem) if old_key < new_key else (new_mem, old)

            loser["contradiction_count"] = int(loser.get("contradiction_count", 0)) + 1
            loser["confidence"] = round(
                _clamp_confidence(loser.get("confidence", 1.0)) * WEAKEN_FACTOR, 4
            )
            loser["contradicts"] = _clean_str_list(
                list(loser.get("contradicts") or []) + [winner.get("id")]
            )
            loser["contradiction_reason"] = (
                f"Contradicted by {winner.get('id')}: {reason}"
            )
            if loser is old:
                loser["status"] = "weakened"
                loser["superseded_by"] = winner.get("id")
                loser["superseded_at"] = _now()
                weakened.append(old["id"])
                winner["supersedes"] = loser.get("id")
            else:
                # The new memory is the weaker side: keep it, but not as a fact.
                loser["status"] = "weakened"
        return weakened

    def record_use(self, mem_ids: List[str]) -> None:
        """Mark memories as retrieved (``last_used``, ``use_count``).

        This is what makes decay depend on *use* rather than mere age: a memory
        that keeps being retrieved keeps its recency factor.
        """
        ids = {i for i in (mem_ids or []) if i}
        if not ids:
            return
        for target_model in sorted(VALID_TARGET_MODELS):
            touched = [m for m in self.memories[target_model] if m.get("id") in ids]
            if not touched:
                continue
            with self._transaction(target_model):
                for mem in touched:
                    mem["last_used"] = _now()
                    mem["use_count"] = int(mem.get("use_count", 0)) + 1

    def apply_decay(self, now: Optional[datetime] = None) -> Dict[str, int]:
        """Recompute decay for every memory and archive the clearly worthless.

        A memory is archived only when it is low-value (low confidence, low
        importance, not an explicit user fact, not a governing boundary, not an
        active contradiction) AND has gone unused for a long time. Age alone is
        never sufficient. Superseded memories are left alone - they are already
        history and their decay factor is applied at read time.
        """
        now = now or datetime.now(timezone.utc)
        archived = 0
        for target_model in sorted(VALID_TARGET_MODELS):
            with self._transaction(target_model):
                for mem in self.memories[target_model]:
                    status = mem.get("status")
                    if status not in ("active", "weakened"):
                        continue
                    strength = _authority.effective_strength(mem, now)
                    mem["effective_strength"] = round(strength, 4)

                    if _authority.is_governing_eligible(mem):
                        continue  # core boundaries never decay into worthlessness

                    age = _days_since(mem.get("created_at") or mem.get("timestamp"), now)
                    unused = _days_since(mem.get("last_used") or mem.get("timestamp"), now)
                    if age is None or unused is None:
                        continue
                    # Archiving needs ALL of: weak, low confidence, old, unused.
                    # Age alone is never enough, and nothing is deleted outright.
                    if (strength < _authority.ARCHIVE_STRENGTH_THRESHOLD
                            and _clamp_confidence(mem.get("confidence"), 0.0) < _authority.ARCHIVE_CONFIDENCE_THRESHOLD
                            and age >= _authority.ARCHIVE_MIN_AGE_DAYS
                            and unused >= _authority.ARCHIVE_UNUSED_DAYS):
                        mem["status"] = "archived"
                        mem["archived_at"] = _now()
                        mem["archive_reason"] = (
                            f"decayed (strength={strength:.3f}, unused={unused:.0f}d)"
                        )
                        archived += 1
        return {"archived": archived}

    def maybe_maintain(self, *, force: bool = False) -> bool:
        """Run periodic maintenance (decay + archival) on the prompt path.

        Throttled by ``decay_every`` so it does not rewrite files every turn.
        Returns True when maintenance actually ran.
        """
        self._maintenance_counter += 1
        if not force and (self._maintenance_counter % self.decay_every) != 0:
            return False
        self.expire_governing_slots()
        self.apply_decay()
        return True

    def expire_governing_slots(self) -> int:
        """Supersede stale 'current state' memories within each slot.

        Slot memories (e.g. how Roum wants to be addressed) are mutually
        exclusive: only the newest active one stays authoritative, older ones
        are marked superseded so the model never has to pick between them.
        """
        expired = 0
        for target_model in sorted(VALID_TARGET_MODELS):
            slots: Dict[str, Dict[str, Any]] = {}
            for mem in self.memories[target_model]:
                slot = mem.get("slot")
                if not slot or mem.get("status") != "active":
                    continue
                current = slots.get(slot)
                if current is None or str(mem.get("timestamp", "")) > str(current.get("timestamp", "")):
                    slots[slot] = mem
            if not slots:
                continue
            with self._transaction(target_model):
                for mem in self.memories[target_model]:
                    slot = mem.get("slot")
                    if not slot or mem.get("status") != "active":
                        continue
                    winner = slots.get(slot)
                    if winner is not None and mem.get("id") != winner.get("id"):
                        mem["status"] = "superseded"
                        mem["superseded_by"] = winner.get("id")
                        mem["superseded_at"] = _now()
                        mem["supersession_reason"] = f"stale slot '{slot}' value"
                        expired += 1
        return expired

    # ---- current state & governing reads ------------------------------
    def get_governing_memories(self, target_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Active memories that must always apply, regardless of the query."""
        models = [target_model] if target_model else sorted(VALID_TARGET_MODELS)
        out: List[Dict[str, Any]] = []
        with self._lock:
            for name in models:
                for mem in self.memories[name]:
                    if _authority.is_governing(mem) and mem.get("content"):
                        out.append(copy.deepcopy(mem))
        out.sort(key=self._authority_key, reverse=True)
        return out

    def get_current_state(self) -> List[Dict[str, Any]]:
        """The single authoritative active memory per slot (e.g. current address)."""
        state: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for name in sorted(VALID_TARGET_MODELS):
                for mem in self.memories[name]:
                    slot = mem.get("slot")
                    if not slot or mem.get("status") != "active":
                        continue
                    current = state.get(slot)
                    if current is None or str(mem.get("timestamp", "")) > str(current.get("timestamp", "")):
                        state[slot] = copy.deepcopy(mem)
        return [state[k] for k in sorted(state)]

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
                         confidence: float = 1.0, reason: str = "Explicit user correction",
                         **extra) -> str:
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

            # A correction inherits the corrected memory's slot, so the new
            # value becomes the current state for that slot (section 12).
            extra.setdefault("slot", old.get("slot"))

            new_mem = self._build_memory(target_model, content, new_type, source,
                                         keywords, tags, confidence, extra)
            new_mem["supersedes"] = old_id
            with self._transaction(target_model):
                old["status"] = "superseded"
                old["superseded_by"] = new_mem["id"]
                old["superseded_at"] = _now()
                old["supersession_reason"] = str(reason)
                old["contradicts"] = _clean_str_list(
                    list(old.get("contradicts") or []) + [new_mem["id"]]
                )
                old["contradiction_count"] = int(old.get("contradiction_count", 0)) + 1
                old["confidence"] = round(
                    _clamp_confidence(old.get("confidence", 1.0)) * WEAKEN_FACTOR, 4
                )
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

    def get_retrievable_memories(self, target_model: str) -> List[Dict[str, Any]]:
        """Active + weakened memories: usable context, ranked by strength later.

        Superseded and archived memories are excluded, so a stale value can
        never be injected as an instruction.
        """
        self._check_target(target_model)
        with self._lock:
            return [
                copy.deepcopy(m) for m in self.memories[target_model]
                if m.get("status") in ("active", "weakened")
            ]

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
                    "by_status": dict(Counter(m.get("status", "unknown") for m in items)),
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
