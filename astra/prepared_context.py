"""Prepared context: a derived index of what may matter next time.

While asleep, Astra can look ahead at the memories, open questions, works and
recent events that are likely to come up in a future interaction. This module
produces that as a small set of *references* - ids and short labels, never
large prose summaries stored as authoritative facts - and persists it in its own
file with its own version.

It is deliberately subordinate to the existing retrieval and authority system:

* It is **not a memory store**. Entries point at records that already exist; they
  never duplicate their content and never become a source of truth.
* The orchestrator decides **whether** prepared context is relevant to the
  actual current turn (a token-overlap filter), so stale preparation cannot leak
  into an unrelated conversation.
* It is **off by default**: no file means the prompt is byte-for-byte unchanged.
* Work isolation holds - a prepared entry carries its ``work_id`` and is dropped
  unless the turn is about that work.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from . import inquiry, working_memory
from .memory import atomic_save, load_json

PREPARED_VERSION = 1
MAX_ENTRIES = 24
MAX_BLOCK_ENTRIES = 4

# Entry kinds.
REF_MEMORY = "memory"
REF_QUESTION = "question"
REF_WORK = "work"
REF_EXPERIENCE = "experience"
REF_TOPIC = "topic"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(text: Any, limit: int = 160) -> str:
    return " ".join(str(text or "").split())[:limit]


class PreparedContext:
    """A bounded, derived set of references for a future turn."""

    def __init__(self, data_dir: str = "./storage") -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "prepared_context.json")
        self._lock = threading.RLock()
        self._entries: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        raw = load_json(self.path, dict, dict)
        items = raw.get("entries") if isinstance(raw, dict) else None
        return [dict(e) for e in items or [] if isinstance(e, dict)]

    def _persist(self) -> None:
        with self._lock:
            atomic_save(self.path, {
                "version": PREPARED_VERSION,
                "prepared_at": _now(),
                "entries": self._entries[-MAX_ENTRIES:],
            })

    def entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entries)

    def replace(self, entries: Iterable[Dict[str, Any]]) -> None:
        """Swap in a freshly prepared set (bounded, references only)."""
        cleaned: List[Dict[str, Any]] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            ref_id = entry.get("ref_id")
            if not ref_id:
                continue  # a reference with no source record is useless
            cleaned.append({
                "kind": str(entry.get("kind") or REF_MEMORY),
                "ref_id": str(ref_id),
                "work_id": entry.get("work_id"),
                "label": _clean(entry.get("label")),
                "reason": _clean(entry.get("reason")),
                "tokens": sorted(working_memory.significant_tokens(entry.get("label"))),
            })
        with self._lock:
            self._entries = cleaned[:MAX_ENTRIES]
            self._persist()

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._persist()


def build(store: Any, *, library: Any = None,
          active_work: Optional[str] = None,
          limit: int = MAX_ENTRIES) -> List[Dict[str, Any]]:
    """Derive references worth carrying into a future turn.

    Reads the governed store read-only; every entry is a pointer to an existing
    record. Nothing is written here - the caller decides whether to persist.
    """
    from . import memory_priority

    entries: List[Dict[str, Any]] = []

    # 1. Highest-priority retrievable memories (derived index, never stored).
    try:
        self_mems = store.get_retrievable_memories("self")
        roum_mems = store.get_retrievable_memories("roum")
    except Exception:
        self_mems, roum_mems = [], []
    pool = list(self_mems) + list(roum_mems)
    for mem in memory_priority.top(pool, active_work=active_work, limit=8):
        entries.append({
            "kind": REF_MEMORY,
            "ref_id": mem.get("id"),
            "work_id": inquiry.work_of(mem),
            "label": _clean(mem.get("content")),
            "reason": "high derived priority",
        })

    # 2. Open questions and conversation candidates (preserved, not scheduled).
    try:
        questions = store.get_questions(status="active")
    except Exception:
        questions = []
    for cand in inquiry.conversation_candidates(questions, limit=6):
        entries.append({
            "kind": REF_QUESTION if cand.get("question_id") else REF_TOPIC,
            "ref_id": cand.get("question_id") or f"cand:{cand.get('kind')}",
            "work_id": cand.get("work_id"),
            "label": _clean(cand.get("reason")),
            "reason": f"candidate: {cand.get('kind')}",
        })

    # 3. Unfinished works (partway through) so a session can pick up where it
    #    left off. The reading *position* stays in the library, not here.
    works: List[Dict[str, Any]] = []
    if library is not None and callable(getattr(library, "list_works", None)):
        try:
            works = library.list_works() or []
        except Exception:
            works = []
    for work in works:
        entries.append({
            "kind": REF_WORK,
            "ref_id": work.get("work_id"),
            "work_id": work.get("work_id"),
            "label": _clean(work.get("title") or work.get("work_id")),
            "reason": "unfinished work",
        })

    return entries[:limit]


def relevance(entries: Iterable[Dict[str, Any]], user_input: str,
              *, limit: int = MAX_BLOCK_ENTRIES) -> List[Dict[str, Any]]:
    """Entries whose label shares content tokens with the current turn.

    This is what keeps prepared context from leaking into an unrelated
    conversation: only a real topical overlap earns a mention.
    """
    turn = working_memory.significant_tokens(user_input)
    if not turn:
        return []
    scored: List[tuple] = []
    for entry in entries or []:
        tokens = set(entry.get("tokens") or [])
        if not tokens:
            tokens = working_memory.significant_tokens(entry.get("label"))
        overlap = len(turn & tokens)
        if overlap:
            scored.append((overlap, str(entry.get("ref_id") or ""), entry))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [entry for _, _, entry in scored[:limit]]


def render_block(entries: Iterable[Dict[str, Any]]) -> Optional[str]:
    """A small prompt block that references records by their short label.

    Subordinate context, clearly marked, so it can never be mistaken for a fact
    or an instruction.
    """
    items = [e for e in entries or [] if e.get("label")]
    if not items:
        return None
    lines = ["=== PREPARED CONTEXT (DERIVED, MAY BE RELEVANT) ==="]
    for entry in items:
        work = f" [{entry['work_id']}]" if entry.get("work_id") else ""
        lines.append(f"- ({entry.get('kind')}{work}) {entry.get('label')}")
    return "\n".join(lines)
