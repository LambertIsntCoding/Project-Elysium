"""Astra's conversational working memory: the middle layer between the current
turn and durable memory.

Long-term memory is governed and durable; the current turn is just the latest
exchange. Neither is the right place for the *shape of the conversation*: what
it is about, what she is currently trying to answer, which details matter for
the next few turns, unresolved references ("that thing we talked about"), the
temporary assumptions she is holding, and threads she set aside to resume.

This module supplies exactly that, and nothing more. It is deliberately:

* **not a memory layer.** Nothing here is written through the memory store, so
  it never becomes a durable memory automatically, never decays like one, and
  never appears in generic retrieval. It is conversation scratch, in the same
  spirit as ``memory.TemporaryContext`` - in-process, capped, and expiring.
* **not persisted.** Like ``TemporaryContext``, it lives for the life of the
  process. A conversation that matters can still be consolidated into real
  memories through the ordinary governed path; working memory never promotes
  itself.
* **not model-driven.** Topic, goal and reference detection are deterministic
  heuristics over the turn and the recent history, so the application decides
  what the conversation is about rather than asking the generator.
* **read-only at prompt time.** ``prompt_block`` renders current state; it never
  mutates it, so building a prompt stays side-effect free. The live turn path
  (the session) is what updates it, exactly like the relational/affect state.

A thread is *suspended*, not deleted, and can be *resumed* - so "back to what
we were saying" is a real operation rather than a guess.
"""
from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# How many turns a scratch item is useful for. Bounded on purpose: working
# memory is about *this* conversation, not about accumulating history.
DEFAULT_TTL_TURNS = 12
DEFAULT_CAP = 24

# The slots. Kept small and concrete so a reader can tell them apart.
SLOT_TOPIC = "topic"
SLOT_GOAL = "goal"
SLOT_DETAIL = "detail"
SLOT_REFERENCE = "reference"
SLOT_ASSUMPTION = "assumption"
SLOT_THREAD = "thread"

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "for", "and", "or", "but", "it", "this", "that", "these", "those",
    "i", "you", "she", "he", "they", "them", "her", "his", "their", "me", "my",
    "with", "as", "at", "by", "from", "about", "into", "if", "not", "no",
    "do", "does", "did", "so", "we", "us", "our", "what", "why", "how", "when",
    "where", "who", "which", "just", "really", "very", "can", "could", "would",
    "should", "will", "have", "has", "had", "get", "got", "there", "here",
}

# An unresolved reference: wording that points back at something shared without
# naming it, so it needs the conversation state to be understood.
_REFERENCE_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:that|this|the) (?:thing|stuff|one|idea|point|part|bit|topic|question)\b",
    r"\bwhat (?:we|you) (?:were|was) (?:saying|talking about|discussing)\b",
    r"\b(?:earlier|before|previously|a moment ago|just now)\b",
    r"\b(?:remember|recall) (?:when|what|that)\b",
    r"\bback to\b", r"\b(?:the )?(?:same|other) (?:thing|topic|one)\b",
    r"\byou know what i mean\b", r"\blike i said\b",
))
# A temporary assumption: something held only for now, explicitly revisable.
_ASSUMPTION_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:let'?s|i'?ll) (?:say|assume|pretend)\b",
    r"\bassuming\b", r"\bfor now\b", r"\btentatively\b", r"\bprovisionally\b",
    r"\bif (?:we|you) assume\b", r"\bsuppose\b", r"\bworking assumption\b",
))
# A goal: what she is currently trying to answer or do in the conversation.
_GOAL_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:i'?m|i am|we'?re|we are) trying to\b",
    r"\bi need to (?:figure out|find out|understand|work out|answer)\b",
    r"\bhelp me\b", r"\bcan you (?:explain|tell me|show me|help)\b",
    r"\bwhat i want to know\b", r"\bthe question is\b", r"\bi'?m working on\b",
))
# Wording that sets a thread aside, to be resumed later.
_SUSPEND_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:set|put) (?:that|this|it) aside\b", r"\bback to that later\b",
    r"\bwe'?ll come back to\b", r"\blet'?s park\b", r"\bsave (?:that|it) for later\b",
    r"\btangent\b", r"\banyway\b", r"\boff topic\b",
))
_RESUME_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\bback to\b", r"\bwhere were we\b", r"\bas i was saying\b",
    r"\breturn to\b", r"\bpick (?:that|this|it) (?:back )?up\b",
    r"\bcontinuing (?:from|where)\b",
))


def _matches(patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)


def significant_tokens(text: Any) -> set:
    return {t for t in re.findall(r"[a-z0-9']+", str(text or "").casefold())
            if len(t) >= 3 and t not in _STOP}


def _clean(text: Any, limit: int = 240) -> str:
    return " ".join(str(text or "").split())[:limit]


def looks_like_reference(text: Any) -> bool:
    return _matches(_REFERENCE_PATTERNS, str(text or ""))


def looks_like_assumption(text: Any) -> bool:
    return _matches(_ASSUMPTION_PATTERNS, str(text or ""))


def looks_like_goal(text: Any) -> bool:
    return _matches(_GOAL_PATTERNS, str(text or ""))


def looks_like_suspend(text: Any) -> bool:
    return _matches(_SUSPEND_PATTERNS, str(text or ""))


def looks_like_resume(text: Any) -> bool:
    return _matches(_RESUME_PATTERNS, str(text or ""))


class ConversationState:
    """The current conversation's working state: topic, goal, refs, assumptions,
    and suspended threads.

    In-process only (never persisted), capped, and expiring by turn count, so it
    can never become durable memory by accident. It is updated on the live turn
    path and only *read* during prompt assembly.
    """

    def __init__(self, *, ttl_turns: int = DEFAULT_TTL_TURNS,
                 cap: int = DEFAULT_CAP):
        self.ttl_turns = max(1, int(ttl_turns))
        self.cap = max(1, int(cap))
        self._lock = threading.RLock()
        self._turn = 0
        self.topic: str = ""
        self.topic_tokens: set = set()
        self.goal: str = ""
        self._items: List[Dict[str, Any]] = []      # details/refs/assumptions/threads
        self._threads: List[Dict[str, Any]] = []    # suspended, resumable

    def _salient_detail(self, text: str) -> str:
        """The next-turn detail: the most *specific* fragment of the last turn.

        Only meaningful when the turn has more than one clause - for a single
        clause the topic already carries the whole thing, and duplicating it
        would just pad the prompt. Concrete tokens (names, numbers, unusual
        words) matter for the next few turns far more than the sentence frame,
        so the detail is the fragment with the most significant tokens.
        """
        pieces = [p.strip() for p in re.split(r"[.!?;\n]+", text) if p.strip()]
        if len(pieces) < 2:
            return ""
        best = max(pieces, key=lambda p: len(significant_tokens(p)))
        return _clean(best, 160) if significant_tokens(best) else ""

    # -- updating (live path only) -------------------------------------
    def observe_turn(self, user_input: str, response: str = "",
                     history: Optional[List[Dict[str, str]]] = None) -> None:
        """Fold one turn into the working state.

        Deterministic: the topic is the most recent turn's salient words; a goal
        is recorded only when the turn actually states one; a detail, reference,
        assumption, or suspended thread is recorded only when the turn warrants
        one. Nothing here is remembered durably.
        """
        text = _clean(user_input)
        if not text:
            return
        with self._lock:
            self._turn += 1
            tokens = significant_tokens(text)
            previous_tokens = self.topic_tokens
            # The topic is the turn's own subject when it has one; a very short
            # or purely referential turn keeps the previous topic.
            if len(tokens) >= 2 or not self.topic:
                self.topic = text
                self.topic_tokens = tokens
            # A topic shift invalidates temporary assumptions from the old topic.
            # Compare against the *previous* topic's tokens, not the new ones.
            if (tokens and previous_tokens and not (tokens & previous_tokens)
                    and len(tokens) >= 3):
                self._items = [i for i in self._items
                               if i["slot"] != SLOT_ASSUMPTION]
            # The next-turn detail: the most specific fragment of this turn.
            detail = self._salient_detail(text)
            if detail and detail != self.topic:
                self._add(SLOT_DETAIL, detail)
            if looks_like_goal(text):
                self.goal = text
            if looks_like_assumption(text):
                self._add(SLOT_ASSUMPTION, text)
            if looks_like_reference(text):
                self._add(SLOT_REFERENCE, text)
            if looks_like_suspend(text) and self.topic:
                self.suspend(self.topic)
            if looks_like_resume(text):
                self.resume()
            self._expire()

    def set_topic(self, topic: str) -> None:
        with self._lock:
            self.topic = _clean(topic)
            self.topic_tokens = significant_tokens(topic)

    def set_goal(self, goal: str) -> None:
        with self._lock:
            self.goal = _clean(goal)

    def add_reference(self, text: str) -> None:
        self._add(SLOT_REFERENCE, text)

    def add_assumption(self, text: str) -> None:
        self._add(SLOT_ASSUMPTION, text)

    def _add(self, slot: str, text: str) -> None:
        text = _clean(text)
        if not text:
            return
        with self._lock:
            for item in self._items:
                if item["slot"] == slot and item["content"] == text:
                    item["turn"] = self._turn
                    return
            self._items.append({"slot": slot, "content": text, "turn": self._turn})
            del self._items[:-self.cap]

    def _expire(self) -> None:
        cutoff = self._turn - self.ttl_turns
        self._items = [i for i in self._items if i["turn"] >= cutoff]

    # -- threads -------------------------------------------------------
    def suspend(self, thread: str) -> None:
        """Set a thread aside so it can be resumed later. Never deleted."""
        thread = _clean(thread)
        if not thread:
            return
        with self._lock:
            self._threads = [t for t in self._threads if t["content"] != thread]
            self._threads.append({"content": thread, "turn": self._turn})
            del self._threads[:-self.cap]

    def resume(self, thread: str = "") -> Optional[str]:
        """Resume the most recent suspended thread (or a named one).

        Returns the resumed thread's text, or ``None`` when there is none.
        """
        with self._lock:
            if not self._threads:
                return None
            if thread:
                target = _clean(thread)
                for t in self._threads:
                    if t["content"] == target:
                        self._threads.remove(t)
                        self.topic = t["content"]
                        self.topic_tokens = significant_tokens(t["content"])
                        return t["content"]
                return None
            t = self._threads.pop()
            self.topic = t["content"]
            self.topic_tokens = significant_tokens(t["content"])
            return t["content"]

    def suspended_threads(self) -> List[str]:
        with self._lock:
            return [t["content"] for t in self._threads]

    # -- views ---------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """A plain-data view of the working state (never a memory record)."""
        with self._lock:
            return {
                "turn": self._turn,
                "current_topic": self.topic,
                "answer_goal": self.goal,
                "next_turn_details": [i["content"] for i in self._items
                                      if i["slot"] == SLOT_DETAIL],
                "unresolved_references": [i["content"] for i in self._items
                                          if i["slot"] == SLOT_REFERENCE],
                "expiring_assumptions": [i["content"] for i in self._items
                                         if i["slot"] == SLOT_ASSUMPTION],
                "threads": [t["content"] for t in self._threads],
                # Back-compat aliases (older diagnostics read these names).
                "topic": self.topic,
                "goal": self.goal,
                "references": [i["content"] for i in self._items
                               if i["slot"] == SLOT_REFERENCE],
                "assumptions": [i["content"] for i in self._items
                                if i["slot"] == SLOT_ASSUMPTION],
            }

    def is_empty(self) -> bool:
        with self._lock:
            return not (self.topic or self.goal or self._items or self._threads)

    def prompt_block(self) -> Optional[str]:
        """The working-memory block, or ``None`` when there is nothing to say.

        Labelled explicitly as *this conversation only* and *not a memory*, so
        the model uses it for continuity without treating it as durable.
        """
        state = self.snapshot()
        if not (state["current_topic"] or state["answer_goal"]
                or state["next_turn_details"] or state["unresolved_references"]
                or state["expiring_assumptions"] or state["threads"]):
            return None
        lines = [
            "=== CURRENT CONVERSATION (WORKING MEMORY - THIS SESSION ONLY, NOT A MEMORY) ===",
            "Scratch state for the conversation in progress. Use it for continuity. "
            "Do not treat any of it as a durable fact, do not record it as one, and "
            "do not mention that you are tracking it.",
        ]
        if state["current_topic"]:
            lines.append(f"- The conversation is currently about: {state['current_topic']}")
        if state["answer_goal"]:
            lines.append(f"- Currently trying to answer: {state['answer_goal']}")
        for detail in state["next_turn_details"]:
            lines.append(f"- Detail from the last turn that may matter next: {detail}")
        for ref in state["unresolved_references"]:
            lines.append(f"- Unresolved reference to keep in mind: {ref}")
        for asm in state["expiring_assumptions"]:
            lines.append(f"- Temporary assumption (expires when the topic changes): {asm}")
        for thread in state["threads"]:
            lines.append(f"- Set aside earlier, may be resumed: {thread}")
        return "\n".join(lines)


__all__ = [
    "ConversationState",
    "SLOT_TOPIC", "SLOT_GOAL", "SLOT_DETAIL", "SLOT_REFERENCE",
    "SLOT_ASSUMPTION", "SLOT_THREAD",
    "DEFAULT_TTL_TURNS", "DEFAULT_CAP",
    "looks_like_reference", "looks_like_assumption", "looks_like_goal",
    "looks_like_suspend", "looks_like_resume", "significant_tokens",
]
