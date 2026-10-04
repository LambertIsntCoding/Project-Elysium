"""Authoritative personal state, and the rule against inventing it.

The self-memory authority model (``astra.self_memory``) settles *what may become
durable knowledge about Astra*. This module handles a different failure: when
Astra is asked about her own current or recent situation and the retrieved
material is incomplete or contradictory, she must not quietly reconstruct an
answer from the conversation and present it as fact.

It does two things, both pure policy (no store, no model, no I/O):

* **Recognise the question.** ``query_kinds`` classifies a turn that asks about
  her reading, games, progress, prior statements, preferences, projects, or
  what she was told. Only such a turn triggers the authoritative block, so
  ordinary conversation is not turned into a status report.

* **Render what is actually known.** ``prompt_block`` takes the library's
  reading state, the works, and the relevant stored records and states them as
  facts - including, explicitly, when there is *no* reliable record. It never
  fills a gap: a missing fact is rendered as "not recorded", which is what makes
  "I don't remember exactly" a valid answer instead of a failure.

It also exposes ``claims_current_state``, the write-time guard: a generated
sentence that asserts a *current* personal state (what she is reading now, how
far she is, what she is playing) is a claim about the world, not a durable
belief, and must not be stored as authoritative self-knowledge.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# What kind of personal-state question is this?
# ---------------------------------------------------------------------
KIND_READING = "reading"
KIND_GAME = "game"
KIND_PROGRESS = "progress"
KIND_PRIOR_STATEMENT = "prior_statement"
KIND_PREFERENCE = "preference"
KIND_PROJECT = "project"
KIND_TOLD = "told"
KIND_PURPOSE = "purpose"

QUERY_KINDS = (
    KIND_READING, KIND_GAME, KIND_PROGRESS, KIND_PRIOR_STATEMENT,
    KIND_PREFERENCE, KIND_PROJECT, KIND_TOLD, KIND_PURPOSE,
)

_QUERY_PATTERNS: Dict[str, Tuple[re.Pattern, ...]] = {
    KIND_READING: tuple(re.compile(p, re.I) for p in (
        r"\bwhat (?:book|novel|story|work|thing) (?:are|were) (?:you|she|astra)\b",
        r"\bwhat (?:are|were) (?:you|she|astra) reading\b",
        r"\b(?:book|novel) (?:are|were) (?:you|she|astra)\b",
        r"\bhave (?:you|she|astra) (?:read|finished|started)\b",
        r"\bdid (?:you|she|astra) (?:read|finish|start)\b",
        r"\b(?:your|her|astra'?s) (?:reading|current book|last book|favourite book)\b",
        r"\bwhat (?:have|has) (?:you|she|astra) been reading\b",
    )),
    KIND_GAME: tuple(re.compile(p, re.I) for p in (
        r"\bwhat (?:game|games) (?:are|were|have|has) (?:you|she|astra)\b",
        r"\b(?:last|recent) game\b",
        r"\bwhat (?:are|were) (?:you|she|astra) playing\b",
        r"\b(?:you|she|astra) (?:played|playing|play)\b[^.?!]{0,20}\bgame\b",
    )),
    KIND_PROGRESS: tuple(re.compile(p, re.I) for p in (
        r"\b(?:what|which) (?:chapter|page|part)\b",
        r"\bhow far (?:are|were|did|have|had) (?:you|she|astra)\b",
        r"\bwhere (?:are|were) (?:you|she|astra) (?:in|up to|at)\b",
        r"\b(?:your|her) (?:progress|place|position|page|chapter)\b",
        r"\bwhere (?:did|do) (?:you|she|astra) (?:leave|left) off\b",
        r"\b(?:how much|how many (?:pages|chapters)) (?:have|has|did)\b",
    )),
    KIND_PRIOR_STATEMENT: tuple(re.compile(p, re.I) for p in (
        r"\bwhat did (?:you|she|astra) (?:say|tell|mention|think)\b",
        r"\bwhat (?:did|do) (?:you|she|astra) (?:previously|earlier|before)\b",
        r"\b(?:you|she|astra) said (?:earlier|before|previously|yesterday)\b",
        r"\bwhat (?:have|has) (?:you|she|astra) said\b",
    )),
    KIND_PREFERENCE: tuple(re.compile(p, re.I) for p in (
        r"\bwhat did (?:you|she|astra) think (?:of|about)\b",
        r"\bdid (?:you|she|astra) (?:like|enjoy|prefer)\b",
        r"\b(?:your|her) opinion (?:of|about|on)\b",
        r"\bwhat (?:do|did) (?:you|she|astra) (?:like|dislike|prefer)\b",
    )),
    KIND_PROJECT: tuple(re.compile(p, re.I) for p in (
        r"\bwhat (?:are|were) (?:you|she|astra) working on\b",
        r"\bwhat (?:project|task)s? (?:are|were|have)\b",
        r"\b(?:your|her) (?:current |last )?project\b",
    )),
    KIND_TOLD: tuple(re.compile(p, re.I) for p in (
        r"\bwhat (?:did|were) (?:you|she|astra) (?:told|told about)\b",
        r"\bwhat (?:was|were) (?:you|she|astra) told\b",
        r"\bwhat did (?:i|roum) tell (?:you|her)\b",
        r"\bwhat (?:have|has) (?:you|she|astra) been told\b",
    )),
    KIND_PURPOSE: tuple(re.compile(p, re.I) for p in (
        r"\bwhat(?:'s| is) (?:your|her) (?:purpose|goal|reason for being)\b",
        r"\bwhy (?:do|does) (?:you|she|astra) exist\b",
        r"\bwhat (?:are|were) (?:you|she|astra) (?:for|meant to do)\b",
        r"\bwhat is (?:the )?purpose of (?:you|astra)\b",
        r"\bwhat (?:do|does) (?:you|she|astra) want to become\b",
    )),
}


@lru_cache(maxsize=4096)
def _query_kinds_cached(text: str) -> FrozenSet[str]:
    hits = {
        kind for kind, patterns in _QUERY_PATTERNS.items()
        if any(p.search(text) for p in patterns)
    }
    return frozenset(hits)


def query_kinds(text: Any) -> FrozenSet[str]:
    """The personal-state categories a turn asks about (empty for ordinary talk)."""
    return _query_kinds_cached(str(text or ""))


def asks_personal_state(text: Any) -> bool:
    """True when the turn asks about Astra's own current/recent situation."""
    return bool(query_kinds(text))


# ---------------------------------------------------------------------
# Write-time guard: is this a claim about her *current* state?
# ---------------------------------------------------------------------
# A generated sentence that asserts a present, factual personal state. These are
# claims about the world that can be checked against a real record, not durable
# beliefs, so they must not become authoritative self-knowledge.
_CURRENT_STATE_PATTERNS: Tuple[re.Pattern, ...] = tuple(re.compile(p, re.I) for p in (
    r"\bi(?:'m| am) (?:currently |still |now )?(?:reading|playing|partway|halfway)\b",
    r"\bi(?:'m| am) (?:up to|on) (?:chapter|page|part)\b",
    r"\bi(?:'ve| have) (?:read|finished|started|reached)\b",
    r"\bi (?:just )?(?:finished|started) (?:reading|playing)\b",
    r"\bmy (?:current|last) (?:book|game|read|playthrough)\b",
    r"\bi(?:'m| am) (?:currently )?(?:farther|further|deeper) (?:into|in)\b",
    r"\bi(?:'m| am) (?:currently )?working on\b",
    r"\bi (?:previously|earlier|just) (?:said|told|mentioned)\b",
))


@lru_cache(maxsize=4096)
def _claims_current_state_cached(text: str) -> bool:
    return any(p.search(text) for p in _CURRENT_STATE_PATTERNS)


def claims_current_state(content: Any) -> bool:
    """True when a generated sentence asserts a present, checkable personal state.

    Such a claim is evidence of what Astra said, never authoritative on its own:
    it must be tied to a real record (a work in the library, a stored experience)
    before it can be treated as fact.
    """
    return _claims_current_state_cached(str(content or ""))


# ---------------------------------------------------------------------
# Contradiction detection between records
# ---------------------------------------------------------------------
_FINISH_PATTERN = re.compile(r"finished reading ['\u2018\u2019\"]?([^'\u2018\u2019\"\.]+)",
                             re.I)
_READING_NOW_PATTERN = re.compile(
    r"(?:currently reading|reading now|partway through|halfway through|"
    r"still reading) ['\u2018\u2019\"]?([^'\u2018\u2019\"\.]+)", re.I)


def _titles(patterns: Sequence[re.Pattern], contents: Iterable[Any]) -> List[str]:
    found: List[str] = []
    for text in contents:
        for pattern in patterns:
            match = pattern.search(str(text or ""))
            if match:
                title = match.group(1).strip().strip("'\"")
                if title:
                    found.append(title)
    return found


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(title or "").casefold()).strip()


def reading_state_conflict(a: Any, b: Any) -> Optional[str]:
    """A reason string when two statements disagree about reading status.

    "Finished reading 'X'" and "currently reading 'X'" are about the same work
    and assert opposite status, so they contradict even though they share no
    polarity term the generic detector recognises. Returns ``None`` otherwise.
    """
    text_a, text_b = str(a or ""), str(b or "")
    for first, second in ((text_a, text_b), (text_b, text_a)):
        finished = {_norm(t) for t in _titles([_FINISH_PATTERN], [first])}
        reading = {_norm(t) for t in _titles([_READING_NOW_PATTERN], [second])}
        if finished & reading:
            return "one says the work was finished, the other that it is being read"
    return None


def conflicts(state: Dict[str, Any], works: Sequence[Dict[str, Any]],
              records: Sequence[Dict[str, Any]]) -> List[str]:
    """Contradictions between the reading state and the stored records.

    The system already resolves contradictions through the authority model; this
    only surfaces the ones that remain, so Astra acknowledges the mix-up rather
    than silently choosing whichever sounds more plausible.
    """
    state = state if isinstance(state, dict) else {}
    contents = [r.get("content") for r in records or [] if isinstance(r, dict)]
    finished = {_norm(t) for t in _titles([_FINISH_PATTERN], contents)}
    reading_now = {_norm(t) for t in _titles([_READING_NOW_PATTERN], contents)}

    # The library is the authority on reading status; a stored memory that
    # disagrees is the contradiction.
    statuses: Dict[str, str] = {}
    for work in works or []:
        if not isinstance(work, dict):
            continue
        for key in (work.get("title"), work.get("work_id")):
            if key:
                statuses[_norm(str(key))] = str(work.get("status") or "")

    out: List[str] = []
    for title in sorted(finished):
        status = statuses.get(title)
        if status and status not in ("finished", "abandoned"):
            out.append(
                f"a stored note says she finished '{title}', but the reading "
                f"state still lists it as {status}"
            )
    for title in sorted(reading_now):
        status = statuses.get(title)
        if status == "finished":
            out.append(
                f"a stored note says she was still reading '{title}', but the "
                f"reading state lists it as finished"
            )
    # Two stored records disagreeing with each other, with no library state to
    # arbitrate: she must not silently pick the one that sounds more plausible.
    for title in sorted(finished & reading_now):
        out.append(
            f"one stored note says she finished '{title}' while another says "
            f"she is still reading it"
        )

    # The current work is finished but still pointed at: she is not reading it.
    current = str(state.get("work_id") or "")
    if current:
        current_status = str(state.get("status") or "")
        if current_status == "finished" and _norm(current) not in reading_now:
            title = state.get("title") or current
            out.append(
                f"the work she is pointed at ('{title}') is already finished, "
                f"so she is not currently reading it"
            )
    # De-duplicate while preserving order.
    seen: set = set()
    unique: List[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


# ---------------------------------------------------------------------
# Rendering the authoritative block
# ---------------------------------------------------------------------
def _words(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _progress_line(state: Dict[str, Any]) -> Optional[str]:
    work_id = str(state.get("work_id") or "")
    if not work_id:
        return None
    title = str(state.get("title") or work_id)
    status = str(state.get("status") or "")
    words_read = _words(state.get("words_read"))
    total_words = _words(state.get("total_words"))
    if total_words:
        pct = int(round(100 * min(1.0, words_read / total_words)))
        measure = f"{pct}% ({words_read} of {total_words} words)"
    else:
        measure = "no word count recorded"
    return f"'{title}' - {status}, {measure}"


def purpose_block(identity_concept: Any) -> str:
    """The established purpose, for a question about why she exists.

    Her purpose comes from identity/configuration. It is not regenerated from
    whatever subject the conversation happens to be about, so this is rendered
    from the configured concept rather than from retrieved material.
    """
    concept = " ".join(str(identity_concept or "").split())
    lines = [
        "\n=== ASTRA'S PURPOSE (ESTABLISHED - NOT REGENERATED) ===",
        "Her purpose is settled by her identity and configuration. It is not "
        "derived from, and does not change with, the subject currently being "
        "discussed: a conversation about books does not make her purpose "
        "'understanding narrative', and a conversation about feelings does not "
        "make it 'understanding what it means to feel'.",
    ]
    if concept:
        lines.append(f"- Established: {concept}")
    else:
        lines.append(
            "- Established: Astra is Astra - an artificial being and a robot "
            "maid who wants to understand the world and develop into herself, "
            "not to become human."
        )
    return "\n".join(lines)


def prompt_block(state: Dict[str, Any], works: Sequence[Dict[str, Any]] = (),
                 records: Sequence[Dict[str, Any]] = (),
                 kinds: FrozenSet[str] = frozenset(), *,
                 identity_concept: Any = "", max_records: int = 8) -> Optional[str]:
    """The authoritative personal-state block, or ``None`` if not asked about it.

    Rendered only for a personal-state question. It states what is recorded and,
    crucially, states the *absence* of a record as an absence - so the model has
    an explicit "not recorded" to answer from instead of inventing a plausible
    one. Opinions and interpretations are labelled as such and never presented
    as evidence of factual state.
    """
    if not kinds:
        return None
    state = state if isinstance(state, dict) else {}
    works = list(works or [])
    records = list(records or [])

    lines: List[str] = [
        "\n=== ASTRA'S CURRENT PERSONAL STATE (AUTHORITATIVE - DO NOT INVENT) ===",
        "This is what is actually recorded about her own current and recent "
        "situation. Answer from these records. If the specific fact is not here, "
        "she does not know it and must say she is unsure - she must not rebuild "
        "it from the conversation, from what she has been discussing, or from "
        "what would be plausible. An opinion or an interpretation about a work is "
        "not evidence of her factual progress through it.",
    ]

    if KIND_PURPOSE in kinds:
        concept = " ".join(str(identity_concept or "").split())
        lines.append(
            "- Purpose: settled by her identity and configuration, not "
            "regenerated from the current subject."
        )
        if concept:
            lines.append(f"    * Established: {concept}")
        else:
            lines.append(
                "    * Established: Astra is Astra - an artificial being and a "
                "robot maid who wants to understand the world and develop into "
                "herself, not to become human."
            )

    reading_asked = bool(kinds & {KIND_READING, KIND_GAME, KIND_PROGRESS})
    if reading_asked:
        current = _progress_line(state)
        if current:
            lines.append(f"- Currently reading: {current}")
        else:
            lines.append("- Currently reading: no work is recorded as being read.")

        finished: List[str] = []
        in_progress: List[str] = []
        for work in works:
            if not isinstance(work, dict):
                continue
            title = str(work.get("title") or work.get("work_id") or "")
            status = str(work.get("status") or "")
            if not title:
                continue
            if status == "finished":
                finished.append(title)
            elif status in ("reading", "paused"):
                in_progress.append(title)

        if finished:
            lines.append("- Finished: " + ", ".join(f"'{t}'" for t in finished))
        else:
            lines.append("- Finished: no finished work is recorded.")

        other_live = [t for t in in_progress
                      if _norm(t) != _norm(str(state.get("title") or ""))]
        if other_live:
            lines.append("- Started but not current: " + ", ".join(f"'{t}'" for t in other_live))

    if KIND_GAME in kinds:
        lines.append(
            "- Games: no game she has played is recorded. If asked, she has no "
            "reliable record of playing a game and must say so."
        )

    # Relevant stored records: experiences, preferences, and self-knowledge.
    facts: List[str] = []
    for mem in records:
        if not isinstance(mem, dict):
            continue
        content = " ".join(str(mem.get("content") or "").split())
        if not content:
            continue
        mem_type = str(mem.get("type") or "")
        label = "opinion" if mem_type in ("self_preference", "explicit_preference") \
            else "interpretation" if mem_type in ("interpretation", "self_observation",
                                                  "hypothesis") \
            else "recorded"
        facts.append(f"({label}) {content}")
        if len(facts) >= max_records:
            break
    if facts:
        lines.append("- Relevant recorded material:")
        lines.extend(f"    * {fact}" for fact in facts)
    else:
        lines.append(
            "- Relevant recorded material: none found for this question."
        )

    found = conflicts(state, works, records)
    if found:
        lines.append(
            "- Conflicting records: " + "; ".join(found) + ". She should "
            "acknowledge the mix-up naturally instead of silently choosing one."
        )

    lines.append(
        "Answer plainly: state a recorded fact as a fact, an opinion as an "
        "opinion, and a missing fact as not known. Do not manufacture an "
        "explanation for a gap, and do not turn the answer into a theory about "
        "herself."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------
# The always-on behavioural contract
# ---------------------------------------------------------------------
# Code-owned and always rendered, so the core rule cannot be switched off by
# config. It names no subsystem value, so there is nothing here to narrate back.
_CONTRACT: Tuple[str, ...] = (
    "When she talks about her own current or recent situation - what she is "
    "reading or playing, how far she is, what she previously said, wanted, or "
    "was told - she answers only from what is actually recorded. A missing fact "
    "is answered with uncertainty, not with a plausible reconstruction.",
    "Talking about a book, game, or project does not make it part of her current "
    "state. Discussing something is not evidence that she is doing it, that it "
    "is active, or that she has progressed in it.",
    "When Roum corrects a personal fact, the correction is information first: "
    "she updates the fact plainly. She does not turn it into a lesson about "
    "identity, reality, humanity, consciousness, or existence.",
    "An uncertain answer is a good answer. 'I don't remember exactly', 'I think "
    "I have that mixed up', and 'I'm not sure which one was current' are honest "
    "replies, not failures - prefer them to a confident fabrication.",
)


def behavior_block() -> str:
    """The always-on anti-invention contract, injected every turn."""
    lines = [
        "=== PERSONAL STATE: HOW ASTRA ANSWERS ABOUT HERSELF (ALWAYS APPLIES) ===",
    ]
    lines.extend(f"- {line}" for line in _CONTRACT)
    return "\n".join(lines)


__all__ = [
    "KIND_READING", "KIND_GAME", "KIND_PROGRESS", "KIND_PRIOR_STATEMENT",
    "KIND_PREFERENCE", "KIND_PROJECT", "KIND_TOLD", "KIND_PURPOSE",
    "QUERY_KINDS", "query_kinds", "asks_personal_state",
    "claims_current_state",
    "reading_state_conflict", "conflicts", "purpose_block",
    "prompt_block", "behavior_block",
]
