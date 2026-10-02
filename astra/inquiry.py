"""Astra's persistent questions, uncertainty, and revisable understanding.

This is deliberately not a "curiosity engine" and it does not invent a parallel
belief store. A question is an ordinary memory in the self model; an
interpretation is an ordinary memory in the Roum model. Everything they need -
evidence counts, confidence, contradiction handling, decay, dormancy,
supersession - already exists in :mod:`astra.memory`. This module supplies only
the vocabulary that lets that machinery express *not knowing*:

* the memory types (an open question, an observation, a tentative
  interpretation, an unconfirmed hypothesis);
* the labels that keep "known" distinct from "tentative" distinct from
  "unresolved" (they are stored as tags, so no new field is invented);
* deterministic detectors for whether a piece of material looks like a
  question, a conflict, or something Astra wants to raise with Roum;
* rendering (the prompt block for open questions and for work-specific
  knowledge, and the conversational *candidates* for a later initiative slice);
* and the small event -> experience mapping that lets a resolution, a revision,
  or a conflict nudge the existing experiential-affect state.

Nothing here writes to a store, and nothing here resolves an ambiguity. A
question stays open until evidence or an explicit correction closes it, and a
contradiction is recorded rather than silently decided.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------
# Memory types
# ---------------------------------------------------------------------
# Astra's own observed history, distinct from a durable self_fact. An
# observation carries what she actually noticed; it never becomes a trait
# without repeated evidence (see SELF_DURABLE_* in astra.memory).
TYPE_OBSERVATION = "observation"
# A tentative interpretation of something observed - explicitly revisable.
TYPE_INTERPRETATION = "interpretation"
# A possibility Astra is entertaining that the evidence does not yet support.
TYPE_HYPOTHESIS = "hypothesis"
# An unresolved line of inquiry. Its status is deliberately NOT the memory
# status: a question can be open, answered, or abandoned while the underlying
# record stays "active" (readable and retrievable). That separation is what
# lets a question keep its history after it is answered.
TYPE_QUESTION = "open_question"

KNOWLEDGE_TYPES = {TYPE_OBSERVATION, TYPE_INTERPRETATION, TYPE_HYPOTHESIS}
INQUIRY_TYPES = {TYPE_QUESTION}

# ---------------------------------------------------------------------
# Epistemic labels
# ---------------------------------------------------------------------
# Stored as a tag (``epistemic:<label>``) so the distinction survives retrieval
# without adding a field to every record. The label is what stops a confident
# sentence from being read as an established fact.
EPISTEMIC_KNOWN = "known"
EPISTEMIC_STATED = "stated"
EPISTEMIC_INTERPRETATION = "interpretation"
EPISTEMIC_SUSPECTED = "suspected"
EPISTEMIC_POSSIBLE = "possible"
EPISTEMIC_UNKNOWN = "unknown"

EPISTEMIC_LABELS = (
    EPISTEMIC_KNOWN, EPISTEMIC_STATED, EPISTEMIC_INTERPRETATION,
    EPISTEMIC_SUSPECTED, EPISTEMIC_POSSIBLE, EPISTEMIC_UNKNOWN,
)

# Default label per type; a caller may override it (an observation may be
# ``stated`` when it quotes the work, an interpretation is usually
# ``interpretation``).
DEFAULT_EPISTEMIC = {
    TYPE_OBSERVATION: EPISTEMIC_KNOWN,
    TYPE_INTERPRETATION: EPISTEMIC_INTERPRETATION,
    TYPE_HYPOTHESIS: EPISTEMIC_POSSIBLE,
    TYPE_QUESTION: EPISTEMIC_UNKNOWN,
}

# How confident a memory may claim to be at each label. An interpretation is
# never allowed to *look* certain, no matter how confidently the model worded
# it; this is the guard against "the LLM produced confident wording, therefore
# it is a durable belief".
CONFIDENCE_CEILING = {
    EPISTEMIC_STATED: 1.0,
    EPISTEMIC_KNOWN: 0.9,
    EPISTEMIC_INTERPRETATION: 0.7,
    EPISTEMIC_SUSPECTED: 0.55,
    EPISTEMIC_POSSIBLE: 0.45,
    EPISTEMIC_UNKNOWN: 0.3,
}

# Question lifecycle. These are the question's *own* status, carried as a tag
# (``qstatus:<value>``) rather than the memory status, so answered and abandoned
# questions remain readable history.
Q_OPEN = "open"
Q_PARTIAL = "partially_answered"
Q_ANSWERED = "answered"
Q_REOPENED = "reopened"
Q_ABANDONED = "abandoned"
QUESTION_STATUSES = (Q_OPEN, Q_PARTIAL, Q_ANSWERED, Q_REOPENED, Q_ABANDONED)
# Open and reopened questions are live lines of inquiry; partially answered ones
# are live but explicitly not settled.
LIVE_QUESTION_STATUSES = (Q_OPEN, Q_REOPENED, Q_PARTIAL)

# Why a question exists. Preserved rather than flattened, because "I want
# Roum's view" is a different reason to speak than "I need a fact".
ORIGIN_READING = "reading"
ORIGIN_INTERACTION = "interaction"
ORIGIN_CONTRADICTION = "contradiction"
ORIGIN_ASSOCIATION = "association"
ORIGIN_EXPERIENCE = "experience"
ORIGIN_INTEREST = "interest"
ORIGIN_SPONTANEOUS = "spontaneous"
ORIGIN_LABELS = (
    ORIGIN_READING, ORIGIN_INTERACTION, ORIGIN_CONTRADICTION, ORIGIN_ASSOCIATION,
    ORIGIN_EXPERIENCE, ORIGIN_INTEREST, ORIGIN_SPONTANEOUS,
)

# Why Astra would raise it. ``factual`` means she wants the answer; ``roum_view``
# means she wants Roum's perspective specifically and would still want it even
# if the fact were available elsewhere.
MOTIVE_FACTUAL = "factual"
MOTIVE_ROUM_VIEW = "roum_view"
MOTIVE_UNDERSTANDING = "understanding"
MOTIVE_CONNECTION = "connection"
MOTIVE_LABELS = (MOTIVE_FACTUAL, MOTIVE_ROUM_VIEW, MOTIVE_UNDERSTANDING, MOTIVE_CONNECTION)

# Candidate kinds a later conversational-initiative slice may draw on. Slice 2
# only *preserves* them; it never acts on them.
CANDIDATE_UNRESOLVED_QUESTION = "unresolved_question"
CANDIDATE_CONFLICT = "conflict"
CANDIDATE_REALIZATION = "realization"
CANDIDATE_DISCOVERY = "discovery"
CANDIDATE_ASSOCIATION = "association"
CANDIDATE_WANTS_ROUM_VIEW = "wants_roum_view"

_MAX_BLOCK_QUESTIONS = 8


# ---------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------
_QUESTION_START = re.compile(
    r"^\s*(?:who|what|when|where|why|how|which|whose|whether|whom)\b", re.I)
_QUESTION_MARK = re.compile(r"\?\s*$")
# Wording that signals an open inquiry even without a question mark.
_INQUIRY_PHRASES = tuple(re.compile(p, re.I) for p in (
    r"\bi (?:don'?t|do not) (?:know|understand)\b",
    r"\bi wonder\b", r"\bi'?m not sure\b", r"\bnot sure (?:why|how|whether|if)\b",
    r"\bdoesn'?t (?:make sense|add up)\b", r"\bunclear\b",
    r"\bi can'?t tell\b", r"\bwonder whether\b",
    r"\bwhat does .* mean\b", r"\bwhy did\b", r"\bwhy does\b",
))
# Wording that signals an interpretation rather than an observation.
_INTERPRETATION_PHRASES = tuple(re.compile(p, re.I) for p in (
    r"\bi think\b", r"\bi suspect\b", r"\bi believe\b", r"\bprobably\b",
    r"\bperhaps\b", r"\bmaybe\b", r"\bseems?\b", r"\bappears?\b",
    r"\bmight (?:be|have|mean)\b", r"\bcould (?:be|have|mean)\b",
    r"\bmy (?:guess|reading|interpretation)\b", r"\bi took it (?:to mean|as)\b",
))
# Wording that explicitly asks for Roum's own view rather than a fact.
_ROUM_VIEW_PHRASES = tuple(re.compile(p, re.I) for p in (
    r"\broum'?s (?:take|view|opinion|perspective|reading|interpretation|thoughts)\b",
    r"\bwhat (?:would|does) roum\b",
    r"\bi wonder (?:what|how) roum\b",
    r"\brather hear (?:what|how) roum\b",
    r"\broum would (?:think|see|read)\b",
    r"\bask roum\b",
))
# Wording that signals a recognised conflict rather than a settled reading.
_CONFLICT_PHRASES = tuple(re.compile(p, re.I) for p in (
    r"\bcontradict\w*\b", r"\bconflict\w*\b", r"\bdoesn'?t (?:agree|match|fit)\b",
    r"\bdon'?t (?:agree|match|fit)\b", r"\binconsistent\b",
    r"\b(?:but|yet|however|though)\b.*\b(?:earlier|before|previously|said|thought)\b",
    r"\bi thought\b.*\bbut\b", r"\bon the other hand\b", r"\bnot consistent\b",
))
# An association with something Astra already knows or has encountered.
_ASSOCIATION_PHRASES = tuple(re.compile(p, re.I) for p in (
    r"\breminds me of\b", r"\bsimilar to\b", r"\brecalls\b",
    r"\blike (?:the|our|that|what) (?:time|conversation|earlier|thing)\b",
    r"\bsame as\b", r"\bechoes\b",
))


def _matches(patterns: Iterable, text: str) -> bool:
    return any(p.search(text) for p in patterns)


def looks_like_question(text: Any) -> bool:
    """True when ``text`` is an unresolved line of inquiry, not a statement.

    A trailing question mark or an interrogative opener counts, as does
    explicit "I don't know / I wonder / it doesn't add up" wording. Kept
    deterministic and narrow: a false question would fill the store with
    noise, and the brief is explicit that not every uncertainty is a question.
    """
    text = str(text or "").strip()
    if not text:
        return False
    if _QUESTION_MARK.search(text) or _QUESTION_START.search(text):
        return True
    return _matches(_INQUIRY_PHRASES, text)


def looks_like_interpretation(text: Any) -> bool:
    """True when ``text`` reads as a tentative interpretation, not an observation."""
    return _matches(_INTERPRETATION_PHRASES, str(text or ""))


def looks_like_conflict(text: Any) -> bool:
    """True when ``text`` describes two things that do not agree."""
    return _matches(_CONFLICT_PHRASES, str(text or ""))


def looks_like_association(text: Any) -> bool:
    """True when ``text`` connects the current material to something else."""
    return _matches(_ASSOCIATION_PHRASES, str(text or ""))


def wants_roum_view(text: Any) -> bool:
    """True when Astra wants Roum's perspective specifically, not just a fact."""
    return _matches(_ROUM_VIEW_PHRASES, str(text or ""))


# ---------------------------------------------------------------------
# Tags and labels
# ---------------------------------------------------------------------
def epistemic_tag(label: str) -> str:
    return f"epistemic:{label}"


def qstatus_tag(status: str) -> str:
    return f"qstatus:{status}"


def origin_tag(origin: str) -> str:
    return f"origin:{origin}"


def motive_tag(motive: str) -> str:
    return f"motive:{motive}"


def work_tag(work_id: Any) -> Optional[str]:
    work_id = str(work_id or "").strip()
    return f"work:{work_id}" if work_id else None


def tags_for(mem: Dict[str, Any]) -> List[str]:
    return [str(t) for t in (mem.get("tags") or [])]


def epistemic_of(mem: Dict[str, Any]) -> str:
    """The memory's epistemic label, defaulting from its type."""
    for tag in tags_for(mem):
        if tag.startswith("epistemic:"):
            return tag.split(":", 1)[1]
    return DEFAULT_EPISTEMIC.get(str(mem.get("type") or ""), EPISTEMIC_UNKNOWN)


def question_status_of(mem: Dict[str, Any]) -> str:
    for tag in tags_for(mem):
        if tag.startswith("qstatus:"):
            return tag.split(":", 1)[1]
    return Q_OPEN


def origin_of(mem: Dict[str, Any]) -> Optional[str]:
    for tag in tags_for(mem):
        if tag.startswith("origin:"):
            return tag.split(":", 1)[1]
    return None


def motive_of(mem: Dict[str, Any]) -> Optional[str]:
    for tag in tags_for(mem):
        if tag.startswith("motive:"):
            return tag.split(":", 1)[1]
    return None


def work_of(mem: Dict[str, Any]) -> Optional[str]:
    """The work a record belongs to, from the structured field or its tag."""
    work_id = str(mem.get("work_id") or "").strip()
    if work_id:
        return work_id
    for tag in tags_for(mem):
        if tag.startswith("work:"):
            return tag.split(":", 1)[1]
    return None


def is_question(mem: Dict[str, Any]) -> bool:
    return str(mem.get("type") or "") == TYPE_QUESTION


def is_open_question(mem: Dict[str, Any]) -> bool:
    return is_question(mem) and question_status_of(mem) in LIVE_QUESTION_STATUSES


def is_knowledge(mem: Dict[str, Any]) -> bool:
    return str(mem.get("type") or "") in KNOWLEDGE_TYPES


def is_inquiry_memory(mem: Dict[str, Any]) -> bool:
    """True for anything this module owns, so retrieval can treat it specially."""
    return str(mem.get("type") or "") in (KNOWLEDGE_TYPES | INQUIRY_TYPES)


def confidence_ceiling_for(label: str) -> float:
    return CONFIDENCE_CEILING.get(label, 0.5)


def clamped_confidence(label: str, confidence: Any, default: float = 0.5) -> float:
    """Clamp a proposed confidence to what its epistemic label may claim."""
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = default
    ceiling = confidence_ceiling_for(label)
    return round(max(0.0, min(value, ceiling)), 4)


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------
def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def render_question(mem: Dict[str, Any]) -> str:
    status = question_status_of(mem)
    marker = {
        Q_OPEN: "open", Q_REOPENED: "reopened (was previously answered)",
        Q_PARTIAL: "partially answered", Q_ANSWERED: "answered",
        Q_ABANDONED: "set aside",
    }.get(status, status)
    motive = motive_of(mem)
    suffix = ""
    if motive == MOTIVE_ROUM_VIEW:
        suffix = " [wants Roum's own view]"
    elif motive == MOTIVE_CONNECTION:
        suffix = " [relates to an earlier exchange]"
    work = work_of(mem)
    work_suffix = f" (re: {work})" if work else ""
    return f"{_clean(mem.get('content'))}{work_suffix} [{marker}]{suffix}"


def open_questions(memories: List[Dict[str, Any]], *, work_id: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Live questions, optionally scoped to one work, newest-relevant first."""
    result = [m for m in memories if is_open_question(m) and m.get("content")]
    if work_id is not None:
        result = [m for m in result if work_of(m) == work_id]
    result.sort(key=lambda m: str(m.get("timestamp") or ""), reverse=True)
    return result[:limit] if limit else result


def questions_prompt_block(memories: List[Dict[str, Any]]) -> Optional[str]:
    """The open-questions block: what Astra does not know, stated as such.

    Only live questions appear, and they are explicitly framed as unresolved so
    the model cannot quietly answer them with general knowledge.
    """
    live = open_questions(memories, limit=_MAX_BLOCK_QUESTIONS)
    if not live:
        return None
    lines = ["\n=== OPEN QUESTIONS (UNRESOLVED - ASTRA DOES NOT KNOW YET) ==="]
    lines.append(
        "These are things Astra has not resolved. Do not answer them with general "
        "knowledge and do not present them as settled. She may wonder about them, "
        "say she is unsure, or ask Roum - she should not pretend to know."
    )
    for mem in live:
        lines.append(f"  ? {render_question(mem)}")
    return "\n".join(lines)


def work_knowledge_prompt_block(memories: List[Dict[str, Any]], work_id: str) -> Optional[str]:
    """Everything Astra has for one work, labelled by how well she knows it.

    This is the boundary that keeps work-specific understanding distinct from
    the base model's knowledge: only what Astra actually recorded about the work
    appears here, and each line says whether it is known, tentative, or merely
    suspected.
    """
    scoped = [m for m in memories if is_knowledge(m) and work_of(m) == work_id and m.get("content")]
    if not scoped:
        return None
    scoped.sort(key=lambda m: (epistemic_of(m) != EPISTEMIC_STATED, str(m.get("timestamp") or "")))
    lines = [f"\n=== WORK CONTEXT: {work_id} (ONLY WHAT ASTRA HAS ENCOUNTERED) ==="]
    lines.append(
        "These are Astra's own notes from this work, not general knowledge. A "
        "tentative line is her interpretation, not the work's statement, and it "
        "may be revised; do not upgrade it to fact."
    )
    for mem in scoped:
        label = epistemic_of(mem)
        lines.append(f"  - ({label}) {_clean(mem.get('content'))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Conversational candidates (preserved, not acted on)
# ---------------------------------------------------------------------
def conversation_candidates(memories: List[Dict[str, Any]],
                            experiences: Optional[List[Dict[str, Any]]] = None,
                            *, limit: int = 5) -> List[Dict[str, Any]]:
    """Reasons Astra *might* later want to speak - preserved, never triggered.

    Each candidate carries its cause so a later initiative slice can weigh it
    against Astra's other motivations. Nothing here schedules or sends
    anything; there is no timer and no "if idle, speak" rule.
    """
    candidates: List[Dict[str, Any]] = []
    for mem in open_questions(memories):
        motive = motive_of(mem)
        kind = CANDIDATE_WANTS_ROUM_VIEW if motive == MOTIVE_ROUM_VIEW else CANDIDATE_UNRESOLVED_QUESTION
        candidates.append({
            "kind": kind,
            "reason": _clean(mem.get("content")),
            "question_id": mem.get("id"),
            "work_id": work_of(mem),
            "motive": motive,
            "weight": _candidate_weight(mem),
        })
    for mem in memories:
        if str(mem.get("type") or "") == TYPE_INTERPRETATION and mem.get("contradicts"):
            candidates.append({
                "kind": CANDIDATE_CONFLICT,
                "reason": _clean(mem.get("content")),
                "question_id": None,
                "work_id": work_of(mem),
                "motive": MOTIVE_UNDERSTANDING,
                "weight": 0.6,
            })
    for exp in (experiences or []):
        kind = str(exp.get("experience_kind") or "")
        if kind == "realization":
            candidates.append({
                "kind": CANDIDATE_REALIZATION, "reason": _clean(exp.get("content")),
                "question_id": None, "work_id": exp.get("work_id"),
                "motive": MOTIVE_UNDERSTANDING, "weight": 0.5,
            })
        elif kind == "discovered":
            candidates.append({
                "kind": CANDIDATE_DISCOVERY, "reason": _clean(exp.get("content")),
                "question_id": None, "work_id": exp.get("work_id"),
                "motive": MOTIVE_FACTUAL, "weight": 0.45,
            })
    candidates.sort(key=lambda c: c["weight"], reverse=True)
    return candidates[:limit]


def _candidate_weight(mem: Dict[str, Any]) -> float:
    """How much a question deserves attention - relevance, not truth.

    This never affects whether an interpretation is *correct*; it only orders
    what is worth thinking or talking about.
    """
    base = 0.4
    try:
        base += 0.3 * float(mem.get("confidence", 0.5) or 0.0)
    except (TypeError, ValueError):
        pass
    if motive_of(mem) == MOTIVE_ROUM_VIEW:
        base += 0.15
    if str(mem.get("status") or "active") != "active":
        base -= 0.2  # dormant questions still exist, but do not dominate
    return round(max(0.0, min(base, 1.0)), 4)


# ---------------------------------------------------------------------
# Experience mapping (event -> affect nudge)
# ---------------------------------------------------------------------
def resolution_kind(*, reopened: bool = False, contradiction: bool = False,
                    wrong: bool = False, partial: bool = False) -> str:
    """The experience kind that best describes how a question moved.

    Returned as a *name*; the caller records the experience through the store
    (Slice 1), so affect stays grounded in events rather than in question mood.
    """
    if contradiction:
        return "unexpected"
    if wrong:
        return "realization"
    if reopened:
        return "discovered"
    if partial:
        return "discovered"
    return "completed"


def question_evidence_span(question: Dict[str, Any],
                           evidence: Dict[str, Any]) -> bool:
    """True when ``evidence`` plausibly bears on ``question``.

    Deliberately lexical rather than exact-string: a later note need not repeat
    the question's wording to be about the same thing. Uses the shared-token
    containment the rest of the project already uses, plus a shared work.
    """
    q_work, e_work = work_of(question), work_of(evidence)
    q_text = _clean(question.get("content"))
    e_text = _clean(evidence.get("content"))
    if not q_text or not e_text:
        return False
    q_tokens = _tokens(q_text) - _STOP
    e_tokens = _tokens(e_text) - _STOP
    if not q_tokens or not e_tokens:
        return False
    shared = q_tokens & e_tokens
    if len(shared) >= 3:
        return True
    # Same work makes a weaker overlap meaningful: a shared subject word (the
    # character or thing the question is about) is enough when both records
    # belong to the same work, and a distinctive token is enough on its own.
    if q_work and q_work == e_work:
        if len(shared) >= 2:
            return True
        if shared and max(len(t) for t in shared) >= 6:
            return True
    return False


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "for", "and", "or", "but", "it", "this", "that", "these", "those",
    "did", "does", "do", "why", "how", "what", "who", "when", "where", "i",
    "she", "he", "they", "them", "her", "his", "their", "me", "my", "you",
    "with", "as", "at", "by", "from", "about", "into", "if", "not", "no",
    "would", "could", "should", "might", "can", "will", "think", "wonder",
}


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9']+", str(text or "").casefold()) if len(t) >= 3}


@lru_cache(maxsize=16384)
def _significant_tokens_cached(text: str) -> frozenset:
    return frozenset(_tokens(text) - _STOP)


def significant_tokens(text: Any) -> frozenset:
    """Content tokens of ``text`` with stopwords removed.

    The shared retriever scores raw token overlap and does not filter stopwords,
    so a turn containing "the" can match unrelated records. Work-scoping and
    question-relevance use this stricter view so a common word never drags in
    another work's context.

    Cached: the same record content is re-tokenized on every turn (and, for the
    question set, several times within one turn), so the analysis is pure and the
    result is safe to reuse. Callers only test membership/intersection, so the
    frozenset return keeps a cached value from being mutated.
    """
    return _significant_tokens_cached(str(text or ""))
