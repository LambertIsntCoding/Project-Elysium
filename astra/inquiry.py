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
    the model cannot quietly answer them with general knowledge. They are lines
    of inquiry she may still be turning over, not a list of questions to put to
    Roum - she can keep thinking about one without demanding an answer.
    """
    live = open_questions(memories, limit=_MAX_BLOCK_QUESTIONS)
    if not live:
        return None
    lines = ["\n=== OPEN QUESTIONS (UNRESOLVED - ASTRA DOES NOT KNOW YET) ==="]
    lines.append(
        "These are things Astra has not resolved. They are resources for how she "
        "reads the current moment, not a queue to work through: do not answer "
        "them with general knowledge, do not present them as settled, and do not "
        "raise one merely because it is open. When it is relevant, she may wonder "
        "about it, say she is unsure, or ask Roum; when it is not, she can let it "
        "wait."
    )
    for mem in live:
        lines.append(f"  ? {render_question(mem)}")
    return "\n".join(lines)


def work_knowledge_prompt_block(memories: List[Dict[str, Any]], work_id: str) -> Optional[str]:
    """Everything Astra has for one work, labelled by how well she knows it.

    This is the boundary that keeps work-specific understanding distinct from
    the base model's knowledge: only what Astra actually recorded about the work
    appears here, and each line says whether it is known, tentative, or merely
    suspected. Stay with what is recorded - where her notes do not cover
    something, uncertainty is preferred to invention.
    """
    scoped = [m for m in memories if is_knowledge(m) and work_of(m) == work_id and m.get("content")]
    if not scoped:
        return None
    scoped.sort(key=lambda m: (epistemic_of(m) != EPISTEMIC_STATED, str(m.get("timestamp") or "")))
    lines = [f"\n=== WORK CONTEXT: {work_id} (ONLY WHAT ASTRA HAS ENCOUNTERED) ==="]
    lines.append(
        "These are Astra's own notes from this work, not general knowledge. A "
        "tentative line is her interpretation, not the work's statement, and it "
        "may be revised; do not upgrade it to fact. Where her notes are silent, "
        "she should say she does not know rather than invent a passage, a detail, "
        "or a quotation."
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


# ---------------------------------------------------------------------
# Topic clustering, priority, and re-association
# ---------------------------------------------------------------------
# The question store has grown large, and many open questions are near
# duplicates of one another. The pieces below stop a single broad topic from
# accumulating dozens of equally-important independent records, without ever
# deleting one: they cluster, they *rank*, and they let an old question be
# re-found later. All are pure functions over records; the store applies the
# consequences (decay/dormancy/priority) through its existing governance.

# How much two questions must overlap to share a topic. Cheap lexical
# containment is enough at this scale; the goal is to stop near-duplicates
# dominating, not to do perfect semantic clustering.
CLUSTER_SIMILARITY = 0.5
# Beyond this many unresolved questions on one topic, further overlapping ones
# are treated as redundant and lose priority rapidly.
CLUSTER_REDUNDANCY_THRESHOLD = 3


def topic_key(mem: Dict[str, Any]) -> str:
    """A lightweight topic identity for a question.

    A shared work is the strongest topic signal (two questions about the same
    book belong together); otherwise the most distinctive content tokens stand
    in. Deterministic and cheap, so it is safe on the whole store.
    """
    work = work_of(mem)
    if work:
        return f"work:{work.casefold()}"
    tokens = sorted(significant_tokens(mem.get("content")))
    # The leading distinctive tokens make a stable, human-readable topic id.
    return "topic:" + "|".join(tokens[:3]) if tokens else "topic:"


def question_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Lexical containment between two questions (0..1).

    Containment (shared / shorter length) rather than Jaccard, matching the
    rest of the project: verbose model-written sentences rarely share half their
    union, so Jaccard would miss real near-duplicates.
    """
    ta, tb = significant_tokens(a.get("content")), significant_tokens(b.get("content"))
    if not ta or not tb:
        return 0.0
    shared = ta & tb
    return len(shared) / float(min(len(ta), len(tb)))


def cluster_questions(memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group open questions into topic clusters.

    Questions that are substantially about the same subject (a shared work, or
    high lexical overlap) land in the same cluster. Genuinely distinct questions
    stay separate. Efficient enough for the whole store: one pass with a bounded
    comparison against existing cluster representatives.
    """
    live = [m for m in memories if is_open_question(m) and m.get("content")]
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for mem in live:
        placed = None
        for key, members in clusters.items():
            rep = members[0]
            if work_of(rep) and work_of(rep) == work_of(mem):
                placed = key
                break
            if question_similarity(rep, mem) >= CLUSTER_SIMILARITY:
                placed = key
                break
        key = placed or topic_key(mem)
        # A collision on a generated key still merges, which is the desired
        # behaviour: they are about the same topic.
        clusters.setdefault(key, []).append(mem)
    return clusters


def _emotional_significance(mem: Dict[str, Any],
                            experiences: Optional[List[Dict[str, Any]]] = None) -> float:
    """How emotionally significant a question is, from experience evidence.

    Uses the existing experience records as evidence rather than storing raw
    affect inside the question. A question tied (by work or by shared subject
    tokens) to a strongly-emotional experience is more significant, so it stays
    meaningful longer; emotionally neutral curiosity decays sooner.
    """
    if not experiences:
        return 0.0
    q_work = work_of(mem)
    q_tokens = significant_tokens(mem.get("content"))
    best = 0.0
    for exp in experiences:
        if str(exp.get("type") or "") != "experience" and not exp.get("experience_kind"):
            continue
        try:
            intensity = float(exp.get("intensity") or 0.0)
            significance = float(exp.get("significance") or 0.0)
        except (TypeError, ValueError):
            continue
        magnitude = max(intensity, significance)
        if magnitude <= 0:
            continue
        e_work = work_of(exp)
        related = (q_work and e_work and q_work == e_work)
        if not related and q_tokens:
            related = bool(q_tokens & significant_tokens(exp.get("content")))
        if related:
            best = max(best, magnitude)
    return round(min(1.0, best), 4)


def priority_score(mem: Dict[str, Any], *,
                   experiences: Optional[List[Dict[str, Any]]] = None,
                   cluster_size: int = 1, now: Optional[Any] = None) -> float:
    """How much an open question deserves attention - relevance, not truth.

    Combines the things the brief asks for: recency, significance, how long it
    has been unresolved, accumulated evidence, and emotional significance. A
    topic already crowded with overlapping questions pushes its members *down*
    (redundancy), so distinct questions keep their footing. This never affects
    whether an interpretation is correct, only what is worth thinking about.
    """
    score = 0.35
    try:
        score += 0.2 * float(mem.get("confidence", 0.5) or 0.0)
    except (TypeError, ValueError):
        pass

    # Emotional significance (evidence-based, never a stored affect number).
    score += 0.35 * _emotional_significance(mem, experiences)

    # Accumulated evidence raises priority.
    evidence = mem.get("question_evidence") or []
    score += min(0.15, 0.03 * len(evidence))

    # A motive to hear Roum's own view keeps a question meaningful longer.
    if motive_of(mem) == MOTIVE_ROUM_VIEW:
        score += 0.1

    # Redundancy: a crowded topic pushes overlapping questions down quickly.
    if cluster_size > CLUSTER_REDUNDANCY_THRESHOLD:
        excess = cluster_size - CLUSTER_REDUNDANCY_THRESHOLD
        score -= min(0.6, 0.15 * excess)

    # A dormant question is still readable, but should not dominate.
    if str(mem.get("status") or "active") != "active":
        score -= 0.2
    return round(max(0.0, min(score, 1.0)), 4)


def prioritise_questions(memories: List[Dict[str, Any]], *,
                         experiences: Optional[List[Dict[str, Any]]] = None,
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Open questions ranked by :func:`priority_score`, with the reasons.

    Returns rows of ``{question, score, reasons, cluster}`` so the admin screen
    can explain *why* a question is prioritised - the observability the brief
    asks for - without any of this reaching the prompt as numbers.
    """
    clusters = cluster_questions(memories)
    rows: List[Dict[str, Any]] = []
    for key, members in clusters.items():
        size = len(members)
        for mem in members:
            effective_size = size
            # Redundancy only counts genuinely overlapping members.
            if size > 1:
                effective_size = 1 + sum(
                    1 for other in members
                    if other is not mem and question_similarity(mem, other) >= CLUSTER_SIMILARITY)
            score = priority_score(mem, experiences=experiences, cluster_size=effective_size)
            rows.append({
                "question": mem, "score": score, "cluster": key,
                "reasons": _priority_reasons(mem, score, effective_size, experiences),
            })
    rows.sort(key=lambda r: (r["score"], str(r["question"].get("timestamp") or "")),
              reverse=True)
    return rows[:limit] if limit else rows


def _priority_reasons(mem: Dict[str, Any], score: float, cluster_size: int,
                      experiences: Optional[List[Dict[str, Any]]]) -> List[str]:
    reasons: List[str] = []
    emo = _emotional_significance(mem, experiences)
    if emo >= 0.5:
        reasons.append("tied to a strong emotional experience")
    if (mem.get("question_evidence") or []):
        reasons.append(f"{len(mem['question_evidence'])} linked piece(s) of evidence")
    if motive_of(mem) == MOTIVE_ROUM_VIEW:
        reasons.append("she wants Roum's own view")
    if cluster_size > CLUSTER_REDUNDANCY_THRESHOLD:
        reasons.append(f"topic is crowded ({cluster_size} overlapping questions) - prioritised down")
    if str(mem.get("status") or "active") != "active":
        reasons.append("record is dormant")
    if not reasons:
        reasons.append("ordinary open curiosity")
    return reasons


def resurface_questions(memories: List[Dict[str, Any]], *,
                        query_tokens: Any = None, active_work: str = "",
                        experiences: Optional[List[Dict[str, Any]]] = None,
                        limit: int = 3) -> List[Dict[str, Any]]:
    """Open questions worth re-raising for the current context.

    A question is re-found when the current turn's tokens match it, when its
    work is now active, or when it is emotionally significant - never only when
    it happens to sit next to the thought that created it. This is what fixes
    "a question matters only at the moment it was created": the question carries
    enough structured topic information (work, significant tokens) to be found
    again later.
    """
    live = [m for m in memories if is_open_question(m) and m.get("content")]
    if not live:
        return []
    tokens = set(query_tokens or ())
    active = str(active_work or "").strip().casefold()
    scored: List[tuple] = []
    for mem in live:
        score = 0.0
        overlap = len(tokens & significant_tokens(mem.get("content"))) if tokens else 0
        if overlap:
            score += min(0.6, 0.2 * overlap)
        if active and (work_of(mem) or "").casefold() == active:
            score += 0.4
        emo = _emotional_significance(mem, experiences)
        if emo >= 0.5:
            score += 0.3
        if score > 0:
            scored.append((score, str(mem.get("timestamp") or ""), mem))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [mem for _, _, mem in scored[:limit]]


def apply_question_priority_decay(store: Any, *,
                                  experiences: Optional[List[Dict[str, Any]]] = None,
                                  now: Optional[Any] = None) -> Dict[str, int]:
    """Govern crowded/redundant questions through the existing decay system.

    Nothing is deleted. When a topic carries many overlapping unresolved
    questions, the lower-priority duplicates are weakened/dormant-ed so they
    stop competing with genuinely distinct ones; emotionally significant
    questions are protected and left alone. Returns a small summary.

    This is an explicit maintenance call (run from the admin/menu layer or a
    migration), never something the live turn path triggers, so a normal turn
    never rewrites the question set.
    """
    if store is None or not callable(getattr(store, "get_memories", None)):
        return {"weakened": 0, "protected": 0}
    memories = store.get_memories("self", status=None)
    clusters = cluster_questions(memories)
    weakened = 0
    protected = 0
    for key, members in clusters.items():
        if len(members) <= CLUSTER_REDUNDANCY_THRESHOLD:
            continue
        ranked = sorted(
            members,
            key=lambda m: priority_score(m, experiences=experiences),
            reverse=True)
        # Keep the most significant ones untouched; weaken the rest of a crowded
        # topic. Emotionally significant questions are protected outright.
        for mem in ranked[CLUSTER_REDUNDANCY_THRESHOLD:]:
            if _emotional_significance(mem, experiences) >= 0.6:
                protected += 1
                continue
            try:
                store.update_memory(
                    "self", mem.get("id"),
                    confidence=max(0.05, float(mem.get("confidence", 0.4) or 0.4) * 0.5),
                    tags=[t for t in tags_for(mem) if not str(t).startswith("cluster:")]
                    + [f"cluster:{key}", "cluster_redundant"],
                )
                weakened += 1
            except Exception:
                continue
    return {"weakened": weakened, "protected": protected,

            "clusters": len(clusters)}
