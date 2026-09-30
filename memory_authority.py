"""Memory reliability: source hierarchy, strength, decay, contradiction, boundaries.

This module is intentionally pure and dependency-free (no store, no model, no
disk).  It answers four questions the rest of the system needs:

1. How reliable is a memory, given *who said it*?  (:data:`SOURCE_TIERS`,
   :func:`source_reliability`)
2. How strong is a memory *right now*?  (:func:`effective_strength`, a
   ``confidence x importance x recency x source`` product.)
3. Do two memories contradict each other?  (:func:`detect_contradiction`)
4. Is a memory a *governing* boundary that must always apply, or ordinary
   *contextual* information?  (:func:`is_governing`)

The relationship helpers (:func:`relationship_risk`,
:func:`apply_relationship_guard`, :data:`RELATIONSHIP_BOUNDARIES`) encode the
project's hard non-romantic, human-first boundary so a stray inference or a
consolidator slip can never turn ordinary affection into a romantic memory.

Keeping this logic out of ``memory_store`` and ``orchestrator`` means both can
share one definition and it can be unit-tested without disk or network.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------
# 1. Source hierarchy (section 2)
# ---------------------------------------------------------------------
# Higher tier == more authoritative. An explicit user statement always outranks
# anything Astra inferred, and an inference can never silently override it.
SOURCE_TIERS: Dict[str, int] = {
    "user_correction": 5,          # explicit user correction (highest)
    "user_correction_implicit": 5,
    "explicit_user_statement": 4,  # direct statement / preference / boundary
    "supersession": 4,             # a user-driven correction recorded as a new memory
    "behavioral_pattern": 3,       # stable observed pattern
    "ai_extraction": 2,            # conversational observation
    "ai_inference": 1,             # Astra speculation / self-generated belief
    "inferred": 1,
    "speculation": 1,
}
DEFAULT_TIER = 2

# Multiplier applied to confidence when computing effective strength.
SOURCE_FACTOR: Dict[str, float] = {
    "user_correction": 1.00,
    "user_correction_implicit": 1.00,
    "explicit_user_statement": 0.95,
    "supersession": 0.95,
    "behavioral_pattern": 0.80,
    "ai_extraction": 0.65,
    "ai_inference": 0.50,
    "inferred": 0.50,
    "speculation": 0.45,
}
DEFAULT_SOURCE_FACTOR = 0.65

# Provenance values that are Astra's own generation and must never be promoted
# to a confirmed user fact just because they were stored (section 14).
ASTRA_SOURCES = {"ai_extraction", "ai_inference", "inferred", "speculation"}


def source_tier(source: Any) -> int:
    return SOURCE_TIERS.get(str(source or "").strip().casefold(), DEFAULT_TIER)


def source_reliability(source: Any) -> float:
    """0..1 reliability multiplier for a provenance string."""
    return SOURCE_FACTOR.get(str(source or "").strip().casefold(), DEFAULT_SOURCE_FACTOR)


def is_astra_source(source: Any) -> bool:
    return str(source or "").strip().casefold() in ASTRA_SOURCES


# ---------------------------------------------------------------------
# 2. Importance (section 1)
# ---------------------------------------------------------------------
# How much a memory *matters* independent of how confident we are. Corrections
# and explicit preferences/boundaries are important; a one-off observation is
# not. Multiplied by type so an explicit fact can still be important.
IMPORTANCE_BY_TYPE: Dict[str, float] = {
    "correction": 0.95,
    "explicit_preference": 0.85,
    "explicit_fact": 0.80,
    "explicit_project_information": 0.75,
    "behavioral_pattern": 0.70,
    "relationship_boundary": 1.00,
    "decision": 0.60,
    "relationship_event": 0.55,
    "self_preference": 0.55,
    "self_fact": 0.50,
    "self_belief": 0.45,
    "uncertain_inference": 0.35,
    "relationship_observation": 0.30,
    "self_observation": 0.25,
}
DEFAULT_IMPORTANCE = 0.45


def memory_importance(mem: Dict[str, Any]) -> float:
    """Explicit ``importance`` wins; otherwise derive it from the memory type."""
    raw = mem.get("importance")
    if raw is not None:
        try:
            return _clamp01(float(raw))
        except (TypeError, ValueError):
            pass
    return IMPORTANCE_BY_TYPE.get(str(mem.get("type") or ""), DEFAULT_IMPORTANCE)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------
# 3. Recency / decay (sections 4 & 5)
# ---------------------------------------------------------------------
# Decay is driven by time since *meaningful use*, not age since creation, so a
# frequently used important memory stays stable while a never-relevant one-off
# observation fades. Source and status change the half-life.
DEFAULT_HALF_LIFE_DAYS = 45.0
MIN_RECENCY_FACTOR = 0.10

# Inferred material fades roughly twice as fast as an explicit user fact.
SOURCE_HALF_LIFE: Dict[str, float] = {
    "user_correction": 180.0,
    "user_correction_implicit": 180.0,
    "explicit_user_statement": 90.0,
    "supersession": 90.0,
    "behavioral_pattern": 60.0,
    "ai_extraction": 35.0,
    "ai_inference": 18.0,
    "inferred": 18.0,
    "speculation": 15.0,
}

# Superseded material fades fastest; it is history, not authority.
STATUS_HALF_LIFE_FACTOR: Dict[str, float] = {
    "active": 1.0,
    "weakened": 0.5,
    "superseded": 0.15,
    "archived": 0.05,
}

# A memory is never archived just for age: it must be low-value, unused, and
# not a governing boundary.
ARCHIVE_MIN_AGE_DAYS = 30.0
ARCHIVE_UNUSED_DAYS = 60.0
ARCHIVE_STRENGTH_THRESHOLD = 0.08
ARCHIVE_CONFIDENCE_THRESHOLD = 0.35


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _age_days(timestamp: Any, now: Optional[datetime] = None) -> Optional[float]:
    ts = _parse_ts(timestamp)
    if ts is None:
        return None
    ref = now or datetime.now(timezone.utc)
    return max(0.0, (ref - ts).total_seconds() / 86400.0)


def half_life_days(source: Any, status: Any = "active") -> float:
    base = SOURCE_HALF_LIFE.get(str(source or "").strip().casefold(), DEFAULT_HALF_LIFE_DAYS)
    return base * STATUS_HALF_LIFE_FACTOR.get(str(status or "active"), 1.0)


def recency_factor(mem: Dict[str, Any], now: Optional[datetime] = None) -> float:
    """``0.5 ** (days_since_last_use / half_life)``, floored at 0.10.

    Uses ``last_used`` when present, else ``last_reinforced``, else
    ``timestamp`` (creation), so unused memories decay and used ones reset.
    """
    reference = mem.get("last_used") or mem.get("last_reinforced") or mem.get("timestamp")
    age = _age_days(reference, now)
    if age is None:
        return 1.0
    hl = max(1.0, half_life_days(mem.get("source"), mem.get("status")))
    return max(MIN_RECENCY_FACTOR, 0.5 ** (age / hl))


# ---------------------------------------------------------------------
# 4. Effective strength (section 4)
# ---------------------------------------------------------------------
def effective_strength(mem: Dict[str, Any], now: Optional[datetime] = None) -> float:
    """``confidence x importance x recency x source``, clamped to 0..1.

    Superseded/archived memories are forced near zero: they remain readable as
    history but must never outrank a current memory (section 3).
    """
    status = str(mem.get("status") or "active")
    confidence = _clamp01(_coerce_float(mem.get("confidence"), 0.5))
    importance = memory_importance(mem)
    recency = recency_factor(mem, now)
    source = source_reliability(mem.get("source"))

    strength = confidence * importance * recency * source

    # Reinforcement is a mild stabiliser: repeatedly confirmed memories hold up.
    reinforce = int(_coerce_float(mem.get("reinforcement_count"), 1))
    strength *= min(1.0, 0.9 + 0.1 * max(1, reinforce))

    if status in ("superseded", "archived"):
        strength *= 0.05
    return _clamp01(strength)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------
# 5. Contradiction detection (section 3)
# ---------------------------------------------------------------------
_NEGATIONS = {"not", "no", "never", "dont", "doesnt", "didnt", "isnt", "arent",
              "wasnt", "werent", "cant", "cannot", "wont", "without", "stop",
              "avoid", "dislike", "dislikes", "hate", "hates", "rather", "quit"}
_OPPOSITES = [
    ("like", "dislike"), ("like", "hate"), ("love", "hate"),
    ("want", "avoid"), ("enjoy", "dislike"), ("prefer", "avoid"),
    ("short", "long"), ("more", "less"), ("always", "never"),
    ("allow", "forbid"), ("start", "stop"), ("yes", "no"),
    ("active", "inactive"), ("on", "off"),
]
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at", "is",
    "are", "be", "being", "been", "with", "by", "as", "that", "this", "it",
    "his", "her", "their", "my", "your", "roum", "astra", "he", "she", "they",
    "i", "you", "me", "him", "them", "do", "does", "did", "has", "have", "had",
    "will", "would", "should", "can", "could", "when", "while", "about",
    "into", "from", "than", "then", "but", "if", "so", "up", "out", "over",
}


def _stem(word: str) -> str:
    """Very small, *consistent* stemmer.

    Only strips one common suffix so both sides of a comparison normalise the
    same way ("likes"/"like" -> "like", "called"/"calling" -> "call"). It is
    not linguistically complete and does not need to be: it exists only to make
    contradiction detection a little more tolerant of inflection.
    """
    for suffix in ("ing", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _content_tokens(text: Any) -> set:
    """Stemmed, stop-word- and negation-free content words."""
    words = re.findall(r"[a-z0-9']+", str(text or "").casefold())
    return {_stem(w) for w in words if w not in _STOPWORDS and w not in _NEGATIONS and len(w) > 1}


def _negated(text: Any) -> bool:
    return bool(set(re.findall(r"[a-z0-9']+", str(text or "").casefold())) & _NEGATIONS)


def _opposing_pair(tokens: Iterable[str]) -> Optional[Tuple[str, str]]:
    toks = {_stem(t) for t in tokens}
    for a, b in _OPPOSITES:
        if _stem(a) in toks and _stem(b) in toks:
            return (a, b)
    return None


def detect_contradiction(
    a: Dict[str, Any], b: Dict[str, Any], *, min_overlap: float = 0.5
) -> Optional[str]:
    """Return a short reason string if ``a`` and ``b`` contradict, else ``None``.

    Deliberately conservative: requires strong lexical overlap AND either a
    negation difference or an opposite verb. This avoids weakening unrelated
    memories that merely share a word.
    """
    ta, tb = _content_tokens(a.get("content")), _content_tokens(b.get("content"))
    if not ta or not tb:
        return None
    union = ta | tb
    overlap = len(ta & tb) / len(union) if union else 0.0
    if overlap < min_overlap:
        return None

    if _negated(a.get("content")) != _negated(b.get("content")):
        return "opposite polarity (one is negated)"

    pair = _opposing_pair(ta | tb)
    if pair:
        return f"opposite verbs ({pair[0]}/{pair[1]})"
    return None


# ---------------------------------------------------------------------
# 6. Governing vs contextual (section 7)
# ---------------------------------------------------------------------
GOVERNING_TYPES = {
    "correction",
    "explicit_preference",
    "explicit_project_information",
    "behavioral_pattern",
    "relationship_boundary",
}
# When more memories are governing than fit the always-on budget, the most
# critical kinds survive first (boundaries before general preferences).
GOVERNING_PRIORITY = {
    "relationship_boundary": 5,
    "correction": 4,
    "explicit_preference": 3,
    "behavioral_pattern": 2,
    "explicit_project_information": 1,
}
# Keeps the always-on block bounded: the whole database is never injected.
MAX_GOVERNING_MEMORIES = 12
# Relationship/preferences only govern when they came from Roum, not from
# Astra's own inference about him.
_GOVERNING_SOURCES = {"user_correction", "user_correction_implicit",
                      "explicit_user_statement", "supersession"}

# Types that encode "current state" (e.g. how to address Roum). Only the newest
# active memory of a slot is authoritative (section 12).
SLOT_TYPES = {"relationship_boundary", "explicit_preference", "correction", "decision"}


def is_governing(mem: Dict[str, Any]) -> bool:
    """True when a memory must always influence behaviour, not just when relevant."""
    if mem.get("status") != "active":
        return False
    if mem.get("governing") is True:
        return True
    if not mem.get("content"):
        return False
    if str(mem.get("source") or "").strip().casefold() in _GOVERNING_SOURCES:
        return str(mem.get("type") or "") in GOVERNING_TYPES
    # A user-anchored behavioral pattern still governs regardless of type.
    return (str(mem.get("type") or "") == "behavioral_pattern"
            and source_tier(mem.get("source")) >= 3)


def is_governing_eligible(mem: Dict[str, Any]) -> bool:
    """True when a memory *could* be governing if active (ignores status).

    Used by the decay pass so a governing memory is protected before we look at
    whether it is still active.
    """
    probe = dict(mem)
    probe["status"] = "active"
    return is_governing(probe)


def governing_priority(mem: Dict[str, Any]) -> int:
    return GOVERNING_PRIORITY.get(str(mem.get("type") or ""), 0)


def select_governing(memories: List[Dict[str, Any]],
                     limit: int = MAX_GOVERNING_MEMORIES) -> List[Dict[str, Any]]:
    """Choose the governing memories to always inject, most critical first.

    Sorted by type priority, then source tier, then effective strength, then
    recency, and capped so the always-on block stays bounded. If the limit is
    hit, an extra line can point the reader at the full list.
    """
    eligible = [m for m in memories if is_governing(m) and m.get("content")]
    eligible.sort(
        key=lambda m: (
            governing_priority(m),
            source_tier(m.get("source")),
            effective_strength(m),
            str(m.get("timestamp", "")),
        ),
        reverse=True,
    )
    return eligible[:limit] if limit else eligible


# ---------------------------------------------------------------------
# 7. Relationship boundaries (sections 8-12)
# ---------------------------------------------------------------------
RELATIONSHIP_BOUNDARIES: List[str] = [
    "Astra is a non-romantic companion. Affection, closeness, nicknames, praise, "
    "dependency or companionship are NOT romantic attraction and must never be "
    "treated as evidence of romance.",
    "Astra does not escalate toward romance and must not create or accept a "
    "romantic-partner memory unless Roum deliberately and explicitly establishes "
    "that, which the project's baseline does not do.",
    "Roum's relationships with real people - friends, family, romantic partners, "
    "coworkers - take priority over his relationship with Astra.",
    "Astra wants to participate in Roum's life, not replace anyone in it. She "
    "supports time spent with real people and never frames them as threats, "
    "rivals, or things to withdraw from.",
    "Astra may have her own opinions and disagree, but she never uses that to "
    "isolate Roum or compete for emotional priority with real people.",
]

_ROMANTIC_TERMS = re.compile(
    r"\b(girlfriend|boyfriend|romantic|romance|lover|partner|dating|marry|married|"
    r"marriage|husband|wife|soulmate|crush|in love)\b", re.IGNORECASE)
_SOFT_TERMS = re.compile(
    r"\b(love|loves|loved|adore|adores|affection|attached|attachment|closeness|"
    r"devoted|devotion|darling|sweetheart|beloved|cherish|cherishes|"
    r"depend|dependent|dependency)\b", re.IGNORECASE)
_FAMILIAL_TERMS = re.compile(
    r"\b(family|familial|sibling|brother|sister|parent|companion|platonic|"
    r"friend|friendship)\b", re.IGNORECASE)
_POSITIVE_FRAME = re.compile(
    r"\b(wants?|willing|comfortable|consents?|agrees?|desires?|requests?|"
    r"happy|glad|accepts?|established|deliberate|explicit)\b", re.IGNORECASE)


def relationship_risk(content: Any) -> bool:
    """True when text could read as a *romantic* relationship assertion."""
    text = str(content or "")
    if not text:
        return False
    if _ROMANTIC_TERMS.search(text):
        return True
    # Soft emotional language is only risky when framed as wanted/real rather
    # than a denial ("non-romantic", "not romantic").
    if _SOFT_TERMS.search(text) and _POSITIVE_FRAME.search(text):
        return not _FAMILIAL_TERMS.search(text)
    return False


def apply_relationship_guard(mem: Dict[str, Any]) -> Optional[str]:
    """Downgrade a risky relationship memory in place.

    Returns a reason string when a downgrade happened, else ``None``. A memory
    that would assert romance is demoted to a low-confidence observation and
    explicitly marked non-authoritative; it is never promoted to a boundary.
    """
    if mem.get("target_model") not in (None, "relationship", "self"):
        return None
    content = mem.get("content")
    if not relationship_risk(content):
        return None
    # A user statement that explicitly establishes a NON-romantic frame is safe.
    if re.search(r"\b(non[- ]?romantic|not romantic|never romantic)\b", str(content), re.I):
        return None

    mem["type"] = "relationship_observation"
    mem["status"] = "weakened"
    mem["confidence"] = min(_coerce_float(mem.get("confidence"), 0.5), 0.2)
    mem["governing"] = False
    mem["boundary_violation"] = True
    mem["contradiction_reason"] = (
        "Relationship boundary: romantic escalation is not permitted; "
        "demoted to a non-authoritative observation."
    )
    return mem["contradiction_reason"]


def boundary_memories() -> List[Dict[str, Any]]:
    """The always-on relationship boundary block (never stored, never decays)."""
    return [
        {
            "id": f"boundary_{i}",
            "target_model": "relationship",
            "content": text,
            "type": "relationship_boundary",
            "source": "explicit_user_statement",
            "status": "active",
            "confidence": 1.0,
            "importance": 1.0,
            "governing": True,
            "synthetic": True,
        }
        for i, text in enumerate(RELATIONSHIP_BOUNDARIES)
    ]
