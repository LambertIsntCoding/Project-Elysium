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

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import relational


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

# Provenance that is genuinely traceable to something Roum said or that the
# user explicitly directed. Only these are surfaced as sourceable material;
# everything else is Astra's own generation and is handled separately (see
# ``is_sourced``), so an unsourced sentence can never be presented as fact.
SOURCED_SOURCES = {"explicit_user_statement", "user_correction",
                   "user_correction_implicit", "supersession"}

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


def is_sourced(mem: Dict[str, Any]) -> bool:
    """True when a memory traces to an explicit user statement or correction.

    Everything else (Astra's own extraction/inference, or an inferred
    behavioral pattern) is *unsourced*: it is still usable as an execution
    directive, but it must never be presented to Roum as a fact about him.
    """
    return str(mem.get("source") or "").strip().casefold() in SOURCED_SOURCES


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
    # A dormant memory is still retrievable, but only weakly: it can wake up if
    # it turns out to be relevant, yet it never governs (section 6).
    if is_dormant(mem):
        strength *= 0.5
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


# Terms whose polarity (asserted vs negated) actually decides a contradiction.
# A shared negation elsewhere ("not intended as a disclaimer") is a detail, not
# a reversal, so only these carry polarity.
_POLARITY_TERMS = {
    "like", "love", "enjoy", "prefer", "want", "need", "hate", "dislike",
    "favor", "favourite", "favorite", "care", "support", "agree", "approve",
    "allow", "permit", "believe", "trust", "accept", "tolerate", "mind",
    "interested", "oppose", "reject", "recommend", "value", "desire",
}


def _negated_terms(text: Any, vocabulary: set) -> set:
    """Shared terms the text negates, e.g. ``not like`` -> ``{"like"}``.

    Scope is narrow on purpose: a negation only counts when it lands within a
    few words of a term both statements share, so a negation about something
    else in the sentence never registers as a reversal of that term.
    """
    words = re.findall(r"[a-z0-9']+", str(text or "").casefold())
    hit = set()
    for idx, word in enumerate(words):
        if word not in _NEGATIONS:
            continue
        for nxt in words[idx + 1:idx + 4]:
            stem = _stem(nxt)
            if stem in vocabulary:
                hit.add(stem)
                break
    return hit


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

    Deliberately conservative: requires that the two statements are about the
    same subject (most of the *shorter* one's content words recur in the other)
    AND that a shared term's polarity differs, or an opposite verb is present.
    This avoids weakening unrelated memories that merely share a word.
    """
    ta, tb = _content_tokens(a.get("content")), _content_tokens(b.get("content"))
    if not ta or not tb:
        return None
    shared = ta & tb
    # Containment, not Jaccard: a restatement of the same fact is often longer
    # or shorter than the original ("Roum's favorite game is Sonic Adventure."
    # vs a sentence about Sonic Adventure), which inflates the union and hides
    # the shared subject. What matters is that most of the smaller statement's
    # content is accounted for.
    overlap = len(shared) / min(len(ta), len(tb))
    if overlap < min_overlap:
        return None

    # Explicitly stated as temporary ("right now", "today"): a passing state,
    # not a durable claim, so it never contradicts a standing fact.
    if _matches(_TEMPORARY_PATTERNS, str(a.get("content") or "").casefold()) or \
            _matches(_TEMPORARY_PATTERNS, str(b.get("content") or "").casefold()):
        return None

    # Polarity is only compared on shared terms that carry a stance, and only
    # when the statements share a subject beyond that stance word. A negation
    # about an incidental detail ("not his absolute favorite") is therefore not
    # mistaken for a reversal, and "likes tea" never contradicts "does not like
    # coffee" just because both contain a negated "like".
    vocabulary = shared & _POLARITY_TERMS
    if vocabulary and (shared - _POLARITY_TERMS):
        negated_a = _negated_terms(a.get("content"), vocabulary)
        negated_b = _negated_terms(b.get("content"), vocabulary)
        if bool(negated_a) != bool(negated_b):
            return "opposite polarity (one is negated)"

    pair = _opposing_pair(ta | tb)
    if pair:
        return f"opposite verbs ({pair[0]}/{pair[1]})"
    return None


# Tags the consolidator attaches to behavioural/style feedback. Two of these
# memories are near-duplicate restatements even when they share few words.
_FEEDBACK_TAGS = {"personality_feedback", "style_critique", "style_preference",
                  "conversational_vibe", "communication_style"}

# Types that encode a revisable behavioural stance, and so may be collapsed as
# restatements. Plain facts are excluded: two similar facts are still two facts.
_COLLAPSIBLE_TYPES = {"explicit_preference", "correction", "behavioral_pattern",
                      "self_preference", "self_observation", "relationship_observation"}


def _feedback_keywords(mem: Dict[str, Any]) -> set:
    """Keyword tokens of a feedback memory, used as a topic signature.

    The consolidator generates short keywords ("stop analyzing", "quirky",
    "brevity") that are far more stable across restatements than the verbose
    sentence it also generates, so they are the reliable overlap signal.
    """
    return _content_tokens(" ".join(_clean_str_list(mem.get("keywords"))))


def detect_restatement(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[str]:
    """Return a reason when ``a`` and ``b`` state the *same* thing, else None.

    ``detect_contradiction`` is deliberately conservative and only fires on a
    polarity reversal, so a restated preference with different wording stays
    active alongside its original. That is how the store accumulated dozens of
    near-duplicate style/tone preferences. This is the complementary,
    non-destructive signal: it never marks a memory false, it only lets the
    newer wording supersede the older (exactly like ``is_project_state`` does
    for project updates), so history is preserved and the pair stops competing.

    Conservative on purpose: it fires only when the two memories share a
    feedback tag or a topic keyword *and* their content has real subject
    overlap. It is therefore only ever consulted for memories whose content is
    feedback about Astra's behaviour.
    """
    if _matches(_TEMPORARY_PATTERNS, str(a.get("content") or "").casefold()) or \
            _matches(_TEMPORARY_PATTERNS, str(b.get("content") or "").casefold()):
        return None
    if str(a.get("type") or "") != str(b.get("type") or ""):
        return None

    shared_tags = (set(_clean_str_list(a.get("tags"))) & set(_clean_str_list(b.get("tags"))))
    tag_overlap = bool(shared_tags & _FEEDBACK_TAGS)
    kw_shared = _feedback_keywords(a) & _feedback_keywords(b)

    ta, tb = _content_tokens(a.get("content")), _content_tokens(b.get("content"))
    if not ta or not tb:
        return None
    shared = ta & tb
    containment = len(shared) / min(len(ta), len(tb))
    jaccard = len(shared) / len(ta | tb)

    # A shared *feedback* tag is what makes two memories comparable at all; a
    # shared topic keyword only corroborates. Requiring the tag is what keeps
    # two genuinely different traits that happen to share a word (e.g. both
    # mention "feelings") from being collapsed.
    if tag_overlap:
        # Real subject overlap, or a shared topic keyword plus some overlap.
        if containment >= 0.30 and len(shared) >= 3:
            return "restates the same feedback (shared tag and subject)"
        if kw_shared and len(shared) >= 2:
            return "restates the same feedback (shared tag and topic keyword)"
    # Same content with only reordered/verbose wording. Needs substantial shared
    # content, not a common sentence frame.
    if len(shared) >= 4 and (containment >= 0.55 or jaccard >= 0.45):
        return "restates the same preference (near-identical wording)"
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
    # Dormant memories are soft-demoted: still retrievable, but never governing.
    if is_dormant(mem):
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


logger = logging.getLogger("astra.memory")

VALID_TARGET_MODELS = {"roum", "self", "relationship"}

# --- Phase 1 memory governance -----------------------------------------
# Every candidate must land in exactly one bucket. Only PERSISTENT_* become
# durable memories; everything else stays out of the memory files.
CLASSIFICATION_PERSISTENT = {
    "persistent_user_fact", "persistent_user_preference",
    "persistent_project_information", "behavioral_pattern",
    "uncertain_inference", "self_fact", "self_preference", "self_observation",
    "self_belief", "relationship_event", "relationship_observation", "decision",
}
CLASSIFICATION_TRANSIENT = {"temporary_context", "no_memory"}
VALID_CLASSIFICATIONS = CLASSIFICATION_PERSISTENT | CLASSIFICATION_TRANSIENT

# Classification -> (target model, stored memory type).
CLASSIFICATION_TYPES: Dict[str, Tuple[str, str]] = {
    "persistent_user_fact": ("roum", "explicit_fact"),
    "persistent_user_preference": ("roum", "explicit_preference"),
    "persistent_project_information": ("roum", "explicit_project_information"),
    "behavioral_pattern": ("roum", "behavioral_pattern"),
    "uncertain_inference": ("roum", "uncertain_inference"),
    "self_fact": ("self", "self_fact"),
    "self_preference": ("self", "self_preference"),
    "self_observation": ("self", "self_observation"),
    "self_belief": ("self", "self_belief"),
    "relationship_event": ("relationship", "relationship_event"),
    "relationship_observation": ("relationship", "relationship_observation"),
    "decision": ("roum", "decision"),
}

# Reverse of CLASSIFICATION_TYPES, used by ``add_memory`` to recover the
# classification label for a stored type so the routing guard can run on direct
# writes too (not just consolidator decisions).
_TYPE_TO_CLASSIFICATION: Dict[str, str] = {
    mem_type: classification for classification, (_, mem_type) in CLASSIFICATION_TYPES.items()
}

# A self-memory may only be durable when the declaration was deliberate
# (Astra's governing system) or backed by repetition. A single generated
# sentence can never establish a self-preference (sections 4).
SELF_DURABLE_CLASSIFICATIONS = {"self_fact", "self_preference"}
DELIBERATE_SELF_SOURCES = {"governing_declaration", "explicit_declaration", "user_correction"}
SELF_DURABLE_MIN_EVIDENCE = 3

# Only an explicit current user statement (or correction) may supersede.
EXPLICIT_USER_SOURCES = {"explicit_user_statement", "user_correction", "user_correction_implicit"}

# Temporary context defaults: how long a scratch item stays available, and how
# many turns of scratch are kept at once.
DEFAULT_TEMPORARY_TTL_SECONDS = 3600
DEFAULT_TEMPORARY_CAP = 20

# Utility buckets (section 6). Dormant is a *soft* demotion: a dormant memory
# is still retrievable (so it can wake up) but never governing.
UTILITY_ACTIVE = "active"
UTILITY_DORMANT = "dormant"
VALID_UTILITIES = {UTILITY_ACTIVE, UTILITY_DORMANT}
DORMANT_MIN_AGE_DAYS = 45
DORMANT_UNUSED_DAYS = 45
DORMANT_STRENGTH_THRESHOLD = 0.18
# A governing memory stays active unless its confidence has fallen below this,
# so a low-confidence user preference can go dormant but a firm one cannot.
GOVERNING_ACTIVE_MIN_CONFIDENCE = 0.5


def classification_to_type(classification: str) -> Tuple[str, str]:
    """Map a classification label to ``(target_model, memory_type)``.

    Unknown or transient labels resolve to ``("roum", "uncertain_inference")``
    so a classifier that drifts can never create an explicit user fact.
    """
    return CLASSIFICATION_TYPES.get(classification, ("roum", "uncertain_inference"))


def is_persistent_classification(classification: str) -> bool:
    return classification in CLASSIFICATION_PERSISTENT


# Wording that addresses Astra directly rather than describing Roum. A sentence
# like "Astra should stop narrating her analysis" is a *directive to Astra*, not
# a fact about Roum, so it belongs in the self model where it can actually shape
# behaviour. This is what let 69 of 169 roum memories become instructions aimed
# at Astra instead of facts about him.
_ASTRA_DIRECTED_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:astra|the (?:ai|assistant|companion)|you)\b[^.?!]{0,40}?"
    r"\b(?:should|must|needs? to|has to|ought to|is to|is not to|will not|won't|"
    r"can't|cannot|do not|don't|never|always|stop|avoid|refrain)\b",
    r"\b(?:do not|don't|never|please)\s+\w+",
    r"\byour (?:personality|tone|style|behaviou?r|responses?|replies|language|"
    r"emotional state|memory|memories)\b",
    r"\byou (?:sound|are being|'re being|talk like|feel like|should|must|need)\b",
))


def is_astra_directed(content: Any) -> bool:
    """True when the statement is an instruction about Astra, not about Roum."""
    return _matches(_ASTRA_DIRECTED_PATTERNS, str(content or "").casefold())


# Feedback about Astra's behaviour is a self-model observation, never an
# automatic edit to her personality or a "fact" about Roum (section 7).
_FEEDBACK_TAG_NAMES = {"personality_feedback", "style_critique"}


def route_candidate_target(classification: str, mem_type: str, target_model: str,
                    content: str, tags: Any) -> Tuple[str, str]:
    """Route a persistent candidate to ``(target_model, memory_type)``.

    The classifier already chose a bucket; this corrects the one case it gets
    structurally wrong: material that is *about Astra* being filed under Roum.
    """
    if target_model == "self":
        return "self", mem_type
    if classification == "behavioral_pattern":
        # A pattern about Astra's conduct is a self-pattern, not a trait of Roum.
        return ("self", "behavioral_pattern") if is_astra_directed(content) else ("roum", mem_type)
    if classification in ("persistent_user_preference", "persistent_user_fact",
                          "persistent_project_information", "decision"):
        tag_names = {str(t).strip().casefold() for t in _clean_str_list(tags)}
        if tag_names & _FEEDBACK_TAG_NAMES or is_astra_directed(content):
            return "self", "self_observation"
    # Anything else keeps the bucket the classifier already chose.
    return target_model, mem_type


def classify_candidate(content: str, *, explicit: bool, mem_type: str,
                       target_model: str, source: str, confidence: float = 0.7) -> str:
    """Decide which bucket a candidate memory belongs in.

    Deterministic and model-independent: this is the *policy* layer, and it
    exists so the system decides what to remember rather than the generator.
    """
    text = str(content or "").strip()
    if not text:
        return "no_memory"
    lowered = text.casefold()

    # A question, or a one-off activity report, is context rather than a fact -
    # unless the wording itself establishes a recurring trait ("I always...").
    if text.endswith("?") or (_matches(_TEMPORARY_PATTERNS, lowered)
                              and not _matches(_PREFERENCE_PATTERNS, lowered)):
        return "temporary_context"

    if explicit:
        if _matches(_DECISION_PATTERNS, lowered):
            return "decision"
        if mem_type == "explicit_project_information":
            return "persistent_project_information"
        if mem_type in ("explicit_preference", "correction"):
            return "persistent_user_preference"
        # A trait stated in preference terms is a preference even when the
        # caller labelled it a plain fact.
        if _matches(_PREFERENCE_PATTERNS, lowered):
            return "persistent_user_preference"
        if mem_type == "explicit_fact":
            return "persistent_user_fact"
        # An explicit statement is trusted; route by the model it targets.
        return {
            "self": "self_fact",
            "relationship": "relationship_event",
        }.get(target_model, "persistent_user_fact")

    # Non-explicit: never allowed to become an explicit fact (provenance guard).
    if source in ("behavioral_pattern",):
        return "behavioral_pattern"
    if target_model == "self":
        return "self_observation"
    if target_model == "relationship":
        return "relationship_observation"
    return "uncertain_inference"


def is_durable_self_memory(classification: str, source: str,
                           reinforcement_count: int) -> bool:
    """Gate for self-model protection (section 4).

    A self-preference/fact is durable only when Astra's governing system
    declared it deliberately or it has been reinforced across conversations.
    """
    if classification not in SELF_DURABLE_CLASSIFICATIONS:
        return True
    if source in DELIBERATE_SELF_SOURCES:
        return True
    return int(reinforcement_count or 1) >= SELF_DURABLE_MIN_EVIDENCE


def may_supersede(new_source: str, new_classification: str) -> bool:
    """Only an explicit user statement/correction may supersede a memory."""
    return new_source in EXPLICIT_USER_SOURCES and new_classification != "temporary_context"


def mark_dormant(mem: Dict[str, Any], reason: str) -> None:
    """Soft-demote a stale, low-value memory in place (never deletes it)."""
    mem["utility"] = UTILITY_DORMANT
    mem["dormant_at"] = _now()
    mem["dormant_reason"] = str(reason)
    mem["governing"] = False


def is_dormant(mem: Dict[str, Any]) -> bool:
    return str(mem.get("utility") or UTILITY_ACTIVE) == UTILITY_DORMANT


def memory_utility(mem: Dict[str, Any]) -> str:
    """Classify a memory as ``active`` or ``dormant`` from its metadata.

    ``old but important`` (recently used, high confidence, governing) stays
    active; ``old and irrelevant`` (weak, unused, low confidence) is dormant.
    """
    if is_dormant(mem):
        return UTILITY_DORMANT
    if mem.get("status") not in ("active", "weakened"):
        return UTILITY_DORMANT
    # Genuinely important memories stay active: a governing memory that still
    # carries reasonable confidence, or one that has actually been used.
    if int(mem.get("use_count", 0)) > 0:
        return UTILITY_ACTIVE
    if (is_governing_eligible(mem)
            and _coerce_float(mem.get("confidence"), 1.0) >= GOVERNING_ACTIVE_MIN_CONFIDENCE):
        return UTILITY_ACTIVE
    now = datetime.now(timezone.utc)
    age = _age_days(mem.get("created_at") or mem.get("timestamp"), now)
    unused = _age_days(mem.get("last_used") or mem.get("timestamp"), now)
    if age is None or unused is None:
        return UTILITY_ACTIVE
    if (unused >= DORMANT_UNUSED_DAYS
            and effective_strength(mem, now) < DORMANT_STRENGTH_THRESHOLD
            and _coerce_float(mem.get("confidence"), 1.0) < ARCHIVE_CONFIDENCE_THRESHOLD):
        return UTILITY_DORMANT
    return UTILITY_ACTIVE


# Transient cues: things worth understanding now but not worth remembering.
_TEMPORARY_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(yesterday|last night|this morning|today|tomorrow|tonight|just now|earlier today)\b",
    r"\bi (?:just|currently) (?:played|watched|ate|read|bought|finished|started|did)\b",
    r"\bright now\b", r"\bat the moment\b", r"\bcurrently (?:talking|discussing|working on|playing)\b",
    r"\bwhat (?:i am|i'm) (?:talking|asking|saying) about\b",
))
_DECISION_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:we|i) (?:decided|agreed|chose|settled on|will go with)\b",
    r"\b(?:let'?s|we'?ll) (?:go with|use|do|stick with)\b",
    r"\bthe plan is\b", r"\bfinal decision\b",
))
# Preference wording: a durable trait rather than a one-off statement.
_PREFERENCE_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:my|his) favou?rite\b", r"\bi (?:really )?(?:like|love|enjoy|prefer|hate|dislike)\b",
    r"\bi(?:'m| am) (?:really )?into\b", r"\bi always\b", r"\bi never\b",
    r"\bi(?:'d| would) rather\b", r"\bplease (?:always|never|don't|do not)\b",
    r"\bi want you to\b", r"\bi prefer\b",
))
# Project state is revisable: it describes how things are now, not an
# immutable historical fact (section 1, tests I).
_PROJECT_STATE_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:is|are|was|were) (?:currently |still |now )?"
    r"(?:stalled|blocked|paused|on hold|delayed|in progress|underway|waiting|pending|done|finished)\b",
    r"\b(?:waiting|blocked) (?:on|for)\b",
))


def _matches(patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)


def is_project_state(content: str) -> bool:
    return _matches(_PROJECT_STATE_PATTERNS, str(content or "").casefold())


def same_project_subject(a: str, b: str) -> bool:
    """True when two project-state statements are about the same subject.

    Requires a shared *distinctive* token (a project name or other unusual
    word), not merely shared filler, so two different projects are never
    treated as one another's stale state. A restatement naturally adds detail,
    so overall overlap is deliberately not required.
    """
    ta, tb = _content_tokens(a), _content_tokens(b)
    shared = ta & tb
    return any(
        tok not in _COMMON_PROJECT_WORDS and len(tok) >= 3 for tok in shared
    )


# Filler that appears in almost any project sentence and therefore does not
# identify *which* project is being discussed.
_COMMON_PROJECT_WORDS = {
    "project", "currently", "current", "still", "now", "work", "working",
    "on", "in", "progress", "stalled", "blocked", "paused", "hold", "delayed",
    "underway", "waiting", "pending", "done", "finished", "because", "approval",
    "publisher", "wait", "start", "started", "develop", "development",
}


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
def atomic_save(filepath: str, data: Any, backup: bool = True,
                *, compact: bool = True) -> None:
    """
    Save JSON atomically: write to a temp file, fsync, then replace.
    If `backup` is set, the previous version is kept as "<file>.bak".

    Files are written in a dense one-record-per-line layout by default. The
    format is plain JSON (no custom decoder), it is still readable, and it is
    ~30% smaller than ``indent=2``; `compact=False` keeps the pretty form for
    one-off, human-facing dumps.
    """
    filepath = os.path.abspath(filepath)
    dir_name = os.path.dirname(filepath)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=dir_name, prefix=os.path.basename(filepath) + ".", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if compact:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            else:
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
# Compact on-disk layout
# ---------------------------------------------------------------------
# A record's *meaning* is a small core; the rest is either derived at read time
# or a no-op default. Storing only the core keeps the files from growing without
# bound, without changing a single reader - every reader already uses ``.get``
# with a safe default. These are the fields dropped on write:
#
#   * ``effective_strength`` - recomputed by ``effective_strength()``; the cached
#     copy was written by ``apply_decay`` and never read.
#   * ``source_type``       - a duplicate of ``source_tier(source)``.
#   * ``created_at``/``last_used``/``last_reinforced`` - all default to
#     ``timestamp``; only kept when they actually differ (i.e. after real use).
#   * empty ``tags``/``keywords``/``contradicts``, and the no-op defaults
#     ``use_count``/``contradiction_count``/``reinforcement_count`` - all
#     re-created on read.
#   * ``None`` supersession pointers and ``slot``.
_DERIVED_FIELDS = ("effective_strength", "source_type")
_TIMESTAMP_ALIASES = ("created_at", "last_used", "last_reinforced")
_NOOP_DEFAULTS = {"use_count": 0, "contradiction_count": 0, "reinforcement_count": 1}
_EMPTY_LIST_FIELDS = ("tags", "keywords", "contradicts")
_NULL_DROP_FIELDS = ("supersedes", "superseded_by", "slot")


def compact_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``record`` with redundant/no-op fields removed (never mutates)."""
    out = dict(record)
    for field in _DERIVED_FIELDS:
        out.pop(field, None)
    timestamp = out.get("timestamp")
    for field in _TIMESTAMP_ALIASES:
        if out.get(field) == timestamp:
            out.pop(field, None)
    for field, default in _NOOP_DEFAULTS.items():
        if out.get(field) == default:
            out.pop(field, None)
    for field in _EMPTY_LIST_FIELDS:
        if not out.get(field):
            out.pop(field, None)
    for field in _NULL_DROP_FIELDS:
        if out.get(field) is None:
            out.pop(field, None)
    return out


# Fields a reader may subscript directly. The compact layout omits them when
# they hold their default, so reads re-materialise them: an absent counter means
# "never used/contradicted", and an absent lifecycle stamp means "same as
# creation". Callers keep the old contract without the bytes on disk.
_READ_DEFAULTS: Dict[str, Any] = {
    "use_count": 0,
    "contradiction_count": 0,
    "reinforcement_count": 1,
}
_NULL_READ_FIELDS = ("supersedes", "superseded_by")
_LIST_READ_FIELDS = ("tags", "keywords", "contradicts")


def _present(mem: Dict[str, Any]) -> Dict[str, Any]:
    """A deep copy of ``mem`` with compact-layout defaults restored for reading."""
    out = copy.deepcopy(mem)
    for field, default in _READ_DEFAULTS.items():
        out.setdefault(field, default)
    for field in _NULL_READ_FIELDS:
        out.setdefault(field, None)
    for field in _LIST_READ_FIELDS:
        out.setdefault(field, [])  # a fresh list per call, never shared
    timestamp = out.get("timestamp")
    for field in _TIMESTAMP_ALIASES:
        out.setdefault(field, timestamp)
    return out


# ---------------------------------------------------------------------
# Time-based views
# ---------------------------------------------------------------------
TIMELINE_GRANULARITIES = ("day", "month", "year")
_UNKNOWN_PERIOD = "unknown"


def _period_key(timestamp: Any, granularity: str) -> str:
    ts = _parse_ts(timestamp)
    if ts is None:
        return _UNKNOWN_PERIOD
    if granularity == "day":
        return ts.strftime("%Y-%m-%d")
    if granularity == "year":
        return ts.strftime("%Y")
    return ts.strftime("%Y-%m")


def timeline(memories: Iterable[Dict[str, Any]], granularity: str = "month") -> Dict[str, List[Dict[str, Any]]]:
    """Bucket memories by day/month/year (from ``timestamp``) - a cheap index.

    Pure and read-only: it groups the copies the store already hands out, so it
    adds no storage and cannot disturb the byte-for-byte read guarantees.
    """
    if granularity not in TIMELINE_GRANULARITIES:
        raise ValueError(f"granularity must be one of {TIMELINE_GRANULARITIES}")
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for mem in memories:
        buckets.setdefault(_period_key(mem.get("timestamp"), granularity), []).append(mem)
    return buckets


def timeline_summary(memories: Iterable[Dict[str, Any]],
                     granularity: str = "month") -> List[Tuple[str, int]]:
    """``[(period, count), ...]`` newest first, with any undated bucket last."""
    buckets = timeline(memories, granularity)
    ordered = sorted((k for k in buckets if k != _UNKNOWN_PERIOD), reverse=True)
    if _UNKNOWN_PERIOD in buckets:
        ordered.append(_UNKNOWN_PERIOD)
    return [(period, len(buckets[period])) for period in ordered]


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
class TemporaryContext:
    """Conversation scratch space: available now, never written to disk.

    Temporary context is for things worth *understanding* this session but not
    worth permanently remembering - what Roum is currently talking about, a
    short-term plan, a passing emotional note. It is deliberately kept out of
    the persistent memory files and is capped and time-limited so it cannot
    grow without bound.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TEMPORARY_TTL_SECONDS,
                 cap: int = DEFAULT_TEMPORARY_CAP):
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.cap = max(1, int(cap))
        self._items: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def add(self, content: str, *, kind: str = "context", source: str = "ai_extraction",
            conversation_id: Optional[str] = None) -> Optional[str]:
        text = str(content or "").strip()
        if not text:
            return None
        item = {
            "id": _new_id("tmp"),
            "content": text,
            "kind": str(kind or "context"),
            "source": str(source or "ai_extraction"),
            "conversation_id": conversation_id,
            "created_at": _now(),
        }
        with self._lock:
            self._items.append(item)
            del self._items[:-self.cap]  # keep only the newest `cap` items
        return item["id"]

    def _live(self, now: datetime) -> List[Dict[str, Any]]:
        live: List[Dict[str, Any]] = []
        for item in self._items:
            age = _days_since(item.get("created_at"), now)
            if age is not None and age * 86400.0 > self.ttl_seconds:
                continue
            live.append(item)
        return live

    def get_context(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Unexpired scratch items, newest first."""
        with self._lock:
            live = self._live(datetime.now(timezone.utc))
        live.reverse()
        if limit is not None:
            live = live[:limit]
        return copy.deepcopy(live)

    def purge(self, now: Optional[datetime] = None) -> int:
        """Drop expired items; returns how many were removed."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            before = len(self._items)
            self._items = self._live(now)
            return before - len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items = []

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


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
        # Conversation scratch: deliberately NOT a file. It lives only for the
        # life of the process so transient context never clutters the stores.
        self.temporary = TemporaryContext()

    # ---- temporary context (section 5) -------------------------------
    def add_temporary_context(self, content: str, *, kind: str = "context",
                              source: str = "ai_extraction",
                              conversation_id: Optional[str] = None) -> Optional[str]:
        return self.temporary.add(content, kind=kind, source=source,
                                  conversation_id=conversation_id)

    def get_temporary_context(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.temporary.get_context(limit=limit)

    def purge_temporary_context(self) -> int:
        return self.temporary.purge()

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
        data = self._get_list(key)
        if key != "journal":
            data = [compact_record(m) for m in data]
        atomic_save(self._path(key), data)

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
        # Only the meaningful core is seeded. Everything a reader can derive or
        # default (created_at/last_used, use_count, contradiction_count,
        # reinforcement_count, empty lists, null pointers, source_type) is left
        # off disk entirely - ``compact_record`` enforces the same shape on the
        # way out, so a record stays small however it was built.
        memory: Dict[str, Any] = {
            "id": _new_id("mem"),
            "target_model": target_model,
            "content": content,
            "type": mem_type,
            "source": source,
            "status": "active",
            "timestamp": now,
            "confidence": _clamp_confidence(confidence),
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
        if is_astra_source(source) and memory.get("type") in {
            "explicit_fact", "explicit_preference", "explicit_project_information",
        }:
            memory["type"] = "uncertain_inference"
        # Relationship guard (sections 8-10): ordinary affection must never
        # become a romantic instruction.
        apply_relationship_guard(memory)
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

        # Structural routing guard (section 7): an instruction about Astra is a
        # self-observation, never a durable fact about Roum. Applied here rather
        # than only in the consolidator so every write path is covered. The
        # self-protection block below then decides whether it stays durable.
        classification = _TYPE_TO_CLASSIFICATION.get(mem_type, "")
        target_model, mem_type = route_candidate_target(
            classification, mem_type, target_model, content, tags
        )

        with self._lock:
            # Self-model protection (section 4): a single generated sentence
            # must not create a durable self-fact/self-preference. It is stored
            # as an observation instead, and only *repeated* evidence promotes
            # it. A deliberate declaration is trusted immediately.
            if (target_model == "self" and mem_type in SELF_DURABLE_CLASSIFICATIONS
                    and not is_durable_self_memory(mem_type, source, 1)):
                prior = self._find_duplicate("self", "self_observation", content)
                if prior is None:
                    with self._transaction(target_model):
                        observed = self._build_memory(
                            "self", content, "self_observation", source,
                            keywords, tags, min(_clamp_confidence(confidence), 0.5), extra,
                        )
                        self.memories["self"].append(observed)
                    return observed["id"]
                with self._transaction(target_model):
                    self._reinforce(prior, _clean_str_list(keywords), _clean_str_list(tags))
                    if is_durable_self_memory(mem_type, source,
                                              int(prior.get("reinforcement_count", 1))):
                        prior["type"] = mem_type
                        prior["promoted_at"] = _now()
                        prior["confidence"] = _clamp_confidence(confidence)
                return prior["id"]

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
            source_tier(mem.get("source")),
            effective_strength(mem),
            str(mem.get("last_used") or mem.get("timestamp") or ""),
        )

    def _resolve_contradictions(self, target_model: str, new_mem: Dict[str, Any],
                                candidates: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Weaken older memories the new one contradicts. Never deletes them.

        Returns the ids of the memories that were weakened. The weaker side
        keeps its content (history is preserved) but loses authority, gains a
        ``contradiction_reason`` and a ``superseded_by`` pointer. ``candidates``
        overrides which memories are compared (used by reconciliation to
        consider only the strictly older records).
        """
        weakened: List[str] = []
        for old in (self.memories[target_model] if candidates is None else candidates):
            if old is new_mem or old.get("status") in ("superseded", "archived"):
                continue
            if old.get("id") == new_mem.get("id"):
                continue
            # Already linked: re-learning the same content must not re-count the
            # contradiction or weaken the same memory twice.
            if new_mem.get("id") in (old.get("contradicts") or []):
                continue
            reason = detect_contradiction(old, new_mem)

            # Revisable project state (section 1, tests I): a newer statement
            # about how a project currently *is* supersedes the older state
            # without marking the older one false - it was true at the time.
            # Checked before lexical contradiction, because two states of the
            # same project need not be worded as opposites.
            if (is_project_state(old.get("content"))
                    and is_project_state(new_mem.get("content"))
                    and same_project_subject(old.get("content"), new_mem.get("content"))
                    and new_mem.get("source") in EXPLICIT_USER_SOURCES):
                if old.get("status") == "active":
                    old["status"] = "superseded"
                    old["superseded_by"] = new_mem.get("id")
                    old["superseded_at"] = _now()
                    old["supersession_reason"] = "project state updated (state, not a false claim)"
                    old["contradiction_reason"] = (
                        f"Superseded by project-state update {new_mem.get('id')}"
                    )
                    old["contradicts"] = _clean_str_list(
                        list(old.get("contradicts") or []) + [new_mem.get("id")]
                    )
                    old["contradiction_count"] = int(old.get("contradiction_count", 0)) + 1
                    new_mem["supersedes"] = old.get("id")
                    weakened.append(old["id"])
                continue

            if not reason:
                # Not a contradiction (no polarity reversal): the two may still
                # be the *same* preference said differently. Collapsing those is
                # how the store stops accumulating near-duplicate restatements.
                # Gated on ``not reason`` so a genuine reversal keeps its
                # "weakened" treatment and is never downgraded to a restatement.
                if (old.get("type") in _COLLAPSIBLE_TYPES
                        and new_mem.get("type") in _COLLAPSIBLE_TYPES
                        and old.get("status") == "active"):
                    restatement = detect_restatement(old, new_mem)
                    if restatement:
                        old["status"] = "superseded"
                        old["superseded_by"] = new_mem.get("id")
                        old["superseded_at"] = _now()
                        old["supersession_reason"] = restatement
                        old["contradiction_reason"] = (
                            f"Restated by {new_mem.get('id')}: {restatement}"
                        )
                        new_mem["supersedes"] = old.get("id")
                        weakened.append(old["id"])
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

    # ---- relational preference (Astra -> Roum) -------------------------
    def accumulate_relational_event(self, event: str, *, subject: str = relational.ROUM,
                                    text: str = "", **context) -> Dict[str, Any]:
        """Fold one command-fulfilment event into Astra's relational state.

        The state lives in a single relationship-model memory so it persists,
        stays auditable (via ``/relationship``), and keeps its evidence on disk
        alongside everything else. Only an *event* moves it - never a prompt
        build - and each subject accumulates separately, so nothing here can
        generalize a Roum-specific preference to another person.
        """
        content = relational.memory_content(subject)
        with self._lock:
            existing = None
            for mem in self.memories["relationship"]:
                if (mem.get("status") == "active"
                        and relational.affinity_memory_filter(mem)):
                    existing = mem
                    break
            state = relational.record_event(
                existing.get("affinity_state") if existing else None,
                event, subject=subject, text=text, **context,
            )
            if existing is None:
                mem_id = self.add_memory(
                    target_model="relationship",
                    content=content,
                    mem_type="relationship_observation",
                    source="ai_extraction",
                    tags=relational.memory_tags(),
                    keywords=relational.memory_keywords(subject),
                    confidence=0.5,
                    affinity_state=state,
                )
                return self.get_memory("relationship", mem_id)
            self.update_memory(
                "relationship", existing["id"],
                tags=relational.memory_tags(),
                keywords=relational.memory_keywords(subject),
                affinity_state=state,
            )
            return self.get_memory("relationship", existing["id"])

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
                    strength = effective_strength(mem, now)

                    if is_governing_eligible(mem):
                        continue  # core boundaries never decay into worthlessness

                    age = _days_since(mem.get("created_at") or mem.get("timestamp"), now)
                    unused = _days_since(mem.get("last_used") or mem.get("timestamp"), now)
                    if age is None or unused is None:
                        continue
                    # Archiving needs ALL of: weak, low confidence, old, unused.
                    # Age alone is never enough, and nothing is deleted outright.
                    if (strength < ARCHIVE_STRENGTH_THRESHOLD
                            and _clamp_confidence(mem.get("confidence"), 0.0) < ARCHIVE_CONFIDENCE_THRESHOLD
                            and age >= ARCHIVE_MIN_AGE_DAYS
                            and unused >= ARCHIVE_UNUSED_DAYS):
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
        self.apply_utility_decay()
        self.purge_temporary_context()
        return True

    def apply_utility_decay(self) -> Dict[str, int]:
        """Mark stale, low-value memories dormant (section 6).

        This is the soft alternative to deletion: a dormant memory keeps its
        content and stays retrievable, but it stops governing and ranks lower.
        Age alone is never enough - it must also be weak, unused and
        low-confidence. Nothing is deleted.
        """
        dormant = 0
        for target_model in sorted(VALID_TARGET_MODELS):
            with self._transaction(target_model):
                for mem in self.memories[target_model]:
                    if is_dormant(mem) or mem.get("status") not in ("active", "weakened"):
                        continue
                    if memory_utility(mem) != UTILITY_DORMANT:
                        continue
                    mark_dormant(mem, "stale and low-value (unused, weak, low confidence)")
                    dormant += 1
        return {"dormant": dormant}

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

    def reconcile_contradictions(self) -> int:
        """Link same-subject contradictions that were never resolved.

        A store written before the detector recognised restatements can hold
        memories that should have superseded one another but are all still
        ``active``. Replaying resolution oldest-first lets the newer statement
        win, exactly as if it had just been learned, so the store heals without
        any memory being deleted.
        """
        resolved = 0
        for target_model in sorted(VALID_TARGET_MODELS):
            ordered = sorted(
                self.memories[target_model],
                key=lambda m: str(m.get("timestamp", "")),
            )
            with self._lock:
                snapshot = copy.deepcopy(self.memories[target_model])
                earlier: List[Dict[str, Any]] = []
                for mem in ordered:
                    if mem.get("status") == "active":
                        resolved += len(self._resolve_contradictions(target_model, mem, earlier))
                    earlier.append(mem)
                if self.memories[target_model] != snapshot:
                    self._save_model(target_model)
                else:
                    self.memories[target_model] = snapshot
        return resolved

    # ---- current state & governing reads ------------------------------
    def get_governing_memories(self, target_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Active memories that must always apply, regardless of the query."""
        models = [target_model] if target_model else sorted(VALID_TARGET_MODELS)
        out: List[Dict[str, Any]] = []
        with self._lock:
            for name in models:
                for mem in self.memories[name]:
                    if is_governing(mem) and mem.get("content"):
                        out.append(_present(mem))
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
                        state[slot] = _present(mem)
        return [state[k] for k in sorted(state)]

    def update_memory(self, target_model: str, mem_id: str, *, content: Optional[str] = None,
                      confidence: Optional[float] = None, keywords: Optional[List[str]] = None,
                      tags: Optional[List[str]] = None, mem_type: Optional[str] = None,
                      **extra) -> Dict[str, Any]:
        """Edit selected fields of a memory; returns a copy of the updated memory.

        ``extra`` lets a caller persist a structured field that belongs to the
        record (e.g. accumulated relational state). Store-managed identity
        fields stay off-limits; everything else is written as given.
        """
        self._check_target(target_model)
        if all(v is None for v in (content, confidence, keywords, tags, mem_type)) and not extra:
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
                for field, value in extra.items():
                    if field in _STORE_MANAGED_FIELDS:
                        raise ValueError(f"'{field}' is managed by the store and cannot be overridden")
                    mem[field] = value
                mem["updated_at"] = _now()
            return _present(mem)

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
            return _present(self._locate(target_model, mem_id)[1])

    def get_memories(self, target_model: str, status: Optional[str] = "active",
                     mem_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Copies of memories filtered by status (None = all) and optionally type."""
        self._check_target(target_model)
        with self._lock:
            return [
                _present(m) for m in self.memories[target_model]
                if (status is None or m.get("status") == status)
                and (mem_type is None or m.get("type") == mem_type)
            ]

    def get_active_memories(self, target_model: str) -> List[Dict[str, Any]]:
        return self.get_memories(target_model, status="active")

    def find_contradiction(self, target_model: str, content: str,
                           mem_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the first active memory that ``content`` contradicts, if any.

        Used by the consolidator to decide *before* writing whether a new
        explicit statement should supersede an existing memory rather than
        coexist with it (section 2).
        """
        self._check_target(target_model)
        probe = {"content": self._clean_content(content), "type": mem_type,
                 "status": "active", "source": "explicit_user_statement"}
        with self._lock:
            for mem in self.memories[target_model]:
                if mem.get("status") != "active":
                    continue
                if detect_contradiction(mem, probe):
                    return _present(mem)
        return None

    def get_retrievable_memories(self, target_model: str) -> List[Dict[str, Any]]:
        """Active + weakened memories: usable context, ranked by strength later.

        Superseded and archived memories are excluded, so a stale value can
        never be injected as an instruction.
        """
        self._check_target(target_model)
        with self._lock:
            return [
                _present(m) for m in self.memories[target_model]
                if m.get("status") in ("active", "weakened")
            ]

    def get_timeline(self, target_model: Optional[str] = None, *,
                     granularity: str = "month") -> List[Tuple[str, int]]:
        """``[(period, count), ...]`` newest first, across one or all models.

        A cheap date index over the records already in memory - no extra storage
        and no prompt cost, so "what happened in August" is a single call.
        """
        memories: List[Dict[str, Any]] = []
        for name in ([target_model] if target_model else sorted(VALID_TARGET_MODELS)):
            memories.extend(self.get_memories(name, status=None))
        return timeline_summary(memories, granularity)

    def get_period(self, period: str, target_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every memory in a day/month/year bucket (e.g. ``"2026-09"``), newest first."""
        text = str(period).strip()
        granularity = "year" if len(text) == 4 else "day" if len(text) == 10 else "month"
        memories: List[Dict[str, Any]] = []
        for name in ([target_model] if target_model else sorted(VALID_TARGET_MODELS)):
            memories.extend(self.get_memories(name, status=None))
        bucket = [m for m in memories if _period_key(m.get("timestamp"), granularity) == text]
        bucket.sort(key=lambda m: str(m.get("timestamp", "")), reverse=True)
        return bucket

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
                    "by_utility": dict(Counter(memory_utility(m) for m in active)),
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
