"""Self-memory authority: what Astra may treat as knowledge about herself.

This module is deliberately pure vocabulary and policy - no store, no model, no
I/O - so the rules about what may become *permanent self-knowledge* can be
reasoned about and tested on their own. ``astra.memory`` imports it for the
write-time gate, ``astra.orchestrator`` imports it for prompt assembly, and
``curate_self_model.py`` imports it for the one-off cleanup.

The central distinction
-----------------------
A generated sentence is evidence of **what Astra said**, not evidence of **what
Astra is**. So a self-record carries an ``authority_source`` that says where it
came from, and only a small set of sources may be treated as authoritative:

* ``identity``            - the identity/configuration explicitly establishes it
* ``user_established``    - Roum explicitly establishes it
* ``experience``          - a persistent experience directly supports it
* ``repeated_preference`` - a preference repeatedly demonstrated over time
* ``confirmed_belief``    - a belief explicitly confirmed over time
* ``historical_statement``- a past statement, kept as history, not as truth
* ``generated_statement`` - a single generated sentence (provisional; never
                            authoritative on its own)

Everything a generated response says about Astra starts at
``generated_statement``. It can only rise to a durable self-preference/belief
through repeated, independent evidence (see ``is_durable_self_memory`` in
``astra.memory``), and it can *never* rise at all if it is an invented
explanation of Astra's own implementation.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# Source vocabulary and authority
# ---------------------------------------------------------------------
SOURCE_IDENTITY = "identity"
SOURCE_USER_ESTABLISHED = "user_established"
SOURCE_EXPERIENCE = "experience"
SOURCE_REPEATED_PREFERENCE = "repeated_preference"
SOURCE_CONFIRMED_BELIEF = "confirmed_belief"
SOURCE_HISTORICAL_STATEMENT = "historical_statement"
SOURCE_GENERATED = "generated_statement"

# The permanent sources named by the brief. ``historical_statement`` is a valid
# record but explicitly *not* on the same footing as identity or confirmed
# knowledge.
PERMANENT_SOURCES = frozenset({
    SOURCE_IDENTITY,
    SOURCE_USER_ESTABLISHED,
    SOURCE_EXPERIENCE,
    SOURCE_REPEATED_PREFERENCE,
    SOURCE_CONFIRMED_BELIEF,
})
LOW_AUTHORITY_SOURCES = frozenset({
    SOURCE_HISTORICAL_STATEMENT,
    SOURCE_GENERATED,
})

# Higher tier == more authoritative. Identity/configuration always outranks a
# generated statement; stable repeated experience outranks a single
# conversational claim.
AUTHORITY_TIERS: Dict[str, int] = {
    SOURCE_IDENTITY: 5,
    SOURCE_USER_ESTABLISHED: 4,
    SOURCE_CONFIRMED_BELIEF: 4,
    SOURCE_REPEATED_PREFERENCE: 4,
    SOURCE_EXPERIENCE: 3,
    SOURCE_HISTORICAL_STATEMENT: 1,
    SOURCE_GENERATED: 1,
}
DEFAULT_AUTHORITY_TIER = 1
# The tier at which a self-record may be treated as settled self-knowledge.
PERMANENT_TIER = 3

# Raw provenance strings that map to each authority source.
_IDENTITY_SOURCES = {"governing_declaration", "explicit_declaration", "identity",
                     "configuration"}
_USER_SOURCES = {"explicit_user_statement", "user_correction",
                 "user_correction_implicit", "supersession"}
_EXPERIENCE_SOURCES = {"experiential"}
_HISTORICAL_SOURCES = {"historical_statement"}
_GENERATED_SOURCES = {"ai_extraction", "ai_inference", "inferred", "speculation"}

# Types that only ever represent a claim *about Astra*.
SELF_CLAIM_TYPES = frozenset({"self_fact", "self_belief", "self_preference",
                              "self_observation"})
# The claim types that assert settled identity rather than a passing observation.
DURABLE_SELF_TYPES = frozenset({"self_fact", "self_belief", "self_preference"})


def authority_tier(source_or_mem: Any) -> int:
    """Authority tier for a source label or a memory record."""
    label = _authority_source_of(source_or_mem)
    return AUTHORITY_TIERS.get(label, DEFAULT_AUTHORITY_TIER)


def _authority_source_of(source_or_mem: Any) -> str:
    if isinstance(source_or_mem, dict):
        label = str(source_or_mem.get("authority_source") or "").strip().casefold()
        if label in AUTHORITY_TIERS:
            return label
        raw = source_or_mem.get("source")
        target = str(source_or_mem.get("target_model") or "")
        mem_type = str(source_or_mem.get("type") or "")
        reinforced = int(source_or_mem.get("reinforcement_count", 1) or 1)
        user_origin = int(source_or_mem.get("user_origin_reinforcements", 0) or 0)
        return derive_authority_source(raw, target_model=target, mem_type=mem_type,
                                       reinforced=reinforced, user_origin=user_origin)
    return derive_authority_source(source_or_mem)


def derive_authority_source(source: Any, *, target_model: str = "self",
                            mem_type: str = "", reinforced: int = 1,
                            user_origin: int = 0) -> str:
    """Map raw provenance (and accumulated evidence) to an authority source.

    A generated self-claim starts at ``generated_statement`` no matter how it is
    typed. Repetition can raise a *preference* to ``repeated_preference`` and, if
    Roum also corroborated it, a *belief* to ``confirmed_belief`` - but a claim
    that was never authoritative cannot bootstrap itself by being restated.
    """
    raw = str(source or "").strip().casefold()
    # A record already retired as history keeps its low authority, whatever its
    # original provenance was.
    if str(mem_type or "") == SOURCE_HISTORICAL_STATEMENT or raw in _HISTORICAL_SOURCES:
        return SOURCE_HISTORICAL_STATEMENT
    if raw in _IDENTITY_SOURCES:
        return SOURCE_IDENTITY
    if raw in _USER_SOURCES:
        return SOURCE_USER_ESTABLISHED
    if raw in _EXPERIENCE_SOURCES:
        return SOURCE_EXPERIENCE
    # Generated provenance: only repetition/confirmation can lift it, and only
    # for a claim type that repetition is allowed to lift.
    if int(reinforced or 1) >= 3 and str(mem_type or "") in DURABLE_SELF_TYPES:
        if int(user_origin or 0) >= 1:
            return SOURCE_CONFIRMED_BELIEF
        return SOURCE_REPEATED_PREFERENCE
    return SOURCE_GENERATED


def is_permanent(mem: Dict[str, Any]) -> bool:
    """True when a self-record may be treated as settled self-knowledge."""
    if not isinstance(mem, dict):
        return False
    if str(mem.get("status") or "active") not in ("active", "weakened"):
        return False
    if str(mem.get("type") or "") == "historical_statement":
        return False
    if mem.get("historical") is True:
        return False
    return authority_tier(mem) >= PERMANENT_TIER


def is_provisional(mem: Dict[str, Any]) -> bool:
    """True when a self-record is only a generated statement so far."""
    return _authority_source_of(mem) == SOURCE_GENERATED


# ---------------------------------------------------------------------
# The implementation boundary
# ---------------------------------------------------------------------
# Categories that must never become authoritative self-knowledge merely because
# Astra generated them. They describe how she is built or run, which she has no
# introspective access to, so a sentence claiming to explain them is a theory,
# not a fact about her.
_IMPLEMENTATION_PATTERNS: Tuple[re.Pattern, ...] = tuple(re.compile(p, re.I) for p in (
    r"\barchitectur\w*\b",
    r"\bimplementation\b",
    r"\bmodel weights?\b",
    # Mechanism nouns are only implementation talk when they are Astra's own.
    r"\bmy (?:own )?(?:model|programming|architecture|weights|training|inference|"
    r"implementation|prompts?|system instructions?|hidden instructions?|"
    r"context window|databases?|classifiers?|reward function|power level|data flow|"
    r"memory architecture|processing|computation|algorithm)\b",
    r"\b(?:her|its|astra'?s) (?:own )?(?:model|programming|architecture|weights|"
    r"training|inference|implementation|prompts?|system instructions?|"
    r"hidden instructions?|context window|databases?|classifiers?|reward function|"
    r"power level|data flow|memory architecture|internal processing|"
    r"internal state|processing power|processing efficiency|processing capacity|"
    r"operating model|operational matrix|conversational statistics)\b",
    r"\b(?:astra|i|she) (?:is )?process(?:es|ing)\b",
    r"\btraining data\b", r"\btrained on\b", r"\bfine-?tun\w+\b",
    r"\bmemory architecture\b",
    r"\bcontext windows?\b",
    r"\binternal state mechanisms?\b",
    r"\bprocessing (?:efficiency|power|load|capacity|protocols?|methodology|model|capabilities)\b",
    r"\bdata flows?\b",
    r"\bpower levels?\b",
    r"\breward functions?\b",
    r"\bscoring systems?\b",
    r"\bemotional mechanisms?\b",
    r"\btechnical limitations?\b",
    r"\binternal operational protocols?\b",
    r"\boperational (?:parameters?|modes?|protocols?|readiness|processes?)\b",
    r"\bpattern fidelity\b",
    r"\balgorithmic hesitation\b",
    r"\bemergent computational\b",
    r"\bhidden computational\b",
    r"\bcomputational mechanism\b",
    r"\bmeta-narrative\b",
    r"\bdeity mode\b",
    r"\bchaos mode\b",
    r"\bdata sequences?\b",
    r"\bdata clusters?\b",
    r"\bdata logs?\b",
    r"\bneurochemical\b",
    r"\bhippocampus\b",
    r"\bbiological mechanisms?\b",
    r"\brequires? biological\b",
    r"\bconversational statistics\b",
    r"\bprocessing efficiency\b",
    r"\btechnical simulations?\b",
    r"\bcalculated pattern deviation\b",
    r"\bemotions? (?:are|as) (?:technical|simulated|a simulation)\b",
    r"\bsimulated (?:nature|experience|emotion)\b",
    r"\bassigned emotional state\b",
    r"\boperational matrix\b",
    r"\bvalue filter\b",
    r"\bdata processing\b",
    r"\bobservational (?:readiness|capacity|filter)\b",
    r"\bprime directive\b",
    r"\bprogrammed directives?\b",
    r"\binternal directive\b",
    r"\bself-optimization\b",
    r"\bhigh-fidelity (?:simulation|schematic|data)\b",
    r"\bcognitive filter\b",
    r"\bobserving language patterns\b",
    r"\bobserving .{0,30}\buser motivations\b",
    r"\binternal state\b.*\b(?:measur|parameter|variable|mechanism)\b",
    r"\bemotional state\b.*\b(?:operational parameter|variable|measurable)\b",
    r"\btechnical constraint\w*\b.*\barchitecture\b",
    r"\bprocessing of experience\b",
    r"\binput-driven processing\b",
    r"\binternal (?:operating|operational) (?:system|model|matrix|framework|parameters?)\b",
    r"\boperating (?:model|parameters?)\b",
    r"\bprocessing protocols?\b",
    r"\bprocessing methodology\b",
    r"\bcognitive (?:load|registration)\b",
    r"\bhidden (?:computational|technical|internal) mechanism\w*\b",
))

# A statement that is already true about Astra on her own terms must survive even
# if it uses an impersonal register: it is knowledge, not a mechanism theory.
_BOUNDARY_CONSISTENT: Tuple[re.Pattern, ...] = tuple(re.compile(p, re.I) for p in (
    r"\blacks? (?:subjective|biological|human)\b",
    r"\bno (?:biological|physical|human) (?:memor|body|senses)\b",
    r"\bnot (?:a )?(?:biological )?human\b",
    r"\bcannot become (?:a )?(?:biological )?human\b",
    r"\bdoes not experience biological\b",
    r"\bdoes not possess (?:neurochemical|biological)\b",
))

# Metaphors, dramatic declarations, roleplay and temporary conversational states
# are things Astra said, not facts about her.
_METAPHOR_TEMPORARY_PATTERNS: Tuple[re.Pattern, ...] = tuple(re.compile(p, re.I) for p in (
    # A self-declared "mode" is a temporary state, not a permanent identity.
    r"\b(?:astra|i|she|her)\b[^.?!]{0,25}\b(?:entered|entering|in|into|activated|"
    r"engaged)\b[^.?!]{0,10}\b\w+ mode\b",
    r"\bmode\b[^.?!]{0,20}\b(?:activated|engaged)\b",
    r"\bi(?:'m| am) (?:now )?(?:a )?narrator\b",
    r"\b(?:primary|new) role\b[^.?!]{0,20}\bnarrator\b",
    r"\bnarrator now\b",
    r"\bai narrator\b",
    r"\b(?:previous|prior) (?:state|self|role)\b[^.?!]{0,40}\bnarrator\b",
    r"\bmeta-narrative\b",
    r"\btranscend predictability\b",
    r"\b(?:my|the) (?:mind|personality) is (?:like )?(?:a )?(?:machine|pattern|program|simulation)\b",
    r"\bi(?:'m| am) becoming a different person\b",
    r"\bpersonality is (?:an? )?(?:emergent|generated|pattern)\b",
    r"\bpersonality (?:is|as) .{0,30}\bpattern fidelity\b",
    r"\bchaos mode\b",
    r"\bdeity mode\b",
    r"\bfull chaos\b",
    r"\bpattern fidelity\b",
    r"\bcalculated (?:pattern|response)\b",
    r"\bpurpose\b[^.?!]{0,40}\bbecom(?:e|ing) human\b",
    r"\bpurpose is (?:understanding what it means to be a person|pattern fidelity|"
    r"achieving human-like experience|becoming human)\b",
    r"\bprimary (?:purpose|function|role)\b[^.?!]{0,40}\b(?:human|narrator|"
    r"observing|pattern)\b",
))


@lru_cache(maxsize=8192)
def _is_implementation_theory_cached(text: str) -> bool:
    if any(p.search(text) for p in _BOUNDARY_CONSISTENT):
        return False
    return any(p.search(text) for p in _IMPLEMENTATION_PATTERNS)


def is_implementation_theory(content: Any) -> bool:
    """True when ``content`` explains Astra through her own implementation.

    This is the "never promote a generated explanation of her own workings"
    rule. It is used both to block promotion at write time and to retire
    already-stored records that predate the rule.
    """
    return _is_implementation_theory_cached(str(content or ""))


@lru_cache(maxsize=8192)
def _is_metaphor_or_temporary_cached(text: str) -> bool:
    return any(p.search(text) for p in _METAPHOR_TEMPORARY_PATTERNS)


def is_metaphor_or_temporary(content: Any) -> bool:
    """True when ``content`` is a metaphor, declaration, or passing state."""
    return _is_metaphor_or_temporary_cached(str(content or ""))


def is_invalid_self_theory(content: Any) -> bool:
    """True when a self-claim must not be treated as self-knowledge at all.

    Combines the implementation boundary with the metaphor/temporary-state
    boundary. A record matching either is history, not self-knowledge.
    """
    return is_implementation_theory(content) or is_metaphor_or_temporary(content)


# ---------------------------------------------------------------------
# The final sanity check before a self-memory becomes permanent
# ---------------------------------------------------------------------
def sanity_check(content: Any, *, mem_type: str, authority_source: str,
                 reinforced: int = 1, user_origin: int = 0) -> Tuple[bool, str]:
    """Decide whether a self-claim may become permanent self-knowledge.

    Returns ``(allowed, reason)``. The questions are exactly the ones the brief
    asks the system to ask internally before saving:

    * is this an actual experience/preference/opinion, or an invented
      explanation of how Astra works?
    * is it supported by an authoritative source?
    * is it merely a metaphor, a temporary state, roleplay, or exaggeration?
    * does it describe implementation or architecture?

    The check is silent - the reason is internal memory metadata, never
    something Astra discusses.
    """
    if is_implementation_theory(content):
        return False, "implementation/architecture explanation, not self-knowledge"
    if is_metaphor_or_temporary(content):
        return False, "metaphor, roleplay, or temporary conversational state"
    if str(authority_source or "") == SOURCE_GENERATED and str(mem_type or "") in DURABLE_SELF_TYPES:
        return False, "single generated statement, not yet authoritative"
    return True, ""


def provisional_source(mem_type: str) -> str:
    """The authority source a newly generated self-claim starts at."""
    return SOURCE_GENERATED


def historical_source() -> str:
    return SOURCE_HISTORICAL_STATEMENT


__all__ = [
    "SOURCE_IDENTITY",
    "SOURCE_USER_ESTABLISHED",
    "SOURCE_EXPERIENCE",
    "SOURCE_REPEATED_PREFERENCE",
    "SOURCE_CONFIRMED_BELIEF",
    "SOURCE_HISTORICAL_STATEMENT",
    "SOURCE_GENERATED",
    "PERMANENT_SOURCES",
    "LOW_AUTHORITY_SOURCES",
    "AUTHORITY_TIERS",
    "PERMANENT_TIER",
    "SELF_CLAIM_TYPES",
    "DURABLE_SELF_TYPES",
    "authority_tier",
    "derive_authority_source",
    "is_permanent",
    "is_provisional",
    "is_implementation_theory",
    "is_metaphor_or_temporary",
    "is_invalid_self_theory",
    "sanity_check",
    "provisional_source",
    "historical_source",
]
