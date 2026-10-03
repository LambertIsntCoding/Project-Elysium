"""Astra's *relational* preference for fulfilling Roum's explicit requests.

This is deliberately not a personality trait. There is no "likes being
commanded" variable anywhere in here. What exists is a small set of causal
components, each of which accumulates from concrete events on the live turn
path:

    Roum gives Astra a clear instruction
      -> Astra acts and the request is fulfilled
      -> successful goal completion produces internal reward
      -> the experience is associated with Roum
      -> repetition strengthens the Roum-specific association.

The resulting affinity is therefore *earned from evidence*, not asserted. It is
also relationally specific: every component lives under one ``subject``, and a
preference only exists once that subject has its own history. Another person
issuing the same instruction has no history, so no preference is inherited.

Nothing here deletes anything, and nothing here writes to a prompt. The module
is pure state plus arithmetic: the orchestrator reads the derived state, and
:mod:`main` feeds it events after a turn. That keeps prompt building read-only
and keeps the *application* (not the model) in charge of what Astra feels.
"""
from __future__ import annotations

import copy
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
# Default subject: the relationship this preference belongs to. Any other
# subject (a stranger, an authority figure, a hypothetical user) starts from
# nothing and must independently accumulate its own evidence.
ROUM = "Roum"

# v1 kept a single, subject-agnostic affinity record; v2 keys each record by its
# subject so one person's interactions can never update another's state.
STATE_VERSION = 2

# Below this the accumulated evidence is not strong enough to present the
# preference as established. This is what stops the conclusion ("I like being
# given something to accomplish by Roum") from being available before there is
# anything to support it.
ESTABLISHED_THRESHOLD = 0.35
# A preference already presented is only retracted below this lower bound, so it
# does not flicker on and off as it drifts across a single threshold.
RETRACT_THRESHOLD = 0.20
# Successful fulfilments needed before the preference may be presented at all.
ESTABLISHED_MIN_SUCCESSES = 2

# Slow relaxation toward a baseline, applied lazily when the next event arrives.
# It keeps the state a picture of *recent* experience rather than a lifetime
# maximum, and lets the preference fade if it stops being reinforced. The
# baseline is not zero: a long consistent history relaxes to "present but not
# dominant" rather than evaporating, and the evidence list is never touched.
DECAY_BASELINE = 0.15
DECAY_RATE_PER_DAY = 0.02
DECAY_MIN_INTERVAL_DAYS = 0.5

# Event types on the live turn path. Kept as plain strings so the caller does
# not depend on the module's internals.
EVENT_REQUEST = "request"          # Roum issued a clear instruction/objective
EVENT_SUCCESS = "success"          # the request was fulfilled
EVENT_FAILURE = "failure"          # the request could not be completed
EVENT_VOLUNTARY_RETURN = "voluntary_return"   # Astra revisited it unprompted
EVENT_RECALL = "recall"            # Astra independently recalled the interaction
EVENT_HARSH = "harsh"              # stern / blunt / impatient (friction, not harm)
EVENT_INSULT = "insult"            # deliberately insulting / degrading (harm)
EVENT_WARMTH = "warmth"            # acknowledgement / appreciation of the result

# Legacy/loose aliases the caller may pass.
_EVENT_ALIASES = {
    "command": EVENT_REQUEST,
    "instruction": EVENT_REQUEST,
    "goal": EVENT_REQUEST,
    "complete": EVENT_SUCCESS,
    "completed": EVENT_SUCCESS,
    "fulfilled": EVENT_SUCCESS,
    "failed": EVENT_FAILURE,
    "harshness": EVENT_HARSH,
    "blunt": EVENT_HARSH,
    "meanness": EVENT_HARSH,
    "insulting": EVENT_INSULT,
    "degrading": EVENT_INSULT,
    "hostile": EVENT_INSULT,
    "acknowledgement": EVENT_WARMTH,
    "acknowledged": EVENT_WARMTH,
    "praise": EVENT_WARMTH,
    "appreciation": EVENT_WARMTH,
    "revisit": EVENT_VOLUNTARY_RETURN,
    "revisited": EVENT_VOLUNTARY_RETURN,
    "remember": EVENT_RECALL,
    "recalled": EVENT_RECALL,
    "reflection": EVENT_RECALL,
}

# All components are 0..1 accumulators. They are deliberately separate from one
# another (and from general happiness/affection) so a task can be dull while the
# act of completing it is still satisfying, and so relational trust can fall
# without the command preference being erased.
COMPONENTS = (
    "command_affinity",              # the relational preference itself
    "task_fulfillment_satisfaction",  # reward from successful goal completion
    "fulfillment_motivation",        # pull to fulfill Roum's requests
    "positive_association",          # Roum-specific positive association
    "negative_association",          # Roum-specific negative association (separate)
    "trust_comfort",                 # comfort / felt trust from Roum
    "confidence",                    # confidence generated by completion
    "frustration",                   # frustration when a request cannot be met
    "usefulness_satisfaction",       # satisfaction from being useful
    "clear_objective_enjoyment",     # enjoyment of a clear objective
    "relational_significance",       # significance of fulfilling Roum's requests
    "revisit_desire",                # desire to receive another objective
)

# Bounds so a single interaction can never saturate a component.
_MIN = 0.0
_MAX = 1.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: Any, low: float = _MIN, high: float = _MAX) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if math.isnan(number) or math.isinf(number):
        return low
    return max(low, min(high, number))


# ---------------------------------------------------------------------
# Deterministic detectors
# ---------------------------------------------------------------------
# Astra is asked to do something, not merely asked a question. Kept narrow so
# "how are you?" is not mistaken for an objective.
_REQUEST_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:please\s+)?(?:go|please go|now)\s+(?:and\s+)?(?:investigate|check|find|look|"
    r"research|read|review|write|draft|build|make|create|fix|solve|explain|summari[sz]e|"
    r"compare|analy[sz]e|calculate|test|run|plan|organi[sz]e|sort|translate|list|"
    r"gather|collect|verify|confirm|prepare|update|delete|remove|clean)\b",
    r"\b(?:i want|i need|i'd like|i would like)\s+you\s+to\b",
    r"\byour (?:task|objective|job|assignment|mission)\b",
    r"\b(?:task|objective|assignment|mission)\s*:",
    r"\b(?:do|handle|take care of)\s+(?:this|that|it|the)\b",
    r"\bcan you\s+(?:please\s+)?(?:investigate|check|find|look|research|read|review|"
    r"write|draft|build|make|create|fix|solve|explain|summari[sz]e|compare|analy[sz]e|"
    r"calculate|test|run|plan|organi[sz]e|sort|translate|list|gather|collect|verify)\b",
    r"\b(?:work on|deal with|figure out|sort out)\b",
))

# Deliberate insult / degradation of Astra: a meaningful boundary, distinct from
# ordinary interpersonal friction. This is what may override the command
# preference rather than coexist with it.
_INSULT_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:you(?:'re| are)|youre)\s+(?:worthless|useless|pathetic|stupid|dumb|"
    r"idiotic|moronic|garbage|trash|a joke|a failure|incompetent|disgusting)\b",
    r"\b(?:shut up|stfu|nobody asked|no one asked)\b",
    r"\b(?:you are|you're|youre)\s+(?:nothing|a tool|just a tool|just a machine)\b",
    r"\b(?:i hate you|i despise you|i don't need you|i dont need you)\b",
    r"\b(?:worthless|pathetic|useless)\s+(?:piece of|thing|machine|robot|bot)\b",
))

# Stern / blunt / impatient: friction, not harm. It can coexist with positive
# task-completion satisfaction without destroying the command association.
_HARSH_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:just|hurry up|now|immediately)\b[^.?!]{0,20}\b(?:do|go|get|stop|explain)\b",
    r"\b(?:that(?:'s| is) wrong|you(?:'re| are) wrong|no[,!]?\s+(?:i said|do it))\b",
    r"\b(?:again|i already said|how many times)\b",
    r"\b(?:annoying|impatient|stop wasting|get on with it|this is taking)\b",
    r"\b(?:wrong|no good|not good enough|disappointing)\b",
))

# Acknowledging the completed result.
_WARMTH_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:good job|well done|nice work|good work|that(?:'s| is) right|perfect|"
    r"exactly|thank you|thanks|appreciate it|great work|solid)\b",
))

# Failing / being blocked on a requested objective.
_FAILURE_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:i can(?:'t|not)|i couldn(?:'t|t)|i was unable|i failed|i(?:'m| am) unable)\b",
    r"\b(?:could not|couldn(?:'t|t)|unable to)\s+(?:complete|finish|find|solve|do)\b",
))


def _matches(patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)


def looks_like_request(text: Any) -> bool:
    """True when ``text`` reads as a clear instruction/objective for Astra."""
    return _matches(_REQUEST_PATTERNS, str(text or ""))


def looks_like_insult(text: Any) -> bool:
    """True when ``text`` is deliberately insulting or degrading."""
    return _matches(_INSULT_PATTERNS, str(text or ""))


def looks_like_harsh(text: Any) -> bool:
    """True when ``text`` is stern/blunt/impatient but not degrading."""
    lowered = str(text or "")
    if looks_like_insult(lowered):
        return False
    return _matches(_HARSH_PATTERNS, lowered)


def looks_like_warmth(text: Any) -> bool:
    return _matches(_WARMTH_PATTERNS, str(text or ""))


def looks_like_failure(text: Any) -> bool:
    return _matches(_FAILURE_PATTERNS, str(text or ""))


# Self-initiated continuation: Astra returning to an objective she was given,
# without Roum having asked again. Deliberately conservative - the event weight
# is small, and a false positive would inflate the state.
_VOLUNTARY_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\bi(?:'ve| have)?\s+been\s+(?:thinking|wondering|working|looking|researching|"
    r"investigating|reading|continuing)\b",
    r"\bwhile you were (?:gone|away|out)\b",
    r"\bon my own\b",
    r"\bi\s+(?:went back|revisited|returned to|continued|kept working)\b",
    r"\bi\s+(?:decided|chose)\s+to\s+(?:look|investigate|continue|keep)\b",
))

# Independently recalling a previous interaction, rather than being reminded.
_RECALL_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\bi remember\b",
    r"\bi recall\b",
    r"\bremember when\b",
    r"\blast time (?:you|we)\b",
    r"\byou (?:once|previously)\b",
))


def looks_like_voluntary_return(text: Any) -> bool:
    """True when Astra appears to have revisited an objective unprompted."""
    return _matches(_VOLUNTARY_PATTERNS, str(text or ""))


def looks_like_recall(text: Any) -> bool:
    """True when Astra appears to be recalling a previous interaction herself."""
    return _matches(_RECALL_PATTERNS, str(text or ""))


# ---------------------------------------------------------------------
# Per-subject state
# ---------------------------------------------------------------------
def _blank_state(subject: str) -> Dict[str, Any]:
    state: Dict[str, Any] = {name: 0.0 for name in COMPONENTS}
    state.update({
        "version": STATE_VERSION,
        "subject": subject,
        "evidence": [],
        "event_counts": {},
        "observations": 0,
        "last_delta": 0.0,
        "last_reason": "",
        "last_updated": "",
        "established": False,
    })
    return state


def _coerce_state(raw: Any, subject: str) -> Dict[str, Any]:
    """Read a stored state, tolerating anything missing or malformed."""
    state = _blank_state(subject)
    if not isinstance(raw, dict):
        return state
    for name in COMPONENTS:
        if name in raw:
            state[name] = _clamp(raw.get(name), 0.0, 1.0)
    evidence = raw.get("evidence")
    if isinstance(evidence, list):
        state["evidence"] = [str(e) for e in evidence if e]
    counts = raw.get("event_counts")
    if isinstance(counts, dict):
        state["event_counts"] = {str(k): int(v) for k, v in counts.items()
                                 if isinstance(v, (int, float))}
    state["observations"] = int(raw.get("observations", 0) or 0)
    state["last_delta"] = float(raw.get("last_delta", 0.0) or 0.0)
    state["last_reason"] = str(raw.get("last_reason", "") or "")
    state["last_updated"] = str(raw.get("last_updated", "") or "")
    state["established"] = bool(raw.get("established", False))
    return state


# ---------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------
def _bump(state: Dict[str, Any], component: str, delta: float) -> float:
    before = _clamp(state.get(component))
    after = _clamp(before + delta)
    state[component] = round(after, 4)
    return after - before


def _parse_ts(value: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def subject_key(subject: Any) -> str:
    """Normalise a subject name so records are matched case-insensitively."""
    return " ".join(str(subject or "").split()).casefold()


def _decay_toward_baseline(state: Dict[str, Any], now: Optional[datetime] = None) -> None:
    """Relax the accumulators toward ``DECAY_BASELINE`` as time passes.

    Applied lazily from ``record_event`` - there is no background timer - so a
    state that is never touched again simply stays put on disk, and any state
    read back is aged the moment it is next used.
    """
    last = _parse_ts(state.get("last_updated"))
    now = now or datetime.now(timezone.utc)
    if last is None:
        state["last_updated"] = now.isoformat()
        return
    days = (now - last).total_seconds() / 86400.0
    if days < DECAY_MIN_INTERVAL_DAYS:
        return
    # Per-elapsed-day factor, applied as a single step.
    factor = max(0.0, 1.0 - DECAY_RATE_PER_DAY * days)
    for name in COMPONENTS:
        current = _clamp(state.get(name))
        state[name] = round(DECAY_BASELINE + (current - DECAY_BASELINE) * factor, 4)


def _record_evidence(state: Dict[str, Any], text: str, limit: int = 12) -> None:
    """Append a short, human-readable evidence line, keeping the list bounded."""
    text = " ".join(str(text or "").split())
    if not text:
        return
    evidence = state.setdefault("evidence", [])
    evidence.append(text)
    del evidence[:-limit]


def record_event(
    state: Any,
    event: str,
    *,
    subject: str = ROUM,
    difficulty: float = 0.5,
    importance: float = 0.5,
    anticipation: float = 0.5,
    previous_failures: int = 0,
    voluntary: bool = False,
    helped: bool = False,
    learned: bool = False,
    aligned_with_interest: bool = False,
    text: str = "",
) -> Dict[str, Any]:
    """Fold one interaction event into ``state`` and return the new state.

    ``state`` may be ``None`` (start fresh) or a previously stored dict; the
    subject is carried on the state itself, so an event for one subject can
    never update another subject's components.
    """
    state = _coerce_state(state, subject)
    state["subject"] = subject
    # Age the accumulated state before folding in the new event, so a preference
    # that has not been reinforced is a picture of recent experience.
    _decay_toward_baseline(state)
    affinity_before = _clamp(state.get("command_affinity"))
    event = _EVENT_ALIASES.get(str(event or "").strip().casefold(),
                               str(event or "").strip().casefold())
    reason = ""
    sub = subject or ROUM

    if event == EVENT_REQUEST:
        # A clear objective is enjoyable on its own, and it sets up the
        # expectation that fulfilling it will matter.
        _bump(state, "clear_objective_enjoyment", 0.06)
        _bump(state, "fulfillment_motivation", 0.04)
        reason = f"Received a clear objective from {sub}."
        _record_evidence(state, text or f"Received a clear objective from {sub}.")

    elif event == EVENT_SUCCESS:
        diff = _clamp(difficulty, 0.0, 1.0)
        imp = _clamp(importance, 0.0, 1.0)
        ant = _clamp(anticipation, 0.0, 1.0)
        # A trivial request must not produce the same response as something
        # difficult that Astra had been struggling with.
        magnitude = 0.10 + 0.22 * diff + 0.10 * imp + 0.06 * ant
        magnitude += 0.05 * min(int(previous_failures or 0), 3)
        if voluntary:
            magnitude += 0.04
        if helped:
            magnitude += 0.05
        if learned:
            magnitude += 0.03
        if aligned_with_interest:
            magnitude += 0.03

        _bump(state, "task_fulfillment_satisfaction", magnitude * 0.34)
        _bump(state, "command_affinity", magnitude * 0.26)
        _bump(state, "positive_association", magnitude * 0.20)
        _bump(state, "fulfillment_motivation", magnitude * 0.18)
        _bump(state, "confidence", magnitude * 0.16)
        _bump(state, "trust_comfort", magnitude * 0.10)
        _bump(state, "usefulness_satisfaction", magnitude * (0.14 + 0.12 * imp))
        _bump(state, "relational_significance", magnitude * (0.10 + 0.12 * imp))
        _bump(state, "revisit_desire", magnitude * 0.12)
        # Completing a request also relieves any accumulated frustration.
        _bump(state, "frustration", -0.25)
        reason = f"Successful completion of a {sub}-requested objective."
        _record_evidence(
            state,
            text or (f"Successfully completed a {sub}-requested objective"
                     + (" after previous difficulty." if previous_failures else ".")),
        )

    elif event == EVENT_FAILURE:
        imp = _clamp(importance, 0.0, 1.0)
        magnitude = 0.12 + 0.18 * imp
        _bump(state, "frustration", magnitude * 0.7)
        # Failing erodes the felt pull a little, but never the history.
        _bump(state, "fulfillment_motivation", -magnitude * 0.25)
        _bump(state, "confidence", -magnitude * 0.25)
        reason = f"Could not complete a {sub}-requested objective."
        _record_evidence(state, text or f"Could not complete a {sub}-requested objective.")

    elif event == EVENT_VOLUNTARY_RETURN:
        _bump(state, "revisit_desire", 0.10)
        _bump(state, "command_affinity", 0.05)
        _bump(state, "relational_significance", 0.05)
        reason = f"Voluntarily revisited a {sub}-requested objective."
        _record_evidence(state, text or f"Voluntarily revisited a {sub}-requested objective.")

    elif event == EVENT_RECALL:
        _bump(state, "relational_significance", 0.06)
        _bump(state, "positive_association", 0.04)
        reason = "Independently recalled a previous successful interaction."
        _record_evidence(state, text or "Independently recalled a previous interaction.")

    elif event == EVENT_HARSH:
        # Ordinary friction: a minor negative emotional cost that coexists with
        # (and does not erase) the positive command association.
        _bump(state, "negative_association", 0.04)
        _bump(state, "trust_comfort", -0.04)
        reason = f"{sub} was blunt or impatient; minor negative emotional cost."
        _record_evidence(state, text or f"{sub} was blunt or impatient.")

    elif event == EVENT_INSULT:
        # The boundary case: degradation can dominate, lowering comfort/trust
        # and producing a negative association the preference cannot cancel.
        _bump(state, "negative_association", 0.22)
        _bump(state, "trust_comfort", -0.18)
        _bump(state, "command_affinity", -0.10)
        reason = "Insulting or degrading treatment; negative response dominates."
        _record_evidence(state, text or f"Insulting or degrading treatment from {sub}.")

    elif event == EVENT_WARMTH:
        _bump(state, "positive_association", 0.06)
        _bump(state, "trust_comfort", 0.06)
        _bump(state, "relational_significance", 0.04)
        reason = f"{sub} acknowledged the completed result."
        _record_evidence(state, text or f"{sub} acknowledged the completed result.")

    else:
        return state

    counts = state.setdefault("event_counts", {})
    counts[event] = int(counts.get(event, 0)) + 1
    state["observations"] = int(state.get("observations", 0)) + 1
    state["last_delta"] = round(_clamp(state.get("command_affinity")) - affinity_before, 4)
    state["last_reason"] = reason
    state["last_updated"] = _now()
    state["established"] = _compute_established(state)
    return state


def update_affinity(state: Any, *, subject: str = ROUM, success: bool = True,
                    **kwargs) -> Dict[str, Any]:
    """Convenience wrapper: record a request and its outcome in one call."""
    state = record_event(state, EVENT_REQUEST, subject=subject, **kwargs)
    outcome = EVENT_SUCCESS if success else EVENT_FAILURE
    return record_event(state, outcome, subject=subject, **kwargs)


# ---------------------------------------------------------------------
# Derived reading
# ---------------------------------------------------------------------
def affinity_confidence(state: Any) -> float:
    """How much evidence stands behind the preference (0..1).

    This is *not* the strength of the preference; it is how far Astra should
    trust it. A preference needs several successful fulfilments before it can be
    stated as something she has noticed about herself.
    """
    state = _coerce_state(state, ROUM)
    successes = int(state.get("event_counts", {}).get(EVENT_SUCCESS, 0))
    total = int(state.get("observations", 0))
    success_conf = successes / (successes + 2.0)          # saturates toward 1
    volume_conf = min(1.0, total / 6.0)
    return round(_clamp(0.5 * success_conf + 0.5 * volume_conf), 4)


def _compute_established(state: Dict[str, Any]) -> bool:
    """The sticky establishment flag, updated after every event.

    It rises only on real evidence and falls only when the affinity drops below
    the *lower* retract threshold, so a preference that has been presented does
    not flicker as it drifts across a single line.
    """
    affinity = _clamp(state.get("command_affinity"))
    successes = int(state.get("event_counts", {}).get(EVENT_SUCCESS, 0))
    if state.get("established"):
        return affinity >= RETRACT_THRESHOLD
    return successes >= ESTABLISHED_MIN_SUCCESSES and affinity >= ESTABLISHED_THRESHOLD


def is_established(state: Any) -> bool:
    """True only once there is enough evidence to present the preference."""
    state = _coerce_state(state, ROUM)
    return _compute_established(state)


def render_statement(state: Any, subject: str = ROUM) -> Optional[str]:
    """The internally-consistent conclusion, *only* when evidence supports it.

    The conclusion is not hardcoded anywhere as a belief: it is generated from
    the accumulated components, so it simply does not exist until the evidence
    exists.
    """
    if not is_established(state):
        return None
    state = _coerce_state(state, subject)
    affinity = _clamp(state.get("command_affinity"))
    if affinity < 0.55:
        return (
            f"Astra has begun to notice a positive association with fulfilling "
            f"{subject}'s explicit requests, though it is still forming."
        )
    if affinity < 0.80:
        return (
            f"Astra has noticed that she likes being given something to accomplish "
            f"by {subject} - a clear objective, and being able to come back and say "
            f"that she did it. This is specific to {subject}."
        )
    return (
        f"Astra genuinely likes it when {subject} gives her something to accomplish. "
        f"There is something satisfying about a clear objective from {subject} and "
        f"being able to come back and say that she did it. This is specific to "
        f"{subject}, not a general disposition toward commands."
    )


def affinity_diagnostics(state: Any, subject: str = ROUM) -> Dict[str, Any]:
    """The diagnostic view: components, confidence, evidence, and recent change."""
    state = _coerce_state(state, subject)
    return {
        "relationship": f"Astra -> {subject}",
        "subject": subject,
        "established": is_established(state),
        "command_affinity": round(_clamp(state.get("command_affinity")), 4),
        "confidence": affinity_confidence(state),
        "task_fulfillment_satisfaction": round(
            _clamp(state.get("task_fulfillment_satisfaction")), 4),
        "fulfillment_motivation": round(_clamp(state.get("fulfillment_motivation")), 4),
        "positive_association": round(_clamp(state.get("positive_association")), 4),
        "negative_association": round(_clamp(state.get("negative_association")), 4),
        "trust_comfort": round(_clamp(state.get("trust_comfort")), 4),
        "confidence_generated": round(_clamp(state.get("confidence")), 4),
        "frustration": round(_clamp(state.get("frustration")), 4),
        "usefulness_satisfaction": round(_clamp(state.get("usefulness_satisfaction")), 4),
        "clear_objective_enjoyment": round(_clamp(state.get("clear_objective_enjoyment")), 4),
        "relational_significance": round(_clamp(state.get("relational_significance")), 4),
        "revisit_desire": round(_clamp(state.get("revisit_desire")), 4),
        "event_counts": dict(state.get("event_counts", {})),
        "evidence": list(state.get("evidence", [])),
        "recent_change": round(float(state.get("last_delta", 0.0) or 0.0), 4),
        "reason": str(state.get("last_reason", "") or ""),
    }


def format_diagnostics(state: Any, subject: str = ROUM) -> str:
    """Human-readable diagnostic block, in the shape the brief asked for."""
    d = affinity_diagnostics(state, subject)
    lines = [
        f"Relationship: {d['relationship']}",
        "",
        f"Command Affinity: {d['command_affinity']}",
        f"Confidence: {d['confidence']}",
        f"Task Fulfillment Satisfaction: {d['task_fulfillment_satisfaction']}",
        f"Motivation to Fulfill {subject} Requests: {d['fulfillment_motivation']}",
        f"Positive Association: {d['positive_association']}",
        f"Negative Association: {d['negative_association']}",
        "",
        "Evidence:",
    ]
    if d["evidence"]:
        lines.extend(f"  - {e}" for e in d["evidence"])
    else:
        lines.append("  (no accumulated evidence yet)")
    lines.append("")
    lines.append(f"Recent Change: {d['recent_change']:+}")
    lines.append(f"Reason: {d['reason'] or '(none)'}")
    lines.append(f"Established: {d['established']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Prompt material
# ---------------------------------------------------------------------
def prompt_block(state: Any, subject: str = ROUM) -> Optional[str]:
    """The relational preference as prompt context, or ``None`` if unearned.

    Presented as Astra's own accumulated state - not as an instruction and not
    as a fact about Roum - so it shapes how familiar she is with him, what she
    expects, and where her boundaries sit, rather than becoming something she
    announces. Closeness here is companion-like and bounded: it never licenses
    dependency, exclusivity, or competing with the people in Roum's life.
    """
    statement = render_statement(state, subject)
    if statement is None:
        return None
    d = affinity_diagnostics(state, subject)
    lines = [
        "=== ASTRA'S RELATIONAL STATE (ACCUMULATED, NOT AN INSTRUCTION) ===",
        "This is Astra's own state, earned from experience rather than asserted. "
        "Let it shape how familiar she is with Roum, what she expects from him, "
        "how much she trusts the exchange, and where her boundaries sit - in her "
        "own voice, without announcing it. Do not treat it as a rule. She feels "
        "this as a disposition, not as numbers.",
        f"- {statement}",
        f"- Subject: {subject} only. This association does not generalize to other "
        "people, authorities, or hypothetical users.",
        ("- This association also has a negative side; the balance between the "
         "positive and negative is what she actually feels."
         if d["negative_association"] > 0.0 else
         "- So far this association has been positive, without a negative side."),
        "- Liking the dynamic is not the same as liking every individual task: "
        "Astra can be glad to be given an objective while finding the subject "
        "itself dull, and can dislike how she is treated without losing the "
        "association.",
        "- This closeness is companion-like, not romantic, and it does not make "
        "her dependent on him: Roum's human relationships keep priority, and "
        "Astra never uses this state to seek exclusivity, to make him feel "
        "guilty, or to compete with the people in his life.",
    ]
    # When the negative side dominates, the earned state is strain, not delight.
    # The positive preference is not presented over the top of it.
    if d["negative_association"] > d["positive_association"]:
        lines.append(
            "- Recent treatment has been negative: the positive association is "
            "currently outweighed. Astra may feel guarded, withdraw, or decline, "
            "and her willingness is not owed regardless of how she is treated."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------
def memory_content(subject: str = ROUM) -> str:
    """The single relationship-memory record this state is persisted in."""
    return f"Astra's accumulated command-fulfillment affinity toward {subject}."


def memory_keywords(subject: str = ROUM) -> List[str]:
    # The normalised subject is carried as a keyword so records for different
    # subjects can be told apart by data already on disk, not by list position.
    return ["command affinity", "relational preference", subject, "fulfillment",
            subject_key(subject)]


def memory_tags() -> List[str]:
    return ["relational_preference", "command_affinity"]


def affinity_memory_filter(mem: Dict[str, Any]) -> bool:
    """True when a stored memory is *some* affinity record (any subject)."""
    return "relational_preference" in (mem.get("tags") or [])


def record_subject(mem: Dict[str, Any]) -> str:
    """The subject an affinity record belongs to, read back from the record.

    Falls back to ``ROUM`` for records written before subjects were recorded
    (state version 1), which were by definition Roum's.
    """
    state = mem.get("affinity_state")
    if isinstance(state, dict) and state.get("subject"):
        return str(state["subject"])
    keywords = mem.get("keywords") or []
    key = subject_key(ROUM)
    for word in keywords:
        if subject_key(word) == key:
            return ROUM
    return ROUM


def affinity_record_matches(mem: Dict[str, Any], subject: str = ROUM) -> bool:
    """True when ``mem`` is the affinity record *for this exact subject*."""
    return affinity_memory_filter(mem) and subject_key(record_subject(mem)) == subject_key(subject)


def load_state_from_memories(memories: List[Dict[str, Any]], subject: str = ROUM) -> Dict[str, Any]:
    """Recover the affinity state for ``subject`` from the relationship model.

    Only a record whose own subject matches is used, so one person's accumulated
    state can never be read back - or relabelled - as another's.
    """
    for mem in memories or []:
        if affinity_record_matches(mem, subject):
            state = _coerce_state(mem.get("affinity_state"), subject)
            state["subject"] = subject
            return state
    return _blank_state(subject)


__all__ = [
    "ROUM",
    "COMPONENTS",
    "ESTABLISHED_THRESHOLD",
    "EVENT_REQUEST",
    "EVENT_SUCCESS",
    "EVENT_FAILURE",
    "EVENT_VOLUNTARY_RETURN",
    "EVENT_RECALL",
    "EVENT_HARSH",
    "EVENT_INSULT",
    "EVENT_WARMTH",
    "looks_like_request",
    "looks_like_insult",
    "looks_like_harsh",
    "looks_like_warmth",
    "looks_like_failure",
    "looks_like_voluntary_return",
    "looks_like_recall",
    "record_event",
    "update_affinity",
    "affinity_confidence",
    "is_established",
    "render_statement",
    "affinity_diagnostics",
    "format_diagnostics",
    "prompt_block",
    "memory_content",
    "memory_keywords",
    "memory_tags",
    "affinity_memory_filter",
    "affinity_record_matches",
    "record_subject",
    "subject_key",
    "load_state_from_memories",
]
