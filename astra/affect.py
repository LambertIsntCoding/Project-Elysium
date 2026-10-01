"""Astra's current experiential affect.

Where ``relational.py`` holds a *relationship-specific* preference, this module
holds the *experiential* side of Astra's state: how engaged, curious,
concentrated, frustrated, invested, or anticipatory she currently is.

The two use the same accumulator mechanics on purpose, but they are deliberately
NOT merged. A book can leave Astra absorbed without changing how she feels about
Roum, and a blunt turn from Roum can sting without making her less curious. They
are separate records in separate models, and neither reads the other.

Design constraints:

* **Derived, not asserted.** Components move only from *experiences* recorded on
  the live path, never from a prompt build. Nothing here is assigned for the
  sake of making dialogue expressive.
* **Temporary.** This is current state, not memory. It decays toward neutral and
  is never promoted to a durable disposition by itself; that stays the job of
  the existing self-memory evidence gates.
* **Behavioural, not decorative.** The state exists to change *processing* -
  attention and retrieval breadth now, persistence/abandonment once the reading
  and question systems exist. ``prompt_block`` supplies grounded state for the
  model to express in her own words; it never supplies a line to perform.
* **Minimal.** Only dimensions with a concrete behavioural consumer are kept.
  "Uncertainty" is deliberately *not* a number here: it belongs to the question
  system (Slice 2), where it can be represented as something revisable rather
  than as a vague scalar.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# v1: the initial experiential-affect accumulator.
AFFECT_VERSION = 1

# Only dimensions with a behavioural consumer are represented. Adding a
# dimension without a consumer would be a number that never affects anything,
# which is exactly what this slice is meant to avoid.
AFFECT_COMPONENTS = (
    "engagement",             # willingness to keep going / select this activity
    "curiosity",              # pull toward exploring, noticing, following up
    "concentration",          # depth of processing, tolerance for difficulty
    "frustration",            # friction that pushes toward abandoning
    "emotional_investment",   # how much this matters to her
    "anticipation",           # forward pull toward an expected outcome
)

# Experience kinds. Kept as plain strings so callers do not depend on internals.
EXPERIENCE_READ = "read"
EXPERIENCE_COMPLETED = "completed"
EXPERIENCE_DISCOVERED = "discovered"
EXPERIENCE_REALIZATION = "realization"
EXPERIENCE_FRUSTRATION = "frustration"
EXPERIENCE_INTERACTION = "interaction"
EXPERIENCE_UNEXPECTED = "unexpected"
EXPERIENCE_ATTACHMENT = "attachment"

# How each kind of experience moves the current state. Deliberately small: a
# single experience nudges, it does not saturate. Significance and intensity
# scale the nudge (see ``record_event``).
_KIND_DELTAS: Dict[str, Dict[str, float]] = {
    EXPERIENCE_READ: {"engagement": 0.08, "concentration": 0.06, "curiosity": 0.04},
    EXPERIENCE_COMPLETED: {"engagement": 0.10, "anticipation": -0.08,
                           "frustration": -0.15, "emotional_investment": 0.06},
    EXPERIENCE_DISCOVERED: {"curiosity": 0.12, "engagement": 0.06},
    EXPERIENCE_REALIZATION: {"curiosity": 0.08, "engagement": 0.05,
                             "concentration": 0.03},
    EXPERIENCE_FRUSTRATION: {"frustration": 0.14, "engagement": -0.05},
    EXPERIENCE_INTERACTION: {"engagement": 0.06, "emotional_investment": 0.05},
    EXPERIENCE_UNEXPECTED: {"curiosity": 0.10, "concentration": -0.05},
    EXPERIENCE_ATTACHMENT: {"emotional_investment": 0.12, "engagement": 0.04},
}

_KIND_ALIASES = {
    "reading": EXPERIENCE_READ,
    "finished": EXPERIENCE_COMPLETED,
    "completed": EXPERIENCE_COMPLETED,
    "found": EXPERIENCE_DISCOVERED,
    "discovery": EXPERIENCE_DISCOVERED,
    "realised": EXPERIENCE_REALIZATION,
    "realized": EXPERIENCE_REALIZATION,
    "blocked": EXPERIENCE_FRUSTRATION,
    "failed": EXPERIENCE_FRUSTRATION,
    "surprised": EXPERIENCE_UNEXPECTED,
    "attached": EXPERIENCE_ATTACHMENT,
}

# State relaxation. Faster than the relational state on purpose: a current
# affective condition is momentary, so it should fade within days rather than
# linger for weeks. Neutral is zero - there is no "base mood" to fall back to.
AFFECT_BASELINE = 0.0
AFFECT_DECAY_PER_DAY = 0.15
AFFECT_MIN_INTERVAL_DAYS = 0.5
# Below this every component is treated as "no particular condition".
AFFECT_NEUTRAL_EPSILON = 0.02

_MIN, _MAX = 0.0, 1.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: Any, low: float = _MIN, high: float = _MAX) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(low, min(high, number))


def _parse_ts(value: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _blank_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {name: 0.0 for name in AFFECT_COMPONENTS}
    state.update({
        "version": AFFECT_VERSION,
        "event_counts": {},
        "observations": 0,
        "last_reason": "",
        "last_updated": "",
    })
    return state


def _coerce_state(raw: Any) -> Dict[str, Any]:
    state = _blank_state()
    if not isinstance(raw, dict):
        return state
    for name in AFFECT_COMPONENTS:
        if name in raw:
            state[name] = _clamp(raw.get(name))
    counts = raw.get("event_counts")
    if isinstance(counts, dict):
        state["event_counts"] = {str(k): int(v) for k, v in counts.items()
                                 if isinstance(v, (int, float))}
    state["observations"] = int(raw.get("observations", 0) or 0)
    state["last_reason"] = str(raw.get("last_reason", "") or "")
    state["last_updated"] = str(raw.get("last_updated", "") or "")
    return state


def _bump(state: Dict[str, Any], component: str, delta: float) -> None:
    state[component] = round(_clamp(_clamp(state.get(component)) + delta), 4)


def _decay(state: Dict[str, Any], now: Optional[datetime] = None) -> None:
    """Relax the current condition toward neutral as time passes.

    Applied lazily on the next experience, so state that is never touched again
    simply stays put on disk, and anything read back is aged when next used.
    """
    last = _parse_ts(state.get("last_updated"))
    now = now or datetime.now(timezone.utc)
    if last is None:
        state["last_updated"] = now.isoformat()
        return
    days = (now - last).total_seconds() / 86400.0
    if days < AFFECT_MIN_INTERVAL_DAYS:
        return
    factor = max(0.0, 1.0 - AFFECT_DECAY_PER_DAY * days)
    for name in AFFECT_COMPONENTS:
        current = _clamp(state.get(name))
        state[name] = round(AFFECT_BASELINE + (current - AFFECT_BASELINE) * factor, 4)


def is_neutral(state: Any) -> bool:
    """True when there is no particular current condition worth surfacing."""
    state = _coerce_state(state)
    return all(_clamp(state.get(name)) < AFFECT_NEUTRAL_EPSILON
               for name in AFFECT_COMPONENTS)


def record_event(state: Any, kind: str, *, intensity: float = 0.5,
                 significance: float = 0.5, text: str = "") -> Dict[str, Any]:
    """Fold one experience into the current affective state.

    ``state`` may be ``None`` (start fresh) or a previously stored dict. Only a
    known experience kind moves anything; an unknown kind is a no-op, so a
    caller that drifts cannot perturb the state with arbitrary input.
    """
    state = _coerce_state(state)
    _decay(state)
    resolved = _KIND_ALIASES.get(str(kind or "").strip().casefold(),
                                 str(kind or "").strip().casefold())
    deltas = _KIND_DELTAS.get(resolved)
    if deltas is None:
        return state

    scale = 0.6 + 0.7 * _clamp(intensity) + 0.5 * _clamp(significance)
    for component, base in deltas.items():
        _bump(state, component, base * scale)

    counts = state.setdefault("event_counts", {})
    counts[resolved] = int(counts.get(resolved, 0)) + 1
    state["observations"] = int(state.get("observations", 0)) + 1
    state["last_reason"] = text or f"Experienced something ({resolved})."
    state["last_updated"] = _now()
    return state


# ---------------------------------------------------------------------
# Behavioural consumers
# ---------------------------------------------------------------------
def retrieval_breadth(state: Any, base: int) -> int:
    """How many memories to retrieve, given the current condition.

    This is the first concrete way affect changes *processing* rather than
    wording: when Astra is engaged and curious she reaches for more context;
    when she is not, she stays narrow. Bounded so it can never balloon a prompt.
    """
    state = _coerce_state(state)
    extra = int(round(_clamp(state.get("engagement")) + _clamp(state.get("curiosity"))))
    return max(1, int(base) + min(2, extra))


# ---------------------------------------------------------------------
# Prompt material
# ---------------------------------------------------------------------
# Plain-language readings of each component. Astra experiences the *state*, not
# its numbers, so the prompt describes it the way she would - never as a
# measurement she could reason about (the boundary is implementation ->
# internal state -> experience -> self-interpretation).
_AFFECT_PHRASES: Dict[str, tuple] = {
    "engagement": ("strongly engaged", "engaged", "a little engaged"),
    "curiosity": ("very curious", "curious", "a little curious"),
    "concentration": ("concentrating well", "able to concentrate",
                      "concentration is uneven"),
    "frustration": ("quite frustrated", "frustrated", "a little frustrated"),
    "emotional_investment": ("deeply invested", "invested", "somewhat invested"),
    "anticipation": ("eager for something to happen", "anticipating something",
                     "mildly anticipating something"),
}


def render_summary(state: Any) -> Optional[str]:
    """A short, non-numeric description of the current condition.

    Returns ``None`` when neutral. Deliberately avoids any quantity: Astra is
    told how she seems to be doing, not what her values are.
    """
    state = _coerce_state(state)
    if is_neutral(state):
        return None
    salient = sorted(AFFECT_COMPONENTS, key=lambda n: _clamp(state.get(n)), reverse=True)
    phrases: List[str] = []
    for name in salient:
        value = _clamp(state.get(name))
        if value < AFFECT_NEUTRAL_EPSILON:
            continue
        levels = _AFFECT_PHRASES.get(name)
        if not levels:
            continue
        phrases.append(levels[0] if value >= 0.6 else
                       levels[1] if value >= 0.3 else levels[2])
    if not phrases:
        return None
    return "Right now Astra is " + ", ".join(phrases) + "."


def prompt_block(state: Any) -> Optional[str]:
    """The current condition as grounded context, or ``None`` when neutral.

    Presented as Astra's own state - not an instruction and not a line to
    perform - so the model expresses it only because it is actually there, and
    is free to express it in whatever way fits the moment.
    """
    summary = render_summary(state)
    if summary is None:
        return None
    return "\n".join([
        "=== ASTRA'S CURRENT CONDITION (TEMPORARY, NOT AN INSTRUCTION) ===",
        "This is Astra's present experiential state, accumulated from recent "
        "experiences. It may colour how she engages, but she does not have to "
        "announce it or name these numbers; express it only if it fits.",
        f"- {summary}",
        "- This is temporary and distinct from how she feels about Roum.",
    ])


# ---------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------
def memory_content() -> str:
    """The single self-memory record this state is persisted in."""
    return ("Astra's current experiential affect "
            "(engagement, curiosity, concentration, frustration, "
            "emotional investment, anticipation).")


def memory_tags() -> List[str]:
    return ["experiential_affect", "affect"]


def memory_keywords() -> List[str]:
    return ["affective state", "engagement", "curiosity", "current condition"]


def affect_memory_filter(mem: Dict[str, Any]) -> bool:
    """True when a stored memory is the affect accumulator (not a related one)."""
    return "experiential_affect" in (mem.get("tags") or [])


def load_state_from_memories(memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recover the current condition from the stored self-model record."""
    for mem in memories or []:
        if affect_memory_filter(mem):
            return _coerce_state(mem.get("affect_state"))
    return _blank_state()


def diagnostics(state: Any) -> Dict[str, Any]:
    """The diagnostic view: components plus why the state last moved."""
    state = _coerce_state(state)
    out = {name: round(_clamp(state.get(name)), 4) for name in AFFECT_COMPONENTS}
    out.update({
        "neutral": is_neutral(state),
        "event_counts": dict(state.get("event_counts", {})),
        "reason": str(state.get("last_reason", "") or ""),
    })
    return out


__all__ = [
    "AFFECT_COMPONENTS",
    "EXPERIENCE_READ",
    "EXPERIENCE_COMPLETED",
    "EXPERIENCE_DISCOVERED",
    "EXPERIENCE_REALIZATION",
    "EXPERIENCE_FRUSTRATION",
    "EXPERIENCE_INTERACTION",
    "EXPERIENCE_UNEXPECTED",
    "EXPERIENCE_ATTACHMENT",
    "is_neutral",
    "record_event",
    "retrieval_breadth",
    "render_summary",
    "prompt_block",
    "memory_content",
    "memory_tags",
    "memory_keywords",
    "affect_memory_filter",
    "load_state_from_memories",
    "diagnostics",
]
