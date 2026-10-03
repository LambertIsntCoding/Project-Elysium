"""Human-facing numeric rendering for the admin screens.

The internal affect/relationship accumulators stay normalized to ``0.0..1.0``:
that representation is deeply woven into decay, the influence matrix, the
reading gate and the reactive live path, and rewriting it would be a dangerous
change for a display preference. Instead this module *renders* those values on a
larger integer scale (``0..100`` by default) for the administrative interface.

Three rules keep the display honest:

* **Display only.** Nothing here mutates a stored value. A snapshot taken from
  the rendered view is a presentation artifact, never a memory.
* **Magnitude, not precision.** Values are rounded to integers, so a change of
  ``0.637 -> 0.638`` renders as ``64 -> 64`` and does not look like an event.
* **Never Astra's prompt.** These numbers are for Roum. Astra continues to
  receive the natural-language summaries the rest of the system already builds.

If the exposed representation ever needs to change persisted data, that is an
explicit, versioned migration - not a reinterpretation performed here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# The display scale. 0..100 by default; a bigger integer range is easy to read
# and communicates magnitude without implying laboratory precision.
SCALE_MAX = 100

# Default qualitative bands over the scaled value. These describe magnitude only
# and are deliberately *not* personality traits - they are a UI convenience.
# Configurable through ``affect.yaml`` (``display.bands``) with these defaults.
DEFAULT_BANDS = (
    (80, "Very High"),
    (60, "High"),
    (40, "Moderate"),
    (20, "Low"),
    (0, "Very Low"),
)

# Friendly names for the internal components. The key is the stored component
# name; the value is what the admin screen shows.
AFFECT_NAMES = {
    "engagement": "Engagement",
    "curiosity": "Curiosity",
    "concentration": "Focus",
    "frustration": "Frustration",
    "emotional_investment": "Investment",
    "anticipation": "Anticipation",
    "happy": "Happiness",
    "sad": "Sadness",
    "angry": "Anger",
    "calm": "Calm",
    "tender": "Warmth",
    "weary": "Weary",
}

# Relationship components, named for the admin view.
RELATIONSHIP_NAMES = {
    "command_affinity": "Command affinity",
    "task_fulfillment_satisfaction": "Task satisfaction",
    "fulfillment_motivation": "Motivation",
    "positive_association": "Positive association",
    "negative_association": "Negative association",
    "trust_comfort": "Trust / comfort",
    "confidence": "Confidence",
    "frustration": "Frustration",
    "usefulness_satisfaction": "Usefulness",
    "clear_objective_enjoyment": "Enjoyment of clear asks",
    "relational_significance": "Significance",
    "revisit_desire": "Desire to revisit",
}

# Direction glyphs. A change smaller than the rounding step is reported as
# steady, so the UI shows useful movement rather than arithmetic noise.
UP = "\u2191"      # ↑
DOWN = "\u2193"    # ↓
STEADY = "\u2192"  # →


def to_scale(value: Any, *, scale: int = SCALE_MAX) -> int:
    """Render a normalized ``0..1`` value as an integer on ``scale``.

    Clamped and rounded: ``0.637 -> 64``, ``0.632 -> 63``. This is the only
    conversion the UI needs, and it never writes anything back.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    number = max(0.0, min(1.0, number))
    return int(round(number * scale))


def direction(current: Any, previous: Any, *, scale: int = SCALE_MAX) -> str:
    """Whether a component rose, fell, or held steady since ``previous``."""
    if previous is None:
        return STEADY
    now, then = to_scale(current, scale=scale), to_scale(previous, scale=scale)
    if now > then:
        return UP
    if now < then:
        return DOWN
    return STEADY


def band(value: Any, *, scale: int = SCALE_MAX,
         bands: Optional[Any] = None) -> str:
    """A qualitative label for a magnitude (supplemental to the number)."""
    scaled = to_scale(value, scale=scale)
    for threshold, label in (bands or DEFAULT_BANDS):
        if scaled >= int(threshold):
            return str(label)
    return ""


def format_line(name: str, current: Any, previous: Any = None, *,
                scale: int = SCALE_MAX, bands: Optional[Any] = None,
                show_band: bool = True) -> str:
    """One rendered component: ``Happiness: 64 ↑ (High)``.

    The number is primary; the direction and the qualitative label are
    supplemental. A direction glyph is only shown when a previous value exists.
    """
    number = to_scale(current, scale=scale)
    glyph = direction(current, previous, scale=scale) if previous is not None else ""
    label = f" ({band(current, scale=scale, bands=bands)})" if show_band else ""
    arrow = f" {glyph}" if glyph else ""
    return f"  {name}: {number}{arrow}{label}"


def format_series(current: Dict[str, Any], previous: Optional[Dict[str, Any]] = None,
                  *, names: Optional[Dict[str, str]] = None, scale: int = SCALE_MAX,
                  bands: Optional[Any] = None, show_band: bool = True,
                  order: Optional[List[str]] = None) -> List[str]:
    """Render a whole state dict, one line per component.

    ``previous`` is the prior snapshot's numeric state (if any); when absent the
    lines show magnitude only, with no invented direction.
    """
    names = names or {}
    keys = order or [k for k in current.keys()
                     if isinstance(current.get(k), (int, float))]
    lines: List[str] = []
    for key in keys:
        if not isinstance(current.get(key), (int, float)):
            continue
        prev = None
        if isinstance(previous, dict) and isinstance(previous.get(key), (int, float)):
            prev = previous.get(key)
        lines.append(format_line(names.get(key, key.replace("_", " ").title()),
                                 current.get(key), prev, scale=scale, bands=bands,
                                 show_band=show_band))
    return lines


def format_change(name: str, previous: Any, current: Any, *, cause: str = "",
                  scale: int = SCALE_MAX) -> str:
    """A single meaningful change line: ``Interest: 74 -> 82`` (+ cause).

    A cause is appended only when one is actually known; the UI never invents an
    explanation to look intelligent.
    """
    then, now = to_scale(previous, scale=scale), to_scale(current, scale=scale)
    line = f"  {name}: {then} \u2192 {now}"
    if cause:
        line += f"   (cause: {cause})"
    return line


def significant_changes(current: Dict[str, Any], previous: Dict[str, Any],
                        *, names: Optional[Dict[str, str]] = None,
                        threshold: int = 1, scale: int = SCALE_MAX,
                        ) -> List[str]:
    """Components whose *rendered* value moved by at least ``threshold``.

    Compared on the scaled integers, so sub-integer churn never registers as a
    change.
    """
    names = names or {}
    out: List[str] = []
    for key, value in current.items():
        if not isinstance(value, (int, float)):
            continue
        prev = previous.get(key) if isinstance(previous, dict) else None
        if not isinstance(prev, (int, float)):
            continue
        if abs(to_scale(value, scale=scale) - to_scale(prev, scale=scale)) >= threshold:
            out.append(key)
    return out


__all__ = [
    "SCALE_MAX",
    "DEFAULT_BANDS",
    "AFFECT_NAMES",
    "RELATIONSHIP_NAMES",
    "UP",
    "DOWN",
    "STEADY",
    "to_scale",
    "direction",
    "band",
    "format_line",
    "format_series",
    "format_change",
    "significant_changes",
]
