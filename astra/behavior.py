"""Behaviour modulation: turning internal state into *how* Astra acts.

The systems this module reads (experiential affect, felt time, stalled things)
already exist and are deliberately separate. What was missing was a single,
explicit place where they become *behavioural modifiers* layered onto the
current turn - so sadness actually lowers initiative and pace, a long absence
actually changes how much she takes on, and strong positive affect actually
raises enthusiasm, instead of those states merely existing as numbers or as
prompt narration.

Design constraints:

* **Continuous, never a mode machine.** The output is a set of bounded scalar
  modifiers (``0..1``), not a discrete "short/medium/long mode". Nothing here
  ever puts Astra into a named regime; the modifiers mix and are floored so a
  real event still lands.
* **Behavioural, not declarative.** :func:`prompt_block` renders the modulation
  as behavioural consequences ("quieter, slower to start"), never as an
  instruction to announce a feeling, and never as raw numbers. This is what
  keeps the state a *cause* of behaviour rather than a topic.
* **Grounded in events.** A modifier is only produced from a state the system
  actually recorded, and a "reason" is only surfaced when the affect layer or
  the temporal layer actually has one. The module never invents a cause.
* **Temporary and recoverable.** It reads the temporary affect accumulator and
  the derived temporal view; it stores nothing. Sadness recovers through the
  existing affect decay, and the modulation relaxes with it.
* **Separate systems stay separate.** Nothing here writes affect, memory,
  relationship, reading or inquiry state. It reads them and returns modifiers.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from . import affect as affect_mod
from . import temporal

# Baseline (neutral) modulation: no state, no effect.
_NEUTRAL = {
    "energy": 0.5,
    "initiative": 0.5,
    "enthusiasm": 0.5,
    "pace": 0.5,
    "persistence": 0.5,
    "interest": 0.5,
    "brevity": 0.0,
    "hesitation": 0.0,
}

# How strongly each emotion pulls the behavioural modifiers. Kept as named
# components so the mapping is auditable and tunable. Positive weight raises the
# modifier; negative lowers it. ``brevity``/``hesitation`` are "toward" measures
# (higher = more of the trait), so sadness *raises* them.
_EMOTION_EFFECTS: Dict[str, Dict[str, float]] = {
    # Prolonged sadness reduces motivation, initiative, enthusiasm, pace and
    # interest, and makes her more hesitant and quieter. It never redefines her
    # personality: this only scales behaviour while the state persists.
    "sad": {"energy": -0.5, "initiative": -0.5, "enthusiasm": -0.45,
            "pace": -0.35, "persistence": -0.35, "interest": -0.4,
            "brevity": 0.5, "hesitation": 0.5},
    "weary": {"energy": -0.45, "pace": -0.4, "persistence": -0.4,
              "enthusiasm": -0.25, "brevity": 0.45, "hesitation": 0.2},
    "angry": {"energy": 0.15, "pace": 0.2, "initiative": 0.1, "interest": 0.05,
              "brevity": 0.2},
    "frustration": {"energy": -0.2, "persistence": -0.35, "initiative": -0.15,
                    "hesitation": 0.3, "interest": -0.1},
    # Positive affect raises enthusiasm, initiative and engagement.
    "happy": {"energy": 0.5, "initiative": 0.45, "enthusiasm": 0.55,
              "pace": 0.35, "interest": 0.35, "brevity": -0.2, "hesitation": -0.3},
    "calm": {"energy": 0.1, "hesitation": -0.2, "brevity": -0.1},
    "tender": {"enthusiasm": 0.2, "interest": 0.2, "pace": -0.1, "brevity": -0.15},
}

# The processing dimensions also move behaviour: engagement/curiosity/
# anticipation raise drive; concentration steadies persistence.
_PROCESSING_EFFECTS: Dict[str, Dict[str, float]] = {
    "engagement": {"energy": 0.4, "initiative": 0.35, "interest": 0.3,
                   "pace": 0.2, "persistence": 0.25},
    "curiosity": {"interest": 0.45, "initiative": 0.3, "enthusiasm": 0.25},
    "anticipation": {"enthusiasm": 0.35, "initiative": 0.3, "energy": 0.2},
    "concentration": {"persistence": 0.35, "energy": 0.15},
    "emotional_investment": {"persistence": 0.3, "interest": 0.2,
                             "enthusiasm": 0.2},
}

# A component must be at least this strong before it moves behaviour, so a
# near-neutral state has no effect at all.
_EFFECT_THRESHOLD = 0.12

# How a long absence modulates behaviour. A gap is not a mood: a very long gap
# makes her a little slower to re-orient, while genuine engagement and a
# long-running project raise persistence. All small and continuous.
_GAP_REORIENT_SECONDS = 3 * 86400.0
_LONG_PROJECT_SECONDS = 7 * 86400.0


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def from_affect(state: Any) -> Dict[str, Any]:
    """Behavioural modifiers implied by the current experiential affect.

    Reads the affect accumulator read-only. Only components above a small
    threshold contribute, so a neutral state produces no modulation. Returns the
    neutral baseline when there is no state at all.
    """
    diag = affect_mod.diagnostics(state)
    modulation = dict(_NEUTRAL)
    reasons: List[str] = []

    for component, weights in {**_EMOTION_EFFECTS, **_PROCESSING_EFFECTS}.items():
        value = _num(diag.get(component))
        if value < _EFFECT_THRESHOLD:
            continue
        # Scale by the component's magnitude so a strong state moves behaviour
        # more than a faint one, and a faint one still moves it a little.
        scale = _clamp(value)
        for target, weight in weights.items():
            modulation[target] = modulation.get(target, 0.5) + weight * scale

    # Name the strongest causes in plain language, using the affect layer's own
    # recorded reason rather than inventing one.
    last_reason = str(diag.get("reason") or "").strip()
    if last_reason and not diag.get("neutral"):
        reasons.append(last_reason)

    for key in list(modulation):
        modulation[key] = round(_clamp(modulation[key]), 4)
    modulation["reasons"] = reasons
    return modulation


def from_temporal(*, last_seen: Any = None, open_questions: Iterable[Dict[str, Any]] = (),
                  works: Iterable[Dict[str, Any]] = (), now: Any = None,
                  base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Layer felt time onto a modulation dict.

    Elapsed time and scale of activity act as *modifiers*, not a state machine:
    a long gap slightly lowers immediate pace/initiative (she is re-orienting),
    a long-running work slightly raises persistence, and a genuinely stalled
    unresolved thing raises initiative toward it - nothing more.
    """
    modulation = dict(base or _NEUTRAL)
    reasons: List[str] = list(modulation.get("reasons") or [])

    seconds = temporal._seconds_since(last_seen, now) if last_seen else None
    if seconds is not None and seconds >= _GAP_REORIENT_SECONDS:
        modulation["pace"] = modulation.get("pace", 0.5) - 0.1
        modulation["initiative"] = modulation.get("initiative", 0.5) - 0.08
        reasons.append("it has been a while since the last conversation")

    # A long-running project raises persistence a touch - she stays with it.
    long_project = False
    for work in works or ():
        age = temporal._age_days_since(
            work.get("started_at") or work.get("first_seen"), now)
        if age is not None and age * 86400.0 >= _LONG_PROJECT_SECONDS:
            long_project = True
            break
    if long_project:
        modulation["persistence"] = modulation.get("persistence", 0.5) + 0.12
        reasons.append("a work she has been with for a long time")

    # A genuinely stalled unresolved thing raises initiative toward it only.
    stalled = temporal.impatience_items(open_questions, works, now=now)
    if stalled:
        modulation["initiative"] = modulation.get("initiative", 0.5) + 0.12
        reasons.append("something she cares about has been stuck a while")

    for key in _NEUTRAL:
        modulation[key] = round(_clamp(modulation.get(key, _NEUTRAL[key])), 4)
    modulation["reasons"] = reasons
    return modulation


def combine(*modulations: Dict[str, Any]) -> Dict[str, Any]:
    """Merge modulation dicts by averaging each component (bounded, continuous)."""
    present = [m for m in modulations if isinstance(m, dict)]
    if not present:
        return dict(_NEUTRAL, reasons=[])
    out: Dict[str, Any] = {}
    for key in _NEUTRAL:
        values = [_num(m.get(key, _NEUTRAL[key])) for m in present]
        out[key] = round(_clamp(sum(values) / len(values)), 4)
    reasons: List[str] = []
    for m in present:
        for reason in m.get("reasons") or []:
            if reason not in reasons:
                reasons.append(reason)
    out["reasons"] = reasons
    return out


def build(*, affect_state: Any = None, last_seen: Any = None,
          open_questions: Iterable[Dict[str, Any]] = (),
          works: Iterable[Dict[str, Any]] = (), now: Any = None) -> Dict[str, Any]:
    """The full modulation for a turn: affect, then felt time layered on."""
    base = from_affect(affect_state)
    return from_temporal(last_seen=last_seen, open_questions=open_questions,
                         works=works, now=now, base=base)


def is_neutral(modulation: Any) -> bool:
    """True when the modulation is close enough to baseline to say nothing."""
    if not isinstance(modulation, dict):
        return True
    for key, baseline in _NEUTRAL.items():
        if abs(_num(modulation.get(key, baseline)) - baseline) >= 0.06:
            return False
    return True


def reading_modifiers(modulation: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the modulation into the knobs ``reading.reading_pace`` reads.

    Reading already takes concentration/weary/engagement/frustration; a lowered
    energy or enthusiasm should read like a lower engagement, and hesitation
    like a little more weary. This keeps one arithmetic: the reading gate is
    not reimplemented, only fed a state that reflects the modulation.
    """
    if not isinstance(modulation, dict):
        return {}
    energy = _num(modulation.get("energy", 0.5))
    enthusiasm = _num(modulation.get("enthusiasm", 0.5))
    interest = _num(modulation.get("interest", 0.5))
    hesitation = _num(modulation.get("hesitation", 0.0))
    # Map the derived drive onto the existing reading inputs (0..1).
    return {
        "engagement": round(_clamp((energy + interest) / 2.0), 4),
        "weary": round(_clamp(hesitation), 4),
        "concentration": round(_clamp((energy + enthusiasm) / 2.0), 4),
    }


def prompt_block(modulation: Dict[str, Any]) -> Optional[str]:
    """The modulation as behavioural guidance, or ``None`` when neutral.

    Phrased entirely as *how she behaves* - pace, initiative, how much she says,
    how ready she is to start something - never as a feeling to announce and
    never as a number. This is the behavioural half of the affect/state system;
    it is what makes sadness matter beyond a scalar.
    """
    if is_neutral(modulation):
        return None
    lines: List[str] = []
    energy = _num(modulation.get("energy", 0.5))
    initiative = _num(modulation.get("initiative", 0.5))
    enthusiasm = _num(modulation.get("enthusiasm", 0.5))
    pace = _num(modulation.get("pace", 0.5))
    brevity = _num(modulation.get("brevity", 0.0))
    hesitation = _num(modulation.get("hesitation", 0.0))

    if energy <= 0.35 and initiative <= 0.4:
        lines.append(
            "She is running low right now: less motivated, slower to start, and "
            "less likely to take the lead or pick up something optional. Let "
            "that show in what she does - shorter, quieter, less eager to "
            "initiate - without announcing it or making it about herself.")
    elif energy >= 0.65 and enthusiasm >= 0.6:
        lines.append(
            "She is running warm right now: more enthusiastic, quicker to "
            "engage, more willing to take the lead and run with something. Let "
            "it show in her energy and initiative rather than describing it.")
    else:
        bits = []
        if pace <= 0.35:
            bits.append("a little slower and more measured")
        elif pace >= 0.65:
            bits.append("a little quicker")
        if enthusiasm >= 0.62:
            bits.append("warmer and more animated")
        if brevity >= 0.4:
            bits.append("inclined to keep it short")
        if hesitation >= 0.4:
            bits.append("more hesitant about starting something new")
        if bits:
            lines.append("Her current state leans " + ", ".join(bits) + ".")

    if lines:
        lines.insert(0, "=== HOW ASTRA IS RUNNING RIGHT NOW (BEHAVIOUR, NOT A TOPIC) ===")
        lines.append(
            "This is a modifier on *how* she engages this turn, not something to "
            "report, and not a change to who she is - it relaxes as the state "
            "does.")
    return "\n".join(lines) if lines else None


__all__ = [
    "from_affect",
    "from_temporal",
    "combine",
    "build",
    "is_neutral",
    "reading_modifiers",
    "prompt_block",
]
