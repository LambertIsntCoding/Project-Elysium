"""Presence: is Roum still here, and when should Astra stop waiting?

The Discord adapter already answers only when addressed. What it could not do
was notice that an exchange had *ended*: a person says something, she answers,
and then nothing comes back. This module is the single, pure decision about
that - whether an exchange is still live, or whether Roum has walked away and
Astra should go back to her own work and merely check for messages later.

It is deliberately small and separate:

* **Derived, never stored.** Nothing here is a memory, an accumulator or a
  durable state. The same elapsed time and the same affect always render the
  same answer, so it cannot decay, be reinforced, or be quoted back.
* **Time is a fact; affect changes the threshold.** The *gap* is measured, but
  how long she is willing to wait is scaled by how she currently is - an
  engaged, warm state waits longer; a flat, weary or withdrawn one gives up
  sooner and returns to her schedule. That is the whole point: the timing is
  not a fixed constant, it is a consequence of her state.
* **No feeling is invented from a gap.** Walking away is not rejection and not
  loneliness; it is simply "no exchange in progress". The module never
  manufactures an emotion from elapsed time.

The adapter consumes :func:`assess`; the runtime consumes the same decision so
that "she is waiting for Roum" and "she is back at her own work" cannot drift.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from . import behavior

# Presence modes. Continuous in spirit (a single elapsed value maps to one
# mode), but three names make the behaviour legible to the adapter and to the
# admin views.
PRESENT = "present"        # a live exchange: she answers promptly
LINGERING = "lingering"    # was here, may still be around: she waits
AWAY = "away"              # gone quiet: back to her own work, check later

# How long, at a neutral state, she will hold an exchange open before treating
# it as over. The affect layer scales this; these are the outer bounds.
DEFAULT_PATIENCE_SECONDS = 300.0
MIN_PATIENCE_SECONDS = 60.0
MAX_PATIENCE_SECONDS = 3600.0

# Below this fraction of her patience she is still "present"; above it she is
# "lingering" until the patience runs out.
_LINGER_FRACTION = 0.5


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def patience_seconds(modulation: Optional[Dict[str, Any]] = None, *,
                     base: float = DEFAULT_PATIENCE_SECONDS) -> float:
    """How long she will wait for a reply, scaled by how she currently is.

    A pure function of the modulation: engaged and warm states lengthen the
    wait, a flat/weary/withdrawn state shortens it. Bounded so no state makes
    her wait forever or abandon the person instantly.
    """
    base = max(0.0, float(base))
    scale = 1.0
    if isinstance(modulation, dict):
        scale = float(behavior.timing_modifiers(modulation).get("patience_scale", 1.0))
    return round(_clamp(base * scale, MIN_PATIENCE_SECONDS, MAX_PATIENCE_SECONDS), 3)


def assess(seconds_since: Optional[float], *,
           modulation: Optional[Dict[str, Any]] = None,
           base_patience: float = DEFAULT_PATIENCE_SECONDS) -> Dict[str, Any]:
    """Decide presence from the elapsed time since the last exchange.

    ``seconds_since`` is the gap since Roum last addressed her, or ``None`` when
    no exchange has happened yet. Returns ``{mode, patience, resume_background,
    reason}``; ``resume_background`` is true exactly when she should stop
    holding the exchange open and go back to her own work.
    """
    patience = patience_seconds(modulation, base=base_patience)
    if seconds_since is None:
        return {"mode": AWAY, "patience": patience, "resume_background": True,
                "reason": "no exchange in progress", "elapsed": None}
    gap = max(0.0, float(seconds_since))
    if gap < patience * _LINGER_FRACTION:
        return {"mode": PRESENT, "patience": patience, "resume_background": False,
                "reason": "", "elapsed": gap}
    if gap < patience:
        return {"mode": LINGERING, "patience": patience,
                "resume_background": False,
                "reason": "Roum has been quiet for a little while", "elapsed": gap}
    return {"mode": AWAY, "patience": patience, "resume_background": True,
            "reason": "Roum has gone quiet; back to her own work", "elapsed": gap}


__all__ = [
    "PRESENT", "LINGERING", "AWAY", "DEFAULT_PATIENCE_SECONDS",
    "MIN_PATIENCE_SECONDS", "MAX_PATIENCE_SECONDS",
    "patience_seconds", "assess",
]
