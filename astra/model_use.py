"""When the language model may be invoked - one central, config-driven answer.

Astra is not the model. The model is her *voice*: it renders a reply she has
already decided to give, or extracts a memory from a turn she already had. It is
not who decides her state, her memory, her schedule, or her behaviour - all of
that is application code. The only question this module answers is the narrow
one the project owner asked: *at which moments is that voice allowed to be
invoked?*

The model must be active only when it is actually doing one of those jobs. This
maps the project owner's examples onto the *real* model call sites in the code:

* ``conversation``   - Roum is talking to her and she is composing the reply.
                       Only ever runs during a live turn, so it is not the
                       "active on its own" case and is on by default.
* ``consolidation``  - turning a turn just had into governed memories (a second
                       model call per turn). On by default.
* ``elysium``        - structuring an explicitly-requested command from Roum's
                       own text. On by default.
* ``proactive_compose`` - phrasing an *already-decided* proactive message in her
                       own voice. Only ever runs after the deterministic
                       notification gate has permitted a message, so it can
                       never originate a reason to speak. On by default; the
                       deterministic summary is the fallback when it is off.
* ``reading``        - the *background* reader summarising a passage while Roum
                       is away. This is where she forms her own view on parts of
                       a book. It is the only job that runs the model while Roum
                       is not interacting, so it is **off by default**: her
                       voice should not be running just because the program is
                       on.

Proactive *deciding* ("pinging to talk to me") is deliberately *not* a key
here: the notification decision layer does not call the model at all. Only the
optional phrasing of a permitted message is gated, by ``proactive_compose``.

This module stores nothing and decides nothing about *what* is said - it is a
pure read of configuration. Disabling a feature never disables the deterministic
work behind it (scheduling, retrieval, memory governance, file maintenance all
keep running); it only stops the model call.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

# The model-invoking features. Order is for stable reporting only.
FEATURE_CONVERSATION = "conversation"
FEATURE_CONSOLIDATION = "consolidation"
FEATURE_ELYSIUM = "elysium"
FEATURE_PROACTIVE_COMPOSE = "proactive_compose"
FEATURE_READING = "reading"

FEATURES = (FEATURE_CONVERSATION, FEATURE_CONSOLIDATION, FEATURE_ELYSIUM,
            FEATURE_PROACTIVE_COMPOSE, FEATURE_READING)

# Conservative defaults. Every interaction-time job is on; the background
# reader - the one that would keep the model busy while Roum is away - is off.
_DEFAULT_ENABLED = {
    FEATURE_CONVERSATION: True,
    FEATURE_CONSOLIDATION: True,
    FEATURE_ELYSIUM: True,
    FEATURE_PROACTIVE_COMPOSE: True,
    FEATURE_READING: False,
}


class ModelUse:
    """Which model-invoking features are permitted, resolved from configuration."""

    def __init__(self, enabled: Dict[str, bool]) -> None:
        self._enabled = {f: bool(enabled.get(f, _DEFAULT_ENABLED[f])) for f in FEATURES}

    def enabled(self, feature: str) -> bool:
        """Whether ``feature`` may invoke the model. Unknown features are off."""
        return bool(self._enabled.get(feature, False))

    @property
    def enabled_features(self) -> Iterable[str]:
        return [f for f in FEATURES if self._enabled[f]]

    def as_dict(self) -> Dict[str, bool]:
        return dict(self._enabled)


def from_config(config: Any) -> ModelUse:
    """Resolve the gate from a runtime config mapping.

    Reads ``runtime.model_use`` (or a top-level ``model_use``). A missing or
    malformed block yields the conservative defaults rather than failing, so an
    unconfigured runtime is the least-active one.
    """
    cfg = config if isinstance(config, dict) else {}
    block = cfg.get("model_use")
    if not isinstance(block, dict):
        runtime = cfg.get("runtime")
        block = runtime.get("model_use") if isinstance(runtime, dict) else None
    block = block if isinstance(block, dict) else {}
    enabled = dict(_DEFAULT_ENABLED)
    for feature in FEATURES:
        if feature in block:
            enabled[feature] = bool(block[feature])
    return ModelUse(enabled)
