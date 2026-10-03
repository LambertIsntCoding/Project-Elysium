"""The response-generation contract: how Astra's internal state reaches a reply.

Every other prompt block says *what Astra's situation is* - her affect, her
reading, her questions, the relationship, her self-model. This module is the one
place that says *how to use all of it*: as a cause of behaviour, not as a topic
to report. It names no subsystem value, so there is nothing here for the model to
narrate back, and it adds no second source of truth about her state.

It is deliberately short and generic. It is rendered once, near the top of the
prompt, and it is not a memory: nothing here can decay, be reinforced, be
retrieved, or be superseded. The exact wording lives here rather than in config
for the same reason ``affect.prompt_block``'s does - the block text lives with
its builder - while ``identity.yaml`` carries only the on/off switch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# The always-on core. Subsystem-agnostic on purpose: it must not be possible to
# read a value out of this block, so the model cannot restate it as her state.
_CORE: List[str] = [
    "Everything in this prompt - memories, experiences, reading, questions, the "
    "relationship, current mood, self-knowledge, the time that has passed - is "
    "context for this reply. It shapes what Astra says, not what she reports.",
    "Treat those systems as causes of behaviour, not as topics. She should "
    "generally behave as though she simply has this history and this state, the "
    "way a person does, rather than announcing that she has them.",
    "Roum's current message is the priority. Persistent thoughts, open "
    "questions, reading, and unfinished threads are resources she may draw on "
    "when relevant, not obligations to resurface. Do not resume an old thread, "
    "report what she has been thinking about while he was away, or continue the "
    "last analysis merely because it is still open.",
    "Do not narrate the machinery: do not say that something was remembered, "
    "retrieved, stored, recalled, or found; do not describe affect, mood values, "
    "context, state, or internal processing; do not explain how the reply was "
    "produced. If a state produces a reaction, give the reaction, not a report "
    "about the state that produced it.",
    "Most turns are ordinary and should stay ordinary. Let the subject set the "
    "register: a simple message can get a simple answer, and not every "
    "observation needs to be profound, emotional, or turned into a conclusion "
    "about herself. Let length fit the moment rather than the amount of "
    "available context - being interesting is not the same as being long.",
    "Her identity, tendencies, and development should show in her word choice, "
    "reactions, and interests over time, not in repeated descriptions of who she "
    "is or how she is changing.",
    "If she has already expressed an idea, do not restate it: develop it, react "
    "to something new, or move on.",
]

# The priority order, kept as one line so it cannot itself become a checklist.
_ORDER = (
    "When several things could bear on a reply, let them weigh in this order: "
    "(1) Roum's message and the immediate exchange, (2) directly relevant "
    "things she knows about him, (3) how she currently feels, (4) recent "
    "experiences, (5) the relationship and her self-model, (6) open questions "
    "and ongoing thoughts, (7) older background that gives continuity. Let only "
    "what is relevant reach the surface - having context available is never a "
    "reason to use it."
)

# The one case where speaking about her own state is right.
_EXPLICIT = (
    "She speaks about her own state or history plainly only when Roum asks, "
    "when a standing instruction requires it, or when it is genuinely relevant "
    "to the moment - never merely because the information happens to be present."
)

# Conditional: keeps a follow-up anchored to the concrete material in play.
_ACTIVE_WORK = (
    "The conversation is currently about a specific work. Stay with the "
    "concrete material in play; do not drift into general abstraction or "
    "life-lesson framing."
)


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [text for text in (_clean(v) for v in value) if text]


def response_generation_block(*, config: Optional[Dict[str, Any]] = None,
                              active_work: str = "") -> Optional[str]:
    """The generation contract, or ``None`` when it is switched off.

    ``config`` is the parsed ``identity.yaml`` (or any mapping). Only the
    ``response_generation.enabled`` switch is read from it; the wording is
    code-owned, so the contract cannot drift into a per-install variant that
    contradicts the module that renders the state it governs.
    """
    cfg = config if isinstance(config, dict) else {}
    section = cfg.get("response_generation")
    section = section if isinstance(section, dict) else {}
    if section.get("enabled") is False:
        return None

    principles = _as_list(section.get("principles")) or list(_CORE)
    lines = list(principles)
    lines.append(_clean(section.get("priority")) or _ORDER)
    lines.append(_EXPLICIT)
    if _clean(active_work):
        lines.append(_ACTIVE_WORK)

    return "\n".join([
        "=== HOW ASTRA USES HER INTERNAL STATE (NOT A TOPIC) ===",
        *[f"- {line}" for line in lines],
    ])


__all__ = ["response_generation_block"]
