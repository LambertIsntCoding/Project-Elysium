"""Response planning and the speech renderer boundary.

Astra's internal state and response *planning* are separated from the final
speech renderer. Between the runtime and the language model / voice system sits
a :class:`ResponsePlan`: a small description of how a response should be
delivered - intent, conversational function, urgency, energy, emotional
expression, and whether it suits an interruption.

Two hard rules:

* **The plan is internal.** Its values are never injected into Astra's
  self-description and never shown to her as numeric introspection. The plan is
  a delivery hint for the voice layer, not something she introspects.
* **The voice system is not the source of personality.** It renders a plan and a
  text response that the language model produced; it does not decide who she is.
  A text-only path and a voice path share the same underlying response, so the
  same reply can be spoken or displayed unchanged.

The renderer is an interface (:class:`VoiceRenderer`) with a no-op default, so
voice can be replaced independently of memory, reasoning, scheduling and Discord.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

# -- conversational function -------------------------------------------------
FUNC_ANSWER = "answer"
FUNC_QUESTION = "question"
FUNC_GREETING = "greeting"
FUNC_ACKNOWLEDGEMENT = "acknowledgement"
FUNC_INITIATIVE = "initiative"        # Astra proactively raising something
FUNC_NOTIFICATION = "notification"    # a task/result notice
FUNC_STATUS = "status"                # a status report
FUNC_ERROR = "error"

# -- intent ------------------------------------------------------------------
INTENT_INFORM = "inform"
INTENT_ASK = "ask"
INTENT_SHARE = "share"
INTENT_WARN = "warn"
INTENT_CONFIRM = "confirm"

_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass
class ResponsePlan:
    """How a response should be delivered. Internal; never introspected."""

    function: str = FUNC_ANSWER
    intent: str = INTENT_INFORM
    urgency: float = 0.0          # 0..1
    energy: float = 0.5           # 0..1
    emotional_expression: str = "neutral"   # a word, not a number
    interruption_suitability: float = 0.0   # 0..1
    proactive: bool = False
    source: str = "conversation"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        """A short human-readable summary for diagnostics (not for the prompt)."""
        return (f"{self.function}/{self.intent} "
                f"urgency={self.urgency:.2f} energy={self.energy:.2f} "
                f"interrupt={self.interruption_suitability:.2f}")


def _has_question(text: str) -> bool:
    return "?" in text or bool(re.search(r"\b(what|why|how|when|where|who)\b", text, re.I))


def plan_response(user_input: str, response: str, *,
                  error: bool = False, proactive: bool = False,
                  notification: bool = False,
                  emotional_expression: str = "neutral") -> ResponsePlan:
    """Derive a delivery plan from the turn, without a model call.

    Deterministic and cheap: the plan is a property of the exchange, decided by
    the application, not a second generation step. It never becomes part of
    Astra's self-description.
    """
    text = str(response or "")
    user = str(user_input or "")

    if error or text.startswith("[Error"):
        return ResponsePlan(function=FUNC_ERROR, intent=INTENT_WARN,
                            urgency=0.6, energy=0.4,
                            interruption_suitability=0.0,
                            source="error")
    if notification:
        return ResponsePlan(function=FUNC_NOTIFICATION, intent=INTENT_INFORM,
                            urgency=0.4, energy=0.5,
                            interruption_suitability=0.3,
                            proactive=proactive, source="task")
    if proactive:
        return ResponsePlan(function=FUNC_INITIATIVE, intent=INTENT_SHARE,
                            urgency=0.5, energy=0.6,
                            interruption_suitability=0.7,
                            proactive=True, source="proactive")

    function = FUNC_ANSWER
    intent = INTENT_INFORM
    if _has_question(text) and not _has_question(user):
        function = FUNC_QUESTION
        intent = INTENT_ASK
    elif not user.strip():
        function = FUNC_GREETING
    elif len(_WORD_RE.findall(text)) <= 6 and _has_question(user):
        function = FUNC_ACKNOWLEDGEMENT

    return ResponsePlan(function=function, intent=intent,
                        urgency=0.1, energy=0.5,
                        emotional_expression=emotional_expression,
                        interruption_suitability=0.0,
                        source="conversation")


class VoiceRenderer:
    """Interface for turning a plan + text into something renderable.

    The default implementation is a pass-through: it returns the same text, so
    text-only and voice paths share one underlying response. A real voice backend
    (TTS, a speech service) subclasses this and may *only* change how the text is
    rendered - never what Astra says, and never who she is.
    """

    name = "text"

    def available(self) -> bool:
        return True

    def render(self, text: str, plan: Optional[ResponsePlan] = None) -> str:
        """Render ``text`` for delivery. The default returns it unchanged."""
        return text


class NullVoiceRenderer(VoiceRenderer):
    """Explicitly no voice: text only. Used when voice is not configured."""

    name = "none"

    def available(self) -> bool:
        return False


class RecordingVoiceRenderer(VoiceRenderer):
    """A test/diagnostic renderer that records what it was asked to render."""

    name = "recording"

    def __init__(self) -> None:
        self.calls = []

    def render(self, text: str, plan: Optional[ResponsePlan] = None) -> str:
        self.calls.append({"text": text, "plan": plan.to_dict() if plan else None})
        return text
