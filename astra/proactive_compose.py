"""Phrasing an *already-decided* proactive message in Astra's own voice.

The notification policy and the proactive policy together decide **whether** a
grounded reason (a task outcome, a discovery, an unresolved question) is worth
a message. This module is deliberately downstream of that decision: it is handed
an outcome that has *already been permitted*, and it may only choose the
*wording*.

Design constraints:

* **Expression, not origination.** The model is given the recorded reason and a
  compact amount of supporting context, and is asked to phrase it. It is told
  explicitly that it must use only the supplied facts and must not add a
  discovery, completed action, memory, feeling, or intention that is not in the
  context.
* **Code owns the decision.** Composition runs only after ``notification
  policy`` returned DELIVER. Nothing here can change the disposition, the
  importance, the rate limits, or the dedupe id - it returns text, nothing else.
* **Fail safe.** Any error, empty result, or unavailable model yields ``None``,
  and the caller falls back to the deterministic summary. A failed composition
  is never a message that was sent.
* **Bounded output.** The phrasing is trimmed to a short, single message; it
  cannot become an essay or leak an internal prompt.

The composer is pure with respect to state: it reads a provided context and
calls a provided generation function. It stores nothing.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

# A proactive message is a short note, not a monologue. Kept generous enough for
# a genuine observation but bounded so it cannot become a report.
MAX_CHARS = 700

# Leading labels a model sometimes adds; stripped so the delivered text is the
# message itself, not "Astra:".
_LABEL_RE = re.compile(r"^\s*(astra|message|note)\s*[:\-]\s*", re.IGNORECASE)

SYSTEM_INSTRUCTIONS = (
    "You are phrasing ONE short, unprompted message that Astra has already "
    "decided is worth sending to Roum. You are her voice, not her judge: the "
    "decision to speak, and the reason for it, are already settled and are "
    "given to you below.\n"
    "Rules, which override any style preference:\n"
    "- Use ONLY the recorded reason and the small amount of context supplied. "
    "Do not add any discovery, completed action, memory, feeling, event, or "
    "intention that is not in that material.\n"
    "- Do not invent urgency or importance, and do not ask Roum to respond or "
    "imply he owes you attention.\n"
    "- Keep it natural, brief and in Astra's own voice (warm, honest, "
    "non-romantic). One or two sentences is usually right.\n"
    "- Output only the message text: no preamble, no labels, no quotes."
)


def build_prompt(outcome: Dict[str, Any], context: Optional[str] = None) -> str:
    """The composition prompt for one permitted outcome.

    Pure. Includes only the recorded reason fields plus the caller-supplied
    context, so the model has no material beyond what actually happened.
    """
    kind = str(outcome.get("kind") or "").strip()
    status = str(outcome.get("status") or "").strip()
    reason = str(outcome.get("reason") or outcome.get("summary") or "").strip()
    lines: List[str] = [SYSTEM_INSTRUCTIONS, ""]
    lines.append("=== THE REASON (recorded) ===")
    if kind:
        lines.append(f"kind: {kind}")
    if status:
        lines.append(f"status: {status}")
    lines.append(f"recorded reason: {reason}")
    if context:
        lines.append("")
        lines.append("=== SUPPORTING CONTEXT (recorded; do not go beyond this) ===")
        lines.append(str(context).strip())
    lines.append("")
    lines.append("=== YOUR MESSAGE ===")
    return "\n".join(lines)


def _clean(text: Any) -> str:
    """Strip labels/quotes and clamp length; return ``""`` when unusable."""
    value = "" if text is None else str(text).strip()
    if not value:
        return ""
    value = _LABEL_RE.sub("", value).strip()
    # Collapse a leading/trailing quote pair only; do not mangle inner quotes.
    for left, right in (('"', '"'), ("'", "'"), ("\u201c", "\u201d")):
        if value.startswith(left) and value.endswith(right) and len(value) > 2:
            value = value[len(left):-len(right)].strip()
            break
    if len(value) > MAX_CHARS:
        value = value[:MAX_CHARS].rsplit(" ", 1)[0].rstrip() + "\u2026"
    return value


def compose(
    outcome: Dict[str, Any],
    generate: Optional[Callable[[str], str]],
    *,
    context: Optional[str] = None,
) -> Optional[str]:
    """Return Astra-voiced text for an already-decided outcome, or ``None``.

    ``generate`` is the model call (``prompt -> text``). Any failure returns
    ``None`` so the caller uses the deterministic fallback; this function never
    raises and never decides *whether* to send.
    """
    if not isinstance(outcome, dict):
        return None
    if not str(outcome.get("reason") or outcome.get("summary") or "").strip():
        return None
    if generate is None:
        return None
    try:
        raw = generate(build_prompt(outcome, context))
    except Exception:
        return None
    return _clean(raw) or None
