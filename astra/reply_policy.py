"""When a reply should visually reference a specific Discord message.

Discord's reply feature attaches a quote of one earlier message. It is useful
exactly when Astra is answering *one specific message's point* rather than
continuing a thread, and it is noise otherwise - so the default is **not** to
reply-attach, and this module is the single, code-owned place that decides.

The decision is deterministic and needs no model cooperation beyond an explicit
opt-in marker the response may carry (``EMPHASIS_MARKER``, a leading ``"\\u21aa "``).
This mirrors how :mod:`astra.generation` owns the always-on response contract:
the rule lives in code, not scattered across the personality prompt.

When a reply *is* warranted the adapter also pings the message's author (the
user's own "reply + ping" convention), and when it is not, no mention is sent -
so Astra never pings by accident.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

# A response may opt into a reply reference by beginning with this marker. It is
# stripped from the delivered text. Chosen to be unambiguous and unlikely to
# appear in ordinary prose.
EMPHASIS_MARKER = "\u21aa "


def parse(text: Any) -> Tuple[str, bool]:
    """Return ``(clean_text, wants_reply)``.

    Pure. The marker is recognised only at the very start (after trimming
    leading whitespace) so a mid-sentence glyph is not misread.
    """
    raw = "" if text is None else str(text)
    stripped = raw.lstrip()
    if stripped.startswith(EMPHASIS_MARKER):
        return stripped[len(EMPHASIS_MARKER):].lstrip(), True
    return raw, False


def should_reply(text: Any, *, incoming_count: int = 1) -> bool:
    """Whether this response should reply-attach (and ping).

    The explicit marker is authoritative. A batch of a single message never
    forces a reply; a multi-message batch is exactly where a quote helps, so the
    marker is honoured there with no extra condition.
    """
    _, wants = parse(text)
    return wants
