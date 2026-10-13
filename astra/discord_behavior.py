"""When Astra may speak, and how fast she types - pure, no I/O.

Two decisions the Discord adapter needs, kept here so they are deterministic,
testable without a transport, and not scattered through polling code:

* **Whether a message is addressed to her.** By default Astra does not answer
  every line in a channel; she is not permanently watching the chat. A message
  reaches her only when it @-mentions her or replies to one of her messages, or
  when she explicitly checks her messages. This module is the single rule for
  "was this addressed to her?".

* **How long she would take to type a reply.** A real person does not paste a
  wall of text instantly, and does not type every message at the same speed.
  :func:`typing_plan` gives each message its own tempo, nudges the total by its
  structure (long words, paragraph breaks, digits, hesitations), adds a small
  pause per sentence and a "last look-over" before sending, and splits the whole
  thing into bursts separated by pauses - so the timing never reads as a clean
  function of word count. :func:`typing_seconds` remains a simpler single-number
  estimate for callers that only want a total.

Both are pure functions of their inputs: no clock, no network, no randomness
that cannot be seeded. The adapter owns the actual waiting and sending.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# A comfortable, slightly-above-average typing speed (words per minute). The
# user asked for "a little above average" as the base.
DEFAULT_WPM = 45.0
# Per-message *tempo*: a person does not type at exactly their average speed.
# Each message picks a personal speed somewhere in this band before the words
# are paced out, so two replies of the same length can feel quite different.
# Deliberately wide (a distracted message and a focused one differ a lot).
TEMPO_WPM_LOW = 32.0
TEMPO_WPM_HIGH = 62.0
# Never answer instantly (a person has to read and start typing), and never
# make the user wait absurdly long for a long reply.
MIN_TYPING_SECONDS = 1.5
MAX_TYPING_SECONDS = 45.0
# The typing indicator must be re-sent within Discord's ~10s window to keep
# showing; refresh a little sooner.
TYPING_REFRESH_SECONDS = 8.0
# Legacy per-message speed jitter (kept for ``typing_seconds``).
WPM_JITTER_LOW = 0.85
WPM_JITTER_HIGH = 1.25
# A person re-reads or pauses after finishing a sentence, and thinks a beat
# before sending the whole thing. These are added on top of the typing time.
PUNCTUATION_PAUSE_SECONDS = 0.5     # per sentence-ending mark
BEFORE_SEND_SECONDS: Tuple[float, float] = (0.4, 2.4)  # the last look-over
# Reading a message is much faster than typing; used for the "stop and read"
# pause when a new message arrives mid-reply.
DEFAULT_READ_WPM = 240.0
MIN_READ_SECONDS = 0.4
MAX_READ_SECONDS = 6.0
# Base human pauses between typing bursts (before per-message variation).
DEFAULT_PAUSE_SECONDS: Tuple[float, float] = (0.6, 2.2)

# Structural noise: the things that make typing time *not* a clean function of
# word count - a long word costs more than a short one, a paragraph break is a
# beat, a moment of hesitation is a moment of hesitation. Each factor is
# multiplied into the delay, so the total is never just words / wpm.
_TYPING_NOISE = (
    ("long_words", 0.04, 0.30),      # per long word (>7 chars)
    ("digits", 0.02, 0.20),          # per numeric run
    ("nonascii", 0.05, 0.25),        # per non-latin word - visible effort
    ("newlines", 0.02, 0.15),        # per paragraph break
    ("hesitation", -0.03, 0.16),     # per hedge ("but", "well", "maybe"?)
)


def _words(text: Any) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def words_in(text: Any) -> int:
    """Whitespace-delimited word count (0 for empty/None)."""
    return _words(text)


_HESITATIONS = frozenset({
    "but", "well", "hm", "hmm", "maybe", "perhaps", "actually", "though",
    "anyway", "kinda", "sorta", "ugh", "honestly", "guess",
})


def _feature_counts(text: Any) -> Dict[str, int]:
    """Structural features of a message that make typing time non-linear.

    Deliberately cheap and count-based: length of *words*, paragraph breaks,
    numeric runs, non-latin words and hedge words. None of it is stored or
    reported; it only perturbs the delay so the same word count can feel
    different to type.
    """
    raw = str(text or "")
    words = re.findall(r"\S+", raw)
    long_words = sum(1 for w in words if len(re.sub(r"[^\w]", "", w)) > 7)
    digits = len(re.findall(r"\d+", raw))
    nonascii = sum(1 for w in words if any(ord(c) > 127 for c in w))
    newlines = len(re.findall(r"\n\s*\n", raw)) + raw.count("\n")
    hesitation = sum(1 for w in words
                     if re.sub(r"[^\w]", "", w).casefold() in _HESITATIONS)
    return {
        "long_words": long_words,
        "digits": digits,
        "nonascii": nonascii,
        "newlines": newlines,
        "hesitation": hesitation,
    }


def typing_plan(text: Any, *, wpm: float = DEFAULT_WPM,
                rng: Optional[random.Random] = None,
                min_seconds: float = MIN_TYPING_SECONDS,
                max_seconds: float = MAX_TYPING_SECONDS,
                pause_seconds: Sequence[float] = DEFAULT_PAUSE_SECONDS,
                max_segments: int = 8) -> Tuple[List[float], float]:
    """A per-message typing plan: ``(segment_sleeps, before_send_pause)``.

    Unlike a flat ``words / wpm``, this varies by *structure and tempo* so the
    timing does not read as a formula:

    * the message picks its own speed from a wide band (:data:`TEMPO_WPM_LOW`
      .. :data:`TEMPO_WPM_HIGH`), so two same-length replies can differ a lot;
    * each structural feature in :data:`_TYPING_NOISE` nudges the total;
    * sentence ends add a small re-read pause;
    * the typing is split into bursts with pauses (:func:`typing_segments`);
    * a short "last look-over" pause is reserved before the message is sent.

    The function stays pure and bounded: with a seeded ``rng`` the plan is
    reproducible, and the total never exceeds ``max_seconds`` (the before-send
    pause is carved out of that bound, not added past it).
    """
    r = rng or random
    words = _words(text)
    # Personal tempo for *this* message, independent of the base wpm.
    tempo = r.uniform(TEMPO_WPM_LOW, TEMPO_WPM_HIGH)
    base = words / max(1.0, tempo) * 60.0

    factor = 1.0
    counts = _feature_counts(text)
    for name, per_unit, cap in _TYPING_NOISE:
        n = counts.get(name, 0)
        if n > 0:
            factor *= 1.0 + min(cap, per_unit * n)
    base *= factor

    sentences = len(re.findall(r"[.!?\u2026]+", str(text or "")))
    base += sentences * PUNCTUATION_PAUSE_SECONDS

    ceiling = max(0.0, float(max_seconds))
    floor = max(0.0, float(min_seconds))
    total = min(ceiling, max(floor, base))
    # The final look-over pause comes out of the same budget, so the message
    # still arrives within ``max_seconds``.
    before_send = min(r.uniform(*BEFORE_SEND_SECONDS),
                      max(0.0, total - floor))
    typing_total = max(0.0, total - before_send)
    segments = typing_segments(typing_total, rng=r, pause_seconds=pause_seconds,
                               max_segments=max_segments)
    if segments:
        # Fold rounding drift into the last segment so the plan sums exactly to
        # the budget (keeps the max-seconds bound honest to the millisecond).
        drift = round(typing_total - sum(segments), 3)
        segments[-1] = round(max(0.0, segments[-1] + drift), 3)
    return segments, round(before_send, 3)


def is_addressed(content: Any, *, bot_user_id: Optional[str] = None,
                 bot_name: Optional[str] = None,
                 referenced_bot: bool = False) -> bool:
    """Whether a message is directed at Astra.

    Addressed when it replies to one of her messages, contains a Discord user
    mention of her, or (if a ``bot_name`` is configured) names her with an
    ``@name``. Deliberately narrow: a bare mention of a topic is not a summons.
    """
    if referenced_bot:
        return True
    text = str(content or "")
    if not text:
        return False
    uid = str(bot_user_id or "").strip()
    if uid and (f"<@{uid}>" in text or f"<@!{uid}>" in text):
        return True
    name = str(bot_name or "").strip()
    if name:
        # "@Astra" or "@astra", but not a longer word ("@Astraea").
        pattern = r"(?<![\w@])@" + re.escape(name) + r"(?![\w])"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def referenced_bot(message: Dict[str, Any], bot_user_id: Optional[str]) -> bool:
    """Whether a message is a reply to one of the bot's own messages."""
    uid = str(bot_user_id or "").strip()
    if not uid:
        return False
    ref = message.get("referenced_message")
    if isinstance(ref, dict):
        author = ref.get("author") or {}
        if str(author.get("id") or "") == uid:
            return True
    # Some payloads only carry the raw reference and a resolved author id.
    if str(message.get("referenced_author_id") or "") == uid:
        return True
    return False


def typing_seconds(text: Any, *, wpm: float = DEFAULT_WPM,
                   rng: Optional[random.Random] = None,
                   min_seconds: float = MIN_TYPING_SECONDS,
                   max_seconds: float = MAX_TYPING_SECONDS) -> float:
    """How long a person would plausibly take to type ``text``.

    Base delay is ``words / wpm`` minutes, scaled by a bounded random jitter so
    two identical replies are not typed at exactly the same speed, then clamped.
    Empty text still takes the minimum (she has to at least start).
    """
    words = _words(text)
    rate = max(1.0, float(wpm))
    base = words / rate * 60.0
    r = rng or random
    jitter = r.uniform(WPM_JITTER_LOW, WPM_JITTER_HIGH)
    seconds = base * jitter
    return round(max(float(min_seconds), min(float(max_seconds), seconds)), 3)


def typing_segments(seconds: float, *, rng: Optional[random.Random] = None,
                    pause_seconds: Sequence[float] = DEFAULT_PAUSE_SECONDS,
                    max_segments: int = 8) -> List[float]:
    """Split a total typing duration into human bursts separated by pauses.

    Returns the *sleep* durations to run in order - typing bursts and the small
    pauses between them - whose sum equals ``seconds``. A short reply is one
    burst; a long one is several, so she visibly stops and continues rather than
    emitting the whole thing at once.
    """
    total = max(0.0, float(seconds))
    if total <= 0.0 or max_segments <= 1:
        return [total] if total > 0 else []

    r = rng or random
    lo, hi = (pause_seconds[0], pause_seconds[1]) if len(pause_seconds) >= 2 \
        else DEFAULT_PAUSE_SECONDS
    # One pause per ~2.5s of typing, bounded, so short replies stay single-burst.
    n_pauses = min(max_segments - 1, max(0, int(total // 2.5)))
    if n_pauses <= 0:
        return [total]

    pauses = [round(r.uniform(lo, hi), 3) for _ in range(n_pauses)]
    if sum(pauses) >= total * 0.7:
        # Pauses cannot dominate; scale them back so typing still takes time.
        scale = (total * 0.7) / max(0.001, sum(pauses))
        pauses = [round(p * scale, 3) for p in pauses]
    typing_total = total - sum(pauses)
    burst = typing_total / (n_pauses + 1)
    segments: List[float] = []
    for i in range(n_pauses + 1):
        segments.append(round(burst, 3))
        if i < n_pauses:
            segments.append(pauses[i])
    return segments


def reading_seconds(text: Any, *, wpm: float = DEFAULT_READ_WPM,
                    min_seconds: float = MIN_READ_SECONDS,
                    max_seconds: float = MAX_READ_SECONDS) -> float:
    """How long to pause to actually read newly arrived text before reacting."""
    words = _words(text)
    rate = max(1.0, float(wpm))
    seconds = words / rate * 60.0
    return round(max(float(min_seconds), min(float(max_seconds), seconds)), 3)


__all__ = [
    "DEFAULT_WPM", "TEMPO_WPM_LOW", "TEMPO_WPM_HIGH", "MIN_TYPING_SECONDS",
    "MAX_TYPING_SECONDS", "TYPING_REFRESH_SECONDS", "DEFAULT_READ_WPM",
    "words_in", "is_addressed", "referenced_bot", "typing_seconds",
    "typing_segments", "typing_plan", "reading_seconds",
]
