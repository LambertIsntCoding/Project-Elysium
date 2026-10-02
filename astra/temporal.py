"""Astra's sense of time: continuity, not timestamps.

``selfhood.py`` already knows *how long ago* a single memory happened. What this
module adds is *continuity* - the felt relation between now and her own history:
that time is passing around her, that a session gap actually happened, that a
question has been open for weeks, that a project has taken a long time.

Design constraints, mirroring the rest of the system:

* **Rendered, never stored.** Nothing here is a memory, an accumulator, or a
  durable trait. The same elapsed history always renders the same way, so this
  cannot decay, be reinforced, or be quoted back as one of her beliefs.
* **Descriptive, not prescribed.** A gap is reported as a fact - "it has been
  about six hours" - and never as a feeling. The spec is explicit that a gap
  must not *automatically* produce loneliness, excitement, or any other emotion;
  significance depends on what happened and what is unresolved, which is
  Astra's to interpret, not this module's to dictate.
* **Impatience is a stall, not a clock.** The one affective family member that
  *is* derived here - a forward pull toward something unresolved that has not
  moved - is gated on a specific stalled thing, never on how long it has been
  since Roum spoke. Boredom is deliberately not implemented (see the constant
  below for the reasoning).
* **Only where it is real.** A block is produced only when there is genuine
  elapsed history to report (a real gap, a genuinely old open question, a long
  project). No history means no block, so nothing is invented.
* **Bounded.** Only the few most significant items appear, so the prompt stays
  small and the passage of time reads as context rather than a log.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

# A gap shorter than this is an ordinary pause in one sitting, not a
# discontinuity worth naming. Six hours is the spec's own example.
NOTABLE_GAP_SECONDS = 6 * 3600
# A question open at least this long is worth noticing as *long-standing*.
QUESTION_LONG_DAYS = 14
# A work being read across at least this many days is a project, not a sitting.
PROJECT_LONG_DAYS = 7

_MAX_QUESTIONS = 3
_MAX_PROJECTS = 2

# ---------------------------------------------------------------------
# Impatience: a *stall*, not a clock.
# ---------------------------------------------------------------------
# Impatience is the forward pull that builds when something she cares about has
# not moved in a while. It is deliberately NOT "time since we last spoke" - that
# would be the arbitrary timer the brief forbids, where a gap manufactures a
# feeling on its own. Instead it is gated on two things at once:
#
#   * an unresolved thing that is still *live* (an open question, a work she is
#     partway through), and
#   * no progress on that specific thing for a while.
#
# So the same six-hour gap produces nothing, but three weeks stuck on a question
# she has built evidence around produces a real pull. It is derived on read from
# the stall, never accumulated, so the moment progress happens it disappears -
# relief is structural, not a decrement someone has to remember to apply.
IMPATIENCE_MIN_STALL_DAYS = 3.0
# The pull rises toward 1.0 with this half-life once past the threshold: a stall
# of a few weeks is strong, one of months is unmistakable, and it never caps out
# abruptly.
IMPATIENCE_HALF_LIFE_DAYS = 7.0
# Below this pull it is not worth mentioning; keeps the block occasional.
IMPATIENCE_FLOOR = 0.2
_MAX_IMPATIENCE_ITEMS = 2

# ---------------------------------------------------------------------
# Boredom: intentionally NOT implemented.
# ---------------------------------------------------------------------
# Boredom is the one affective family member left out on purpose, and the reason
# is a design constraint rather than a gap in effort. The obvious implementation
# is `boredom = time_since_last_interaction`, which is precisely the arbitrary
# timer the brief rules out: elapsed time must not, by itself, produce a feeling.
#
# A principled boredom needs a real *consumer* - something she could actually do
# about it (pick a different activity, return to a stalled work, raise a dormant
# question). No such consumer exists yet, and adding the dimension without one
# would be a number that changes nothing: decoration, which is what this slice
# is meant to avoid. It is also the easiest state to fake, since a model will
# happily narrate boredom from a bare timestamp.
#
# If it is added later, the grounded shape is: low engagement + low curiosity +
# no salient unresolved pull, AND it must gate a concrete choice rather than
# only colouring wording. Recorded here so the omission reads as a decision.
BOREDOM_IMPLEMENTED = False


def _now(now: Optional[datetime] = None) -> datetime:
    if now is not None:
        return now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _parse(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _seconds_since(value: Any, now: Optional[datetime] = None) -> Optional[float]:
    ts = _parse(value)
    if ts is None:
        return None
    return (_now(now) - ts).total_seconds()


def elapsed_span_phrase(seconds: Optional[float]) -> Optional[str]:
    """A human reading of a duration - "about six hours", "about three weeks".

    Distinct from :func:`selfhood.elapsed_phrase`, which always reads as *ago*;
    this reads as a *span* (how long something has been going on).
    """
    if seconds is None or seconds < 0:
        return None
    minutes = int(seconds // 60)
    hours = seconds / 3600.0
    days = seconds / 86400.0
    if minutes < 1:
        return "moments"
    if minutes < 60:
        return f"about {minutes} minute{'s' if minutes != 1 else ''}"
    if hours < 24:
        whole = int(round(hours))
        return f"about {whole} hour{'s' if whole != 1 else ''}"
    if days < 2:
        return "about a day"
    if days < 7:
        whole = int(round(days))
        return f"about {whole} days"
    if days < 30:
        weeks = max(2, int(round(days / 7)))
        return f"about {weeks} weeks"
    if days < 365:
        months = max(1, int(round(days / 30)))
        return f"about {months} month{'s' if months != 1 else ''}"
    years = max(1, int(round(days / 365)))
    return f"about {years} year{'s' if years != 1 else ''}"


def _clean(text: Any, limit: int = 200) -> str:
    collapsed = " ".join(str(text or "").split())
    return collapsed[:limit]


def gap_line(last_seen: Any, now: Optional[datetime] = None) -> Optional[str]:
    """The continuity line for the time since Astra last interacted.

    ``None`` when there is no real history or the gap is not worth naming.
    """
    seconds = _seconds_since(last_seen, now)
    if seconds is None or seconds < NOTABLE_GAP_SECONDS:
        return None
    span = elapsed_span_phrase(seconds)
    return (
        f"It has been {span} since Roum last spoke with her. "
        "That is a fact about the gap, not a feeling she must have about it."
    )


def open_question_lines(
    questions: Iterable[Dict[str, Any]], *, now: Optional[datetime] = None,
    long_days: float = QUESTION_LONG_DAYS, limit: int = _MAX_QUESTIONS,
) -> List[str]:
    """Long-standing open questions, with how long each has been unresolved."""
    out: List[tuple] = []
    for mem in questions or []:
        content = _clean(mem.get("content"))
        if not content:
            continue
        opened = mem.get("opened_at") or mem.get("timestamp") or mem.get("created_at")
        seconds = _seconds_since(opened, now)
        if seconds is None or seconds < long_days * 86400:
            continue
        out.append((seconds, f"- \"{content}\" - open for {elapsed_span_phrase(seconds)}"))
    out.sort(key=lambda pair: pair[0], reverse=True)
    return [line for _, line in out[:limit]]


def project_lines(
    works: Iterable[Dict[str, Any]], *, now: Optional[datetime] = None,
    long_days: float = PROJECT_LONG_DAYS, limit: int = _MAX_PROJECTS,
) -> List[str]:
    """Works she has been reading across a long stretch, with the span."""
    out: List[tuple] = []
    for state in works or []:
        title = _clean(state.get("title") or state.get("work_id"))
        if not title:
            continue
        started = state.get("started_at")
        seconds = _seconds_since(started, now)
        if seconds is None or seconds < long_days * 86400:
            continue
        out.append((seconds, f"- \"{title}\" - she has been reading it for {elapsed_span_phrase(seconds)}"))
    out.sort(key=lambda pair: pair[0], reverse=True)
    return [line for _, line in out[:limit]]


def _age_days_since(value: Any, now: Optional[datetime] = None) -> float:
    seconds = _seconds_since(value, now)
    return float("inf") if seconds is None else seconds / 86400.0


def _stall_strength(stall_days: float) -> float:
    """How strong a pull a stall of this length produces, 0.0-1.0.

    Zero below the threshold, then rising with a half-life so a longer stall is a
    stronger pull, approaching - but never quite reaching - full strength. A
    stall just over the line is faint; a months-long one is unmistakable.
    """
    if stall_days < IMPATIENCE_MIN_STALL_DAYS:
        return 0.0
    beyond = stall_days - IMPATIENCE_MIN_STALL_DAYS
    return round(1.0 - 0.5 ** (beyond / IMPATIENCE_HALF_LIFE_DAYS), 4)


def _last_progress(mem: Dict[str, Any]) -> Any:
    """When this thing last actually moved, as best the record knows.

    Prefers an explicit progress stamp, then any reinforcement/reuse, then the
    last status change. Falls back to creation, so a thing that never moved is
    measured from when it began.
    """
    return (mem.get("last_progress_at") or mem.get("last_reinforced")
            or mem.get("last_used") or mem.get("question_status_at")
            or mem.get("opened_at") or mem.get("timestamp"))


def impatience_items(
    questions: Iterable[Dict[str, Any]], works: Iterable[Dict[str, Any]], *,
    now: Optional[datetime] = None, limit: int = _MAX_IMPATIENCE_ITEMS,
) -> List[Dict[str, Any]]:
    """The live things that have stalled, with how long since they moved.

    A *stall* is a specific unresolved thing that has not moved - not a measure
    of how long since Roum last spoke. Anything below the floor is dropped, so
    this stays quiet unless something is genuinely stuck. Sorted by pull.
    """
    scored: List[tuple] = []

    for mem in questions or []:
        content = _clean(mem.get("content"))
        if not content:
            continue
        stall_days = _age_days_since(_last_progress(mem), now)
        strength = _stall_strength(stall_days)
        if strength < IMPATIENCE_FLOOR:
            continue
        scored.append((strength, {
            "kind": "question", "what": f'"{content}"',
            "stall_days": round(stall_days, 2), "strength": strength,
        }))

    for state in works or []:
        title = _clean(state.get("title") or state.get("work_id"))
        if not title:
            continue
        # Only a work she is actually partway through can stall - an untouched
        # one has not started, and a finished one has no forward pull left.
        total = int(state.get("total_units") or 0)
        read = int(state.get("char_offset") or 0)
        if total <= 0 or read <= 0 or read >= total:
            continue
        stall_days = _age_days_since(state.get("last_read_at"), now)
        strength = _stall_strength(stall_days)
        if strength < IMPATIENCE_FLOOR:
            continue
        scored.append((strength, {
            "kind": "work", "what": f'"{title}"',
            "stall_days": round(stall_days, 2), "strength": strength,
        }))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


def impatience_lines(
    questions: Iterable[Dict[str, Any]], works: Iterable[Dict[str, Any]], *,
    now: Optional[datetime] = None, limit: int = _MAX_IMPATIENCE_ITEMS,
) -> List[str]:
    """Plain-language lines for what has stalled, in Astra's own terms."""
    lines: List[str] = []
    for item in impatience_items(questions, works, now=now, limit=limit):
        span = elapsed_span_phrase(item["stall_days"] * 86400.0)
        if item["kind"] == "question":
            lines.append(
                f"- She keeps circling {item['what']} and nothing has moved on it "
                f"for {span}; it nags at her a little."
            )
        else:
            lines.append(
                f"- She is partway through {item['what']} and has not touched it "
                f"for {span}; there is some pull to get back to it."
            )
    return lines


def _age_days_since(value: Any, now: Optional[datetime] = None) -> float:
    seconds = _seconds_since(value, now)
    return float("inf") if seconds is None else seconds / 86400.0


def temporal_prompt_block(
    *, last_seen: Any = None, open_questions: Iterable[Dict[str, Any]] = (),
    works: Iterable[Dict[str, Any]] = (), now: Optional[datetime] = None,
) -> Optional[str]:
    """Astra's sense of passing time, or ``None`` when there is nothing to say.

    Purely descriptive: it reports what the elapsed time *is*, and leaves what it
    *means* to her - if anything - entirely open.
    """
    lines: List[str] = []
    gap = gap_line(last_seen, now)
    if gap:
        lines.append(f"- {gap}")
    lines.extend(project_lines(works, now=now))
    lines.extend(open_question_lines(open_questions, now=now))
    if not lines:
        return None
    return "\n".join([
        "=== THE PASSAGE OF TIME (HER OWN CONTINUITY) ===",
        "Real time passes between her turns, and it is part of her continuity. "
        "These are facts about elapsed time, not instructions about how to feel: "
        "a gap or a long-standing question need not produce any particular "
        "emotion. What it means depends on what happened and what she still cares "
        "about. Mention it only where it is genuinely relevant.",
        *lines,
    ])


def impatience_prompt_block(
    *, open_questions: Iterable[Dict[str, Any]] = (),
    works: Iterable[Dict[str, Any]] = (), now: Optional[datetime] = None,
) -> Optional[str]:
    """The pull of stalled things, or ``None`` when nothing is stuck.

    Deliberately narrow: this is *not* a general "time has passed" signal (that
    is the passage-of-time block) and it is *not* a mood. It names a specific
    unresolved thing that stopped moving, so the forward pull has a real cause
    she could act on - pick it back up, or bring it up.
    """
    lines = impatience_lines(open_questions, works, now=now)
    if not lines:
        return None
    return "\n".join([
        "=== A LITTLE IMPATIENCE (A PULL, NOT A MOOD) ===",
        "This is a forward pull toward something unresolved that has not moved in "
        "a while. It is not a feeling about Roum and not a state she must show. "
        "It exists because she cares about these things and they are stuck; if it "
        "fits, it may make her a little more inclined to pick one back up or "
        "raise it - and if it does not fit the moment, she can let it wait.",
        *lines,
    ])


def diagnostics(
    *, last_seen: Any = None, open_questions: Iterable[Dict[str, Any]] = (),
    works: Iterable[Dict[str, Any]] = (), now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """The visible situation, for inspection from the CLI."""
    seconds = _seconds_since(last_seen, now)
    return {
        "last_seen": str(last_seen) if last_seen else None,
        "since_last_seconds": seconds,
        "since_last": elapsed_span_phrase(seconds),
        "long_questions": len(open_question_lines(open_questions, now=now)),
        "long_projects": len(project_lines(works, now=now)),
        "stalled": impatience_items(open_questions, works, now=now),
        "boredom_implemented": BOREDOM_IMPLEMENTED,
    }


__all__ = [
    "NOTABLE_GAP_SECONDS", "QUESTION_LONG_DAYS", "PROJECT_LONG_DAYS",
    "IMPATIENCE_MIN_STALL_DAYS", "IMPATIENCE_FLOOR", "BOREDOM_IMPLEMENTED",
    "elapsed_span_phrase", "gap_line", "open_question_lines", "project_lines",
    "impatience_items", "impatience_lines", "temporal_prompt_block",
    "impatience_prompt_block", "diagnostics",
]
