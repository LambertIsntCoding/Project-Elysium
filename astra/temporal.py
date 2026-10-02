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
    }


__all__ = [
    "NOTABLE_GAP_SECONDS", "QUESTION_LONG_DAYS", "PROJECT_LONG_DAYS",
    "elapsed_span_phrase", "gap_line", "open_question_lines", "project_lines",
    "temporal_prompt_block", "diagnostics",
]
