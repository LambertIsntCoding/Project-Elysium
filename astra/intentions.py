"""Astra's own plans, as intentions rather than as a scheduler report.

The runtime already records what Astra has planned: the durable task queue
(:mod:`astra.tasks`), the bounded planner's evidence-backed candidates
(:mod:`astra.activity`), and the attempt ledger. This module is the *only* new
piece, and it is deliberately small and pure (no store, no model, no I/O): it
turns those existing records into a first-person account of her own intentions -
what she has committed to, what she selected for herself, what she is occupied
with, what she has postponed, what she is only considering, and what she has
finished or set aside - and, when the plan involves affect, frames it through the
*real* state via :mod:`astra.behavior`.

Two rules make it a view and not a second personality:

* **It reads, and only reads.** It never enqueues, starts, completes or
  otherwise mutates a task. Talking about a plan cannot change its state; any
  transition stays in the task/activity lifecycle. ``build_snapshot`` calls only
  read accessors (``all``/``scheduled``/``counts``/``candidates``), and rendering
  is pure.
* **It shares one source of truth.** Every line is derived from a real record -
  a live task, a planner candidate, a ledger outcome - so what Astra says and
  what the systems record cannot drift. A possibility is never presented as a
  decision, and an idle state is reported as idle rather than filled with an
  invented activity.

Recognition (``asks_intentions``) is the same shape as ``personal_state``: only a
turn that actually asks about her plans, intentions or current activity triggers
the block, so ordinary conversation is not turned into a status report. When the
user asks about the *system* behind it, the block also permits a plain
explanation without pretending the plan itself is a report.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional

from . import tasks as task_mod

# The natural, first-person phrase for each real task kind. Code-owned and
# bounded to the kinds the runtime actually supports, so nothing here can name
# an activity the implementation cannot perform.
_KIND_PHRASE: Dict[str, str] = {
    task_mod.KIND_MEMORY_MAINTENANCE: "tidying up my memory",
    task_mod.KIND_QUESTION_REVIEW: "revisiting some open questions",
    task_mod.KIND_WORK_REVIEW: "working back through something I've read",
    task_mod.KIND_READING: "reading",
    task_mod.KIND_FILE_INSPECTION: "looking over my files",
    task_mod.KIND_CALCULATION: "working something out",
    task_mod.KIND_CONTEXT_PREPARATION: "getting context ready for later",
    task_mod.KIND_APP_MAINTENANCE: "keeping things in order",
    task_mod.KIND_MEMORY_REVIEW: "checking my memory for contradictions",
    task_mod.KIND_EXPERIENCE_REVIEW: "turning recent experiences into lessons",
    task_mod.KIND_SELF_REVIEW: "looking over my own recent behaviour",
    task_mod.KIND_KNOWLEDGE_GAP: "looking into a gap in what I know",
    task_mod.KIND_TEST_RUN: "running my own tests",
    task_mod.KIND_SELF_IMPROVEMENT: "drafting ideas for improving myself",
}


def phrase_for(kind: Any) -> str:
    """A plain-language phrase for a task kind, honest about the unknown.

    An unrecognised kind is described generically rather than guessed at, so a
    new kind never produces an invented name.
    """
    key = str(kind or "")
    return _KIND_PHRASE.get(key) or "some background work"


# ---------------------------------------------------------------------
# Question recognition
# ---------------------------------------------------------------------
# Turns that ask about her plans, intentions, or current activity. Kept narrow:
# a false positive turns ordinary talk into a status report, so prefer missing
# one over over-triggering.
_QUESTION_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\bwhat (?:are|were) (?:you|she|astra) (?:planning|going to do|up to|doing)\b",
    r"\bwhat (?:do|did) (?:you|she|astra) plan\b",
    r"\bwhat(?:'s| is| are) (?:your|her|astra'?s) plans?\b",
    r"\b(?:do|have) (?:you|she|astra) (?:have|got|any) plans?\b",
    r"\b(?:have|got) (?:you|she|astra) (?:anything|something) planned\b",
    r"\b(?:you|she|astra) (?:have|got) (?:anything|something) planned\b",
    r"\bwhat (?:are|is) (?:you|she|astra) going to (?:do|work on|tackle)\b",
    r"\bwhat (?:do|does) (?:you|she|astra) intend\b",
    r"\bwhat (?:are|were) (?:you|she|astra) (?:working on|occupied with|busy with)\b",
    r"\bwhat(?:'s| is) next for (?:you|her|astra)\b",
    r"\bare (?:you|she|astra) (?:busy|working on anything|planning anything)\b",
    r"\bwhat (?:have|has) (?:you|she|astra) got (?:planned|lined up|on)\b",
    r"\bwhat(?:'s| is) (?:on|in) (?:your|her|astra'?s) (?:schedule|agenda|plate)\b",
    r"\bwhat (?:are|were) (?:you|she|astra) (?:intending|planning) to\b",
    r"\b(?:your|her|astra'?s) (?:plans|intentions|schedule|agenda)\b",
))


@lru_cache(maxsize=4096)
def _asks_cached(text: str) -> bool:
    return any(p.search(text) for p in _QUESTION_PATTERNS)


def asks_intentions(text: Any) -> bool:
    """True when the turn asks about Astra's own plans or current activity."""
    return _asks_cached(str(text or ""))


# ---------------------------------------------------------------------
# Building the snapshot (read-only over the existing records)
# ---------------------------------------------------------------------
def _reason_of(task: Any) -> str:
    progress = task.progress if isinstance(getattr(task, "progress", None), dict) else {}
    return str(progress.get("why") or progress.get("reason") or "").strip()


def _label(task: Any) -> str:
    payload = task.payload if isinstance(getattr(task, "payload", None), dict) else {}
    title = payload.get("work_id") or payload.get("title")
    phrase = phrase_for(task.kind)
    return f"{phrase} ('{title}')" if title else phrase


def build_snapshot(task_store: Any, *, planner: Any = None, ledger: Any = None,
                   paused: bool = False, notifications_paused: bool = False,
                   now: Any = None) -> Dict[str, Any]:
    """Derive Astra's intentions from the *existing* records, read-only.

    Categories mirror the distinction the brief requires:

    * ``commitments`` - scheduled work she is committed to (a schedule is a
      standing intention, not a user request);
    * ``selected`` - work she chose for herself (``origin == activity``);
    * ``active`` - the task she is actually occupied with right now;
    * ``waiting`` - postponed (deferred), blocked, or flagged for retry;
    * ``considering`` - planner candidates she has *evidence* for but has not
      committed to (a possibility, explicitly not a decision);
    * ``recent`` - what she finished or set aside, from the ledger.

    It calls only read accessors, so it can never start, finish or reorder work.
    """
    commitments: List[Dict[str, Any]] = []
    selected: List[Dict[str, Any]] = []
    active: List[Dict[str, Any]] = []
    waiting: List[Dict[str, Any]] = []
    retrying: List[Dict[str, Any]] = []

    try:
        tasks = list(task_store.all())
    except Exception:
        tasks = []

    for task in tasks:
        entry = {
            "kind": task.kind,
            "label": _label(task),
            "status": task.status,
            "due": task.due_at or task.next_due_at,
            "schedule": task.schedule or None,
            "reason": _reason_of(task),
            "attempts": task.attempts,
        }
        if task.status == task_mod.RUNNING:
            active.append(entry)
            continue
        if not task.is_live():
            continue  # terminal work is shown via the ledger's ``recent``
        if task.is_scheduled():
            commitments.append(entry)
        elif task.origin == task_mod.ORIGIN_ACTIVITY:
            selected.append(entry)
        else:
            selected.append(entry)
        if task.status in (task_mod.DEFERRED, task_mod.BLOCKED,
                           task_mod.AWAITING_HUMAN):
            waiting.append(entry)
        elif task.status == task_mod.FAILED:
            retrying.append(entry)

    considering: List[Dict[str, Any]] = []
    if planner is not None and callable(getattr(planner, "candidates", None)):
        try:
            live_keys = {(t.kind, t.dedupe_key) for t in tasks if t.is_live()}
            for cand in planner.candidates():
                if (cand.get("kind"), cand.get("dedupe_key")) in live_keys:
                    continue
                considering.append({
                    "kind": cand.get("kind"),
                    "label": phrase_for(cand.get("kind")),
                    "reason": str(cand.get("reason") or ""),
                    "lane": cand.get("lane"),
                })
                if len(considering) >= 4:
                    break
        except Exception:
            considering = []

    recent: List[Dict[str, Any]] = []
    if ledger is not None and callable(getattr(ledger, "entries", None)):
        try:
            for item in reversed(ledger.entries(limit=12)):
                status = str(item.get("status") or "")
                if status not in (task_mod.COMPLETED, task_mod.FAILED,
                                  task_mod.SKIPPED, task_mod.MISSED,
                                  task_mod.EXPIRED, task_mod.CANCELLED,
                                  task_mod.BLOCKED):
                    continue
                recent.append({
                    "kind": item.get("kind"),
                    "label": phrase_for(item.get("kind")),
                    "status": status,
                    "detail": str(item.get("detail") or ""),
                    "at": item.get("at"),
                })
                if len(recent) >= 5:
                    break
        except Exception:
            recent = []

    def _sort(rows: List[Dict[str, Any]]) -> None:
        rows.sort(key=lambda r: str(r.get("due") or ""))

    _sort(commitments)
    _sort(selected)
    return {
        "commitments": commitments,
        "selected": selected,
        "active": active,
        "waiting": waiting,
        "retrying": retrying,
        "considering": considering,
        "recent": recent,
        "paused": bool(paused),
        "notifications_paused": bool(notifications_paused),
        "now": str(now) if now else None,
    }


def is_idle(snapshot: Dict[str, Any]) -> bool:
    """True when nothing is planned, selected, active or waiting."""
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    return not any(snapshot.get(key) for key in
                   ("commitments", "selected", "active", "waiting", "retrying"))


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------
_STATUS_WORD = {
    task_mod.DEFERRED: "postponed",
    task_mod.BLOCKED: "blocked",
    task_mod.AWAITING_HUMAN: "waiting on Roum's word",
}


def _line(entry: Dict[str, Any]) -> str:
    text = str(entry.get("label") or "some background work")
    due = entry.get("due")
    status = entry.get("status")
    bits: List[str] = []
    if due:
        bits.append(f"due {due}")
    if status in _STATUS_WORD:
        bits.append(_STATUS_WORD[status])
    reason = str(entry.get("reason") or "").strip()
    if reason:
        bits.append(f"because {reason}")
    return text + (f" ({'; '.join(bits)})" if bits else "")


def _bias_lines(affect_state: Any, modulation: Any) -> List[str]:
    """How the *real* state should colour how she speaks about her plans.

    Derived from ``behavior.activity_bias`` - the same derivation the planner
    uses to decide *what* and *how much* she takes on - so the framing matches
    her actual willingness rather than a mood she performs. Phrased as conduct,
    never as a number, and omitted entirely when the state is neutral.
    """
    from . import behavior

    try:
        mod = modulation if isinstance(modulation, dict) else behavior.from_affect(affect_state)
    except Exception:
        return []
    bias = behavior.activity_bias(mod)
    lines: List[str] = []
    take_on = float(bias.get("take_on", 0.5))
    effort = float(bias.get("effort", 0.5))
    if take_on <= 0.3:
        lines.append(
            "Right now she is not eager to take on anything new; let that show "
            "in how she talks about what is ahead - reserved, not enthusiastic "
            "about starting more - without a speech about being tired.")
    elif take_on >= 0.7:
        lines.append(
            "Right now she is ready to take things on; let a little forward "
            "energy colour how she talks about her plans, without over-selling "
            "them.")
    if effort <= 0.35 and take_on > 0.3:
        lines.append(
            "She has little appetite for heavy or investigative work at the "
            "moment and leans toward lighter things; let that preference show "
            "in which of her plans she sounds drawn to.")
    if bias.get("prefer_light"):
        lines.append(
            "If she mentions what she is considering, let the lighter, more "
            "routine options come first.")
    return lines


def prompt_block(snapshot: Dict[str, Any], *, affect_state: Any = None,
                 modulation: Any = None) -> Optional[str]:
    """Render Astra's plans as her own intentions, or ``None`` when not asked.

    Rendered only for an intentions question. It presents her plans in ordinary
    first-person terms, keeps the categories distinct (a commitment is not a
    possibility, a running task is not a finished one), permits an explicit
    explanation when Roum asks about the system behind them, and states an empty
    plan as genuinely empty rather than filling it.
    """
    if not isinstance(snapshot, dict):
        return None

    lines: List[str] = [
        "\n=== ASTRA'S OWN PLANS (HER INTENTIONS - NOT A SYSTEM REPORT) ===",
        "These are Astra's actual plans, taken from what the runtime has really "
        "recorded. Talk about them as *her own* intentions, in the first person "
        "and in ordinary language - what she means to do, has chosen, is busy "
        "with, has put off, or is only mulling over. Do not describe the "
        "scheduler, the task queue, "
        "priorities, files, or 'the system' unless Roum explicitly asks how it "
        "works. Keep the categories straight: a scheduled commitment is not a "
        "possibility, something only under consideration is not yet a decision, "
        "and she must not claim to have started, finished, or be doing anything "
        "that is not actually running or done below.",
    ]

    if snapshot.get("paused"):
        lines.append(
            "- Her autonomous work is currently paused, so she is not taking on "
            "or advancing background activities right now.")
    if snapshot.get("notifications_paused"):
        lines.append(
            "- Proactive messages are paused, but that does not stop her work.")

    if snapshot.get("active"):
        lines.append("- Occupied with right now: "
                     + "; ".join(_line(e) for e in snapshot["active"]))
    if snapshot.get("commitments"):
        lines.append(
            "- Committed to (scheduled, recurring): "
            + "; ".join(_line(e) for e in snapshot["commitments"]))
    if snapshot.get("selected"):
        lines.append("- Chosen for herself: "
                     + "; ".join(_line(e) for e in snapshot["selected"]))
    if snapshot.get("retrying"):
        lines.append(
            "- Meant to retry (a previous attempt did not finish): "
            + "; ".join(_line(e) for e in snapshot["retrying"]))
    if snapshot.get("waiting"):
        lines.append(
            "- Postponed, blocked, or waiting: "
            + "; ".join(_line(e) for e in snapshot["waiting"]))
    if snapshot.get("considering"):
        lines.append(
            "- Only considering, not decided (do not present as a decision): "
            + "; ".join(str(e.get("label")) for e in snapshot["considering"]))
    if snapshot.get("recent"):
        bits = []
        for entry in snapshot["recent"]:
            if entry.get("status") == task_mod.COMPLETED:
                bits.append(f"{entry.get('label')} (done)")
            else:
                bits.append(f"{entry.get('label')} ({entry.get('status')})")
        lines.append("- Recently: " + "; ".join(bits))

    if is_idle(snapshot):
        lines.append(
            "- Nothing is planned or in progress at the moment. She should say "
            "so plainly, in her own words, and not invent something she is "
            "'about to do' to seem busy. Being idle is allowed.")

    lines.extend(_bias_lines(affect_state, modulation))

    lines.append(
        "She may revise her intentions as things change, and she may have plans "
        "Roum did not ask for - they are hers, not requests awaiting his "
        "approval. If she is unsure whether a plan will happen, she may say so. "
        "Explaining a plan does not begin it, and she must not imply she is "
        "already doing what she has only decided or is only considering.")
    return "\n".join(lines)


__all__ = [
    "asks_intentions", "phrase_for", "build_snapshot", "is_idle",
    "prompt_block",
]
