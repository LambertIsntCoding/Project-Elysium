"""Proactive communication policy: whether an existing candidate may surface.

Astra may hold reasons to speak (the inquiry system's conversation candidates).
This module is the *separate decision* the spec requires about them, and it
keeps four questions distinct:

1. should the candidate be **retained**?
2. should it be **surfaced** when the user next interacts?
3. is it important enough to **interrupt** the user?
4. does the current environment **permit** an interruption?

Everything here is a pure decision over explicit inputs plus a little persisted
bookkeeping (cooldowns, recently-surfaced ids). It never generates an emotion:
in particular, there is **no loneliness or boredom timer**. Elapsed time may
raise the strength of an *existing unresolved condition*, but time alone never
manufactures a reason to speak. If there is no concrete candidate, nothing is
produced - the system does not generate activity to justify itself.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .memory import atomic_save, load_json

POLICY_VERSION = 1

# Dispositions for a candidate after the decision.
DISPOSITION_SURFACE = "surface"       # say it when the user next interacts
DISPOSITION_INTERRUPT = "interrupt"   # important enough to speak unprompted
DISPOSITION_RETAIN = "retain"         # keep for later; do not speak now
DISPOSITION_DROP = "drop"             # not worth keeping

# Suppression reasons (why an interruption is not permitted right now).
SUPPRESS_USER_ACTIVE = "user_active"
SUPPRESS_MUTED = "muted"
SUPPRESS_GAME = "game_active"
SUPPRESS_PROHIBITED = "prohibited"
SUPPRESS_BUSY = "busy"
SUPPRESS_COOLDOWN = "cooldown"

# A candidate must clear this derived strength to be worth interrupting for.
INTERRUPT_THRESHOLD = 0.72
SURFACE_THRESHOLD = 0.40

# Cooldowns (seconds). Time is only ever used to *hold back*, never to prompt.
DEFAULT_INTERRUPT_COOLDOWN = 20 * 60
DEFAULT_MUTED_COOLDOWN = 0.0


def _now(now: Optional[datetime] = None) -> datetime:
    return now or datetime.now(timezone.utc)


def _parse(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


class ProactivePolicy:
    """Decides retain / surface / interrupt / permitted, and enforces cooldowns."""

    def __init__(self, data_dir: str = "./storage", *,
                 interrupt_cooldown: float = DEFAULT_INTERRUPT_COOLDOWN,
                 muted: bool = False) -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "proactive_state.json")
        self.interrupt_cooldown = float(interrupt_cooldown)
        self._lock = threading.RLock()
        self._state = self._load()
        self._muted = bool(muted)

    # -- persisted bookkeeping ------------------------------------------
    def _load(self) -> Dict[str, Any]:
        raw = load_json(self.path, dict, dict)
        return {
            "version": POLICY_VERSION,
            "last_interrupt_at": raw.get("last_interrupt_at") if isinstance(raw, dict) else None,
            "surfaced_ids": list(raw.get("surfaced_ids") or []) if isinstance(raw, dict) else [],
        }

    def reload(self) -> None:
        """Re-read the persisted state, for a separate inspection process."""
        with self._lock:
            self._state = self._load()

    def _persist(self) -> None:
        with self._lock:
            atomic_save(self.path, self._state)

    # -- mute ------------------------------------------------------------
    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)

    # -- cooldown --------------------------------------------------------
    def in_cooldown(self, now: Optional[datetime] = None) -> bool:
        last = _parse(self._state.get("last_interrupt_at"))
        if last is None:
            return False
        elapsed = (_now(now) - last).total_seconds()
        return elapsed < self.interrupt_cooldown

    def note_interrupt(self, now: Optional[datetime] = None) -> None:
        with self._lock:
            self._state["last_interrupt_at"] = _now(now).isoformat()
            self._persist()

    def note_surfaced(self, ref_id: Any) -> None:
        if not ref_id:
            return
        with self._lock:
            ids = list(self._state.get("surfaced_ids") or [])
            ids.append(str(ref_id))
            self._state["surfaced_ids"] = ids[-50:]
            self._persist()

    def was_surfaced(self, ref_id: Any) -> bool:
        return str(ref_id) in (self._state.get("surfaced_ids") or [])

    # -- suppression -----------------------------------------------------
    def suppression(self, *, user_active: bool = False, game: Optional[str] = None,
                    busy: Optional[str] = None, prohibited: bool = False,
                    now: Optional[datetime] = None) -> Optional[str]:
        """The first condition suppressing an interruption, or ``None``.

        Order is deliberate: the loudest, most concrete reasons come first, and
        "Astra is being interacted with" beats everything.
        """
        if user_active:
            return SUPPRESS_USER_ACTIVE
        if prohibited:
            return SUPPRESS_PROHIBITED
        if self._muted:
            return SUPPRESS_MUTED
        if game:
            return SUPPRESS_GAME
        if busy:
            return SUPPRESS_BUSY
        if self.in_cooldown(now):
            return SUPPRESS_COOLDOWN
        return None

    # -- the decision ----------------------------------------------------
    def evaluate(self, candidate: Dict[str, Any], *, strength: float,
                 user_active: bool = False, game: Optional[str] = None,
                 busy: Optional[str] = None, prohibited: bool = False,
                 high_priority_event: bool = False,
                 allow_high_priority_bypass: bool = False,
                 now: Optional[datetime] = None) -> Dict[str, Any]:
        """Decide what to do with one candidate.

        ``high_priority_event`` with ``allow_high_priority_bypass`` is the only
        way past ordinary conversational suppression, and only when
        configuration explicitly permits it. It never bypasses the user's own
        standing decisions (user is active, an explicit mute, or a prohibition).
        """
        ref_id = candidate.get("question_id") or candidate.get("ref_id") or candidate.get("reason")
        strength = max(0.0, min(1.0, float(strength)))

        # Retention is independent of whether we can speak now.
        retain = strength >= SURFACE_THRESHOLD

        suppress = self.suppression(user_active=user_active, game=game, busy=busy,
                                    prohibited=prohibited, now=now)
        # A high-priority event may bypass ordinary conversational courtesies
        # (game, load, cooldown) only when configuration opts in. It may never
        # override a standing human decision - being present, an explicit mute,
        # or a prohibition - because those are the user's authority, not the
        # moment's.
        bypass = bool(high_priority_event and allow_high_priority_bypass
                      and suppress not in (SUPPRESS_USER_ACTIVE, SUPPRESS_MUTED,
                                           SUPPRESS_PROHIBITED))

        # Surface-when-next-spoken-to: allowed whenever the candidate is strong
        # enough; this is not an interruption, so suppression does not apply.
        if strength >= SURFACE_THRESHOLD:
            disposition = DISPOSITION_SURFACE
        elif retain:
            disposition = DISPOSITION_RETAIN
        else:
            disposition = DISPOSITION_DROP

        # Interruption: strong enough, and the environment permits it.
        interrupt = False
        if strength >= INTERRUPT_THRESHOLD and (suppress is None or bypass):
            disposition = DISPOSITION_INTERRUPT
            interrupt = True

        if suppress is not None and not interrupt and disposition == DISPOSITION_SURFACE:
            # We would surface it, but the moment is suppressed: keep it for
            # later rather than discarding it.
            disposition = DISPOSITION_RETAIN

        return {
            "ref_id": str(ref_id) if ref_id else None,
            "kind": candidate.get("kind"),
            "strength": round(strength, 4),
            "retain": retain,
            "surface": disposition in (DISPOSITION_SURFACE, DISPOSITION_INTERRUPT),
            "interrupt": interrupt,
            "permitted": suppress is None or bypass,
            "suppressed_by": suppress,
            "bypassed": bool(bypass and suppress is not None),
            "disposition": disposition,
        }

    def filter_candidates(self, decisions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only candidates that should be surfaced when next interacted."""
        return [d for d in decisions if d.get("surface") and not d.get("interrupt")]


def candidate_strength(candidate: Dict[str, Any], *,
                       unresolved_days: Optional[float] = None,
                       relevance: float = 0.0) -> float:
    """Derive a candidate's strength from its own weight plus a stall factor.

    Elapsed time can *raise* the strength of an already-unresolved thing, but
    with no candidate there is nothing to raise - this function cannot invent a
    reason to speak. ``relevance`` is a 0..1 contextual boost.
    """
    base = max(0.0, min(1.0, float(candidate.get("weight", 0.0) or 0.0)))
    stall = 0.0
    if unresolved_days is not None:
        # Saturating curve: a few weeks is meaningful, months no more so.
        stall = min(0.25, max(0.0, float(unresolved_days)) / 60.0 * 0.25)
    return max(0.0, min(1.0, base + stall + 0.15 * max(0.0, min(1.0, relevance))))


# -- candidate -> notification-outcome mapping (the missing wire) -----------
#
# A conversation candidate is not a task outcome, so its fields are mapped
# explicitly rather than passed through as if they were the same thing:
#
# * ``reason``/``summary`` - the candidate's own recorded reason (the *retained*
#   basis). Both keys are populated so either notification path reads it, but
#   the reported message is always the recorded reason, never an invented one.
# * ``ref_id`` - derived from the candidate's *persistent* identity (its
#   question id; else its kind + referenced work + a content hash). The index is
#   never used, because the same candidate must dedupe across reorderings.
# * ``importance`` - the candidate's *weight* is a 0..1 *relevance* (how much a
#   question deserves attention), which is not the same as the importance of a
#   completed or failed task. Treating it as equivalent would make every ordinary
#   open question an interruption-worthy event. It is therefore *discounted* by
#   ``_CANDIDATE_IMPORTANCE_SCALE`` into the notification-importance scale, so a
#   median question maps below ``min_importance`` (digest only) and only a
#   genuinely strong reason - or a borderline one lifted by her own anticipation
#   (Change 2) - clears the delivery threshold.

# A preserved reason is less urgent than a task outcome, so its relevance is
# discounted before it meets the notification thresholds. Calibrated so a median
# open question (weight ~0.55) lands below the default ``min_importance`` (0.55)
# while a strong "wants Roum's view" question (~0.70) clears it.
_CANDIDATE_IMPORTANCE_SCALE = 0.8


def candidate_ref(candidate: Dict[str, Any]) -> Optional[str]:
    """A stable, content-derived identity for a conversation candidate."""
    if not isinstance(candidate, dict):
        return None
    qid = candidate.get("question_id")
    if qid:
        return f"q:{qid}"
    kind = str(candidate.get("kind") or "cand").strip()
    work = str(candidate.get("work_id") or "").strip()
    reason = str(candidate.get("reason") or "").strip()
    if reason:
        import hashlib
        digest = hashlib.sha1(reason.encode("utf-8", "ignore")).hexdigest()[:12]
        return f"{kind}:{work}:{digest}" if work else f"{kind}:{digest}"
    return f"{kind}:{work}" if work else None


def _candidate_importance(candidate: Dict[str, Any]) -> float:
    """The 0..1 importance a candidate warrants for the notification decision.

    The input is the candidate's own *weight* (relevance), discounted into the
    notification-importance scale so the two are not conflated. An explicit
    ``importance`` (unused today) is honoured verbatim and not discounted twice.
    """
    try:
        explicit = float(candidate.get("importance"))
        return max(0.0, min(1.0, explicit))
    except (TypeError, ValueError):
        pass
    try:
        weight = float(candidate.get("weight") or 0.0)
    except (TypeError, ValueError):
        weight = 0.0
    return round(max(0.0, min(1.0, weight * _CANDIDATE_IMPORTANCE_SCALE)), 4)


def candidate_outcome(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map a conversation candidate to a notification outcome, or ``None``.

    Returns ``None`` when the candidate carries no recorded reason or no stable
    identity, so nothing without a grounded basis and a dedupe key can ever
    reach the delivery path.
    """
    if not isinstance(candidate, dict):
        return None
    reason = str(candidate.get("reason") or "").strip()
    ref_id = candidate_ref(candidate)
    if not reason or not ref_id:
        return None
    return {
        "ref_id": ref_id,
        "kind": str(candidate.get("kind") or "candidate"),
        "status": "candidate",
        "problem": False,
        "reason": reason,
        "summary": reason,
        "importance": round(_candidate_importance(candidate), 4),
        "candidate": dict(candidate),
    }


# -- affect -> outreach relevance (Change 2) --------------------------------
#
# Her *current* affective state already influences *what she does* through
# ``behavior.from_affect`` -> ``activity_bias``. It should also influence
# whether a grounded reason is worth saying. The semantically appropriate
# existing signal is ``anticipation``: affect.py documents it as the forward
# pull toward an *expected outcome*, and a reason to speak (a question awaiting
# resolution, a work about to finish) is exactly such a forward pull. It is NOT
# the same as ``activity_bias`` (effort/take-on for a task), which would be the
# wrong meaning to feed into a communication decision.
#
# The projection is scaled small so mood *tilts* the relevance of a candidate
# rather than making her indiscriminately talkative: a fully-anticipated reason
# rises by ``_ANTICIPATION_GAIN``, a flat one is unchanged. ``candidate_strength``
# then applies its own 0.15 relevance weight, keeping the total effect bounded.

_ANTICIPATION_GAIN = 0.3


def outreach_relevance(state: Any) -> float:
    """The 0..1 relevance her current affect adds to an outreach decision.

    A pure read of the existing affect state via ``affect.diagnostics`` - no
    new store, no second personality simulation. Flat/neutral affect yields
    ``0.0`` so the decision is unchanged when she has no strong forward pull.
    """
    try:
        from . import affect as affect_mod

        diag = affect_mod.diagnostics(state)
    except Exception:
        return 0.0
    try:
        anticipation = float(diag.get("anticipation") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    anticipation = max(0.0, min(1.0, anticipation))
    return round(anticipation * _ANTICIPATION_GAIN, 4)
