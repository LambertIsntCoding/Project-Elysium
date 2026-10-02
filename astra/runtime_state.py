"""Astra's application/runtime state: ACTIVE and SLEEP.

This is the application's own state machine, not an emotion. It answers one
question - "is Astra currently engaged with a human interaction, or is she free
to do bounded background work?" - and nothing else. It is deliberately separate
from :mod:`astra.affect` (temporary experiential state) and
:mod:`astra.temporal` (felt elapsed time): the runtime state is a fact about the
process, never a feeling, and it must never be turned into one.

Design constraints:

* **SLEEP is caused by an explicit human/application decision, never by
  silence.** Merely having no incoming message does not put Astra to sleep, and
  going quiet is never an emotional event. The only ways in are a human/CLI
  command, an explicit application lifecycle decision, or (when configured) an
  idle *policy* that the application chooses to honour.
* **Waking is a human-caused wake event.** A transition out of SLEEP records
  *who/what* woke her, so a background task can never wake her by itself.
* **Persisted with its own contract.** The state lives in its own JSON file with
  an explicit version and a tolerant loader. It is not a memory, is never
  written into the memory models, and a corrupt or missing file degrades to
  ACTIVE rather than crashing a session.
* **Pure transitions.** :func:`transition` is a pure function over a state dict,
  so the whole machine is testable without I/O.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from .memory import atomic_save, load_json

STATE_VERSION = 1

ACTIVE = "active"
SLEEP = "sleep"
RUNTIME_STATES = (ACTIVE, SLEEP)

# Why a transition happened. The reason is recorded for auditing and for the
# wake event; it is never shown to Astra as a feeling.
REASON_HUMAN_COMMAND = "human_command"       # Roum/CLI explicitly asked
REASON_LIFECYCLE = "lifecycle"               # application startup/shutdown decision
REASON_IDLE_POLICY = "idle_policy"           # a configured idle policy fired
REASON_HUMAN_MESSAGE = "human_message"       # a human spoke: wake
REASON_HUMAN_WAKE = "human_wake"             # explicit wake command
REASON_TASK = "background_task"              # must NOT wake her (guarded)

# The only reasons that may move ACTIVE -> SLEEP. A background task is absent by
# construction: no scheduler cycle can put her to sleep.
_SLEEP_CAUSES = (REASON_HUMAN_COMMAND, REASON_LIFECYCLE, REASON_IDLE_POLICY)
# The only reasons that may move SLEEP -> ACTIVE. All are human/application
# caused; REASON_TASK is explicitly rejected so work can never wake her.
_WAKE_CAUSES = (REASON_HUMAN_MESSAGE, REASON_HUMAN_WAKE, REASON_HUMAN_COMMAND,
                REASON_LIFECYCLE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def blank_state() -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "state": ACTIVE,
        "since": _now(),
        "reason": REASON_LIFECYCLE,
        "wake_reason": None,
        "wake_events": 0,
        "cycles": 0,
    }


def coerce_state(raw: Any) -> Dict[str, Any]:
    """Tolerant read: anything malformed degrades to a fresh ACTIVE state."""
    state = blank_state()
    if not isinstance(raw, dict):
        return state
    value = str(raw.get("state") or "").strip().lower()
    if value in RUNTIME_STATES:
        state["state"] = value
    for key in ("since", "reason", "wake_reason"):
        if raw.get(key):
            state[key] = str(raw[key])
    for key in ("wake_events", "cycles"):
        try:
            state[key] = max(0, int(raw.get(key, 0) or 0))
        except (TypeError, ValueError):
            pass
    return state


def is_sleeping(state: Any) -> bool:
    return coerce_state(state)["state"] == SLEEP


def is_active(state: Any) -> bool:
    return coerce_state(state)["state"] == ACTIVE


def transition(state: Any, target: str, *, reason: str,
               now: Optional[str] = None) -> Dict[str, Any]:
    """Pure transition. Returns a *new* state; illegal moves are no-ops.

    ``ACTIVE -> SLEEP`` requires one of ``_SLEEP_CAUSES``; ``SLEEP -> ACTIVE``
    requires one of ``_WAKE_CAUSES`` (a background task is rejected). A no-op
    move leaves the state untouched, so a stray call cannot flip her.
    """
    current = coerce_state(state)
    stamp = now or _now()
    target = str(target or "").strip().lower()
    if target not in RUNTIME_STATES or target == current["state"]:
        return current

    if target == SLEEP:
        if reason not in _SLEEP_CAUSES:
            return current
        current["state"] = SLEEP
        current["reason"] = reason
        current["since"] = stamp
        current["wake_reason"] = None
        return current

    # target == ACTIVE: waking.
    if reason not in _WAKE_CAUSES:
        return current
    current["state"] = ACTIVE
    current["reason"] = reason
    current["wake_reason"] = reason
    current["since"] = stamp
    current["wake_events"] = int(current.get("wake_events", 0)) + 1
    return current


def render_summary(state: Any) -> str:
    """A plain-language one-liner for the CLI/Discord, never an emotion."""
    current = coerce_state(state)
    if current["state"] == SLEEP:
        return "Astra is asleep (bounded background work only)."
    return "Astra is active."


class RuntimeStateStore:
    """Persistent holder of the runtime state, with its own serialization.

    Separate file, separate version, separate loader - it is application state,
    not a memory, and never appears in the memory models or the prompt.
    """

    def __init__(self, data_dir: str = "./storage", *,
                 clock: Optional[Callable[[], str]] = None) -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "runtime_state.json")
        self._clock = clock or _now
        self._lock = threading.RLock()
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        raw = load_json(self.path, dict, dict)
        return coerce_state(raw)

    def current(self) -> Dict[str, Any]:
        with self._lock:
            return coerce_state(self._state)

    def _persist(self) -> None:
        with self._lock:
            atomic_save(self.path, self._state)

    def set_state(self, target: str, *, reason: str) -> Dict[str, Any]:
        """Apply a transition and persist it if it was legal."""
        with self._lock:
            new_state = transition(self._state, target, reason=reason,
                                   now=self._clock())
            if new_state != self._state:
                self._state = new_state
                self._persist()
            return coerce_state(self._state)

    def sleep(self, *, reason: str = REASON_HUMAN_COMMAND) -> Dict[str, Any]:
        return self.set_state(SLEEP, reason=reason)

    def wake(self, *, reason: str = REASON_HUMAN_WAKE) -> Dict[str, Any]:
        return self.set_state(ACTIVE, reason=reason)

    def note_cycle(self) -> None:
        """Record that one scheduler cycle ran (bookkeeping only)."""
        with self._lock:
            self._state["cycles"] = int(self._state.get("cycles", 0)) + 1
            self._persist()

    def is_sleeping(self) -> bool:
        return is_sleeping(self.current())

    def is_active(self) -> bool:
        return is_active(self.current())

    def render(self) -> str:
        return render_summary(self.current())
