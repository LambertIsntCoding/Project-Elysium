"""The background reader: Astra reads a little, then goes quiet again.

This is the only piece of the reading system that runs on its own. It is a
low-priority daemon that *wakes only when conditions are suitable*, reads a
bounded portion of one work, saves its progress and the resulting understanding,
and then idles. It is not a continuously running model process, and it is not on
a timer that decides to be spontaneous:

* the decision to read is the deterministic policy in :mod:`astra.reading`
  (Roum idle, no game, machine not loaded);
* a game - or an explicitly busy GPU - always wins, because the reader shares the
  machine with whatever Roum is doing;
* each cycle is bounded, and the pause conditions are re-checked *between*
  chunks, so an interaction or a game takes effect immediately rather than after
  a whole work;
* progress is saved after every chunk, so an interruption loses at most the
  chunk in flight and never forces a reread.

The reader owns no memory of its own. It records the *understanding* through the
ordinary store (``TripleMemoryStore.apply_reading_results``), so it inherits the
same evidence/decay/contradiction machinery as everything else, and it records
the *position* through :class:`astra.library.Library`. When it is done it reports
what it did; it never decides to speak.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from . import reading
from .library import Library

DEFAULT_POLL_SECONDS = 5.0
DEFAULT_IDLE_SECONDS = reading.DEFAULT_IDLE_SECONDS
DEFAULT_LOAD_CEILING = reading.DEFAULT_LOAD_CEILING


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_cpu_load() -> Optional[float]:
    """CPU load fraction (0.0-1.0), or ``None`` where it is not observable.

    Uses ``os.getloadavg`` where available and normalises by the core count. On
    platforms without it (e.g. Windows) this returns ``None``, and the reader
    simply does not stand down for load - it never guesses.
    """
    try:
        import os

        if hasattr(os, "getloadavg"):
            load1, _, _ = os.getloadavg()
            cores = os.cpu_count() or 1
            return max(0.0, min(1.0, load1 / cores))
    except (OSError, AttributeError, ValueError):
        return None
    return None


def default_process_names() -> Optional[List[str]]:
    """Running process names, if a provider is installed; otherwise ``None``.

    ``psutil`` is optional. Without it the reader cannot detect a game by itself,
    so game pausing relies on the caller signalling one (see
    ``note_game_pause``); the reader never invents a reason to stop.
    """
    try:
        import psutil  # optional dependency

        names: List[str] = []
        for proc in psutil.process_iter(["name"]):
            try:
                name = proc.info.get("name")
                if name:
                    names.append(str(name))
            except Exception:
                continue
        return names
    except Exception:
        return None


class BackgroundReader:
    """A bounded, low-priority reading daemon over a :class:`Library`.

    ``model_fn`` is the one expensive call. It is injected so the reader can be
    exercised deterministically in tests and so it reuses whatever local backend
    the application already uses - it is never hardcoded here.
    """

    def __init__(
        self,
        library: Library,
        store: Any,
        model_fn: Callable[[str], str],
        *,
        enabled: bool = True,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        load_ceiling: float = DEFAULT_LOAD_CEILING,
        chunk_chars: int = reading.DEFAULT_CHUNK_CHARS,
        chunks_per_cycle: int = reading.DEFAULT_CHUNKS_PER_CYCLE,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        cpu_load_fn: Optional[Callable[[], Optional[float]]] = None,
        process_names_fn: Optional[Callable[[], Optional[List[str]]]] = None,
        process_policy: Optional[reading.ProcessPolicy] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.library = library
        self.store = store
        self.model_fn = model_fn
        self.enabled = bool(enabled)
        self.idle_seconds = float(idle_seconds)
        self.load_ceiling = float(load_ceiling)
        self.chunk_chars = int(chunk_chars)
        self.chunks_per_cycle = max(1, int(chunks_per_cycle))
        self.poll_seconds = float(poll_seconds)
        self.cpu_load_fn = cpu_load_fn or default_cpu_load
        self.process_names_fn = process_names_fn or default_process_names
        # One shared policy says what counts as busy and what to ignore, so every
        # background task consults the same answer instead of re-implementing it.
        self.process_policy = process_policy or reading.ProcessPolicy()
        self._clock = clock
        self._sleep = sleep_fn

        self._lock = threading.RLock()
        self._last_activity = 0.0          # monotonic timestamp of last interaction
        self._game_override: Optional[str] = None
        self._gpu_busy = False
        self._busy_process: Optional[str] = None
        self._busy_flag = False            # a non-game busy task was signalled
        self._last_game: Optional[str] = None   # last process seen (for diagnostics)
        self._last_busy: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_cycle: Dict[str, Any] = {}
        self.last_reason: str = reading.IDLE_NO_WORK
        # The current pace (chunks this cycle, rest afterwards) and the state it
        # was derived from, kept for diagnostics. Set each cycle.
        self.last_pace: Dict[str, Any] = {}
        self._rest_seconds = DEFAULT_POLL_SECONDS

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Start the daemon once. Idempotent and safe to call when disabled."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="astra-background-reader", daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                # A background courtesy must never take down the conversation.
                pass
            # Reading takes time: after a cycle, rest for the pace the current
            # state calls for rather than hammering the work. An interaction or
            # a game is still noticed during the wait.
            self._stop.wait(max(self.poll_seconds, self._rest_seconds))

    # -- signals -----------------------------------------------------------
    def note_activity(self) -> None:
        """Record that Roum just interacted, so the reader yields immediately."""
        with self._lock:
            self._last_activity = self._clock()

    def clear_activity(self) -> None:
        """Forget the last interaction, as if Roum had been quiet all along.

        Used by an explicit ``/read`` request: the request itself is interaction,
        and it should not be what stops the cycle it just asked for.
        """
        with self._lock:
            self._last_activity = 0.0

    def note_game_pause(self, process: Optional[str], active: bool = True) -> None:
        """Signal that a game is (or is no longer) running."""
        with self._lock:
            self._game_override = str(process) if (active and process) else None

    def note_busy_pause(self, process: Optional[str], active: bool = True) -> None:
        """Signal a non-game *busy* task (a render, build, call) is running.

        Treated like load rather than a game: it stands the reader down but does
        not claim the GPU. An ignored process is refused here, so a task on the
        ignore list can never be used to stop reading.
        """
        with self._lock:
            text = str(process or "").strip()
            if active and text and not self.process_policy.is_ignored(text):
                self._busy_process = text
                self._busy_flag = True
            else:
                self._busy_process = None
                self._busy_flag = False

    def note_gpu_busy(self, busy: bool) -> None:
        with self._lock:
            self._gpu_busy = bool(busy)

    def _seconds_since_input(self) -> Optional[float]:
        with self._lock:
            last = self._last_activity
        if not last:
            return None  # no interaction observed yet: not "active"
        return max(0.0, self._clock() - last)

    def _affect_snapshot(self) -> Dict[str, Any]:
        """The current experiential-affect state, read from the store.

        Read-only and tolerant: a store without the affect method (or an empty
        state) simply yields neutral values, so the gate never invents a reason
        to stop and reading is never silently disabled by a missing record.
        """
        getter = getattr(self.store, "current_affect", None)
        if not callable(getter):
            return {}
        try:
            return getter() or {}
        except Exception:
            return {}

    def _gap_seconds(self) -> Optional[float]:
        """Seconds since Astra was last present, or ``None`` if not recorded."""
        last = getattr(self.store, "last_present", None)
        if not callable(last):
            return None
        try:
            stamp = last()
        except Exception:
            return None
        if not stamp:
            return None
        try:
            from datetime import datetime as _dt
            ts = _dt.fromisoformat(str(stamp))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return max(0.0, (_dt.now(timezone.utc) - ts).total_seconds())
        except (TypeError, ValueError):
            return None

    def _conditions(self) -> Dict[str, Any]:
        with self._lock:
            game = self._game_override
            gpu_busy = self._gpu_busy
            busy_signalled = self._busy_flag
        # A game is inferred from the process list (its patterns are built in); a
        # configured "busy" process stands down on load but does not claim the
        # GPU. Ignored names are filtered inside the policy, so a trusted task
        # never blocks reading.
        busy_scanned = False
        busy_name = self._busy_process
        if self.process_names_fn is not None:
            try:
                found_game, found_busy = self.process_policy.classify(
                    self.process_names_fn() or [])
                game = game or found_game
                busy_scanned = bool(found_busy)
                busy_name = busy_name or found_busy
            except Exception:
                pass
        self._last_game = game
        self._last_busy = busy_name
        cpu_load = None
        if self.cpu_load_fn is not None:
            try:
                cpu_load = self.cpu_load_fn()
            except Exception:
                cpu_load = None
        affect = self._affect_snapshot()
        return {
            "enabled": self.enabled,
            "seconds_since_input": self._seconds_since_input(),
            "game": game,
            "gpu_busy": gpu_busy,
            "machine_busy_flag": busy_signalled or busy_scanned,
            "cpu_load": cpu_load,
            # Reading requires concentration: the gate reads her current state.
            "concentration": affect.get("concentration"),
            "weary": affect.get("weary"),
            "frustration": affect.get("frustration"),
        }

    def may_read(self, *, force: bool = False) -> tuple:
        """Whether the reader may run now, and why (for diagnostics)."""
        conditions = self._conditions()
        has_work = bool(self.library.next_work_id())
        allowed, reason = reading.should_read(
            has_work=has_work, idle_threshold=self.idle_seconds,
            load_ceiling=self.load_ceiling, force=force, **conditions,
        )
        # Cache the reason so the prompt path can describe the reader without
        # re-running the gate (which may scan processes).
        self.last_reason = reason
        return allowed, reason

    # -- one bounded cycle -------------------------------------------------
    def run_once(self, *, allow_model: bool = True, force: bool = False) -> Dict[str, Any]:
        """One wake: gate, then read at most ``chunks_per_cycle`` chunks.

        Returns a small report (what it read, or why it did not) and leaves the
        position saved. Pure enough to call directly in tests.

        ``force`` runs the cycle against the courtesy conditions (idle, load, a
        busy task) - Roum's explicit request. The between-chunk pause check is
        *not* forced, so the cycle still yields to an interaction or a game after
        this chunk; force starts a cycle, it does not silence the reader.
        """
        allowed, reason = self.may_read(force=force)
        if not allowed:
            return self._report(False, reason, [])

        work_id = self.library.current_work_id() or self.library.next_work_id()
        if not work_id:
            return self._report(False, reading.IDLE_NO_WORK, [])

        text = self.library.text_for(work_id)
        if not text:
            # The cached text is gone; nothing to read, but do not crash.
            self.library.set_status(work_id, reading.WORK_ABANDONED)
            return self._report(False, reading.IDLE_NO_WORK, [])
        if not allow_model:
            return self._report(False, reading.IDLE_HELD, [])

        state = self.library.get_state(work_id)
        read_chunks: List[Dict[str, Any]] = []

        # How much to read is a function of her current state and her sense of
        # elapsed time - not a fixed count. Reading takes concentration, and a
        # long absence lets her catch up a little when she is engaged.
        affect = self._affect_snapshot()
        pace = reading.reading_pace(
            concentration=affect.get("concentration"),
            weary=affect.get("weary"),
            engagement=affect.get("engagement"),
            frustration=affect.get("frustration"),
            gap_seconds=self._gap_seconds(),
            base_chunks=self.chunks_per_cycle,
        )
        self.last_pace = pace
        self._rest_seconds = float(pace.get("rest_seconds") or 0.0)
        if not pace.get("allowed", True):
            return self._report(False, reading.IDLE_NOT_CONCENTRATING, [])
        chunks_this_cycle = int(pace.get("chunks") or 0)

        for _ in range(chunks_this_cycle):
            offset = int(state.get("char_offset") or 0)
            if offset >= len(text):
                self._finish(work_id, state)
                break
            # Re-check between chunks so a game or an interaction stops us now.
            # A forced cycle keeps the same courtesy bypass here, so forcing
            # against a busy task actually reads; a game still stops it.
            pause = reading.pause_reason(
                idle_threshold=self.idle_seconds, load_ceiling=self.load_ceiling,
                force=force, **self._conditions(),
            )
            if pause is not None:
                break

            start, end = reading.chunk_bounds(text, offset, max_chars=self.chunk_chars)
            if end <= start:
                break
            passage = text[start:end]
            digest = self._extract(work_id, state, passage) if allow_model else {}
            if digest is None:
                # The model call failed, so this passage was *not* processed.
                # Do not advance: leave the offset where it is so the next cycle
                # retries it. Losing one passage is better than skipping it.
                break
            applied = self._apply(work_id, state, digest, passage) if digest else {"memories": 0}
            state = reading.advance(
                state, char_offset=end, units_read=int(state.get("units_read") or 0) + 1,
                words_read=reading.words_at(text, end),
                resume_context=reading.resume_context_from(digest) or state.get("resume_context", ""),
                finished=end >= len(text),
            )
            state["last_read_at"] = _now()
            state = self.library.save_state(work_id, state)
            read_chunks.append({"work_id": work_id, "start": start, "end": end,
                                "words": reading.words_at(text, end)
                                - reading.words_at(text, start),
                                "memories": applied.get("memories", 0)})
            if state.get("status") == reading.WORK_FINISHED:
                self._finish(work_id, state)
                break

        if not read_chunks:
            # The gate said yes but nothing advanced. Distinguish the cases:
            # no model call was permitted, the model was unreachable, or there
            # was simply nothing left to read.
            if not allow_model:
                reason = reading.IDLE_HELD
            elif text and int(self.library.get_state(work_id).get("char_offset") or 0) < len(text):
                reason = reading.IDLE_MODEL_FAILED
            else:
                reason = reading.IDLE_WORK_FINISHED
            return self._report(False, reason, [])
        return self._report(True, reason, read_chunks)

    def _extract(self, work_id: str, state: Dict[str, Any], passage: str) -> Optional[Dict[str, Any]]:
        """One model call for one bounded passage, parsed into the digest shape.

        Returns ``None`` when the call *failed* (so the caller does not advance
        past an unprocessed passage) and ``{}`` when it succeeded but produced
        nothing usable (an empty passage is still processed).
        """
        prompt = reading.reading_prompt(
            state.get("title") or work_id, passage,
            context=state.get("resume_context") or "",
        )
        try:
            raw = self.model_fn(prompt)
        except Exception:
            return None
        if isinstance(raw, str) and raw.startswith("[Error"):
            return None
        return reading.parse_extraction(raw)

    def _apply(self, work_id: str, state: Dict[str, Any],
               digest: Dict[str, Any], passage: str) -> Dict[str, Any]:
        apply_results = getattr(self.store, "apply_reading_results", None)
        if not callable(apply_results):
            return {"memories": 0}
        try:
            return apply_results(
                work_id=work_id, title=state.get("title") or work_id, digest=digest,
                passage=passage,
            )
        except Exception:
            return {"memories": 0}

    def _finish(self, work_id: str, state: Dict[str, Any]) -> None:
        """Mark a work finished and record the completion as an experience."""
        if state.get("status") != reading.WORK_FINISHED:
            state = dict(state)
            state["status"] = reading.WORK_FINISHED
            self.library.save_state(work_id, state)
        finish = getattr(self.store, "record_reading_finish", None)
        if callable(finish):
            try:
                finish(work_id=work_id, title=state.get("title") or work_id,
                       chunks=int(state.get("chunks") or 0))
            except Exception:
                pass
        # Advance to the next live work, if any.
        nxt = self.library.next_work_id()
        if nxt and nxt != work_id:
            self.library.set_current(nxt)

    def _report(self, read: bool, reason: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            game = self._game_override
        report = {
            "read": bool(read),
            "reason": reason,
            "chunks": len(chunks),
            # Words read this cycle: the human-facing measure, summed from the
            # chunks actually processed (a chunk carries how many words it held).
            "words": sum(int(c.get("words") or 0) for c in chunks),
            "work_id": chunks[-1]["work_id"] if chunks else self.library.current_work_id(),
            "detail": chunks,
            "outcome": "ok" if read else reason,
            "at": _now(),
            "pace": dict(self.last_pace or {}),
        }
        self.last_cycle = report
        return report

    # -- diagnostics -------------------------------------------------------
    def diagnostics(self) -> Dict[str, Any]:
        state = self.library.current_state()
        _, reason = self.may_read()
        queue = [w.get("work_id") for w in self.library.list_works()
                 if reading.is_live(w)]
        return reading.diagnostics(
            state, enabled=self.enabled, reason=reason, game=self._last_game,
            busy=self._last_busy, policy=self.process_policy.as_dict(),
            work_queue=queue, last_cycle=self.last_cycle,
        )

    def format_diagnostics(self) -> str:
        d = self.diagnostics()
        # The human-facing measure is words; a legacy state carries only character
        # offsets, so fill the word counts in (read-only) for the display.
        state = self.library.current_state()
        measure = getattr(self.library, "word_measure", None)
        if callable(measure):
            words_read, total_words = measure(state)
            state = dict(state) | {"words_read": words_read,
                                   "total_words": total_words}
        return reading.format_diagnostics(
            state, enabled=d["enabled"], reason=d["reason"],
            game=d["game"], busy=d["busy"], work_queue=d["queue"],
            last_cycle=d["last_cycle"],
        )


__all__ = ["BackgroundReader", "default_cpu_load", "default_process_names"]
