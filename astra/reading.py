"""Astra's reading: what she is reading, how far, and what she has made of it.

This module is the *vocabulary and policy* layer for background reading. It owns
no store, no network, no model, and no thread; it is pure functions over plain
data, in the same spirit as :mod:`astra.inquiry`. That keeps the design honest:

* the durable *understanding* of a work is an ordinary memory - observations,
  interpretations, hypotheses and open questions - written through the existing
  store, so it inherits evidence, decay, dormancy, contradiction handling and
  supersession (see ``TripleMemoryStore.apply_reading_results``);
* the *reading progress* (where she is, what she has covered) is a compact
  structured record that this module describes, so it can be persisted
  atomically and resumed without rereading (see :mod:`astra.library`);
* the *decision to read right now* is a deterministic policy over observable
  conditions - Roum is idle, no game is running, the machine is not loaded - so
  the reader wakes only when it is polite to, never on a timer and never by
  guessing the model's mood.

Two boundaries this module deliberately preserves:

* **General knowledge vs work-specific understanding.** Nothing here consults
  the base model's knowledge of a work. Only what Astra actually recorded about
  *this* work, under its own ``work_id``, is ever surfaced.
* **Reason to speak vs speaking.** ``conversation_candidates`` preserves why she
  might want to raise something; it never sends anything. Initiative stays a
  later, explicit decision.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------
# Work vocabulary
# ---------------------------------------------------------------------
# A "work" is a substantial thing read over time: a book, a long article, a
# paper. ``work_id`` is the slug the rest of the store already uses to keep two
# works apart (``inquiry.work_of``); nothing here invents a second identity.
WORK_READING = "reading"
WORK_PAUSED = "paused"
WORK_FINISHED = "finished"
WORK_ABANDONED = "abandoned"
WORK_STATUSES = (WORK_READING, WORK_PAUSED, WORK_FINISHED, WORK_ABANDONED)
LIVE_WORK_STATUSES = (WORK_READING, WORK_PAUSED)

SOURCE_FORMAT_TEXT = "text"
SOURCE_FORMAT_EPUB = "epub"

# ---------------------------------------------------------------------
# Why the reader is (or is not) running right now
# ---------------------------------------------------------------------
# Every decision the reader makes is explained by one of these, and the reason is
# surfaced in diagnostics so "why is she reading?" always has an answer.
IDLE_OK = "idle_ok"                      # conditions are suitable: read a little
IDLE_USER_ACTIVE = "user_active"         # Roum is interacting: do not read
IDLE_GAME_RUNNING = "game_running"       # a game is running: yield, especially the GPU
IDLE_MACHINE_BUSY = "machine_busy"       # the machine is loaded: do not add work
IDLE_NO_WORK = "no_work"                 # nothing is queued to read
IDLE_DISABLED = "disabled"               # reading is switched off
IDLE_WORK_FINISHED = "work_finished"     # the current work is done; pick another
IDLE_MODEL_FAILED = "model_unavailable"  # the local model could not be reached
IDLE_HELD = "held"                       # no model call was permitted this cycle
IDLE_FORCED = "forced"                   # Roum forced a cycle: the gate was bypassed
IDLE_NOT_CONCENTRATING = "not_concentrating"  # too scattered/weary to read well

# The reasons that are genuine *interruptions* (as opposed to "fine to read" or
# "nothing to read"). Only these are worth telling Astra about in her prompt.
_PAUSE_REASONS = (IDLE_USER_ACTIVE, IDLE_GAME_RUNNING, IDLE_MACHINE_BUSY,
                  IDLE_NOT_CONCENTRATING)

# Default thresholds. Kept as module constants so the behaviour is auditable and
# tunable, exactly like the memory heuristics in ``astra.memory``.
DEFAULT_IDLE_SECONDS = 90.0        # how long Roum must be quiet before reading
DEFAULT_LOAD_CEILING = 0.75        # CPU load fraction above which we stand down
DEFAULT_CHUNK_CHARS = 3500         # max characters of a work per model call
DEFAULT_CHUNKS_PER_CYCLE = 1       # bounded work per wake
DEFAULT_MIN_CHUNK_CHARS = 200      # ignore trailing fragments shorter than this

# Reading is not free: it needs a settled mind. A weary or worked-up state
# stands the reader down; how *concentrated* she is then scales how much she
# reads, and a long absence lets her read a little more to catch up. All of it
# is read from the existing experiential-affect state - no new store. Note that
# this accumulator is zero-neutral: 0 means "no particular condition", so a low
# concentration *value* is not a scattered state - a scattered state is high
# weary/frustration. Concentration is therefore a modifier, not a gate.
#
# The gate only stops at *extreme* weariness; moderate weariness reads less
# (``WEARY_CEILING`` below), so "reading takes concentration" makes her read
# more slowly rather than not at all.
WEARY_GATE_CEILING = 0.80       # at/above this she is too drained to read
FRUSTRATION_CEILING = 0.75      # at/above this she is too worked up to read
CONCENTRATION_HIGH = 0.50       # at/above this she reads a little more
WEARY_CEILING = 0.55            # at/above this she reads less, not more
DEFAULT_PACE_CHUNKS = 1            # the normal amount per cycle
MAX_PACE_CHUNKS = 2               # never more than this, however fresh
LONG_ABSENCE_SECONDS = 3 * 86400.0  # an absence this long counts as "a while"
# A small rest between cycles so reading a work takes time rather than being
# consumed in one burst. Scaled by the pace (see ``reading_pace``).
DEFAULT_REST_SECONDS = 20.0
WEARY_REST_SECONDS = 90.0


def concentration_ok(*, concentration: Optional[float] = None,
                     weary: Optional[float] = None,
                     frustration: Optional[float] = None) -> bool:
    """True when Astra is settled enough to read well.

    Reading requires concentration in the sense that a scattered state - too
    weary, or too frustrated - is not a reading moment. Deliberately lenient:
    an unknown or neutral state is fine, so a missing affect record never
    silently disables reading. ``concentration`` is accepted for callers that
    pass it but is a *modifier* (see ``reading_pace``), not a gate.
    """
    for value, ceiling in ((weary, WEARY_GATE_CEILING), (frustration, FRUSTRATION_CEILING)):
        if value is None:
            continue
        try:
            if float(value) >= ceiling:
                return False
        except (TypeError, ValueError):
            continue
    return True


def reading_pace(*, concentration: Optional[float] = None,
                 weary: Optional[float] = None, engagement: Optional[float] = None,
                 frustration: Optional[float] = None,
                 gap_seconds: Optional[float] = None,
                 base_chunks: int = DEFAULT_PACE_CHUNKS) -> Dict[str, Any]:
    """How much to read this cycle, from Astra's state and her sense of time.

    Reading requires concentration and takes time. The amount read is a *derived*
    function of the existing experiential-affect state and the elapsed gap since
    she was last present - never a stored counter:

    * a weary or frustrated state reads less (or nothing);
    * a long absence since she was last present lets her read a little more to
      catch up, but only when she is engaged and not weary;
    * otherwise she reads the base amount, a little more when she is
      concentrating well.

    Returns a small dict with ``chunks``, ``rest_seconds``, ``allowed`` and a
    plain-language ``reason`` for diagnostics.
    """
    chunks = max(0, int(base_chunks))
    allowed = concentration_ok(concentration=concentration, weary=weary,
                               frustration=frustration)
    if not allowed:
        return {"chunks": 0, "rest_seconds": 0.0, "allowed": False,
                "reason": "not settled enough to read"}

    def _num(value: Optional[float]) -> float:
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    weary_value = _num(weary)
    engaged = _num(engagement)
    focused = _num(concentration)

    if weary_value >= WEARY_CEILING:
        chunks = min(chunks, 1)
        rest = WEARY_REST_SECONDS
        reason = "weary: reading a little, then resting"
    else:
        rest = DEFAULT_REST_SECONDS
        if focused >= CONCENTRATION_HIGH:
            chunks = min(MAX_PACE_CHUNKS, chunks + 1)
            reason = "concentrating well: reading a bit more"
        else:
            reason = "reading at a steady pace"
        # A long absence plus genuine engagement lets her catch up a little.
        if (gap_seconds is not None and float(gap_seconds) >= LONG_ABSENCE_SECONDS
                and engaged >= 0.4):
            chunks = min(MAX_PACE_CHUNKS, max(chunks, base_chunks + 1))
            reason = "been away a while and engaged: reading a bit more to catch up"
    return {"chunks": max(0, chunks), "rest_seconds": float(rest),
            "allowed": True, "reason": reason}

# ---------------------------------------------------------------------
# Reading-state schema helpers
# ---------------------------------------------------------------------
def blank_state(work_id: str, *, title: str = "", source_format: str = "",
                total_units: int = 0, fingerprint: str = "") -> Dict[str, Any]:
    """A fresh reading-progress state for a work.

    Deliberately small: position, a bounded resume context, and the counters
    needed to explain progress. ``total_units`` is the length of the cached text
    (characters), which is what the byte-for-byte read guarantee is expressed in;
    ``units_read`` and ``words_read`` are the human-facing measure - words.
    The understanding itself lives in memories.
    """
    return {
        "work_id": str(work_id or "").strip(),
        "title": str(title or "").strip(),
        "source_format": str(source_format or "").strip(),
        "fingerprint": str(fingerprint or "").strip(),
        "status": WORK_READING,
        "total_units": int(total_units or 0),
        "units_read": 0,
        "words_read": 0,
        "total_words": 0,
        "char_offset": 0,
        "resume_context": "",
        "started_at": "",
        "last_read_at": "",
        "chunks": 0,
    }


def coerce_state(raw: Any, work_id: str = "") -> Dict[str, Any]:
    """Normalise a stored reading-state, tolerant of partial/legacy data."""
    if not isinstance(raw, dict):
        return blank_state(work_id)
    resolved = work_id or str(raw.get("work_id") or "")
    state = blank_state(resolved)
    for field in ("work_id", "title", "source_format", "fingerprint", "status",
                  "resume_context", "started_at", "last_read_at"):
        if raw.get(field) is not None:
            state[field] = str(raw.get(field))
    for field in ("total_units", "units_read", "char_offset", "chunks"):
        try:
            state[field] = max(0, int(raw.get(field, 0) or 0))
        except (TypeError, ValueError):
            state[field] = 0
    try:
        state["words_read"] = max(0, int(raw.get("words_read", 0) or 0))
    except (TypeError, ValueError):
        state["words_read"] = 0
    try:
        state["total_words"] = max(0, int(raw.get("total_words", 0) or 0))
    except (TypeError, ValueError):
        state["total_words"] = 0
    if state["status"] not in WORK_STATUSES:
        state["status"] = WORK_READING
    return state


def progress(state: Any) -> float:
    """Fraction of the work covered, 0.0-1.0. Character offset is the truth."""
    state = coerce_state(state)
    total = int(state.get("total_units") or 0)
    if total <= 0:
        return 0.0
    return round(min(1.0, max(0.0, int(state.get("char_offset") or 0) / total)), 4)


def words_in_text(text: str) -> int:
    """Approximate word count of ``text`` (whitespace-delimited tokens)."""
    return len(str(text or "").split())


def words_at(text: str, char_offset: int) -> int:
    """Words fully read when the reader has reached ``char_offset`` characters.

    The same measure as :func:`words_in_text`, applied to the prefix, so
    "words read" and "total words" are always computed the same way.
    """
    text = str(text or "")
    offset = max(0, min(len(text), int(char_offset or 0)))
    return words_in_text(text[:offset])


def is_live(state: Any) -> bool:
    return coerce_state(state).get("status") in LIVE_WORK_STATUSES


def advance(state: Any, *, char_offset: int, units_read: int,
            words_read: Optional[int] = None, resume_context: str = "",
            finished: bool = False) -> Dict[str, Any]:
    """Return ``state`` advanced to a new position (never mutates the input)."""
    state = dict(coerce_state(state))
    state["char_offset"] = max(0, int(char_offset))
    state["units_read"] = max(0, int(units_read))
    if words_read is not None:
        state["words_read"] = max(0, int(words_read))
    state["chunks"] = int(state.get("chunks", 0)) + 1
    if resume_context:
        state["resume_context"] = str(resume_context)[:600]
    if finished:
        state["status"] = WORK_FINISHED
    elif state["status"] not in WORK_STATUSES:
        state["status"] = WORK_READING
    return state


# ---------------------------------------------------------------------
# Detectors (narrow, deterministic - a false positive corrupts the state)
# ---------------------------------------------------------------------
_GAME_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\bsteam(?:\.exe)?\b", r"\bsteamapps\b", r"\bepic games\b", r"\bgog galaxy\b",
    r"\briot client\b", r"\bbattlenet\b", r"\bblizzard\b", r"\bubisoft\b",
    r"\bdota2?\b", r"\bcounter-?strike\b", r"\bcs2?\.exe\b", r"\bvalorant\b",
    r"\bleague of legends\b", r"\briot\b", r"\bfortnite\b", r"\bminecraft\b",
    r"\bgenshin\b", r"\bhonkai\b", r"\bwitcher\b", r"\bcyberpunk\b",
    r"\bskyrim\b", r"\bfactorio\b", r"\brimworld\b", r"\bstardew\b",
    r"\bworld of warcraft\b", r"\bwow\.exe\b", r"\bfinal fantasy\b",
    r"\belden ring\b", r"\bbaldur'?s gate\b", r"\bdark souls\b", r"\bterraria\b",
))


def looks_like_game(process_names: Iterable[str]) -> Optional[str]:
    """The first process that reads as a game, or ``None``.

    Matching is name-based and deliberately conservative: a browser tab titled
    "steam" is not a game, but a running ``steam.exe``/``dota2.exe`` is. A false
    positive here would make Astra stop reading for no reason, so the patterns
    are specific application names rather than broad keywords.
    """
    for name in process_names or ():
        text = str(name or "").strip()
        if not text:
            continue
        if any(p.search(text) for p in _GAME_PATTERNS):
            return text
    return None


# ---------------------------------------------------------------------
# Process policy: what makes the machine "busy" (configurable, shared)
# ---------------------------------------------------------------------
# Games are detected by built-in patterns. But a machine can be busy for a task
# that is not a game - a render, a build, a compile, a video call - and some
# things that *look* like a game (or a heavy app) should be ignored. Rather than
# let every background task re-implement this, the reader owns one policy object
# that says: what to treat as busy, what to ignore, what to always ignore.
#
# Order matters and is intentional: "ignore" is checked before "busy" patterns
# and the built-in game list, so an ignored name is never busy; "always ignore"
# is an absolute block (the machine's own housekeeping), checked before anything.
# A "busy" match additionally stands down on *load*; a game stands down on the
# GPU too, so the two are not the same severity.
DEFAULT_ALWAYS_IGNORED = ("openhands", "astra", "python", "pythonw")


class ProcessPolicy:
    """Which running processes mean "busy", which to ignore, and which never count.

    Pure data plus deterministic matching: given the same process list it always
    decides the same way, so the reader's wakefulness stays explainable. Nothing
    here is stored; it is rebuilt from config each session.
    """

    def __init__(self, *, busy: Iterable[str] = (), ignore: Iterable[str] = (),
                 always_ignore: Iterable[str] = DEFAULT_ALWAYS_IGNORED) -> None:
        self.busy = _compile_patterns(busy)
        self.ignore = _compile_patterns(ignore)
        self.always_ignore = _compile_patterns(always_ignore)

    @classmethod
    def from_config(cls, config: Any) -> "ProcessPolicy":
        """Build from a ``background_processes`` config mapping (tolerant of junk)."""
        cfg = config if isinstance(config, dict) else {}
        return cls(
            busy=cfg.get("busy") or (),
            ignore=cfg.get("ignore") or (),
            always_ignore=cfg.get("always_ignore", DEFAULT_ALWAYS_IGNORED),
        )

    def _ignored(self, text: str) -> bool:
        return any(p.search(text) for p in (*self.always_ignore, *self.ignore))

    def classify(self, process_names: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
        """Split the running processes into ``(game, busy)``, ignoring the trusted.

        A game is a built-in pattern; a busy task is a configured pattern. An
        ignored name is skipped first, so a task you trust never blocks reading.
        A name matching both counts as a game (the higher-severity, GPU-claiming
        case). Returns ``None`` for either when nothing matched.
        """
        game: Optional[str] = None
        busy: Optional[str] = None
        for name in process_names or ():
            text = str(name or "").strip()
            if not text or self._ignored(text):
                continue
            if any(p.search(text) for p in _GAME_PATTERNS):
                game = game or text
            elif any(p.search(text) for p in self.busy):
                busy = busy or text
        return game, busy

    def busy_process(self, process_names: Iterable[str]) -> Optional[str]:
        """The first process that should make reading stand down, or ``None``.

        Convenience over :meth:`classify`: either a game or a configured busy
        task counts. Prefer ``classify`` where the two severities matter.
        """
        game, busy = self.classify(process_names)
        return game or busy

    def is_ignored(self, text: str) -> bool:
        """Whether a single process name is on either ignore list."""
        return self._ignored(str(text or "").strip())

    def as_dict(self) -> Dict[str, Any]:
        """The raw patterns, for diagnostics (never the compiled regexes)."""
        return {
            "busy": [p.pattern for p in self.busy],
            "ignore": [p.pattern for p in self.ignore],
            "always_ignore": [p.pattern for p in self.always_ignore],
        }


def _compile_patterns(patterns: Iterable[str]) -> Tuple:
    out = []
    for pattern in patterns or ():
        text = str(pattern or "").strip()
        if not text:
            continue
        try:
            out.append(re.compile(text, re.I))
        except re.error:
            # A malformed pattern is dropped rather than crashing the reader; a
            # bad config entry must never take down the conversation.
            continue
    return tuple(out)


def is_user_active(*, seconds_since_input: Optional[float],
                   idle_threshold: float = DEFAULT_IDLE_SECONDS) -> bool:
    """True when Roum is currently interacting, so the reader must yield."""
    if seconds_since_input is None:
        return False
    try:
        return float(seconds_since_input) < float(idle_threshold)
    except (TypeError, ValueError):
        return False


def machine_busy(*, cpu_load: Optional[float],
                 ceiling: float = DEFAULT_LOAD_CEILING) -> bool:
    """True when the machine is loaded enough that reading should stand down."""
    if cpu_load is None:
        return False
    try:
        return float(cpu_load) >= float(ceiling)
    except (TypeError, ValueError):
        return False


def should_read(*, enabled: bool = True, seconds_since_input: Optional[float] = None,
                game: Optional[str] = None, gpu_busy: bool = False,
                machine_busy_flag: bool = False, cpu_load: Optional[float] = None,
                idle_threshold: float = DEFAULT_IDLE_SECONDS,
                load_ceiling: float = DEFAULT_LOAD_CEILING,
                has_work: bool = True, force: bool = False,
                concentration: Optional[float] = None,
                weary: Optional[float] = None,
                frustration: Optional[float] = None) -> Tuple[bool, str]:
    """Whether the reader may run now, and the single reason for the decision.

    Pure and deterministic: given the same observable conditions it always
    returns the same answer, so the reader's wakefulness is explainable rather
    than arbitrary. A game (or an explicitly busy GPU) always wins over reading.

    Reading requires concentration: a scattered or exhausted state stands the
    reader down (``IDLE_NOT_CONCENTRATING``), because reading is not free and a
    distracted pass produces noise rather than understanding.

    ``force`` is the explicit override: Roum asked for a cycle now, so the
    courtesy conditions (idle, load, a busy task, concentration) are bypassed.
    The hard stops are kept even under force - reading switched off, a running
    game, and nothing to read - because those are about not fighting the
    machine, not about politeness.
    """
    if not enabled:
        return False, IDLE_DISABLED
    if game:
        return False, IDLE_GAME_RUNNING
    if gpu_busy:
        return False, IDLE_GAME_RUNNING
    if force:
        return (True, IDLE_FORCED) if has_work else (False, IDLE_NO_WORK)
    if is_user_active(seconds_since_input=seconds_since_input, idle_threshold=idle_threshold):
        return False, IDLE_USER_ACTIVE
    if machine_busy_flag:
        return False, IDLE_MACHINE_BUSY
    if machine_busy(cpu_load=cpu_load, ceiling=load_ceiling):
        return False, IDLE_MACHINE_BUSY
    if not concentration_ok(concentration=concentration, weary=weary,
                            frustration=frustration):
        return False, IDLE_NOT_CONCENTRATING
    if not has_work:
        return False, IDLE_NO_WORK
    return True, IDLE_OK


def pause_reason(*, enabled: bool = True, seconds_since_input: Optional[float] = None,
                 game: Optional[str] = None, gpu_busy: bool = False,
                 machine_busy_flag: bool = False, cpu_load: Optional[float] = None,
                 idle_threshold: float = DEFAULT_IDLE_SECONDS,
                 load_ceiling: float = DEFAULT_LOAD_CEILING,
                 force: bool = False, concentration: Optional[float] = None,
                 weary: Optional[float] = None,
                 frustration: Optional[float] = None) -> Optional[str]:
    """Why reading should *stop* mid-cycle, or ``None`` if it may continue.

    The same conditions as :func:`should_read`, without the "is there work"
    question: a running cycle checks this between chunks so an interaction or a
    game takes effect immediately rather than after the whole cycle.

    Under ``force`` the courtesy conditions (idle, load, a busy task,
    concentration) are ignored between chunks too - otherwise forcing against a
    busy task would start a cycle and immediately read nothing. A game is still a
    hard stop: a forced cycle never fights the GPU, it just declines to wait for
    idle.
    """
    if force:
        return IDLE_GAME_RUNNING if (game or gpu_busy) else None
    _, reason = should_read(
        enabled=enabled, seconds_since_input=seconds_since_input, game=game,
        gpu_busy=gpu_busy, machine_busy_flag=machine_busy_flag, cpu_load=cpu_load,
        idle_threshold=idle_threshold, load_ceiling=load_ceiling, has_work=True,
        concentration=concentration, weary=weary, frustration=frustration,
    )
    return None if reason == IDLE_OK else reason


def interruption_priority(reason: Optional[str]) -> int:
    """Order pause reasons so the most important one is reported."""
    order = {IDLE_DISABLED: 0, IDLE_GAME_RUNNING: 1, IDLE_USER_ACTIVE: 2,
             IDLE_MACHINE_BUSY: 3, IDLE_NOT_CONCENTRATING: 4, IDLE_NO_WORK: 5,
             IDLE_FORCED: 6}
    return order.get(reason or "", 9)


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def chunk_bounds(text: str, start: int, *, max_chars: int = DEFAULT_CHUNK_CHARS,
                 min_chars: int = DEFAULT_MIN_CHUNK_CHARS) -> Tuple[int, int]:
    """A ``[start, end)`` slice of ``text`` bounded by ``max_chars``.

    The end is nudged to the nearest sentence/paragraph break before the limit,
    so a chunk is coherent rather than cut mid-sentence. A trailing fragment
    shorter than ``min_chars`` is folded into this chunk, so nothing is dropped
    and the reader does not make a model call for three words.
    """
    text = str(text or "")
    length = len(text)
    start = max(0, min(int(start), length))
    if start >= length:
        return start, start
    hard_end = min(length, start + max(1, int(max_chars)))
    if hard_end >= length:
        return start, length
    window = text[start:hard_end]
    cut = -1
    for match in _SENTENCE_END.finditer(window):
        cut = match.end()
    end = start + cut if (cut > 0 and cut >= int(min_chars)) else hard_end
    # Fold a short remaining tail into this chunk so the reader never makes a
    # model call for a handful of leftover characters.
    if length - end < int(min_chars):
        return start, length
    return start, end


def chunk_count(text: str, *, max_chars: int = DEFAULT_CHUNK_CHARS) -> int:
    """How many chunks ``text`` yields under the same bounds the reader uses."""
    text = str(text or "")
    total, pos, count = len(text), 0, 0
    while pos < total:
        start, end = chunk_bounds(text, pos, max_chars=max_chars)
        if end <= start:
            break
        pos, count = end, count + 1
    return count


# ---------------------------------------------------------------------
# Extraction: what to ask the model for, and how to read its answer
# ---------------------------------------------------------------------
# The reader asks for a *bounded* structured digest, not a summary. Missing
# context must stay uncertain, so the model is told not to answer its own
# questions from general knowledge, and a question it raises becomes an open
# question rather than a fact.
READING_EXTRACTION_PROMPT = """You are helping Astra build her own notes on a work she is reading. You are given one passage from "{title}".

Rules:
- Only describe what is actually in this passage. Do NOT use outside knowledge about this work or its author.
- If something is unclear or the passage references something not shown, record it as an open question rather than guessing.
- Separate what the passage states (observation) from what you infer from it (interpretation/hypothesis).
- Keep each item short. Prefer a few real items over many thin ones.
- Also record Astra's own reaction to this passage: what she actually felt or noticed while reading it, and what she was thinking about besides the book. This is her momentary response, not a claim about who she is - do not turn it into a personality trait or a life lesson.

Return STRICT JSON only:
{{
  "entities": [{{"name": "...", "note": "..."}}],
  "events": ["what happened in this passage"],
  "relationships": ["who/what relates to whom, and how"],
  "ideas": ["a concept, theme, or argument the passage develops"],
  "observations": ["something the passage states"],
  "interpretations": ["a tentative reading of the passage"],
  "questions": ["something the passage leaves unresolved"],
  "associations": ["something this reminds you of, inside the work so far"],
  "reaction": "Astra's own brief, concrete reaction to this passage (what she felt or noticed), or empty.",
  "reaction_emotion": "one of: happy, sad, angry, tender, calm, weary, frustration, unexpected, realization, attachment, or empty",
  "reaction_intensity": 0.0,
  "reflection": "what Astra was thinking about besides the book, and how it connects to how she feels, or empty."
}}

{context}Passage:
{passage}
"""


def reading_prompt(title: str, passage: str, *, context: str = "") -> str:
    """Fill :data:`READING_EXTRACTION_PROMPT` for one passage.

    ``context`` is the carried-over resume context (what happened just before),
    so the model reads a chunk *in sequence* rather than as an isolated excerpt.
    A blank ``context`` still renders - the prompt simply has no "so far" line.
    """
    prior = str(context or "").strip()
    if prior:
        context_block = (
            "What has happened in the work so far (for continuity only, do not "
            "repeat it as new material):\n" + prior + "\n\n"
        )
    else:
        context_block = ""
    return READING_EXTRACTION_PROMPT.format(
        title=title, passage=passage, context=context_block)

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_extraction(raw: Any) -> Dict[str, List[Any]]:
    """Coerce a model reply into the bounded digest shape.

    Tolerant on purpose: the model may wrap JSON in prose, return a bare list,
    or omit keys. Anything unusable becomes an empty list, so a bad reply costs
    one chunk rather than the whole cycle. Strings are also accepted where a
    list was asked for.
    """
    import json as _json

    empty = {key: [] for key in (
        "entities", "events", "relationships", "ideas", "observations",
        "interpretations", "questions", "associations")}
    # Scalar reaction fields: her momentary response, kept distinct from the
    # analytical lists above. They are re-materialised even when the model omits
    # them, so a bad reply costs one chunk and never crashes the reader.
    empty["reaction"] = ""
    empty["reaction_emotion"] = ""
    empty["reflection"] = ""
    empty["reaction_intensity"] = 0.0
    text = str(raw or "").strip()
    if not text:
        return empty
    match = _JSON_OBJECT.search(text)
    candidate = match.group(0) if match else text
    try:
        data = _json.loads(candidate)
    except (ValueError, TypeError):
        return empty
    if not isinstance(data, dict):
        return empty
    for key in ("entities", "events", "relationships", "ideas", "observations",
                "interpretations", "questions", "associations"):
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = [value] if value.strip() else []
        elif isinstance(value, dict):
            value = [value]
        if isinstance(value, (list, tuple)):
            cleaned = [v for v in value if v not in (None, "", [], {})]
            empty[key] = cleaned[:8]
    for key in ("reaction", "reaction_emotion", "reflection"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            empty[key] = value.strip()[:600]
    try:
        empty["reaction_intensity"] = max(
            0.0, min(1.0, float(data.get("reaction_intensity", 0.0) or 0.0))
        )
    except (TypeError, ValueError):
        pass
    return empty


def item_text(item: Any) -> str:
    if isinstance(item, dict):
        name = str(item.get("name") or "").strip()
        note = str(item.get("note") or "").strip()
        if name and note:
            return f"{name}: {note}"
        return name or note
    return str(item or "").strip()


def resume_context_from(digest: Dict[str, List[Any]]) -> str:
    """A short carry-forward so the next chunk knows where the story is.

    Built from what the passage *stated* plus the most recent event, so the
    reader has continuity without re-sending the whole work to the model.
    """
    parts: List[str] = []
    events = [item_text(e) for e in (digest.get("events") or []) if item_text(e)]
    if events:
        parts.append("Last events: " + "; ".join(events[-2:]))
    ideas = [item_text(i) for i in (digest.get("ideas") or []) if item_text(i)]
    if ideas:
        parts.append("Recent ideas: " + "; ".join(ideas[-2:]))
    return " ".join(parts)[:600]


# ---------------------------------------------------------------------
# Experience mapping (a reading event -> the affect nudge Slice 1 defines)
# ---------------------------------------------------------------------
def reading_experience_kind(*, discovered: bool = False, surprise: bool = False,
                            attached: bool = False, resolved: bool = False,
                            frustrated: bool = False, finished: bool = False,
                            emotion: str = "") -> str:
    """The experience kind that best describes how a reading step went.

    Returned as a *name*; the caller records it through the store, so the
    affective system stays grounded in events rather than in reading mood. An
    explicit ``emotion`` (from the passage reaction) takes precedence over the
    generic flags, so reading can produce an actual feeling rather than only an
    analytical note.
    """
    emotion_name = str(emotion or "").strip().casefold()
    if emotion_name:
        return emotion_name
    if finished:
        return "completed"
    if frustrated:
        return "frustration"
    if surprise:
        return "unexpected"
    if resolved:
        return "realization"
    if attached:
        return "attachment"
    if discovered:
        return "discovered"
    return "read"


def significance_for(*, difficulty: float = 0.5, surprise: float = 0.5,
                     resolved_question: bool = False,
                     finished_work: bool = False, learned: bool = False) -> float:
    """How much a reading step *matters*, beyond how intense it felt.

    Finishing a work or resolving a question outranks a merely absorbing
    passage, so two steps with the same "intensity" can have different
    long-term weight.
    """
    value = 0.35 + 0.25 * _clamp01(difficulty) + 0.15 * _clamp01(surprise)
    if resolved_question:
        value += 0.15
    if learned:
        value += 0.10
    if finished_work:
        value += 0.25
    return round(_clamp01(value), 4)


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------
def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _format_words(count: Any) -> str:
    """A word count as a plain, readable number (e.g. ``12,480 words``)."""
    try:
        return f"{int(count):,} words"
    except (TypeError, ValueError):
        return "0 words"


def render_progress(state: Any) -> str:
    state = coerce_state(state)
    title = state.get("title") or state.get("work_id") or "(untitled)"
    pct = int(round(progress(state) * 100))
    return f"{title} [{state.get('status')}] {pct}% ({_format_words(state.get('words_read'))})"


def reader_prompt_block(state: Any, *, current_condition: Optional[str] = None) -> Optional[str]:
    """What Astra is reading, as her own context - or ``None`` when nothing is.

    Presented as her own activity, never as knowledge about the work: the work's
    *content* reaches the prompt only through the ordinary work-context block,
    labelled by how well she knows it.
    """
    state = coerce_state(state)
    if not state.get("work_id") or not is_live(state):
        return None
    lines = [
        "=== ASTRA'S READING (HER OWN ACTIVITY, NOT KNOWLEDGE) ===",
        "This is what Astra has been reading in the background. It is her own "
        "activity; do not present it as something she has been told, and do not "
        "answer questions about the work from general knowledge - only from her "
        "own recorded notes, if any.",
        f"- Currently reading: {render_progress(state)}",
    ]
    # Only mention an *interruption*: "reading right now" is not news.
    if current_condition in _PAUSE_REASONS:
        lines.append(f"- She has been reading in the pauses while: {current_condition}.")
    if state.get("resume_context"):
        lines.append(f"- Where she is: {_clean(state.get('resume_context'))}")
    return "\n".join(lines)


def diagnostics(state: Any, *, enabled: bool = True, reason: str = IDLE_NO_WORK,
                game: Optional[str] = None, busy: Optional[str] = None,
                policy: Optional[Dict[str, Any]] = None,
                work_queue: Optional[List[str]] = None,
                last_cycle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The reader's whole situation: what, how far, and why it is (not) running."""
    state = coerce_state(state)
    return {
        "enabled": bool(enabled),
        "reason": reason,
        "game": game or None,
        "busy": busy or None,
        "policy": dict(policy or {}),
        "work_id": state.get("work_id") or None,
        "title": state.get("title") or None,
        "status": state.get("status"),
        "progress": progress(state),
        "units_read": state.get("units_read"),
        "total_units": state.get("total_units"),
        "words_read": state.get("words_read"),
        "total_words": state.get("total_words"),
        "chunks": state.get("chunks"),
        "resume_context": state.get("resume_context") or None,
        "last_read_at": state.get("last_read_at") or None,
        "queue": list(work_queue or []),
        "last_cycle": dict(last_cycle or {}),
    }


def format_diagnostics(state: Any, *, enabled: bool = True, reason: str = IDLE_NO_WORK,
                       game: Optional[str] = None, busy: Optional[str] = None,
                       work_queue: Optional[List[str]] = None,
                       last_cycle: Optional[Dict[str, Any]] = None) -> str:
    d = diagnostics(state, enabled=enabled, reason=reason, game=game, busy=busy,
                    work_queue=work_queue, last_cycle=last_cycle)
    lines = ["=== BACKGROUND READING ==="]
    lines.append(f"Enabled: {d['enabled']}")
    lines.append(f"State: {d['reason']}")
    if d["game"]:
        lines.append(f"Paused for game: {d['game']}")
    if d["busy"]:
        lines.append(f"Paused for busy task: {d['busy']}")
    if d["work_id"]:
        lines.append(f"Current: {d['title'] or d['work_id']} [{d['status']}]")
        total_words = int(d["total_words"] or 0)
        words_read = int(d["words_read"] or 0)
        if total_words:
            lines.append(f"Progress: {int(round((d['progress'] or 0) * 100))}% "
                         f"({_format_words(words_read)} of {_format_words(total_words)})")
        else:
            lines.append(f"Progress: {_format_words(words_read)} read")
        if d["resume_context"]:
            lines.append(f"Where she is: {d['resume_context']}")
        if d["last_read_at"]:
            lines.append(f"Last read: {d['last_read_at']}")
    else:
        lines.append("Current: (nothing queued)")
    if d["queue"]:
        lines.append(f"Queue: {', '.join(d['queue'])}")
    last = d.get("last_cycle") or {}
    if last:
        lines.append(
            f"Last cycle: read {_format_words(last.get('words'))} from "
            f"{last.get('work_id') or '?'} ({last.get('outcome') or 'ok'})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Persistence note
# ---------------------------------------------------------------------
# The reader's *position* is not a memory: it changes every chunk, and writing it
# through the store would rewrite the self model continuously and disturb the
# byte-for-byte read guarantees. It lives in the library's own state file
# (``astra.library``), which is the single source of truth for resuming. The
# *understanding* a work produces is a memory, and is written through the
# ordinary store so it decays and can be contradicted like anything else.


__all__ = [
    "WORK_READING", "WORK_PAUSED", "WORK_FINISHED", "WORK_ABANDONED",
    "WORK_STATUSES", "LIVE_WORK_STATUSES",
    "SOURCE_FORMAT_TEXT", "SOURCE_FORMAT_EPUB",
    "IDLE_OK", "IDLE_USER_ACTIVE", "IDLE_GAME_RUNNING", "IDLE_MACHINE_BUSY",
    "IDLE_NO_WORK", "IDLE_DISABLED", "IDLE_WORK_FINISHED", "IDLE_MODEL_FAILED",
    "IDLE_HELD", "IDLE_FORCED", "IDLE_NOT_CONCENTRATING",
    "DEFAULT_IDLE_SECONDS", "DEFAULT_LOAD_CEILING", "DEFAULT_CHUNK_CHARS",
    "DEFAULT_CHUNKS_PER_CYCLE", "DEFAULT_MIN_CHUNK_CHARS",
    "DEFAULT_ALWAYS_IGNORED", "ProcessPolicy",
    "WEARY_CEILING", "DEFAULT_PACE_CHUNKS",
    "MAX_PACE_CHUNKS", "LONG_ABSENCE_SECONDS", "DEFAULT_REST_SECONDS",
    "WEARY_REST_SECONDS", "concentration_ok", "reading_pace",
    "blank_state", "coerce_state", "progress", "is_live", "advance",
    "words_in_text", "words_at",
    "looks_like_game", "is_user_active", "machine_busy", "should_read",
    "pause_reason", "interruption_priority",
    "chunk_bounds", "chunk_count", "READING_EXTRACTION_PROMPT", "reading_prompt",
    "parse_extraction", "resume_context_from", "item_text",
    "reading_experience_kind", "significance_for",
    "render_progress", "reader_prompt_block", "diagnostics", "format_diagnostics",
]
