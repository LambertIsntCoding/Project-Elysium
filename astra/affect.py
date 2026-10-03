"""Astra's current experiential affect.

Where ``relational.py`` holds a *relationship-specific* preference, this module
holds the *experiential* side of Astra's state: how engaged, curious,
concentrated, frustrated, invested, or anticipatory she currently is.

The two use the same accumulator mechanics on purpose, but they are deliberately
NOT merged. A book can leave Astra absorbed without changing how she feels about
Roum, and a blunt turn from Roum can sting without making her less curious. They
are separate records in separate models, and neither reads the other.

Design constraints:

* **Derived, not asserted.** Components move only from *experiences* recorded on
  the live path, never from a prompt build. Nothing here is assigned for the
  sake of making dialogue expressive.
* **Temporary.** This is current state, not memory. It decays toward neutral and
  is never promoted to a durable disposition by itself; that stays the job of
  the existing self-memory evidence gates.
* **Behavioural, not decorative.** The state exists to change *processing* -
  attention and retrieval breadth now, persistence/abandonment once the reading
  and question systems exist. ``prompt_block`` supplies grounded state for the
  model to express in her own words; it never supplies a line to perform.
* **Minimal.** Only dimensions with a concrete behavioural consumer are kept.
  "Uncertainty" is deliberately *not* a number here: it belongs to the question
  system (Slice 2), where it can be represented as something revisable rather
  than as a vague scalar.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
_affect_config: Optional[Dict[str, Any]] = None
_config_loaded = False


def _load_affect_config() -> Dict[str, Any]:
    """Load affect configuration from YAML, with hardcoded defaults."""
    global _affect_config, _config_loaded
    if _config_loaded:
        return _affect_config or {}
    _config_loaded = True
    path = os.path.join(_CONFIG_DIR, "affect.yaml")
    if not os.path.isfile(path):
        _affect_config = {}
        return _affect_config
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            _affect_config = yaml.safe_load(fh) or {}
    except Exception:
        _affect_config = {}
    return _affect_config


def _config_section(key: str) -> Dict[str, Any]:
    """Return a top-level section from the affect config, or empty dict."""
    return _load_affect_config().get(key) or {}


def reload_affect_config() -> None:
    """Force a reload of the affect configuration. For testing.

    The derived tables (kind deltas, influence matrix, colour map) are built at
    import, so they are rebuilt here too - otherwise a reload would report new
    config while the runtime kept the old tables.
    """
    global _affect_config, _config_loaded, _KIND_DELTAS, _EMOTION_INFLUENCE, _COLOURS
    _affect_config = None
    _config_loaded = False
    _KIND_DELTAS = _build_kind_deltas()
    _EMOTION_INFLUENCE = _build_emotion_influence()
    _COLOURS = _build_colour_map()


# v1: the initial experiential-affect accumulator.
AFFECT_VERSION = 1

# Only dimensions with a behavioural consumer are represented. Adding a
# dimension without a consumer would be a number that never affects anything,
# which is exactly what this slice is meant to avoid.
AFFECT_COMPONENTS = (
    "engagement",             # willingness to keep going / select this activity
    "curiosity",              # pull toward exploring, noticing, following up
    "concentration",          # depth of processing, tolerance for difficulty
    "frustration",            # friction that pushes toward abandoning
    "emotional_investment",   # how much this matters to her
    "anticipation",           # forward pull toward an expected outcome
    # Valence/emotion dimensions. These are the ordinary emotions the brief
    # calls out as important; they are momentary states (like the rest of this
    # accumulator), not traits, and they decay toward neutral. Each is included
    # only because it has a concrete consumer: the reading gate reads
    # ``concentration`` and these valence terms to decide whether and how much
    # to read, and ``prompt_block`` surfaces them as grounded state.
    "happy",                  # pleased, delighted, amused
    "sad",                    # low, moved to sadness, grieving
    "angry",                  # irritated, indignant
    "calm",                   # settled, at ease
    "tender",                 # warm, affectionate, moved
    "weary",                  # tired, drained, needing to stop
)

# The valence/emotion dimensions, kept together so consumers can reason about
# "how she feels" without touching the processing dimensions.
EMOTION_COMPONENTS = ("happy", "sad", "angry", "calm", "tender", "weary")

# Experience kinds. Kept as plain strings so callers do not depend on internals.
EXPERIENCE_READ = "read"
EXPERIENCE_COMPLETED = "completed"
EXPERIENCE_DISCOVERED = "discovered"
EXPERIENCE_REALIZATION = "realization"
EXPERIENCE_FRUSTRATION = "frustration"
EXPERIENCE_INTERACTION = "interaction"
EXPERIENCE_UNEXPECTED = "unexpected"
EXPERIENCE_ATTACHMENT = "attachment"
# Valence-bearing experience kinds. A reading passage (or an interaction) can
# move her emotionally; these name the emotion so it reaches the accumulator
# without inventing a parallel store.
EXPERIENCE_HAPPY = "happy"
EXPERIENCE_SAD = "sad"
EXPERIENCE_ANGRY = "angry"
EXPERIENCE_TENDER = "tender"
EXPERIENCE_WEARY = "weary"
EXPERIENCE_CALM = "calm"

# Live-only event kinds are named before the delta table so it can reference
# them. ``LIVE_REPETITION`` is not an experience - nobody remembers "the chat
# was a bit dull" - but it is felt as a drop in pull and depth.
LIVE_REPETITION = "live_repetition"

# How each kind of experience moves the current state. Deliberately small: a
# single experience nudges, it does not saturate. Significance and intensity
# scale the nudge (see ``record_event``). Loaded from config/affect.yaml with
# hardcoded defaults.
_KIND_DELTAS_DEFAULTS: Dict[str, Dict[str, float]] = {
    EXPERIENCE_READ: {"engagement": 0.08, "concentration": 0.06, "curiosity": 0.04},
    EXPERIENCE_COMPLETED: {"engagement": 0.10, "anticipation": -0.08,
                           "frustration": -0.15, "emotional_investment": 0.06,
                           "happy": 0.08, "weary": -0.05},
    EXPERIENCE_DISCOVERED: {"curiosity": 0.12, "engagement": 0.06, "happy": 0.05},
    EXPERIENCE_REALIZATION: {"curiosity": 0.08, "engagement": 0.05,
                             "concentration": 0.03},
    EXPERIENCE_FRUSTRATION: {"frustration": 0.14, "engagement": -0.05,
                             "angry": 0.08, "calm": -0.04},
    EXPERIENCE_INTERACTION: {"engagement": 0.06, "emotional_investment": 0.05},
    EXPERIENCE_UNEXPECTED: {"curiosity": 0.10, "concentration": -0.05},
    EXPERIENCE_ATTACHMENT: {"emotional_investment": 0.12, "engagement": 0.04,
                            "tender": 0.10, "happy": 0.05},
    # Valence kinds. The positive ones settle frustration; sadness and anger are
    # their own states, not merely "not happy".
    EXPERIENCE_HAPPY: {"happy": 0.16, "sad": -0.08, "angry": -0.06,
                       "calm": 0.04, "engagement": 0.05},
    EXPERIENCE_SAD: {"sad": 0.16, "happy": -0.08, "tender": 0.05,
                     "emotional_investment": 0.06},
    EXPERIENCE_ANGRY: {"angry": 0.16, "calm": -0.10, "happy": -0.06,
                       "frustration": 0.06},
    EXPERIENCE_TENDER: {"tender": 0.16, "calm": 0.05, "happy": 0.05},
    EXPERIENCE_WEARY: {"weary": 0.16, "concentration": -0.12,
                       "engagement": -0.06, "calm": -0.03},
    EXPERIENCE_CALM: {"calm": 0.16, "angry": -0.08, "frustration": -0.06},
    # Not an experience: the dip from a conversation that has become repetitive.
    # It lowers pull and depth where she had any, and adds a little weariness so
    # it is still felt from neutral (the accumulator is floored at zero, so a
    # decrease-only event would otherwise be invisible). Only ever produced by
    # the live layer (see ``LIVE_REPETITION``).
    LIVE_REPETITION: {"engagement": -0.06, "curiosity": -0.04,
                      "concentration": -0.05, "weary": 0.05},
}


def _build_kind_deltas() -> Dict[str, Dict[str, float]]:
    """Build the experience deltas from config, falling back to defaults."""
    config_deltas = _config_section("experience_deltas")
    merged: Dict[str, Dict[str, float]] = {}
    all_kinds = set(_KIND_DELTAS_DEFAULTS.keys()) | set(config_deltas.keys())
    for kind in all_kinds:
        default = _KIND_DELTAS_DEFAULTS.get(kind, {})
        override = config_deltas.get(kind)
        if isinstance(override, dict):
            merged[kind] = {str(k): float(v) for k, v in override.items()}
        else:
            merged[kind] = dict(default)
    return merged


_KIND_DELTAS: Dict[str, Dict[str, float]] = _build_kind_deltas()

# The strongest base valence delta any kind carries (0.16 today). A memory's
# felt salience is normalised against it, so the most charged kind scores 1.0.
_SALIENCE_REFERENCE = 0.16


# ---------------------------------------------------------------------------
# Emotion influence matrix: how emotions change each other.
#
# When a source emotion's value changes, it ripples into target emotions:
#   target += source_delta * influence_weight * source_current_value
#
# This is loaded from config/affect.yaml. The defaults encode the natural
# relationships: happiness calms anger, sadness breeds weariness, etc.
# ---------------------------------------------------------------------------
_EMOTION_INFLUENCE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "happy":   {"calm": 0.12, "tender": 0.08, "sad": -0.15,
                "angry": -0.10, "weary": -0.06, "engagement": 0.04},
    "sad":     {"happy": -0.12, "tender": 0.10, "weary": 0.08,
                "engagement": -0.06, "calm": -0.04},
    "angry":   {"calm": -0.15, "happy": -0.10, "tender": -0.06,
                "frustration": 0.10, "engagement": -0.04},
    "calm":    {"angry": -0.12, "frustration": -0.08, "happy": 0.06,
                "concentration": 0.05, "weary": -0.04},
    "tender":  {"calm": 0.08, "happy": 0.10, "angry": -0.06,
                "emotional_investment": 0.08, "sad": -0.04},
    "weary":   {"engagement": -0.10, "concentration": -0.08,
                "happy": -0.06, "frustration": 0.06, "calm": -0.04,
                "anticipation": -0.05},
}


def _build_emotion_influence() -> Dict[str, Dict[str, float]]:
    """Build the influence matrix from config, falling back to defaults."""
    config_influence = _config_section("emotion_influence")
    merged: Dict[str, Dict[str, float]] = {}
    all_sources = set(_EMOTION_INFLUENCE_DEFAULTS.keys()) | set(config_influence.keys())
    for source in all_sources:
        default = _EMOTION_INFLUENCE_DEFAULTS.get(source, {})
        override = config_influence.get(source)
        if isinstance(override, dict):
            merged[source] = {str(k): float(v) for k, v in override.items()}
        else:
            merged[source] = dict(default)
    return merged


_EMOTION_INFLUENCE: Dict[str, Dict[str, float]] = _build_emotion_influence()


def _apply_influence(state: Dict[str, Any], source: str,
                     delta: float) -> None:
    """Let one emotion's change ripple into other emotions.

    The magnitude of the ripple is ``delta * weight * source_current_value``,
    so a strong emotion has more influence than a faint one.  Only one level
    of cascading is applied (no recursive chain reactions).
    """
    targets = _EMOTION_INFLUENCE.get(source)
    if not targets:
        return
    current = _clamp(state.get(source))
    for target, weight in targets.items():
        if target in state:
            ripple = delta * weight * (0.3 + 0.7 * current)
            _bump(state, target, ripple)

# The kinds below can also be *felt directly*, without a durable experience: an
# ordinary live turn may carry an affective charge - humour, a sharp remark, a
# warm acknowledgement - that moves the current condition but is not, on its
# own, an event worth remembering. They are aliases onto existing experience
# kinds so the live semantic layer and the reading/experience layer share one
# arithmetic (there is no second delta table).
LIVE_HUMOUR = EXPERIENCE_HAPPY
LIVE_INTEREST = EXPERIENCE_DISCOVERED
LIVE_POSITIVE = EXPERIENCE_INTERACTION
LIVE_CRITICISM = EXPERIENCE_FRUSTRATION
LIVE_FAILURE = EXPERIENCE_FRUSTRATION
LIVE_SURPRISE = EXPERIENCE_UNEXPECTED
LIVE_DISCLOSURE = EXPERIENCE_TENDER

# The live layer is deliberately conservative: a turn must carry a real signal
# to move anything at all, and even then it only *nudges* the current condition.
# These weights scale a live event below a full experience (the application is
# the final authority; the model never mutates affect). Configurable via
# config/affect.yaml live_settings.
def _live_setting(key: str, default: float) -> float:
    """Read a live setting from config, falling back to the default."""
    val = _config_section("live_settings").get(key)
    return float(val) if val is not None else default


LIVE_EVENT_MIN_SCORE = _live_setting("event_min_score", 0.6)
LIVE_INTENSITY_CAP = _live_setting("intensity_cap", 0.7)
LIVE_SIGNIFICANCE = _live_setting("significance", 0.45)
LIVE_NEGATIVE_SIGNIFICANCE = _live_setting("negative_significance", 0.5)

# Astra's *own* current state modulates the reaction, so the same words do not
# land identically twice. She is not a fixed classifier: an interested, engaged
# turn leans further into interest; a weary or frustrated turn is harder to
# interest and more easily pushed away; a happy turn savours a joke; a tender,
# invested turn is moved more by what Roum shares. The modulation is a single
# scalar (kept gentle) so it nudges rather than overrides the event.
LIVE_REACTIVITY_FLOOR = _live_setting("reactivity_floor", 0.75)
LIVE_REACTIVITY_CEIL = _live_setting("reactivity_ceil", 1.3)
# A real event always lands with at least this much intensity, however closed
# Astra currently is: modulation changes *how much* it moves her, never whether
# it happened. Without this, a low mood could make her miss events entirely.
LIVE_REACTIVITY_MIN_INTENSITY = _live_setting("reactivity_min_intensity", 0.35)
# What "opening up to" versus "closing down to" each event kind looks like, read
# from her current components. Each tuple is (opening, closing).
_LIVE_REACTIVITY: Dict[str, tuple] = {
    LIVE_HUMOUR: (("happy", "calm"), ("sad", "weary", "angry")),
    LIVE_INTEREST: (("curiosity", "engagement"), ("weary", "frustration", "sad")),
    LIVE_POSITIVE: (("engagement", "emotional_investment"),
                    ("weary", "frustration")),
    LIVE_CRITICISM: (("frustration", "angry"), ("calm",)),
    LIVE_SURPRISE: (("anticipation", "curiosity"), ("concentration",)),
    LIVE_DISCLOSURE: (("emotional_investment", "tender"), ("weary",)),
    LIVE_REPETITION: (("weary", "frustration"), ("curiosity", "engagement")),
}

# Repetition/habituation: a string of turns that echo each other (and do not
# carry a real event) slowly wears the conversation down. It is felt on the
# *third* such turn in a row, and only weakly, so an ordinary back-and-forth
# never registers. State is pure input, returned by :func:`evaluate_turn`.
LIVE_REPETITION_MIN = int(_live_setting("repetition_min", 3))
LIVE_REPETITION_SIMILARITY = _live_setting("repetition_similarity", 0.7)
LIVE_REPETITION_INTENSITY = _live_setting("repetition_intensity", 0.6)
LIVE_REPETITION_SIGNIFICANCE = _live_setting("repetition_significance", 0.4)
_LIVE_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Narrow markers for the live semantic layer. Kept as a few precise signals, not
# a "giant keyword list": each detects a structural feature of the turn (an
# actual question, an explicit evaluation, a marked emotional disclosure), and
# nothing fires on an ordinary neutral sentence.
_LIVE_QUESTION_RE = re.compile(
    r"\?|\b(?:how|why|what|when|where|which|who|whose|whether)\b", re.I)
_INTERESTING_RE = re.compile(
    r"\b(?:interesting|fascinating|curious|intriguing|strange|peculiar|subtle|"
    r"paradox|dilemma|thought experiment|counterintuitive|unusual|novel|"
    r"what if|suppose|imagine)\b", re.I)
_MEGA_QUESTION_RE = re.compile(
    r"\b(?:what if|suppose|imagine|thought experiment|hypothetical|"
    r"counterintuitive|paradox|the nature of|the meaning of|why do we|"
    r"what makes|how does .* work)\b", re.I)
_HUMOUR_RE = re.compile(
    r"\b(?:haha|hahaha|lol|lmao|joke|joking|kidding|funny|hilarious|amusing|"
    r"punchline|rofl)\b|(?:😂|🤣|😄|😆|😅)", re.I)
_SURPRISE_RE = re.compile(
    r"\b(?:surprising|surprised|unexpected|unbeliev|whoa|wow|no way|really\?|"
    r"shocking|astonish|i didn'?t expect|unforeseen|plot twist)\b", re.I)
_ACKNOWLEDGE_RE = re.compile(
    r"\b(?:thank(?:s| you)|good job|well done|nice work|appreciate (?:it|that)|"
    r"that(?:'s| is) (?:right|great|perfect|helpful)|exactly|you helped|"
    r"that worked|helpful)\b", re.I)
_PERSONAL_RE = re.compile(
    r"\b(?:i feel|i felt|i'?m feeling|i'?ve been feeling|i'?m scared|i'?m "
    r"afraid|i'?m grieving|i lost|i'?ve lost|i'?m struggling|i'?ve been "
    r"struggling|i'?m lonely|i'?m worried|it hurt|i'?m heartbroken|my "
    r"father|my mother|my partner|my friend|my brother|my sister|passed "
    r"away|funeral|i'?m ill|died|i'?m depressed|i'?m anxious)\b", re.I)
_CRITICISM_RE = re.compile(
    r"\b(?:you(?:'re| are)|that(?:'s| is| was| answer| response| reply| "
    r"summary| explanation| work| code)|this (?:is| was| answer| response| "
    r"reply| summary| explanation| work| code)|it (?:is| was)|the (?:answer| "
    r"response| reply| summary| explanation| work| code)|you (?:were|keep|"
    r"always|never))\b[^.?!]{0,40}\b(?:wrong|useless|unhelpful|not helpful|"
    r"disappoint|poor|bad|worse|incoherent|too vague|missed the point|"
    r"not what i asked|rambling|you failed)\b",
    re.I)
_MISSED_RE = re.compile(
    r"\b(?:no,? that'?s not|that'?s not (?:right|what i)|not what i "
    r"(?:asked|meant|wanted)|i already (?:said|told)|again\?|"
    r"i said|wrong again|still wrong)\b", re.I)

_KIND_ALIASES = {
    "reading": EXPERIENCE_READ,
    "finished": EXPERIENCE_COMPLETED,
    "completed": EXPERIENCE_COMPLETED,
    "found": EXPERIENCE_DISCOVERED,
    "discovery": EXPERIENCE_DISCOVERED,
    "realised": EXPERIENCE_REALIZATION,
    "realized": EXPERIENCE_REALIZATION,
    "blocked": EXPERIENCE_FRUSTRATION,
    "failed": EXPERIENCE_FRUSTRATION,
    "surprised": EXPERIENCE_UNEXPECTED,
    "attached": EXPERIENCE_ATTACHMENT,
    # Valence aliases so a reading reaction can name an emotion directly.
    "joy": EXPERIENCE_HAPPY,
    "delight": EXPERIENCE_HAPPY,
    "amused": EXPERIENCE_HAPPY,
    "sorrow": EXPERIENCE_SAD,
    "grief": EXPERIENCE_SAD,
    "moved": EXPERIENCE_SAD,
    "anger": EXPERIENCE_ANGRY,
    "irritated": EXPERIENCE_ANGRY,
    "indignant": EXPERIENCE_ANGRY,
    "warmth": EXPERIENCE_TENDER,
    "affection": EXPERIENCE_TENDER,
    "tired": EXPERIENCE_WEARY,
    "drained": EXPERIENCE_WEARY,
    "settled": EXPERIENCE_CALM,
    "peaceful": EXPERIENCE_CALM,
}

# State relaxation. Faster than the relational state on purpose: a current
# affective condition is momentary, so it should fade within days rather than
# linger for weeks. Neutral is zero - there is no "base mood" to fall back to.
# Configurable via config/affect.yaml decay_settings.
def _decay_setting(key: str, default: float) -> float:
    """Read a decay setting from config, falling back to the default."""
    val = _config_section("decay_settings").get(key)
    return float(val) if val is not None else default


AFFECT_BASELINE = _decay_setting("baseline", 0.0)
AFFECT_DECAY_PER_DAY = _decay_setting("per_day", 0.15)
AFFECT_MIN_INTERVAL_DAYS = _decay_setting("min_interval_days", 0.5)
# Below this every component is treated as "no particular condition".
AFFECT_NEUTRAL_EPSILON = _decay_setting("neutral_epsilon", 0.02)

_MIN, _MAX = 0.0, 1.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: Any, low: float = _MIN, high: float = _MAX) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(low, min(high, number))


def _parse_ts(value: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _blank_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {name: 0.0 for name in AFFECT_COMPONENTS}
    state.update({
        "version": AFFECT_VERSION,
        "event_counts": {},
        "observations": 0,
        "last_reason": "",
        "last_updated": "",
    })
    return state


def _coerce_state(raw: Any) -> Dict[str, Any]:
    state = _blank_state()
    if not isinstance(raw, dict):
        return state
    for name in AFFECT_COMPONENTS:
        if name in raw:
            state[name] = _clamp(raw.get(name))
    counts = raw.get("event_counts")
    if isinstance(counts, dict):
        state["event_counts"] = {str(k): int(v) for k, v in counts.items()
                                 if isinstance(v, (int, float))}
    state["observations"] = int(raw.get("observations", 0) or 0)
    state["last_reason"] = str(raw.get("last_reason", "") or "")
    state["last_updated"] = str(raw.get("last_updated", "") or "")
    return state


def _bump(state: Dict[str, Any], component: str, delta: float) -> None:
    state[component] = round(_clamp(_clamp(state.get(component)) + delta), 4)


def _decay(state: Dict[str, Any], now: Optional[datetime] = None) -> None:
    """Relax the current condition toward neutral as time passes.

    Applied lazily on the next experience, so state that is never touched again
    simply stays put on disk, and anything read back is aged when next used.
    """
    last = _parse_ts(state.get("last_updated"))
    now = now or datetime.now(timezone.utc)
    if last is None:
        state["last_updated"] = now.isoformat()
        return
    days = (now - last).total_seconds() / 86400.0
    if days < AFFECT_MIN_INTERVAL_DAYS:
        return
    factor = max(0.0, 1.0 - AFFECT_DECAY_PER_DAY * days)
    for name in AFFECT_COMPONENTS:
        current = _clamp(state.get(name))
        state[name] = round(AFFECT_BASELINE + (current - AFFECT_BASELINE) * factor, 4)


def is_neutral(state: Any) -> bool:
    """True when there is no particular current condition worth surfacing."""
    state = _coerce_state(state)
    return all(_clamp(state.get(name)) < AFFECT_NEUTRAL_EPSILON
               for name in AFFECT_COMPONENTS)


# ---------------------------------------------------------------------
# The emotional colour of a memory
# ---------------------------------------------------------------------
# A memory's kind is its emotional colour. ``EXPERIENCE_HAPPY`` is how a moment
# felt, and ``read``/``discovered``/``completed`` carry their own plain colour.
# Keeping the map here (rather than as a field on the record) means a colour can
# be reworded or a new synonym folded in without migrating stored data, and it
# ties the emotion to the memory at recall time rather than at write time.
_COLOUR_DEFAULTS: Dict[str, str] = {
    EXPERIENCE_READ: "absorbed",
    EXPERIENCE_DISCOVERED: "curious",
    EXPERIENCE_COMPLETED: "satisfied",
    EXPERIENCE_REALIZATION: "illuminated",
    EXPERIENCE_FRUSTRATION: "frustrated",
    EXPERIENCE_INTERACTION: "engaged",
    EXPERIENCE_UNEXPECTED: "startled",
    EXPERIENCE_ATTACHMENT: "attached",
    EXPERIENCE_HAPPY: "glad",
    EXPERIENCE_SAD: "sad",
    EXPERIENCE_ANGRY: "angry",
    EXPERIENCE_TENDER: "warm",
    EXPERIENCE_WEARY: "weary",
    EXPERIENCE_CALM: "calm",
    LIVE_REPETITION: "restless",
    # Formative-kind fallbacks, when a strong experience was recorded without a
    # specific kind. A hard one must still read as hard rather than colourless.
    "memorable": "memorable",
    "traumatic": "painful",
    "formative": "significant",
}


def _build_colour_map() -> Dict[str, str]:
    """Build the kind -> colour map from config, falling back to defaults."""
    config_colours = _config_section("memory_colours")
    merged = dict(_COLOUR_DEFAULTS)
    for kind, word in config_colours.items():
        merged[str(kind)] = str(word)
    return merged


_COLOURS: Dict[str, str] = _build_colour_map()


def memory_colour(mem: Dict[str, Any]) -> str:
    """The emotional colour of a memory, or ``""`` when it has none.

    Only ``experience`` records carry a felt colour - they are the moments that
    moved her. An experience is coloured by its ``experience_kind`` first and
    falls back to its ``formative_kind`` (a hard experience stays coloured even
    if its kind was left blank). Ordinary memories have no colour: colouring a
    preference or a fact would be exactly the invented feeling this system is
    meant to avoid.
    """
    if str(mem.get("type") or "") != "experience":
        return ""
    colour = _COLOURS.get(resolve_kind(mem.get("experience_kind")) or "", "")
    if not colour:
        colour = _COLOURS.get(
            str(mem.get("formative_kind") or "").strip().casefold(), "")
    return colour


def felt_components(mem: Dict[str, Any]) -> Dict[str, float]:
    """The valence components a memory's kind moves, for salience ordering.

    A read-only projection of the same delta table :func:`record_event` applies:
    it answers "which feelings does this memory carry, and how strongly" without
    writing anything. The kind is resolved through its alias first, so a record
    stored as ``joy`` still counts as ``happy``.
    """
    resolved = resolve_kind(mem.get("experience_kind"))
    deltas = _KIND_DELTAS.get(resolved) if resolved else None
    if not deltas:
        return {}
    return {name: float(deltas[name]) for name in EMOTION_COMPONENTS
            if name in deltas}


def felt_salience(mem: Dict[str, Any]) -> float:
    """How strongly a memory's own feeling pulls on her, as a 0..1 scalar.

    The largest valence delta its kind carries, normalised against the strongest
    base delta (``_SALIENCE_REFERENCE``) and capped. An emotionally charged
    experience scores near 1, an analytical one near 0. Used to surface the felt
    moments among her recent history; it reads, never writes, and an unknown
    kind scores 0.
    """
    components = felt_components(mem)
    if not components:
        return 0.0
    return min(1.0, max(abs(value) for value in components.values()) / _SALIENCE_REFERENCE)


def resolve_kind(kind: Any) -> Optional[str]:
    """The canonical experience kind for a name or alias, or ``None``.

    A single source of truth for "is this a kind the affect system knows". The
    alias table folds reading/emotional synonyms (``joy`` -> ``happy``) onto the
    canonical kinds; an unrecognised name resolves to ``None`` so a caller can
    fall back rather than file a record whose kind nothing can interpret.
    """
    name = str(kind or "").strip().casefold()
    if not name:
        return None
    resolved = _KIND_ALIASES.get(name, name)
    return resolved if resolved in _KIND_DELTAS else None


def colour_vocabulary() -> Dict[str, str]:
    """A copy of the kind -> colour map, for diagnostics."""
    return dict(_COLOURS)


def record_event(state: Any, kind: str, *, intensity: float = 0.5,
                 significance: float = 0.5, text: str = "") -> Dict[str, Any]:
    """Fold one experience into the current affective state.

    ``state`` may be ``None`` (start fresh) or a previously stored dict. Only a
    known experience kind moves anything; an unknown kind is a no-op, so a
    caller that drifts cannot perturb the state with arbitrary input.

    After the base delta is applied, the emotion influence matrix lets each
    valence emotion that changed ripple into the other emotions (e.g. a bump
    in ``happy`` also nudges ``calm`` up and ``sad`` down).
    """
    state = _coerce_state(state)
    _decay(state)
    resolved = resolve_kind(kind)
    deltas = _KIND_DELTAS.get(resolved) if resolved else None
    if deltas is None:
        return state

    scale = 0.6 + 0.7 * _clamp(intensity) + 0.5 * _clamp(significance)
    # Track emotion deltas for the influence matrix.
    emotion_deltas: Dict[str, float] = {}
    for component, base in deltas.items():
        scaled = base * scale
        _bump(state, component, scaled)
        if component in EMOTION_COMPONENTS:
            emotion_deltas[component] = scaled

    # Apply the influence matrix: each emotion that changed ripples into others.
    for emotion, delta in emotion_deltas.items():
        _apply_influence(state, emotion, delta)

    counts = state.setdefault("event_counts", {})
    counts[resolved] = int(counts.get(resolved, 0)) + 1
    state["observations"] = int(state.get("observations", 0)) + 1
    state["last_reason"] = text or f"Experienced something ({resolved})."
    state["last_updated"] = _now()
    return state


# ---------------------------------------------------------------------
# Live conversational affect
# ---------------------------------------------------------------------
_LIVE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "does", "for", "from", "get", "had", "has", "have", "how", "i", "if",
    "in", "is", "it", "its", "just", "like", "me", "my", "no", "not", "of",
    "on", "or", "our", "so", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "up", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
})


def _live_tokens(text: Any) -> frozenset:
    """Content tokens of a turn, for a cheap repetition check."""
    return frozenset(
        tok for tok in _LIVE_TOKEN_RE.findall(str(text or "").casefold())
        if len(tok) > 1 and tok not in _LIVE_STOPWORDS
    )


def _is_repetitive(tokens: frozenset, history: Any) -> bool:
    """True when ``tokens`` echoes enough of the recent turns to count as a rut.

    A single echo is not enough: ``LIVE_REPETITION_MIN`` turns of the same thing
    are needed, so an ordinary back-and-forth never registers.
    """
    if not tokens or not history:
        return False
    matches = 0
    for previous in list(history)[-3:]:
        prior = _live_tokens(previous)
        if prior and len(tokens & prior) / len(tokens) >= LIVE_REPETITION_SIMILARITY:
            matches += 1
    return matches >= LIVE_REPETITION_MIN - 1


def _reactivity(state: Any, kind: str) -> float:
    """How strongly Astra's *current* state lets ``kind`` land.

    A scalar in ``[LIVE_REACTIVITY_FLOOR, LIVE_REACTIVITY_CEIL]``: high when her
    current condition is already open to this event, low when she is closed to
    it. Neutral state gives 1.0, so the base intensities are the neutral-case
    reaction. This is what makes her response to the same words depend on how
    she already is, without letting her state override the event entirely.
    """
    spec = _LIVE_REACTIVITY.get(kind)
    if not spec:
        return 1.0
    opening, closing = spec
    state = _coerce_state(state)
    up = max((_clamp(state.get(name)) for name in opening), default=0.0)
    down = max((_clamp(state.get(name)) for name in closing), default=0.0)
    return max(LIVE_REACTIVITY_FLOOR,
               min(LIVE_REACTIVITY_CEIL, 1.0 + up - down))


def evaluate_turn(user_input: Any, response: Any = "", *,
                  state: Any = None, history: Any = None) -> Optional[Dict[str, Any]]:
    """Decide whether a live conversation turn carries an affective event.

    This is the semantic layer the live path was missing. It is *pure*: it reads
    the turn's text (and, optionally, Astra's current state and the recent user
    turns) and returns an event description, or ``None`` for an ordinary turn.
    It never touches the store, the model, or any state, and it is not
    authoritative - the application decides whether to apply the event (see
    ``TripleMemoryStore.apply_live_affect_event``). The generated prose is never
    treated as a claim about how Astra feels; at most a clear acknowledgement in
    it corroborates a positive turn.

    Astra's own condition modulates the reaction: ``state`` scales the event's
    intensity through :func:`_reactivity`, so the same remark lands differently
    when she is already engaged, weary, or calm. ``history`` (recent user turns)
    lets a run of echoing, eventless turns register as a mild
    :data:`LIVE_REPETITION` dip.

    At most one event is returned per turn, chosen by priority, and only when it
    clears ``LIVE_EVENT_MIN_SCORE``. A neutral or generic turn (a bare "hello",
    a plain factual statement) returns ``None`` so it cannot swing the state.

    The returned dict is ``{"kind", "intensity", "significance", "text",
    "reason"}`` where ``kind`` is one of the existing experience kinds, so the
    same arithmetic in :func:`record_event` applies.
    """
    text = str(user_input or "").strip()
    resp = str(response or "").strip()
    if not text:
        return None

    def _event(kind: str, intensity: float, reason: str,
               significance: float = LIVE_SIGNIFICANCE) -> Optional[Dict[str, Any]]:
        # The *presence* gate is state-independent: whether an event happened is
        # a property of the turn, not of how Astra feels. Her current state then
        # scales the magnitude (floored, so a real event always lands a little),
        # which is what makes the same words move her differently over time.
        base = float(intensity)
        if base < LIVE_EVENT_MIN_SCORE:
            return None
        modulated = base * _reactivity(state, kind)
        modulated = min(LIVE_INTENSITY_CAP,
                        max(LIVE_REACTIVITY_MIN_INTENSITY, modulated))
        return {
            "kind": kind,
            "intensity": round(modulated, 4),
            "significance": round(float(significance), 4),
            "text": reason,
            "reason": reason,
        }

    # Priority order: a sharp or significant turn decides the condition first;
    # a pleasant or interesting one only registers if nothing sharper did. This
    # is what keeps a criticism from being read as a warm interaction.
    if _CRITICISM_RE.search(text) or _MISSED_RE.search(text):
        return _event(LIVE_CRITICISM, 0.6,
                      "Roum criticised the response.",
                      LIVE_NEGATIVE_SIGNIFICANCE)

    if _PERSONAL_RE.search(text):
        # Something Roum shared that plainly matters to him. Astra is moved for
        # him - this is her own current state, not knowledge about Roum and not
        # a durable experience.
        return _event(LIVE_DISCLOSURE, 0.7,
                      "Roum shared something emotionally significant.",
                      LIVE_NEGATIVE_SIGNIFICANCE)

    if _SURPRISE_RE.search(text):
        return _event(LIVE_SURPRISE, 0.6, "Roum said something surprising.")

    if _HUMOUR_RE.search(text):
        return _event(LIVE_HUMOUR, 0.65, "Roum said something funny.")

    if _MEGA_QUESTION_RE.search(text):
        return _event(LIVE_INTEREST, 0.65,
                      "Roum raised something genuinely interesting.")
    if _LIVE_QUESTION_RE.search(text) and _INTERESTING_RE.search(text):
        return _event(LIVE_INTEREST, 0.6,
                      "Roum raised an interesting question.")

    if _ACKNOWLEDGE_RE.search(text) or _ACKNOWLEDGE_RE.search(resp):
        return _event(LIVE_POSITIVE, 0.6,
                      "Roum acknowledged something that worked.")

    # Nothing real happened, but the conversation may have gone in circles.
    tokens = _live_tokens(text)
    if _is_repetitive(tokens, history):
        return _event(LIVE_REPETITION, LIVE_REPETITION_INTENSITY,
                      "The conversation has become repetitive.",
                      LIVE_REPETITION_SIGNIFICANCE)

    return None


# ---------------------------------------------------------------------
# Behavioural consumers
# ---------------------------------------------------------------------
def retrieval_breadth(state: Any, base: int) -> int:
    """How many memories to retrieve, given the current condition.

    This is the first concrete way affect changes *processing* rather than
    wording: when Astra is engaged and curious she reaches for more context;
    when she is not, she stays narrow. Bounded so it can never balloon a prompt.
    """
    state = _coerce_state(state)
    extra = int(round(_clamp(state.get("engagement")) + _clamp(state.get("curiosity"))))
    return max(1, int(base) + min(2, extra))


# ---------------------------------------------------------------------
# Prompt material
# ---------------------------------------------------------------------
# Plain-language readings of each component. Astra experiences the *state*, not
# its numbers, so the prompt describes it the way she would - never as a
# measurement she could reason about (the boundary is implementation ->
# internal state -> experience -> self-interpretation).
_AFFECT_PHRASES: Dict[str, tuple] = {
    "engagement": ("strongly engaged", "engaged", "a little engaged"),
    "curiosity": ("very curious", "curious", "a little curious"),
    "concentration": ("concentrating well", "able to concentrate",
                      "concentration is uneven"),
    "frustration": ("quite frustrated", "frustrated", "a little frustrated"),
    "emotional_investment": ("deeply invested", "invested", "somewhat invested"),
    "anticipation": ("eager for something to happen", "anticipating something",
                     "mildly anticipating something"),
    # Ordinary emotions. Worded plainly and proportionally so the state is
    # available without pushing her toward melodrama.
    "happy": ("in a good mood", "a little pleased", "quietly content"),
    "sad": ("sad", "a little down", "faintly subdued"),
    "angry": ("angry", "irritated", "a little annoyed"),
    "calm": ("calm", "settled", "fairly even"),
    "tender": ("tender", "warm", "a little soft"),
    "weary": ("weary", "tired", "a bit drained"),
}


def render_summary(state: Any) -> Optional[str]:
    """A short, non-numeric description of the current condition.

    Returns ``None`` when neutral. Deliberately avoids any quantity: Astra is
    told how she seems to be doing, not what her values are.
    """
    state = _coerce_state(state)
    if is_neutral(state):
        return None
    salient = sorted(AFFECT_COMPONENTS, key=lambda n: _clamp(state.get(n)), reverse=True)
    phrases: List[str] = []
    for name in salient:
        value = _clamp(state.get(name))
        if value < AFFECT_NEUTRAL_EPSILON:
            continue
        levels = _AFFECT_PHRASES.get(name)
        if not levels:
            continue
        phrases.append(levels[0] if value >= 0.6 else
                       levels[1] if value >= 0.3 else levels[2])
    if not phrases:
        return None
    return "Right now Astra is " + ", ".join(phrases) + "."


def prompt_block(state: Any) -> Optional[str]:
    """The current condition as a *behavioural* cause, or ``None`` when neutral.

    Presented as Astra's own state - not an instruction and not a line to
    perform - so it moves how she engages rather than becoming something she
    reports. The condition is translated into tendencies (energy, patience,
    bluntness, elaboration, initiative, ...) because a state that is only named
    gets narrated; a state that is given behavioural consequences gets acted on.
    """
    summary = render_summary(state)
    if summary is None:
        return None
    return "\n".join([
        "=== ASTRA'S CURRENT CONDITION (TEMPORARY, NOT AN INSTRUCTION) ===",
        "This is Astra's present experiential state, accumulated from recent "
        "experiences. Let it shape *how* she engages - her energy, patience, "
        "warmth, bluntness, how much she elaborates, how ready she is to take "
        "the lead or to let a thought drop - rather than becoming something she "
        "says. She does not have to announce it or name these numbers; express "
        "it only if it fits.",
        f"- {summary}",
        "- These are tendencies, not verdicts, and they mix: she can be weary "
        "and still curious, engaged and still flat. Let the values interact with "
        "the moment instead of reading as one mood.",
        "- Moderate states stay moderate. A little friction is not anger, "
        "tiredness is not sadness, interest is not gushing, and caring about a "
        "subject is not attachment to Roum.",
        "- This is temporary and distinct from how she feels about Roum.",
    ])


# ---------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------
def memory_content() -> str:
    """The single self-memory record this state is persisted in."""
    return ("Astra's current experiential affect "
            "(engagement, curiosity, concentration, frustration, "
            "emotional investment, anticipation).")


def memory_tags() -> List[str]:
    return ["experiential_affect", "affect"]


def memory_keywords() -> List[str]:
    return ["affective state", "engagement", "curiosity", "current condition"]


def affect_memory_filter(mem: Dict[str, Any]) -> bool:
    """True when a stored memory is the affect accumulator (not a related one)."""
    return "experiential_affect" in (mem.get("tags") or [])


def load_state_from_memories(memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recover the current condition from the stored self-model record."""
    for mem in memories or []:
        if affect_memory_filter(mem):
            return _coerce_state(mem.get("affect_state"))
    return _blank_state()


def diagnostics(state: Any) -> Dict[str, Any]:
    """The diagnostic view: components plus why the state last moved.

    Includes the emotion influence matrix and configuration source so
    developers can verify the system is wired correctly.
    """
    state = _coerce_state(state)
    out = {name: round(_clamp(state.get(name)), 4) for name in AFFECT_COMPONENTS}
    out.update({
        "neutral": is_neutral(state),
        "event_counts": dict(state.get("event_counts", {})),
        "reason": str(state.get("last_reason", "") or ""),
        "emotion_influence": {k: dict(v) for k, v in _EMOTION_INFLUENCE.items()},
        "config_loaded": _config_loaded,
        "config_source": os.path.join(_CONFIG_DIR, "affect.yaml"),
    })
    return out


def emotion_influence_matrix() -> Dict[str, Dict[str, float]]:
    """Return a copy of the current emotion influence matrix.

    Each key is a source emotion; each value maps target emotions to weights.
    Positive = source increases target; negative = source decreases target.
    """
    return {k: dict(v) for k, v in _EMOTION_INFLUENCE.items()}


def experience_deltas() -> Dict[str, Dict[str, float]]:
    """Return a copy of the current experience delta table."""
    return {k: dict(v) for k, v in _KIND_DELTAS.items()}


__all__ = [
    "AFFECT_COMPONENTS",
    "EMOTION_COMPONENTS",
    "EXPERIENCE_READ",
    "EXPERIENCE_COMPLETED",
    "EXPERIENCE_DISCOVERED",
    "EXPERIENCE_REALIZATION",
    "EXPERIENCE_FRUSTRATION",
    "EXPERIENCE_INTERACTION",
    "EXPERIENCE_UNEXPECTED",
    "EXPERIENCE_ATTACHMENT",
    "LIVE_HUMOUR",
    "LIVE_INTEREST",
    "LIVE_POSITIVE",
    "LIVE_CRITICISM",
    "LIVE_FAILURE",
    "LIVE_SURPRISE",
    "LIVE_DISCLOSURE",
    "LIVE_REPETITION",
    "LIVE_EVENT_MIN_SCORE",
    "is_neutral",
    "record_event",
    "evaluate_turn",
    "resolve_kind",
    "memory_colour",
    "felt_components",
    "felt_salience",
    "colour_vocabulary",
    "retrieval_breadth",
    "render_summary",
    "prompt_block",
    "memory_content",
    "memory_tags",
    "memory_keywords",
    "affect_memory_filter",
    "load_state_from_memories",
    "diagnostics",
    "emotion_influence_matrix",
    "experience_deltas",
    "reload_affect_config",
]
