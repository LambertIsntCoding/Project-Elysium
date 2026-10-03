import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import requests
import yaml

from . import elysium as _elysium
from . import affect
from . import behavior
from . import generation
from . import inquiry
from . import prepared_context
from . import reading
from . import relational
from . import selfhood
from . import temporal
from . import working_memory
from .elysium import (  # noqa: F401 - re-exported for import compatibility
    CommandExtractor,
    ElysiumCommandRecorder,
    ElysiumDirective,
    ElysiumOrchestrator,
    validate_command,
)
from .memory import (
    RELATIONSHIP_BOUNDARIES,
    CommandStore,
    TripleMemoryStore,
    effective_strength,
    is_authoritative_constraint,
    is_governing,
    is_sourced,
    memory_importance,
    rank_governing,
    select_governing,
    source_tier,
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("ASTRA_MODEL", "gemma4:e4b")

_TOKEN_RE = re.compile(r"\b\w{3,}\b")

# Elysium governing directives are answered by the application-level Elysium
# handler and are deliberately NOT injected into Astra's prompt.
_MAX_COMMAND_CONTEXT = 3


# ---------------------------------------------------------------------
# Small helpers for reading loosely-typed YAML / memory fields safely
# ---------------------------------------------------------------------
def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clean_text(value: Any) -> str:
    """Collapse whitespace/newlines into a single line; None -> ''."""
    return " ".join(str(value).split()) if value is not None else ""


def _clean_list(values: Any) -> List[str]:
    """Return non-empty cleaned strings from a list; anything else -> []."""
    if not isinstance(values, list):
        return []
    return [t for t in (_clean_text(v) for v in values) if t]


class DeterministicLexicalRetriever:
    """Rank memories by lexical relevance *and* current reliability.

    Retrieval deliberately mixes two signals so a highly relevant but weak or
    inferred memory cannot automatically outrank a slightly less similar
    explicit fact (section 6):

    * ``relevance`` - saturating lexical overlap (content/keywords/tags)
    * ``strength``  - :func:`memoryeffective_strength`
    """

    RELEVANCE_WEIGHT = 0.55
    STRENGTH_WEIGHT = 0.45
    NON_RETRIEVABLE_STATUSES = {"superseded", "archived"}

    @staticmethod
    def _tokenize(text: Any) -> Set[str]:
        return set(_TOKEN_RE.findall(str(text or "").lower()))

    @classmethod
    def _tokenize_many(cls, items: Any) -> Set[str]:
        # Tokenize each keyword/tag so multi-word entries ("machine learning")
        # can match individual query words.
        tokens: Set[str] = set()
        for item in items or []:
            tokens |= cls._tokenize(item)
        return tokens

    @staticmethod
    def _recency_bonus(timestamp: Any) -> float:
        try:
            ts = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            days_old = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0)
            return max(0.1, 1.0 - (days_old / 30.0))
        except (TypeError, ValueError):
            return 0.5

    @classmethod
    def lexical_score(cls, query_tokens: Set[str], mem: Dict[str, Any]) -> float:
        content_tokens = cls._tokenize(mem.get("content"))
        keywords = cls._tokenize_many(mem.get("keywords"))
        tags = cls._tokenize_many(mem.get("tags"))
        return (
            len(query_tokens & content_tokens) * 1.0
            + len(query_tokens & keywords) * 2.0
            + len(query_tokens & tags) * 2.5
        )

    @classmethod
    def is_retrievable(cls, mem: Dict[str, Any]) -> bool:
        """Active/weakened only; superseded and archived are never injected."""
        return mem.get("status") not in cls.NON_RETRIEVABLE_STATUSES

    @classmethod
    def score_memory(cls, query: str, query_tokens: Set[str], mem: Dict[str, Any]) -> float:
        raw = cls.lexical_score(query_tokens, mem)
        if raw <= 0:
            return 0.0
        relevance = raw / (raw + 3.0)  # saturates toward 1.0
        strength = effective_strength(mem)
        return round(relevance * cls.RELEVANCE_WEIGHT + strength * cls.STRENGTH_WEIGHT, 6)

    @classmethod
    def diagnose(cls, query: str, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Per-candidate score breakdown for the retrieval diagnostics (section 15)."""
        query_tokens = cls._tokenize(query)
        report: List[Dict[str, Any]] = []
        for mem in memories or []:
            raw = cls.lexical_score(query_tokens, mem) if query_tokens else 0.0
            report.append({
                "id": mem.get("id"),
                "content": _clean_text(mem.get("content")),
                "type": mem.get("type"),
                "source": mem.get("source"),
                "status": mem.get("status"),
                "confidence": mem.get("confidence"),
                "importance": memory_importance(mem),
                "effective_strength": round(effective_strength(mem), 4),
                "lexical": raw,
                "relevance": round(raw / (raw + 3.0), 4) if raw else 0.0,
                "score": cls.score_memory(query, query_tokens, mem),
                "contradicted": bool(mem.get("contradiction_count")),
                "superseded_by": mem.get("superseded_by"),
                "governing": is_governing(mem),
                "retrievable": cls.is_retrievable(mem),
            })
        report.sort(key=lambda r: r["score"], reverse=True)
        return report

    @classmethod
    def retrieve(cls, query: str, memories: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        query_tokens = cls._tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for m in memories or []:
            if not cls.is_retrievable(m):
                continue
            score = cls.score_memory(query, query_tokens, m)
            if score > 0.0:
                scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]


class BehavioralAdaptationCompiler:
    """
    Translates learned behavioral memories into execution directives.
    Restricted to: correction, explicit_preference, self_preference, behavioral_pattern.
    Does NOT include self_observation or factual memories.
    Does NOT modify stored memories.
    """

    BEHAVIORAL_MEMORY_TYPES = {
        "explicit_preference",
        "correction",
        "self_preference",
        "behavioral_pattern",
    }

    MAX_ADAPTATIONS = 12  # keeps the prompt from growing without bound

    _LABELS = {
        "correction": "EXECUTION DIRECTIVE (correction to apply)",
        "explicit_preference": "EXECUTION DIRECTIVE (user preference to honor)",
        "self_preference": "EXECUTION DIRECTIVE (your own preference to maintain)",
        "behavioral_pattern": "BEHAVIORAL PATTERN DIRECTIVE (maintain this pattern)",
    }

    @classmethod
    def is_behavioral_memory(cls, mem: Dict[str, Any]) -> bool:
        return mem.get("type") in cls.BEHAVIORAL_MEMORY_TYPES

    @classmethod
    def compile_adaptation(cls, mem: Dict[str, Any]) -> str:
        """Label the memory with how it should be applied; keep its wording intact."""
        content = _clean_text(mem.get("content"))
        label = cls._LABELS.get(mem.get("type", ""), "EXECUTION DIRECTIVE")
        return f"{label}: {content}"

    @classmethod
    def _priority(cls, mem: Dict[str, Any]) -> tuple:
        # Effective strength already folds in confidence, importance, recency and
        # source reliability, so a decayed or inferred preference ranks below a
        # reinforced explicit one.
        return (
            effective_strength(mem),
            source_tier(mem.get("source")),
            str(mem.get("timestamp", "")),
        )

    @classmethod
    def select_active_adaptation_memories(cls, all_memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        behavioral = [
            m for m in all_memories
            if cls.is_behavioral_memory(m) and m.get("status") == "active" and m.get("content")
        ]
        behavioral.sort(key=cls._priority, reverse=True)

        selected: List[Dict[str, Any]] = []
        seen = set()
        for mem in behavioral:
            compiled = cls.compile_adaptation(mem)
            if compiled not in seen:
                selected.append(mem)
                seen.add(compiled)
            if len(selected) >= cls.MAX_ADAPTATIONS:
                break
        return selected

    @classmethod
    def extract_active_adaptations(cls, all_memories: List[Dict[str, Any]]) -> List[str]:
        return [cls.compile_adaptation(m) for m in cls.select_active_adaptation_memories(all_memories)]


class CompanionOrchestrator:
    MAX_STYLE_EXAMPLES = 4   # set higher (or None) to include every example
    MAX_HISTORY_TURNS = 6

    def __init__(self, store: TripleMemoryStore, cmd_store: Any = None, *,
                 config_dir: str = "./config", elysium: Any = None,
                 enable_elysium_commands: bool = False):
        # Support every call form used across the project:
        #   CompanionOrchestrator(store)
        #   CompanionOrchestrator(store, config_dir="./config")
        #   CompanionOrchestrator(store, cmd_store)
        #   CompanionOrchestrator(store, cmd_store, config_dir="./config")
        if isinstance(cmd_store, (str, os.PathLike)):
            config_dir = os.fspath(cmd_store)  # legacy positional config path
            cmd_store = None
        elif cmd_store is not None and not isinstance(cmd_store, CommandStore):
            raise TypeError(
                "second argument must be a CommandStore or a config path, "
                f"got {type(cmd_store).__name__}"
            )
        self.store = store
        self.config_dir = os.path.abspath(config_dir)
        self._yaml_cache: Dict[str, tuple] = {}  # filename -> (mtime, data)
        self._session = requests.Session()
        # The background reader is injected by the session once it exists; the
        # orchestrator only ever *reads* its position for the prompt, never
        # starts or advances it (prompt building stays read-only).
        self.reader: Any = None
        # Prepared context (derived references written by the sleep scheduler) is
        # likewise only *read* here, and only when its file exists. When nothing
        # is attached, the prompt is byte-for-byte unchanged.
        self.prepared_context: Any = None
        # Conversational working memory: the middle layer between the current
        # turn and durable memory. It is updated on the live turn path (by the
        # session) and only *read* here. When no session is present, the
        # orchestrator keeps a local instance so prompt building still works.
        self.conversation = working_memory.ConversationState()

        # Per-build read cache: the same model is read from the store several
        # times while assembling one prompt, and each read deep-copies every
        # record. Caching it for the duration of a build removes the repeated
        # copies; it is cleared at the start of each public entry point, so a
        # write between turns is always seen.
        self._read_cache: Dict[Any, List[Dict[str, Any]]] = {}

        # --- Optional ELYSIUM layer -------------------------------------
        # All of this is off unless requested, so the default constructor
        # behaves exactly as before.
        self.elysium = elysium
        if self.elysium is None and os.path.exists(
            os.path.join(self.config_dir, "elysium_state.json")
        ):
            # Auto-enable only when a state file is actually present, so
            # building a prompt never fabricates a file as a side effect.
            self.elysium = _elysium.ElysiumOrchestrator(config_dir=self.config_dir)

        self.cmd_store = cmd_store
        if self.cmd_store is None and enable_elysium_commands:
            self.cmd_store = CommandStore(data_dir=self.config_dir)
        self.command_recorder = None
        if enable_elysium_commands:
            self.command_recorder = _elysium.ElysiumCommandRecorder(
                self.cmd_store, _elysium.CommandExtractor()
            )

    # -----------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.config_dir, filename)
        if not os.path.exists(path):
            return {}
        try:
            mtime = os.path.getmtime(path)
            cached = self._yaml_cache.get(filename)
            if cached and cached[0] == mtime:
                return cached[1]
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            data = _as_dict(data)
            self._yaml_cache[filename] = (mtime, data)
            return data
        except Exception as e:
            print(f"[Warning] Failed to read config file '{filename}': {e}")
            return {}

    # -----------------------------------------------------------------
    # Prompt section builders (each silently skips malformed/empty input
    # so one bad config field can't break the whole conversation)
    # -----------------------------------------------------------------
    @staticmethod
    def _append_facts(parts: List[str], heading: str, memories: List[Dict[str, Any]]) -> bool:
        facts = [t for t in (_clean_text(m.get("content")) for m in memories) if t]
        if not facts:
            return False
        parts.append(heading)
        parts.extend(f"  * {fact}" for fact in facts)
        return True

    @staticmethod
    def _sourced(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [m for m in memories if is_sourced(m)]

    @staticmethod
    def _unsourced(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [m for m in memories if not is_sourced(m)]

    @staticmethod
    def _append_unsourced_context(parts: List[str], memories: List[Dict[str, Any]]) -> None:
        """Present inferred material as provisional, never as established fact."""
        items = [t for t in (_clean_text(m.get("content")) for m in memories) if t]
        if not items:
            return
        parts.append("\n=== TENTATIVE INFERENCES (UNVERIFIED - NOT STATED BY ROUM) ===")
        parts.append(
            "These are Astra's own guesses, not things Roum said. Treat them as "
            "provisional: do not present them as fact and do not repeat them back "
            "as if Roum told you. Use one only where it is relevant, and if it "
            "matters, ask him to confirm it first."
        )
        parts.extend(f"  ~ {item}" for item in items)

    @staticmethod
    def _append_list_section(parts: List[str], heading: str, values: Any) -> None:
        items = _clean_list(values)
        if not items:
            return
        parts.append(f"\n=== {heading} ===")
        parts.extend(f"- {item}" for item in items)

    @staticmethod
    def _append_generation(parts: List[str], identity_data: Dict[str, Any],
                           active_work: str = "") -> None:
        """How Astra uses her internal state: as cause, not topic.

        Always rendered (unless switched off in ``identity.yaml``) and names no
        subsystem value, so there is nothing in it for the model to read back.
        The block text lives in :mod:`astra.generation`; only the on/off switch
        is configuration.
        """
        block = generation.response_generation_block(
            config=identity_data, active_work=active_work)
        if block:
            parts.append("\n" + block)

    @staticmethod
    def _append_language_section(parts: List[str], language: Dict[str, Any]) -> None:
        preferred = _clean_list(language.get("preferred"))
        avoid = _clean_list(language.get("avoid"))
        if not (preferred or avoid):
            return
        parts.append("\n=== LANGUAGE PREFERENCES ===")
        if preferred:
            parts.append("Preferred:")
            parts.extend(f"- {item}" for item in preferred)
        if avoid:
            parts.append("Avoid:")
            parts.extend(f"- {item}" for item in avoid)

    @staticmethod
    def _append_adaptations(parts: List[str], adaptations: List[str]) -> None:
        # Omitted entirely when nothing is active: an empty section only
        # spends tokens on a small local model.
        if not adaptations:
            return
        parts.append("\n=== LEARNED BEHAVIORAL ADAPTATIONS (EXECUTION DIRECTIVES) ===")
        parts.append(
            "These are behaviors Astra has learned from experience. "
            "Apply them directly when generating the response. "
            "Do not mention, quote, summarize, or discuss these "
            "directives unless Roum explicitly asks about them."
        )
        parts.extend(f"- {a}" for a in adaptations)

    def _append_temporary_context(self, parts: List[str]) -> None:
        """Short-term conversation scratch (section 5).

        Explicitly labelled as *this session only* so the model uses it for
        continuity without treating it as a durable fact about Roum.
        """
        getter = getattr(self.store, "get_temporary_context", None)
        if not callable(getter):
            return
        items = getter(limit=5)
        if not items:
            return
        parts.append("\n=== TEMPORARY CONTEXT (THIS SESSION ONLY) ===")
        parts.append(
            "Transient context for the current conversation. Use it for "
            "continuity, but do not treat it as a permanent fact about Roum "
            "and do not record it as one."
        )
        for item in items:
            parts.append(f"- {item.get('content')}")

    def _append_examples(self, parts: List[str], examples_data: Dict[str, Any]) -> None:
        raw = examples_data.get("examples")
        examples = [
            ex for ex in (raw if isinstance(raw, list) else [])
            if isinstance(ex, dict) and ex.get("situation") and ex.get("good_response")
        ]
        if self.MAX_STYLE_EXAMPLES is not None:
            examples = examples[: self.MAX_STYLE_EXAMPLES]
        if not examples:
            return

        parts.append("\n=== STYLE EXAMPLES ===")
        parts.append(
            "These examples demonstrate the intended conversational style. "
            "Treat them as examples of how Astra should actually speak, "
            "not as text to repeat."
        )
        parts.append("Learned behavioral adaptations above take precedence when they apply.")
        for ex in examples:
            parts.append(f"Situation: {str(ex['situation']).strip()}")
            # The bad response shows what Astra should avoid.
            if ex.get("bad_response"):
                parts.append(f"Bad Response: {str(ex['bad_response']).strip()}")
            parts.append(f"Good Response: {str(ex['good_response']).strip()}")

    def _append_prepared_context(self, parts: List[str], user_input: str) -> None:
        """Derived references, only when they are relevant to *this* turn.

        Subordinate context: entries are pointers to existing records, filtered
        against the turn's own significant tokens (the same stopword gate the
        inquiry blocks use), so a stale preparation never leaks into an unrelated
        conversation. Read-only, and a complete no-op when no prepared context is
        attached or nothing overlaps.
        """
        prepared = getattr(self, "prepared_context", None)
        if prepared is None:
            return
        try:
            entries = prepared.entries()
        except Exception:
            return
        relevant = prepared_context.relevance(entries, user_input)
        block = prepared_context.render_block(relevant)
        if block:
            parts.append("\n" + block)

    def _append_working_memory(self, parts: List[str]) -> None:
        """The conversation's working memory, if there is any.

        Rendered just before the recent conversation, and labelled explicitly as
        *this session only, not a memory*, so the model gets continuity without
        treating scratch state as durable. Read-only: ``prompt_block`` never
        mutates the state.
        """
        conversation = getattr(self, "conversation", None)
        block = conversation.prompt_block() if conversation is not None else None
        if block:
            parts.append("\n" + block)

    def _append_history(self, parts: List[str], history: List[Dict[str, str]]) -> None:
        parts.append("\n=== RECENT CONVERSATION ===")
        for turn in (history or [])[-self.MAX_HISTORY_TURNS:]:
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            role = "ROUM" if turn.get("role") == "user" else "ASTRA"
            parts.append(f"{role}: {content}")

    # -----------------------------------------------------------------
    # Runtime context (clock) & command history
    # -----------------------------------------------------------------
    def _append_clock(self, parts: List[str]) -> None:
        """Inject the current local time so Astra can answer time questions."""
        now = datetime.now().astimezone()
        parts.append(f"\nCURRENT SYSTEM DATE AND TIME:\n{now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        parts.append("Use this time naturally if asked. Do not discuss having access to a clock.")

    def _append_command_history(self, parts: List[str]) -> None:
        if self.cmd_store is None:
            return
        recent = self.cmd_store.get_last_commands_context(_MAX_COMMAND_CONTEXT)
        if not recent:
            return
        parts.append("\n=== DIRECT COMMAND HISTORY ===")
        parts.append("If the user asks about recent commands, reference this log:")
        for entry in recent:
            command = entry.get("structured_command", {}) if isinstance(entry, dict) else {}
            trigger = _elysium.clean_text(command.get("trigger"), 200)
            response = _elysium.clean_text(command.get("response"), 200)
            spoken = _elysium.clean_text(entry.get("user_input") if isinstance(entry, dict) else "", 200)
            parts.append(
                f"  * User said: '{spoken}' -> Stored rule: "
                f"When user says '{trigger}', respond '{response}'"
            )

    def capture_command(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Extract and store a conditional command from ``user_input``.

        No-op unless the orchestrator was built with
        ``enable_elysium_commands=True``.
        """
        if self.command_recorder is None:
            return None
        return self.command_recorder.capture(user_input)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def build_prompt(self, user_input: str, conversation_history: List[Dict[str, str]]) -> str:
        prompt, _ = self.build_prompt_with_diagnostics(
            user_input, conversation_history, detailed=False)
        return prompt

    def note_retrieval(self, diagnostics: Dict[str, Any]) -> None:
        """Record which memories were actually injected as *used*.

        Kept separate from ``build_prompt`` on purpose: prompt building stays
        read-only (the persistence tests rely on that), while the live
        conversation path calls this after a turn. Recording use is what keeps
        ``last_used``/``use_count`` - and therefore decay, dormancy and "used
        memories stay strong" - alive; without it every memory ages from
        creation and nothing is ever reinforced by use.
        """
        recorder = getattr(self.store, "record_use", None)
        ids = [i for i in (diagnostics.get("injected_ids") or []) if i]
        if not callable(recorder) or not ids:
            return
        try:
            recorder(ids)
        except Exception:  # pragma: no cover - defensive: never break a turn
            pass

    def build_prompt_with_diagnostics(
        self, user_input: str, conversation_history: List[Dict[str, str]],
        *, detailed: bool = True,
    ) -> tuple:
        """Assemble the prompt and return ``(prompt, diagnostics)``.

        Read-only: it never writes to the store, so callers that rely on prompt
        building being side-effect free (e.g. the persistence tests) still hold.
        The diagnostics object records, per stage, which memories existed, which
        were retrieved, and which actually reached the prompt (section 15).

        ``detailed=False`` returns the lean diagnostics (see
        :meth:`_build_diagnostics`): the prompt is byte-for-byte identical, but
        the developer-only analyses are skipped. The live turn path uses it so a
        real reply does not pay for ``/debug``-only work.
        """
        # A fresh build must see writes that happened since the last one.
        self.invalidate_read_cache()
        identity_data = self._load_yaml("identity.yaml")
        examples_data = self._load_yaml("behavior_examples.yaml")
        boundaries_cfg = self._load_yaml("relationship.yaml")

        # 1. Load every retrievable (active/weakened) memory; superseded and
        #    archived records are history and never enter a prompt.
        all_roum = self._load_memories("roum")
        all_self = self._load_memories("self")
        all_rel = self._load_memories("relationship")
        # The accumulated relational state is presented through its own block
        # (below), so its raw backing record is kept out of generic retrieval:
        # that way the preference is stated once, and only when it is earned.
        all_rel = [m for m in all_rel if not relational.affinity_memory_filter(m)]
        # The experiential-affect accumulator is likewise presented through its
        # own block, so its backing record stays out of generic retrieval.
        all_self = [m for m in all_self if not affect.affect_memory_filter(m)]
        # A claim that Astra can become biologically human, or to an experience
        # she could not have had (Roum's life, a body, a childhood), or to
        # operational/process chatter, is not knowledge about her. It is kept as
        # history but never surfaced - not as a self-fact and not as one of her
        # own recollections. Filtering here, before the experience split, is what
        # keeps it out of *every* block and stops an already-stored record from
        # reaching the model as settled self-knowledge.
        all_self = [m for m in all_self
                    if not m.get("boundary_violation") and not m.get("absent_experience")
                    and not selfhood.is_boundary_inconsistent(m.get("content"))]
        # Experiences are Astra's own history, not facts about Roum nor
        # inferences about him. They get a dedicated block rather than being
        # mislabelled under FACTUAL CONTEXT / TENTATIVE INFERENCES.
        all_experiences = [m for m in all_self if m.get("type") == "experience"]
        recent_experiences = self._recent_experiences(all_self)
        # Experiences strong enough to have stayed with her (memorable or
        # traumatic) are surfaced separately, with real elapsed time.
        formative_experiences = self._formative_experiences(all_experiences)
        all_self = [m for m in all_self if m.get("type") != "experience"]
        # Questions and work-specific knowledge are presented through their own
        # labelled blocks (below), so their raw records stay out of generic
        # retrieval - otherwise a tentative interpretation would be injected as
        # an unlabelled "fact", and an open question could be answered as if it
        # were knowledge.
        all_questions = [m for m in all_self if inquiry.is_question(m)]
        all_self = [m for m in all_self if not inquiry.is_question(m)]
        all_knowledge = [m for m in all_roum if inquiry.is_knowledge(m)]
        all_roum = [m for m in all_roum if not inquiry.is_knowledge(m)]
        everything = all_roum + all_self + all_rel

        # 2. Governing memories: always applied, regardless of the query words.
        #    One walk collects the eligible records; the count and the ranked
        #    selection are then derived from that same list instead of scanning
        #    every memory twice more (this ran three O(store) passes per turn).
        governing_eligible = [
            m for m in everything if m.get("content") and is_governing(m)
        ]
        total_governing = len(governing_eligible)
        governing = rank_governing(governing_eligible)

        # 3. Current-state slots (e.g. how to address Roum): only the newest
        #    active value per slot, so stale names are never injected.
        current_state = self._select_current_state()

        # 4. Relationship boundaries: config-driven plus built-in non-romantic
        #    baseline, always injected (sections 8-10).
        boundaries = RELATIONSHIP_BOUNDARIES + _clean_list(boundaries_cfg.get("boundaries"))

        # 5. Behavioral adaptations (execution directives), always applied.
        adaptation_memories = BehavioralAdaptationCompiler.select_active_adaptation_memories(everything)
        governing_ids = {m.get("id") for m in governing}
        adaptation_memories = [m for m in adaptation_memories if m.get("id") not in governing_ids]
        active_adaptations = [
            BehavioralAdaptationCompiler.compile_adaptation(m) for m in adaptation_memories
        ]

        # 6. Contextual retrieval: relevance-filtered, strength-weighted. The
        #    current affective condition widens (or narrows) the breadth, so
        #    affect changes *processing*, not just wording.
        affect_state = self._affect_state()
        breadth = affect.retrieval_breadth(affect_state, 4)
        is_behavioral = BehavioralAdaptationCompiler.is_behavioral_memory
        pinned = governing_ids | {m.get("id") for m in current_state}
        retrieve = DeterministicLexicalRetriever.retrieve
        # Explicit behavioural constraints are rendered once, in their own
        # authoritative block, so they are kept out of generic retrieval (as a
        # "fact" or an observation). They still live in ``everything`` so they
        # govern; only the duplicate rendering is suppressed.
        constraints = [m for m in everything if is_authoritative_constraint(m)]
        constraint_ids = {m.get("id") for m in constraints}
        retrieved_roum = retrieve(user_input, [m for m in all_roum
                                               if not is_behavioral(m)
                                               and m.get("id") not in constraint_ids], top_k=breadth)
        retrieved_self = retrieve(user_input, [m for m in all_self
                                               if not is_behavioral(m)
                                               and m.get("id") not in constraint_ids], top_k=3)
        retrieved_rel = retrieve(user_input, [m for m in all_rel
                                              if not is_behavioral(m)
                                              and m.get("id") not in constraint_ids], top_k=3)

        # 6b. Questions and work knowledge. Which questions are relevant is a
        #     relevance question, so it goes through the same retriever - a
        #     question that keeps being surfaced by related context gets more
        #     chance to appear, while an unrelated one stays out. Relevance
        #     never decides whether an interpretation is *true*, only whether it
        #     is worth raising. Nothing here resolves anything.
        active_work = self._active_work(user_input, all_knowledge)
        relevant_questions = self._select_relevant_questions(user_input, all_questions, breadth)
        work_knowledge = self._relevant_work_knowledge(
            user_input, all_knowledge, active_work=active_work)

        # 7. Read identity configuration
        identity = _as_dict(identity_data.get("identity"))
        speech = _as_dict(identity_data.get("speech_style"))
        language = _as_dict(speech.get("language"))

        # 8. Assemble prompt
        parts: List[str] = []

        # Canonical identity first: a settled, non-retrievable source of truth
        # for who is speaking and who she is talking to, rendered ahead of every
        # other block so no memory, inference, or style example can compete with
        # it. This is a source, not another stored fact.
        parts.append(selfhood.canonical_identity_block(
            name=_clean_text(identity.get("name")),
            role=_clean_text(identity.get("core_concept")),
            user_name=_clean_text(identity.get("user_name")) or "Roum",
        ))

        parts.append("\n=== SYSTEM IDENTITY ===")
        parts.append(f"Name: {_clean_text(identity.get('name')) or 'Astra'}")
        concept = _clean_text(identity.get("core_concept"))
        if concept:
            parts.append(f"Concept: {concept}")

        self._append_list_section(parts, "BASE BEHAVIORAL RULES", identity_data.get("behavioral_rules"))
        self._append_list_section(
            parts, "GROUNDED RESPONSE CONSTRAINTS (EXPLICIT, ALWAYS APPLY)",
            identity_data.get("behavioral_constraints"))
        self._append_list_section(parts, "CORE SPEECH STYLE", speech.get("core_style"))
        self._append_list_section(parts, "CONVERSATIONAL HABITS", speech.get("conversational_habits"))
        self._append_list_section(parts, "PERSONALITY TRAITS", speech.get("personality"))
        self._append_language_section(parts, language)
        # The generation contract sits with the identity material: it governs
        # how every state block below is *used*, so it must be read before them.
        # It names no subsystem value, so it adds a way to behave rather than
        # another thing to narrate. Config-gated via identity.yaml.
        self._append_generation(parts, identity_data, active_work)

        self._append_clock(parts)
        self._append_relationship_boundaries(parts, boundaries)
        # What Astra is (settled) and what she may be discovering about herself
        # (revisable) are deliberately adjacent but distinct.
        self._append_selfhood_boundary(parts)
        self._append_affect(parts, affect_state)
        self._append_reading(parts, self._reading_state())
        self._append_experiences(parts, recent_experiences)
        self._append_formative(parts, formative_experiences)
        self._append_dispositions(parts)
        self._append_questions(parts, relevant_questions)
        self._append_work_knowledge(parts, work_knowledge)
        self._append_temporal(parts, all_questions)
        self._append_impatience(parts, all_questions)
        # The behavioural modulation reads the affect and temporal state the
        # blocks above already described and turns them into *how she engages*.
        # It is the piece that makes sadness and elapsed time act on behaviour
        # rather than merely being present; it stores nothing and names no value.
        self._append_modulation(parts, affect_state, all_questions)
        self._append_current_state(parts, current_state)
        self._append_governing(parts, governing, total_governing)
        self._append_adaptations(parts, active_adaptations)
        self._append_temporary_context(parts)

        # Sourced material (things Roum actually said) is presented as fact.
        # Unsourced material (Astra's own extraction/inference) is separated and
        # explicitly marked, so it is never stated back to Roum as something he
        # said. Splitting the two is what lessens the reliance on information
        # that cannot be directly sourced.
        parts.append("\n=== FACTUAL CONTEXT ===")
        has_facts = False
        has_facts |= self._append_facts(parts, "User Facts:", self._sourced(retrieved_roum))
        has_facts |= self._append_facts(parts, "Self Facts & Observations:", self._sourced(retrieved_self))
        has_facts |= self._append_facts(parts, "Relationship Context:", self._sourced(retrieved_rel))
        if has_facts:
            parts.append(
                "These are things Roum has told you. Let the relevant ones inform "
                "the reply naturally, as background you already have - do not "
                "recite the list or announce that you are recalling it."
            )
        else:
            parts.append("No query-relevant background facts were retrieved for this turn.")

        unsourced = (self._unsourced(retrieved_roum)
                     + self._unsourced(retrieved_self)
                     + self._unsourced(retrieved_rel))
        self._append_unsourced_context(parts, unsourced)

        self._append_examples(parts, examples_data)
        self._append_command_history(parts)
        # Prepared context (derived, subordinate, off by default) comes after
        # retrieval and before the live working memory, so it can never outrank
        # a retrieved fact. It is filtered against the actual turn so a stale
        # preparation cannot leak into an unrelated conversation.
        self._append_prepared_context(parts, user_input)
        self._append_working_memory(parts)
        self._append_history(parts, conversation_history)

        parts.append(f"\nROUM: {user_input}")
        parts.append("ASTRA:")

        diagnostics = self._build_diagnostics(
            user_input, everything, governing, current_state, boundaries,
            retrieved_roum, retrieved_self, retrieved_rel, pinned, recent_experiences,
            relevant_questions, work_knowledge, prompt="\n".join(parts),
            detailed=detailed, total_governing=total_governing,
        )
        return "\n".join(parts), diagnostics

    def set_prepared_context(self, prepared: Any) -> None:
        """Attach the sleep-time prepared context (read-only at prompt time)."""
        self.prepared_context = prepared

    # ---- memory section builders ---------------------------------------
    def _load_memories(self, target_model: str) -> List[Dict[str, Any]]:
        """Retrievable memories for one model, tolerant of minimal store stubs.

        Reads the model's full set once (which also populates the retrievable
        view), so the affect/affinity helpers that need every record reuse the
        same read instead of deep-copying the model again.
        """
        if callable(getattr(self.store, "get_memories", None)):
            full = self._read(target_model, "all")
            return [m for m in full if m.get("status") in ("active", "weakened")]
        return self._read(target_model, "retrievable")

    def _store_read(self, target_model: str, mode: str) -> List[Dict[str, Any]]:
        """A single raw read from the store (no caching)."""
        if mode == "retrievable":
            getter = getattr(self.store, "get_retrievable_memories", None)
            if callable(getter):
                return list(getter(target_model) or [])
            getter = getattr(self.store, "get_active_memories", None)
            return list(getter(target_model) or []) if callable(getter) else []
        getter = getattr(self.store, "get_memories", None)
        if callable(getter):
            return list(getter(target_model, status=None) or [])
        return []

    def _read(self, target_model: str, mode: str) -> List[Dict[str, Any]]:
        """Read once per prompt build, then reuse.

        Prompt building pulls the same model from the store several times
        (affect state, affinity state, the main retrieval pass). Each store read
        deep-copies every record, so on a large store this dominates the turn.
        The cache is invalidated at the start of every public entry point, so a
        write between turns is always visible. The returned lists are private
        copies owned by the orchestrator, not by the store.

        ``retrievable`` is a filter over ``all`` (active + weakened), so when the
        full set is already loaded it is derived rather than re-read. That keeps
        it to at most one store read per model per build.
        """
        key = (target_model, mode)
        cached = self._read_cache.get(key)
        if cached is not None:
            return cached

        if mode == "retrievable":
            full = self._read_cache.get((target_model, "all"))
            if full is not None:
                data = [m for m in full
                        if m.get("status") in ("active", "weakened")]
                self._read_cache[key] = data
                return data
        data = self._store_read(target_model, mode)
        self._read_cache[key] = data
        if mode == "all":
            # Serve the retrievable view from the same read; no second copy.
            self._read_cache[(target_model, "retrievable")] = [
                m for m in data if m.get("status") in ("active", "weakened")]
        return data

    def _read_experiences(self, *, status: Optional[str] = "active") -> List[Dict[str, Any]]:
        """Experience records, read once per build from the self model.

        The self-portrait, the recent-history block and the conversation
        candidates all want experiences; without this they each deep-copy the
        whole self model. The two status views are kept apart because they are
        genuinely different sets, but each is read at most once.
        """
        key = ("experiences", status)
        cached = self._read_cache.get(key)
        if cached is not None:
            return cached
        getter = getattr(self.store, "get_memories", None)
        if callable(getter):
            # The full self set is already loaded for this build; derive the
            # experience view instead of re-reading and re-copying it.
            full = self._read("self", "all")
            data = [m for m in full if m.get("type") == "experience"]
            if status is not None:
                data = [m for m in data if (m.get("status") or "active") == status]
        else:
            exp_getter = getattr(self.store, "get_experiences", None)
            data = exp_getter(status=status) if callable(exp_getter) else []
        self._read_cache[key] = data
        return data

    def invalidate_read_cache(self) -> None:
        """Drop the per-build read cache (called on every store mutation)."""
        self._read_cache.clear()

    @staticmethod
    def _select_governing(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return select_governing(memories)

    def _select_current_state(self) -> List[Dict[str, Any]]:
        getter = getattr(self.store, "get_current_state", None)
        return getter() if callable(getter) else []

    def _append_relationship_boundaries(self, parts: List[str], boundaries: List[str]) -> None:
        if not boundaries:
            return
        parts.append("\n=== RELATIONSHIP BOUNDARIES (NON-NEGOTIABLE) ===")
        parts.append(
            "These define the relationship. They apply at all times and are not "
            "replaced by ordinary affection, closeness, or nicknames."
        )
        parts.extend(f"- {b}" for b in boundaries)
        self._append_affinity(parts)

    def _affinity_state(self) -> Dict[str, Any]:
        """Astra's accumulated relational state, read from the relationship model."""
        memories = self._read("relationship", "all")
        return relational.load_state_from_memories(memories)

    def affinity_diagnostics(self) -> Dict[str, Any]:
        """The relational preference as diagnostics (never an instruction)."""
        self.invalidate_read_cache()
        return relational.affinity_diagnostics(self._affinity_state())

    def _append_affinity(self, parts: List[str]) -> None:
        """Inject the earned preference, and only once the evidence supports it."""
        block = relational.prompt_block(self._affinity_state())
        if block:
            parts.append("\n" + block)

    def _affect_state(self) -> Dict[str, Any]:
        """Astra's current experiential affect, read from the self model."""
        memories = self._read("self", "all")
        return affect.load_state_from_memories(memories)

    def affect_diagnostics(self) -> Dict[str, Any]:
        """The current condition as diagnostics (temporary, never an instruction)."""
        return affect.diagnostics(self._affect_state())

    def questions_diagnostics(self) -> Dict[str, Any]:
        """Open lines of inquiry, with the evidence each has accumulated."""
        memories = self._read("self", "all")
        questions = [m for m in memories if inquiry.is_question(m)]
        return {
            "open": [inquiry.render_question(q) for q in inquiry.open_questions(questions)],
            "answered": [inquiry.render_question(q) for q in questions
                         if inquiry.question_status_of(q) == inquiry.Q_ANSWERED],
            "total": len(questions),
        }

    def conversation_candidates(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Reasons Astra might later want to speak - preserved, never triggered."""
        memories = self._read("self", "all")
        return inquiry.conversation_candidates(
            memories, self._read_experiences(status="active"), limit=limit)

    def _append_affect(self, parts: List[str], state: Dict[str, Any]) -> None:
        """Inject the current condition, and only when it is not neutral."""
        block = affect.prompt_block(state)
        if block:
            parts.append("\n" + block)

    # ---- background reading (read-only view) ---------------------------
    def _reading_state(self) -> Dict[str, Any]:
        """Astra's current reading position, read from the library if present."""
        reader = self.reader
        library = getattr(reader, "library", None) if reader is not None else None
        if library is None:
            return reading.blank_state("")
        try:
            return library.current_state()
        except Exception:
            return reading.blank_state("")

    def reading_diagnostics(self) -> Dict[str, Any]:
        """What the reader is doing and why - never an instruction."""
        reader = self.reader
        if reader is not None and callable(getattr(reader, "diagnostics", None)):
            try:
                return reader.diagnostics()
            except Exception:
                pass
        state = self._reading_state()
        return reading.diagnostics(state)

    def _append_reading(self, parts: List[str], state: Dict[str, Any]) -> None:
        """Inject what Astra is reading, only when she is actually reading."""
        reader = self.reader
        # Use the reader's last known reason rather than re-running the gate:
        # building a prompt must not scan processes or touch the machine.
        condition = getattr(reader, "last_reason", None) if reader is not None else None
        block = reading.reader_prompt_block(state, current_condition=condition)
        if block:
            parts.append("\n" + block)

    def _works(self) -> List[Dict[str, Any]]:
        """The works Astra has in her library, or ``[]`` when there is none."""
        library = getattr(getattr(self, "reader", None), "library", None)
        if library is not None and callable(getattr(library, "list_works", None)):
            try:
                return library.list_works()
            except Exception:
                return []
        return []

    def _append_temporal(self, parts: List[str], all_questions: List[Dict[str, Any]]) -> None:
        """How much time has passed around her, and what that time contains.

        Read-only: it reports the gap since she was last present, long-running
        questions, and long projects, but never records presence itself - that is
        the session's job on a real turn, so a prompt build cannot erase a gap.
        """
        last_seen = None
        store = self.store
        if callable(getattr(store, "last_present", None)):
            last_seen = store.last_present()
        block = temporal.temporal_prompt_block(
            last_seen=last_seen, open_questions=all_questions, works=self._works(),
        )
        if block:
            parts.append("\n" + block)

    def _append_impatience(self, parts: List[str], all_questions: List[Dict[str, Any]]) -> None:
        """The forward pull of things that have stalled, when anything has.

        Grounded in a specific unresolved thing that stopped moving - never in
        how long it has been since Roum last spoke. Read-only, like the temporal
        block, and silent unless something is genuinely stuck.
        """
        live = [q for q in all_questions if inquiry.is_open_question(q)]
        block = temporal.impatience_prompt_block(
            open_questions=live, works=self._works(),
        )
        if block:
            parts.append("\n" + block)

    def _append_modulation(self, parts: List[str], affect_state: Dict[str, Any],
                           all_questions: List[Dict[str, Any]]) -> None:
        """Turn the current affect and felt time into behavioural guidance.

        This is the bridge that makes the state act on *how* Astra engages (pace,
        initiative, brevity, hesitation) instead of only existing. Read-only: it
        reads the affect accumulator and the derived temporal view and stores
        nothing, so prompt building stays pure and byte-stable.
        """
        store = self.store
        last_seen = store.last_present() if callable(
            getattr(store, "last_present", None)) else None
        live = [q for q in all_questions if inquiry.is_open_question(q)]
        modulation = behavior.build(
            affect_state=affect_state, last_seen=last_seen,
            open_questions=live, works=self._works(),
        )
        block = behavior.prompt_block(modulation)
        if block:
            parts.append("\n" + block)

    def current_modulation(self) -> Dict[str, Any]:
        """The current behavioural modulation, for the admin/status view only.

        Read-only and external: this is the same derivation the prompt uses, so
        the admin screen can show *why* Astra is behaving a certain way without
        any of it reaching her conversational context as numbers.
        """
        all_questions = self._read("self", "all")
        live = [q for q in all_questions if inquiry.is_open_question(q)]
        store = self.store
        last_seen = store.last_present() if callable(
            getattr(store, "last_present", None)) else None
        return behavior.build(
            affect_state=self._affect_state(), last_seen=last_seen,
            open_questions=live, works=self._works(),
        )

    @staticmethod
    def _recent_experiences(all_self: List[Dict[str, Any]], limit: int = 4) -> List[Dict[str, Any]]:
        """The newest experience records, for Astra's own recent-history block."""
        experiences = [m for m in all_self if m.get("type") == "experience" and m.get("content")]
        experiences.sort(key=lambda m: str(m.get("timestamp") or ""), reverse=True)
        return experiences[:limit]

    def _append_experiences(self, parts: List[str], experiences: List[Dict[str, Any]]) -> None:
        """Astra's recent experiences, as grounded context - never as facts about Roum."""
        if not experiences:
            return
        parts.append("\n=== ASTRA'S RECENT EXPERIENCES (HER OWN HISTORY) ===")
        parts.append(
            "Things Astra has actually done or encountered. These are her own "
            "history, not claims about Roum; let the relevant ones inform how she "
            "reacts rather than listing them. The time is real elapsed time, not "
            "decoration."
        )
        for mem in experiences:
            kind = _clean_text(mem.get("experience_kind")) or "experience"
            work = _clean_text(mem.get("work_id"))
            suffix = f" (re: {work})" if work else ""
            when = selfhood.provenance_note(mem)
            parts.append(
                f"- [{kind}] {_clean_text(mem.get('content'))}{suffix} [{when}]"
            )

    @staticmethod
    def _formative_experiences(all_self: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Experiences strong enough to have stayed with her (memorable/traumatic)."""
        return [m for m in all_self
                if m.get("type") == "experience" and selfhood.is_formative(m)]

    def _append_selfhood_boundary(self, parts: List[str]) -> None:
        """Inject what Astra is: settled, non-negotiable, never an open question.

        The epistemic stance rides alongside it: the boundary is what she does
        not question, and the stance is how she questions everything else.
        """
        parts.append("\n" + selfhood.boundary_prompt_block())
        parts.append(selfhood.epistemic_prompt_block())

    def _append_formative(self, parts: List[str], experiences: List[Dict[str, Any]]) -> None:
        """Experiences that stayed with her - including traumatic ones."""
        block = selfhood.formative_block(experiences)
        if block:
            parts.append(block)

    def _append_dispositions(self, parts: List[str]) -> None:
        """A revisable self-portrait derived on the fly from her own history.

        Never stored as a memory - it is a reading of her experience records,
        so it moves as they do and carries no authority of its own. It stays
        silent until her history actually supports a pattern.
        """
        getter = getattr(self.store, "get_experiences", None)
        experiences = self._read_experiences(status=None) if callable(getter) else []
        block = selfhood.disposition_block(experiences)
        if block:
            parts.append(block)

    def _select_relevant_questions(self, user_input: str,
                                   questions: List[Dict[str, Any]],
                                   breadth: int) -> List[Dict[str, Any]]:
        """Live questions relevant to the current turn, most relevant first.

        Uses the same lexical retriever as everything else, so a question
        resurfaces when related context appears and stays quiet otherwise. The
        affect state widens the pool a little, but never decides truth.
        """
        live = [q for q in questions if inquiry.is_open_question(q)]
        if not live:
            return []
        query_tokens = inquiry.significant_tokens(user_input)
        if not query_tokens:
            # A turn with no distinctive content (e.g. "how are you") surfaces
            # only the most recent question or two, not every open question.
            return sorted(live, key=lambda m: str(m.get("timestamp") or ""), reverse=True)[:2]
        ranked = [
            q for q in live
            if query_tokens & inquiry.significant_tokens(q.get("content"))
        ]
        ranked.sort(
            key=lambda m: len(query_tokens & inquiry.significant_tokens(m.get("content"))),
            reverse=True,
        )
        ranked = ranked[:max(3, breadth)]
        # Always keep a couple of the most recent questions in view so an
        # unresolved line of inquiry is not forgotten the moment the topic
        # shifts; relevance still decides the rest.
        recent = sorted(live, key=lambda m: str(m.get("timestamp") or ""), reverse=True)[:2]
        seen, ordered = set(), []
        for mem in ranked + recent:
            if mem.get("id") not in seen:
                seen.add(mem.get("id"))
                ordered.append(mem)
        return ordered

    @staticmethod
    def _relevant_work_knowledge(user_input: str,
                                 knowledge: List[Dict[str, Any]],
                                 active_work: str = "") -> Dict[str, List[Dict[str, Any]]]:
        """Work-specific knowledge grouped by work, but only for works in play.

        A work counts as "in play" when the turn mentions it, when one of its
        records is lexically relevant, or when it is the work the conversation is
        currently *about* (``active_work``, from the working-memory state). That
        last case is what keeps a follow-up - "what does that part mean to you?"
        - anchored to the material it refers to instead of drifting to generic
        abstraction; without it, a turn with no work-specific words loses the
        work and the model fills the gap with life-lesson language.

        Grouping by work is what keeps two contexts apart: knowledge is only ever
        shown under its own work's heading, so a fact about one work can never
        leak into another - an active work pulls in *its own* records only, never
        a second work's.
        """
        if not knowledge:
            return {}
        query_tokens = inquiry.significant_tokens(user_input)
        haystack = str(user_input or "").casefold()
        active = str(active_work or "").strip().casefold()
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for mem in knowledge:
            work = inquiry.work_of(mem)
            if not work:
                continue
            mentions_work = work.casefold() in haystack
            # Significant overlap with the turn (never a bare stopword), or the
            # work named outright. This is what keeps two works apart: another
            # work's records are never pulled in on a common word alone.
            overlap = query_tokens & inquiry.significant_tokens(mem.get("content"))
            if mentions_work or overlap or (active and work.casefold() == active):
                grouped.setdefault(work, []).append(mem)
        return grouped

    def _active_work(self, user_input: str = "",
                     knowledge: Optional[List[Dict[str, Any]]] = None) -> str:
        """The work the conversation is currently about, if any.

        Read from the turn and the working-memory topic by matching a work_id or
        title, so the active work is a *deterministic* fact (a real work Astra
        has notes on, or one in her library), not a guess. Returns ``""`` when
        the conversation is not about a work.
        """
        candidates: Dict[str, str] = {}
        for mem in (knowledge or []):
            work = inquiry.work_of(mem)
            if work:
                candidates.setdefault(work.casefold(), work)
        library = getattr(getattr(self, "reader", None), "library", None)
        getter = getattr(library, "list_works", None)
        if callable(getter):
            try:
                for work in getter() or []:
                    work_id = str(work.get("work_id") or "")
                    title = str(work.get("title") or "")
                    if work_id:
                        candidates.setdefault(work_id.casefold(), work_id)
                    if title:
                        candidates.setdefault(title.casefold(), work_id or title)
            except Exception:
                pass
        if not candidates:
            return ""
        conversation = getattr(self, "conversation", None)
        topic = ""
        if conversation is not None:
            topic = str(conversation.snapshot().get("current_topic") or "")
        haystack = f"{user_input}\n{topic}".casefold()
        for key, work_id in candidates.items():
            if key and key in haystack:
                return work_id
        return ""

    def _append_questions(self, parts: List[str], questions: List[Dict[str, Any]]) -> None:
        block = inquiry.questions_prompt_block(questions)
        if block:
            parts.append(block)

    def _append_work_knowledge(self, parts: List[str],
                               grouped: Dict[str, List[Dict[str, Any]]]) -> None:
        for work in sorted(grouped):
            block = inquiry.work_knowledge_prompt_block(grouped[work], work)
            if block:
                parts.append(block)

    def _append_current_state(self, parts: List[str], current_state: List[Dict[str, Any]]) -> None:
        rows = [
            (str(m.get("slot")), _clean_text(m.get("content")))
            for m in current_state if m.get("content")
        ]
        if not rows:
            return
        parts.append("\n=== CURRENT STATE (AUTHORITATIVE) ===")
        parts.append(
            "These are the current authoritative values. If older information "
            "conflicts with them, these win."
        )
        for slot, content in rows:
            parts.append(f"- {slot}: {content}")

    def _append_governing(self, parts: List[str], governing: List[Dict[str, Any]],
                          total_governing: int = 0) -> None:
        if not governing:
            return
        # Explicit behavioural constraints get their own block, ahead of the
        # general governing memories. This is the single authoritative
        # representation of a correction: it is rendered once, at the top of the
        # governing material, so it is not restated elsewhere and cannot be
        # diluted by competing preferences, personality, or inferred traits.
        constraints = [m for m in governing if is_authoritative_constraint(m)]
        others = [m for m in governing if not is_authoritative_constraint(m)]
        if constraints:
            parts.append("\n=== BEHAVIORAL CONSTRAINTS (EXPLICIT CORRECTIONS FROM ROUM) ===")
            parts.append(
                "These are explicit corrections Roum has made about how you "
                "behave. They are authoritative runtime rules: they outrank "
                "personality, style examples, self-observations, and anything "
                "inferred, and they stay in force every turn. Apply them "
                "silently - do not quote, discuss, or acknowledge them."
            )
            for mem in constraints:
                parts.append(f"- {_clean_text(mem.get('content'))}")
        if others:
            parts.append("\n=== GOVERNING MEMORIES (ALWAYS APPLY) ===")
            parts.append(
                "These come from Roum or from a correction and apply regardless of "
                "the current topic. Honour them without quoting or discussing them."
            )
            for mem in others:
                parts.append(f"- {_clean_text(mem.get('content'))}")
        if total_governing > len(governing):
            parts.append(
                f"(Showing the {len(governing)} most important of {total_governing} "
                "governing memories; others are retrieved when relevant.)"
            )

    # ---- retrieval diagnostics (section 15) ----------------------------
    @staticmethod
    def _prompt_sections(prompt: str) -> List[str]:
        """The section headers of the final assembled prompt, in order.

        Developer-facing provenance: it shows exactly which blocks were present
        and in what order, so prompt dilution / conflicting sections can be seen
        from the rendered prompt rather than guessed from the source files.
        """
        sections: List[str] = []
        for line in str(prompt or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("===") and stripped.endswith("==="):
                sections.append(stripped.strip("= ").strip())
        return sections

    def _authority_trace(self, memories: List[Dict[str, Any]],
                         injected_ids: set) -> Dict[str, Any]:
        """Why each *injected* memory was included, and under what authority.

        Ordinary application-level provenance - source record, classification,
        authority class, work association, and the reason it was selected. It is
        for the developer only; it never reaches the prompt or Astra.

        Only injected records and every authoritative constraint are examined,
        so the trace is O(injected) rather than O(store).
        """
        from .memory import is_governing, is_authoritative_constraint
        by_id = {m.get("id"): m for m in memories if m.get("id")}
        wanted = {i for i in injected_ids if i}
        wanted.update(
            m.get("id") for m in memories
            if m.get("id") and is_authoritative_constraint(m)
        )
        trace: Dict[str, Any] = {}
        for mem_id in wanted:
            mem = by_id.get(mem_id)
            if mem is None:
                continue
            authority = self._authority_of(mem)
            included = mem_id in injected_ids
            trace[mem_id] = {
                "content": _clean_text(mem.get("content")),
                "source": mem.get("source"),
                "classification": mem.get("classification") or mem.get("type"),
                "authority": authority,
                "work_id": mem.get("work_id"),
                "governing": bool(is_governing(mem)),
                "included": included,
                "reason": ("authoritative constraint" if authority == "authoritative_constraint"
                           else "governing" if is_governing(mem)
                           else "retrieved"),
            }
        return trace

    @staticmethod
    def _authority_of(mem: Dict[str, Any]) -> str:
        """The authority class of a memory: constraint > explicit > inference."""
        from .memory import is_authoritative_constraint, source_tier
        if is_authoritative_constraint(mem):
            return "authoritative_constraint"
        source = str(mem.get("source") or "")
        tier = source_tier(source)
        if tier >= 4:
            return "explicit"
        if source in ("ai_inference", "inferred", "speculation"):
            return "inference"
        return "observation"

    def _build_diagnostics(
        self, user_input: str, everything: List[Dict[str, Any]],
        governing: List[Dict[str, Any]], current_state: List[Dict[str, Any]],
        boundaries: List[str], retrieved_roum, retrieved_self, retrieved_rel,
        pinned: set, recent_experiences: List[Dict[str, Any]],
        relevant_questions: List[Dict[str, Any]],
        work_knowledge: Dict[str, List[Dict[str, Any]]],
        prompt: str = "", detailed: bool = True, total_governing: int = 0,
    ) -> Dict[str, Any]:
        """Assemble the turn's diagnostics.

        ``detailed=True`` (the default) computes the full developer picture:
        per-candidate scores, affinity/affect breakdowns, prompt sections, and
        the authority trace. The live conversation path asks for the *lean*
        form, which keeps the fields it actually consumes (``injected_ids`` for
        ``note_retrieval``, plus the injected/retrieved id sets) and skips the
        O(store) analyses that only ``/debug`` ever reads. The returned dict
        always carries the same keys, so a consumer never sees a KeyError.
        """
        retrieved_ids = {
            m.get("id") for m in (retrieved_roum + retrieved_self + retrieved_rel)
        }
        injected_ids = (
            set(retrieved_ids)
            | {m.get("id") for m in governing}
            | {m.get("id") for m in current_state}
        )
        # Explicit behavioural constraints (authoritative runtime rules), with
        # their origin preserved so a future debug can tell "never received" from
        # "received and ignored" from "displaced during prompt construction".
        constraints = [m for m in everything if is_authoritative_constraint(m)]
        conversation = getattr(self, "conversation", None)
        diagnostics: Dict[str, Any] = {
            "query": user_input,
            "candidates": [],
            "governing_ids": [m.get("id") for m in governing],
            "total_governing": total_governing,
            "current_state_slots": {m.get("slot"): _clean_text(m.get("content")) for m in current_state},
            "boundaries": boundaries,
            "relational_affinity": {},
            "experiential_affect": {},
            "experience_ids": [m.get("id") for m in recent_experiences],
            "question_ids": [m.get("id") for m in relevant_questions],
            "work_context": {work: [m.get("id") for m in mems]
                             for work, mems in work_knowledge.items()},
            "retrieved_ids": sorted(i for i in retrieved_ids if i),
            "injected_ids": sorted(i for i in injected_ids if i),
            "omitted_ids": [],
            "non_retrievable_ids": [],
            "pinned_ids": sorted(i for i in pinned if i),
            "prompt_sections": [],
            "authority_trace": {},
            "authoritative_constraints": [
                {
                    "id": m.get("id"),
                    "content": _clean_text(m.get("content")),
                    "source": m.get("source"),
                    "included": m.get("id") in injected_ids,
                    "origin_turn": m.get("origin_turn") or m.get("turn"),
                    "intended_behavior": m.get("corrects") or m.get("intent"),
                }
                for m in constraints
            ],
            "working_memory": conversation.snapshot() if conversation is not None else {},
        }
        if not detailed:
            return diagnostics

        candidates = DeterministicLexicalRetriever.diagnose(user_input, everything)
        diagnostics.update({
            "candidates": candidates,
            "relational_affinity": self.affinity_diagnostics(),
            "experiential_affect": self.affect_diagnostics(),
            "omitted_ids": sorted(
                c["id"] for c in candidates
                if c["id"] not in injected_ids and c.get("retrievable")
            ),
            "non_retrievable_ids": sorted(
                c["id"] for c in candidates if not c.get("retrievable")
            ),
            "prompt_sections": self._prompt_sections(prompt),
            "authority_trace": self._authority_trace(everything, injected_ids),
        })
        return diagnostics

    def query_gemma(self, prompt: str) -> str:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",  # avoid reloading the model between turns
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                # Stops the model from writing Roum's next line for him.
                "stop": ["\nROUM:", "\nASTRA:"],
            },
        }
        try:
            # (connect, read): fail fast if Ollama is down, but allow a cold model load.
            res = self._session.post(OLLAMA_URL, json=payload, timeout=(5, 120))
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except Exception as e:
            return f"[Error communicating with local Gemma endpoint: {e}]"