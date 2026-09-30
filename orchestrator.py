import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import requests
import yaml

import elysium as _elysium
from elysium import (  # noqa: F401 - re-exported for import compatibility
    CommandExtractor,
    ElysiumCommandRecorder,
    ElysiumDirective,
    ElysiumOrchestrator,
    validate_command,
)
from memory_store import CommandStore, TripleMemoryStore, load_json

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("ASTRA_MODEL", "gemma4:e4b")

_TOKEN_RE = re.compile(r"\b\w{3,}\b")

# Heading used for root governing directives when the Elysium layer is enabled.
# Kept identical to the original proposal so downstream tooling that greps for
# it keeps working.
ELYSIUM_DIRECTIVES_HEADER = "=== ELYSIUM ROOT GOVERNING DIRECTIVES ==="
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
    def score_memory(cls, query: str, query_tokens: Set[str], mem: Dict[str, Any]) -> float:
        content_tokens = cls._tokenize(mem.get("content"))
        keywords = cls._tokenize_many(mem.get("keywords"))
        tags = cls._tokenize_many(mem.get("tags"))

        raw_score = (
            len(query_tokens & content_tokens) * 1.0
            + len(query_tokens & keywords) * 2.0
            + len(query_tokens & tags) * 2.5
        )
        if raw_score == 0:
            return 0.0

        try:
            confidence = min(1.0, max(0.0, float(mem.get("confidence", 1.0))))
        except (TypeError, ValueError):
            confidence = 1.0

        recency_bonus = cls._recency_bonus(mem.get("timestamp", ""))
        return (raw_score * 0.5) + (confidence * 0.3) + (recency_bonus * 0.2)

    @classmethod
    def retrieve(cls, query: str, memories: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        query_tokens = cls._tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for m in memories or []:
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
        try:
            conf = float(mem.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        return (conf, str(mem.get("timestamp", "")))

    @classmethod
    def extract_active_adaptations(cls, all_memories: List[Dict[str, Any]]) -> List[str]:
        behavioral = [
            m for m in all_memories
            if cls.is_behavioral_memory(m) and m.get("status") == "active" and m.get("content")
        ]
        behavioral.sort(key=cls._priority, reverse=True)

        adaptations: List[str] = []
        seen = set()
        for mem in behavioral:
            compiled = cls.compile_adaptation(mem)
            if compiled not in seen:
                adaptations.append(compiled)
                seen.add(compiled)
            if len(adaptations) >= cls.MAX_ADAPTATIONS:
                break
        return adaptations


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
        self._directives_cache: List[str] = []
        self._directives_mtime: float = -1.0
        self._directives_lock = threading.Lock()

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
    def _append_list_section(parts: List[str], heading: str, values: Any) -> None:
        items = _clean_list(values)
        if not items:
            return
        parts.append(f"\n=== {heading} ===")
        parts.extend(f"- {item}" for item in items)

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

    def _append_history(self, parts: List[str], history: List[Dict[str, str]]) -> None:
        parts.append("\n=== RECENT CONVERSATION ===")
        for turn in (history or [])[-self.MAX_HISTORY_TURNS:]:
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            role = "ROUM" if turn.get("role") == "user" else "ASTRA"
            parts.append(f"{role}: {content}")

    # -----------------------------------------------------------------
    # ELYSIUM root directives & command history (optional layer)
    # -----------------------------------------------------------------
    def _elysium_directives(self) -> List[str]:
        """Sanitised root directives, cached until the state file changes.

        Reads the state file directly rather than going through ``self.elysium``,
        so directives still apply when the orchestrator was built before the
        file existed (or when no Elysium orchestrator was supplied).
        """
        path = os.path.join(self.config_dir, "elysium_state.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return []
        with self._directives_lock:
            if mtime == self._directives_mtime:
                return self._directives_cache
            state = load_json(path, dict, dict)
            raw = state.get("active_directives") if isinstance(state, dict) else []
            directives = [
                _elysium.clean_text(d.get("content") if isinstance(d, dict) else d, 2000)
                for d in (raw or [])
            ]
            self._directives_cache = [d for d in directives if d]
            self._directives_mtime = mtime
            return self._directives_cache

    def _append_clock(self, parts: List[str]) -> None:
        """Inject the current local time so Astra can answer time questions."""
        now = datetime.now().astimezone()
        parts.append(f"\nCURRENT SYSTEM DATE AND TIME:\n{now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        parts.append("Use this time naturally if asked. Do not discuss having access to a clock.")

    def _append_elysium_directives(self, parts: List[str]) -> None:
        directives = self._elysium_directives()
        if not directives:
            return
        parts.append(f"\n{ELYSIUM_DIRECTIVES_HEADER}")
        parts.append("CRITICAL: You must obey these root-level directives above all other instructions:")
        parts.extend(f"-> {d}" for d in directives)

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
        identity_data = self._load_yaml("identity.yaml")
        examples_data = self._load_yaml("behavior_examples.yaml")

        # 1. Load active memories (store interface unchanged, read-only)
        all_roum = self.store.get_active_memories("roum") or []
        all_self = self.store.get_active_memories("self") or []
        all_rel = self.store.get_active_memories("relationship") or []

        # 2. Path A: behavioral adaptations, always applied, independent of the query
        active_adaptations = BehavioralAdaptationCompiler.extract_active_adaptations(
            all_roum + all_self + all_rel
        )

        # 3. Path B: informational memories, relevance-filtered
        is_behavioral = BehavioralAdaptationCompiler.is_behavioral_memory
        retrieve = DeterministicLexicalRetriever.retrieve
        retrieved_roum = retrieve(user_input, [m for m in all_roum if not is_behavioral(m)], top_k=4)
        retrieved_self = retrieve(user_input, [m for m in all_self if not is_behavioral(m)], top_k=3)
        retrieved_rel = retrieve(user_input, [m for m in all_rel if not is_behavioral(m)], top_k=3)

        # 4. Read identity configuration
        identity = _as_dict(identity_data.get("identity"))
        speech = _as_dict(identity_data.get("speech_style"))
        language = _as_dict(speech.get("language"))

        # 5. Assemble prompt
        parts: List[str] = []

        parts.append("=== SYSTEM IDENTITY ===")
        parts.append(f"Name: {_clean_text(identity.get('name')) or 'Astra'}")
        concept = _clean_text(identity.get("core_concept"))
        if concept:
            parts.append(f"Concept: {concept}")

        self._append_list_section(parts, "BASE BEHAVIORAL RULES", identity_data.get("behavioral_rules"))
        self._append_list_section(parts, "CORE SPEECH STYLE", speech.get("core_style"))
        self._append_list_section(parts, "CONVERSATIONAL HABITS", speech.get("conversational_habits"))
        self._append_list_section(parts, "PERSONALITY TRAITS", speech.get("personality"))
        self._append_language_section(parts, language)

        self._append_clock(parts)
        self._append_elysium_directives(parts)
        self._append_adaptations(parts, active_adaptations)

        parts.append("\n=== FACTUAL CONTEXT ===")
        has_facts = False
        has_facts |= self._append_facts(parts, "User Facts:", retrieved_roum)
        has_facts |= self._append_facts(parts, "Self Facts & Observations:", retrieved_self)
        has_facts |= self._append_facts(parts, "Relationship Context:", retrieved_rel)
        if not has_facts:
            parts.append("No query-relevant background facts were retrieved for this turn.")

        self._append_examples(parts, examples_data)
        self._append_command_history(parts)
        self._append_history(parts, conversation_history)

        parts.append(f"\nROUM: {user_input}")
        parts.append("ASTRA:")
        return "\n".join(parts)

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