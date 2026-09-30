import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

import requests
import yaml

from memory_store import TripleMemoryStore

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("ASTRA_MODEL", "gemma4:e4b")

_TOKEN_RE = re.compile(r"\b\w{3,}\b")


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

    def __init__(self, store: TripleMemoryStore, config_dir: str = "./config"):
        self.store = store
        self.config_dir = os.path.abspath(config_dir)
        self._yaml_cache: Dict[str, tuple] = {}  # filename -> (mtime, data)
        self._session = requests.Session()

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

        self._append_adaptations(parts, active_adaptations)

        parts.append("\n=== FACTUAL CONTEXT ===")
        has_facts = False
        has_facts |= self._append_facts(parts, "User Facts:", retrieved_roum)
        has_facts |= self._append_facts(parts, "Self Facts & Observations:", retrieved_self)
        has_facts |= self._append_facts(parts, "Relationship Context:", retrieved_rel)
        if not has_facts:
            parts.append("No query-relevant background facts were retrieved for this turn.")

        self._append_examples(parts, examples_data)
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