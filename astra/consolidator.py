"""Turn a conversation turn into governed memory candidates.

This is the *policy* layer between the model and the store. The model only
proposes candidates; this module decides, deterministically, what actually
happens to each one:

* is it worth remembering at all (classification),
* should it be a durable memory, temporary context, or nothing,
* does it contradict an existing memory (and therefore supersede it),
* is it a self-memory that is not yet allowed to be durable,
* is it feedback about Astra's own behaviour (stored as a relationship/self
  observation, never as an automatic personality edit).

It deliberately keeps the heuristics here rather than in the prompt so the
system, not the generator, is the thing that decides what Astra knows.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from .memory import (
    VALID_CLASSIFICATIONS,
    VALID_MEMORY_TYPES,
    VALID_TARGET_MODELS,
    TripleMemoryStore,
    classify_candidate,
    classification_to_type,
    is_persistent_classification,
    may_supersede,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e4b"

# Feedback about Astra's conversational behaviour is a relationship/self
# observation, never a silent personality edit (section 7).
_FEEDBACK_PATTERNS = (
    "you sound", "you are being", "you're being", "be more", "be less",
    "you talk like", "you feel like", "stop sounding", "too generic",
    "generic chatbot", "sound like a", "more casual", "less formal",
    "your personality", "your tone", "talk more", "act more",
)

# Classifications whose stored type is an explicit user fact.
_EXPLICIT_CLASSIFICATIONS = {
    "persistent_user_fact", "persistent_user_preference",
    "persistent_project_information", "decision",
}


CONSOLIDATION_PROMPT = """
Analyze this conversation turn between Roum (User) and Astra (AI Companion).

For every candidate, choose exactly ONE classification:

PERSISTENT (becomes a durable memory):
- 'persistent_user_fact', 'persistent_user_preference',
  'persistent_project_information', 'behavioral_pattern',
  'uncertain_inference', 'self_fact', 'self_preference', 'self_observation',
  'self_belief', 'relationship_event', 'relationship_observation', 'decision'

TRANSIENT (NOT remembered):
- 'temporary_context' (worth understanding now, not worth remembering)
- 'no_memory'

CLASSIFICATION RULES:
1. Only classify as a persistent_* type when Roum directly stated it.
2. A one-off activity ("I played X yesterday") is 'temporary_context'.
   A durable trait ("my favourite game is X") is 'persistent_user_preference'.
   Do not turn a single statement into a 'behavioral_pattern' unless the wording
   itself clearly establishes a recurring trait ("I always...", "I never...").
3. Project information that describes current state ("stalled", "waiting on")
   is 'persistent_project_information' and is revisable, not an immutable fact.
4. 'self_preference' / 'self_fact' about Astra require repeated evidence or an
   explicit declaration by Astra's governing system. Otherwise use
   'self_observation'.
5. Questions and passing context are 'temporary_context'.

Conversation Turn:
User (Roum): {user_input}
AI (Astra): {ai_response}

Output STRICT JSON ONLY:
{{
  "candidates": [
    {{
      "classification": "one of the labels above",
      "content": "Clear statement",
      "confidence": 0.0 to 1.0,
      "is_explicit_user_statement": true/false,
      "tags": ["tag1"],
      "keywords": ["key1"]
    }}
  ],
  "journal_worthy": false,
  "journal_title": "",
  "journal_observation": ""
}}
"""


def looks_like_personality_feedback(text: str) -> bool:
    """True when ``text`` is feedback about Astra's conversational behaviour."""
    lowered = str(text or "").casefold()
    return any(p in lowered for p in _FEEDBACK_PATTERNS)


def resolve_candidate(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise a raw candidate into a governed decision, or None to drop it.

    The model's proposed classification is only a hint: the deterministic
    policy layer can downgrade it but never promote an inference to a fact.
    """
    content = str(candidate.get("content", "")).strip()
    if not content:
        return None

    classification = str(candidate.get("classification", "")).strip()
    if classification not in VALID_CLASSIFICATIONS:
        classification = "no_memory"

    explicit = bool(candidate.get("is_explicit_user_statement", False))
    proposed_type = str(candidate.get("type", "") or "")
    proposed_target = str(candidate.get("target_model", "") or "")
    if proposed_target not in VALID_TARGET_MODELS:
        proposed_target = ""
    source = "explicit_user_statement" if explicit else "ai_extraction"

    # Deterministic policy re-derivation. It is the safety net for transient
    # cues and empty content, and it reconciles a fact/preference mismatch, but
    # a well-formed persistent label is otherwise preserved.
    derived = classify_candidate(
        content, explicit=explicit,
        mem_type=proposed_type or "explicit_fact",
        target_model=proposed_target or "roum",
        source=source,
        confidence=float(candidate.get("confidence", 0.7) or 0.7),
    )
    if not is_persistent_classification(derived):
        classification = derived
    elif not is_persistent_classification(classification):
        classification = derived
    elif classification in _EXPLICIT_CLASSIFICATIONS and derived in _EXPLICIT_CLASSIFICATIONS:
        classification = derived

    if not is_persistent_classification(classification):
        return {
            "classification": classification,
            "content": content,
            "transient": True,
            "source": source,
            "tags": candidate.get("tags", []),
            "keywords": candidate.get("keywords", []),
        }

    if classification in _EXPLICIT_CLASSIFICATIONS and not explicit:
        classification = "uncertain_inference"

    target_model, mem_type = classification_to_type(classification)
    if mem_type not in VALID_MEMORY_TYPES:
        mem_type = "uncertain_inference"

    return {
        "classification": classification,
        "content": content,
        "target_model": target_model,
        "type": mem_type,
        "source": source,
        "confidence": 1.0 if explicit else float(candidate.get("confidence", 0.7) or 0.7),
        "tags": candidate.get("tags", []),
        "keywords": candidate.get("keywords", []),
        "transient": False,
    }


def apply_decision(store: TripleMemoryStore, decision: Dict[str, Any],
                   conversation_id: str, turn_num: int) -> Optional[str]:
    """Route one governed decision to the store (or to temporary context)."""
    if decision["transient"]:
        if decision["classification"] == "temporary_context":
            return store.add_temporary_context(
                decision["content"], kind="conversation",
                source=decision["source"], conversation_id=conversation_id,
            )
        return None

    target = decision["target_model"]
    content = decision["content"]

    # Contradiction -> supersede, but only an explicit statement may do so.
    if may_supersede(decision["source"], decision["classification"]):
        conflict = store.find_contradiction(target, content)
        if conflict is not None:
            return store.supersede_memory(
                target_model=target,
                old_id=conflict["id"],
                content=content,
                mem_type=decision["type"],
                source="user_correction",
                keywords=decision.get("keywords"),
                tags=decision.get("tags"),
                confidence=decision["confidence"],
                reason="Explicit user statement contradicts an existing memory",
            )

    return store.add_memory(
        target_model=target,
        content=content,
        mem_type=decision["type"],
        source=decision["source"],
        confidence=decision["confidence"],
        tags=decision.get("tags", []),
        keywords=decision.get("keywords", []),
        conversation_id=conversation_id,
        turn=turn_num,
    )


def consolidate_turn(
    user_input: str,
    ai_response: str,
    store: TripleMemoryStore,
    conversation_id: str,
    turn_num: int,
) -> None:
    """Classify, then persist/route, the memories from one conversation turn."""
    # Personality feedback is captured deterministically so it is never lost to
    # a classifier miss and never becomes an automatic personality edit.
    if looks_like_personality_feedback(user_input):
        store.add_memory(
            target_model="relationship",
            content=(
                "Roum perceives Astra's conversational style as needing adjustment: "
                f"{user_input.strip()}"
            ),
            mem_type="relationship_observation",
            source="explicit_user_statement",
            confidence=0.9,
            tags=["personality_feedback"],
            keywords=["personality", "style", "tone"],
            conversation_id=conversation_id,
            turn=turn_num,
        )

    prompt = CONSOLIDATION_PROMPT.format(user_input=user_input, ai_response=ai_response)
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json"}

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
        data = json.loads(res.json().get("response", "{}"))

        raw = data.get("candidates", data.get("extracted_memories", []))
        for candidate in raw:
            decision = resolve_candidate(candidate)
            if decision is None:
                continue
            apply_decision(store, decision, conversation_id, turn_num)

        if data.get("journal_worthy", False) and data.get("journal_observation"):
            store.add_journal_entry(
                title=data.get("journal_title", "Key Milestone"),
                observation=data.get("journal_observation"),
                trigger_conv=conversation_id,
            )
    except Exception:
        # Consolidation failures must never crash the conversation loop.
        pass
