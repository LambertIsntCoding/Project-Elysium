import json
import requests
from typing import Dict, Any, List
from memory_store import TripleMemoryStore, VALID_MEMORY_TYPES, VALID_TARGET_MODELS

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e4b"

CONSOLIDATION_PROMPT = """
Analyze this conversation turn between Roum (User) and Astra (AI Companion).
Extract any persistent facts, explicit preferences, self-model observations, or relationship milestones.

CLASSIFICATION RULES:
1. 'explicit_fact', 'explicit_preference', 'explicit_project_information': ONLY if directly stated by Roum.
2. 'uncertain_inference', 'behavioral_pattern': Derived patterns or AI speculation about Roum.
3. 'self_fact', 'self_observation', 'self_belief', 'self_preference': Information Astra stated or observed about herself.
   - NOTE: 'self_preference' requires repeated evidence or explicit declaration.
4. 'relationship_event', 'relationship_observation', 'decision': Relationship developments.

Conversation Turn:
User (Roum): {user_input}
AI (Astra): {ai_response}

Output STRICT JSON ONLY matching this format:
{{
  "extracted_memories": [
    {{
      "target_model": "roum | self | relationship",
      "content": "Clear statement of fact",
      "type": "explicit_fact | explicit_preference | explicit_project_information | behavioral_pattern | uncertain_inference | self_fact | self_observation | self_belief | self_preference | relationship_event | decision",
      "confidence": 0.0 to 1.0,
      "is_explicit_user_statement": true/false,
      "tags": ["tag1", "tag2"],
      "keywords": ["key1", "key2"]
    }}
  ],
  "journal_worthy": false,
  "journal_title": "",
  "journal_observation": ""
}}
"""

def consolidate_turn(
    user_input: str,
    ai_response: str,
    store: TripleMemoryStore,
    conversation_id: str,
    turn_num: int
) -> None:
    prompt = CONSOLIDATION_PROMPT.format(user_input=user_input, ai_response=ai_response)
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
        data = json.loads(res.json().get("response", "{}"))

        for mem in data.get("extracted_memories", []):
            target = mem.get("target_model")
            m_type = mem.get("type")
            content = mem.get("content", "").strip()
            is_explicit = mem.get("is_explicit_user_statement", False)

            if not content or target not in VALID_TARGET_MODELS:
                continue

            # GUARDRAIL: AI-generated extractions MUST NOT become explicit user facts unless verified
            if not is_explicit and m_type in {"explicit_fact", "explicit_preference", "explicit_project_information"}:
                m_type = "uncertain_inference"

            if m_type not in VALID_MEMORY_TYPES:
                m_type = "uncertain_inference" if target == "roum" else "self_observation"

            source = "explicit_user_statement" if is_explicit else "ai_extraction"
            confidence = 1.0 if is_explicit else float(mem.get("confidence", 0.7))

            store.add_memory(
                target_model=target,
                content=content,
                mem_type=m_type,
                source=source,
                confidence=confidence,
                tags=mem.get("tags", []),
                keywords=mem.get("keywords", []),
                conversation_id=conversation_id,
                turn=turn_num
            )

        # Handle Journal Entry (Only if marked journal_worthy and contains content)
        if data.get("journal_worthy", False) and data.get("journal_observation"):
            store.add_journal_entry(
                title=data.get("journal_title", "Key Milestone"),
                observation=data.get("journal_observation"),
                trigger_conv=conversation_id
            )

    except Exception as e:
        # Consolidation failures should never crash the conversation loop
        pass