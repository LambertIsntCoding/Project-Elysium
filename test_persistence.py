import os
import shutil
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import requests

from memory_store import TripleMemoryStore
from orchestrator import (
    CompanionOrchestrator,
    BehavioralAdaptationCompiler,
    DeterministicLexicalRetriever,
)

TEST_DIR = "./test_storage_behavior_final"
BEHAVIOR_HEADER = "=== LEARNED BEHAVIORAL ADAPTATIONS (EXECUTION DIRECTIVES) ==="
FACT_HEADER = "=== FACTUAL CONTEXT ==="


def calculate_file_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def snapshot_store_hashes(directory: str) -> dict:
    """Hash every file in the store directory, so the check doesn't depend on
    knowing each model's filename (roum, self, relationship, ...)."""
    hashes = {}
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            path = os.path.join(root, name)
            hashes[os.path.relpath(path, directory)] = calculate_file_hash(path)
    return hashes


def behavioral_section(prompt: str) -> str:
    return prompt.split(BEHAVIOR_HEADER)[1].split(FACT_HEADER)[0]


class _StubStore:
    """Minimal in-memory stand-in for unit tests that don't need disk."""
    def get_active_memories(self, model: str):
        return []


# ---------------------------------------------------------------------
# Integration tests (real store on disk)
# ---------------------------------------------------------------------
def run_integration_tests():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    # SESSION 1: learn memories & verify baseline
    store_session1 = TripleMemoryStore(data_dir=TEST_DIR)

    store_session1.add_memory(
        target_model="roum",
        content="Roum dislikes when I narrate internal processing.",
        mem_type="explicit_preference",
        source="explicit_user_statement",
    )
    store_session1.add_memory(
        target_model="self",
        content="Astra felt nervous during a conversation about emotions.",
        mem_type="self_observation",
        source="ai_extraction",
    )
    store_session1.add_memory(
        target_model="roum",
        content="Roum holds a Master degree in Computer Science.",
        mem_type="explicit_fact",
        source="explicit_user_statement",
    )

    hashes_before = snapshot_store_hashes(TEST_DIR)
    assert hashes_before, "Expected the store to have written files to disk."

    # TEST A: zero lexical overlap behavioral adaptation
    orchestrator_s1 = CompanionOrchestrator(store_session1)
    prompt_a = orchestrator_s1.build_prompt("How was your day?", [])

    assert BEHAVIOR_HEADER in prompt_a
    section_a = behavioral_section(prompt_a)
    assert "EXECUTION DIRECTIVE" in section_a
    assert "narrate internal processing" in section_a.lower()
    print("✓ Test A: behavioral memory becomes an execution directive with zero token overlap.")

    # TEST B: self_observation is NOT compiled as a behavioral rule
    assert "felt nervous" not in section_a
    print("✓ Test B: self_observation is excluded from execution directives.")

    # TEST C: byte-for-byte preservation across ALL store files
    hashes_after = snapshot_store_hashes(TEST_DIR)
    assert hashes_before == hashes_after, (
        "Store files changed while building a prompt! Differences: "
        f"{ {k for k in hashes_after if hashes_before.get(k) != hashes_after[k]} }"
    )
    print("✓ Test C: every memory file is byte-for-byte identical after prompt building.")

    # SESSION 2: cross-session persistence
    del store_session1
    del orchestrator_s1

    store_session2 = TripleMemoryStore(data_dir=TEST_DIR)
    orchestrator_s2 = CompanionOrchestrator(store_session2)
    prompt_s2 = orchestrator_s2.build_prompt("What is the capital of France?", [])

    section_s2 = behavioral_section(prompt_s2)
    assert "narrate internal processing" in section_s2.lower()
    assert "EXECUTION DIRECTIVE" in section_s2
    assert snapshot_store_hashes(TEST_DIR) == hashes_before
    print("✓ Test D: behavioral adaptation persists into a fresh session, and files are still unchanged.")


# ---------------------------------------------------------------------
# Unit tests (no disk, no Ollama)
# ---------------------------------------------------------------------
def run_unit_tests():
    # TEST E: compiler keeps wording intact instead of mangling it
    pref = {"type": "explicit_preference", "status": "active",
            "content": "Roum prefers short answers."}
    corr = {"type": "correction", "status": "active",
            "content": "I was too verbose."}
    pattern = {"type": "behavioral_pattern", "status": "active",
               "content": "Astra opens with a light joke."}

    compiled_pref = BehavioralAdaptationCompiler.compile_adaptation(pref)
    compiled_corr = BehavioralAdaptationCompiler.compile_adaptation(corr)
    compiled_pattern = BehavioralAdaptationCompiler.compile_adaptation(pattern)

    assert compiled_pref.endswith("Roum prefers short answers.")
    assert "ensure you short" not in compiled_pref
    assert compiled_corr.endswith("I was too verbose.")
    assert "Avoid I was" not in compiled_corr
    assert compiled_pattern.startswith("BEHAVIORAL PATTERN DIRECTIVE")
    print("✓ Test E: directives keep the original wording under a type label.")

    # TEST F: filtering, dedupe, ordering, cap
    memories = [
        {"type": "correction", "status": "archived", "content": "old rule", "confidence": 1.0},
        {"type": "self_observation", "status": "active", "content": "felt calm", "confidence": 1.0},
        {"type": "explicit_fact", "status": "active", "content": "likes tea", "confidence": 1.0},
        {"type": "correction", "status": "active", "content": "dup rule", "confidence": 0.5},
        {"type": "correction", "status": "active", "content": "dup rule", "confidence": 0.5},
    ]
    memories += [
        {"type": "correction", "status": "active", "content": f"rule {i}", "confidence": i / 20}
        for i in range(20)
    ]
    result = BehavioralAdaptationCompiler.extract_active_adaptations(memories)
    joined = "\n".join(result)

    assert len(result) == BehavioralAdaptationCompiler.MAX_ADAPTATIONS
    assert "old rule" not in joined and "felt calm" not in joined and "likes tea" not in joined
    assert result[0].endswith("rule 19"), "Highest-confidence rule should come first."
    assert len(result) == len(set(result)), "Duplicates should be removed."

    small = BehavioralAdaptationCompiler.extract_active_adaptations(memories[3:5])
    assert len(small) == 1, "Identical directives should collapse to one."
    print("✓ Test F: inactive/non-behavioral excluded, deduped, ranked by confidence, capped.")

    # TEST G: retriever robustness
    now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    old = {"content": "pizza night", "confidence": 1.0,
           "timestamp": (now_naive - timedelta(days=60)).isoformat()}
    fresh = {"content": "pizza night", "confidence": 1.0,
             "timestamp": now_naive.isoformat()}
    ranked = DeterministicLexicalRetriever.retrieve("pizza", [old, fresh], top_k=2)
    assert ranked[0] is fresh, "Naive timestamps should be treated as UTC and rank by recency."

    sparse = {"content": "pizza night", "keywords": None, "tags": None, "confidence": None}
    assert DeterministicLexicalRetriever.retrieve("pizza", [sparse]) == [sparse]

    multiword = {"content": "unrelated words here", "keywords": ["machine learning"]}
    assert DeterministicLexicalRetriever.retrieve("tell me about learning", [multiword]) == [multiword]

    assert DeterministicLexicalRetriever.retrieve("a I", [fresh]) == [], "No usable tokens -> no results."
    print("✓ Test G: retriever handles naive timestamps, null fields, and multi-word keywords.")

    # TEST H: query_gemma payload and error handling (mocked, no Ollama needed)
    orch = CompanionOrchestrator(_StubStore())

    ok_response = MagicMock()
    ok_response.json.return_value = {"response": "  hello there  "}
    orch._session.post = MagicMock(return_value=ok_response)

    assert orch.query_gemma("prompt") == "hello there"
    payload = orch._session.post.call_args.kwargs["json"]
    assert "\nROUM:" in payload["options"]["stop"]
    assert payload["stream"] is False

    orch._session.post = MagicMock(side_effect=requests.ConnectionError("down"))
    assert orch.query_gemma("prompt").startswith("[Error communicating")
    print("✓ Test H: query_gemma sends stop sequences and reports connection errors.")


def run_tests():
    print("=== RUNNING PERSISTENCE, INTEGRATION & UNIT TESTS ===")
    try:
        run_integration_tests()
        run_unit_tests()
    finally:
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
    print("\n=== ALL TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_tests()