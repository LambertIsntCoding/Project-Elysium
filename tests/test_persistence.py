import os
import shutil
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import requests

from astra.memory import TripleMemoryStore
from astra.orchestrator import (
    CompanionOrchestrator,
    BehavioralAdaptationCompiler,
    DeterministicLexicalRetriever,
)

TEST_DIR = "./test_storage_behavior_final"
GOVERNING_HEADER = "=== GOVERNING MEMORIES (ALWAYS APPLY) ==="
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

    # TEST A: an explicit preference is governing (always applied) even when the
    # user's question shares no tokens with it.
    orchestrator_s1 = CompanionOrchestrator(store_session1)
    prompt_a = orchestrator_s1.build_prompt("How was your day?", [])

    assert GOVERNING_HEADER in prompt_a
    governing_a = prompt_a.split(GOVERNING_HEADER)[1].split(FACT_HEADER)[0]
    assert "narrate internal processing" in governing_a.lower()
    print("✓ Test A: explicit preference becomes a governing memory with zero token overlap.")

    # TEST B: self_observation is NOT treated as a governing/behavioral rule
    assert "felt nervous" not in governing_a
    print("✓ Test B: self_observation is excluded from governing memories.")

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

    governing_s2 = prompt_s2.split(GOVERNING_HEADER)[1].split(FACT_HEADER)[0]
    assert "narrate internal processing" in governing_s2.lower()
    assert snapshot_store_hashes(TEST_DIR) == hashes_before
    print("✓ Test D: governing memory persists into a fresh session, and files are still unchanged.")


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


def run_config_robustness_tests():
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as cfg:
        with open(os.path.join(cfg, "identity.yaml"), "w", encoding="utf-8") as f:
            f.write(textwrap.dedent("""
                identity: just a string
                behavioral_rules: "not a list"
                speech_style:
                  core_style: [" calm ", null, ""]
                  language:
                    preferred: "plain string"
                    avoid: ["corporate speak"]
            """))
        with open(os.path.join(cfg, "behavior_examples.yaml"), "w", encoding="utf-8") as f:
            f.write("examples:\n  - situation: hi\n    good_response: hello\n  - not a dict\n")

        class Store:
            def get_active_memories(self, model: str):
                return [{"content": None, "keywords": ["hello"],
                         "type": "explicit_fact", "status": "active"}]

        orch = CompanionOrchestrator(Store(), config_dir=cfg)
        history = [{"role": "user", "content": None}, {"role": "assistant", "content": "hi"}]
        prompt = orch.build_prompt("hello", history)

    assert "Name: Astra" in prompt, "Non-dict identity should fall back to defaults."
    assert "- calm" in prompt and "- corporate speak" in prompt
    assert "Preferred:" not in prompt and "plain string" not in prompt, \
        "String-valued list fields should be ignored, not iterated per character."
    assert "BASE BEHAVIORAL RULES" not in prompt
    assert "LEARNED BEHAVIORAL ADAPTATIONS" not in prompt, "Empty adaptations section should be omitted."
    assert "No query-relevant background facts" in prompt, "None-content memory should not render."
    assert "Good Response: hello" in prompt
    assert "ASTRA: hi" in prompt and prompt.count("ROUM:") == 1, "Empty history turns should be skipped."
    print("✓ Test I: malformed config, None memory content and empty history turns are handled cleanly.")


def run_tests():
    print("=== RUNNING PERSISTENCE, INTEGRATION & UNIT TESTS ===")
    try:
        run_integration_tests()
        run_unit_tests()
        run_config_robustness_tests()
    finally:
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
    print("\n=== ALL TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_tests()
