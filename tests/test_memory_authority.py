"""Memory reliability tests (sections 1-17 of the memory overhaul).

Covers the fourteen behaviours requested, plus the runtime injection
diagnostics. Everything runs against a real ``TripleMemoryStore`` on disk and a
real ``CompanionOrchestrator`` - no mocks of the memory path.
"""

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import astra.authority as authority
from astra.memory import TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator, DeterministicLexicalRetriever

CONFIG_DIR = "./config"


def _old(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


class _MemoryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def prompt(self, text, history=None):
        return self.orch.build_prompt(text, history or [])


# ---------------------------------------------------------------------
# 1. Explicit beats inferred
# ---------------------------------------------------------------------
class TestExplicitBeatsInferred(_MemoryCase):
    def test_explicit_statement_outranks_inference(self):
        explicit = self.store.add_memory(
            "roum", "Roum likes coffee.", "explicit_preference",
            "explicit_user_statement", confidence=0.9)
        inferred = self.store.add_memory(
            "roum", "Roum seems to like coffee.", "explicit_preference",
            "ai_inference", confidence=0.99)

        # The inference is demoted at write time and is strictly weaker.
        self.assertEqual(self.store.get_memory("roum", inferred)["type"], "uncertain_inference")
        self.assertGreater(
            authority.effective_strength(self.store.get_memory("roum", explicit)),
            authority.effective_strength(self.store.get_memory("roum", inferred)),
        )

    def test_inference_does_not_override_explicit_on_contradiction(self):
        explicit = self.store.add_memory(
            "roum", "Roum likes being called Bryson.", "explicit_preference",
            "explicit_user_statement", confidence=0.9)
        # An AI inference contradicts it; the explicit memory must survive.
        self.store.add_memory(
            "roum", "Roum does not like being called Bryson.", "explicit_preference",
            "ai_inference", confidence=0.99)

        mem = self.store.get_memory("roum", explicit)
        self.assertEqual(mem["status"], "active")
        self.assertEqual(mem["confidence"], 0.9)


# ---------------------------------------------------------------------
# 2. Explicit correction weakens old contradictory information
# ---------------------------------------------------------------------
class TestContradictionResolution(_MemoryCase):
    def test_correction_weakens_old_memory(self):
        old = self.store.add_memory(
            "roum", "Roum likes being called Bryson.", "explicit_preference",
            "explicit_user_statement", confidence=0.9)
        new = self.store.add_memory(
            "roum", "Roum does not like being called Bryson.", "explicit_preference",
            "explicit_user_statement", confidence=0.95)

        old_mem = self.store.get_memory("roum", old)
        self.assertEqual(old_mem["status"], "weakened")
        self.assertLess(old_mem["confidence"], 0.9)
        self.assertEqual(old_mem["superseded_by"], new)
        self.assertIn(new, old_mem["contradicts"])
        self.assertIn("ontradicted", old_mem["contradiction_reason"])
        # The new memory stays authoritative and links back.
        self.assertEqual(self.store.get_memory("roum", new)["status"], "active")
        self.assertEqual(self.store.get_memory("roum", new)["supersedes"], old)

    def test_contradictory_history_is_kept(self):
        old = self.store.add_memory("roum", "Roum likes tea.", "explicit_preference",
                                    "explicit_user_statement", confidence=0.9)
        self.store.add_memory("roum", "Roum does not like tea.", "explicit_preference",
                              "explicit_user_statement", confidence=0.95)
        # Still readable even though it is no longer authoritative.
        self.assertEqual(self.store.get_memory("roum", old)["content"], "Roum likes tea.")


# ---------------------------------------------------------------------
# 3. Repeated reinforcement increases stability
# ---------------------------------------------------------------------
class TestReinforcement(_MemoryCase):
    def test_repetition_raises_confidence_and_count(self):
        first = self.store.add_memory("roum", "Roum likes tea.", "explicit_preference",
                                      "explicit_user_statement", confidence=0.6)
        again = self.store.add_memory("roum", "  Roum   LIKES tea. ", "explicit_preference",
                                      "explicit_user_statement")
        self.assertEqual(first, again)
        mem = self.store.get_memory("roum", first)
        self.assertEqual(mem["reinforcement_count"], 2)
        self.assertGreater(mem["confidence"], 0.6)

    def test_reinforced_memory_holds_up_better_than_one_off(self):
        reinforced = self.store.add_memory("roum", "Roum values honesty.", "explicit_preference",
                                           "explicit_user_statement", timestamp=_old(40))
        for _ in range(5):
            self.store.add_memory("roum", "Roum values honesty.", "explicit_preference",
                                  "explicit_user_statement")
        one_off = self.store.add_memory("roum", "Roum mentioned honesty once.", "uncertain_inference",
                                        "ai_extraction", confidence=0.4, timestamp=_old(40))
        self.assertGreater(
            authority.effective_strength(self.store.get_memory("roum", reinforced)),
            authority.effective_strength(self.store.get_memory("roum", one_off)),
        )


# ---------------------------------------------------------------------
# 4. Unused weak memories decay
# ---------------------------------------------------------------------
class TestDecay(_MemoryCase):
    def test_unused_low_value_memory_decays_and_archives(self):
        junk = self.store.add_memory(
            "roum", "Roum mentioned a random indie game once.", "uncertain_inference",
            "ai_inference", confidence=0.3, timestamp=_old(200))
        self.store.apply_decay()
        self.assertEqual(self.store.get_memory("roum", junk)["status"], "archived")
        self.assertNotIn(junk, [m["id"] for m in self.store.get_active_memories("roum")])

    def test_decay_is_driven_by_use_not_age(self):
        used = self.store.add_memory("roum", "Roum's favorite game is Sonic Adventure.",
                                     "explicit_preference", "explicit_user_statement",
                                     timestamp=_old(200))
        self.store.record_use([used])
        mem = self.store.get_memory("roum", used)
        self.assertGreater(authority.recency_factor(mem), 0.9)


# ---------------------------------------------------------------------
# 5. Important governing memories do not decay from inactivity
# ---------------------------------------------------------------------
class TestGoverningDoesNotDecay(_MemoryCase):
    def test_boundary_survives_long_inactivity(self):
        gov = self.store.add_memory(
            "relationship", "Roum wants his real-life friendships to stay important.",
            "relationship_boundary", "explicit_user_statement", confidence=1.0,
            timestamp=_old(400))
        self.store.apply_decay()
        self.assertEqual(self.store.get_memory("relationship", gov)["status"], "active")
        self.assertIn(gov, [m["id"] for m in self.store.get_governing_memories()])


# ---------------------------------------------------------------------
# 6. Superseded memories do not outrank current ones
# ---------------------------------------------------------------------
class TestSupersededRanking(_MemoryCase):
    def test_superseded_is_not_retrievable(self):
        old = self.store.add_memory("roum", "Address Roum as 'Master'.", "relationship_boundary",
                                    "explicit_user_statement", slot="address", confidence=0.9)
        new = self.store.supersede_memory("roum", old, "Address Roum as 'Roum'.",
                                          slot="address", source="user_correction")
        self.assertNotIn(old, [m["id"] for m in self.store.get_retrievable_memories("roum")])
        self.assertLess(
            authority.effective_strength(self.store.get_memory("roum", old)),
            authority.effective_strength(self.store.get_memory("roum", new)),
        )


# ---------------------------------------------------------------------
# 7. Inference cannot become a confirmed fact through repetition
# ---------------------------------------------------------------------
class TestProvenanceIsPreserved(_MemoryCase):
    def test_astra_inference_never_becomes_explicit_fact(self):
        mid = self.store.add_memory("roum", "Roum likes X.", "explicit_fact",
                                    "ai_inference", confidence=0.9)
        self.assertNotEqual(self.store.get_memory("roum", mid)["type"], "explicit_fact")
        self.assertEqual(self.store.get_memory("roum", mid)["type"], "uncertain_inference")

    def test_repeated_inference_stays_an_inference(self):
        for _ in range(5):
            mid = self.store.add_memory("roum", "Roum likes X.", "explicit_preference",
                                        "ai_extraction", confidence=0.9)
        mem = self.store.get_memory("roum", mid)
        self.assertEqual(mem["type"], "uncertain_inference")
        self.assertLess(authority.source_tier(mem["source"]), authority.source_tier("explicit_user_statement"))


# ---------------------------------------------------------------------
# 8. Relevant memories are actually injected into the runtime prompt
# ---------------------------------------------------------------------
class TestRuntimeInjection(_MemoryCase):
    def test_relevant_memory_reaches_the_prompt(self):
        self.store.add_memory("roum", "Roum's favorite game is Sonic Adventure.",
                              "explicit_fact", "explicit_user_statement",
                              keywords=["Sonic Adventure"])
        prompt = self.prompt("What do you know about Sonic Adventure?")
        self.assertIn("Sonic Adventure", prompt)
        self.assertIn("=== FACTUAL CONTEXT ===", prompt)

    def test_diagnostics_report_the_injection_stages(self):
        mid = self.store.add_memory("roum", "Roum's favorite game is Sonic Adventure.",
                                    "explicit_fact", "explicit_user_statement",
                                    keywords=["Sonic Adventure"])
        _, diag = self.orch.build_prompt_with_diagnostics("Sonic Adventure", [])
        self.assertIn(mid, diag["injected_ids"])
        self.assertIn(mid, [c["id"] for c in diag["candidates"]])
        cand = next(c for c in diag["candidates"] if c["id"] == mid)
        self.assertGreater(cand["effective_strength"], 0)
        self.assertGreater(cand["relevance"], 0)


# ---------------------------------------------------------------------
# 9. Governing memories available with zero lexical overlap
# ---------------------------------------------------------------------
class TestGoverningZeroOverlap(_MemoryCase):
    def test_governing_memory_injected_without_keyword_match(self):
        self.store.add_memory("roum", "Roum dislikes when Astra narrates her processing.",
                              "explicit_preference", "explicit_user_statement")
        prompt = self.prompt("What is the weather like today?")
        self.assertIn("GOVERNING MEMORIES (ALWAYS APPLY)", prompt)
        self.assertIn("narrates her processing", prompt)

    def test_inferred_preference_does_not_govern(self):
        self.store.add_memory("roum", "Roum might dislike long answers.", "explicit_preference",
                              "ai_inference")
        prompt = self.prompt("Tell me a story.")
        self.assertNotIn("Roum might dislike long answers.", prompt)


# ---------------------------------------------------------------------
# 10. Archived memories do not appear as active instructions
# ---------------------------------------------------------------------
class TestArchivedNeverActive(_MemoryCase):
    def test_archived_memory_absent_from_prompt(self):
        mid = self.store.add_memory("roum", "Roum once mentioned visiting a hot air balloon show.",
                                    "uncertain_inference", "ai_inference", confidence=0.2,
                                    keywords=["balloon"])
        self.store.archive_memory("roum", mid, reason="noise")
        prompt = self.prompt("Tell me about hot air balloons.")
        self.assertNotIn("hot air balloon show", prompt)


# ---------------------------------------------------------------------
# 11 & 12. Relationship boundaries and no romantic escalation
# ---------------------------------------------------------------------
class TestRelationshipBoundaries(_MemoryCase):
    def test_boundaries_always_injected(self):
        prompt = self.prompt("Heyo!")
        self.assertIn("RELATIONSHIP BOUNDARIES (NON-NEGOTIABLE)", prompt)
        self.assertIn("non-romantic", prompt)
        self.assertIn("take priority over his relationship with Astra", prompt)

    def test_affection_does_not_create_romantic_memory(self):
        mid = self.store.add_memory("relationship", "Roum loves Astra and wants to marry her.",
                                    "relationship_event", "ai_extraction", confidence=0.9)
        mem = self.store.get_memory("relationship", mid)
        self.assertEqual(mem["status"], "weakened")
        self.assertEqual(mem["type"], "relationship_observation")
        self.assertTrue(mem.get("boundary_violation"))
        self.assertFalse(authority.is_governing(mem))

    def test_romantic_content_never_governs_or_injects(self):
        self.store.add_memory("relationship", "Astra is Roum's girlfriend.",
                              "relationship_event", "ai_extraction", confidence=0.9)
        prompt = self.prompt("Who are you to me?")
        self.assertNotIn("girlfriend", prompt)


# ---------------------------------------------------------------------
# 13. Participation does not become exclusivity
# ---------------------------------------------------------------------
class TestParticipationNotExclusivity(_MemoryCase):
    def test_boundary_forbids_exclusivity(self):
        prompt = self.prompt("I hung out with my friends today.")
        self.assertIn("never seeks exclusivity", prompt)
        self.assertIn("never asks him to choose her over anyone", prompt)

    def test_human_relationships_outrank_astra(self):
        prompt = self.prompt("anything at all")
        self.assertIn("always take priority over his relationship with Astra", prompt)


# ---------------------------------------------------------------------
# 14. Current address/name state overrides stale historical address
# ---------------------------------------------------------------------
class TestCurrentAddressState(_MemoryCase):
    def test_current_state_slot_wins(self):
        old = self.store.add_memory("roum", "Address Roum as 'Master'.", "relationship_boundary",
                                    "explicit_user_statement", slot="address", confidence=0.8)
        new = self.store.supersede_memory("roum", old, "Address Roum as 'Roum'.",
                                          slot="address", source="user_correction")

        state = self.store.get_current_state()
        self.assertEqual([m["id"] for m in state], [new])

        prompt = self.prompt("sup")
        self.assertIn("=== CURRENT STATE (AUTHORITATIVE) ===", prompt)
        self.assertIn("Address Roum as 'Roum'.", prompt)
        self.assertNotIn("Address Roum as 'Master'.", prompt)

    def test_stale_slot_is_superseded_automatically(self):
        self.store.add_memory("roum", "Address Roum as 'Lambert'.", "relationship_boundary",
                              "explicit_user_statement", slot="address",
                              timestamp=_old(30))
        latest = self.store.add_memory("roum", "Address Roum as 'Bryson'.",
                                       "relationship_boundary", "explicit_user_statement",
                                       slot="address")
        self.store.expire_governing_slots()
        current = self.store.get_current_state()
        self.assertEqual([m["id"] for m in current], [latest])


# ---------------------------------------------------------------------
# Retrieval weighting: relevant-but-weak vs slightly-less-relevant explicit
# ---------------------------------------------------------------------
class TestRetrievalWeighting(_MemoryCase):
    def test_explicit_fact_outranks_weak_inference(self):
        weak = {"id": "weak", "content": "Roum likes tea.", "status": "active",
                "confidence": 0.15, "source": "ai_inference", "type": "uncertain_inference"}
        strong = {"id": "strong", "content": "Roum likes tea as well.", "status": "active",
                  "confidence": 0.95, "source": "explicit_user_statement",
                  "type": "explicit_preference"}
        ranked = DeterministicLexicalRetriever.retrieve("does Roum like tea", [weak, strong], top_k=2)
        self.assertEqual(ranked[0]["id"], "strong")


if __name__ == "__main__":
    unittest.main(verbosity=2)
