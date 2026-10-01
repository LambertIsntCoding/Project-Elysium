"""Memory governance tests (Phase 1, sections 1-9).

Everything runs against a real ``TripleMemoryStore`` and the real consolidator
decision layer - no mocks. The tests cover the behaviours the Phase 1 brief
asked for, labelled A-J to match the brief:

A. temporary context is not stored as a persistent memory
B. contradiction handling updates rather than duplicating
C. confidence hierarchy keeps explicit above inferred
D. self-model protection blocks single-sentence self-memories
E. temporary context is available this session and does not persist
F. memory utility keeps important things active and marks stale ones dormant
G. personality feedback is captured as an observation, not a personality edit
H. style examples reach the prompt as concrete positive examples
I. project state is revisable rather than a permanent contradiction
J. no memory is ever deleted - superseded/dormant history stays readable
"""

import json
import os
import shutil
import tempfile
import unittest

from astra import authority
from astra.consolidator import (
    apply_decision,
    looks_like_personality_feedback,
    resolve_candidate,
)
from astra.memory import (
    TripleMemoryStore,
    classify_candidate,
    memory_utility,
)

CONFIG_DIR = "./config"


class _GovernanceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def ingest(self, candidate, conversation_id="c1", turn=1):
        """Run one candidate through the real decision layer."""
        decision = resolve_candidate(candidate)
        self.assertIsNotNone(decision)
        mem_id = apply_decision(self.store, decision, conversation_id, turn)
        return decision, mem_id

    def candidates(self, *candidates):
        return [self.ingest(c, turn=i + 1) for i, c in enumerate(candidates)]


# ---------------------------------------------------------------------
# A. Temporary context is not stored as a persistent memory
# ---------------------------------------------------------------------
class TestATemporaryContext(unittest.TestCase):
    def test_one_off_activity_is_not_persisted(self):
        decision = resolve_candidate({
            "classification": "persistent_user_fact",
            "content": "I played Sonic Adventure yesterday",
            "is_explicit_user_statement": True,
        })
        self.assertEqual(decision["classification"], "temporary_context")
        self.assertTrue(decision["transient"])

    def test_durable_trait_is_persisted(self):
        decision = resolve_candidate({
            "classification": "persistent_user_preference",
            "content": "my favorite game is Sonic Adventure",
            "is_explicit_user_statement": True,
        })
        self.assertEqual(decision["classification"], "persistent_user_preference")
        self.assertFalse(decision["transient"])

    def test_question_is_temporary(self):
        self.assertEqual(
            classify_candidate("What is my favourite game?", explicit=True,
                               mem_type="explicit_fact", target_model="roum",
                               source="explicit_user_statement"),
            "temporary_context",
        )


# ---------------------------------------------------------------------
# B. Contradiction handling updates rather than duplicating
# ---------------------------------------------------------------------
class TestBContradictionUpdates(_GovernanceCase):
    def test_new_explicit_statement_supersedes_the_old_one(self):
        _, old_id = self.ingest({
            "classification": "persistent_user_preference",
            "content": "Roum likes being called Bryson",
            "is_explicit_user_statement": True,
        })
        _, new_id = self.ingest({
            "classification": "persistent_user_preference",
            "content": "Roum does not like being called Bryson",
            "is_explicit_user_statement": True,
        }, turn=2)

        old = self.store.get_memory("roum", old_id)
        new = self.store.get_memory("roum", new_id)
        self.assertEqual(old["status"], "superseded")
        self.assertEqual(old["superseded_by"], new_id)
        self.assertEqual(new["status"], "active")
        self.assertEqual(new["supersedes"], old_id)
        # Exactly one authoritative memory remains for that topic.
        active = [m for m in self.store.get_active_memories("roum")
                  if "called" in m["content"] and "Bryson" in m["content"]]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["id"], new_id)

    def test_unrelated_statements_do_not_collide(self):
        _, a = self.ingest({
            "classification": "persistent_user_fact",
            "content": "Roum's computer has 32GB of RAM",
            "is_explicit_user_statement": True,
        })
        self.ingest({
            "classification": "persistent_user_fact",
            "content": "Roum lives in a city with a river",
            "is_explicit_user_statement": True,
        }, turn=2)
        self.assertEqual(self.store.get_memory("roum", a)["status"], "active")


# ---------------------------------------------------------------------
# C. Confidence hierarchy keeps explicit above inferred
# ---------------------------------------------------------------------
class TestCConfidenceHierarchy(_GovernanceCase):
    def test_explicit_outranks_inference(self):
        _, explicit = self.ingest({
            "classification": "persistent_user_preference",
            "content": "Roum likes coffee",
            "is_explicit_user_statement": True,
        })
        _, inferred = self.ingest({
            "classification": "uncertain_inference",
            "content": "Roum seems to like coffee",
            "is_explicit_user_statement": False,
            "confidence": 0.99,
        }, turn=2)
        self.assertGreater(
            authority.effective_strength(self.store.get_memory("roum", explicit)),
            authority.effective_strength(self.store.get_memory("roum", inferred)),
        )

    def test_inference_cannot_supersede_explicit(self):
        _, explicit = self.ingest({
            "classification": "persistent_user_preference",
            "content": "Roum likes tea",
            "is_explicit_user_statement": True,
        })
        self.ingest({
            "classification": "uncertain_inference",
            "content": "Roum does not like tea",
            "is_explicit_user_statement": False,
        }, turn=2)
        self.assertEqual(self.store.get_memory("roum", explicit)["status"], "active")

    def test_inference_never_stored_as_an_explicit_fact(self):
        _, mid = self.ingest({
            "classification": "persistent_user_fact",
            "content": "Roum likes X",
            "is_explicit_user_statement": False,
        })
        self.assertEqual(self.store.get_memory("roum", mid)["type"], "uncertain_inference")


# ---------------------------------------------------------------------
# D. Self-model protection
# ---------------------------------------------------------------------
class TestDSelfModelProtection(_GovernanceCase):
    def test_single_sentence_self_preference_is_demoted(self):
        _, mid = self.ingest({
            "classification": "self_preference",
            "content": "Astra likes this game",
            "is_explicit_user_statement": False,
            "confidence": 0.9,
        })
        mem = self.store.get_memory("self", mid)
        self.assertEqual(mem["type"], "self_observation")
        self.assertLessEqual(mem["confidence"], 0.5)

    def test_reinforced_self_preference_becomes_durable(self):
        mid = None
        for turn in range(3):
            _, mid = self.ingest({
                "classification": "self_preference",
                "content": "Astra prefers concise answers",
                "is_explicit_user_statement": False,
                "reinforcement_count": turn + 1,
            }, turn=turn + 1)
        self.assertEqual(self.store.get_memory("self", mid)["type"], "self_preference")

    def test_deliberate_declaration_is_durable(self):
        mid = self.store.add_memory("self", "Astra values honesty.",
                                    "self_preference", "governing_declaration")
        self.assertEqual(self.store.get_memory("self", mid)["type"], "self_preference")


# ---------------------------------------------------------------------
# E. Temporary context: available this session, never persisted
# ---------------------------------------------------------------------
class TestETemporaryContextStore(_GovernanceCase):
    def test_temporary_context_available_and_not_on_disk(self):
        decision = resolve_candidate({
            "classification": "persistent_user_fact",
            "content": "I am currently working on the memory layer",
            "is_explicit_user_statement": True,
        })
        apply_decision(self.store, decision, "c1", 1)

        items = self.store.get_temporary_context()
        self.assertTrue(any("memory layer" in i["content"] for i in items))
        # It is not in any persistent model.
        for target in ("roum", "self", "relationship"):
            self.assertEqual(self.store.get_memories(target), [])

    def test_temporary_context_does_not_survive_a_new_store(self):
        self.store.add_temporary_context("scratch note")
        fresh = TripleMemoryStore(data_dir=self.tmp)
        self.assertEqual(fresh.get_temporary_context(), [])

    def test_temporary_context_is_capped(self):
        for i in range(30):
            self.store.add_temporary_context(f"note {i}")
        self.assertLessEqual(len(self.store.get_temporary_context(limit=1000)), 20)

    def test_prompt_labels_temporary_context_as_session_only(self):
        from astra.orchestrator import CompanionOrchestrator

        self.store.add_temporary_context("we are talking about memory today")
        prompt = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR).build_prompt(
            "what were we saying?", [])
        self.assertIn("TEMPORARY CONTEXT (THIS SESSION ONLY)", prompt)
        self.assertIn("do not treat it as a permanent fact", prompt)


# ---------------------------------------------------------------------
# F. Memory utility: keep important, mark stale dormant
# ---------------------------------------------------------------------
class TestFMemoryUtility(_GovernanceCase):
    def test_recently_used_memory_stays_active(self):
        mid = self.store.add_memory("roum", "Roum's favourite game is Sonic Adventure.",
                                    "explicit_preference", "explicit_user_statement")
        self.store.record_use([mid])
        self.assertEqual(memory_utility(self.store.get_memory("roum", mid)), "active")

    def test_stale_weak_memory_becomes_dormant_but_survives(self):
        from datetime import datetime, timedelta, timezone

        old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        mid = self.store.add_memory("roum", "Roum mentioned a random indie game once.",
                                    "uncertain_inference", "ai_inference",
                                    confidence=0.25, timestamp=old)
        self.store.apply_utility_decay()
        mem = self.store.get_memory("roum", mid)
        self.assertEqual(mem["utility"], "dormant")
        # Dormant is a soft demotion: the content is still there and retrievable.
        self.assertEqual(mem["content"], "Roum mentioned a random indie game once.")
        self.assertIn(mid, [m["id"] for m in self.store.get_retrievable_memories("roum")])

    def test_dormant_memory_never_governs(self):
        from datetime import datetime, timedelta, timezone

        old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        mid = self.store.add_memory("roum", "Roum might dislike long answers.",
                                    "explicit_preference", "explicit_user_statement",
                                    confidence=0.2, timestamp=old)
        self.store.apply_utility_decay()
        self.assertNotIn(mid, [m["id"] for m in self.store.get_governing_memories("roum")])

    def test_governing_memory_never_goes_dormant(self):
        from datetime import datetime, timedelta, timezone

        old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        mid = self.store.add_memory("roum", "Roum dislikes being narrated to.",
                                    "explicit_preference", "explicit_user_statement",
                                    timestamp=old)
        self.store.apply_utility_decay()
        self.assertEqual(memory_utility(self.store.get_memory("roum", mid)), "active")


# ---------------------------------------------------------------------
# G. Personality feedback is captured, not silently applied
# ---------------------------------------------------------------------
class TestGPersonalityFeedback(_GovernanceCase):
    def test_feedback_is_detected(self):
        self.assertTrue(looks_like_personality_feedback("You sound like a generic chatbot."))
        self.assertFalse(looks_like_personality_feedback("What is the capital of France?"))

    def test_feedback_stored_as_observation_not_personality_edit(self):
        mid = self.store.add_memory(
            "relationship",
            "Roum perceives Astra's conversational style as needing adjustment: "
            "you sound too formal",
            "relationship_observation", "explicit_user_statement",
            tags=["personality_feedback"],
        )
        mem = self.store.get_memory("relationship", mid)
        # It is a record of feedback, not a governing behavioural directive.
        self.assertEqual(mem["type"], "relationship_observation")
        self.assertFalse(authority.is_governing(mem))


# ---------------------------------------------------------------------
# H. Style examples reach the prompt as concrete positive examples
# ---------------------------------------------------------------------
class TestHStyleExamples(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        from astra.orchestrator import CompanionOrchestrator

        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_examples_present_with_good_and_bad_responses(self):
        prompt = self.orch.build_prompt("Heyo!", [])
        self.assertIn("=== STYLE EXAMPLES ===", prompt)
        self.assertIn("Good Response:", prompt)
        self.assertIn("Bad Response:", prompt)

    def test_examples_include_disagreement(self):
        prompt = self.orch.build_prompt("Let's rewrite the whole thing with microservices.", [])
        self.assertIn("adds three new failure modes", prompt)


# ---------------------------------------------------------------------
# I. Project state is revisable, not a permanent contradiction
# ---------------------------------------------------------------------
class TestIProjectState(_GovernanceCase):
    def test_new_state_supersedes_old_without_calling_it_false(self):
        _, old_id = self.ingest({
            "classification": "persistent_project_information",
            "content": "Project Elysium is in progress",
            "is_explicit_user_statement": True,
        })
        _, new_id = self.ingest({
            "classification": "persistent_project_information",
            "content": "Project Elysium is currently stalled because I am waiting for publisher approval",
            "is_explicit_user_statement": True,
        }, turn=2)

        old = self.store.get_memory("roum", old_id)
        self.assertEqual(old["status"], "superseded")
        self.assertIn("state, not a false claim", old["supersession_reason"])
        # The historical record is intact.
        self.assertEqual(old["content"], "Project Elysium is in progress")
        self.assertEqual(self.store.get_memory("roum", new_id)["status"], "active")

    def test_new_project_fact_does_not_supersede_unrelated_state(self):
        _, a = self.ingest({
            "classification": "persistent_project_information",
            "content": "Project Elysium is in progress",
            "is_explicit_user_statement": True,
        })
        self.ingest({
            "classification": "persistent_project_information",
            "content": "Project Nova is currently stalled",
            "is_explicit_user_statement": True,
        }, turn=2)
        self.assertEqual(self.store.get_memory("roum", a)["status"], "active")


# ---------------------------------------------------------------------
# J. Nothing is ever deleted
# ---------------------------------------------------------------------
class TestJHistoryPreserved(_GovernanceCase):
    def test_superseded_history_stays_readable(self):
        _, old_id = self.ingest({
            "classification": "persistent_user_preference",
            "content": "Roum likes being called Master",
            "is_explicit_user_statement": True,
        })
        self.ingest({
            "classification": "persistent_user_preference",
            "content": "Roum does not like being called Master",
            "is_explicit_user_statement": True,
        }, turn=2)
        # Still on disk, still readable, just not authoritative.
        self.assertEqual(self.store.get_memory("roum", old_id)["content"],
                         "Roum likes being called Master")

    def test_storage_files_keep_every_record(self):
        self.ingest({
            "classification": "persistent_user_fact",
            "content": "Roum has a cat",
            "is_explicit_user_statement": True,
        })
        self.ingest({
            "classification": "persistent_user_fact",
            "content": "Roum has a cat",
            "is_explicit_user_statement": True,
        }, turn=2)
        all_mems = self.store.get_memories("roum", status=None)
        self.assertEqual(len(all_mems), 1)  # exact duplicate reinforced, not duplicated
        self.assertGreaterEqual(all_mems[0]["reinforcement_count"], 2)


# ---------------------------------------------------------------------
# K. Restatements are recognised even when worded differently
# ---------------------------------------------------------------------
class TestKRestatementSupersession(_GovernanceCase):
    def test_restated_preference_supersedes_the_original(self):
        _, old_id = self.ingest({
            "classification": "persistent_user_preference",
            "content": "Sonic Adventure is Roum's favorite game.",
            "is_explicit_user_statement": True,
        })
        _, new_id = self.ingest({
            "classification": "persistent_user_preference",
            "content": "He likes Sonic Adventure even though it is not his absolute favorite game.",
            "is_explicit_user_statement": True,
        }, turn=2)

        old = self.store.get_memory("roum", old_id)
        self.assertIn(old["status"], ("superseded", "weakened"))
        self.assertEqual(old["superseded_by"], new_id)
        self.assertEqual(self.store.get_memory("roum", new_id)["status"], "active")

    def test_incidental_negation_does_not_reverse_the_subject(self):
        # The second sentence negates a detail, not the shared preference, so it
        # must not be read as a reversal of the first.
        self.ingest({
            "classification": "persistent_user_fact",
            "content": "The AI maintains its core function is learning.",
            "is_explicit_user_statement": True,
        })
        _, b = self.ingest({
            "classification": "persistent_user_fact",
            "content": (
                "Astra understands that its core statement regarding its nature "
                "is not intended as a disclaimer."
            ),
            "is_explicit_user_statement": True,
        }, turn=2)
        self.assertEqual(self.store.get_memory("roum", b)["status"], "active")

    def test_learning_again_does_not_re_count_the_contradiction(self):
        _, old_id = self.ingest({
            "classification": "persistent_user_preference",
            "content": "Roum likes being called Bryson.",
            "is_explicit_user_statement": True,
        })
        self.ingest({
            "classification": "persistent_user_preference",
            "content": "Roum does not like being called Bryson.",
            "is_explicit_user_statement": True,
        }, turn=2)
        first = self.store.get_memory("roum", old_id)
        self.ingest({
            "classification": "persistent_user_preference",
            "content": "Roum does not like being called Bryson.",
            "is_explicit_user_statement": True,
        }, turn=3)
        second = self.store.get_memory("roum", old_id)
        self.assertEqual(first["contradiction_count"], second["contradiction_count"])


# ---------------------------------------------------------------------
# L. Reconciliation heals a store written before restatements were detected
# ---------------------------------------------------------------------
class TestLReconciliation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_legacy_store(self, contents):
        """Write a store file the way the pre-fix code left it: all active."""
        seed_dir = tempfile.mkdtemp()
        try:
            seed = TripleMemoryStore(data_dir=seed_dir)
            records = []
            for content in contents:
                mid = seed.add_memory("roum", content, "explicit_preference",
                                      "explicit_user_statement")
                records.append(seed.get_memory("roum", mid))
            for rec in records:  # undo any resolution the live store applied
                rec["status"] = "active"
                rec["superseded_by"] = None
                rec["supersedes"] = None
                rec.pop("contradiction_reason", None)
                rec["contradicts"] = []
                rec["contradiction_count"] = 0
        finally:
            shutil.rmtree(seed_dir, ignore_errors=True)
        with open(os.path.join(self.tmp, "roum_model.json"), "w", encoding="utf-8") as f:
            json.dump(records, f)

    def test_reconcile_links_unresolved_restatements(self):
        self._write_legacy_store([
            "Sonic Adventure is Roum's favorite game.",
            "He likes Sonic Adventure even though it is not his absolute favorite game.",
        ])
        store = TripleMemoryStore(data_dir=self.tmp)
        old, new = (m["id"] for m in store.get_memories("roum", status="active"))
        self.assertEqual(store.get_memory("roum", old)["status"], "active")

        resolved = store.reconcile_contradictions()
        self.assertGreaterEqual(resolved, 1)
        self.assertIn(store.get_memory("roum", old)["status"],
                      ("superseded", "weakened"))
        self.assertEqual(store.get_memory("roum", old)["superseded_by"], new)
        # Nothing is deleted, and a second pass is a no-op.
        self.assertEqual(store.get_memory("roum", old)["content"],
                         "Sonic Adventure is Roum's favorite game.")
        self.assertEqual(store.reconcile_contradictions(), 0)


if __name__ == "__main__":
    unittest.main()
