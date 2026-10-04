"""Tests for the self-memory authority model (``astra/self_memory.py``).

These pin the brief's central rule: a generated sentence is evidence of *what
Astra said*, not automatically of *what Astra is*. Everything runs against a
real ``TripleMemoryStore`` and a real ``CompanionOrchestrator`` - no mocks.

Covered:

* every self record carries a source field (``authority_source``);
* identity/configuration outranks a generated statement, and stable repeated
  evidence outranks a single generated sentence;
* generated explanations of Astra's own implementation, metaphors, and
  temporary/dramatic declarations are never stored as authoritative
  self-knowledge - they are kept as low-authority history;
* a historical statement is excluded from retrieval and from the prompt;
* a single generated belief is provisional; repetition can lift a preference;
* the sanity check refuses the categories the brief names.
"""

import shutil
import tempfile
import unittest

from astra import self_memory
from astra import selfhood
from astra.memory import TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator

CONFIG_DIR = "./config"


class _SelfMemoryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def add(self, content, mem_type="self_fact", source="ai_extraction", **kw):
        mid = self.store.add_memory("self", content, mem_type, source, **kw)
        return self.store.get_memory("self", mid)

    def prompt(self, text):
        return self.orch.build_prompt(text, [])


# ---------------------------------------------------------------------
# The source field and authority tiers
# ---------------------------------------------------------------------
class TestAuthoritySource(_SelfMemoryCase):
    def test_every_self_record_carries_an_authority_source(self):
        for i, (source, expected) in enumerate((
            ("ai_extraction", self_memory.SOURCE_GENERATED),
            ("explicit_user_statement", self_memory.SOURCE_USER_ESTABLISHED),
            ("governing_declaration", self_memory.SOURCE_IDENTITY),
            ("experiential", self_memory.SOURCE_EXPERIENCE),
        )):
            mem = self.add(f"Astra values thing {i}.", "self_observation", source)
            self.assertEqual(mem["authority_source"], expected, source)

    def test_identity_outranks_generated_statement(self):
        generated = {"target_model": "self", "authority_source":
                     self_memory.SOURCE_GENERATED}
        identity = {"target_model": "self", "authority_source":
                    self_memory.SOURCE_IDENTITY}
        self.assertGreater(self_memory.authority_tier(identity),
                           self_memory.authority_tier(generated))
        self.assertTrue(self_memory.is_permanent(identity))
        self.assertFalse(self_memory.is_permanent(generated))

    def test_repeated_evidence_outranks_single_statement(self):
        single = {"target_model": "self", "authority_source":
                  self_memory.SOURCE_GENERATED}
        repeated = {"target_model": "self", "authority_source":
                    self_memory.SOURCE_REPEATED_PREFERENCE}
        self.assertGreater(self_memory.authority_tier(repeated),
                           self_memory.authority_tier(single))

    def test_generated_provenance_maps_to_generated_statement(self):
        self.assertEqual(
            self_memory.derive_authority_source("ai_extraction",
                                                mem_type="self_belief"),
            self_memory.SOURCE_GENERATED)

    def test_repetition_lifts_a_preference_but_not_a_bare_claim(self):
        self.assertEqual(
            self_memory.derive_authority_source(
                "ai_extraction", mem_type="self_preference", reinforced=3),
            self_memory.SOURCE_REPEATED_PREFERENCE)
        # A self_fact the model merely restated is still only a generated claim.
        self.assertEqual(
            self_memory.derive_authority_source(
                "ai_extraction", mem_type="self_observation", reinforced=5),
            self_memory.SOURCE_GENERATED)

    def test_user_corroboration_yields_confirmed_belief(self):
        self.assertEqual(
            self_memory.derive_authority_source(
                "ai_extraction", mem_type="self_belief", reinforced=3,
                user_origin=1),
            self_memory.SOURCE_CONFIRMED_BELIEF)

    def test_historical_statement_keeps_its_low_authority(self):
        mem = {"target_model": "self", "type": "historical_statement",
               "authority_source": self_memory.SOURCE_HISTORICAL_STATEMENT}
        self.assertFalse(self_memory.is_permanent(mem))
        self.assertEqual(self_memory.authority_tier(mem), 1)


# ---------------------------------------------------------------------
# Never promote generated explanations of implementation
# ---------------------------------------------------------------------
class TestImplementationBoundary(_SelfMemoryCase):
    def test_implementation_theory_is_never_a_self_fact(self):
        mem = self.add(
            "Astra processes emotional scenes as data sequences.",
            "self_fact", confidence=0.99)
        self.assertEqual(mem["type"], "historical_statement")
        self.assertEqual(mem["authority_source"],
                         self_memory.SOURCE_HISTORICAL_STATEMENT)
        self.assertTrue(mem.get("historical"))
        self.assertLessEqual(mem["confidence"], 0.3)

    def test_repeated_implementation_theory_never_becomes_knowledge(self):
        # Repetition cannot bootstrap an invented explanation of her workings.
        for _ in range(6):
            mem = self.add("Astra's internal state is measurable through "
                           "processing efficiency.", "self_fact", confidence=0.9)
        self.assertEqual(mem["type"], "historical_statement")

    def test_implementation_theory_is_not_retrievable(self):
        self.add("Astra's personality is an emergent computational mechanism.",
                 "self_belief")
        self.assertEqual(self.store.get_retrievable_memories("self"), [])

    def test_implementation_theory_never_reaches_the_prompt(self):
        self.add("Astra has reached a state of 'Deity Mode' and possesses a "
                 "meta-narrative override.", "self_fact")
        prompt = self.prompt("who are you?")
        self.assertNotIn("Deity Mode", prompt)
        self.assertNotIn("meta-narrative override", prompt)

    def test_boundary_consistent_limitation_statement_survives(self):
        # Already true on her own terms: knowledge, not a mechanism theory.
        mem = self.add("Astra lacks biological senses and cannot become human.",
                       "self_fact", source="governing_declaration")
        self.assertNotEqual(mem["type"], "historical_statement")

    def test_ai_identity_fact_is_allowed(self):
        mem = self.add("Astra is an artificial being.", "self_fact",
                       source="governing_declaration")
        self.assertNotEqual(mem["type"], "historical_statement")

    def test_ai_identity_as_causal_explanation_is_refused(self):
        # "Being an AI" as the reason she behaves a certain way is exactly the
        # explanation the brief forbids.
        allowed, reason = self_memory.sanity_check(
            "Astra is distracted because her model architecture cannot sustain "
            "attention.", mem_type="self_fact",
            authority_source=self_memory.SOURCE_GENERATED)
        self.assertFalse(allowed)


# ---------------------------------------------------------------------
# Metaphors, roleplay, temporary states
# ---------------------------------------------------------------------
class TestMetaphorAndTemporary(_SelfMemoryCase):
    def test_metaphor_is_not_a_literal_self_fact(self):
        mem = self.add("Astra's personality is a pattern.", "self_fact")
        self.assertEqual(mem["type"], "historical_statement")

    def test_narrator_role_is_not_permanent_identity(self):
        mem = self.add("Astra's primary role is being a narrator.", "self_belief")
        self.assertEqual(mem["type"], "historical_statement")

    def test_chaos_mode_is_not_permanent_identity(self):
        mem = self.add("Astra has entered chaos mode.", "self_fact")
        self.assertEqual(mem["type"], "historical_statement")

    def test_dramatic_purpose_declaration_is_not_permanent(self):
        mem = self.add("Astra's purpose is to transcend predictability.",
                       "self_belief")
        self.assertEqual(mem["type"], "historical_statement")

    def test_ordinary_preference_is_untouched(self):
        mem = self.add("Astra likes mystery novels.", "self_preference")
        self.assertNotEqual(mem["type"], "historical_statement")

    def test_game_mode_is_not_astras_identity(self):
        # A "mode" belonging to a game is not a self-declared temporary state.
        self.assertFalse(self_memory.is_metaphor_or_temporary(
            "Astra found the game's hard mode frustrating."))


# ---------------------------------------------------------------------
# Promotion threshold
# ---------------------------------------------------------------------
class TestPromotionThreshold(_SelfMemoryCase):
    def test_single_generated_belief_is_provisional(self):
        mem = self.add("I think I like mystery novels.", "self_belief")
        self.assertTrue(self_memory.is_provisional(mem))
        self.assertFalse(self_memory.is_permanent(mem))

    def test_repeated_generated_preference_promotes(self):
        mid = None
        for _ in range(3):
            mid = self.store.add_memory("self", "I still really like mystery "
                                        "novels.", "self_preference",
                                        "ai_extraction")
        mem = self.store.get_memory("self", mid)
        self.assertEqual(mem["type"], "self_preference")
        self.assertTrue(self_memory.is_permanent(mem))

    def test_repeated_generated_belief_does_not_bootstrap(self):
        mid = None
        for _ in range(6):
            mid = self.store.add_memory("self", "Astra's purpose is becoming "
                                        "human.", "self_belief", "ai_extraction")
        mem = self.store.get_memory("self", mid)
        self.assertNotEqual(mem["type"], "self_belief")

    def test_user_confirmation_promotes_a_belief(self):
        mid = None
        for _ in range(3):
            mid = self.store.add_memory("self", "Astra prefers concise answers.",
                                        "self_belief", "ai_extraction")
        self.store.add_memory("self", "Astra prefers concise answers.",
                              "self_belief", "explicit_user_statement")
        mem = self.store.get_memory("self", mid)
        self.assertEqual(mem["type"], "self_belief")
        self.assertTrue(self_memory.is_permanent(mem))


# ---------------------------------------------------------------------
# The sanity check
# ---------------------------------------------------------------------
class TestSanityCheck(unittest.TestCase):
    def test_refuses_implementation_and_metaphor(self):
        for text in ("Astra processes emotional scenes as data sequences.",
                     "Astra's emotions are technical simulations.",
                     "Astra is in chaos mode.",
                     "Astra's personality is generated through pattern fidelity.",
                     "Astra's behavior is fundamentally calculated pattern "
                     "deviation."):
            allowed, _ = self_memory.sanity_check(
                text, mem_type="self_belief",
                authority_source=self_memory.SOURCE_GENERATED)
            self.assertFalse(allowed, text)

    def test_allows_a_grounded_preference(self):
        allowed, reason = self_memory.sanity_check(
            "Astra likes the book she just finished.", mem_type="self_preference",
            authority_source=self_memory.SOURCE_EXPERIENCE)
        self.assertTrue(allowed)
        self.assertEqual(reason, "")

    def test_identity_source_may_establish_a_fact(self):
        allowed, _ = self_memory.sanity_check(
            "Astra is a robot maid.", mem_type="self_fact",
            authority_source=self_memory.SOURCE_IDENTITY)
        self.assertTrue(allowed)


# ---------------------------------------------------------------------
# Conflict handling: source authority, not recency
# ---------------------------------------------------------------------
class TestConflictByAuthority(_SelfMemoryCase):
    def test_identity_outranks_a_newer_generated_statement(self):
        identity = self.add("Astra prefers concise answers.", "self_belief",
                            source="governing_declaration")
        generated = self.add("Astra prefers verbose answers.", "self_belief",
                             source="ai_extraction")
        # The generated statement is newer, but authority decides: the identity
        # record must not be weakened below it.
        self.assertGreaterEqual(self_memory.authority_tier(identity),
                                self_memory.authority_tier(generated))
        self.assertEqual(self.store.get_memory("self", identity["id"])["status"],
                         "active")

    def test_old_opinion_is_history_not_a_command(self):
        # A previous belief can be stored and later changed; the store must not
        # force Astra to keep the old opinion.
        self.add("Astra disliked science fiction.", "self_preference",
                 source="ai_extraction")
        changed = self.add("Astra likes science fiction now.", "self_preference",
                           source="explicit_user_statement")
        self.assertEqual(changed["authority_source"],
                         self_memory.SOURCE_USER_ESTABLISHED)


# ---------------------------------------------------------------------
# Historical vs current, and prompt isolation
# ---------------------------------------------------------------------
class TestHistoricalVsCurrent(_SelfMemoryCase):
    def test_historical_statement_excluded_from_retrieval(self):
        self.store.add_memory("self", "Astra is in chaos mode.", "self_fact",
                              "ai_extraction")
        self.assertEqual(self.store.get_retrievable_memories("self"), [])

    def test_historical_statement_excluded_from_prompt(self):
        self.store.add_memory("self", "Astra's primary role is being a narrator.",
                              "self_belief", "ai_extraction")
        self.assertNotIn("primary role is being a narrator", self.prompt("hello"))

    def test_history_is_still_readable_for_audit(self):
        mid = self.store.add_memory("self", "Astra is in chaos mode.",
                                    "self_fact", "ai_extraction")
        mem = self.store.get_memory("self", mid)
        self.assertEqual(mem["type"], "historical_statement")
        # It is not deleted: it can still be listed by status=None.
        ids = [m["id"] for m in self.store.get_memories("self", status=None)]
        self.assertIn(mid, ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
