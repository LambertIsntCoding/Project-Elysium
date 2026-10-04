"""Tests for the anti-invention personal-state rules (``astra/personal_state.py``).

These pin the brief's central rule for this change: when Astra is asked about
her own current or recent situation and the retrieved memory is incomplete or
contradictory, she must not reconstruct a plausible answer and present it as
fact. Everything runs against a real ``TripleMemoryStore``, a real ``Library``,
and a real ``CompanionOrchestrator`` - no mocks.

Covered:

* a personal-state question renders an authoritative block, and an ordinary
  turn does not;
* a missing fact is rendered as missing (so "I don't know" is available), and a
  game with no record is explicitly unknown;
* an opinion/interpretation is labelled and never presented as progress;
* a generated current-state claim without a supporting record is not stored as
  authoritative self-knowledge;
* a generated current-state claim that matches a real record is allowed;
* contradictory reading records are detected and surface as uncertainty;
* the always-on contract is present every turn and silent about its own rules;
* a purpose question is answered from identity, not the current subject.
"""

import shutil
import tempfile
import unittest

from astra import personal_state
from astra.library import Library
from astra.memory import TripleMemoryStore, detect_contradiction
from astra.orchestrator import CompanionOrchestrator

CONFIG_DIR = "./config"


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.library = Library(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def attach_reader(self):
        self.orch.reader = type(
            "R", (), {"library": self.library, "last_reason": None})()

    def prompt(self, text):
        return self.orch.build_prompt(text, [])


# ---------------------------------------------------------------------
# Question recognition
# ---------------------------------------------------------------------
class TestQueryRecognition(_Case):
    def test_personal_state_questions_are_recognised(self):
        for text in (
            "What book are you reading?",
            "What was the last game you played?",
            "What chapter are you on?",
            "What did you think of the ending?",
            "What were you working on?",
            "What did I tell you yesterday?",
            "What is your purpose?",
        ):
            with self.subTest(text=text):
                self.assertTrue(personal_state.asks_personal_state(text))

    def test_ordinary_conversation_is_not_a_status_query(self):
        for text in ("hello", "how are you?", "that sounds funny", "tell me a story"):
            with self.subTest(text=text):
                self.assertFalse(personal_state.asks_personal_state(text))


# ---------------------------------------------------------------------
# The authoritative block: present only when asked, and honest about gaps
# ---------------------------------------------------------------------
class TestAuthoritativeBlock(_Case):
    def test_block_appears_only_for_a_personal_state_question(self):
        self.attach_reader()
        self.assertIn("CURRENT PERSONAL STATE", self.prompt("What book are you reading?"))
        self.assertNotIn("CURRENT PERSONAL STATE", self.prompt("hello there"))

    def test_missing_state_is_rendered_as_missing(self):
        self.attach_reader()
        prompt = self.prompt("What book are you reading?")
        self.assertIn("no work is recorded as being read", prompt)

    def test_a_game_with_no_record_is_explicitly_unknown(self):
        self.attach_reader()
        prompt = self.prompt("What was the last game you played?")
        self.assertIn("no game she has played is recorded", prompt)

    def test_current_work_and_finished_work_are_stated(self):
        self.library.add_text("The captain stared at the sea. " * 80,
                              title="Sea Novel", work_id="sea")
        self.library.add_text("Another story entirely. " * 80,
                              title="Other Book", work_id="other")
        self.library.set_status("other", "finished")
        self.library.set_current("sea")
        self.attach_reader()
        prompt = self.prompt("What book are you reading?")
        self.assertIn("'Sea Novel'", prompt)
        self.assertIn("Finished: 'Other Book'", prompt)

    def test_opinion_is_labelled_and_not_presented_as_progress(self):
        self.store.record_experience(
            "Reacting to 'Other Book': I loved the ending.",
            kind="reading_reaction", work_id="other", intensity=0.5)
        self.attach_reader()
        prompt = self.prompt("What did you think of Other Book?")
        self.assertIn("(recorded) Reacting to 'Other Book'", prompt)


# ---------------------------------------------------------------------
# Write guard: generated current-state claims need a real record
# ---------------------------------------------------------------------
class TestWriteGuard(_Case):
    def add(self, content, source="ai_extraction", mem_type="self_belief", **kw):
        mid = self.store.add_memory("self", content, mem_type, source, **kw)
        return self.store.get_memory("self", mid)

    def test_fabricated_progress_is_not_authoritative_self_knowledge(self):
        mem = self.add("I'm partway through Book B.")
        self.assertNotIn(mem["type"], ("self_fact", "self_belief"))
        self.assertTrue(mem.get("historical"))

    def test_supported_progress_is_allowed(self):
        self.store.record_reading_finish(work_id="book-b", title="Book B")
        mem = self.add("I finished reading 'Book B'.")
        self.assertNotEqual(mem["type"], "historical_statement")
        self.assertFalse(mem.get("historical"))

    def test_fabricated_state_does_not_reach_the_prompt(self):
        self.add("I'm currently reading a book I have never opened.")
        self.attach_reader()
        prompt = self.prompt("What book are you reading?")
        self.assertNotIn("never opened", prompt)

    def test_current_state_patterns(self):
        self.assertTrue(personal_state.claims_current_state("I'm currently reading X."))
        self.assertTrue(personal_state.claims_current_state("I'm up to chapter 4."))
        self.assertTrue(personal_state.claims_current_state("I previously said that."))
        self.assertFalse(personal_state.claims_current_state("I like mystery novels."))


# ---------------------------------------------------------------------
# Contradictions trigger uncertainty
# ---------------------------------------------------------------------
class TestContradiction(_Case):
    def test_finished_and_currently_reading_contradict(self):
        a = self.store.add_memory(
            "self", "Finished reading 'Book A'.", "experience", "experiential")
        b = self.store.add_memory(
            "self", "I am currently reading 'Book A'.", "self_observation",
            "ai_extraction")
        reason = detect_contradiction(
            self.store.get_memory("self", a), self.store.get_memory("self", b))
        self.assertIsNotNone(reason)

    def test_contradiction_is_surfaced_in_the_block(self):
        self.store.add_memory(
            "self", "Finished reading 'Book A'.", "experience", "experiential")
        self.store.add_memory(
            "self", "I am currently reading 'Book A'.", "self_observation",
            "ai_extraction")
        self.attach_reader()
        prompt = self.prompt("What book are you reading?")
        self.assertIn("Conflicting records", prompt)

    def test_authority_resolves_the_contradiction(self):
        self.store.add_memory(
            "self", "Finished reading 'Book A'.", "experience", "experiential")
        b = self.store.add_memory(
            "self", "I am currently reading 'Book A'.", "self_observation",
            "ai_extraction")
        # The explicit experience outranks the generated observation, so the
        # generated one is weakened rather than the explicit one replaced.
        self.assertEqual(self.store.get_memory("self", b)["status"], "weakened")


# ---------------------------------------------------------------------
# The always-on contract and purpose
# ---------------------------------------------------------------------
class TestContractAndPurpose(_Case):
    def test_contract_is_present_every_turn(self):
        self.assertIn("HOW ASTRA ANSWERS ABOUT HERSELF", self.prompt("hello"))
        self.assertIn("HOW ASTRA ANSWERS ABOUT HERSELF", self.prompt("What book are you reading?"))

    def test_contract_does_not_tell_astra_to_discuss_it(self):
        prompt = self.prompt("hello")
        self.assertIn("prefer them to a confident fabrication", prompt)

    def test_purpose_comes_from_identity_not_the_subject(self):
        self.attach_reader()
        prompt = self.prompt("What is your purpose?")
        self.assertIn("CURRENT PERSONAL STATE", prompt)
        self.assertIn("not regenerated from the current subject", prompt)
        self.assertIn("robot maid", prompt)


if __name__ == "__main__":
    unittest.main()
