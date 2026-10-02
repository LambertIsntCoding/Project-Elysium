"""Tests for derived prepared context: references, relevance, work isolation."""

import shutil
import tempfile
import unittest

from astra import prepared_context as pc
from astra.memory import TripleMemoryStore


class TestPreparedContextStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = pc.PreparedContext(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_references_require_a_source_record(self):
        self.store.replace([{"kind": "memory", "label": "no id"}])
        self.assertEqual(self.store.entries(), [])

    def test_entries_carry_tokens_and_persist(self):
        self.store.replace([{"kind": "memory", "ref_id": "m1",
                             "label": "the lighthouse keeper"}])
        reloaded = pc.PreparedContext(data_dir=self.tmp)
        self.assertEqual(len(reloaded.entries()), 1)
        self.assertTrue(reloaded.entries()[0]["tokens"])

    def test_it_is_not_a_memory_store(self):
        self.store.replace([{"kind": "memory", "ref_id": "m1", "label": "x"}])
        # No memory model file appears; prepared context is its own contract.
        reloaded_store = TripleMemoryStore(data_dir=self.tmp)
        self.assertEqual(reloaded_store.get_memories("self"), [])


class TestRelevance(unittest.TestCase):
    def test_unrelated_entry_is_not_relevant(self):
        entries = [{"ref_id": "m1", "label": "sourdough starter hydration",
                    "tokens": ["sourdough", "starter", "hydration"]}]
        self.assertEqual(pc.relevance(entries, "what is the weather"), [])

    def test_overlapping_entry_is_relevant(self):
        entries = [{"ref_id": "m1", "label": "sourdough starter hydration",
                    "tokens": ["sourdough", "starter", "hydration"]}]
        rel = pc.relevance(entries, "how is my sourdough starter doing")
        self.assertEqual(len(rel), 1)

    def test_block_marks_itself_derived(self):
        block = pc.render_block([{"ref_id": "m1", "kind": "memory", "label": "x"}])
        self.assertIn("DERIVED", block)


class TestBuildFromStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_references_existing_records_only(self):
        mem_id = self.store.add_memory(
            target_model="roum", content="Roum is learning to sail.",
            mem_type="explicit_fact", source="explicit_user_statement")
        entries = pc.build(self.store)
        self.assertTrue(entries)
        # Every entry points at a real record id.
        ids = {e["ref_id"] for e in entries}
        self.assertIn(mem_id, ids)

    def test_work_scoping_is_preserved(self):
        self.store.add_question("What does the lighthouse mean?",
                                work_id="book-a", tags=["work:book-a"])
        entries = [e for e in pc.build(self.store) if e["kind"] == pc.REF_QUESTION]
        self.assertTrue(all(e.get("work_id") in (None, "book-a") for e in entries))


if __name__ == "__main__":
    unittest.main(verbosity=2)
