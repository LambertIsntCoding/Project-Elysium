"""Tests for the turn write-batch and the orchestrator read cache.

These lock in the performance work: a turn's small writes coalesce into one
save per touched model but remain durable at the turn boundary, and a write
between turns is always visible to the next prompt build.
"""

import os
import shutil
import tempfile
import unittest

from astra.memory import TripleMemoryStore, atomic_save
from astra.orchestrator import CompanionOrchestrator


class TestTurnBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.mid = self.store.add_memory(
            target_model="roum", content="RouM likes strong coffee",
            mem_type="explicit_fact", source="explicit_user_statement")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _writes(self, fn):
        seen = []
        import astra.memory as mem
        original = mem.atomic_save

        def spy(path, *a, **k):
            seen.append(os.path.basename(path))
            return original(path, *a, **k)

        mem.atomic_save = spy
        try:
            fn()
        finally:
            mem.atomic_save = original
        return seen

    def test_batch_coalesces_writes_and_persists_on_exit(self):
        def work():
            with self.store.turn_batch():
                self.store.record_use([self.mid])
                self.store.record_use([self.mid])
                self.store.mark_present()

        writes = self._writes(work)
        # One write per touched model plus one for presence, not one per call.
        self.assertEqual(sorted(writes), ["presence.json", "roum_model.json"])

    def test_batched_use_is_durable_after_exit(self):
        with self.store.turn_batch():
            self.store.record_use([self.mid])
        reopened = TripleMemoryStore(data_dir=self.tmp)
        self.assertGreaterEqual(reopened.get_memory("roum", self.mid)["use_count"], 1)

    def test_batched_presence_is_durable_after_exit(self):
        stamp = None
        with self.store.turn_batch():
            stamp = self.store.mark_present()
            # Visible to a reader while still batched (in-memory).
            self.assertEqual(self.store.last_present(), stamp)
        reopened = TripleMemoryStore(data_dir=self.tmp)
        self.assertEqual(reopened.last_present(), stamp)

    def test_writes_outside_a_batch_persist_immediately(self):
        writes = self._writes(lambda: self.store.record_use([self.mid]))
        self.assertEqual(writes, ["roum_model.json"])

    def test_nested_batches_flush_once(self):
        def work():
            with self.store.turn_batch():
                self.store.record_use([self.mid])
                with self.store.turn_batch():
                    self.store.record_use([self.mid])

        writes = self._writes(work)
        self.assertEqual(writes, ["roum_model.json"])

    def test_exception_still_restores_and_flushes(self):
        with self.assertRaises(RuntimeError):
            with self.store.turn_batch():
                self.store.record_use([self.mid])
                raise RuntimeError("boom")
        # The turn's writes are still committed; only the failing transaction
        # would roll back, and none did here.
        self.assertGreaterEqual(self.store.get_memory("roum", self.mid)["use_count"], 1)


class TestReadCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir="./config")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_between_builds_is_visible(self):
        self.store.add_memory(target_model="roum", content="RouM has a blue bicycle",
                              mem_type="explicit_fact", source="explicit_user_statement")
        p1, _ = self.orch.build_prompt_with_diagnostics("blue bicycle", [], detailed=False)
        self.assertIn("blue bicycle", p1)
        self.store.add_memory(target_model="roum", content="RouM has a red canoe",
                              mem_type="explicit_fact", source="explicit_user_statement")
        p2, _ = self.orch.build_prompt_with_diagnostics("red canoe", [], detailed=False)
        self.assertIn("red canoe", p2)

    def test_build_is_deterministic_with_cache(self):
        self.store.add_memory(target_model="roum", content="RouM enjoys chess",
                              mem_type="explicit_fact", source="explicit_user_statement")
        p1, _ = self.orch.build_prompt_with_diagnostics("chess", [], detailed=False)
        p2, _ = self.orch.build_prompt_with_diagnostics("chess", [], detailed=False)
        self.assertEqual(p1, p2)

    def test_repeated_reads_do_not_re_copy_from_store(self):
        self.store.add_memory(target_model="roum", content="RouM enjoys chess",
                              mem_type="explicit_fact", source="explicit_user_statement")
        calls = {"n": 0}
        original = self.store.get_memories

        def spy(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        self.store.get_memories = spy
        self.orch.build_prompt_with_diagnostics("chess", [], detailed=False)
        # One read per model (roum, self, relationship) plus the experience
        # view, not one read per prompt section.
        self.assertLessEqual(calls["n"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
