"""Compact storage layout and the date/time memory index.

Two behaviours are pinned here, both of them about *keeping the store small
without changing what it means*:

* the on-disk record omits fields a reader can derive or default, and reads
  restore them, so every existing caller keeps working;
* memories can be grouped by day/month/year from the records already in memory,
  with no extra storage and no prompt-path cost.
"""
import json
import os
import tempfile
import unittest

from astra.memory import (
    TIMELINE_GRANULARITIES,
    TripleMemoryStore,
    compact_record,
    timeline,
    timeline_summary,
)

TS = "2026-09-02T10:00:00+00:00"


class TestCompactLayout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _raw(self, target="roum"):
        with open(os.path.join(self.tmp, f"{target}_model.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_compact_record_drops_only_derivable_or_default_fields(self):
        record = {
            "id": "mem_1", "content": "x", "type": "explicit_fact", "source": "t",
            "status": "active", "timestamp": TS, "confidence": 0.6,
            "created_at": TS, "last_used": TS, "last_reinforced": TS,
            "use_count": 0, "contradiction_count": 0, "reinforcement_count": 1,
            "tags": [], "keywords": [], "contradicts": [],
            "supersedes": None, "superseded_by": None,
            "effective_strength": 0.42, "source_type": "tier_1",
        }
        out = compact_record(record)
        for dropped in ("created_at", "last_used", "last_reinforced", "use_count",
                        "contradiction_count", "reinforcement_count", "tags",
                        "keywords", "contradicts", "supersedes", "superseded_by",
                        "effective_strength", "source_type"):
            self.assertNotIn(dropped, out)
        # The meaningful core is untouched.
        self.assertEqual(out["content"], "x")
        self.assertEqual(out["confidence"], 0.6)
        self.assertEqual(out["status"], "active")

    def test_meaningful_values_are_kept(self):
        record = {
            "id": "mem_1", "timestamp": TS, "content": "x",
            "last_used": "2026-09-30T00:00:00+00:00",   # differs -> kept
            "use_count": 4, "reinforcement_count": 3,   # non-default -> kept
            "contradiction_count": 2,
            "tags": ["a"], "keywords": ["b"], "contradicts": ["mem_2"],
            "superseded_by": "mem_2",
        }
        out = compact_record(record)
        for kept in ("last_used", "use_count", "reinforcement_count",
                     "contradiction_count", "tags", "keywords", "contradicts",
                     "superseded_by"):
            self.assertIn(kept, out)
        self.assertNotIn("created_at", out)  # still equal to timestamp -> dropped

    def test_files_on_disk_are_compact_but_reads_are_complete(self):
        mid = self.store.add_memory("roum", "Roum likes tea.", "explicit_fact",
                                    "explicit_user_statement", tags=["drinks"])
        on_disk = self._raw()[0]
        # Compact: no null pointers, no derived duplicates, no no-op defaults.
        self.assertNotIn("supersedes", on_disk)
        self.assertNotIn("superseded_by", on_disk)
        self.assertNotIn("effective_strength", on_disk)
        self.assertNotIn("source_type", on_disk)
        self.assertNotIn("use_count", on_disk)
        self.assertIn("tags", on_disk)  # a real value stays

        # Reads restore the contract, so a caller may still subscript directly.
        mem = self.store.get_memory("roum", mid)
        self.assertEqual(mem["use_count"], 0)
        self.assertIsNotNone(mem["created_at"])
        self.assertIsNone(mem["superseded_by"])
        self.assertEqual(mem["reinforcement_count"], 1)
        self.assertEqual(mem["tags"], ["drinks"])

    def test_used_memory_persists_its_counters(self):
        mid = self.store.add_memory("roum", "Roum likes tea.", "explicit_fact",
                                    "explicit_user_statement")
        self.store.record_use([mid])
        reopened = TripleMemoryStore(data_dir=self.tmp)
        self.assertGreaterEqual(reopened.get_memory("roum", mid)["use_count"], 1)
        self.assertIsNotNone(reopened.get_memory("roum", mid)["last_used"])

    def test_dense_format_is_smaller_than_pretty(self):
        for i in range(30):
            self.store.add_memory("roum", f"fact number {i} about Roum", "explicit_fact",
                                  "explicit_user_statement", tags=["topic"],
                                  keywords=["fact", str(i)])
        path = os.path.join(self.tmp, "roum_model.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pretty = len(json.dumps(data, indent=2, ensure_ascii=False))
        self.assertLess(os.path.getsize(path), pretty)
        # And it is still plain, readable JSON.
        with open(path, encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)), 30)

    def test_reads_still_never_write(self):
        self.store.add_memory("roum", "a fact", "explicit_fact", "t")

        def hashes():
            out = {}
            for name in sorted(os.listdir(self.tmp)):
                if name.endswith(".json"):
                    with open(os.path.join(self.tmp, name), "rb") as f:
                        out[name] = f.read()
            return out

        before = hashes()
        reopened = TripleMemoryStore(data_dir=self.tmp)
        reopened.get_memories("roum", status=None)
        reopened.get_timeline()
        reopened.stats()
        self.assertEqual(hashes(), before)


class TestTimeline(unittest.TestCase):
    def test_grouping_by_day_month_year(self):
        mems = [
            {"id": "a", "timestamp": "2026-08-15T10:00:00+00:00"},
            {"id": "b", "timestamp": "2026-09-02T10:00:00+00:00"},
            {"id": "c", "timestamp": "2026-09-02T23:00:00+00:00"},
            {"id": "d", "timestamp": "2025-12-31T00:00:00+00:00"},
        ]
        self.assertEqual(timeline_summary(mems, "month"),
                         [("2026-09", 2), ("2026-08", 1), ("2025-12", 1)])
        self.assertEqual(timeline_summary(mems, "year"),
                         [("2026", 3), ("2025", 1)])
        self.assertEqual(timeline_summary(mems, "day"),
                         [("2026-09-02", 2), ("2026-08-15", 1), ("2025-12-31", 1)])

    def test_undated_records_go_last_and_never_crash(self):
        mems = [{"id": "a", "timestamp": "2026-09-02T10:00:00+00:00"},
                {"id": "b", "timestamp": None},
                {"id": "c"}]
        summary = timeline_summary(mems, "month")
        self.assertEqual(summary[-1], ("unknown", 2))
        self.assertEqual(summary[0], ("2026-09", 1))

    def test_invalid_granularity_is_rejected(self):
        with self.assertRaises(ValueError):
            timeline([], "fortnight")

    def test_store_timeline_and_period_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TripleMemoryStore(data_dir=tmp)
            store.add_memory("roum", "aug", "explicit_fact", "t",
                             timestamp="2026-08-15T10:00:00+00:00")
            store.add_memory("roum", "sep", "explicit_fact", "t",
                             timestamp="2026-09-02T10:00:00+00:00")
            store.add_memory("self", "sep self", "self_observation", "t",
                             timestamp="2026-09-02T11:00:00+00:00")
            self.assertEqual(store.get_timeline(granularity="month"),
                             [("2026-09", 2), ("2026-08", 1)])
            self.assertEqual(store.get_timeline("roum", granularity="month"),
                             [("2026-09", 1), ("2026-08", 1)])
            self.assertEqual(len(store.get_period("2026-09")), 2)
            self.assertEqual(len(store.get_period("2026")), 3)
            self.assertEqual([m["content"] for m in store.get_period("2026-09-02")],
                             ["sep self", "sep"])

    def test_timeline_adds_no_storage(self):
        """The index is computed, not persisted - the point of 'not a whole thing'."""
        with tempfile.TemporaryDirectory() as tmp:
            store = TripleMemoryStore(data_dir=tmp)
            store.add_memory("roum", "a", "explicit_fact", "t")
            before = sorted(os.listdir(tmp))
            store.get_timeline()
            store.get_period("2026-09")
            self.assertEqual(sorted(os.listdir(tmp)), before)


class TestTimelineCommand(unittest.TestCase):
    def test_help_and_dispatch(self):
        import main as cli
        from astra.memory import CommandStore
        from astra.orchestrator import CompanionOrchestrator

        self.assertIn("/timeline", cli.HELP_TEXT)
        self.assertEqual(TIMELINE_GRANULARITIES, ("day", "month", "year"))

        with tempfile.TemporaryDirectory() as tmp:
            store = TripleMemoryStore(data_dir=tmp)
            store.add_memory("roum", "sep", "explicit_fact", "t",
                             timestamp="2026-09-02T10:00:00+00:00")
            orch = CompanionOrchestrator(store, config_dir="./config")
            emitted = []
            session = cli.ChatSession(orch, CommandStore(data_dir=tmp),
                                      output_fn=emitted.append, input_fn=lambda _: "")
            session._handle_slash("/timeline")
            text = "\n".join(emitted)
            self.assertIn("MEMORY TIMELINE", text)
            self.assertIn("2026-09 : 1", text)

            emitted.clear()
            session._handle_slash("/timeline day roum")
            self.assertIn("day, roum", "\n".join(emitted))


if __name__ == "__main__":
    unittest.main()
