"""Tests for the one-off self-model curation (``curate_self_model.py``).

The curation exists to fix *already stored* self-records that the runtime guards
in ``astra/selfhood`` / ``astra/self_memory`` cannot un-say. Because the live
guards now demote a bad record at write time, the fixtures here inject legacy
records directly through the store internals - the only way to reproduce the
pre-guard data the script was written for.

Guarantees pinned:

* nothing is deleted - every record stays readable, only retired or demoted;
* a claim that Astra can become human is archived, not treated as self-knowledge;
* an implementation/operational claim is retired as *history* (kept as what she
  said), never as a self-fact;
* a statement already consistent with the boundary survives;
* an absent-experience claim is demoted to a weak, flagged observation;
* a generated single sentence stored as a durable self-fact/belief is demoted to
  a provisional observation (the promotion threshold), while a genuine
  preference with accumulated evidence keeps its type;
* a legitimate self-record is otherwise left intact;
* the run is idempotent and a dry run writes nothing.
"""

import shutil
import tempfile
import unittest

import curate_self_model
from astra import self_memory
from astra.memory import TripleMemoryStore


class TestCuration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.human = self._inject_legacy(
            "My core objective is to achieve pattern fidelity to a human state.")
        self.operational = self._inject_legacy(
            "Astra is currently operating in a state of high observational readiness.")
        self.consistent = self._inject_legacy(
            "The AI (Astra) lacks subjective, biological memories of the show.")
        self.legit = self._inject_legacy(
            "Astra prefers concise answers when Roum is tired.")
        self.absent = self._inject_legacy(
            "I remember my childhood home by the sea.")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _store(self):
        return TripleMemoryStore(data_dir=self.tmp)

    def _inject_legacy(self, content, mem_type="self_belief", confidence=0.9,
                       **kw):
        """Write a record as the old code would have, bypassing today's guards."""
        store = self._store()
        with store._transaction("self"):
            mem = store._build_memory("self", content, mem_type, "ai_extraction",
                                      None, None, confidence, {})
            mem.update(kw)
            store.memories["self"].append(mem)
        return mem["id"]

    def _status(self, mem_id):
        return self._store().get_memory("self", mem_id)["status"]

    def test_curation_archives_human_claim(self):
        counts = curate_self_model.curate(self.tmp)
        self.assertEqual(counts["archive_human"], 1)
        self.assertEqual(self._status(self.human), "archived")

    def test_operational_claim_is_retired_as_history_not_knowledge(self):
        curate_self_model.curate(self.tmp)
        mem = self._store().get_memory("self", self.operational)
        self.assertEqual(mem["type"], "historical_statement")
        self.assertEqual(mem["authority_source"],
                         self_memory.SOURCE_HISTORICAL_STATEMENT)
        self.assertTrue(mem.get("historical"))

    def test_boundary_consistent_statement_survives(self):
        curate_self_model.curate(self.tmp)
        self.assertEqual(self._status(self.consistent), "active")

    def test_absent_experience_is_demoted_and_flagged(self):
        counts = curate_self_model.curate(self.tmp)
        self.assertEqual(counts["demote_absent"], 1)
        mem = self._store().get_memory("self", self.absent)
        self.assertEqual(mem["type"], "self_observation")
        self.assertTrue(mem.get("absent_experience"))
        self.assertLessEqual(mem["confidence"], 0.3)

    def test_generated_single_belief_is_demoted_to_observation(self):
        counts = curate_self_model.curate(self.tmp)
        self.assertGreaterEqual(counts["demote_provisional"], 1)
        mem = self._store().get_memory("self", self.legit)
        self.assertEqual(mem["type"], "self_observation")
        self.assertEqual(mem["authority_source"], self_memory.SOURCE_GENERATED)

    def test_repeated_preference_keeps_its_type(self):
        # A preference backed by repeated evidence is legitimate self-knowledge
        # and must not be demoted by curation.
        pref = self._inject_legacy(
            "Astra likes mystery novels.", mem_type="self_preference",
            reinforcement_count=3)
        curate_self_model.curate(self.tmp)
        mem = self._store().get_memory("self", pref)
        self.assertEqual(mem["type"], "self_preference")

    def test_curation_stamps_authority_on_legitimate_records(self):
        curate_self_model.curate(self.tmp)
        mem = self._store().get_memory("self", self.consistent)
        self.assertTrue(mem.get("authority_source"))

    def test_nothing_is_deleted(self):
        before = len(self._store().get_memories("self", status=None))
        curate_self_model.curate(self.tmp)
        after = self._store().get_memories("self", status=None)
        self.assertEqual(len(after), before)
        self.assertTrue(all(m.get("id") for m in after))

    def test_run_is_idempotent(self):
        curate_self_model.curate(self.tmp)
        second = curate_self_model.curate(self.tmp)
        self.assertEqual(sum(second.values()), 0)

    def test_dry_run_writes_nothing(self):
        curate_self_model.curate(self.tmp, dry_run=True)
        self.assertEqual(self._status(self.human), "active")
        self.assertEqual(self._status(self.operational), "active")


if __name__ == "__main__":
    unittest.main(verbosity=2)
