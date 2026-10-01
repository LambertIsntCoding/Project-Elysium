"""Tests for the one-off self-model curation (``curate_self_model.py``).

The curation exists to fix *already stored* self-records that the runtime guards
in ``astra/selfhood`` cannot un-say. Because the live guards now demote a bad
record at write time, the fixtures here inject legacy records directly through
the store internals - the only way to reproduce the pre-guard data the script
was written for.

Guarantees pinned:

* nothing is deleted - every record stays readable, only retired or demoted;
* a claim that Astra can become human is retired, not treated as self-knowledge;
* operational/implementation chatter is retired, but a statement that is
  already consistent with the boundary survives;
* an absent-experience claim is demoted to a weak, flagged observation;
* a legitimate self-record is left completely untouched;
* the run is idempotent.
"""

import shutil
import tempfile
import unittest

import curate_self_model
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

    def _inject_legacy(self, content, mem_type="self_belief", confidence=0.9):
        """Write a record as the old code would have, bypassing today's guards."""
        store = self._store()
        with store._transaction("self"):
            mem = store._build_memory("self", content, mem_type, "ai_extraction",
                                      None, None, confidence, {})
            store.memories["self"].append(mem)
        return mem["id"]

    def _status(self, mem_id):
        return self._store().get_memory("self", mem_id)["status"]

    def test_curation_retires_human_and_operational_claims(self):
        counts = curate_self_model.curate(self.tmp)
        self.assertEqual(counts["archive_human"], 1)
        self.assertEqual(counts["archive_operational"], 1)
        self.assertEqual(self._status(self.human), "archived")
        self.assertEqual(self._status(self.operational), "archived")

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

    def test_legitimate_record_is_untouched(self):
        curate_self_model.curate(self.tmp)
        self.assertEqual(self._status(self.legit), "active")
        self.assertEqual(self._store().get_memory("self", self.legit)["type"],
                         "self_belief")

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
