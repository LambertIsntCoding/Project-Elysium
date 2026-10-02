"""Tests for the Active/Sleep runtime state machine and its persistence."""

import os
import shutil
import tempfile
import threading
import unittest

from astra import runtime_state as rs


class TestTransitionIsPure(unittest.TestCase):
    def test_blank_state_is_active(self):
        state = rs.blank_state()
        self.assertEqual(state["state"], rs.ACTIVE)
        self.assertTrue(rs.is_active(state))

    def test_human_command_sleeps_and_human_wake_wakes(self):
        state = rs.blank_state()
        state = rs.transition(state, rs.SLEEP, reason=rs.REASON_HUMAN_COMMAND)
        self.assertTrue(rs.is_sleeping(state))
        state = rs.transition(state, rs.ACTIVE, reason=rs.REASON_HUMAN_WAKE)
        self.assertTrue(rs.is_active(state))
        self.assertEqual(state["wake_reason"], rs.REASON_HUMAN_WAKE)
        self.assertEqual(state["wake_events"], 1)

    def test_background_task_can_never_wake(self):
        # REASON_TASK is deliberately not a wake cause.
        state = rs.blank_state()
        state = rs.transition(state, rs.SLEEP, reason=rs.REASON_HUMAN_COMMAND)
        after = rs.transition(state, rs.ACTIVE, reason=rs.REASON_TASK)
        self.assertTrue(rs.is_sleeping(after))

    def test_silence_cannot_sleep(self):
        # There is no reason value that means "no message arrived"; a bare
        # transition without a human/lifecycle cause is a no-op.
        state = rs.blank_state()
        after = rs.transition(state, rs.SLEEP, reason="idle_timeout_guess")
        self.assertTrue(rs.is_active(after))

    def test_noop_transition_is_identity(self):
        state = rs.blank_state()
        self.assertEqual(rs.transition(state, rs.ACTIVE, reason=rs.REASON_HUMAN_WAKE),
                         rs.coerce_state(state))

    def test_coerce_degrades_bad_input_to_active(self):
        self.assertEqual(rs.coerce_state("nonsense")["state"], rs.ACTIVE)
        self.assertEqual(rs.coerce_state({"state": "banana"})["state"], rs.ACTIVE)


class TestRuntimeStateStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = rs.RuntimeStateStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_persists_across_reload(self):
        self.store.sleep()
        reloaded = rs.RuntimeStateStore(data_dir=self.tmp)
        self.assertTrue(reloaded.is_sleeping())
        self.assertEqual(reloaded.current()["reason"], rs.REASON_HUMAN_COMMAND)

    def test_wake_persists(self):
        self.store.sleep()
        self.store.wake()
        reloaded = rs.RuntimeStateStore(data_dir=self.tmp)
        self.assertTrue(reloaded.is_active())
        self.assertEqual(reloaded.current()["wake_reason"], rs.REASON_HUMAN_WAKE)

    def test_state_is_its_own_file_not_a_memory(self):
        self.store.sleep()
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "runtime_state.json")))
        # It never creates memory model files.
        for name in ("roum_model.json", "self_model.json", "relationship_model.json"):
            self.assertFalse(os.path.exists(os.path.join(self.tmp, name)))

    def test_concurrent_transitions_are_serialized(self):
        errors = []

        def worker(i):
            try:
                if i % 2:
                    self.store.sleep()
                else:
                    self.store.wake()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        reloaded = rs.RuntimeStateStore(data_dir=self.tmp)
        self.assertIn(reloaded.current()["state"], rs.RUNTIME_STATES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
