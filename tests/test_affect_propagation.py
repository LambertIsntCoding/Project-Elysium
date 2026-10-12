"""Tests that Astra's emotional state *causes* work, not merely a mood report.

These verify the propagation the brief asks for: the affect -> behaviour
modulation actually changes which activities are selected, how many are taken
on, and whether the reader reads - rather than only changing the wording of a
reply. They use the real scheduler, planner, reader gate and machine state; no
network, no model.
"""

import shutil
import tempfile
import unittest

from astra import behavior


class _Sleeping:
    def is_sleeping(self):
        return True


class _Ledger:
    def __init__(self):
        self.entries_ = []

    def record(self, **entry):
        self.entries_.append(entry)

    def entries(self, limit=None):
        return self.entries_[-limit:] if limit else list(self.entries_)


class TestActivityBias(unittest.TestCase):
    def test_sad_state_lowers_effort_and_take_on(self):
        sad = behavior.activity_bias(behavior.from_affect({"sad": 0.9, "weary": 0.7}))
        glad = behavior.activity_bias(behavior.from_affect({"happy": 0.8, "curiosity": 0.7}))
        self.assertLess(sad["effort"], glad["effort"])
        self.assertLess(sad["take_on"], glad["take_on"])
        self.assertTrue(sad["prefer_light"])
        self.assertFalse(glad["prefer_light"])

    def test_bias_is_bounded_and_neutral_is_even(self):
        for state in ({}, {"sad": 1.0}, {"happy": 1.0, "weary": 1.0}):
            bias = behavior.activity_bias(behavior.from_affect(state))
            for key in ("effort", "take_on"):
                self.assertGreaterEqual(bias[key], 0.0)
                self.assertLessEqual(bias[key], 1.0)
        neutral = behavior.activity_bias(behavior.from_affect({}))
        self.assertFalse(neutral["prefer_light"])

    def test_reading_modifiers_are_worsened_by_sadness(self):
        sad = behavior.reading_modifiers(behavior.from_affect({"sad": 0.9, "weary": 0.9}))
        neutral = behavior.reading_modifiers(behavior.from_affect({}))
        self.assertLess(sad["concentration"], neutral["concentration"])
        self.assertGreater(sad["weary"], neutral["weary"])


class TestSchedulerHonoursState(unittest.TestCase):
    def setUp(self):
        from astra import tasks as T
        from astra.scheduler import Scheduler

        self.T = T
        self.tmp = tempfile.mkdtemp()
        self.tasks = T.TaskStore(data_dir=self.tmp)
        self.ledger = _Ledger()
        self.sched = Scheduler(
            task_store=self.tasks, ledger=self.ledger, runtime_state=_Sleeping(),
            process_names_fn=lambda: [],
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _register(self):
        for kind in self.T.TASK_KINDS:
            self.sched.register(kind, lambda task: {"status": self.T.COMPLETED})

    def test_low_intensity_runs_only_light_work(self):
        self._register()
        self.tasks.enqueue(self.T.BackgroundTask(kind=self.T.KIND_READING,
                                                 dedupe_key="r", priority=90))
        self.tasks.enqueue(self.T.BackgroundTask(kind=self.T.KIND_MEMORY_MAINTENANCE,
                                                 dedupe_key="m", priority=10))
        self.sched.intensity_provider = lambda: 0.05
        report = self.sched.run_cycle()
        ran = {self.tasks.get(tid).kind for tid in
               [t.id for t in self.tasks.all() if t.status == self.T.COMPLETED]}
        self.assertEqual(ran, {self.T.KIND_MEMORY_MAINTENANCE})
        self.assertTrue(any(s["reason"] == "low_energy" for s in report["skipped"]))

    def test_strong_affect_allows_intensive_work(self):
        self._register()
        self.tasks.enqueue(self.T.BackgroundTask(kind=self.T.KIND_READING,
                                                 dedupe_key="r", priority=90))
        self.sched.intensity_provider = lambda: 0.9
        report = self.sched.run_cycle()
        self.assertEqual(report["tasks"], 1)

    def test_cold_machine_runs_only_light_work(self):
        self._register()
        self.tasks.enqueue(self.T.BackgroundTask(kind=self.T.KIND_TEST_RUN,
                                                 dedupe_key="t", priority=90))
        self.tasks.enqueue(self.T.BackgroundTask(kind=self.T.KIND_MEMORY_REVIEW,
                                                 dedupe_key="mr", priority=10))
        self.sched.machine_ready_provider = lambda: False
        report = self.sched.run_cycle()
        ran = {t.kind for t in self.tasks.all() if t.status == self.T.COMPLETED}
        self.assertEqual(ran, {self.T.KIND_MEMORY_REVIEW})
        self.assertTrue(any(s["reason"] == "machine_cold" for s in report["skipped"]))

    def test_no_provider_means_state_changes_nothing(self):
        self._register()
        self.tasks.enqueue(self.T.BackgroundTask(kind=self.T.KIND_READING,
                                                 dedupe_key="r", priority=90))
        self.assertIsNone(self.sched.intensity())
        report = self.sched.run_cycle()
        self.assertEqual(report["tasks"], 1)


class TestPlannerHonoursState(unittest.TestCase):
    def setUp(self):
        from astra import tasks as T
        from astra.activity import ActivityPlanner

        self.T = T
        self.tmp = tempfile.mkdtemp()
        self.tasks = T.TaskStore(data_dir=self.tmp)
        self.planner = ActivityPlanner(self.tasks, store=None, library=None,
                                       ledger=_Ledger(), config={"max_new_per_pass": 2})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_sad_state_takes_on_fewer_activities(self):
        # A store that reports a strongly sad state.
        class Store:
            def current_affect(self):
                return {"sad": 0.9, "weary": 0.7}
        self.planner.store = Store()
        report = self.planner.plan()
        self.assertEqual(report["decision"], "idle")
        self.assertIn("low energy", report["reason"])

    def test_a_neutral_state_may_take_on_the_bound(self):
        baseline = self.planner.plan()
        self.assertEqual(baseline["decision"], "plan")
        self.assertLessEqual(len(baseline["proposals"]), 2)
        self.assertGreater(len(baseline["proposals"]), 0)

    def test_bias_is_reported_with_the_decision(self):
        class Store:
            def current_affect(self):
                return {"happy": 0.8, "curiosity": 0.7}
        self.planner.store = Store()
        report = self.planner.plan()
        self.assertIn("bias", report)


class TestReaderGateHonoursState(unittest.TestCase):
    """A sad/weary state can stand the reader down, through the one gate."""

    def test_sad_modulation_stands_reading_down(self):
        from astra import reading
        mods = behavior.reading_modifiers(behavior.from_affect({"sad": 0.9, "weary": 0.9}))
        allowed, reason = reading.should_read(
            concentration=mods["concentration"], weary=mods["weary"],
            frustration=0.0, has_work=True)
        self.assertFalse(allowed)
        self.assertEqual(reason, reading.IDLE_NOT_CONCENTRATING)

    def test_neutral_state_reads(self):
        from astra import reading
        allowed, reason = reading.should_read(has_work=True)
        self.assertTrue(allowed)


class TestRealStorePropagation(unittest.TestCase):
    """End-to-end over a real store: an affect event changes real planning."""

    def setUp(self):
        from astra.memory import TripleMemoryStore
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_deep_sad_state_holds_off_on_new_activities(self):
        from astra import tasks as T
        from astra.activity import ActivityPlanner

        # A sustained low mood (several sad experiences), the way it accumulates.
        for _ in range(6):
            self.store.record_experience("Everything feels heavy today.",
                                         kind="sad", intensity=1.0, significance=1.0)
        self.assertLessEqual(self.store.current_affect().get("sad"), 1.0)

        planner = ActivityPlanner(
            T.TaskStore(data_dir=self.tmp), store=self.store, library=None,
            ledger=_Ledger(), config={"max_new_per_pass": 2},
            affect_provider=self.store.current_affect)
        report = planner.plan()
        # A strongly sad state takes on no new work this pass, honestly idle.
        self.assertEqual(report["decision"], "idle")
        self.assertIn("low energy", report["reason"])
        self.assertTrue(report["bias"]["prefer_light"])

    def test_a_glad_state_takes_on_work(self):
        from astra import tasks as T
        from astra.activity import ActivityPlanner

        for _ in range(6):
            self.store.record_experience("Something wonderful happened.",
                                         kind="happy", intensity=1.0, significance=1.0)
        planner = ActivityPlanner(
            T.TaskStore(data_dir=self.tmp), store=self.store, library=None,
            ledger=_Ledger(), config={"max_new_per_pass": 2},
            affect_provider=self.store.current_affect)
        report = planner.plan()
        self.assertEqual(report["decision"], "plan")
        self.assertFalse(report["bias"]["prefer_light"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
