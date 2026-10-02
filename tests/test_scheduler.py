"""Tests for the deterministic background scheduler."""

import shutil
import tempfile
import unittest

from astra import runtime_state as rs
from astra import tasks as T
from astra import reading
from astra.scheduler import Scheduler


class _FakeState:
    """A controllable runtime-state double (is_sleeping only)."""

    def __init__(self, sleeping=True):
        self.sleeping = sleeping

    def is_sleeping(self):
        return self.sleeping


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tasks = T.TaskStore(data_dir=self.tmp)
        self.ledger = T.TaskLedger(data_dir=self.tmp)
        self.state = _FakeState(sleeping=True)
        self.executed = []
        self.sched = Scheduler(
            task_store=self.tasks, ledger=self.ledger, runtime_state=self.state,
            process_names_fn=lambda: [], max_tasks_per_cycle=2,
            cycle_budget_seconds=10.0,
        )
        for kind in T.TASK_KINDS:
            self.sched.register(kind, self._handler)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _handler(self, task):
        self.executed.append(task.id)
        return {"status": T.COMPLETED, "detail": "done"}

    # -- no-op and gating ------------------------------------------------
    def test_zero_tasks_is_a_legitimate_cycle(self):
        report = self.sched.run_cycle()
        self.assertEqual(report["tasks"], 0)
        self.assertEqual(report["reason"], "no_work")

    def test_active_state_runs_nothing(self):
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="a"))
        self.state.sleeping = False
        report = self.sched.run_cycle()
        self.assertEqual(report["reason"], "active")
        self.assertEqual(self.executed, [])
        # The task is untouched - it will run when sleep resumes.
        self.assertEqual(self.tasks.all()[0].status, T.PENDING)

    def test_game_stands_down_intensive_but_light_may_run(self):
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_FILE_INSPECTION, dedupe_key="f", priority=50))
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_MEMORY_MAINTENANCE, dedupe_key="m", priority=90))
        self.sched.process_names_fn = lambda: ["dota2.exe"]
        report = self.sched.run_cycle()
        kinds_run = {self.tasks.get(i).kind for i in self.executed}
        self.assertEqual(kinds_run, {T.KIND_MEMORY_MAINTENANCE})
        self.assertTrue(any(s["reason"] == "game" for s in report["skipped"]))

    def test_busy_task_stands_down_intensive(self):
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_FILE_INSPECTION, dedupe_key="f"))
        self.sched.process_policy = reading.ProcessPolicy(busy=[r"blender"])
        self.sched.process_names_fn = lambda: ["blender"]
        self.sched.run_cycle()
        self.assertEqual(self.executed, [])

    def test_load_ceiling_blocks_intensive(self):
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="r"))
        self.sched.cpu_load_fn = lambda: 0.99
        self.sched.run_cycle()
        self.assertEqual(self.executed, [])

    # -- budgets ---------------------------------------------------------
    def test_bounded_tasks_per_cycle(self):
        for i in range(5):
            self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_READING,
                                                dedupe_key=f"r{i}", priority=10 + i))
        report = self.sched.run_cycle()
        self.assertEqual(report["tasks"], 2)  # max_tasks_per_cycle
        self.assertEqual(report["reason"], "budget")

    def test_stop_between_tasks(self):
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="a", priority=50))

        def stopping_handler(task):
            self.executed.append(task.id)
            self.sched.stop()
            return {"status": T.COMPLETED}

        self.sched.register(T.KIND_READING, stopping_handler)
        report = self.sched.run_cycle()
        self.assertTrue(report["stopped"])
        self.assertEqual(report["reason"], "stopped")

    def test_cycle_budget_stops_after_spending(self):
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="a", priority=50))
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="b", priority=40))
        self.sched.max_tasks_per_cycle = 5
        self.sched.cycle_budget_seconds = 0.0  # already spent
        report = self.sched.run_cycle()
        self.assertEqual(report["tasks"], 0)
        self.assertEqual(report["reason"], "budget")

    # -- outcomes --------------------------------------------------------
    def test_handler_failure_is_recorded_as_failed_not_success(self):
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="r"))
        self.sched.register(T.KIND_READING,
                            lambda t: {"status": T.FAILED, "detail": "boom"})
        self.sched.run_cycle()
        task = self.tasks.all()[0]
        self.assertEqual(task.status, T.FAILED)
        self.assertEqual(task.last_error, "boom")
        # The ledger carries the failure, so it is never counted as success.
        self.assertEqual(self.ledger.entries()[-1]["status"], T.FAILED)

    def test_repeated_failure_becomes_blocked_not_infinite_loop(self):
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="r", max_attempts=2))
        self.sched.register(T.KIND_READING, lambda t: {"status": T.FAILED})
        self.sched.run_cycle()
        self.sched.run_cycle()
        task = self.tasks.all()[0]
        self.assertEqual(task.status, T.BLOCKED)
        # No longer eligible, so it cannot loop forever.
        self.assertNotIn(task.id, [t.id for t in self.tasks.eligible()])

    def test_missing_handler_blocks_rather_than_silently_completes(self):
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_WORK_REVIEW, dedupe_key="w"))
        del self.sched._handlers[T.KIND_WORK_REVIEW]
        self.sched.run_cycle()
        self.assertEqual(self.tasks.all()[0].status, T.BLOCKED)

    def test_running_state_is_persisted_before_side_effect(self):
        seen = {}

        def handler(task):
            seen["status_during"] = self.tasks.get(task.id).status
            return {"status": T.COMPLETED}

        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_READING, dedupe_key="r"))
        self.sched.register(T.KIND_READING, handler)
        self.sched.run_cycle()
        # During the side effect the task must already read as RUNNING, so a
        # crash is visible and recoverable rather than a silent success.
        self.assertEqual(seen["status_during"], T.RUNNING)

    def test_scheduler_is_deterministic(self):
        def run_once(keys):
            tmp = tempfile.mkdtemp()
            try:
                store = T.TaskStore(data_dir=tmp)
                ledger = T.TaskLedger(data_dir=tmp)
                ran = []
                sched = Scheduler(task_store=store, ledger=ledger,
                                  runtime_state=_FakeState(True),
                                  process_names_fn=lambda: [],
                                  max_tasks_per_cycle=10)
                sched.register(T.KIND_READING,
                               lambda t: (ran.append(t.dedupe_key),
                                          {"status": T.COMPLETED})[1])
                for key, prio in keys:
                    store.enqueue(T.BackgroundTask(kind=T.KIND_READING,
                                                   dedupe_key=key, priority=prio))
                sched.run_cycle()
                return ran
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        keys = [("a", 10), ("b", 30), ("c", 20)]
        self.assertEqual(run_once(keys), run_once(keys))
        # Same priority -> oldest first; highest priority first otherwise.
        self.assertEqual(run_once(keys), ["b", "c", "a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
