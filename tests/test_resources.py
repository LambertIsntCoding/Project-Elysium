"""Tests for resource history and per-task cost estimation.

These are the "can this actually be done?" decisions: a task is deferred only
when a resource it *genuinely uses* would be pushed past saturation, never
merely because a headline number is high. GPU and CPU/memory are separate axes,
and an unmeasurable resource never blocks anything. A defer is a skip, never a
failure. Everything is driven through injected readings, so no host is touched.
"""

import shutil
import tempfile
import unittest

from astra import resources as R
from astra import tasks as T
from astra.machine import Sample
from astra.scheduler import Scheduler


class _State:
    def is_sleeping(self):
        return True


def _sample(*, cpu=None, mem=None, gpu=None, vram=None, vram_total=8e9):
    used = None if vram is None else vram * vram_total
    return Sample(cpu_load=cpu, memory_pressure=mem, gpu_util=gpu,
                  vram_used=used, vram_total=vram_total)


class TestAffordable(unittest.TestCase):
    def test_cpu_only_task_is_never_blocked_by_a_busy_gpu(self):
        # The motivating correction: a busy GPU does not make a CPU-only task
        # impossible. The GPU axis is not consulted for work that does not use it.
        ok, reason = R.affordable(R.ResourceCost(cpu=0.05),
                                  _sample(cpu=0.4, mem=0.4, gpu=0.99, vram=0.98))
        self.assertTrue(ok)
        self.assertEqual(reason, R.AFFORD_OK)

    def test_gpu_task_is_blocked_when_the_gpu_is_saturated(self):
        ok, reason = R.affordable(R.ResourceCost(gpu=0.05),
                                  _sample(cpu=0.1, mem=0.1, gpu=0.99))
        self.assertFalse(ok)
        self.assertEqual(reason, R.LIMITED_GPU)

    def test_vram_pressure_blocks_only_gpu_work(self):
        # VRAM near capacity blocks a VRAM-using task...
        ok, reason = R.affordable(R.ResourceCost(vram=0.05),
                                  _sample(gpu=0.1, vram=0.96))
        self.assertFalse(ok)
        self.assertEqual(reason, R.LIMITED_VRAM)
        # ...but the same machine does not block a CPU-only task.
        ok2, _ = R.affordable(R.ResourceCost(cpu=0.05),
                              _sample(gpu=0.1, vram=0.96))
        self.assertTrue(ok2)

    def test_memory_holding_is_not_the_same_as_no_headroom(self):
        # 80% memory used is *holding* memory, not saturation: a small task fits.
        ok, _ = R.affordable(R.ResourceCost(mem=0.05),
                             _sample(cpu=0.1, mem=0.80))
        self.assertTrue(ok)
        # At 94% the same task would push past the 0.95 bound, so it defers.
        ok2, reason = R.affordable(R.ResourceCost(mem=0.05),
                                   _sample(cpu=0.1, mem=0.94))
        self.assertFalse(ok2)
        self.assertEqual(reason, R.LIMITED_MEMORY)

    def test_cpu_saturation_defers_cpu_work(self):
        ok, reason = R.affordable(R.ResourceCost(cpu=0.05),
                                  _sample(cpu=0.93, mem=0.1))
        self.assertFalse(ok)
        self.assertEqual(reason, R.LIMITED_CPU)

    def test_unmeasurable_resource_never_blocks(self):
        ok, reason = R.affordable(R.ResourceCost(cpu=0.5, mem=0.5, gpu=0.5,
                                                 vram=0.5),
                                  _sample(cpu=None, mem=None, gpu=None, vram=None))
        self.assertTrue(ok)
        self.assertEqual(reason, R.AFFORD_OK)

    def test_a_mapping_sample_is_accepted(self):
        ok, _ = R.affordable(R.ResourceCost(gpu=0.5),
                             {"gpu_util": 0.99, "vram_used": 0, "vram_total": 0})
        self.assertFalse(ok)


class TestResourceHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.h = R.ResourceHistory(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_active_and_idle_baselines_are_separated(self):
        for _ in range(3):
            self.h.observe(_sample(cpu=0.05, mem=0.4, gpu=0.02), active=False)
            self.h.observe(_sample(cpu=0.7, mem=0.6, gpu=0.6), active=True)
        idle = self.h.idle_baseline()
        active = self.h.active_baseline()
        self.assertAlmostEqual(idle["cpu"], 0.05, places=3)
        self.assertAlmostEqual(active["cpu"], 0.7, places=3)
        self.assertEqual(idle["samples"], 3)
        self.assertEqual(active["samples"], 3)

    def test_history_is_bounded(self):
        for _ in range(R.MAX_OBSERVATIONS + 50):
            self.h.observe(_sample(cpu=0.1), active=False)
        self.assertEqual(len(self.h._observations), R.MAX_OBSERVATIONS)

    def test_cost_is_the_prior_until_enough_observations(self):
        self.assertEqual(self.h.cost(T.KIND_READING).as_dict(),
                         R.prior_cost(T.KIND_READING).as_dict())

    def test_cost_is_learned_from_measured_deltas(self):
        before = _sample(cpu=0.10, mem=0.40, gpu=0.10, vram=0.10)
        after = _sample(cpu=0.30, mem=0.42, gpu=0.30, vram=0.20)
        for _ in range(R.MIN_COST_SAMPLES):
            self.h.record_task(T.KIND_READING, before, after)
        cost = self.h.cost(T.KIND_READING)
        # The observed deltas (0.20 cpu, 0.02 mem, 0.20 gpu, 0.10 vram) pull the
        # estimate away from the prior and toward what actually happened.
        self.assertGreater(cost.cpu, R.prior_cost(T.KIND_READING).cpu)
        self.assertGreater(cost.gpu, R.prior_cost(T.KIND_READING).gpu)

    def test_missing_delta_teaches_nothing(self):
        before = _sample(cpu=None, mem=0.4, gpu=0.1)
        after = _sample(cpu=None, mem=0.5, gpu=0.2)
        for _ in range(R.MIN_COST_SAMPLES):
            self.h.record_task(T.KIND_READING, before, after)
        # cpu was unmeasurable, so it keeps the prior rather than inventing 0.
        self.assertAlmostEqual(self.h.cost(T.KIND_READING).cpu,
                               R.prior_cost(T.KIND_READING).cpu, places=5)

    def test_reading_claims_gpu_but_light_kinds_do_not(self):
        self.assertIn("gpu", R.KIND_RESOURCES[T.KIND_READING])
        self.assertNotIn("gpu", R.KIND_RESOURCES[T.KIND_MEMORY_MAINTENANCE])
        self.assertFalse(R.prior_cost(T.KIND_MEMORY_MAINTENANCE).uses("gpu"))

    def test_history_persists_across_instances(self):
        self.h.observe(_sample(cpu=0.1), active=False)
        for _ in range(R.MIN_COST_SAMPLES):
            self.h.record_task(T.KIND_TEST_RUN,
                               _sample(cpu=0.1, mem=0.4),
                               _sample(cpu=0.4, mem=0.5))
        reopened = R.ResourceHistory(data_dir=self.tmp)
        self.assertEqual(reopened.idle_baseline()["samples"], 1)
        self.assertGreater(reopened.cost(T.KIND_TEST_RUN).cpu,
                           R.prior_cost(T.KIND_TEST_RUN).cpu)


class TestSchedulerResourceGate(unittest.TestCase):
    """The gate admits what fits and defers what does not - never fails."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tasks = T.TaskStore(data_dir=self.tmp)
        self.ledger = T.TaskLedger(data_dir=self.tmp)
        self.history = R.ResourceHistory(data_dir=self.tmp)
        self.executed = []
        self.sample = _sample(cpu=0.1, mem=0.1)
        self.sched = Scheduler(
            task_store=self.tasks, ledger=self.ledger, runtime_state=_State(),
            process_names_fn=lambda: [], max_tasks_per_cycle=5,
            cycle_budget_seconds=10.0, resource_history=self.history,
            machine_sample_provider=lambda: self.sample,
        )
        for kind in T.TASK_KINDS:
            self.sched.register(kind, self._handler)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _handler(self, task):
        self.executed.append(task.id)
        return {"status": T.COMPLETED, "detail": "done"}

    def test_cpu_only_task_runs_on_a_busy_gpu_machine(self):
        self.sample = _sample(cpu=0.2, mem=0.2, gpu=0.99, vram=0.95)
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_MEMORY_MAINTENANCE,
                                            dedupe_key="m", priority=90))
        report = self.sched.run_cycle()
        self.assertEqual(report["tasks"], 1)
        self.assertEqual(self.tasks.all()[0].status, T.COMPLETED)

    def test_gpu_task_defers_when_the_gpu_is_saturated(self):
        self.sample = _sample(cpu=0.2, mem=0.2, gpu=0.99)
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_READING,
                                            dedupe_key="r", priority=90))
        report = self.sched.run_cycle()
        self.assertEqual(report["tasks"], 0)
        self.assertTrue(any(s["reason"] == "gpu_headroom"
                            for s in report["skipped"]))

    def test_a_defer_is_a_skip_never_a_failure(self):
        self.sample = _sample(cpu=0.99, mem=0.99, gpu=0.99, vram=0.99)
        task = T.BackgroundTask(kind=T.KIND_READING, dedupe_key="r", priority=90)
        self.tasks.enqueue(task)
        self.sched.run_cycle()
        stored = self.tasks.get(task.id)
        # Deferred, not failed: it stays pending with its attempts untouched, so
        # the bounded retry policy is never consumed by a resource hold.
        self.assertEqual(stored.status, T.PENDING)
        self.assertEqual(stored.attempts, 0)
        self.assertEqual(self.ledger.entries(), [])

    def test_no_task_ever_fails_merely_for_resources(self):
        # A fully saturated machine: every kind is enqueued and none may fail.
        self.sample = _sample(cpu=0.99, mem=0.99, gpu=0.99, vram=0.99)
        for i, kind in enumerate(T.TASK_KINDS):
            self.tasks.enqueue(T.BackgroundTask(kind=kind, dedupe_key=f"k{i}"))
        self.sched.run_cycle()
        statuses = {t.status for t in self.tasks.all()}
        self.assertNotIn(T.FAILED, statuses)
        self.assertNotIn(T.BLOCKED, statuses)

    def test_cost_is_learned_from_a_run(self):
        # The scheduler brackets each run with a machine reading, so the kind
        # accrues a real observation (the estimate itself is exercised directly
        # in TestResourceHistory; here we prove the wiring records it).
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_TEST_RUN,
                                            dedupe_key="t", priority=90))
        self.sched.run_cycle()
        self.assertEqual(self.tasks.all()[0].status, T.COMPLETED)
        costs = self.history.kind_costs()
        self.assertIn(T.KIND_TEST_RUN, costs)
        self.assertGreaterEqual(int(costs[T.KIND_TEST_RUN].get("samples", 0)), 1)

    def test_no_history_means_the_prior_gate_only(self):
        sched = Scheduler(
            task_store=self.tasks, ledger=self.ledger, runtime_state=_State(),
            process_names_fn=lambda: [], machine_sample_provider=lambda: _sample(cpu=0.99),
            resource_history=None,
        )
        for kind in T.TASK_KINDS:
            sched.register(kind, self._handler)
        self.tasks.enqueue(T.BackgroundTask(kind=T.KIND_MEMORY_MAINTENANCE,
                                            dedupe_key="m"))
        # With no history the prior is still consulted through the sample, so the
        # saturated CPU defers even the light kind.
        report = sched.run_cycle()
        self.assertEqual(report["tasks"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
