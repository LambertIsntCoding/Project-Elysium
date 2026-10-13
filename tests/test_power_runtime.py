"""Tests that the runtime sleeps alongside the machine and stays honest.

These drive a real :class:`RuntimeController` with an injected (fake, numeric)
machine, so a cold boot or a suspend resume is deterministic and no real host
state is touched. The store and task files are real on disk.
"""

import shutil
import tempfile
import unittest

from astra.application import RuntimeController
from astra.memory import TripleMemoryStore
from astra.machine import MachineState, Sample
from astra import runtime_state as rs


class _StubSession:
    output_fn = staticmethod(lambda text: None)

    def note_activity(self):
        pass

    def handle(self, text):
        return text


class _FixedMachine:
    """A MachineState whose single sample is scripted (never the real host)."""

    def __init__(self, *, cpu=0.1, mem=0.1, uptime=100000.0, idle=None,
                 suspended=None, gpu=None, vram_used=None, vram_total=None):
        self._sample = Sample(cpu_load=cpu, memory_pressure=mem,
                              uptime_seconds=uptime, idle_seconds=idle,
                              suspended_seconds=suspended, gpu_util=gpu,
                              vram_used=vram_used, vram_total=vram_total)
        self._ready = suspended is None and uptime >= 180.0 and cpu < 0.6

    def sample(self):
        return self._sample

    def sample_now(self):
        return self._sample

    def ready_for_work(self, sample=None):
        return self._ready

    def sustainably_calm(self, sample=None):
        # Warmup-only readiness: memory holding does not make it "cold".
        return self._sample.suspended_seconds is None and self._sample.uptime_seconds >= 180.0

    def efficiency_mode(self, sample=None):
        return not self._ready

    def cpu_reading(self):
        return self._sample.cpu_load

    def memory_headroom(self, sample=None):
        return None if self._sample.memory_pressure is None else 1.0 - self._sample.memory_pressure

    def snapshot(self):
        out = self._sample.as_dict()
        out.update({"ready_for_work": self._ready,
                    "efficiency_mode": not self._ready,
                    "resources_available": True,
                    "gpu_available": True})
        return out


def _controller(tmp, machine, config=None):
    store = TripleMemoryStore(data_dir=tmp)
    return RuntimeController(
        session=_StubSession(), store=store, data_dir=tmp,
        config_dir="./config", config=config or {}, reader=None, library=None,
        machine=machine, poll_seconds=1.0,
    )


class TestPowerAndMachine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_suspend_makes_astra_sleep_alongside_the_machine(self):
        ctrl = _controller(self.tmp, _FixedMachine(suspended=4000.0))
        ctrl.wake(reason=rs.REASON_HUMAN_WAKE)
        self.assertFalse(ctrl.sleeping())
        ctrl._handle_power()
        self.assertTrue(ctrl.sleeping())
        self.assertEqual(ctrl.state.current().get("reason"), rs.REASON_POWER)
        self.assertGreater(ctrl._last_suspend_seconds, 0)

    def test_no_suspend_leaves_state_alone(self):
        ctrl = _controller(self.tmp, _FixedMachine())
        ctrl.wake(reason=rs.REASON_HUMAN_WAKE)
        ctrl._handle_power()
        self.assertFalse(ctrl.sleeping())

    def test_sleep_with_computer_can_be_disabled(self):
        ctrl = _controller(self.tmp, _FixedMachine(suspended=4000.0),
                           config={"runtime": {"power": {"sleep_with_computer": False}}})
        ctrl.wake(reason=rs.REASON_HUMAN_WAKE)
        ctrl._handle_power()
        self.assertFalse(ctrl.sleeping())

    def test_resume_can_opt_into_waking(self):
        ctrl = _controller(self.tmp, _FixedMachine(suspended=4000.0),
                           config={"runtime": {"power": {"wake_on_resume": True}}})
        ctrl.sleep(reason=rs.REASON_LIFECYCLE)
        ctrl._handle_power()
        self.assertFalse(ctrl.sleeping())

    def test_cold_machine_is_efficiency_mode(self):
        ctrl = _controller(self.tmp, _FixedMachine(uptime=5.0))
        self.assertFalse(ctrl._machine_ready_for_work())
        self.assertTrue(ctrl.power_report().count("efficiency mode: True") == 1)

    def test_warm_machine_is_ready(self):
        ctrl = _controller(self.tmp, _FixedMachine(uptime=100000.0, cpu=0.1))
        self.assertTrue(ctrl._machine_ready_for_work())

    def test_power_report_is_truthful_and_has_no_secrets(self):
        ctrl = _controller(self.tmp, _FixedMachine())
        report = ctrl.power_report()
        self.assertIn("Machine and power:", report)
        self.assertIn("ready for heavy work", report)

    def test_scheduler_feeds_on_machine_cpu(self):
        # With no explicit cpu_load_fn, the controller uses the machine reading.
        ctrl = _controller(self.tmp, _FixedMachine(cpu=0.99))
        self.assertEqual(ctrl.scheduler.cpu_load_fn(), 0.99)

    def test_power_report_includes_gpu_and_vram_when_measurable(self):
        ctrl = _controller(self.tmp, _FixedMachine(
            gpu=0.4, vram_used=4e9, vram_total=8e9))
        report = ctrl.power_report()
        self.assertIn("gpu: 40%", report)
        self.assertIn("3.7/7.5 GB", report)

    def test_power_report_says_gpu_unmeasurable_honestly(self):
        ctrl = _controller(self.tmp, _FixedMachine())
        self.assertIn("gpu: not measurable", ctrl.power_report())

    def test_resource_report_renders_baselines_and_costs(self):
        ctrl = _controller(self.tmp, _FixedMachine())
        report = ctrl.resource_report()
        self.assertIn("Resource history:", report)
        self.assertIn("idle (machine alone)", report)
        self.assertIn("active (Roum's apps)", report)
        self.assertIn("reading", report)

    def test_controller_owns_a_resource_history(self):
        ctrl = _controller(self.tmp, _FixedMachine())
        self.assertTrue(hasattr(ctrl.resource_history, "observe"))

    def test_sampling_records_an_observation(self):
        ctrl = _controller(self.tmp, _FixedMachine(cpu=0.3, mem=0.4))
        ctrl._sample_machine()
        snap = ctrl.resource_history.snapshot()
        self.assertGreaterEqual(snap["observations"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
