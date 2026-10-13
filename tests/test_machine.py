"""Tests for real machine awareness: resources, uptime, and suspend detection.

Everything here is driven through *injected* providers, so a scenario (cold
boot, loaded machine, post-suspend resume) is deterministic and never touches
the host. ``build_default`` is exercised separately against whatever the real
platform reports, asserting only that a value is a number or ``None`` - never a
fabricated number.
"""

import unittest

from astra import machine


class _Clock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _state(**kwargs) -> machine.MachineState:
    """A MachineState whose providers are drawn from kwargs."""
    values = dict(kwargs)

    def provider(name):
        def get():
            value = values.get(name)
            return value() if callable(value) else value
        return get

    return machine.MachineState(
        cpu_provider=provider("cpu"),
        memory_provider=provider("memory"),
        uptime_provider=provider("uptime"),
        idle_provider=provider("idle"),
        monotonic=kwargs.get("monotonic") or _Clock(),
        warmup_seconds=kwargs.get("warmup_seconds", 180.0),
        sample_window=kwargs.get("sample_window", 6.0),
    )


class TestResourceDecisions(unittest.TestCase):
    def test_missing_measurements_are_not_a_problem(self):
        st = _state(cpu=None, memory=None, uptime=None)
        st.sample()
        self.assertTrue(st.resources_available())
        # Unknown uptime must not trap the machine in a cold-start mode forever.
        self.assertTrue(st.warm())

    def test_high_cpu_or_memory_is_unavailable(self):
        self.assertFalse(_state(cpu=0.9, memory=0.1).resources_available(
            machine.Sample(cpu_load=0.9, memory_pressure=0.1)))
        self.assertFalse(_state().resources_available(
            machine.Sample(cpu_load=0.1, memory_pressure=0.95)))

    def test_ready_requires_warm_and_sustained_calm(self):
        st = _state(cpu=0.1, memory=0.1)
        # Cold: not warm yet (explicit sample avoids the suspend detector, which
        # is exercised separately below).
        cold = machine.Sample(at=1000.0, cpu_load=0.1, memory_pressure=0.1,
                              uptime_seconds=10.0)
        self.assertFalse(st.ready_for_work(cold))
        # Warm, but calm has not yet lasted the sample window.
        warm = machine.Sample(at=1001.0, cpu_load=0.1, memory_pressure=0.1,
                              uptime_seconds=500.0)
        self.assertFalse(st.ready_for_work(warm))
        calm = machine.Sample(at=1008.0, cpu_load=0.1, memory_pressure=0.1,
                              uptime_seconds=500.0)
        self.assertTrue(st.ready_for_work(calm))
        self.assertFalse(st.efficiency_mode(calm))

    def test_a_dip_resets_the_sustained_calm_window(self):
        st = _state(cpu=0.1, memory=0.1)
        calm = machine.Sample(at=1000.0, cpu_load=0.1, memory_pressure=0.1,
                              uptime_seconds=500.0)
        st.ready_for_work(calm)
        later = machine.Sample(at=1007.0, cpu_load=0.1, memory_pressure=0.1,
                               uptime_seconds=500.0)
        self.assertTrue(st.ready_for_work(later))
        # A momentary spike resets calm.
        spike = machine.Sample(at=1008.0, cpu_load=0.95, memory_pressure=0.1,
                               uptime_seconds=500.0)
        self.assertFalse(st.ready_for_work(spike))
        recovered = machine.Sample(at=1012.0, cpu_load=0.1, memory_pressure=0.1,
                                   uptime_seconds=500.0)
        # The window restarted at 1012, so it is not ready yet.
        self.assertFalse(st.ready_for_work(recovered))

    def test_efficiency_mode_is_readiness_inverted(self):
        st = _state(cpu=0.9, memory=0.1, uptime=500.0)
        st.sample()
        self.assertTrue(st.efficiency_mode())
        self.assertFalse(st.ready_for_work())


class TestSuspendDetection(unittest.TestCase):
    def test_uptime_mismatch_detects_suspend(self):
        clock = _Clock()
        st = _state(cpu=0.1, memory=0.1, uptime=100.0, monotonic=clock)
        st.sample()
        # Our monotonic clock advanced 2s, but the machine's uptime advanced
        # 4000s: the machine was suspended for ~3998s.
        clock.advance(2.0)
        st.uptime_provider = lambda: 4100.0
        sample = st.sample()
        self.assertIsNotNone(sample.suspended_seconds)
        self.assertGreater(sample.suspended_seconds, machine.SUSPEND_BACKLOG_SECONDS)

    def test_wall_and_uptime_in_step_is_not_a_suspend(self):
        clock = _Clock()
        st = _state(cpu=0.1, memory=0.1, uptime=100.0, monotonic=clock)
        st.sample()
        clock.advance(10.0)
        st.uptime_provider = lambda: 110.0
        self.assertIsNone(st.sample().suspended_seconds)

    def test_unknown_uptime_never_invents_a_suspend(self):
        st = _state(cpu=0.1, memory=0.1, uptime=None)
        st.sample()
        st.sample()
        self.assertIsNone(st.sample().suspended_seconds)

    def test_cpu_reading_does_not_advance_suspend_bookkeeping(self):
        clock = _Clock()
        st = _state(cpu=0.1, memory=0.1, uptime=100.0, monotonic=clock)
        st.sample()
        # A bare cpu reading (used by the scheduler from another thread) must
        # not perturb the suspend detector.
        st.cpu_reading()
        clock.advance(2.0)
        st.uptime_provider = lambda: 4100.0
        # The detector still sees the gap: the bare read did not reset _last.
        self.assertIsNotNone(st.sample().suspended_seconds)


class TestDefaultBuilder(unittest.TestCase):
    def test_default_builder_returns_numbers_or_none_only(self):
        st = machine.build_default()
        st.sample()
        snap = st.snapshot()
        for key in ("cpu_load", "memory_pressure", "uptime_seconds",
                    "idle_seconds", "suspended_seconds"):
            self.assertTrue(snap[key] is None or isinstance(snap[key], float),
                            f"{key} must be a measured float or None")
        self.assertIsInstance(snap["ready_for_work"], bool)
        self.assertIsInstance(snap["efficiency_mode"], bool)


class TestCpuReadingIsFresh(unittest.TestCase):
    """A status/diagnostic read must not report a stale or missing CPU value.

    Windows' cumulative-counter provider returns ``None`` on its first read
    (there is no previous sample to difference against). The old ``snapshot``
    reused the stored sample, so that one ``None`` was reported forever and the
    status line showed ``cpu=?``. These lock in the fix: a two-read provider
    yields a real fraction on the very first sample, and ``snapshot`` re-reads
    rather than echoing the stored value.
    """

    def _two_read(self):
        calls = {"n": 0}

        def cpu():
            calls["n"] += 1
            return None if calls["n"] == 1 else 0.5

        return cpu, calls

    def test_first_sample_tolerates_a_two_read_provider(self):
        cpu, calls = self._two_read()
        st = machine.MachineState(
            cpu_provider=cpu, memory_provider=lambda: 0.2,
            uptime_provider=lambda: 500.0, idle_provider=lambda: 10.0)
        self.assertEqual(st.sample().cpu_load, 0.5)
        self.assertGreaterEqual(calls["n"], 2)

    def test_snapshot_reflects_a_fresh_reading_not_a_stale_none(self):
        cpu, _ = self._two_read()
        st = machine.MachineState(
            cpu_provider=cpu, memory_provider=lambda: 0.2,
            uptime_provider=lambda: 500.0, idle_provider=lambda: 10.0)
        # A value that changes between reads must be visible immediately.
        st.cpu_provider = lambda: 0.25
        self.assertEqual(st.snapshot()["cpu_load"], 0.25)
        st.cpu_provider = lambda: 0.75
        self.assertEqual(st.snapshot()["cpu_load"], 0.75)

    def test_snapshot_does_not_advance_suspend_bookkeeping(self):
        # A diagnostic read must not mask a real suspend from the power path.
        clock = _Clock()
        st = _state(cpu=0.1, memory=0.1, uptime=100.0, monotonic=clock)
        st.sample()
        st.snapshot()
        clock.advance(2.0)
        st.uptime_provider = lambda: 4100.0
        self.assertIsNotNone(st.sample().suspended_seconds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
