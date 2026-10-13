"""Real, numeric machine awareness: resources, uptime, idle and suspend.

The brief asks for three related things that all need *actual numbers* rather
than an inference from which applications happen to be running:

* **Start conservatively.** When the computer starts, Astra runs in an energy-
  and resource-efficient mode until the machine has been up long enough *and*
  the measured CPU/memory use is low enough to safely do more.
* **Do not fight the machine.** When a heavy workload is actually present, the
  existing policy already stands intensive work down; this module supplies the
  numeric load that decision reads.
* **Sleep with the computer.** If the machine suspends, Astra should recognise
  it (on resume) and sleep alongside it rather than treating the gap as active
  time, and should be able to recognise a machine that is about to sleep.

Design constraints, kept consistent with the rest of the project:

* **Numeric and measured, never guessed.** Every value is either a real
  measurement or ``None``; the module never invents a number. ``None`` always
  means "not observable here", and callers treat that as "no reason to change".
* **Stdlib first, ``psutil`` optional like everywhere else.** Windows paths use
  ``ctypes`` (``GetSystemTimes``, ``GlobalMemoryStatusEx``, ``GetTickCount64``,
  ``GetLastInputInfo``); POSIX uses ``os.getloadavg`` and ``/proc``. ``psutil``
  is *not* required - the project treats it as an optional extra elsewhere, and
  the measurements here are available without it.
* **Pure decisions over sampled facts.** :class:`MachineState` samples, and the
  *decisions* (``ready_for_work``, ``efficiency_mode``) are pure functions of a
  sample, so they are testable without a real machine.
* **Injectable for tests.** The raw providers are constructor arguments, so a
  test can drive any scenario (cold boot, loaded, post-suspend) deterministically
  without touching the host.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

# Default ceilings. Measured as fractions (0..1). Conservative on purpose: a
# cold or busy machine stays in efficiency mode until it is genuinely calm.
DEFAULT_CPU_CEILING = 0.60          # CPU above this -> not ready for heavy work
DEFAULT_MEMORY_CEILING = 0.85       # memory above this -> not ready
# GPU is a *separate* resource from CPU/memory: a machine can be CPU-idle while a
# game or render claims the GPU. Only GPU-claiming work is gated on it, and an
# unmeasurable GPU (``None``) never blocks anything - it is not proof of absence.
DEFAULT_GPU_CEILING = 0.85          # GPU utilisation above this -> GPU work stands down
DEFAULT_VRAM_CEILING = 0.90         # VRAM pressure above this -> GPU work stands down
DEFAULT_WARMUP_SECONDS = 180.0      # how long the machine must have been up
DEFAULT_SAMPLE_WINDOW = 6.0         # how long "sustained" calm must last
# A gap between samples at least this much larger than the wall-clock elapsed
# means the machine was suspended: the wall clock advanced while it did not.
SUSPEND_BACKLOG_SECONDS = 90.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Raw providers (stdlib-first; each returns a real measurement or ``None``)
# ---------------------------------------------------------------------------
def _windows_cpu_load(_state: Dict[str, Any]) -> Optional[float]:
    try:
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD),
                        ("dwHighDateTime", wintypes.DWORD)]

        idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        idle_t = (idle.dwHighDateTime << 32) | idle.dwLowDateTime
        kernel_t = (kernel.dwHighDateTime << 32) | kernel.dwLowDateTime
        user_t = (user.dwHighDateTime << 32) | user.dwLowDateTime
        prev = _state.get("win_cpu")
        _state["win_cpu"] = (idle_t, kernel_t, user_t)
        if prev is None:
            return None  # need two samples
        d_idle = idle_t - prev[0]
        d_total = (kernel_t - prev[1]) + (user_t - prev[2])
        if d_total <= 0:
            return None
        return max(0.0, min(1.0, 1.0 - (d_idle / d_total)))
    except Exception:
        return None


def _windows_memory_pressure() -> Optional[float]:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        return max(0.0, min(1.0, stat.dwMemoryLoad / 100.0))
    except Exception:
        return None


def _windows_uptime_seconds() -> Optional[float]:
    try:
        import ctypes

        return float(ctypes.windll.kernel32.GetTickCount64()) / 1000.0
    except Exception:
        return None


def _windows_idle_seconds() -> Optional[float]:
    try:
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        tick = ctypes.windll.kernel32.GetTickCount64()
        return max(0.0, (tick - info.dwTime) / 1000.0)
    except Exception:
        return None


def _posix_cpu_load() -> Optional[float]:
    try:
        import os

        if hasattr(os, "getloadavg"):
            load1, _, _ = os.getloadavg()
            cores = os.cpu_count() or 1
            return max(0.0, min(1.0, load1 / cores))
    except (OSError, AttributeError, ValueError):
        return None
    return None


def _posix_memory_pressure() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            data = {}
            for line in fh:
                key, _, rest = line.partition(":")
                data[key.strip()] = rest.strip().split()[0] if rest.strip() else "0"
        total = float(data.get("MemTotal", 0))
        avail = float(data.get("MemAvailable", 0))
        if total <= 0:
            return None
        return max(0.0, min(1.0, 1.0 - (avail / total)))
    except Exception:
        return None


def _posix_uptime_seconds() -> Optional[float]:
    try:
        with open("/proc/uptime", "r", encoding="utf-8") as fh:
            return float(fh.read().split()[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GPU + VRAM (optional; each returns a real reading or ``None``)
# ---------------------------------------------------------------------------
# The GPU is a genuinely separate resource: a machine can be CPU-idle while a
# game or a render owns the GPU, and a model can fit in VRAM while system memory
# is tight. So GPU utilisation and VRAM are measured independently and never
# folded into the CPU/memory numbers. Everything here is best-effort and
# stdlib-first (``subprocess`` to ``nvidia-smi`` only if present, ``/sys`` for
# AMD); an unmeasurable GPU is ``None`` and blocks nothing.
def _parse_nvidia_smi_csv(output: str) -> Optional[Dict[str, Optional[float]]]:
    """Parse one ``nvidia-smi --query-gpu=...,memory.used,memory.total`` row."""
    rows = [r for r in str(output or "").strip().splitlines() if r.strip()]
    if not rows:
        return None
    parts = [p.strip() for p in rows[0].split(",")]
    if len(parts) < 3:
        return None

    def num(text: str) -> Optional[float]:
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    util = num(parts[0])
    used = num(parts[1])
    total = num(parts[2])
    return {
        "util": None if util is None else max(0.0, min(1.0, util / 100.0)),
        "vram_used": used if (used is not None and used > 0) else None,
        "vram_total": total if (total is not None and total > 0) else None,
    }


def _nvidia_smi_reading() -> Optional[Dict[str, Optional[float]]]:
    """GPU/VRAM via ``nvidia-smi`` when it is installed, else ``None``."""
    try:
        import shutil
        import subprocess

        exe = shutil.which("nvidia-smi")
        if not exe:
            return None
        proc = subprocess.run(
            [exe, "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
        if proc.returncode != 0:
            return None
        return _parse_nvidia_smi_csv(proc.stdout)
    except Exception:
        return None


def _amdgpu_reading() -> Optional[Dict[str, Optional[float]]]:
    """GPU/VRAM for AMD cards via ``/sys/class/drm`` when present, else ``None``."""
    try:
        import glob
        import os

        def read_int(path: str) -> Optional[float]:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return float(fh.read().strip())
            except Exception:
                return None

        for busy_path in sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")):
            dev = os.path.dirname(busy_path)
            util = read_int(busy_path)
            used = read_int(os.path.join(dev, "mem_info_vram_used"))
            total = read_int(os.path.join(dev, "mem_info_vram_total"))
            if util is None and not (used and total):
                continue
            return {
                "util": None if util is None else max(0.0, min(1.0, util / 100.0)),
                "vram_used": used if (used and used > 0) else None,
                "vram_total": total if (total and total > 0) else None,
            }
    except Exception:
        return None
    return None


def default_gpu_reading() -> Optional[Dict[str, Optional[float]]]:
    """A GPU/VRAM reading from whichever real source is available.

    Order is deliberate: ``nvidia-smi`` (the most common discrete GPU, and the
    one Ollama uses) first, then AMD's ``/sys`` interface. Anything else returns
    ``None`` - the absence of a measurement is never treated as an idle GPU.
    """
    return _nvidia_smi_reading() or _amdgpu_reading()


def default_is_windows() -> bool:
    import sys
    return sys.platform.startswith("win")


# ---------------------------------------------------------------------------
# The sample and the pure decisions over it
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    """One measurement of the machine, at one instant."""

    at: float = 0.0                       # monotonic seconds
    cpu_load: Optional[float] = None      # 0..1 or None
    memory_pressure: Optional[float] = None   # 0..1 or None
    uptime_seconds: Optional[float] = None    # since boot, or None
    idle_seconds: Optional[float] = None      # since last user input, or None
    suspended_seconds: Optional[float] = None  # detected suspend gap, or None
    gpu_util: Optional[float] = None      # 0..1 GPU utilisation, or None
    vram_used: Optional[float] = None     # bytes in use, or None
    vram_total: Optional[float] = None    # bytes total, or None

    def vram_pressure(self) -> Optional[float]:
        """VRAM in use as a fraction (0..1), or ``None`` when not measurable."""
        used, total = self.vram_used, self.vram_total
        if used is None or total is None or total <= 0:
            return None
        return max(0.0, min(1.0, used / total))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cpu_load": self.cpu_load,
            "memory_pressure": self.memory_pressure,
            "uptime_seconds": self.uptime_seconds,
            "idle_seconds": self.idle_seconds,
            "suspended_seconds": self.suspended_seconds,
            "gpu_util": self.gpu_util,
            "vram_used": self.vram_used,
            "vram_total": self.vram_total,
            "vram_pressure": self.vram_pressure(),
        }


@dataclass
class MachineState:
    """Samples the machine and answers bounded policy questions about it.

    All providers are injectable so the behaviour is deterministic in tests.
    Every provider returns a real number or ``None``; ``None`` is never treated
    as a problem, only as "unknown", so a platform without a given measurement
    simply does not change any decision.
    """

    cpu_provider: Optional[Callable[[], Optional[float]]] = None
    memory_provider: Optional[Callable[[], Optional[float]]] = None
    uptime_provider: Optional[Callable[[], Optional[float]]] = None
    idle_provider: Optional[Callable[[], Optional[float]]] = None
    gpu_provider: Optional[Callable[[], Optional[Dict[str, Optional[float]]]]] = None
    monotonic: Callable[[], float] = time.monotonic
    cpu_ceiling: float = DEFAULT_CPU_CEILING
    memory_ceiling: float = DEFAULT_MEMORY_CEILING
    gpu_ceiling: float = DEFAULT_GPU_CEILING
    vram_ceiling: float = DEFAULT_VRAM_CEILING
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS
    sample_window: float = DEFAULT_SAMPLE_WINDOW
    # The CPU sampling on Windows needs two reads to produce a fraction; that
    # small cache lives here rather than in module globals so state is explicit.
    _scratch: Dict[str, Any] = field(default_factory=dict)
    _last: Optional[Sample] = None
    _calm_since: Optional[float] = None
    _settled_since: Optional[float] = None

    # -- sampling --------------------------------------------------------
    def _gpu_measurement(self) -> Dict[str, Optional[float]]:
        """GPU utilisation and VRAM from the injected provider (or empty)."""
        if self.gpu_provider is None:
            return {}
        try:
            value = self.gpu_provider()
        except Exception:
            return {}
        if not isinstance(value, dict):
            return {}

        def num(key: str) -> Optional[float]:
            try:
                raw = value.get(key)
                return None if raw is None else float(raw)
            except (TypeError, ValueError):
                return None

        return {"gpu_util": num("util"), "vram_used": num("vram_used"),
                "vram_total": num("vram_total")}

    def sample(self) -> Sample:
        now = self.monotonic()
        cpu = self._cpu_measurement()
        mem = self._call(self.memory_provider)
        up = self._call(self.uptime_provider)
        idle = self._call(self.idle_provider)
        suspended = self._detect_suspend(now, up)
        gpu = self._gpu_measurement()
        sample = Sample(at=now, cpu_load=cpu, memory_pressure=mem,
                        uptime_seconds=up, idle_seconds=idle,
                        suspended_seconds=suspended, **gpu)
        self._last = sample
        return sample

    def _cpu_measurement(self) -> Optional[float]:
        """The CPU fraction, tolerating a provider that needs two reads.

        A cumulative-counter provider (Windows ``GetSystemTimes``) returns
        ``None`` on its very first read because it has no previous sample to
        difference against. Rather than persist that ``None`` for the whole
        runtime, one immediate re-read is taken; the scratch cache is updated by
        the first call, so the second yields a real fraction. An honestly
        unmeasurable provider still returns ``None``.
        """
        cpu = self._call(self.cpu_provider)
        if cpu is None:
            cpu = self._call(self.cpu_provider)
        return cpu

    def _call(self, provider: Optional[Callable[[], Optional[float]]]) -> Optional[float]:
        if provider is None:
            return None
        try:
            value = provider()
        except Exception:
            return None
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def _detect_suspend(self, now: float,
                        uptime: Optional[float]) -> Optional[float]:
        """Detect a machine suspend from the uptime vs. wall-clock mismatch.

        If the process's own monotonic clock advanced far less than the machine
        uptime since the last sample, the machine was suspended in between. Only
        the *machine* uptime tells us this; a process restart loses the history
        and yields ``None`` (unknown), never a guess.
        """
        last = self._last
        if last is None or uptime is None or last.uptime_seconds is None:
            return None
        wall = now - last.at
        up_delta = uptime - last.uptime_seconds
        if up_delta < 0:
            return None
        # Suspend: the machine's uptime grew while our monotonic clock barely
        # moved. The shortfall is how long it slept.
        gap = up_delta - wall
        if gap >= SUSPEND_BACKLOG_SECONDS:
            return gap
        return None

    # -- pure decisions --------------------------------------------------
    def resources_available(self, sample: Optional[Sample] = None) -> bool:
        """True when measured CPU and memory are both below their ceilings.

        A missing measurement is not a problem (``None`` -> treated as okay), so
        a platform we cannot measure never gets stuck in efficiency mode.
        """
        sample = sample or self._last or self.sample()
        cpu = sample.cpu_load
        mem = sample.memory_pressure
        if cpu is not None and cpu >= self.cpu_ceiling:
            return False
        if mem is not None and mem >= self.memory_ceiling:
            return False
        return True

    def gpu_available(self, sample: Optional[Sample] = None) -> bool:
        """True when a measured GPU/VRAM is below its ceiling.

        GPU is a *separate* axis from CPU/memory, and only GPU-claiming work is
        gated on it. An unmeasurable GPU (``None``) is ``True`` here - the
        absence of a reading is never treated as a busy GPU. Likewise VRAM is
        only a problem when *pressure* is high (used/total near the ceiling);
        a machine holding a lot of VRAM but not near capacity is not blocked,
        which is exactly the "holding more memory than needed is not the same as
        being unable to take the task" case.
        """
        sample = sample or self._last or self.sample()
        util = sample.gpu_util
        if util is not None and util >= self.gpu_ceiling:
            return False
        pressure = sample.vram_pressure()
        if pressure is not None and pressure >= self.vram_ceiling:
            return False
        return True

    def memory_headroom(self, sample: Optional[Sample] = None) -> Optional[float]:
        """Free system memory as a fraction (0..1), or ``None`` when unknown.

        This is the *positive* form of ``memory_pressure``: ``1 - pressure``.
        It is what an affordability estimate multiplies a task's cost against,
        so a machine that is merely holding memory (high usage but not near the
        ceiling) still reports real headroom rather than a refusal.
        """
        sample = sample or self._last or self.sample()
        mem = sample.memory_pressure
        if mem is None:
            return None
        return max(0.0, min(1.0, 1.0 - float(mem)))

    def vram_headroom(self, sample: Optional[Sample] = None) -> Optional[float]:
        """Free VRAM as a fraction (0..1), or ``None`` when not measurable."""
        sample = sample or self._last or self.sample()
        pressure = sample.vram_pressure()
        if pressure is None:
            return None
        return max(0.0, min(1.0, 1.0 - float(pressure)))

    def warm(self, sample: Optional[Sample] = None) -> bool:
        """True when the machine has been up long enough to be fully ready."""
        sample = sample or self._last or self.sample()
        up = sample.uptime_seconds
        if up is None:
            # Unknown uptime: do not keep the machine in a cold-start mode
            # forever; fall back to the sustained-calm signal instead.
            return True
        return up >= self.warmup_seconds

    def ready_for_work(self, sample: Optional[Sample] = None) -> bool:
        """Whether Astra may do ordinary (including heavier) work now.

        Requires both a warm machine and sustainedly-available resources. The
        "sustained" part is what keeps a momentary dip (a program starting) from
        immediately admitting heavy work; calm must last ``sample_window``.

        This is the conservative *efficiency* answer used for display and the
        startup policy. The scheduler's per-task affordability check is finer
        (it compares a task's estimated cost against measured headroom), so a
        machine merely holding memory is not blanket-refused there - see
        :meth:`sustainably_calm` and :mod:`astra.resources`.
        """
        sample = sample or self._last or self.sample()
        if sample.suspended_seconds:
            # Just resumed: stay conservative until calm is re-established.
            self._calm_since = None
            return False
        if not self.resources_available(sample):
            self._calm_since = None
            return False
        if self._calm_since is None:
            self._calm_since = sample.at
        return self.warm(sample) and (sample.at - self._calm_since) >= self.sample_window

    def sustainably_calm(self, sample: Optional[Sample] = None) -> bool:
        """Warm and settled, *ignoring* the resource ceilings.

        The energy/warmup half of :meth:`ready_for_work`, with its own tracker so
        the two do not interfere. Resource affordability is a separate, per-task
        measured question (see :mod:`astra.resources`): a warm machine is warm
        regardless of how much memory it happens to be holding, so this
        deliberately does not consult CPU/memory.
        """
        sample = sample or self._last or self.sample()
        if sample.suspended_seconds:
            self._settled_since = None
            return False
        if self._settled_since is None:
            self._settled_since = sample.at
        return self.warm(sample) and (sample.at - self._settled_since) >= self.sample_window

    def efficiency_mode(self, sample: Optional[Sample] = None) -> bool:
        """True while Astra should run conservatively (low power/resource use).

        This is the inverse of readiness, kept explicit because it is the state
        the runtime reports and gates on at startup.
        """
        return not self.ready_for_work(sample)

    def about_to_sleep(self, sample: Optional[Sample] = None) -> bool:
        """Weak, honest signal that the machine is idle enough to be sleeping.

        We cannot reliably know a host is *about* to suspend, so this only
        reports the observable proxy: the user has been idle a long time and the
        machine is calm. Callers must treat it as "may sleep soon", never as a
        certainty, and must not force a sleep on it alone.
        """
        sample = sample or self._last or self.sample()
        idle = sample.idle_seconds
        if idle is None:
            return False
        return idle >= 900.0 and self.resources_available(sample)

    def cpu_reading(self) -> Optional[float]:
        """A live CPU fraction (0..1) without suspend bookkeeping.

        Used by the scheduler/reader as the numeric load that stands heavy work
        down. Unlike :meth:`sample` it does not advance suspend detection, so it
        is safe to call from another thread without disturbing the power state.
        """
        return self._call(self.cpu_provider)

    def sample_now(self) -> Sample:
        """Take a *fresh* measurement without disturbing suspend bookkeeping.

        The stored ``_last`` sample is only refreshed on the runtime's own
        cadence, so on a platform whose CPU reading needs two reads (Windows) the
        first sample is ``None`` and every later :meth:`snapshot` would keep
        reporting that stale ``None``. A status/diagnostic read wants the current
        value, so this reads the providers again but - like :meth:`cpu_reading` -
        does not update ``_last`` (which would mask a real suspend from the power
        path) and does not invent anything: an unmeasurable value is still
        ``None``.
        """
        return Sample(
            at=self.monotonic(),
            cpu_load=self._cpu_measurement(),
            memory_pressure=self._call(self.memory_provider),
            uptime_seconds=self._call(self.uptime_provider),
            idle_seconds=self._call(self.idle_provider),
            **self._gpu_measurement(),
        )

    def snapshot(self) -> Dict[str, Any]:
        """A bounded, JSON-safe view for diagnostics/status (never a memory)."""
        sample = self.sample_now()
        out = sample.as_dict()
        out.update({
            "ready_for_work": self.ready_for_work(sample),
            "efficiency_mode": self.efficiency_mode(sample),
            "resources_available": self.resources_available(sample),
            "gpu_available": self.gpu_available(sample),
            "memory_headroom": self.memory_headroom(sample),
            "vram_headroom": self.vram_headroom(sample),
            "at": _now(),
        })
        return out


def build_default(*, monotonic: Callable[[], float] = time.monotonic
                  ) -> "MachineState":
    """A :class:`MachineState` wired to this platform's real providers."""
    if default_is_windows():
        scratch: Dict[str, Any] = {}

        def cpu() -> Optional[float]:
            return _windows_cpu_load(scratch)

        return MachineState(
            cpu_provider=cpu, memory_provider=_windows_memory_pressure,
            uptime_provider=_windows_uptime_seconds,
            idle_provider=_windows_idle_seconds,
            gpu_provider=default_gpu_reading, monotonic=monotonic,
        )
    return MachineState(
        cpu_provider=_posix_cpu_load, memory_provider=_posix_memory_pressure,
        uptime_provider=_posix_uptime_seconds,
        gpu_provider=default_gpu_reading, monotonic=monotonic)


__all__ = [
    "MachineState", "Sample", "build_default", "default_gpu_reading",
    "DEFAULT_CPU_CEILING", "DEFAULT_MEMORY_CEILING",
    "DEFAULT_GPU_CEILING", "DEFAULT_VRAM_CEILING",
    "DEFAULT_WARMUP_SECONDS", "DEFAULT_SAMPLE_WINDOW", "SUSPEND_BACKLOG_SECONDS",
]
