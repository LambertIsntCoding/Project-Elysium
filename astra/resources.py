"""Resource history and per-task cost estimation: "can this actually be done?"

The machine module measures the machine *now*. This module remembers what the
machine looks like over time - and, crucially, what each kind of background task
actually *costs* - so a decision to stand heavy work down is based on a measured
estimate rather than a single aggregate number.

The motivating correction: a high headline number is not proof a task is
impossible. A machine holding a lot of memory may still have real headroom; a
busy GPU does not block a CPU-only task. So instead of refusing whenever a
number crosses a policy line, we ask the honest question:

    does this task's *estimated* additional need still fit inside the measured
    headroom, for each resource it actually uses?

Only when a resource the task genuinely needs is *demonstrably* near saturation
(``current + estimated_cost > saturation``) is the task deferred - and a defer is
a *skip*, never a failure (the scheduler keeps the task pending for a later turn).
Crucially, a deferral can never outlast the machine: saturation is a *point in
time*, not a permanent verdict. Before the first measurement of a resource - or
while every measurement is stale - the task is admitted rather than refused, so
an unmeasurable (or never-yet-sampled) resource is never an indefinite blocker.
Reading must be able to start on a fresh machine, not wait forever for a history
that only a first read would create.

Two ideas do the work:

* **Active vs idle.** An observation is tagged with whether Roum was engaged.
  The idle baseline is what the machine looks like when nobody is using it; the
  active baseline is what his applications look like. Comparing them is how we
  separate "the machine is busy because Roum is working" from "the machine is
  busy because of us".
* **Learned cost.** When a task runs, the resource delta across it is measured
  and folded (EWMA) into a per-kind cost, seeded from a conservative prior. So
  the estimate for the *next* task of that kind is grounded in what the previous
  one actually took - the "previous usage case" the brief asks for.

Design constraints (consistent with the rest of the project):

* **Measured, never guessed.** Every value is a real reading or ``None``. An
  unmeasurable resource is never treated as busy, and never blocks anything.
* **Bounded and durable.** One small JSON file, capped, atomically written - the
  same discipline as the task store and ledger.
* **Pure decisions.** :func:`affordable` is a pure function of a cost estimate,
  a sample and the saturation bounds, so the policy is testable without a host.
* **Display vs decision.** The numbers here are for the scheduler's decision and
  the admin's view. They never reach Astra's prompt.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import tasks as task_mod
from .memory import atomic_save, load_json

RESOURCE_VERSION = 1

# How many recent observations to keep. Enough to see a trend, bounded so the
# file cannot grow without limit.
MAX_OBSERVATIONS = 240
# How many recent observations feed a baseline mean.
BASELINE_WINDOW = 60
# How long an observation stays "current" for the affordability check. Beyond
# this the reading is stale and is treated as unmeasurable (admitted, not
# refused): a snapshot taken an hour ago must never hold a task down now.
OBSERVATION_TTL_SECONDS = 120.0
# A kind's learned cost needs at least this many real observations before it
# fully replaces the conservative prior (below it we blend toward the prior).
MIN_COST_SAMPLES = 2
# EWMA weight for a new observation folded into a kind's cost.
COST_EWMA_ALPHA = 0.4

# Why a task was deferred for resources (never a failure).
AFFORD_OK = "ok"
LIMITED_CPU = "cpu_headroom"
LIMITED_MEMORY = "memory_headroom"
LIMITED_GPU = "gpu_headroom"
LIMITED_VRAM = "vram_headroom"

# The *saturation* bounds - genuinely "there is no room for more", as opposed to
# the conservative policy ceilings in :mod:`astra.machine` (which describe
# "efficiency mode", not "impossible"). A task is deferred only when its measured
# cost would push a resource it uses past one of these.
SATURATION_CPU = 0.90
SATURATION_MEMORY = 0.95
SATURATION_GPU = 0.95
SATURATION_VRAM = 0.97


@dataclass
class ResourceCost:
    """The estimated additional load of one task, as fractions (0..1)."""

    cpu: float = 0.0
    mem: float = 0.0
    gpu: float = 0.0
    vram: float = 0.0

    def uses(self, resource: str) -> bool:
        return getattr(self, resource, 0.0) > 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"cpu": self.cpu, "mem": self.mem,
                "gpu": self.gpu, "vram": self.vram}


# Which resources each kind of work actually uses. This is what keeps the axes
# separate: a CPU-only task is never blocked by a busy GPU, and a GPU task is
# never blocked by a high system-memory number it does not touch. ``reading``
# uses the model (Ollama), which may run on the GPU, so it claims GPU/VRAM too.
_ALL = ("cpu", "mem", "gpu", "vram")
_CPU_MEM = ("cpu", "mem")
_CPU_ONLY = ("cpu",)
KIND_RESOURCES: Dict[str, Tuple[str, ...]] = {
    task_mod.KIND_READING: _ALL,
    task_mod.KIND_TEST_RUN: _CPU_MEM,
    task_mod.KIND_FILE_INSPECTION: _CPU_MEM,
    task_mod.KIND_CALCULATION: _CPU_MEM,
    task_mod.KIND_CONTEXT_PREPARATION: _CPU_MEM,
    task_mod.KIND_WORK_REVIEW: _CPU_MEM,
    task_mod.KIND_MEMORY_MAINTENANCE: _CPU_ONLY,
    task_mod.KIND_MEMORY_REVIEW: _CPU_ONLY,
    task_mod.KIND_EXPERIENCE_REVIEW: _CPU_ONLY,
    task_mod.KIND_QUESTION_REVIEW: _CPU_ONLY,
    task_mod.KIND_SELF_REVIEW: _CPU_ONLY,
    task_mod.KIND_KNOWLEDGE_GAP: _CPU_ONLY,
    task_mod.KIND_SELF_IMPROVEMENT: _CPU_ONLY,
    task_mod.KIND_APP_MAINTENANCE: _CPU_ONLY,
}

# Conservative priors used until a kind has been observed. Deliberately modest:
# a prior is a starting guess, and an over-large prior would defer work we could
# actually do. Only the resources the kind uses are non-zero.
DEFAULT_COST: Dict[str, Dict[str, float]] = {
    task_mod.KIND_READING: {"cpu": 0.08, "mem": 0.02, "gpu": 0.05, "vram": 0.04},
    task_mod.KIND_TEST_RUN: {"cpu": 0.20, "mem": 0.06},
    task_mod.KIND_FILE_INSPECTION: {"cpu": 0.05, "mem": 0.02},
    task_mod.KIND_CALCULATION: {"cpu": 0.10, "mem": 0.02},
    task_mod.KIND_CONTEXT_PREPARATION: {"cpu": 0.06, "mem": 0.03},
    task_mod.KIND_WORK_REVIEW: {"cpu": 0.06, "mem": 0.03},
}
DEFAULT_LIGHT_COST = {"cpu": 0.02}


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _utc_epoch() -> float:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).timestamp()


def _epoch_of(value: Any) -> Optional[float]:
    """Parse an ISO-8601 timestamp to a POSIX epoch, or ``None`` if unparseable."""
    if not value:
        return None
    try:
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(str(value))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.timestamp()
    except (TypeError, ValueError):
        return None


def prior_cost(kind: str) -> ResourceCost:
    """The conservative prior for a kind, limited to the resources it uses."""
    used = KIND_RESOURCES.get(kind, _CPU_ONLY)
    table = DEFAULT_COST.get(kind, DEFAULT_LIGHT_COST)
    return ResourceCost(
        cpu=float(table.get("cpu", 0.0)) if "cpu" in used else 0.0,
        mem=float(table.get("mem", 0.0)) if "mem" in used else 0.0,
        gpu=float(table.get("gpu", 0.0)) if "gpu" in used else 0.0,
        vram=float(table.get("vram", 0.0)) if "vram" in used else 0.0,
    )


def _delta(before: Dict[str, Any], after: Dict[str, Any],
           key: str) -> Optional[float]:
    """A non-negative resource delta between two samples, or ``None``."""
    try:
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None:
            return None
        return max(0.0, float(a) - float(b))
    except (TypeError, ValueError):
        return None


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


class ResourceHistory:
    """Durable, bounded history of machine observations and per-kind costs."""

    def __init__(self, data_dir: str = "./storage") -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "resource_history.json")
        self._lock = threading.RLock()
        self._observations: List[Dict[str, Any]] = []
        self._costs: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        raw = load_json(self.path, dict, dict)
        obs = raw.get("observations") if isinstance(raw, dict) else None
        costs = raw.get("kind_costs") if isinstance(raw, dict) else None
        self._observations = [dict(o) for o in obs or [] if isinstance(o, dict)]
        self._costs = {str(k): dict(v) for k, v in (costs or {}).items()
                       if isinstance(v, dict)}

    def reload(self) -> None:
        """Re-read from disk, for a separate read-only inspection process."""
        with self._lock:
            self._load()

    def _persist(self) -> None:
        with self._lock:
            atomic_save(self.path, {
                "version": RESOURCE_VERSION,
                "observations": self._observations[-MAX_OBSERVATIONS:],
                "kind_costs": self._costs,
            })

    # -- recording -------------------------------------------------------
    def observe(self, sample: Any, *, active: bool = False) -> Dict[str, Any]:
        """Record one machine observation (bounded, compact).

        ``sample`` may be a :class:`astra.machine.Sample` or a mapping. Only the
        numeric readings are stored; ``None`` stays ``None``.
        """
        def get(key: str) -> Optional[float]:
            raw = (sample.get(key) if isinstance(sample, dict)
                   else getattr(sample, key, None))
            try:
                return None if raw is None else float(raw)
            except (TypeError, ValueError):
                return None

        vram_pressure = None
        if hasattr(sample, "vram_pressure") and callable(sample.vram_pressure):
            vram_pressure = sample.vram_pressure()
        else:
            used, total = get("vram_used"), get("vram_total")
            if used is not None and total:
                vram_pressure = max(0.0, min(1.0, used / total))

        entry = {
            "at": _now(),
            "cpu": get("cpu_load"),
            "mem": get("memory_pressure"),
            "gpu": get("gpu_util"),
            "vram": vram_pressure,
            "active": bool(active),
        }
        with self._lock:
            self._observations.append(entry)
            if len(self._observations) > MAX_OBSERVATIONS:
                self._observations = self._observations[-MAX_OBSERVATIONS:]
            self._persist()
        return entry

    def record_task(self, kind: str, before: Any, after: Any) -> None:
        """Fold the measured delta of a completed task into its kind's cost.

        ``before``/``after`` are machine readings (mapping or ``Sample``) taken
        around the task. Only resources the kind uses are updated, and only when
        both readings exist - a missing measurement never invents a cost.
        """
        def as_map(obj: Any) -> Dict[str, Any]:
            if isinstance(obj, dict):
                return obj
            out = {}
            for key in ("cpu_load", "memory_pressure", "gpu_util"):
                out[key] = getattr(obj, key, None)
            if hasattr(obj, "vram_pressure") and callable(obj.vram_pressure):
                out["vram_pressure"] = obj.vram_pressure()
            return out

        b, a = as_map(before), as_map(after)
        used = KIND_RESOURCES.get(kind, _CPU_ONLY)
        deltas = {
            "cpu": _delta(b, a, "cpu_load"),
            "mem": _delta(b, a, "memory_pressure"),
            "gpu": _delta(b, a, "gpu_util"),
            "vram": _delta(b, a, "vram_pressure"),
        }
        with self._lock:
            entry = self._costs.get(kind) or prior_cost(kind).as_dict()
            entry = dict(entry)
            samples = int(entry.get("samples", 0))
            for resource in used:
                observed = deltas.get(resource)
                if observed is None:
                    continue
                current = float(entry.get(resource, 0.0) or 0.0)
                entry[resource] = round(
                    (1.0 - COST_EWMA_ALPHA) * current
                    + COST_EWMA_ALPHA * observed, 5)
            entry["samples"] = samples + 1
            self._costs[kind] = entry
            self._persist()

    # -- reads -----------------------------------------------------------
    def cost(self, kind: str) -> ResourceCost:
        """The estimated cost for a kind: learned when observed, else the prior."""
        used = KIND_RESOURCES.get(kind, _CPU_ONLY)
        with self._lock:
            entry = self._costs.get(kind)
        prior = prior_cost(kind)
        if not entry or int(entry.get("samples", 0)) < MIN_COST_SAMPLES:
            return prior
        return ResourceCost(
            cpu=float(entry.get("cpu", prior.cpu)) if "cpu" in used else 0.0,
            mem=float(entry.get("mem", prior.mem)) if "mem" in used else 0.0,
            gpu=float(entry.get("gpu", prior.gpu)) if "gpu" in used else 0.0,
            vram=float(entry.get("vram", prior.vram)) if "vram" in used else 0.0,
        )

    def _baseline(self, *, active: bool) -> Dict[str, Optional[float]]:
        with self._lock:
            rows = [o for o in self._observations[-BASELINE_WINDOW:]
                    if bool(o.get("active")) == active]
        out: Dict[str, Optional[float]] = {}
        for key in ("cpu", "mem", "gpu", "vram"):
            vals = [float(o[key]) for o in rows if o.get(key) is not None]
            out[key] = _mean(vals)
        out["samples"] = len(rows)
        return out

    def idle_baseline(self) -> Dict[str, Optional[float]]:
        """What the machine looks like when Roum is *not* using it."""
        return self._baseline(active=False)

    def active_baseline(self) -> Dict[str, Optional[float]]:
        """What the machine looks like while Roum is using it (his applications)."""
        return self._baseline(active=True)

    def comparison(self) -> Dict[str, Any]:
        """The active-vs-idle comparison the brief asks for (read-only)."""
        return {"idle": self.idle_baseline(), "active": self.active_baseline()}

    def kind_costs(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._costs.items()}

    def freshness(self) -> Dict[str, bool]:
        """Whether each resource has a *current* (non-stale) measurement.

        A resource is "fresh" when the most recent observation that actually
        carried a value for it is within :data:`OBSERVATION_TTL_SECONDS`. This is
        what lets the affordability check distinguish "measured and busy" from
        "never measured / stale", so an unmeasurable or long-ago-sampled
        resource is admitted rather than treated as a permanent block.
        """
        with self._lock:
            rows = list(self._observations)
        now = _utc_epoch()
        out: Dict[str, bool] = {}
        for key in ("cpu", "mem", "gpu", "vram"):
            fresh = False
            for row in reversed(rows):
                if row.get(key) is None:
                    continue
                at = _epoch_of(row.get("at"))
                fresh = at is not None and (now - at) <= OBSERVATION_TTL_SECONDS
                break
            out[key] = fresh
        return out

    def last_observation_at(self) -> Optional[str]:
        with self._lock:
            return str(self._observations[-1].get("at")) if self._observations else None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "observations": len(self._observations),
            "idle_baseline": self.idle_baseline(),
            "active_baseline": self.active_baseline(),
            "kind_costs": self.kind_costs(),
            "freshness": self.freshness(),
            "last_observation_at": self.last_observation_at(),
        }


# ---------------------------------------------------------------------------
# Affinity: which resource a task actually contends for, and by how much
# ---------------------------------------------------------------------------
# A task is deferred only for the resource it genuinely competes for, and only
# when it would push that resource past saturation. Cost alone is not enough:
# e.g. a large model may *hold* a lot of memory (high absolute cost) yet not be
# adding to memory pressure - holding is not contending. So the affordability
# test asks two honest questions per resource the task uses:
#
#   1. is the resource already near saturation? (``current >= saturation``)
#   2. would this task's estimated cost push it over? (``current + cost > sat``)
#
# A resource the machine is not near its limit on never blocks, however large
# the task's absolute footprint looks.
RESOURCE_CPU = "cpu"
RESOURCE_MEMORY = "memory"
RESOURCE_GPU = "gpu"
RESOURCE_VRAM = "vram"
RESOURCE_USAGE: Dict[str, Tuple[str, str]] = {
    "cpu": ("cpu_load", RESOURCE_CPU),
    "mem": ("memory_pressure", RESOURCE_MEMORY),
    "gpu": ("gpu_util", RESOURCE_GPU),
    "vram": ("vram_pressure", RESOURCE_VRAM),
}


def _sample_value(sample: Any, key: str) -> Optional[float]:
    """Read one resource fraction from a ``Sample`` or a mapping, or ``None``."""
    if key == "vram":
        if hasattr(sample, "vram_pressure") and callable(sample.vram_pressure):
            try:
                return sample.vram_pressure()
            except Exception:
                return None
        if isinstance(sample, dict):
            used, total = sample.get("vram_used"), sample.get("vram_total")
            if used is not None and total:
                try:
                    return max(0.0, min(1.0, float(used) / float(total)))
                except (TypeError, ValueError):
                    return None
        return None
    attr = RESOURCE_USAGE[key][0]
    raw = (sample.get(attr) if isinstance(sample, dict)
           else getattr(sample, attr, None))
    try:
        return None if raw is None else float(raw)
    except (TypeError, ValueError):
        return None


def affordable(cost: ResourceCost, sample: Any, *,
               cpu_saturation: float = SATURATION_CPU,
               mem_saturation: float = SATURATION_MEMORY,
               gpu_saturation: float = SATURATION_GPU,
               vram_saturation: float = SATURATION_VRAM) -> Tuple[bool, str]:
    """Whether a task's estimated cost still fits, per resource it actually uses.

    Pure. A resource the task does not use is never consulted, so a busy GPU
    cannot block a CPU-only task and a high system-memory number cannot block a
    GPU-only one. A resource we cannot measure is skipped (unknown is not busy),
    so an unmeasurable resource can never become a permanent reason to refuse
    work. ``sample`` is a *live* reading, so it is the current truth: a resource
    that is genuinely saturated does stand the task down (a legitimate limit),
    but the next cycle re-samples it and re-decides, so a defer never outlasts
    the condition that caused it.

    A resource is only a reason to defer when the task *uses* it and the machine
    is already near saturation such that the task's estimated cost would push it
    over the bound. This is the honest "can this actually be done?" test: a
    machine merely holding memory is not near saturation, so it does not refuse a
    memory-using task.

    Returns ``(True, AFFORD_OK)`` when it fits, else ``(False, <reason>)`` naming
    the first resource that would saturate. A ``False`` is a *defer*, never a
    failure - the caller keeps the task pending for a later turn.
    """
    bounds = {"cpu": cpu_saturation, "mem": mem_saturation,
              "gpu": gpu_saturation, "vram": vram_saturation}
    reasons = {"cpu": LIMITED_CPU, "mem": LIMITED_MEMORY,
               "gpu": LIMITED_GPU, "vram": LIMITED_VRAM}
    for resource in ("cpu", "mem", "gpu", "vram"):
        need = getattr(cost, resource, 0.0)
        if need <= 0.0:
            continue  # the task does not use this resource: never consulted
        now = _sample_value(sample, resource)
        if now is None:
            continue  # unmeasurable: unknown, not busy
        if now + need > bounds[resource]:
            return False, reasons[resource]
    return True, AFFORD_OK


__all__ = [
    "ResourceCost", "ResourceHistory", "affordable", "prior_cost",
    "KIND_RESOURCES", "DEFAULT_COST", "RESOURCE_USAGE",
    "AFFORD_OK", "LIMITED_CPU", "LIMITED_MEMORY", "LIMITED_GPU", "LIMITED_VRAM",
    "SATURATION_CPU", "SATURATION_MEMORY", "SATURATION_GPU", "SATURATION_VRAM",
    "MAX_OBSERVATIONS", "MIN_COST_SAMPLES", "OBSERVATION_TTL_SECONDS",
]
