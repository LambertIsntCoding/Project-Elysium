"""Live observation screens: relationship, status, and affect.

Three continuously-refreshing pages that make Astra's internal state
*observable*. They are external/administrative views for Roum: they may show
useful numeric statistics, but none of this ever reaches Astra's prompt - she
continues to receive the natural-language summaries the rest of the system
builds. Each screen is a pure renderer over the live stores plus a bounded
rolling snapshot buffer (see :mod:`astra.admin`), so it can be watched in real
time instead of inferred from dialogue.

The screens read only. They never write a memory, never mutate affect or the
relationship, and never route anything through the model.

Each builder takes the session/controller and returns a
:class:`astra.admin.LiveScreen`, so the menu layer can open them uniformly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import affect as affect_mod
from . import behavior as behavior_mod
from . import relational
from . import telemetry
from .admin import LiveScreen, SnapshotStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _controller(session: Any) -> Any:
    return getattr(session, "controller", None)


def _orchestrator(session: Any) -> Any:
    return getattr(session, "orchestrator", None)


# ---------------------------------------------------------------------
# Shared keyboard-interaction label
# ---------------------------------------------------------------------
_HELP = ("Enter=refresh  s=snapshot  h=history  q=leave")


# ---------------------------------------------------------------------
# Relationship screen
# ---------------------------------------------------------------------
def relationship_render(session: Any) -> List[str]:
    orch = _orchestrator(session)
    store = getattr(orch, "store", None) or getattr(session, "store", None)
    memories = store.get_memories("relationship", status=None) if store is not None else []
    state = relational.load_state_from_memories(memories)
    diag = relational.affinity_diagnostics(state)

    lines = ["=== RELATIONSHIP (LIVE) ==="]
    lines.append(f"  status: {diag.get('relationship')}   "
                 f"established: {diag.get('established')}")
    lines.append("  current values (0-100):")
    names = telemetry.RELATIONSHIP_NAMES
    order = list(relational.COMPONENTS)
    for key in order:
        value = diag.get(key)
        if isinstance(value, (int, float)):
            lines.append(telemetry.format_line(names.get(key, key), value))
    lines.append(f"  recent change: {diag.get('recent_change'):+}")
    reason = str(diag.get("reason") or "").strip()
    lines.append(f"  reason: {reason or '(none recorded)'}")
    lines.append("  what she currently considers important:")
    evidence = diag.get("evidence") or []
    if evidence:
        for item in evidence[-4:]:
            lines.append(f"    - {item}")
    else:
        lines.append("    (no accumulated evidence yet)")
    lines.append("  astra sees a natural-language summary of this, never the numbers above.")
    return lines


def relationship_snapshot(session: Any) -> Dict[str, Any]:
    orch = _orchestrator(session)
    store = getattr(orch, "store", None) or getattr(session, "store", None)
    memories = store.get_memories("relationship", status=None) if store is not None else []
    state = relational.load_state_from_memories(memories)
    diag = relational.affinity_diagnostics(state)
    values = {k: diag.get(k) for k in relational.COMPONENTS
              if isinstance(diag.get(k), (int, float))}
    return {
        "screen": "relationship",
        "captured_at": _now(),
        "relationship": diag.get("relationship"),
        "established": diag.get("established"),
        "values": values,
        "values_scaled": {k: telemetry.to_scale(v) for k, v in values.items()},
        "recent_change": diag.get("recent_change"),
        "reason": diag.get("reason"),
        "evidence": list(diag.get("evidence") or [])[-4:],
    }


# ---------------------------------------------------------------------
# Status screen
# ---------------------------------------------------------------------
def status_render(session: Any) -> List[str]:
    lines = ["=== STATUS (LIVE) ==="]
    controller = _controller(session)
    orch = _orchestrator(session)

    # Runtime state (Active/Sleep) - a fact about the process, not a feeling.
    if controller is not None:
        state = getattr(controller, "state", None)
        current = state.current() if callable(getattr(state, "current", None)) else {}
        lines.append(f"  runtime: {current.get('state', 'unknown')} "
                     f"(since {current.get('since', '?')}, reason {current.get('reason', '?')})")
        counts = getattr(getattr(controller, "task_store", None), "counts", None)
        if callable(counts):
            try:
                tally = counts() or {}
                visible = ", ".join(f"{k}={v}" for k, v in sorted(tally.items()) if v)
                lines.append(f"  tasks: {visible or 'none'}")
            except Exception:
                pass
        if hasattr(controller, "status_report"):
            try:
                report = controller.status_report()
                lines.extend(f"  {ln}" for ln in report.splitlines() if ln.strip()[:12])
            except Exception:
                pass
    else:
        lines.append("  runtime: (background runtime not attached)")

    # Conversation idle + reader/activity.
    reader = getattr(session, "reader", None)
    if reader is not None:
        last_cycle = getattr(reader, "last_cycle", None)
        reason = getattr(reader, "last_reason", None)
        lines.append(f"  background reading: {'running' if reader.running() else 'stopped'}")
        lines.append(f"  reader gate: {reason or '(not evaluated yet)'}")
        if isinstance(last_cycle, dict):
            lines.append(f"  last read: {last_cycle.get('outcome')} "
                         f"({last_cycle.get('words')} words at {last_cycle.get('at')})")
    else:
        lines.append("  background reading: (not attached)")

    # Current work item + elapsed time.
    works = orch._works() if orch is not None and hasattr(orch, "_works") else []
    live = [w for w in works if str(w.get("status") or "") not in ("finished", "abandoned")]
    if live:
        current = live[0]
        lines.append(f"  current work: {current.get('title') or current.get('work_id')} "
                     f"({current.get('status')})")
        lines.append(f"    progress: {int(current.get('words_read') or 0):,}/"
                     f"{int(current.get('total_words') or 0):,} words")
    else:
        lines.append("  current work: (none)")

    # Behavioural modulation: what her current state is doing to her behaviour.
    if orch is not None and callable(getattr(orch, "current_modulation", None)):
        mod = orch.current_modulation()
        if not behavior_mod.is_neutral(mod):
            lines.append("  behavioural drive (0-100):")
            for key in ("energy", "initiative", "enthusiasm", "pace", "persistence",
                        "interest", "brevity", "hesitation"):
                lines.append(telemetry.format_line(key.title(), mod.get(key, 0.5),
                                                   show_band=False))
            for reason in mod.get("reasons") or []:
                lines.append(f"    cause: {reason}")

    # Temporal state.
    store = getattr(orch, "store", None)
    if store is not None and callable(getattr(store, "last_present", None)):
        lines.append(f"  last present: {store.last_present() or '(not recorded)'}")
    lines.append("  " + _HELP)
    return lines


def status_snapshot(session: Any) -> Dict[str, Any]:
    controller = _controller(session)
    orch = _orchestrator(session)
    state = getattr(controller, "state", None) if controller is not None else None
    current = state.get() if callable(getattr(state, "get", None)) else {}
    reader = getattr(session, "reader", None)
    mod = orch.current_modulation() if (orch is not None and callable(
        getattr(orch, "current_modulation", None))) else {}
    return {
        "screen": "status",
        "captured_at": _now(),
        "runtime": current.get("state"),
        "runtime_since": current.get("since"),
        "reading": bool(reader.running()) if reader is not None else False,
        "reader_reason": getattr(reader, "last_reason", None) if reader else None,
        "modulation": {k: v for k, v in (mod or {}).items() if isinstance(v, (int, float))},
        "modulation_reasons": list((mod or {}).get("reasons") or []),
    }


# ---------------------------------------------------------------------
# Affect screen
# ---------------------------------------------------------------------
def affect_render(session: Any, store: Optional[SnapshotStore] = None) -> List[str]:
    orch = _orchestrator(session)
    state_store = getattr(orch, "store", None)
    memories = state_store.get_memories("self", status=None) if state_store is not None else []
    state = affect_mod.load_state_from_memories(memories)
    diag = affect_mod.diagnostics(state)

    lines = ["=== AFFECT (LIVE) ==="]
    if diag.get("neutral"):
        lines.append("  (neutral - no particular condition right now)")
    # Previous snapshot gives us direction and "recent changes".
    previous = None
    if store is not None:
        hist = store.recent("affect", limit=1)
        if hist:
            previous = hist[-1].get("values")
    lines.append("  current values (0-100):")
    for key in affect_mod.AFFECT_COMPONENTS:
        value = diag.get(key)
        prev = previous.get(key) if isinstance(previous, dict) else None
        lines.append(telemetry.format_line(
            telemetry.AFFECT_NAMES.get(key, key), value, prev))

    if isinstance(previous, dict):
        changed = telemetry.significant_changes(
            {k: diag.get(k) for k in affect_mod.AFFECT_COMPONENTS}, previous)
        if changed:
            lines.append("  recent changes:")
            reason = str(diag.get("reason") or "").strip()
            for key in changed:
                lines.append(telemetry.format_change(
                    telemetry.AFFECT_NAMES.get(key, key),
                    previous.get(key), diag.get(key), cause=reason))
        else:
            lines.append("  recent changes: (none since the last snapshot)")
    else:
        lines.append("  (press 's' to capture a snapshot; direction appears after that)")

    last_updated = state.get("last_updated") if isinstance(state, dict) else ""
    lines.append(f"  last moved: {last_updated or '(never)'}")
    lines.append(f"  last cause: {diag.get('reason') or '(none recorded)'}")
    counts = diag.get("event_counts") or {}
    if counts:
        recent = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())[:6])
        lines.append(f"  event counts: {recent}")
    lines.append("  astra receives a natural-language summary of this, never these numbers.")
    lines.append("  " + _HELP)
    return lines


def affect_snapshot(session: Any) -> Dict[str, Any]:
    orch = _orchestrator(session)
    state_store = getattr(orch, "store", None)
    memories = state_store.get_memories("self", status=None) if state_store is not None else []
    state = affect_mod.load_state_from_memories(memories)
    diag = affect_mod.diagnostics(state)
    values = {k: diag.get(k) for k in affect_mod.AFFECT_COMPONENTS
              if isinstance(diag.get(k), (int, float))}
    return {
        "screen": "affect",
        "captured_at": _now(),
        "values": values,
        "values_scaled": {k: telemetry.to_scale(v) for k, v in values.items()},
        "neutral": diag.get("neutral"),
        "reason": diag.get("reason"),
        "last_updated": state.get("last_updated") if isinstance(state, dict) else "",
        "event_counts": diag.get("event_counts") or {},
    }


# ---------------------------------------------------------------------
# Screen factories
# ---------------------------------------------------------------------
def build_live_screens(session: Any, snapshots: SnapshotStore) -> Dict[str, LiveScreen]:
    """The three live screens, sharing one bounded snapshot store."""
    return {
        "relationship": LiveScreen(
            "relationship", render=lambda: relationship_render(session),
            snapshot=lambda: relationship_snapshot(session), store=snapshots),
        "status": LiveScreen(
            "status", render=lambda: status_render(session),
            snapshot=lambda: status_snapshot(session), store=snapshots),
        "affect": LiveScreen(
            "affect", render=lambda: affect_render(session, snapshots),
            snapshot=lambda: affect_snapshot(session), store=snapshots),
    }


__all__ = [
    "relationship_render",
    "relationship_snapshot",
    "status_render",
    "status_snapshot",
    "affect_render",
    "affect_snapshot",
    "build_live_screens",
]
