"""Derived memory priority: an ordering index, not a truth.

The scheduler and the retriever want to know *which existing memories matter
right now* without asking the model to re-rank the whole store. This module
computes that from metadata the memory system already governs - effective
strength, evidence, confidence, recency/use, reinforcement, unresolvedness,
work relevance, relational relevance, dormancy and conflict state - and returns
a number plus a short breakdown.

Hard rules (from the spec and the project invariants):

* **Derived, never stored.** Nothing here is written to a memory record, and
  there is no second memory database. Call it, use it, discard it.
* **Never a truth category.** Priority orders attention; it does not change a
  memory's classification, authority, or status.
* **Explicit always outranks inferred.** An inferred record can never outrank an
  explicit one merely because it was retrieved often: provenance is applied as a
  hard tier, and the inferred tier is capped below the sourced tier.
* **Nothing is deleted.** This only reads.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from . import inquiry, memory

# Provenance tiers: a sourced record (explicit user statement/correction) sits in
# a strictly higher band than an inferred one, so frequency can never invert the
# authority order. Band width is wider than any in-band bonus can cross.
_TIER_SOURCED = 1000.0
_TIER_INFERRED = 100.0
_TIER_OTHER = 400.0  # relationship/self records that are neither user-sourced nor AI inference


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _tier(mem: Dict[str, Any]) -> float:
    if memory.is_sourced(mem):
        return _TIER_SOURCED
    if memory.is_astra_source(mem.get("source")):
        return _TIER_INFERRED
    return _TIER_OTHER


def _unresolved_bonus(mem: Dict[str, Any]) -> float:
    if inquiry.is_open_question(mem):
        # A live line of inquiry is worth attention; a resolved/abandoned one is
        # not, which is exactly what the qstatus tag records.
        return 30.0 if inquiry.question_status_of(mem) in inquiry.LIVE_QUESTION_STATUSES else 0.0
    if str(mem.get("type") or "") == inquiry.TYPE_INTERPRETATION and mem.get("contradicts"):
        return 25.0
    return 0.0


def _work_bonus(mem: Dict[str, Any], active_work: Optional[str]) -> float:
    work_id = inquiry.work_of(mem)
    if work_id and active_work and work_id == active_work:
        return 20.0
    return 0.0


def _relational_bonus(mem: Dict[str, Any]) -> float:
    tags = mem.get("tags") or []
    if isinstance(tags, (list, tuple)) and any(
            str(t).strip().lower() in ("relational_preference", "relational_event")
            for t in tags):
        return 10.0
    return 0.0


def priority(mem: Dict[str, Any], *, now: Optional[datetime] = None,
             active_work: Optional[str] = None) -> Dict[str, Any]:
    """A priority score and its breakdown for one memory.

    Returns ``{"score", "tier", "factors"}``. The score is monotonic in the
    things that matter and, crucially, tier-separated: no amount of in-band
    evidence lets an inferred record pass a sourced one.
    """
    strength = memory.effective_strength(mem, now)          # 0..1, the workhorse
    importance = memory.memory_importance(mem)              # 0..1
    reinforce = _clamp01(int(mem.get("reinforcement_count", 1) or 1) / 6.0)
    used = _clamp01(int(mem.get("use_count", 0) or 0) / 8.0)
    unresolved = _unresolved_bonus(mem)
    work = _work_bonus(mem, active_work)
    relational = _relational_bonus(mem)
    dormant = memory.is_dormant(mem)

    in_band = (
        200.0 * strength
        + 60.0 * importance
        + 20.0 * reinforce
        + 15.0 * used
        + unresolved + work + relational
    )
    if dormant:
        in_band *= 0.6  # dormant records stay readable but never dominate

    score = _tier(mem) + in_band
    return {
        "score": round(score, 4),
        "tier": ("sourced" if _tier(mem) == _TIER_SOURCED
                 else "inferred" if _tier(mem) == _TIER_INFERRED else "other"),
        "factors": {
            "strength": round(strength, 4),
            "importance": round(importance, 4),
            "reinforcement": round(reinforce, 4),
            "use": round(used, 4),
            "unresolved": unresolved,
            "work": work,
            "relational": relational,
            "dormant": bool(dormant),
        },
    }


def score(mem: Dict[str, Any], *, now: Optional[datetime] = None,
          active_work: Optional[str] = None) -> float:
    return priority(mem, now=now, active_work=active_work)["score"]


def ranked(memories: Iterable[Dict[str, Any]], *, now: Optional[datetime] = None,
           active_work: Optional[str] = None,
           limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Memories annotated with their derived priority, highest first.

    Deterministic: ties break on id so the same input always yields the same
    order. The returned dicts are shallow copies - the stored records are never
    mutated.
    """
    scored: List[tuple] = []
    for mem in memories or []:
        info = priority(mem, now=now, active_work=active_work)
        scored.append((info["score"], str(mem.get("id") or ""), mem, info))
    scored.sort(key=lambda row: (-row[0], row[1]))
    out: List[Dict[str, Any]] = []
    for total, _id, mem, info in scored[:limit] if limit else scored:
        annotated = dict(mem)
        annotated["_priority"] = info
        out.append(annotated)
    return out


def top(memories: Iterable[Dict[str, Any]], *, now: Optional[datetime] = None,
        active_work: Optional[str] = None, limit: int = 8) -> List[Dict[str, Any]]:
    return ranked(memories, now=now, active_work=active_work, limit=limit)


def diagnostics(memories: Iterable[Dict[str, Any]], *,
                active_work: Optional[str] = None,
                limit: int = 5) -> List[Dict[str, Any]]:
    """A compact breakdown of the highest-priority memories, for inspection."""
    rows = []
    for mem in ranked(memories, active_work=active_work, limit=limit):
        info = mem["_priority"]
        rows.append({
            "id": mem.get("id"),
            "type": mem.get("type"),
            "score": info["score"],
            "tier": info["tier"],
            "factors": info["factors"],
        })
    return rows
