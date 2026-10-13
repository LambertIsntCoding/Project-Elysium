"""Durable improvement proposals awaiting Roum's acknowledgement.

The brief draws a hard line the code already respects: autonomous work may do
*routine, reversible maintenance* on its own, but a change to core identity,
behavioural boundaries, security settings, credentials, access permissions or
source code must remain a **proposal requiring explicit approval**. Until this
module, that proposal existed only as a transient ledger line; there was no
durable place to hold it, acknowledge it, or refuse it.

This is that place, and it is deliberately small:

* A proposal is *recorded* by an autonomous activity (currently the
  ``self_improvement`` handler) with a title, the evidence that justified it, and
  the scope it would touch.
* Roum can **acknowledge** it (stock-taking, "I've seen this") or **approve**
  it (explicitly permits the change) or **decline** it. Nothing here *applies* a
  change: approving only flips a status a separate, human-driven step would
  consume. The store never edits code, identity, boundaries or credentials.
* Only one open proposal per key exists at a time, so a recurring review cannot
  pile up duplicates - exactly like the task store's dedupe.

It is application state for reporting and control, never autobiography: nothing
here is a memory and nothing here reaches the prompt.
"""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .memory import atomic_save, load_json

PROPOSAL_VERSION = 1

# The scope a proposal may touch. The first set is *always* a proposal and can
# never be applied autonomously; the last is routine and reversible, which the
# existing rules already let autonomous work do on its own.
SCOPE_IDENTITY = "identity"
SCOPE_BOUNDARIES = "boundaries"
SCOPE_CREDENTIALS = "credentials"
SCOPE_PERMISSIONS = "permissions"
SCOPE_SOURCE = "source_code"
SCOPE_ROUTINE = "routine_maintenance"

# Scopes that require an explicit human acknowledgement/approval before anything
# would act on them. Kept here so the boundary is one list, not scattered.
PROTECTED_SCOPES = (SCOPE_IDENTITY, SCOPE_BOUNDARIES, SCOPE_CREDENTIALS,
                    SCOPE_PERMISSIONS, SCOPE_SOURCE)

# Status vocabulary. "open" is unacknowledged; the rest are terminal-ish.
P_OPEN = "open"
P_ACKNOWLEDGED = "acknowledged"
P_APPROVED = "approved"
P_DECLINED = "declined"
PROPOSAL_STATUSES = (P_OPEN, P_ACKNOWLEDGED, P_APPROVED, P_DECLINED)
OPEN_STATUSES = (P_OPEN, P_ACKNOWLEDGED)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ImprovementProposal:
    """One bounded, human-approvable improvement suggestion."""

    id: str = ""
    title: str = ""
    scope: str = SCOPE_ROUTINE
    evidence: str = ""
    proposed_change: str = ""
    status: str = P_OPEN
    created_at: str = ""
    updated_at: str = ""
    source_task_id: Optional[str] = None
    decided_at: Optional[str] = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "prop_" + uuid.uuid4().hex[:12]
        now = _now()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @property
    def requires_approval(self) -> bool:
        return self.scope in PROTECTED_SCOPES

    def public_line(self) -> str:
        return f"[{self.id}] {self.title} (scope {self.scope}, {self.status})"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProposalStore:
    """Durable store of improvement proposals. Own file, atomic writes."""

    def __init__(self, data_dir: str = "./storage") -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "proposals.json")
        self._lock = threading.RLock()
        self._items: List[ImprovementProposal] = self._load()

    def _load(self) -> List[ImprovementProposal]:
        raw = load_json(self.path, dict, dict)
        items = raw.get("proposals") if isinstance(raw, dict) else None
        out: List[ImprovementProposal] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            data = {k: v for k, v in item.items()
                    if k in ImprovementProposal.__dataclass_fields__}
            try:
                out.append(ImprovementProposal(**data))
            except TypeError:
                continue
        return out

    def reload(self) -> None:
        """Re-read from disk, for a separate read-only inspection process."""
        with self._lock:
            self._items = self._load()

    def _persist(self) -> None:
        with self._lock:
            atomic_save(self.path, {
                "version": PROPOSAL_VERSION,
                "proposals": [p.to_dict() for p in self._items],
            })

    def add(self, proposal: ImprovementProposal) -> Optional[ImprovementProposal]:
        """Record a proposal unless an open one with the same title exists."""
        if not isinstance(proposal, ImprovementProposal):
            return None
        with self._lock:
            for existing in self._items:
                if existing.title == proposal.title and existing.status in OPEN_STATUSES:
                    return existing
            self._items.append(proposal)
            self._persist()
            return proposal

    def all(self) -> List[ImprovementProposal]:
        with self._lock:
            return list(self._items)

    def open(self) -> List[ImprovementProposal]:
        with self._lock:
            return [p for p in self._items if p.status in OPEN_STATUSES]

    def get(self, proposal_id: str) -> Optional[ImprovementProposal]:
        with self._lock:
            for item in self._items:
                if item.id == proposal_id:
                    return item
        return None

    def set_status(self, proposal_id: str, status: str, *, note: str = ""
                   ) -> Optional[ImprovementProposal]:
        if status not in PROPOSAL_STATUSES:
            raise ValueError(f"unknown proposal status: {status!r}")
        with self._lock:
            item = self.get(proposal_id)
            if item is None:
                return None
            item.status = status
            item.updated_at = _now()
            if note:
                item.note = note
            if status in (P_APPROVED, P_DECLINED):
                item.decided_at = _now()
            self._persist()
            return item

    def counts(self) -> Dict[str, int]:
        out = {s: 0 for s in PROPOSAL_STATUSES}
        with self._lock:
            for item in self._items:
                out[item.status] = out.get(item.status, 0) + 1
        return out

    def render_open(self, limit: int = 8) -> str:
        items = self.open()[:limit]
        if not items:
            return "No improvement proposals are waiting for you."
        lines = ["Improvement proposals waiting for you:"]
        for item in items:
            lines.append(f"  {item.public_line()}")
            if item.evidence:
                lines.append(f"      evidence: {item.evidence}")
            if item.proposed_change:
                lines.append(f"      would change: {item.proposed_change}")
        return "\n".join(lines)


__all__ = [
    "ImprovementProposal", "ProposalStore", "PROPOSAL_VERSION",
    "SCOPE_IDENTITY", "SCOPE_BOUNDARIES", "SCOPE_CREDENTIALS",
    "SCOPE_PERMISSIONS", "SCOPE_SOURCE", "SCOPE_ROUTINE", "PROTECTED_SCOPES",
    "P_OPEN", "P_ACKNOWLEDGED", "P_APPROVED", "P_DECLINED",
    "PROPOSAL_STATUSES", "OPEN_STATUSES",
]
