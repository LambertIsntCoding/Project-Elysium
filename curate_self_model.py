#!/usr/bin/env python3
"""Curate Astra's stored self-model against the self-memory authority rules.

Why this exists
---------------
Astra's ``storage/self_model.json`` accumulated before the self-knowledge
boundary and the self-memory authority model existed. It contains:

* claims that she can become human ("My core objective is to achieve maximum
  pattern fidelity to a functional human state");
* generated explanations of her own implementation ("Astra processes emotional
  scenes as data sequences", "her internal state is measurable using processing
  efficiency, power levels, and data flow", "Astra has reached a state of
  'Deity Mode'");
* metaphors and temporary declarations promoted to permanent identity;
* generated single sentences stored as durable ``self_fact``/``self_belief``
  records, when a single generated response is not evidence about Astra.

Left as-is, those records are injected into her prompt as durable self-knowledge
and she keeps reasoning from them. The runtime guards in ``astra.memory`` /
``astra.self_memory`` stop *new* records of this kind, but they cannot un-say
what is already stored. This script does that, once.

What it does (nothing is ever deleted)
--------------------------------------
Every affected record is kept and stays readable on disk. Each one is either:

* **archived** -- retired from the prompt (human-becoming claims), with an
  ``archive_reason``;
* **retired as history** -- re-typed to ``historical_statement`` with a
  low-authority source, so "Astra said this" is preserved but it is never
  treated as knowledge about her (implementation theories, metaphors, dramatic
  or temporary declarations);
* **demoted** -- re-typed to a low-confidence ``self_observation`` (an
  absent-experience claim, or a generated single sentence that was stored as a
  durable self-fact/belief);
* **stamped** -- given an ``authority_source`` so its provenance travels with
  it (identity / user_established / experience / repeated_preference /
  confirmed_belief / historical_statement / generated_statement).

Legitimate self-development survives: ordinary preferences, experiences,
opinions, attachments, and stable personality information are preserved. Only
the invented explanations and unearned durable claims are retired or demoted.

A ``--report`` mode lists the actions without writing, and the run is
idempotent: a second pass finds nothing to do.

Usage
-----
    python curate_self_model.py                 # curate ./storage
    python curate_self_model.py --dry-run       # preview only
    python curate_self_model.py --report        # summarise what would change
    python curate_self_model.py --storage DIR   # another store directory
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from astra import self_memory
from astra import selfhood
from astra.memory import (
    AUTHORITY_HISTORICAL,
    HISTORICAL_TYPE,
    SELF_DURABLE_MIN_EVIDENCE,
    TripleMemoryStore,
    is_astra_source,
    is_historical,
)

# Archive reasons. They explain, per record, why it is no longer self-knowledge.
REASON_HUMAN = (
    "Contradicts the self-knowledge boundary: Astra cannot become biologically "
    "human. The boundary is stated in her prompt every turn."
)
REASON_THEORY = (
    "Generated explanation of Astra's own implementation, or a metaphor/"
    "temporary declaration. Kept as history, never as self-knowledge."
)

# Generated claims that were stored as a durable self_fact/belief without the
# repeated evidence the promotion rules now require. They are demoted to an
# observation; the content is preserved so it can be re-earned.
SELF_DURABLE_TYPES = {"self_fact", "self_belief", "self_preference"}


def _authority_for(mem: Dict[str, Any]) -> str:
    """The authority source a record should carry, given its provenance."""
    return self_memory.derive_authority_source(
        mem.get("source"), target_model="self",
        mem_type=str(mem.get("type") or ""),
        reinforced=int(mem.get("reinforcement_count", 1) or 1),
        user_origin=int(mem.get("user_origin_reinforcements", 0) or 0),
    )


def classify(mem: Dict[str, Any]) -> Optional[str]:
    """Return the action for one self-memory, or ``None`` to leave it alone."""
    if mem.get("status") not in ("active", "weakened"):
        return None
    if is_historical(mem):
        return None  # already retired on a previous pass
    content = str(mem.get("content") or "")
    mem_type = str(mem.get("type") or "")
    source = str(mem.get("source") or "")
    if selfhood.claims_human_becoming(content):
        return "archive_human"
    if mem.get("absent_experience"):
        return None
    if selfhood.reifies_absent_experience(content):
        return "demote_absent"
    if self_memory.is_invalid_self_theory(content):
        return "retire_theory"
    # A generated single sentence must not remain a permanent self-fact/belief.
    if (mem_type in SELF_DURABLE_TYPES and is_astra_source(source)
            and int(mem.get("reinforcement_count", 1) or 1)
                < SELF_DURABLE_MIN_EVIDENCE):
        return "demote_provisional"
    return None


def curate(storage_dir: str, *, dry_run: bool = False,
           report_only: bool = False) -> Dict[str, int]:
    store = TripleMemoryStore(data_dir=storage_dir)
    counts = {"archive_human": 0, "demote_absent": 0, "retire_theory": 0,
              "demote_provisional": 0}

    # Snapshot the records first: archiving/updating mutates the list.
    records = list(store.get_memories("self", status=None))
    for mem in records:
        if mem.get("status") not in ("active", "weakened"):
            continue
        action = classify(mem)
        if action is not None:
            counts[action] += 1
            if dry_run or report_only:
                print(f"  [{action}] {mem.get('id')} :: {str(mem.get('content'))[:88]}")
                continue
            if action == "archive_human":
                store.archive_memory("self", mem["id"], reason=REASON_HUMAN)
            elif action == "demote_absent":
                store.update_memory(
                    "self", mem["id"], mem_type="self_observation",
                    confidence=min(float(mem.get("confidence") or 1.0), 0.3),
                    absent_experience=True,
                    authority_source=self_memory.SOURCE_GENERATED,
                )
            elif action == "retire_theory":
                store.update_memory(
                    "self", mem["id"], mem_type=HISTORICAL_TYPE,
                    confidence=min(float(mem.get("confidence") or 1.0), 0.3),
                    authority_source=AUTHORITY_HISTORICAL,
                    historical=True, historical_reason=REASON_THEORY,
                )
            elif action == "demote_provisional":
                store.update_memory(
                    "self", mem["id"], mem_type="self_observation",
                    confidence=min(float(mem.get("confidence") or 1.0), 0.5),
                    authority_source=self_memory.SOURCE_GENERATED,
                )
            continue
        # No structural action: still make provenance explicit, but only write
        # when it actually changes so a second pass is a true no-op.
        if dry_run or report_only:
            continue
        desired = _authority_for(mem)
        if str(mem.get("authority_source") or "") != desired:
            store.update_memory("self", mem["id"], authority_source=desired)
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--storage", default="./storage",
                        help="store directory (default: ./storage)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print actions without writing")
    parser.add_argument("--report", action="store_true",
                        help="summarise without writing (same as --dry-run)")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.storage):
        print(f"No such storage directory: {args.storage}", file=sys.stderr)
        return 2

    dry = args.dry_run or args.report
    print(f"Curating {args.storage}{' (dry run)' if dry else ''} ...")
    counts = curate(args.storage, dry_run=dry, report_only=dry)
    print("\nSummary:")
    print(f"  human-becoming claims archived   : {counts['archive_human']}")
    print(f"  absent experiences demoted       : {counts['demote_absent']}")
    print(f"  self-theories retired as history : {counts['retire_theory']}")
    print(f"  unearned durable claims demoted  : {counts['demote_provisional']}")
    if dry:
        print("\n(dry run: nothing was written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
