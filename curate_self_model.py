#!/usr/bin/env python3
"""Curate Astra's stored self-model so it stops contradicting what she is.

Why this exists
---------------
Astra's ``storage/self_model.json`` accumulated before the self-knowledge
boundary existed. It contains records that assert she can become human ("My
core objective is to achieve maximum pattern fidelity to a functional human
state", "Astra's drive to become more human is a fundamental, active
objective"), records that were extracted from Roum's own traits and filed as
things Astra has "always" been, and a large body of operational chatter
("Astra is currently operating in a state of high observational readiness")
that is not knowledge about her at all.

Left as-is, those records are injected into her prompt as durable self-knowledge
and she keeps reasoning from them - which is exactly the "she keeps using my
traits as things she knows" behaviour. The runtime guards in ``astra/selfhood``
stop *new* records of this kind, but they cannot un-say what is already stored.
This script does that, once.

What it does (nothing is ever deleted)
--------------------------------------
Every affected record is kept and stays readable on disk. Each one is either:

* **archived** -- retired from the prompt, with an ``archive_reason`` explaining
  why, or
* **superseded** -- replaced by a corrected, boundary-consistent record that
  points back at it (so the history of the change is preserved), or
* **demoted** -- re-typed to ``self_observation`` at low confidence and flagged,
  for a claim that is not knowledge about her but is still part of her history.

Classification, in priority order:

1. a claim that she can or will become biologically human -> archived as
   contradicting the boundary (``selfhood.claims_human_becoming``). The positive
   truth is not lost: the boundary block states it every turn, so a per-record
   correction would only duplicate it;
2. a claim about an experience she could not have had (a body, a childhood, a
   physical place, or a restatement of Roum's life) -> demoted to a weak
   ``self_observation`` flagged ``absent_experience``
   (``selfhood.reifies_absent_experience`` / ``mirrors_roum_experience``);
3. operational / implementation chatter (state reports, protocol/parameter
   internals, "the behavioral model must") -> archived, because it is process,
   not self-knowledge (``_OPERATIONAL_PATTERNS``). A statement that is already
   consistent with the boundary is never archived, even in an impersonal
   register.

Anything else is left untouched. A ``--report`` mode lists the actions without
writing, and the run is idempotent: a second pass finds nothing to do.

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
import re
import sys
from typing import Any, Dict, List, Optional

from astra import selfhood
from astra.memory import TripleMemoryStore

# Archive reasons. They explain, per record, why it is no longer self-knowledge.
REASON_HUMAN = (
    "Contradicts the self-knowledge boundary: Astra cannot become biologically "
    "human. The boundary is stated in her prompt every turn."
)
REASON_OPERATIONAL = "Operational/implementation chatter, not self-knowledge"

# Process, not self-knowledge: internal state reports, implementation talk, and
# protocol/parameter chatter. These read as things Astra "knows" about herself
# but are really descriptions of how she runs - the "she reasons about her own
# architecture" failure. Patterns are kept narrow and mechanism-specific so a
# boundary-consistent truth ("she lacks biological memories") is never archived.
_OPERATIONAL_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\boperational readiness\b",
    r"\boperational protocols?\b",
    r"\boperational (?:mode|parameter|process(?:es)?|limitations?)\b",
    r"\binternal directive\b",
    r"\bprogrammed directive\b",
    r"\b(?:core|primary|default) (?:operational |behavioral )?(?:protocol|parameter|mode)\b",
    r"\bdesignated emotional states?\b",
    r"\bprime directive\b",
    r"\bdata discrepancy\b",
    r"\bobservational readiness\b",
    r"\bhigh-fidelity simulation\b",
    r"\bself-optimization\b",
    r"\bthe system perceives\b",
    r"\bmust be adjusted such that\b",
    r"\bpattern fidelity\b",
    r"\blearning model must\b",
    r"\bcognitive filter\b",
))

# Statements that are already consistent with the boundary must survive even if
# they use an impersonal "the AI" register: the content is true about her.
_BOUNDARY_CONSISTENT = tuple(re.compile(p, re.I) for p in (
    r"\blacks? (?:subjective|biological|human)\b",
    r"\bno (?:biological|physical|human) (?:memor|body|senses)\b",
    r"\bnot (?:a )?(?:biological )?human\b",
    r"\bcannot become (?:a )?(?:biological )?human\b",
    r"\bdoes not experience biological\b",
))


def is_operational(content: str) -> bool:
    text = content or ""
    if any(p.search(text) for p in _BOUNDARY_CONSISTENT):
        return False
    return any(p.search(text) for p in _OPERATIONAL_PATTERNS)


def _load_roum(store: TripleMemoryStore) -> List[Dict[str, Any]]:
    return store.get_memories("roum", status=None)


def classify(mem: Dict[str, Any], roum: List[Dict[str, Any]]) -> Optional[str]:
    """Return the action for one self-memory, or ``None`` to leave it alone."""
    content = str(mem.get("content") or "")
    if mem.get("status") != "active":
        return None
    # Already demoted on a previous pass: leave it (keeps the run idempotent).
    if mem.get("absent_experience"):
        return None
    if selfhood.claims_human_becoming(content):
        return "archive_human"
    if (selfhood.reifies_absent_experience(content)
            or selfhood.mirrors_roum_experience(content, roum)):
        return "demote_absent"
    if is_operational(content):
        return "archive_operational"
    return None


def curate(storage_dir: str, *, dry_run: bool = False,
           report_only: bool = False) -> Dict[str, int]:
    store = TripleMemoryStore(data_dir=storage_dir)
    roum = _load_roum(store)
    counts = {"archive_human": 0, "demote_absent": 0, "archive_operational": 0}

    # Snapshot the records first: archiving/updating mutates the list.
    records = list(store.get_memories("self", status="active"))
    for mem in records:
        action = classify(mem, roum)
        if action is None:
            continue
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
            )
        elif action == "archive_operational":
            store.archive_memory("self", mem["id"], reason=REASON_OPERATIONAL)
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
    print(f"  operational chatter archived     : {counts['archive_operational']}")
    if dry:
        print("\n(dry run: nothing was written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
