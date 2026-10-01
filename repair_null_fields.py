#!/usr/bin/env python3
"""Strip no-op ``null`` placeholder fields from Astra's memory stores.

Why the nulls are there
-----------------------
``TripleMemoryStore._build_memory`` (``astra/memory.py``) seeds *every* new
memory with explicit ``None`` values for the supersession pointers::

    "supersedes": None,
    "superseded_by": None,

Those fields are only assigned a value later, and only if the memory actually
takes part in a supersession. A memory that is never superseded therefore keeps
its ``null`` placeholders on disk forever. Because the seed is copied into every
record, a handful of never-used fields account for the overwhelming majority of
``null`` values in ``storage/*.json``.

Why removing them is safe
-------------------------
Every reader in the codebase uses ``mem.get("supersedes")`` / ``.get(...)`` with
a default (``astra/memory.py``, ``astra/orchestrator.py``, ``main.py``), so a
missing key and a key set to ``null`` are indistinguishable. The nulls carry no
information -- they are pure file bloat. Removing them is behaviour-preserving.

What this script does
---------------------
For each JSON list of objects under the given directories, it drops top-level
keys whose value is ``None`` and rewrites the file atomically (keeping the
usual ``.bak``). It never deletes a record and never changes a field that holds
a value, so it honours the project rule that nothing is deleted. It is
idempotent: a second run reports zero changes.

Usage
-----
    python repair_null_fields.py                 # cleans ./storage
    python repair_null_fields.py --dry-run       # preview only
    python repair_null_fields.py --audit         # integrity check (read-only)
    python repair_null_fields.py storage test_storage

Audit mode
----------
``--audit`` never writes. It reports records whose supersession fields are
inconsistent, so you can tell a harmless placeholder from a real breakage:

* broken -- ``status == "superseded"`` with no ``superseded_by`` (the app always
  sets these together); or a pointer that is self-referential or points at an id
  that does not exist in the same store (dangling).
* warning -- ``status == "weakened"`` with no pointer (valid when this record is
  the newer, weaker side); ``contradiction_count`` with an empty ``contradicts``;
  or a pointer whose target does not link back (normal for slot expiry, which
  only updates the loser).

It exits non-zero when any *broken* finding exists, so it can gate a build.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astra.memory import atomic_save, load_json  # noqa: E402

DEFAULT_DIRS = ["storage"]


def _json_files(dirs):
    for directory in dirs:
        if os.path.isfile(directory):
            yield directory
            continue
        for root, _dirs, names in os.walk(directory):
            for name in sorted(names):
                if name.endswith(".json"):
                    yield os.path.join(root, name)


def _clean_record(record):
    """Return (new_record, removed_field_names)."""
    removed = [key for key, value in record.items() if value is None]
    if not removed:
        return record, removed
    return {k: v for k, v in record.items() if v is not None}, removed


def clean_file(path, dry_run=False):
    data = load_json(path, list, list)
    records = [item for item in data if isinstance(item, dict)]
    if not records:
        return 0, 0, {}

    changed_records = 0
    removed_total = 0
    by_field = {}
    new_data = []
    for record in records:
        cleaned, removed = _clean_record(record)
        if removed:
            changed_records += 1
            removed_total += len(removed)
            for key in removed:
                by_field[key] = by_field.get(key, 0) + 1
        new_data.append(cleaned)

    if not dry_run and changed_records:
        atomic_save(path, new_data)
    return changed_records, removed_total, by_field


def audit_file(path):
    """Read-only integrity check of supersession links. Returns (broken, warnings).

    A "broken" finding is a state the app never produces; a "warning" is
    suspicious but reachable through normal operation, so it needs a human eye.
    """
    records = [item for item in load_json(path, list, list) if isinstance(item, dict)]
    # Only memory records carry an id; other lists (e.g. history_log) are skipped.
    by_id = {r["id"]: r for r in records if r.get("id")}
    broken = []
    warnings = []
    for rec in records:
        rid = rec.get("id")
        if not rid:
            continue
        status = rec.get("status")
        pointer = rec.get("superseded_by")

        if status == "superseded" and not pointer:
            broken.append(f"{rid}: status=superseded but superseded_by is empty")
        if pointer and pointer == rid:
            broken.append(f"{rid}: superseded_by points at itself")
        if pointer and pointer not in by_id:
            broken.append(f"{rid}: superseded_by={pointer} does not exist in this store")

        if status == "weakened" and not pointer:
            warnings.append(f"{rid}: status=weakened with no superseded_by (valid if this is the newer side)")
        if rec.get("contradiction_count") and not rec.get("contradicts"):
            warnings.append(f"{rid}: contradiction_count={rec.get('contradiction_count')} but contradicts is empty")

        back = rec.get("supersedes")
        if back and back not in by_id:
            broken.append(f"{rid}: supersedes={back} does not exist in this store")
        if pointer and pointer in by_id:
            target = by_id[pointer]
            if target.get("supersedes") != rid and rid not in (target.get("contradicts") or []):
                warnings.append(f"{rid}: {pointer} does not link back (normal for slot expiry)")
    return broken, warnings


def run_audit(paths):
    total_broken = total_warn = 0
    for path in _json_files(paths):
        broken, warnings = audit_file(path)
        if not broken and not warnings:
            continue
        total_broken += len(broken)
        total_warn += len(warnings)
        print(f"{path}: {len(broken)} broken, {len(warnings)} warning(s)")
        for line in broken:
            print(f"  BROKEN  {line}")
        for line in warnings:
            print(f"  warn    {line}")
    print(f"\nAudit: {total_broken} broken, {total_warn} warning(s).")
    return 1 if total_broken else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="*", default=DEFAULT_DIRS,
                        help="Files or directories to scan (default: storage)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing")
    parser.add_argument("--audit", action="store_true",
                        help="Read-only check for inconsistent supersession links")
    args = parser.parse_args(argv)
    dirs = args.dirs or DEFAULT_DIRS

    if args.audit:
        return run_audit(dirs)

    grand_records = grand_nulls = 0
    for path in _json_files(dirs):
        records, nulls, by_field = clean_file(path, dry_run=args.dry_run)
        if records:
            grand_records += records
            grand_nulls += nulls
            fields = ", ".join(f"{k}={v}" for k, v in sorted(by_field.items()))
            print(f"{path}: {records} record(s), {nulls} null field(s) removed  [{fields}]")
    verb = "would be removed" if args.dry_run else "removed"
    print(f"\nTotal: {grand_nulls} null field(s) {verb} across {grand_records} record(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
