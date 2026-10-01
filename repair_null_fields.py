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
    python repair_null_fields.py storage test_storage
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="*", default=DEFAULT_DIRS,
                        help="Files or directories to scan (default: storage)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing")
    args = parser.parse_args(argv)
    dirs = args.dirs or DEFAULT_DIRS

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
