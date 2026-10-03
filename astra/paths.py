"""Where Astra's code lives vs. where her accumulated life lives.

The rule this module encodes is the one the brief asks for:

    Code changes through Git. Astra's accumulated state belongs to the runtime
    environment.

The repository checkout is *code* - it is freely edited, pulled and tested. All
of the state that accumulates while Astra runs (memories, affect, relationships,
reading position, questions, presence, background state and admin snapshots) is
*runtime state* and lives under a single runtime root that is deliberately not
tracked by Git.

Resolution order for the runtime root:

1. ``ASTRA_RUNTIME_DIR`` if set (the deployment/supervisor uses this to point a
   live runtime at a stable, out-of-tree directory);
2. otherwise ``<repo>/runtime``.

This is a *seeding* migration, never a destructive move. On first run, if the
runtime storage directory is missing but the old in-repo ``storage/`` exists, the
runtime store is seeded from it by copying files that are not already present.
The originals are left untouched, so nothing is lost and the change is
reversible by simply removing the runtime directory.

``storage/library/`` (the books and their extracted text) is treated as a
fixture: it is version-controlled source material, not accumulated state, so a
seed copy preserves it and it is never deleted.
"""
from __future__ import annotations

import os
import shutil
from typing import List, Optional

# Files/dirs under storage/ that are version-controlled fixtures rather than
# accumulated runtime state. These are copied on seeding but treated as source.
LIBRARY_SUBDIR = "library"

# The environment variable the supervisor/deployment uses to keep runtime state
# outside the checkout entirely.
RUNTIME_ENV = "ASTRA_RUNTIME_DIR"


def source_root() -> str:
    """The directory holding the checked-out code (``main.py`` lives here)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runtime_root(explicit: Optional[str] = None) -> str:
    """The directory holding Astra's accumulated state.

    Deliberately separate from :func:`source_root`: a running process must never
    have its live state living inside the tree an editor/Git is modifying.
    """
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get(RUNTIME_ENV)
    if env:
        return os.path.abspath(env)
    return os.path.join(source_root(), "runtime")


def storage_dir(explicit: Optional[str] = None) -> str:
    """The runtime storage directory (memories, affect, relationships, ...)."""
    if explicit:
        return os.path.abspath(explicit)
    return os.path.join(runtime_root(), "storage")


def logs_dir(explicit: Optional[str] = None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    return os.path.join(runtime_root(), "logs")


def snapshots_dir(explicit: Optional[str] = None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    return os.path.join(runtime_root(), "snapshots")


def deploy_dir(explicit: Optional[str] = None) -> str:
    """Where the deployment system records the currently-running version."""
    if explicit:
        return os.path.abspath(explicit)
    return os.path.join(runtime_root(), "deploy")


def control_dir(explicit: Optional[str] = None) -> str:
    """Where the admin/supervisor leave control requests for the runtime."""
    if explicit:
        return os.path.abspath(explicit)
    return os.path.join(runtime_root(), "control")


def lock_path(explicit: Optional[str] = None) -> str:
    """The single-instance lock file for the runtime process."""
    if explicit:
        return os.path.abspath(explicit)
    return os.path.join(runtime_root(), "runtime.lock")


def ensure_runtime_dirs() -> str:
    """Create the runtime tree if needed and return the runtime root."""
    root = runtime_root()
    for path in (storage_dir(), logs_dir(), snapshots_dir(), deploy_dir(),
                 control_dir()):
        os.makedirs(path, exist_ok=True)
    return root


def legacy_storage_dir() -> str:
    """The old in-repo storage location, kept only as a seed source."""
    return os.path.join(source_root(), "storage")


def is_tracked_fixture(rel_path: str) -> bool:
    """True for files under storage/ that are source fixtures, not runtime state.

    Only ``library/`` (the books and their cached text) qualifies today; the
    JSON/JSONL state files are all accumulated runtime state.
    """
    rel = str(rel_path or "").replace("\\", "/").lstrip("./")
    return rel == LIBRARY_SUBDIR or rel.startswith(LIBRARY_SUBDIR + "/")


def seed_runtime_storage(*, target: Optional[str] = None,
                         source: Optional[str] = None) -> List[str]:
    """Seed the runtime storage from the legacy in-repo storage, non-destructively.

    Copies only files that do not already exist at the target, so running this
    again never overwrites accumulated state. The source is left untouched.
    Returns the list of relative paths actually copied (empty when there was
    nothing to do).
    """
    target = storage_dir(target)
    source = source or legacy_storage_dir()
    if os.path.abspath(target) == os.path.abspath(source):
        return []
    if not os.path.isdir(source):
        return []
    os.makedirs(target, exist_ok=True)
    copied: List[str] = []
    for root, _dirs, files in os.walk(source):
        rel_root = os.path.relpath(root, source)
        for name in files:
            rel = name if rel_root == "." else os.path.join(rel_root, name)
            dest = os.path.join(target, rel)
            if os.path.exists(dest):
                continue
            os.makedirs(os.path.dirname(dest) or target, exist_ok=True)
            try:
                shutil.copy2(os.path.join(root, name), dest)
                copied.append(rel)
            except OSError:
                continue
    return copied


def resolve_storage_dir(explicit: Optional[str] = None, *, seed: bool = True) -> str:
    """The storage dir a runtime should use, seeded once from the legacy location.

    ``explicit`` (a caller-provided path, e.g. from a test) is honoured as-is and
    never seeded. Otherwise the runtime storage is used and, on first setup only
    (when it has no files yet), seeded from the in-repo ``storage/`` so Astra's
    existing state carries over. A populated runtime storage is never re-seeded,
    so normal starts do not walk the legacy tree.
    """
    if explicit:
        return os.path.abspath(explicit)
    ensure_runtime_dirs()
    target = storage_dir()
    if seed and not _has_files(target):
        seed_runtime_storage(target=target)
    return target


def _has_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if files:
            return True
    return False


__all__ = [
    "RUNTIME_ENV",
    "LIBRARY_SUBDIR",
    "source_root",
    "runtime_root",
    "storage_dir",
    "logs_dir",
    "snapshots_dir",
    "deploy_dir",
    "ensure_runtime_dirs",
    "legacy_storage_dir",
    "is_tracked_fixture",
    "seed_runtime_storage",
    "resolve_storage_dir",
]
