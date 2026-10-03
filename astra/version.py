"""Runtime version, start time, and restart bookkeeping.

A long-running Astra and a busy development checkout is exactly how a running
process silently keeps using *old code* while the files on disk say something
newer. This module makes that distinction explicit: the runtime records the code
version/commit it actually started from, when it started, whether it has
restarted recently, and the last deployment/restart time - so the admin status
screen can tell "the code was changed" apart from "the running process changed".

It owns no memory and touches no store. It reads the source tree read-only and
writes one small JSON file under the runtime ``deploy/`` directory.
"""
from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from . import paths

RECORD_NAME = "runtime.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(args: list) -> Optional[str]:
    """Run a read-only git command in the source tree; ``None`` on any failure."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", *args], cwd=paths.source_root(),
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_revision() -> Dict[str, str]:
    """The code revision the checkout currently holds (read-only)."""
    commit = _git(["rev-parse", "--short", "HEAD"]) or ""
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]) or ""
    describe = _git(["describe", "--always", "--dirty", "--tags"]) or ""
    return {"commit": commit, "branch": branch, "describe": describe}


def runtime_id() -> str:
    """A stable identifier for this runtime *process* (host + pid)."""
    return f"{socket.gethostname()}:{os.getpid()}"


class RuntimeInfo:
    """The runtime's own identity/version, persisted under the runtime tree.

    ``record_start`` is called once when the runtime comes up. It keeps a small
    history to detect a *recent restart* and exposes the last deployment time,
    which the supervisor and deployment tooling update as they go.
    """

    def __init__(self, *, deploy_dir: Optional[str] = None) -> None:
        self.dir = deploy_dir or paths.deploy_dir()
        self.path = os.path.join(self.dir, RECORD_NAME)
        os.makedirs(self.dir, exist_ok=True)

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: Dict[str, Any]) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def record_start(self, *, revision: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Record that the runtime started, from the code version now on disk."""
        data = self._load()
        rev = revision or git_revision()
        started = _now()
        history = list(data.get("start_history") or [])
        history.append({"started_at": started, "commit": rev.get("commit", ""),
                        "runtime_id": runtime_id()})
        del history[:-10]
        data.update({
            "runtime_id": runtime_id(),
            "started_at": started,
            "version": rev,
            "start_history": history,
            "state": "running",
        })
        data.setdefault("last_deploy_at", "")
        self._save(data)
        return data

    def record_deploy(self, *, revision: Optional[Dict[str, str]] = None,
                      note: str = "") -> Dict[str, Any]:
        """Record a deliberate deployment (new code promoted to live)."""
        data = self._load()
        data["last_deploy_at"] = _now()
        data["last_deploy_note"] = note
        if revision:
            data["last_deploy_version"] = revision
        self._save(data)
        return data

    def record_stop(self) -> None:
        data = self._load()
        data["state"] = "stopped"
        data["stopped_at"] = _now()
        self._save(data)

    def status(self, now: Optional[str] = None) -> Dict[str, Any]:
        """Everything the admin status screen needs to explain the version."""
        data = self._load()
        on_disk = git_revision()
        started = data.get("started_at") or ""
        restarted_recently = False
        history = data.get("start_history") or []
        if len(history) >= 2:
            restarted_recently = True
        running_version = (data.get("version") or {}).get("commit", "")
        return {
            "runtime_id": data.get("runtime_id") or runtime_id(),
            "state": data.get("state", "unknown"),
            "started_at": started,
            "last_deploy_at": data.get("last_deploy_at") or "",
            "restarted_recently": restarted_recently,
            "restart_count": max(0, len(history) - 1),
            "running_commit": running_version,
            "on_disk_commit": on_disk.get("commit", ""),
            # A clear, single boolean: is the running process on the current code?
            "code_changed_since_start": bool(
                running_version and on_disk.get("commit")
                and running_version != on_disk.get("commit")),
            "on_disk": on_disk,
            "deploy_note": data.get("last_deploy_note", ""),
        }


def status_lines(now: Optional[str] = None) -> list:
    """Runtime version info as status lines, for the admin/status screens."""
    info = RuntimeInfo().status(now)
    lines = [
        f"  runtime id: {info['runtime_id']}",
        f"  running commit: {info['running_commit'] or '(unknown)'}",
        f"  on-disk commit: {info['on_disk_commit'] or '(unknown)'}",
        f"  started: {info['started_at'] or '(unknown)'}",
        f"  last deploy: {info['last_deploy_at'] or '(none recorded)'}",
        f"  restarts: {info['restart_count']}"
        + (" (restarted recently)" if info["restarted_recently"] else ""),
    ]
    if info["code_changed_since_start"]:
        lines.append("  NOTE: code on disk differs from the running process - "
                     "deploy to pick it up.")
    return lines


__all__ = ["RuntimeInfo", "git_revision", "runtime_id", "status_lines", "RECORD_NAME"]
