"""Pull Astra's own code from Git, then restart into it with minimal downtime.

This is the "pull command" the operator asks for: a single action that brings the
checkout up to date with its remote and promotes it to the live runtime, reusing
the *existing* graceful-restart machinery rather than inventing a second one.

Two guarantees, unchanged from :mod:`deploy`:

* **Runtime state is untouched.** Updating never copies, moves or edits the
  runtime tree; the running process is only asked to restart through its own
  flush path (``ChatSession.close`` -> store flush -> drain consolidation).
* **Nothing is destructive.** The working tree must be clean before anything is
  fetched. A failed fetch/pull leaves the checkout exactly as it was, and a
  failed restart leaves the runtime on its current process (the previous commit
  is recorded so it can be restored).

The engine is deliberately small and stdlib-only. The Git command is injected
(``git_runner``) so the whole flow - including a failed pull and a fast-forward -
is testable without a network or a repository.

Downtime is minimal because the checkout is only updated *after* the fetch is
known to be a clean fast-forward, and the runtime is asked to restart only once
the new code is already on disk: the restart window is the runtime's own flush,
typically well under a second.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from . import paths, version

# The default remote and branch a pull targets.
DEFAULT_REMOTE = "origin"
DEFAULT_BRANCH = "main"

# How long to wait for the runtime to record a fresh start after a restart.
RESTART_TIMEOUT_SECONDS = 60.0
RESTART_POLL_SECONDS = 0.5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_git_runner() -> Callable[..., Any]:
    """The real git runner: a read/write subprocess in the source tree.

    Returns a callable ``(args, timeout) -> (returncode, stdout, stderr)``.
    Kept injectable so tests never touch a repository or the network.
    """
    import subprocess

    def run(args: List[str], timeout: float = 60.0):
        result = subprocess.run(
            ["git", *args], cwd=paths.source_root(),
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr

    return run


class UpdateEngine:
    """Fetch, inspect, and (on request) fast-forward the checkout."""

    def __init__(self, *, git_runner: Optional[Callable[..., Any]] = None,
                 remote: str = DEFAULT_REMOTE, branch: str = DEFAULT_BRANCH,
                 runtime_dir: Optional[str] = None,
                 info: Optional[version.RuntimeInfo] = None) -> None:
        self.git = git_runner or _default_git_runner()
        self.remote = remote
        self.branch = branch
        self.runtime_dir = runtime_dir
        self.info = info or version.RuntimeInfo(
            deploy_dir=paths.deploy_dir(runtime_dir) if runtime_dir else None)

    # -- git helpers -----------------------------------------------------
    def _git(self, *args: str, timeout: float = 60.0) -> Dict[str, Any]:
        try:
            code, out, err = self.git(list(args), timeout)
        except Exception as exc:  # a missing git / a timeout is a failed step
            return {"ok": False, "code": None, "stdout": "", "stderr": str(exc)}
        return {"ok": code == 0, "code": code,
                "stdout": (out or "").strip(), "stderr": (err or "").strip()}

    def _rev(self, ref: str) -> str:
        got = self._git("rev-parse", "--short", ref)
        return got["stdout"] if got["ok"] else ""

    def dirty_files(self) -> List[str]:
        """Tracked files with uncommitted changes (untracked files are ignored).

        Only *tracked* modifications would be clobbered by a fast-forward, so a
        stray new file (a scratch note, a downloaded book) never blocks an
        update. An untracked file that the incoming commit introduces would still
        cause Git to refuse the merge, which the pull step reports honestly.
        """
        got = self._git("status", "--porcelain", "--untracked-files=no")
        if not got["ok"]:
            return []
        return [line for line in got["stdout"].splitlines() if line.strip()]

    def check(self, *, fetch: bool = True) -> Dict[str, Any]:
        """Report whether the checkout can be updated, without changing it."""
        dirty = self.dirty_files()
        local = self._rev("HEAD")
        remote_rev = ""
        fetched = False
        if fetch:
            fetched = self._git("fetch", self.remote, self.branch, timeout=120.0)["ok"]
            remote_rev = self._rev(f"{self.remote}/{self.branch}")
        up_to_date = bool(remote_rev and local and remote_rev == local)
        behind = 0
        if fetched and remote_rev and local:
            counted = self._git("rev-list", "--count", f"{local}..{remote_rev}")
            if counted["ok"]:
                try:
                    behind = int(counted["stdout"] or 0)
                except ValueError:
                    behind = 0
        return {
            "local_commit": local,
            "remote_commit": remote_rev,
            "remote": self.remote,
            "branch": self.branch,
            "fetched": fetched,
            "dirty": dirty,
            "dirty_ok": not dirty,
            "up_to_date": up_to_date,
            "behind": behind,
            "can_update": bool(fetched and not dirty and not up_to_date),
        }

    def pull(self, *, mode: str = "ff-only", restart: bool = True,
             wait: bool = True, timeout: float = RESTART_TIMEOUT_SECONDS
             ) -> Dict[str, Any]:
        """Bring the checkout up to date and restart the runtime into it.

        ``mode`` is ``"ff-only"`` (default; refuse if the local branch has
        diverged) or ``"reset"`` (hard-reset the local branch to the remote - a
        deliberate "make my code exactly match upstream" that discards local
        *committed* changes, but only ever after a clean-tree check).

        Returns a structured summary; on any failure the checkout and the
        running runtime are left as they were.
        """
        before = self.info.status()
        previous_commit = before.get("running_commit") or self._rev("HEAD")
        dirty = self.dirty_files()
        if dirty:
            return {"action": "update", "updated": False, "reason": "dirty-tree",
                    "dirty": dirty, "previous_commit": previous_commit}

        fetch = self._git("fetch", self.remote, self.branch, timeout=120.0)
        if not fetch["ok"]:
            return {"action": "update", "updated": False, "reason": "fetch-failed",
                    "error": fetch["stderr"] or fetch["stdout"],
                    "previous_commit": previous_commit}

        local = self._rev("HEAD")
        remote_rev = self._rev(f"{self.remote}/{self.branch}")
        if not remote_rev:
            return {"action": "update", "updated": False, "reason": "no-remote-ref",
                    "previous_commit": previous_commit}
        if remote_rev == local:
            return {"action": "update", "updated": False, "reason": "up-to-date",
                    "local_commit": local, "remote_commit": remote_rev,
                    "previous_commit": previous_commit}

        if mode == "reset":
            merged = self._git("reset", "--hard", f"{self.remote}/{self.branch}")
            merge_mode = "reset"
        else:
            merged = self._git("merge", "--ff-only", f"{self.remote}/{self.branch}")
            merge_mode = "ff-only"
        if not merged["ok"]:
            return {"action": "update", "updated": False, "reason": "merge-failed",
                    "mode": merge_mode, "error": merged["stderr"] or merged["stdout"],
                    "previous_commit": previous_commit}

        new_commit = self._rev("HEAD")
        # The new code is on disk; record the deploy, then ask the runtime to
        # restart through its own flush path so nothing is lost.
        self.info.record_deploy(revision=version.git_revision(),
                                note=f"update.py ({merge_mode})")
        requested = self._request_restart()
        verified = False
        if requested and restart and wait:
            verified = self._wait_for_start(previous_started=before.get("started_at") or "",
                                            timeout=timeout)
        return {
            "action": "update",
            "updated": True,
            "mode": merge_mode,
            "previous_commit": previous_commit,
            "new_commit": new_commit,
            "restart_requested": requested,
            "verified_start": verified,
            "note": "" if (requested and restart) else
                    "code updated; restart the runtime to run it "
                    "(start with: python main.py --supervise)",
        }

    # -- restart plumbing ------------------------------------------------
    def _request_restart(self) -> bool:
        """Leave a restart request for the supervisor (shared with deploy)."""
        from . import supervisor

        return supervisor.request_restart(self.runtime_dir)

    def _wait_for_start(self, *, previous_started: str,
                        timeout: float = RESTART_TIMEOUT_SECONDS) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.info.status()
            if status.get("started_at") and status["started_at"] != previous_started:
                return True
            time.sleep(RESTART_POLL_SECONDS)
        return False


UPDATE_USAGE = (
    "Usage: /update [apply|reset|check]\n"
    "  /update          report whether the code is behind its remote\n"
    "  /update apply    pull (fast-forward) and restart into the new code\n"
    "  /update reset    hard-reset the checkout to the remote, then restart")


def command_text(controller: Any, args: List[str]) -> str:
    """Parse ``/update`` arguments and run the right action, returning text.

    One implementation shared by the CLI session and the application router, so
    the interactive console and Discord cannot drift in what ``/update`` does.
    The apply path uses the runtime-local variant (which does not block waiting
    for its own restart) when the controller offers it - a command issued by the
    running runtime must answer *before* the control watcher restarts it.
    """
    word = (args[0].lower() if args else "").strip()

    def _apply(mode: str) -> str:
        local = getattr(controller, "apply_update_from_runtime", None)
        if callable(local):
            return local(mode=mode)
        return controller.apply_update(mode=mode)

    if word in ("", "status", "check"):
        return controller.update_report(fetch=True)
    if word in ("apply", "pull", "now", "fast-forward", "ff"):
        return _apply("ff-only")
    if word in ("reset", "hard", "force"):
        return _apply("reset")
    return UPDATE_USAGE


def update_report(*, fetch: bool = True,
                  git_runner: Optional[Callable[..., Any]] = None,
                  remote: str = DEFAULT_REMOTE, branch: str = DEFAULT_BRANCH,
                  runtime_dir: Optional[str] = None) -> str:
    """A human-readable "is there an update?" report, for ``/update`` with no arg."""
    engine = UpdateEngine(git_runner=git_runner, remote=remote, branch=branch,
                          runtime_dir=runtime_dir)
    status = engine.check(fetch=fetch)
    lines = ["CODE UPDATE", "-----------"]
    lines.append(f"  branch: {status['branch']} (remote {status['remote']})")
    lines.append(f"  local commit:  {status['local_commit'] or '(unknown)'}")
    lines.append(f"  remote commit: {status['remote_commit'] or '(unknown)'}")
    if not status["fetched"]:
        lines.append("  (could not reach the remote; showing the last known state)")
    if status["dirty"]:
        lines.append(f"  WARNING: {len(status['dirty'])} uncommitted change(s) - "
                     "commit or stash before updating:")
        for line in status["dirty"][:8]:
            lines.append(f"    {line}")
    if status["up_to_date"]:
        lines.append("  status: up to date")
    elif status["can_update"]:
        lines.append(f"  status: {status['behind']} commit(s) behind - "
                     "run /update apply to pull and restart")
    else:
        lines.append("  status: cannot update automatically (see above)")
    return "\n".join(lines)


__all__ = ["UpdateEngine", "update_report", "command_text", "UPDATE_USAGE",
           "DEFAULT_REMOTE", "DEFAULT_BRANCH"]
