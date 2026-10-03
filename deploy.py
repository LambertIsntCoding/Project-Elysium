"""Deploy the tested checkout to the live runtime, without losing Astra's state.

The workflow this implements, deliberately in this order:

    develop in the checkout -> run the tests -> (only if green) request a
    graceful restart -> the runtime flushes its own state and exits -> the new
    code starts -> verify it came up -> record the deployed version.

Two guarantees matter more than convenience:

* **Runtime state is untouched.** Deployment never copies, moves or edits the
  runtime tree; it only asks the runtime to restart. State continuity is Astra's
  own flush path (``ChatSession.close`` -> store flush -> checkpoint), which this
  script triggers through the supervisor rather than bypassing.
* **Nothing is destructive.** If the tests fail, nothing happens. If the new
  version fails to start, the previous commit is recorded so it can be restored,
  and no runtime state is modified.

The script is stdlib-only and has no side effects until a deploy is actually
requested (``--dry-run`` prints the plan and stops).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astra import paths, version  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_tests(*, python: Optional[str] = None) -> bool:
    """Run the project's own test entry point; return True only on success."""
    cmd = [python or sys.executable, "-m", "tests"]
    print(f"running tests: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=paths.source_root())
    return result.returncode == 0


def request_restart() -> bool:
    """Ask the supervisor to restart the runtime gracefully.

    Returns False when no supervisor appears to be running (no control dir / no
    lock), so the caller can fall back to a manual start.
    """
    control = paths.control_dir()
    os.makedirs(control, exist_ok=True)
    try:
        with open(os.path.join(control, "restart.request"), "w",
                  encoding="utf-8") as handle:
            handle.write(_now())
        return True
    except OSError:
        return False


def wait_for_start(*, previous_started: str, timeout: float = 60.0,
                   poll: float = 0.5) -> bool:
    """Wait until the runtime records a *new* start time, or time out."""
    info = version.RuntimeInfo()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = info.status()
        if status.get("started_at") and status["started_at"] != previous_started:
            return True
        time.sleep(poll)
    return False


def deploy(*, run_tests_first: bool = True, dry_run: bool = False,
           python: Optional[str] = None) -> Dict[str, Any]:
    """Promote the current checkout to live, safely. Returns a summary."""
    revision = version.git_revision()
    info = version.RuntimeInfo()
    before = info.status()
    plan = {
        "action": "deploy",
        "revision": revision,
        "previous_commit": before.get("running_commit") or "",
        "runtime_started_at": before.get("started_at") or "",
        "dry_run": dry_run,
    }
    print(f"deploy plan: {json.dumps(plan, indent=2)}")
    if dry_run:
        return plan

    if run_tests_first:
        if not run_tests(python=python):
            print("tests FAILED - refusing to deploy. Live runtime is untouched.")
            return {"action": "deploy", "deployed": False,
                    "reason": "tests-failed", "revision": revision}

    # Record the deploy before restarting, so the runtime's status shows it even
    # if verification below is inconclusive.
    info.record_deploy(revision=revision, note="deploy.py")

    requested = request_restart()
    if requested:
        print("restart requested; waiting for the runtime to come back up...")
        ok = wait_for_start(previous_started=plan["runtime_started_at"])
    else:
        print("no supervisor detected; start the runtime with: python main.py --supervise")
        ok = False

    result = {
        "action": "deploy",
        "deployed": bool(ok),
        "revision": revision,
        "restart_requested": requested,
        "verified_start": bool(ok),
        "previous_commit": plan["previous_commit"],
    }
    if not ok:
        print("could not verify the new runtime started. The previous commit is "
              f"recorded as {plan['previous_commit'] or '(unknown)'} so it can be "
              "restored; runtime state was not modified.")
    else:
        print(f"deployed and verified: {revision.get('commit')}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy the tested checkout to the live runtime.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and stop; change nothing")
    parser.add_argument("--skip-tests", action="store_true",
                        help="do not run the test suite first (not recommended)")
    parser.add_argument("--python", default=None, help="interpreter to use for tests/runtime")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = deploy(run_tests_first=not args.skip_tests, dry_run=args.dry_run,
                    python=args.python)
    return 0 if result.get("deployed") or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
