"""The supervisor: a deliberately boring process manager for the runtime.

Its job is narrow, and it owns *no* of Astra's state:

* start the runtime, exactly one at a time (single-instance lock);
* detect an unexpected exit and restart after it, with backoff so a repeated
  crash cannot become a hot restart loop;
* carry out an intentional graceful restart when the admin asks for one;
* make sure the runtime exits through its own flush path before it is stopped
  (it never kills the process and hopes the JSON survived);
* log runtime failures;
* let the admin console be closed and reopened independently of the runtime.

The rule it enforces is *state continuity, not process continuity*: the runtime
may restart, but it must never lose itself doing so. State continuity is Astra's
own job - the runtime's existing flush/checkpoint/shutdown path - and this module
only ever asks it to stop gracefully and waits for it.

This is intentionally stdlib-only and local: no sockets, no networking. The
runtime and console coordinate through the filesystem (a lock file, a control
directory, and the runtime's own state files), which is the simplest mechanism
that is safe here and avoids adding a listening port to Astra.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import paths

# Backoff between crash restarts (seconds). Grows to avoid a rapid loop.
BASE_BACKOFF = 2.0
MAX_BACKOFF = 60.0
# If the runtime survives longer than this, treat it as a stable run and reset
# the backoff - a crash after hours is not the same as a crash-on-start loop.
STABLE_RUN_SECONDS = 60.0

# The control files under runtime/control/.
RESTART_REQUEST = "restart.request"
STOP_REQUEST = "stop.request"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SingleInstanceLock:
    """A best-effort single-instance lock built from an exclusive-create file.

    Not a replacement for a real OS file lock (the process does not hold the
    handle open), but combined with the PID recorded inside it is enough to stop
    a second runtime from starting and to detect a stale lock left by a crash.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or paths.lock_path()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.acquired = False

    def _read_pid(self) -> Optional[int]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return int(handle.read().strip() or 0) or None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _alive(pid: Optional[int]) -> bool:
        if not pid:
            return False
        if os.name == "nt":
            # On Windows a signal-0 probe is unreliable; use tasklist.
            try:
                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True, text=True, timeout=5).stdout
                return str(pid) in out
            except Exception:
                return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def acquire(self) -> bool:
        pid = os.getpid()
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as handle:
                    handle.write(str(pid))
                self.acquired = True
                return True
            except FileExistsError:
                if not self._alive(self._read_pid()):
                    # Stale lock from a crashed runtime: reclaim it.
                    try:
                        os.remove(self.path)
                    except OSError:
                        return False
                    continue
                return False
        return False

    def release(self) -> None:
        if self.acquired:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.acquired = False


class RuntimeSupervisor:
    """Start, watch, and gracefully restart the Astra runtime.

    The command used to start the runtime is injected (``command``), so tests can
    exercise the whole supervisor against a trivial fake process without a
    terminal, a model, or Discord.
    """

    def __init__(self, *, command: Optional[List[str]] = None,
                 env: Optional[Dict[str, str]] = None,
                 runtime_dir: Optional[str] = None,
                 poll_seconds: float = 1.0) -> None:
        self.command = command or [sys.executable, "main.py"]
        self.env = dict(os.environ)
        if env:
            self.env.update(env)
        self.env.setdefault(paths.RUNTIME_ENV, runtime_dir or paths.runtime_root())
        self.poll_seconds = poll_seconds
        self.control = paths.control_dir(runtime_dir)
        os.makedirs(self.control, exist_ok=True)
        self.log_path = os.path.join(paths.logs_dir(runtime_dir), "supervisor.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._process: Optional[subprocess.Popen] = None
        self._stopping = False
        self._backoff = BASE_BACKOFF

    # -- logging -----------------------------------------------------------
    def log(self, message: str) -> None:
        line = f"[{_now()}] {message}"
        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    # -- control files -----------------------------------------------------
    def request_restart(self) -> None:
        self._write_control(RESTART_REQUEST)

    def request_stop(self) -> None:
        self._write_control(STOP_REQUEST)

    def _write_control(self, name: str) -> None:
        os.makedirs(self.control, exist_ok=True)
        try:
            with open(os.path.join(self.control, name), "w", encoding="utf-8") as handle:
                handle.write(_now())
        except OSError:
            pass

    def _consume_control(self) -> Optional[str]:
        for name, kind in ((STOP_REQUEST, "stop"), (RESTART_REQUEST, "restart")):
            path = os.path.join(self.control, name)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
                return kind
        return None

    def controls_pending(self) -> Optional[str]:
        """Whether a control file is present, without consuming it."""
        if os.path.exists(os.path.join(self.control, STOP_REQUEST)):
            return "stop"
        if os.path.exists(os.path.join(self.control, RESTART_REQUEST)):
            return "restart"
        return None

    # -- process control ---------------------------------------------------
    def start(self) -> subprocess.Popen:
        self.log(f"starting runtime: {' '.join(self.command)}")
        self._process = subprocess.Popen(self.command, env=self.env)
        return self._process

    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        if self._process is None:
            return None
        return self._process.wait(timeout=timeout)

    def _terminate_gracefully(self, timeout: float = 30.0) -> None:
        """Ask the runtime to stop through its own flush path, then wait.

        A graceful SIGTERM is what lets the runtime run its existing shutdown
        hooks (flush the store, drain background consolidation). Only if it does
        not exit in time is it killed - and even then, the atomic writes mean the
        last completed turn is safe.
        """
        process = self._process
        if process is None or process.poll() is not None:
            return
        self.log("requesting graceful shutdown")
        try:
            if os.name == "nt":
                process.terminate()
            else:
                process.send_signal(signal.SIGTERM)
        except Exception:
            process.terminate()
        try:
            process.wait(timeout=timeout)
            self.log("runtime exited gracefully")
        except subprocess.TimeoutExpired:
            self.log("runtime did not exit in time; killing")
            try:
                process.kill()
                process.wait(timeout=10)
            except Exception:
                pass

    def stop(self) -> None:
        """Intentional stop (used on supervisor shutdown)."""
        self._stopping = True
        self._terminate_gracefully()

    # -- main loop ---------------------------------------------------------
    def run(self, *, max_restarts: Optional[int] = None,
            max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Supervise the runtime until stopped or the restart budget is spent.

        Returns a small summary. ``max_restarts``/``max_iterations`` exist so a
        test can bound the loop deterministically.
        """
        lock = SingleInstanceLock(os.path.join(paths.runtime_root(), "supervisor.lock"))
        if not lock.acquire():
            self.log("another supervisor is already running; exiting")
            return {"started": False, "reason": "already-running"}
        restarts = 0
        iterations = 0
        try:
            while not self._stopping:
                iterations += 1
                started_at = time.monotonic()
                intentional_restart = False
                self.start()
                # Watch the process, reacting to control files and exits.
                while self.running():
                    if max_iterations is not None and iterations >= max_iterations:
                        self._terminate_gracefully()
                        return {"started": True, "reason": "iteration-limit",
                                "restarts": restarts}
                    control = self._consume_control()
                    if control == "stop":
                        self.log("stop requested via control file")
                        self._terminate_gracefully()
                        self._stopping = True
                        return {"started": True, "reason": "stopped",
                                "restarts": restarts}
                    if control == "restart":
                        self.log("graceful restart requested")
                        self._terminate_gracefully()
                        restarts += 1
                        intentional_restart = True
                        break  # outer loop restarts the runtime
                    time.sleep(self.poll_seconds)
                if intentional_restart:
                    # A deliberate restart is not a crash: no backoff, no fail
                    # count, and the previous version was already flushed.
                    self._backoff = BASE_BACKOFF
                    continue
                if self._stopping:
                    break
                if self.running():
                    continue
                code = self.wait()
                uptime = time.monotonic() - started_at
                if uptime >= STABLE_RUN_SECONDS:
                    self._backoff = BASE_BACKOFF  # stable run: reset backoff
                if max_restarts is not None and restarts >= max_restarts:
                    self.log(f"restart budget reached; runtime exited code={code}")
                    return {"started": True, "reason": "restart-budget",
                            "restarts": restarts, "last_exit": code}
                restarts += 1
                self.log(f"runtime exited code={code}; restarting in {self._backoff:.1f}s")
                time.sleep(self._backoff)
                self._backoff = min(MAX_BACKOFF, self._backoff * 2)
            return {"started": True, "reason": "stopped", "restarts": restarts}
        finally:
            lock.release()


__all__ = ["RuntimeSupervisor", "SingleInstanceLock", "RESTART_REQUEST", "STOP_REQUEST"]
