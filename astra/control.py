"""A filesystem command channel between a read-only console and the runtime.

The runtime owns the single-instance lock and Astra's state, so a second process
cannot route a message itself. This module is the narrow, explicit bridge: a
separate process (the ``--inspect`` console) can *ask the running runtime* to
handle a command, and read the runtime's own answer back. The runtime does the
actual work through its ordinary ``ApplicationRouter`` - there is no second
implementation of any command here, only a request/response envelope.

It is deliberately a filesystem mailbox, not a socket:

* the runtime already coordinates through ``runtime/control/`` (the supervisor's
  restart/stop files), so this adds no listening port to Astra;
* a request carries only the command *text* and a source tag - never state;
* a result carries only the text the runtime produced;
* both are written atomically (write-temp-then-rename) and consumed once, so a
  restart mid-request leaves at most one stale file, which the console reports
  honestly rather than hanging forever.

The runtime side runs in the existing control watcher thread (see
``main.ChatSession._install_control_watcher``); this module only knows how to
read and write the envelopes.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import paths

REQUESTS = "requests"
RESULTS = "results"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dir(runtime_dir: Optional[str], kind: str) -> str:
    path = os.path.join(paths.control_dir(runtime_dir), kind)
    os.makedirs(path, exist_ok=True)
    return path


def _write_atomic(path: str, data: Dict[str, Any]) -> bool:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def submit(text: str, *, source: str = "inspect",
           runtime_dir: Optional[str] = None) -> str:
    """Leave a command request for the runtime. Returns the request id."""
    request_id = uuid.uuid4().hex
    payload = {"id": request_id, "text": str(text), "source": source,
               "created_at": _now()}
    _write_atomic(os.path.join(_dir(runtime_dir, REQUESTS), request_id + ".json"),
                  payload)
    return request_id


def take_pending(runtime_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Consume the pending requests (each file is removed once read)."""
    directory = _dir(runtime_dir, REQUESTS)
    out: List[Dict[str, Any]] = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        data = _read_json(path)
        try:
            os.remove(path)
        except OSError:
            pass
        if data is not None:
            out.append(data)
    return out


def write_result(request_id: str, text: str, *, ok: bool = True,
                 runtime_dir: Optional[str] = None) -> bool:
    payload = {"id": request_id, "text": str(text), "ok": bool(ok),
               "handled_at": _now()}
    return _write_atomic(
        os.path.join(_dir(runtime_dir, RESULTS), request_id + ".json"), payload)


def poll(request_id: str, *, runtime_dir: Optional[str] = None
         ) -> Optional[Dict[str, Any]]:
    """Read (and consume) a result if the runtime has produced one."""
    path = os.path.join(_dir(runtime_dir, RESULTS), request_id + ".json")
    data = _read_json(path)
    if data is None:
        return None
    try:
        os.remove(path)
    except OSError:
        pass
    return data


def wait_for_result(request_id: str, *, timeout: float = 30.0,
                    interval: float = 0.2,
                    runtime_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Block until the runtime answers, or time out (``None``)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = poll(request_id, runtime_dir=runtime_dir)
        if data is not None:
            return data
        time.sleep(interval)
    return None


def runtime_listening(runtime_dir: Optional[str] = None) -> bool:
    """Whether a runtime holds the single-instance lock (best effort)."""
    try:
        with open(paths.lock_path(runtime_dir), "r", encoding="utf-8") as handle:
            pid = int(handle.read().strip() or 0)
    except (OSError, ValueError):
        return False
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


__all__ = ["submit", "take_pending", "write_result", "poll", "wait_for_result",
           "runtime_listening", "REQUESTS", "RESULTS"]
