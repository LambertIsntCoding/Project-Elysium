"""ELYSIUM governing layer for Astra.

ELYSIUM is an application-level root command interface, not a personality,
prompt section or conversational mode of Astra.  ``is_elysium_invocation``
decides whether a line addresses the root layer, and ``ElysiumCommandHandler``
answers it directly in Python (exact capabilities such as the clock) without
ever entering Astra's generation path.

This module also implements the root-directive ("Elysium") storage concept on
top of the existing Astra storage layer.  It deliberately reuses
``memory_store``'s ``atomic_save`` / ``load_json`` / ``CommandStore`` rather
than re-implementing persistence, so the on-disk format stays the plain-JSON
format the rest of the project already reads and writes.

Design notes
------------
* The LLM endpoint is validated and pinned to loopback by default.  Prompts
  carry the user's private memories, so an unvalidated URL is an exfiltration
  primitive; ``allowed_hosts`` must be passed explicitly to opt out.
* Nothing the model returns is trusted.  A proposed command is only stored when
  its trigger actually appears in the user's own text, and every string that
  reaches a prompt is flattened and stripped of section markers so stored data
  cannot forge a new prompt section.
* Directive mutations are serialised with a process-wide lock and persisted
  through ``atomic_save``; each write keeps the same ``.bak`` convention as the
  rest of the project.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from ipaddress import ip_address
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests

from .memory import CommandStore, atomic_save, load_json

logger = logging.getLogger("astra.elysium")

# Defaults are read from the environment so the layer shares configuration with
# the rest of the project instead of hard-coding its own endpoint.
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL_NAME = os.getenv("ASTRA_MODEL", "gemma4:e4b")
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_COMMAND_CONFIDENCE_THRESHOLD = 0.7

DEFAULT_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_MAX_FIELD_LEN = 4000
_MAX_ITEM_LEN = 2000

_SESSION_LOCK = threading.Lock()
_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    """Process-wide session so TCP connections are pooled across calls."""
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                session = requests.Session()
                session.headers.update({"User-Agent": "astra-elysium/1.0"})
                _SESSION = session
    return _SESSION


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any, limit: int = _MAX_FIELD_LEN) -> str:
    """Flatten untrusted text and strip prompt-structure markers.

    Collapsing whitespace and removing ``===`` / backticks means a stored memory
    or directive cannot inject a fake ``=== SYSTEM ... ===`` section.
    """
    if value is None:
        return ""
    flat = " ".join(str(value).split())
    flat = flat.replace("===", "").replace("```", "")
    return flat[:limit]


# Public alias: other modules sanitise untrusted prompt text with this.
clean_text = _clean_text


def _validate_endpoint(url: str, allowed_hosts: Optional[set] = None) -> str:
    """Reject anything but an http(s) endpoint on an allow-listed host."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme for LLM endpoint: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError(f"LLM endpoint is missing a host: {url!r}")
    host = parsed.hostname.casefold()
    allow = DEFAULT_ALLOWED_HOSTS if allowed_hosts is None else {h.casefold() for h in allowed_hosts}
    if host not in allow:
        raise ValueError(
            f"Refusing to send prompts to untrusted host {host!r}; "
            "pass allowed_hosts to permit it explicitly."
        )
    return url


def _is_loopback(host: str) -> bool:
    try:
        return ip_address(socket.gethostbyname(host)).is_loopback
    except (socket.gaierror, ValueError):
        return False


def _json_object(raw: str) -> Dict[str, Any]:
    """Parse a JSON object, tolerating prose around the object."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass
class ElysiumDirective:
    """A single root governing directive."""

    content: str
    added_at: float = field(default_factory=time.time)
    priority: int = 100
    source: str = "user"

    @classmethod
    def from_dict(cls, data: Any) -> "ElysiumDirective":
        # Older state files stored directives as bare strings.
        if isinstance(data, str):
            return cls(content=data)
        data = data if isinstance(data, dict) else {}
        return cls(
            content=str(data.get("content", "")),
            added_at=_coerce_float(data.get("added_at"), time.time()),
            priority=_coerce_int(data.get("priority"), 100),
            source=str(data.get("source", "user")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "added_at": self.added_at,
            "priority": self.priority,
            "source": self.source,
        }


def validate_command(command: Dict[str, Any], user_input: str) -> bool:
    """True only when the proposed trigger appears in the user's own text.

    Blocks the case where injected memory content causes the model to invent a
    standing command the user never issued.
    """
    trigger = str(command.get("trigger", "")).casefold().strip()
    return bool(trigger) and trigger in user_input.casefold()


class CommandExtractor:
    """Structures a conditional command from user text via the local model.

    The model only ever *proposes*; callers must still pass the result through
    :func:`validate_command` before persisting it.
    """

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        allowed_hosts: Optional[set] = None,
    ):
        self.ollama_url = _validate_endpoint(ollama_url, allowed_hosts)
        self.model_name = model_name
        self.allowed_hosts = allowed_hosts

    def extract(
        self,
        text: str,
        confidence_threshold: float = DEFAULT_COMMAND_CONFIDENCE_THRESHOLD,
    ) -> Optional[Dict[str, Any]]:
        """Return a ``{is_command, trigger, response, once}`` dict, or ``None``."""
        if not isinstance(text, str) or not text.strip():
            return None

        prompt = (
            "Analyze this text. If the user is giving a direct conditional "
            'command (e.g., "Next time I say X, do Y"), extract the exact '
            "trigger and response. If it is not a direct command, return an "
            "empty JSON object {}. Return a confidence score 0-1 indicating "
            "how certain you are this is a valid command.\n\n"
            f'Text: "{_clean_text(text)}"\n\n'
            "Output STRICT JSON ONLY:\n"
            "{\n"
            '    "is_command": true/false,\n'
            '    "confidence": 0.0,\n'
            '    "trigger": "exact word or phrase to listen for",\n'
            '    "response": "exact response required",\n'
            '    "once": true\n'
            "}"
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        try:
            data = _json_object(self._post(payload, timeout=10))
        except Exception as exc:  # unreachable model is not fatal
            logger.debug("Command extraction failed: %s", exc)
            return None

        if not data.get("is_command", False):
            return None
        if _coerce_float(data.get("confidence"), 0.0) < confidence_threshold:
            return None

        trigger = str(data.get("trigger") or "").strip()
        response = str(data.get("response") or "").strip()
        if not trigger or not response:
            return None
        return {
            "is_command": True,
            "trigger": trigger[:_MAX_ITEM_LEN],
            "response": response[:_MAX_ITEM_LEN],
            "once": bool(data.get("once", True)),
        }

    def _post(self, payload: Dict[str, Any], timeout: float) -> str:
        _validate_endpoint(self.ollama_url, self.allowed_hosts)
        response = _session().post(self.ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()
        return str(response.json().get("response", "{}"))


class ElysiumOrchestrator:
    """Reads and mutates the root-directive state file.

    State schema (``config_dir/elysium_state.json``)::

        {"version": 1.1, "active_directives": [<directive dict>, ...],
         "system_flags": {}}

    Legacy files whose ``active_directives`` is a list of bare strings are
    migrated in place on load.
    """

    def __init__(
        self,
        config_dir: str = "./config",
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        allowed_hosts: Optional[set] = None,
    ):
        self.config_dir = config_dir
        self.ollama_url = _validate_endpoint(ollama_url, allowed_hosts)
        self.model_name = model_name
        self.allowed_hosts = allowed_hosts
        self.state_file = os.path.join(config_dir, "elysium_state.json")
        self._state_lock = threading.RLock()
        self._state_last_modified = 0.0
        self.state = self._load_state()

    # -- state -------------------------------------------------------------
    @staticmethod
    def _default_state() -> Dict[str, Any]:
        return {"version": 1.1, "active_directives": [], "system_flags": {}}

    def _load_state(self) -> Dict[str, Any]:
        """Load and migrate the state file; never raises on bad input."""
        os.makedirs(self.config_dir, exist_ok=True)
        raw = load_json(self.state_file, dict, self._default_state)
        if not isinstance(raw, dict):
            raw = self._default_state()
        raw.setdefault("version", 1.1)
        raw.setdefault("system_flags", {})

        directives = raw.get("active_directives")
        if not isinstance(directives, list):
            directives = []
        raw["active_directives"] = [ElysiumDirective.from_dict(d).to_dict() for d in directives]

        try:
            self._state_last_modified = os.path.getmtime(self.state_file)
        except OSError:
            self._state_last_modified = 0.0
        return raw

    def _refresh_if_stale(self) -> None:
        try:
            mtime = os.path.getmtime(self.state_file)
        except OSError:
            return
        if mtime > self._state_last_modified:
            self.state = self._load_state()

    # -- reading -----------------------------------------------------------
    def get_active_directives(self) -> List[ElysiumDirective]:
        """Active directives, highest priority first."""
        with self._state_lock:
            self._refresh_if_stale()
            directives = [
                ElysiumDirective.from_dict(d)
                for d in self.state.get("active_directives", [])
                if _clean_text(d.get("content") if isinstance(d, dict) else d)
            ]
        return sorted(directives, key=lambda d: -d.priority)

    # -- mutating ----------------------------------------------------------
    def add_directive(self, directive: ElysiumDirective) -> None:
        content = _clean_text(directive.content, _MAX_ITEM_LEN)
        if not content:
            raise ValueError("directive content must not be empty")
        with self._state_lock:
            self._refresh_if_stale()
            self.state.setdefault("active_directives", []).append(
                ElysiumDirective(
                    content=content,
                    added_at=directive.added_at,
                    priority=directive.priority,
                    source=directive.source,
                ).to_dict()
            )
            self._persist()

    def remove_directive(self, content: str) -> int:
        """Remove every directive whose content matches; return the count."""
        target = str(content).strip()
        with self._state_lock:
            self._refresh_if_stale()
            existing = self.state.get("active_directives", [])
            kept = [d for d in existing if str(d.get("content")) != target]
            removed = len(existing) - len(kept)
            if removed:
                self.state["active_directives"] = kept
                self._persist()
        return removed

    def _persist(self) -> None:
        atomic_save(self.state_file, self.state)
        try:
            self._state_last_modified = os.path.getmtime(self.state_file)
        except OSError:
            self._state_last_modified = time.time()

    # -- administrative command -------------------------------------------
    def process_command(self, user_input: str) -> str:
        """Parse an administrative command, update directives, confirm.

        Returns the model's confirmation, or an ``[Elysium Error: ...]`` string
        (matching the original contract) when the endpoint is unreachable.
        """
        directives = [d.content for d in self.get_active_directives()]
        prompt = (
            "You are ELYSIUM, the root governing architecture of this system. "
            "You sit above the conversational AI (Astra). Your job is to parse "
            "root-level administrative requests, update system directives, and "
            "confirm execution. Do NOT roleplay as a person. Do NOT simulate "
            "emotions. Respond with cold, administrative precision.\n"
            f"Current Active Directives: {json.dumps(directives)}\n"
            f"User Command: {_clean_text(user_input)}\n\n"
            "Output STRICT JSON ONLY:\n"
            "{\n"
            '    "system_response": "Brief confirmation to the user",\n'
            '    "new_directive_to_add": "Any new rule Astra must follow, or null",\n'
            '    "directive_to_remove": "Any existing rule to remove, or null"\n'
            "}"
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        try:
            _validate_endpoint(self.ollama_url, self.allowed_hosts)
            response = _session().post(self.ollama_url, json=payload, timeout=DEFAULT_REQUEST_TIMEOUT)
            response.raise_for_status()
            data = _json_object(str(response.json().get("response", "{}")))

            to_add = _clean_text(data.get("new_directive_to_add"), _MAX_ITEM_LEN)
            if to_add:
                self.add_directive(ElysiumDirective(content=to_add))
            to_remove = _clean_text(data.get("directive_to_remove"), _MAX_ITEM_LEN)
            if to_remove:
                self.remove_directive(to_remove)
            return str(data.get("system_response") or "Elysium state updated.")
        except Exception as exc:
            logger.error("Elysium processing error: %s", exc)
            return f"[Elysium Error: {exc}]"


def is_elysium_invocation(text: Any) -> bool:
    """True when ``text`` addresses the ELYSIUM root layer rather than Astra.

    The match is on the whole leading word (plus optional ``:``/``>``
    separator), so an ordinary sentence that merely contains "elysium"
    ("I read about Elysium") is *not* an invocation.
    """
    if not isinstance(text, str):
        return False
    return re.match(r"^\s*elysium\b\s*[:>]?", text, re.IGNORECASE) is not None


def _strip_invocation(text: str) -> str:
    """Return the command text following the leading ``elysium`` keyword."""
    return re.sub(r"^\s*elysium\b\s*[:>]?\s*", "", text, count=1, flags=re.IGNORECASE)


class ElysiumCommandHandler:
    """Application-level command interface for the ELYSIUM root layer.

    Elysium is a governing/system layer, not a persona. It answers exact
    requests deterministically in Python and never routes them through the
    conversational model, so it has no access to Astra's tone, memories or
    roleplay behaviour.

    Capabilities are only the ones actually implemented here:

    * ``clock``  -- current time/date from the system clock (``datetime``).

    Administrative directive changes (add/remove) are delegated to the
    supplied :class:`ElysiumOrchestrator`, whose output is used verbatim and
    is not passed through Astra.
    """

    HELP_TEXT = "commands: time | date | help"

    def __init__(self, orchestrator: Optional[Any] = None, *, clock: Optional[Callable[[], datetime]] = None):
        self.orchestrator = orchestrator
        self._clock = clock or (lambda: datetime.now().astimezone())

    # -- exact capabilities -------------------------------------------------
    def _clock_text(self, command: str) -> Optional[str]:
        words = set(re.findall(r"[a-z]+", command.casefold()))
        wants_date = bool(words & {"date", "day", "today"})
        wants_time = bool(words & {"time", "clock", "hour"})
        if not (wants_date or wants_time):
            return None
        now = self._clock()
        if wants_date and wants_time:
            return now.strftime("%Y-%m-%d %H:%M:%S %Z")
        if wants_date:
            return now.strftime("%A, %B %d, %Y")
        # Portable 12-hour form (%-I is glibc-only; Windows rejects it).
        return now.strftime("%I:%M %p %Z").lstrip("0")

    def _directive_command(self, raw: str, command: str) -> Optional[str]:
        if self.orchestrator is None:
            return None
        cmd = command.casefold()
        if cmd.startswith(("add ", "remove ", "delete ", "drop ", "set ")):
            return self.orchestrator.process_command(raw)
        return None

    # -- public API ---------------------------------------------------------
    def handle(self, user_input: str) -> str:
        """Handle one ELYSIUM invocation and return the display line.

        Every return value is prefixed with ``Elysium >`` so a caller can tell
        a root-layer answer apart from an Astra reply.
        """
        command = _strip_invocation(user_input).strip()
        if not command:
            return f"Elysium > {self.HELP_TEXT}"

        exact = self._clock_text(command)
        if exact is not None:
            return f"Elysium > {exact}"

        if command.casefold() in {"help", "?"}:
            return f"Elysium > {self.HELP_TEXT}"

        directive = self._directive_command(user_input, command)
        if directive is not None:
            return f"Elysium > {directive}"

        return (
            "Elysium > Unknown command. "
            "Available: time, date, help (directive changes: add/remove <rule>)."
        )


class ElysiumCommandRecorder:
    """Validated bridge between :class:`CommandExtractor` and ``CommandStore``.

    Extraction is optional and off by default because it costs an LLM round
    trip per turn; the recorder itself is pure and cheap to construct.
    """

    def __init__(
        self,
        cmd_store: CommandStore,
        extractor: Optional[CommandExtractor] = None,
    ):
        self.cmd_store = cmd_store
        self.extractor = extractor

    def capture(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Extract, validate against ``user_input``, and store a command."""
        if self.extractor is None:
            return None
        candidate = self.extractor.extract(user_input)
        if not candidate or not validate_command(candidate, user_input):
            return None
        self.cmd_store.add_command(
            raw_text=user_input,
            trigger=candidate["trigger"],
            response=candidate["response"],
            once=candidate["once"],
        )
        return candidate
