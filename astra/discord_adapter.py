"""Discord as a thin frontend adapter over the shared Astra runtime.

This module contains **transport only**. It translates Discord events into
application-level input and translates the runtime's output back into Discord
messages. It holds no memory, no personality, no scheduling, no reasoning and
no task authority: every message is handed to
:class:`astra.application.ApplicationRouter`, which is the same authority the
CLI uses. There is deliberately no second orchestrator and no direct store
mutation here.

The transport uses Discord's REST API with the standard library/``requests``
(the project already depends on ``requests``), so no Gateway/websocket library
is required. Message IDs are tracked persistently so a reconnect or restart does
not reprocess (and therefore does not re-answer or re-schedule) old messages.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests

from .memory import atomic_save, load_json

DISCORD_API = "https://discord.com/api/v10"
STATE_VERSION = 1
MAX_MESSAGE_CHARS = 1900


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_for_discord(text: Any, *, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Clamp a runtime response to Discord's message size; never alter content."""
    text = "" if text is None else str(text)
    if not text.strip():
        return "(no output)"
    if len(text) <= limit:
        return text
    return text[:limit - 1].rstrip() + "\u2026"


class DiscordTransport:
    """Thin REST client. Fails as a transport error, never as a fake success."""

    def __init__(self, token: str, *, base_url: str = DISCORD_API,
                 session: Optional[requests.Session] = None, timeout: float = 15.0) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bot {self.token}",
                "Content-Type": "application/json"}

    def _request(self, method: str, path: str, *, json_body: Optional[dict] = None,
                 params: Optional[dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(3):
            try:
                res = self._session.request(
                    method, url, headers=self._headers(), json=json_body,
                    params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                return {"ok": False, "error": f"transport: {exc}"}
            if res.status_code == 429:
                retry = 1.0
                try:
                    retry = float(res.json().get("retry_after", 1.0))
                except Exception:
                    pass
                time.sleep(min(max(retry, 0.5), 10.0) + 0.1 * attempt)
                continue
            if res.status_code in (200, 201):
                try:
                    return {"ok": True, "data": res.json()}
                except ValueError:
                    return {"ok": True, "data": {}}
            if res.status_code == 204:
                return {"ok": True, "data": {}}
            return {"ok": False, "error": f"http {res.status_code}"}
        return {"ok": False, "error": "rate-limited"}

    def fetch_messages(self, channel_id: str, *, after: Optional[str] = None,
                       limit: int = 20) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": max(1, min(limit, 100))}
        if after:
            params["after"] = after
        return self._request("GET", f"/channels/{channel_id}/messages", params=params)

    def send_message(self, channel_id: str, content: str) -> Dict[str, Any]:
        body = {"content": format_for_discord(content),
                # Never let a response ping @everyone or roles.
                "allowed_mentions": {"parse": []}}
        return self._request("POST", f"/channels/{channel_id}/messages", json_body=body)


class DiscordState:
    """Persistent, minimal delivery state: the last processed message id."""

    def __init__(self, data_dir: str = "./storage") -> None:
        self.path = os.path.join(data_dir, "discord_state.json")
        raw = load_json(self.path, dict, dict)
        self._last_id = str(raw.get("last_message_id") or "") if isinstance(raw, dict) else ""
        self._lock = threading.RLock()

    @property
    def last_message_id(self) -> str:
        with self._lock:
            return self._last_id

    def set_last_message_id(self, message_id: str) -> None:
        if not message_id:
            return
        with self._lock:
            self._last_id = str(message_id)
            atomic_save(self.path, {"version": STATE_VERSION,
                                    "last_message_id": self._last_id})


class DiscordAdapter:
    """Polls a channel and routes each human message through the shared router."""

    def __init__(self, transport: DiscordTransport, router: Any, *,
                 channel_id: str, data_dir: str = "./storage",
                 bot_user_id: Optional[str] = None,
                 poll_seconds: float = 2.0,
                 stop_event: Optional[threading.Event] = None) -> None:
        self.transport = transport
        self.router = router
        self.channel_id = str(channel_id)
        self.bot_user_id = str(bot_user_id) if bot_user_id else None
        self.state = DiscordState(data_dir=data_dir)
        self.poll_seconds = max(0.5, float(poll_seconds))
        self._stop = stop_event or threading.Event()
        self.last_error: Optional[str] = None
        self.delivered = 0
        self.failed = 0

    def stop(self) -> None:
        self._stop.set()

    def poll_once(self) -> Dict[str, Any]:
        """Fetch new human messages and answer each through the router.

        On the first poll with no stored cursor, the existing channel contents
        are treated as a baseline: they are recorded but *not* answered, so
        starting Astra does not spam answers to history. Returns a report.
        Delivery failures are counted and surfaced, never recorded as successes.
        """
        first_run = not self.state.last_message_id
        fetched = self.transport.fetch_messages(
            self.channel_id, after=self.state.last_message_id or None, limit=20)
        if not fetched.get("ok"):
            self.last_error = fetched.get("error")
            return {"ok": False, "error": fetched.get("error"), "handled": 0}
        messages = fetched.get("data") or []
        # Discord returns newest-first; process oldest-first.
        messages = list(reversed(messages))
        if first_run:
            for message in messages:
                self.state.set_last_message_id(str(message.get("id") or ""))
            return {"ok": True, "handled": 0, "error": None, "baselined": len(messages)}
        handled = 0
        for message in messages:
            message_id = str(message.get("id") or "")
            author = message.get("author") or {}
            is_bot = bool(author.get("bot")) or (
                self.bot_user_id and str(author.get("id")) == self.bot_user_id)
            if is_bot:
                # Never reply to ourselves; advance past it so we don't loop.
                self.state.set_last_message_id(message_id)
                continue
            content = message.get("content") or ""
            self.state.set_last_message_id(message_id)
            if not content.strip():
                continue
            result = self.router.handle(content, source="discord")
            reply = result.get("text") or ""
            if not reply.strip():
                continue
            sent = self.transport.send_message(self.channel_id, reply)
            if sent.get("ok"):
                self.delivered += 1
            else:
                # A transport failure is a failure, not a delivered message.
                self.failed += 1
                self.last_error = sent.get("error")
            handled += 1
        return {"ok": True, "handled": handled, "error": None}

    def run(self) -> None:
        """Poll until stopped. Bounded: one fetch, then sleep."""
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:  # transport hiccup must not kill the loop
                self.last_error = str(exc)
            self._stop.wait(self.poll_seconds)


def build_discord_adapter(router: Any, config: Dict[str, Any], *,
                          data_dir: str = "./storage") -> Optional[DiscordAdapter]:
    """Build an adapter from config, or ``None`` if Discord is not configured.

    The token is read from an environment variable named in config, never from
    the config file, so a secret is not written to disk.
    """
    discord_cfg = config.get("discord") if isinstance(config.get("discord"), dict) else {}
    if not discord_cfg.get("enabled"):
        return None
    token_env = str(discord_cfg.get("bot_token_env") or "DISCORD_BOT_TOKEN")
    token = os.environ.get(token_env)
    channel_id = str(discord_cfg.get("channel_id") or "").strip()
    if not token or not channel_id:
        return None
    transport = DiscordTransport(
        token, base_url=str(discord_cfg.get("base_url") or DISCORD_API))
    return DiscordAdapter(
        transport, router, channel_id=channel_id, data_dir=data_dir,
        bot_user_id=discord_cfg.get("bot_user_id"),
        poll_seconds=float(discord_cfg.get("poll_seconds", 2.0)))
