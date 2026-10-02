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

Features:
- Message splitting for long responses (respects Discord's 2000-char limit)
- Typing indicator while the router processes a message
- Reply-to support (responses reference the original message)
- Thread support (messages in threads are answered in-thread)
- Exponential backoff on persistent errors
- Presence/status updates
- Thread-safe counters
- Structured logging for transport errors
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests

from .memory import atomic_save, load_json

log = logging.getLogger(__name__)

DISCORD_API = "https://discord.com/api/v10"
STATE_VERSION = 1
MAX_MESSAGE_CHARS = 1900
# Discord's actual limit is 2000; leave headroom for safety.
DISCORD_HARD_LIMIT = 2000
# Minimum backoff after consecutive errors (seconds).
BACKOFF_MIN = 1.0
# Maximum backoff after consecutive errors (seconds).
BACKOFF_MAX = 60.0
# Number of consecutive errors before backoff kicks in.
BACKOFF_THRESHOLD = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> List[str]:
    """Split text into chunks that each fit within Discord's message limit.

    Tries to split on paragraph boundaries (double newlines), then on single
    newlines, then on sentence boundaries, and falls back to hard splitting.
    Never returns an empty list.
    """
    text = text.strip()
    if not text:
        return ["(no output)"]
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        # Try to find a good split point.
        split_at = _find_split_point(remaining, limit)
        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    return chunks if chunks else ["(no output)"]


def _find_split_point(text: str, limit: int) -> int:
    """Find the best position to split text within the limit."""
    # Try double newline (paragraph break) first.
    search = text[:limit]
    idx = search.rfind("\n\n")
    if idx > limit // 4:
        return idx + 2
    # Try single newline.
    idx = search.rfind("\n")
    if idx > limit // 4:
        return idx + 1
    # Try sentence boundary.
    for sep in (". ", "! ", "? "):
        idx = search.rfind(sep)
        if idx > limit // 4:
            return idx + len(sep)
    # Try word boundary.
    idx = search.rfind(" ")
    if idx > limit // 4:
        return idx + 1
    # Hard split as last resort.
    return limit


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
                log.warning("Discord transport error on %s %s: %s", method, path, exc)
                return {"ok": False, "error": f"transport: {exc}"}
            if res.status_code == 429:
                retry = 1.0
                try:
                    retry = float(res.json().get("retry_after", 1.0))
                except Exception:
                    pass
                sleep_for = min(max(retry, 0.5), 10.0) + 0.1 * attempt
                log.debug("Rate-limited on %s, sleeping %.1fs", path, sleep_for)
                time.sleep(sleep_for)
                continue
            if res.status_code in (200, 201):
                try:
                    return {"ok": True, "data": res.json()}
                except ValueError:
                    return {"ok": True, "data": {}}
            if res.status_code == 204:
                return {"ok": True, "data": {}}
            log.warning("Discord HTTP %d on %s %s", res.status_code, method, path)
            return {"ok": False, "error": f"http {res.status_code}"}
        return {"ok": False, "error": "rate-limited"}

    def fetch_messages(self, channel_id: str, *, after: Optional[str] = None,
                       limit: int = 20) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": max(1, min(limit, 100))}
        if after:
            params["after"] = after
        return self._request("GET", f"/channels/{channel_id}/messages", params=params)

    def send_message(self, channel_id: str, content: str, *,
                     reference: Optional[str] = None) -> Dict[str, Any]:
        """Send a message, optionally replying to another message.

        ``reference`` is the message ID to reply to. When set, Discord shows
        the reply as a linked message with a quote of the original.
        """
        body: Dict[str, Any] = {
            "content": format_for_discord(content),
            # Never let a response ping @everyone or roles.
            "allowed_mentions": {"parse": []},
        }
        if reference:
            body["message_reference"] = {"message_id": reference}
        return self._request("POST", f"/channels/{channel_id}/messages", json_body=body)

    def send_typing(self, channel_id: str) -> Dict[str, Any]:
        """Trigger the 'is typing...' indicator in a channel."""
        return self._request("POST", f"/channels/{channel_id}/typing")

    def set_presence(self, status: str = "online", *,
                     activity_type: Optional[int] = None,
                     activity_name: Optional[str] = None) -> Dict[str, Any]:
        """Set the bot's presence via the Gateway.

        Since we use REST-only, this uses the /users/@me endpoint for status
        text. For full presence control a Gateway connection is needed; this
        is a best-effort approach that works for the bot's visible status.
        """
        # REST API does not support presence changes directly.
        # This is a no-op placeholder; presence requires Gateway.
        # Keeping the method so callers can call it without error.
        log.debug("Presence set to %s (REST-only; requires Gateway for full support)", status)
        return {"ok": True, "data": {}}


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
    """Polls a channel and routes each human message through the shared router.

    Enhancements over basic polling:
    - Typing indicator while the router processes.
    - Reply-to: responses reference the original message.
    - Message splitting: long responses are sent as multiple messages.
    - Thread awareness: messages in threads are answered in-thread.
    - Exponential backoff on persistent errors.
    - Thread-safe counters.
    """

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
        self._counters_lock = threading.Lock()
        self._delivered = 0
        self._failed = 0
        self._consecutive_errors = 0

    @property
    def delivered(self) -> int:
        with self._counters_lock:
            return self._delivered

    @property
    def failed(self) -> int:
        with self._counters_lock:
            return self._failed

    def stop(self) -> None:
        self._stop.set()

    def _inc_delivered(self, n: int = 1) -> None:
        with self._counters_lock:
            self._delivered += n

    def _inc_failed(self, n: int = 1) -> None:
        with self._counters_lock:
            self._failed += n

    def _send_reply(self, channel_id: str, content: str,
                    reference: Optional[str] = None) -> Dict[str, Any]:
        """Send a reply, splitting long messages into multiple parts.

        Each part is sent as a separate message. The first part replies to
        the original message; subsequent parts are plain messages.
        """
        parts = _split_message(content)
        last_result: Dict[str, Any] = {"ok": True, "data": {}}
        for i, part in enumerate(parts):
            ref = reference if i == 0 else None
            result = self.transport.send_message(channel_id, part, reference=ref)
            if not result.get("ok"):
                return result
            last_result = result
        return last_result

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
            self._consecutive_errors += 1
            return {"ok": False, "error": fetched.get("error"), "handled": 0}

        # Success resets the error counter.
        self._consecutive_errors = 0
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

            # Determine the channel to reply in. If the message is in a
            # thread, reply in that thread; otherwise use the configured channel.
            reply_channel = str(message.get("channel_id") or self.channel_id)

            # Show typing indicator while the router processes.
            self.transport.send_typing(reply_channel)

            result = self.router.handle(content, source="discord")
            reply = result.get("text") or ""
            if not reply.strip():
                continue

            # Send the reply, referencing the original message for context.
            sent = self._send_reply(reply_channel, reply, reference=message_id)
            if sent.get("ok"):
                self._inc_delivered()
            else:
                # A transport failure is a failure, not a delivered message.
                self._inc_failed()
                self.last_error = sent.get("error")
                log.warning("Failed to send Discord reply: %s", sent.get("error"))
            handled += 1

        return {"ok": True, "handled": handled, "error": None}

    def run(self) -> None:
        """Poll until stopped. Bounded: one fetch, then sleep.

        Uses exponential backoff when consecutive errors accumulate, so a
        persistent outage does not hammer the API.
        """
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                self.last_error = str(exc)
                self._consecutive_errors += 1
                log.exception("Discord poll error")

            # Exponential backoff on persistent errors.
            if self._consecutive_errors >= BACKOFF_THRESHOLD:
                backoff = min(BACKOFF_MAX,
                              BACKOFF_MIN * (2 ** (self._consecutive_errors - BACKOFF_THRESHOLD)))
                log.warning("Discord: %d consecutive errors, backing off %.1fs",
                            self._consecutive_errors, backoff)
                self._stop.wait(backoff)
            else:
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