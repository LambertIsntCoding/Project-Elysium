"""Discord as a thin frontend adapter over the shared Astra runtime.

This module contains **transport plus delivery bookkeeping only**. It translates
Discord events into application-level input and translates the runtime's output
back into Discord messages. It holds no memory, no personality, no scheduling
and no task authority: every inbound message is handed to
:class:`astra.application.ApplicationRouter`, the same authority the CLI uses,
and every outbound proactive message is a decision another layer already made
(:mod:`astra.notifications`, :mod:`astra.autonomy`). There is deliberately no
second orchestrator and no direct store mutation here.

The transport uses Discord's REST API with ``requests`` (the project already
depends on it), so no Gateway/websocket library is required.

Inbound:

- messages are **batched** with a short debounce, so a person who sends a
  thought in several lines has it read as one turn rather than answered line by
  line;
- a reply is **not** attached by default; it is only attached (and the author
  pinged) when the response explicitly asks for emphasis
  (:mod:`astra.reply_policy`), which is Astra's own convention;
- message IDs are tracked persistently so a restart does not reprocess history;
- only configured, authorised authors are answered.

Outbound (proactive):

- :meth:`DiscordAdapter.send_proactive` is the single send path. It records the
  *decision*, the *delivery* and any *failure* separately, suppresses duplicates
  by a stable reference id, bounds retries, and never reports a failed send as a
  message the user received;
- message destinations are restricted to the configured channel and authorised
  recipients - there is no way to send elsewhere;
- disconnections and permission problems are surfaced through
  :meth:`diagnostics` and a throttled ``last_error``, never by spamming the
  recipient.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests

from . import reply_policy
from .memory import atomic_save, load_json

log = logging.getLogger(__name__)

DISCORD_API = "https://discord.com/api/v10"
STATE_VERSION = 1
DELIVERY_VERSION = 1
MAX_MESSAGE_CHARS = 1900
# Discord's actual limit is 2000; leave headroom for safety.
DISCORD_HARD_LIMIT = 2000
# Minimum backoff after consecutive errors (seconds).
BACKOFF_MIN = 1.0
# Maximum backoff after consecutive errors (seconds).
BACKOFF_MAX = 60.0
# Number of consecutive errors before backoff kicks in.
BACKOFF_THRESHOLD = 3
# Outbound proactive retry bound (attempts, not infinite).
PROACTIVE_MAX_ATTEMPTS = 3
# Bounded delivery-event log.
DELIVERY_LOG_LIMIT = 200


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
                return {"ok": False, "error": f"transport: {exc}", "status": None}
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
                    return {"ok": True, "data": res.json(), "status": res.status_code}
                except ValueError:
                    return {"ok": True, "data": {}, "status": res.status_code}
            if res.status_code == 204:
                return {"ok": True, "data": {}, "status": res.status_code}
            log.warning("Discord HTTP %d on %s %s", res.status_code, method, path)
            return {"ok": False, "error": f"http {res.status_code}",
                    "status": res.status_code}
        return {"ok": False, "error": "rate-limited", "status": 429}

    def fetch_messages(self, channel_id: str, *, after: Optional[str] = None,
                       limit: int = 20) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": max(1, min(limit, 100))}
        if after:
            params["after"] = after
        return self._request("GET", f"/channels/{channel_id}/messages", params=params)

    def send_message(self, channel_id: str, content: str, *,
                     reference: Optional[str] = None,
                     mention_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Send a message, optionally replying to and pinging a user.

        ``reference`` is the message ID to reply to. When set, Discord shows the
        reply as a linked message with a quote of the original. ``mention_user_id``
        adds that single user to ``allowed_mentions`` (never ``@everyone``/roles)
        and the caller is responsible for including ``<@id>`` in the content.
        """
        allowed: Dict[str, Any] = {"parse": []}
        if mention_user_id:
            allowed["users"] = [str(mention_user_id)]
        body: Dict[str, Any] = {
            "content": format_for_discord(content),
            "allowed_mentions": allowed,
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
        """Best-effort presence. REST cannot change gateway presence; this is a
        no-op placeholder so callers can call it without error."""
        log.debug("Presence set to %s (REST-only; requires Gateway for full support)", status)
        return {"ok": True, "data": {}}


class DiscordState:
    """Persistent, minimal inbound delivery state: the last processed message id."""

    def __init__(self, data_dir: str = "./storage") -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "discord_state.json")
        self._lock = threading.RLock()
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        raw = load_json(self.path, dict, dict)
        return {"version": STATE_VERSION,
                "last_message_id": (raw or {}).get("last_message_id")}

    @property
    def last_message_id(self) -> Optional[str]:
        with self._lock:
            value = self._state.get("last_message_id")
            return str(value) if value else None

    def set_last_message_id(self, message_id: Any) -> None:
        if not message_id:
            return
        with self._lock:
            self._state["last_message_id"] = str(message_id)
            atomic_save(self.path, {"version": STATE_VERSION,
                                    "last_message_id": self._state["last_message_id"]})


class DiscordDeliveryStore:
    """Persistent outbound-delivery bookkeeping: the decision, the result, retries.

    Kept separate from :class:`DiscordState` (the inbound cursor) because a
    successful *delivery* is a different fact from the *decision to send*. The
    store is bounded and atomic, so a restart neither re-sends a delivered
    message nor loses the record of a failure.
    """

    def __init__(self, data_dir: str = "./storage") -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "discord_deliveries.json")
        self._lock = threading.RLock()
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        raw = load_json(self.path, dict, dict)
        raw = raw if isinstance(raw, dict) else {}
        return {
            "version": DELIVERY_VERSION,
            "sends": {str(k): v for k, v in (raw.get("sends") or {}).items()
                      if isinstance(v, dict)},
            "events": [e for e in (raw.get("events") or []) if isinstance(e, dict)][-DELIVERY_LOG_LIMIT:],
        }

    def _persist(self) -> None:
        with self._lock:
            atomic_save(self.path, self._state)

    def flush(self) -> None:
        """Force a save (called by the shared store flush path)."""
        self._persist()

    def get(self, ref_id: Any) -> Dict[str, Any]:
        if not ref_id:
            return {}
        with self._lock:
            return dict(self._state.get("sends", {}).get(str(ref_id)) or {})

    def was_delivered(self, ref_id: Any) -> bool:
        return self.get(ref_id).get("status") == "delivered"

    def attempts(self, ref_id: Any) -> int:
        try:
            return int(self.get(ref_id).get("attempts") or 0)
        except (TypeError, ValueError):
            return 0

    def note_decision(self, ref_id: Any, *, summary: str = "") -> None:
        """Record that delivering this outcome was *considered* (not sent)."""
        if not ref_id:
            return
        with self._lock:
            sends = self._state.setdefault("sends", {})
            entry = sends.setdefault(str(ref_id), {"attempts": 0})
            entry["decided_at"] = _now()
            entry["summary"] = str(summary or entry.get("summary") or "")[:200]
            self._append_event("decided", ref_id)
            self._persist()

    def note_delivered(self, ref_id: Any, *, message_id: Optional[str] = None) -> None:
        if not ref_id:
            return
        with self._lock:
            sends = self._state.setdefault("sends", {})
            entry = sends.setdefault(str(ref_id), {"attempts": 0})
            entry["status"] = "delivered"
            entry["delivered_at"] = _now()
            if message_id:
                entry["message_id"] = str(message_id)
            self._append_event("delivered", ref_id)
            self._persist()

    def note_failure(self, ref_id: Any, error: str, *, attempts: int) -> None:
        if not ref_id:
            return
        with self._lock:
            sends = self._state.setdefault("sends", {})
            entry = sends.setdefault(str(ref_id), {"attempts": 0})
            entry["status"] = "failed"
            entry["last_error"] = str(error)[:200]
            entry["attempts"] = int(attempts)
            entry["failed_at"] = _now()
            self._append_event("failed", ref_id)
            self._persist()

    def _append_event(self, kind: str, ref_id: Any) -> None:
        events = self._state.setdefault("events", [])
        events.append({"at": _now(), "kind": kind, "ref_id": str(ref_id)})
        self._state["events"] = events[-DELIVERY_LOG_LIMIT:]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            sends = self._state.get("sends", {})
            delivered = sum(1 for v in sends.values() if v.get("status") == "delivered")
            failed = sum(1 for v in sends.values() if v.get("status") == "failed")
            last_delivered = max(
                (v.get("delivered_at") for v in sends.values() if v.get("delivered_at")),
                default=None)
        return {"tracked": len(sends), "delivered": delivered, "failed": failed,
                "last_delivered_at": last_delivered}


class DiscordAdapter:
    """Polls a channel, routes inbound messages, and delivers proactive output.

    Inbound enhancements: batching, typing indicator, reply-only-on-emphasis,
    thread awareness, exponential backoff, thread-safe counters.
    Outbound: a single bounded, deduplicated, recorded send path.
    """

    def __init__(self, transport: DiscordTransport, router: Any, *,
                 channel_id: str, data_dir: str = "./storage",
                 bot_user_id: Optional[str] = None,
                 allowed_author_ids: Optional[List[str]] = None,
                 poll_seconds: float = 2.0,
                 batch_seconds: float = 0.0,
                 max_batch: int = 8,
                 record_inbound_delivery: bool = True,
                 stop_event: Optional[threading.Event] = None,
                 now_fn: Optional[Callable[[], datetime]] = None) -> None:
        self.transport = transport
        self.router = router
        self.channel_id = str(channel_id)
        self.bot_user_id = str(bot_user_id) if bot_user_id else None
        self.allowed_author_ids = {str(a) for a in (allowed_author_ids or [])}
        self.state = DiscordState(data_dir=data_dir)
        self.deliveries = DiscordDeliveryStore(data_dir=data_dir)
        self.poll_seconds = max(0.5, float(poll_seconds))
        self.batch_seconds = max(0.0, float(batch_seconds))
        self.max_batch = max(1, int(max_batch))
        self.record_inbound_delivery = bool(record_inbound_delivery)
        self._stop = stop_event or threading.Event()
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self.last_error: Optional[str] = None
        self.last_fetch_ok_at: Optional[str] = None
        self.last_proactive_at: Optional[str] = None
        self._counters_lock = threading.Lock()
        self._delivered = 0
        self._failed = 0
        self._consecutive_errors = 0
        # Inbound batch buffer (scratch, session-only; not persisted).
        self._batch: List[Dict[str, Any]] = []
        self._batch_since: Optional[datetime] = None
        self._send_lock = threading.Lock()

    # -- counters --------------------------------------------------------
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

    def _author_allowed(self, author: Dict[str, Any]) -> bool:
        """Whether an author may be answered. Empty allowlist = anyone (non-bot)."""
        if not self.allowed_author_ids:
            return True
        return str(author.get("id")) in self.allowed_author_ids

    # -- sending ---------------------------------------------------------
    def _send_reply(self, channel_id: str, content: str, *,
                    reference: Optional[str] = None,
                    mention_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Send a reply, splitting long messages into multiple parts.

        Each part is sent as a separate message. The first part carries the
        reference/mention; subsequent parts are plain messages.
        """
        parts = _split_message(content)
        last_result: Dict[str, Any] = {"ok": True, "data": {}}
        for i, part in enumerate(parts):
            kwargs: Dict[str, Any] = {}
            if reference is not None and i == 0:
                kwargs["reference"] = reference
            if mention_user_id and i == 0:
                kwargs["mention_user_id"] = mention_user_id
            result = self.transport.send_message(channel_id, part, **kwargs)
            if not result.get("ok"):
                return result
            last_result = result
        return last_result

    def send_proactive(self, content: str, *, ref_id: Optional[str] = None,
                       mention_user_id: Optional[str] = None,
                       importance: float = 0.0) -> Dict[str, Any]:
        """The single outbound path for an unprompted message.

        Records the decision, suppresses a duplicate, bounds retries, and records
        a delivery or a failure separately. Returns a structured result:
        ``{"ok": bool, "sent": bool, "suppressed": str|None, "message_id": str|None,
        "error": str|None}``. A failure is never reported as sent.
        """
        if not str(content or "").strip():
            return {"ok": False, "sent": False, "error": "empty",
                    "suppressed": "empty", "message_id": None}
        with self._send_lock:
            if ref_id and self.deliveries.was_delivered(ref_id):
                return {"ok": True, "sent": False, "suppressed": "duplicate",
                        "message_id": self.deliveries.get(ref_id).get("message_id"),
                        "error": None}
            attempts = self.deliveries.attempts(ref_id) if ref_id else 0
            if ref_id and attempts >= PROACTIVE_MAX_ATTEMPTS:
                return {"ok": False, "sent": False, "error": "retry limit reached",
                        "suppressed": "retry-limit", "message_id": None}
            if ref_id:
                self.deliveries.note_decision(ref_id, summary=content)
            result = self.transport.send_message(
                self.channel_id, content, mention_user_id=mention_user_id)
            if result.get("ok"):
                message_id = None
                data = result.get("data") if isinstance(result.get("data"), dict) else {}
                message_id = data.get("id")
                if ref_id:
                    self.deliveries.note_delivered(ref_id, message_id=message_id)
                self._inc_delivered()
                self.last_proactive_at = _now()
                self._consecutive_errors = 0
                return {"ok": True, "sent": True, "suppressed": None,
                        "message_id": str(message_id) if message_id else None,
                        "error": None}
            # Failure: record it as a failure, bounded. Never a delivered message.
            error = str(result.get("error") or "send failed")
            status = result.get("status")
            if status in (401, 403):
                # Permission/auth problem: a standing, diagnosable condition, not
                # something to retry against the recipient. Surface it and stop.
                self._note_permission_error(status)
                if ref_id:
                    self.deliveries.note_failure(ref_id, error, attempts=PROACTIVE_MAX_ATTEMPTS)
                self._inc_failed()
                return {"ok": False, "sent": False, "error": error,
                        "suppressed": "permission", "message_id": None}
            if ref_id:
                self.deliveries.note_failure(ref_id, error, attempts=attempts + 1)
            self._inc_failed()
            self.last_error = error
            self._consecutive_errors += 1
            return {"ok": False, "sent": False, "error": error, "suppressed": None,
                    "message_id": None}

    def _note_permission_error(self, status: Optional[int]) -> None:
        self.last_error = (f"discord permission/auth error (http {status}); "
                          "check the bot token and channel permissions")

    # -- inbound ---------------------------------------------------------
    def poll_once(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Fetch new messages, buffer them, and answer a completed batch.

        On the first poll with no stored cursor, existing channel contents are
        treated as a baseline: they are recorded but *not* answered. Delivery
        failures are counted and surfaced, never recorded as successes.
        """
        now = now or self._now()
        first_run = not self.state.last_message_id
        fetched = self.transport.fetch_messages(
            self.channel_id, after=self.state.last_message_id or None, limit=20)
        if not fetched.get("ok"):
            self.last_error = fetched.get("error")
            self._consecutive_errors += 1
            return {"ok": False, "error": fetched.get("error"), "handled": 0}

        # Success resets the transport error counter.
        self._consecutive_errors = 0
        self.last_fetch_ok_at = _now()
        messages = fetched.get("data") or []
        # Discord returns newest-first; process oldest-first.
        messages = list(reversed(messages))
        if first_run:
            for message in messages:
                self.state.set_last_message_id(str(message.get("id") or ""))
            return {"ok": True, "handled": 0, "error": None, "baselined": len(messages)}

        accepted = 0
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
            if not self._author_allowed(author):
                continue
            self._batch.append({
                "id": message_id,
                "content": content,
                "channel": str(message.get("channel_id") or self.channel_id),
                "author_id": str(author.get("id")) if author.get("id") else None,
            })
            accepted += 1
            if self._batch_since is None:
                self._batch_since = now

        answered = self._maybe_flush_batch(now)
        return {"ok": True, "handled": answered, "accepted": accepted,
                "buffered": len(self._batch), "error": None}

    def _maybe_flush_batch(self, now: datetime) -> int:
        """Answer the buffered batch once it is complete (quiet for the debounce)."""
        if not self._batch:
            return 0
        since = self._batch_since or now
        elapsed = (now - since).total_seconds()
        ready = elapsed >= self.batch_seconds or len(self._batch) >= self.max_batch
        if not ready:
            return 0
        return self._answer_batch()

    def _answer_batch(self) -> int:
        batch = self._batch
        self._batch = []
        self._batch_since = None
        if not batch:
            return 0
        # Prefer a single channel (a thread or the configured channel). If a
        # batch somehow spans channels, answer in the last message's channel.
        reply_channel = batch[-1]["channel"]
        combined = "\n".join(m["content"].strip() for m in batch if m["content"].strip())
        if not combined.strip():
            return 0
        last = batch[-1]

        self.transport.send_typing(reply_channel)
        result = self.router.handle(combined, source="discord")
        reply = result.get("text") or ""
        if not reply.strip():
            return 0

        # Reply-attach only when the response asks for emphasis; otherwise send a
        # plain message. A reply also pings the author of the emphasised message.
        clean, wants_reply = reply_policy.parse(reply)
        reference = last["id"] if wants_reply else None
        mention = last.get("author_id") if wants_reply else None
        if wants_reply and mention and "<@" not in clean:
            # Ensure the ping actually resolves by including the mention token.
            clean = f"<@{mention}> {clean}".strip()
        sent = self._send_reply(reply_channel, clean, reference=reference,
                                mention_user_id=mention)
        if sent.get("ok"):
            self._inc_delivered()
            if self.record_inbound_delivery and wants_reply:
                self.deliveries.note_delivered(f"reply:{last['id']}")
        else:
            # A transport failure is a failure, not a delivered message.
            self._inc_failed()
            self.last_error = sent.get("error")
            log.warning("Failed to send Discord reply: %s", sent.get("error"))
        return len(batch)

    def flush(self) -> int:
        """Answer any buffered batch immediately (used on shutdown)."""
        return self._answer_batch()

    def run(self) -> None:
        """Poll until stopped. Bounded: one fetch, then sleep.

        Uses exponential backoff when consecutive errors accumulate, so a
        persistent outage does not hammer the API. While a batch is buffering it
        polls at the debounce cadence so a thought is answered promptly.
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
            elif self._batch:
                # A batch is pending: poll no slower than the debounce so it is
                # answered as soon as the person stops typing.
                self._stop.wait(min(self.poll_seconds, max(0.5, self.batch_seconds)))
            else:
                self._stop.wait(self.poll_seconds)

    # -- diagnostics -----------------------------------------------------
    def diagnostics(self) -> Dict[str, Any]:
        """A truthful transport/delivery snapshot for the admin views."""
        connected = self.last_fetch_ok_at is not None and self.last_error is None
        return {
            "channel_id": self.channel_id,
            "connected": connected,
            "last_fetch_ok_at": self.last_fetch_ok_at,
            "last_error": self.last_error,
            "inbound_delivered": self.delivered,
            "inbound_failed": self.failed,
            "consecutive_errors": self._consecutive_errors,
            "last_proactive_at": self.last_proactive_at,
            "deliveries": self.deliveries.summary(),
            "buffered": len(self._batch),
        }


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
        allowed_author_ids=discord_cfg.get("allowed_author_ids"),
        poll_seconds=float(discord_cfg.get("poll_seconds", 2.0)),
        batch_seconds=float(discord_cfg.get("batch_seconds", 3.0)),
        max_batch=int(discord_cfg.get("max_batch", 8)))
