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
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests

from . import discord_behavior
from . import presence as presence_mod
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
# A transient transport error (timeout / dropped connection) is retried this
# many times *within one request* before it is reported as a failure, so a
# single blip does not fail a poll (and, for a proactive send, burn an attempt
# against the bounded delivery policy). Bounded, never infinite.
TRANSPORT_MAX_ATTEMPTS = 3
# Ceiling on a server-provided Retry-After (seconds); never sleep past this.
RATE_LIMIT_MAX_SLEEP = 10.0


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

    def _rate_limit_sleep(self, res: Any, attempt: int) -> float:
        """The bounded sleep before retrying a rate-limited request.

        Prefers Discord's structured ``retry_after`` (seconds), falls back to the
        ``Retry-After`` header, clamps to ``RATE_LIMIT_MAX_SLEEP`` and adds a
        small per-attempt jitter so concurrent senders do not resynchronise.
        """
        retry = 1.0
        try:
            body = res.json()
            if isinstance(body, dict) and body.get("retry_after") is not None:
                retry = float(body["retry_after"])
        except Exception:
            header = getattr(res, "headers", {}) or {}
            try:
                retry = float(header.get("Retry-After", retry))
            except (TypeError, ValueError):
                retry = 1.0
        return min(max(retry, 0.5), RATE_LIMIT_MAX_SLEEP) + 0.1 * attempt

    def _request(self, method: str, path: str, *, json_body: Optional[dict] = None,
                 params: Optional[dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Optional[str] = None
        for attempt in range(TRANSPORT_MAX_ATTEMPTS):
            try:
                res = self._session.request(
                    method, url, headers=self._headers(), json=json_body,
                    params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                # A timeout or dropped connection is transient: retry within the
                # same request with a short bounded backoff before giving up, so
                # a single blip does not fail the poll or burn a proactive send
                # attempt. Intermediate retries are quiet (info); only the final
                # failure is a warning, so a flapping link is not log-spam.
                last_error = f"transport: {exc}"
                if attempt + 1 < TRANSPORT_MAX_ATTEMPTS:
                    log.info("Discord transport error on %s %s (attempt %d/%d), retrying: %s",
                             method, path, attempt + 1, TRANSPORT_MAX_ATTEMPTS, exc)
                    time.sleep(min(0.5 * (2 ** attempt), 5.0) + 0.1 * random.random())
                    continue
                log.warning("Discord transport error on %s %s (after %d attempts): %s",
                            method, path, TRANSPORT_MAX_ATTEMPTS, exc)
                return {"ok": False, "error": last_error, "status": None}
            if res.status_code == 429:
                sleep_for = self._rate_limit_sleep(res, attempt)
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
            # A 5xx is the server side being briefly unavailable: retry it; a 4xx
            # is a definite answer (auth, permissions, not-found) and is returned
            # at once so it can be diagnosed rather than hammered.
            if 500 <= res.status_code < 600 and attempt + 1 < TRANSPORT_MAX_ATTEMPTS:
                log.info("Discord HTTP %d on %s %s (attempt %d/%d), retrying",
                         res.status_code, method, path, attempt + 1,
                         TRANSPORT_MAX_ATTEMPTS)
                time.sleep(min(0.5 * (2 ** attempt), 5.0))
                continue
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

    def reload(self) -> None:
        """Re-read from disk, for a separate read-only inspection process."""
        with self._lock:
            self._state = self._load()

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

    def reload(self) -> None:
        """Re-read from disk, for a separate read-only inspection process."""
        with self._lock:
            self._state = self._load()

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


class DiscordHealthStore:
    """A tiny persisted record of the live transport's connection health.

    Held separately from the delivery bookkeeping (which is about *outbound*
    messages) and from the inbound cursor. Its whole purpose is that a second,
    read-only process can report whether Discord is reachable, and when it last
    was - without itself opening a connection.
    """

    def __init__(self, data_dir: str = "./storage") -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "discord_health.json")
        self._lock = threading.RLock()
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        raw = load_json(self.path, dict, dict)
        raw = raw if isinstance(raw, dict) else {}
        return {"version": 1,
                "last_fetch_ok_at": raw.get("last_fetch_ok_at"),
                "last_send_ok_at": raw.get("last_send_ok_at"),
                "last_error": raw.get("last_error"),
                "written_at": raw.get("written_at")}

    def reload(self) -> None:
        with self._lock:
            self._state = self._load()

    def read(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def record(self, *, last_fetch_ok_at: Optional[str] = None,
               last_send_ok_at: Optional[str] = None,
               last_error: Optional[str] = None) -> None:
        with self._lock:
            if last_fetch_ok_at:
                self._state["last_fetch_ok_at"] = last_fetch_ok_at
            if last_send_ok_at:
                self._state["last_send_ok_at"] = last_send_ok_at
            self._state["last_error"] = last_error
            self._state["written_at"] = _now()
            atomic_save(self.path, self._state)


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
                 require_mention: bool = False,
                 bot_name: Optional[str] = None,
                 typing_wpm: float = discord_behavior.DEFAULT_WPM,
                 typing_min_seconds: float = discord_behavior.MIN_TYPING_SECONDS,
                 typing_max_seconds: float = discord_behavior.MAX_TYPING_SECONDS,
                 typing_simulation: bool = False,
                 check_on_cycle: bool = False,
                 check_interval_seconds: float = 0.0,
                 modulation_provider: Optional[Callable[[], Dict[str, Any]]] = None,
                 presence_base_patience: float = presence_mod.DEFAULT_PATIENCE_SECONDS,
                 stop_event: Optional[threading.Event] = None,
                 now_fn: Optional[Callable[[], datetime]] = None,
                 sleep_fn: Optional[Callable[[float], None]] = None) -> None:
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
        # Whether a message must address Astra (mention/reply) before she answers
        # it. Off by default so the direct-adapter tests keep their behaviour;
        # the config builder turns it ON, so a running Astra is not permanently
        # watching the channel.
        self.require_mention = bool(require_mention)
        self.bot_name = str(bot_name).strip() if bot_name else None
        # Optional: with the mention gate on, she still looks at the channel once
        # per this many seconds and answers (a low-frequency "check"). 0 = never,
        # though an explicit check_messages() always works.
        self.check_on_cycle = bool(check_on_cycle)
        self.check_interval_seconds = max(0.0, float(check_interval_seconds))
        self._last_check = time.monotonic()
        # Typing realism: delay a reply to roughly how long a person would take
        # to type it (words / wpm), in human bursts. Non-blocking against the
        # poll loop, so a message arriving mid-typing is read first.
        self.typing_wpm = float(typing_wpm)
        self.typing_min_seconds = float(typing_min_seconds)
        self.typing_max_seconds = float(typing_max_seconds)
        # Off by default so direct construction (tests) sends synchronously; the
        # config builder turns it ON for the running runtime.
        self.typing_simulation = bool(typing_simulation)
        # Her current behaviour modulation, read on each reply so the affect of
        # the moment shapes *how long* she types, waits and lingers. Pure read;
        # the adapter never writes affect. ``None`` -> neutral timing.
        self.modulation_provider = modulation_provider
        # The modulation is read at most this often. The poll loop asks about
        # presence each iteration, and a full store read per poll would be
        # wasteful; a few seconds of staleness is irrelevant to a coarse timing
        # decision, and nothing is written.
        self._modulation_cache: Optional[Dict[str, Any]] = None
        self._modulation_cache_at = 0.0
        self.presence_base_patience = float(presence_base_patience)
        # When Roum has gone quiet and she is back at her own work, poll this
        # slowly (still checking for messages, just not hovering).
        self.away_poll_seconds = max(self.poll_seconds, 30.0)
        # When Roum's last addressed message arrived, and whether she is still
        # holding the exchange open. Session-only scratch, never persisted.
        self._last_exchange: Optional[datetime] = None
        self._exchange_open = False
        self._stop = stop_event or threading.Event()
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep_fn or time.sleep
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
        # Typing simulation: one background thread at a time, interruptible when
        # a new message arrives so she stops and reads instead of talking over it.
        self._typing_lock = threading.Lock()
        self._typing_thread: Optional[threading.Thread] = None
        self._typing_interrupt = threading.Event()
        # A reply that has been planned and whose typing is in flight. Held here,
        # not sent from the typing thread, so the poll loop can pre-empt it when
        # a new message arrives. None when nothing is pending.
        self._reply_in_flight: Optional[Dict[str, Any]] = None
        # A tiny persisted health record so a *separate* process (the read-only
        # inspection console) can show whether the live transport is connected.
        # The transient connect/error state is in-memory only; writing the two
        # facts that matter - the last successful poll and the last error - lets
        # the console read what Discord is actually doing without being the
        # process that holds the connection. Best-effort: a health-write failure
        # must never affect delivery.
        self._health = DiscordHealthStore(data_dir=data_dir)
        self._health_last_error: Optional[str] = None
        self._health_written_at: Optional[float] = None

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

    def _drain_on_exit(self) -> None:
        """Deliver anything in flight when the poll loop ends (best-effort)."""
        try:
            self.flush()
        except Exception:
            pass

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

    def _addressed(self, message: Dict[str, Any]) -> bool:
        """Whether Astra is being spoken to rather than merely observing.

        When ``require_mention`` is set (the running default), only an @-mention,
        a reply to one of her messages, or an explicit check counts. When it is
        off (direct construction in tests), every allowed message counts, which
        preserves the original behaviour.
        """
        if not self.require_mention:
            return True
        referenced = discord_behavior.referenced_bot(message, self.bot_user_id)
        return discord_behavior.is_addressed(
            message.get("content") or "",
            bot_user_id=self.bot_user_id, bot_name=self.bot_name,
            referenced_bot=referenced)

    # -- state-shaped timing --------------------------------------------
    def _modulation(self) -> Dict[str, Any]:
        """Her current behaviour modulation, or a neutral one.

        A pure read through the injected provider; never raises into polling.
        The adapter only *reads* her state to shape timing - it writes nothing.
        Cached briefly so the poll loop's per-iteration presence check does not
        re-read the store every couple of seconds.
        """
        provider = self.modulation_provider
        if provider is None:
            return {}
        now = time.monotonic()
        cached = self._modulation_cache
        if cached is not None and (now - self._modulation_cache_at) < 3.0:
            return cached
        try:
            modulation = provider()
            modulation = modulation if isinstance(modulation, dict) else {}
        except Exception:
            modulation = {}
        self._modulation_cache = modulation
        self._modulation_cache_at = now
        return modulation

    def _timing(self) -> Dict[str, float]:
        """The affect-scaled multipliers for this moment (typing/read/patience)."""
        from . import behavior
        return behavior.timing_modifiers(self._modulation())

    def _note_exchange(self, when: Optional[datetime] = None) -> None:
        """Roum addressed her: this starts (or continues) a live exchange."""
        self._last_exchange = when or self._now()
        self._exchange_open = True

    def _presence(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Whether the exchange is still live, given the gap and her state.

        Once Roum has gone quiet past her (state-scaled) patience, the exchange
        closes and she returns to her own work; it reopens the moment he speaks
        again. ``_exchange_open`` is what the poll loop uses to decide whether to
        keep watching *only* for a follow-up or to resume the normal schedule.
        """
        now = now or self._now()
        gap: Optional[float] = None
        if self._last_exchange is not None:
            gap = max(0.0, (now - self._last_exchange).total_seconds())
        view = presence_mod.assess(
            gap, modulation=self._modulation(),
            base_patience=self.presence_base_patience)
        if view.get("resume_background"):
            self._exchange_open = False
        else:
            self._exchange_open = True
        return view

    # -- typing simulation ----------------------------------------------
    def _scale_timing(self, segments: Optional[List[float]],
                      scale: float) -> Optional[List[float]]:
        """Scale sleep segments by a bounded factor, preserving their total.

        The typing thread sleeps each segment in turn, so multiplying each by
        the same factor scales the whole simulation while keeping the burst
        rhythm. A neutral scale returns the segments untouched.
        """
        if not segments:
            return segments
        try:
            factor = max(0.2, min(3.0, float(scale)))
        except (TypeError, ValueError):
            return segments
        if abs(factor - 1.0) < 1e-6:
            return segments
        return [round(seg * factor, 3) for seg in segments]

    def _begin_typing(self, channel: str, content: str,
                      segments: Optional[List[float]] = None) -> None:
        """Start typing the reply at a human pace, in bursts, in the background.

        Starts a daemon thread that keeps the typing indicator alive and sleeps
        in human-sized bursts (with small pauses between them). The final send is
        *not* done here - it happens on the poll loop via :meth:`_flush_pending`,
        which is what lets an arriving message pre-empt it. A message that lands
        mid-typing sets ``_typing_interrupt``, so the thread stops promptly and
        the pending reply is discarded: she reads the new thought first.
        """
        if not self.typing_simulation:
            # Still show the indicator once, so the channel sees her react.
            try:
                self.transport.send_typing(channel)
            except Exception:
                pass
            return
        if segments is None:
            segments, _ = discord_behavior.typing_plan(
                content, wpm=self.typing_wpm,
                min_seconds=self.typing_min_seconds,
                max_seconds=self.typing_max_seconds)
        self._typing_interrupt.clear()

        def _work() -> None:
            last_typing = 0.0
            for seg in segments:
                if self._stop.is_set() or self._typing_interrupt.is_set():
                    return
                now = time.monotonic()
                if now - last_typing >= discord_behavior.TYPING_REFRESH_SECONDS:
                    try:
                        self.transport.send_typing(channel)
                    except Exception:
                        pass
                    last_typing = now
                # Interruptible wait in small steps so a new message stops the
                # typing promptly rather than after the whole segment.
                waited = 0.0
                step = 0.25
                while waited < seg:
                    if self._stop.is_set() or self._typing_interrupt.is_set():
                        return
                    self._stop.wait(min(step, seg - waited))
                    waited += step

        with self._typing_lock:
            thread = threading.Thread(target=_work, name="astra-typing", daemon=True)
            self._typing_thread = thread
        thread.start()

    def _typing_in_flight(self) -> bool:
        with self._typing_lock:
            thread = self._typing_thread
        return thread is not None and thread.is_alive()

    def _join_typing(self, timeout: Optional[float] = None) -> None:
        """Wait for an in-flight typing simulation to finish (or be interrupted)."""
        with self._typing_lock:
            thread = self._typing_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _interrupt_typing(self, now: Optional[datetime] = None) -> None:
        """Stop an in-flight typing simulation and abandon its unsent reply.

        The abandoned reply's source messages are put back at the front of the
        batch, so the next turn reads them together with the thought that just
        arrived - she answers the *full* thought, not the first half of it.
        """
        self._typing_interrupt.set()
        with self._send_lock:
            pending = self._reply_in_flight
            if pending is not None:
                # Nothing was sent, so this is not a delivery failure - the reply
                # is simply superseded by the thought that just arrived.
                self._reply_in_flight = None
                source = pending.get("source_batch") or []
                if source:
                    self._batch = list(source) + self._batch
                    if self._batch_since is None:
                        self._batch_since = now or self._now()
                log.debug("Discord: reply abandoned, new message arrived")

    def _flush_pending(self) -> int:
        """Send a pending reply whose typing has finished. Non-blocking.

        Called on every poll (and on ``flush``). Sends only when the typing
        thread has completed; a still-running simulation is left for the next
        poll. Returns 1 when a message was sent, else 0.
        """
        with self._send_lock:
            pending = self._reply_in_flight
        if pending is None:
            return 0
        if self._typing_in_flight():
            return 0
        with self._send_lock:
            pending = self._reply_in_flight
            self._reply_in_flight = None
        return self._deliver_pending(pending)

    def _deliver_pending(self, pending: Dict[str, Any]) -> int:
        """Deliver a prepared reply and record the outcome honestly."""
        # A short "last look-over" beat before the message actually goes out:
        # the thought is finished, she sends it a moment later, not the instant
        # the typing stops. Interruptible, so a new message still pre-empts it.
        pause = float(pending.get("before_send") or 0.0)
        if pause > 0:
            waited = 0.0
            step = 0.2
            while waited < pause:
                if self._stop.is_set() or self._typing_interrupt.is_set():
                    return 0
                self._stop.wait(min(step, pause - waited))
                waited += step
        result = self._send_reply(
            pending["channel"], pending["content"],
            reference=pending.get("reference"),
            mention_user_id=pending.get("mention"))
        if result.get("ok"):
            self._inc_delivered()
            if self.record_inbound_delivery and pending.get("wants_reply"):
                self.deliveries.note_delivered(f"reply:{pending['last_id']}")
            return 1
        self._inc_failed()
        self.last_error = result.get("error")
        log.warning("Failed to send Discord reply: %s", result.get("error"))
        return 0

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
            if i > 0 and self.typing_simulation:
                # A "no-comment ping": hop to the next part with a small pause and
                # a brief typing refresh, so a split reply reads as one person
                # continuing rather than a burst of pasted messages.
                try:
                    self.transport.send_typing(channel_id)
                except Exception:
                    pass
                self._stop.wait(min(1.5, max(0.6, len(part) / 12.0)))
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
                self._record_health(ok=True, sent=True)
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
                self._record_health(ok=False, error=self.last_error)
                return {"ok": False, "sent": False, "error": error,
                        "suppressed": "permission", "message_id": None}
            if ref_id:
                self.deliveries.note_failure(ref_id, error, attempts=attempts + 1)
            self._inc_failed()
            self.last_error = error
            self._consecutive_errors += 1
            self._record_health(ok=False, error=error)
            return {"ok": False, "sent": False, "error": error, "suppressed": None,
                    "message_id": None}

    def _note_permission_error(self, status: Optional[int]) -> None:
        self.last_error = (f"discord permission/auth error (http {status}); "
                          "check the bot token and channel permissions")

    def _record_health(self, *, ok: bool, error: Optional[str] = None,
                       sent: bool = False) -> None:
        """Persist connection/delivery health, throttled (best-effort).

        A poll succeeds every couple of seconds, so the success path is written
        at most once a minute; a *changed* error is always written at once, so a
        new disconnect is visible immediately without hammering the disk while an
        outage persists. A successful proactive send is likewise written on the
        change (its timestamp advances), so the read-only console can show when
        Astra last actually reached Roum. A write failure never affects polling.
        """
        now = time.monotonic()
        changed = (error or None) != self._health_last_error or sent
        if (not changed and self._health_written_at is not None
                and (now - self._health_written_at) < 60.0):
            return
        try:
            self._health.record(
                last_fetch_ok_at=self.last_fetch_ok_at if (ok and not sent) else None,
                last_send_ok_at=self.last_proactive_at if sent else None,
                last_error=None if ok else (error or self.last_error))
            self._health_last_error = error or None
            self._health_written_at = now
        except Exception:
            pass  # health bookkeeping must never break polling

    # -- inbound ---------------------------------------------------------
    def poll_once(self, *, now: Optional[datetime] = None,
                  force: bool = False) -> Dict[str, Any]:
        """Fetch new messages, buffer them, and answer a completed batch.

        On the first poll with no stored cursor, existing channel contents are
        treated as a baseline: they are recorded but *not* answered. Delivery
        failures are counted and surfaced, never recorded as successes.

        ``force`` answers any buffered batch regardless of whether it addressed
        Astra; it is the "she is checking her messages" path. A normal poll only
        buffers messages that are actually addressed to her (mention or reply)
        when ``require_mention`` is set, so she is not watching the channel.
        """
        now = now or self._now()
        first_run = not self.state.last_message_id
        fetched = self.transport.fetch_messages(
            self.channel_id, after=self.state.last_message_id or None, limit=20)
        if not fetched.get("ok"):
            self.last_error = fetched.get("error")
            self._consecutive_errors += 1
            self._record_health(ok=False, error=self.last_error)
            return {"ok": False, "error": fetched.get("error"), "handled": 0}

        # Success resets the transport error counter.
        self._consecutive_errors = 0
        self.last_fetch_ok_at = _now()
        self._record_health(ok=True)
        messages = fetched.get("data") or []
        # Discord returns newest-first; process oldest-first.
        messages = list(reversed(messages))
        if first_run:
            for message in messages:
                self.state.set_last_message_id(str(message.get("id") or ""))
            return {"ok": True, "handled": 0, "error": None, "baselined": len(messages)}

        accepted = 0
        interrupted = False
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
            # The addressed gate: unless she is explicitly checking her
            # messages, a line not directed at her does not summon her.
            if not force and not self._addressed(message):
                continue
            # Roum is addressing her: a live exchange is (re)opened, so the poll
            # loop watches closely for the rest of the thought. It closes again
            # (state-scaled) once he has gone quiet.
            self._note_exchange(now)
            # A new addressed message arrived: stop any in-flight typing and read
            # it before replying, so she does not talk over the new thought.
            interrupted = True
            self._batch.append({
                "id": message_id,
                "content": content,
                "channel": str(message.get("channel_id") or self.channel_id),
                "author_id": str(author.get("id")) if author.get("id") else None,
            })
            accepted += 1
            if self._batch_since is None:
                self._batch_since = now

        if interrupted:
            self._interrupt_typing()

        # A finished typing simulation's reply is sent first (it was planned
        # before this batch), then a newly-complete batch is answered. Both are
        # non-blocking: a still-typing reply is left for a later poll.
        flushed = self._flush_pending()
        answered = self._maybe_flush_batch(now) + flushed
        return {"ok": True, "handled": answered, "accepted": accepted,
                "buffered": len(self._batch), "error": None}

    def _maybe_flush_batch(self, now: datetime) -> int:
        """Answer the buffered batch once it is complete (quiet for the debounce)."""
        if not self._batch:
            return 0
        # Do not start a new reply while one is still being typed / pending: she
        # finishes (or abandons) the current thought before composing the next.
        with self._send_lock:
            pending = self._reply_in_flight
        if pending is not None or self._typing_in_flight():
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

        # Reply-attach only when the response asks for emphasis *and* we are
        # answering one specific message; otherwise send a plain message. A reply
        # also pings the author of the emphasised message, because a reply that
        # does not ping is easily missed - the two go together by design.
        clean, wants_reply = reply_policy.parse(reply)
        wants_reply = wants_reply and bool(last.get("id"))
        reference = last["id"] if wants_reply else None
        mention = last.get("author_id") if wants_reply else None
        if wants_reply and mention and "<@" not in clean:
            # Ensure the ping actually resolves by including the mention token.
            clean = f"<@{mention}> {clean}".strip()

        if self.typing_simulation:
            # Plan the whole thing at once: a per-message tempo and structure
            # decide the burst pattern plus a short before-send beat. The send
            # happens later from the poll loop, so a message that arrives
            # meanwhile pre-empts it (she stops typing, reads, and re-answers
            # with the new thought).
            segments, before_send = discord_behavior.typing_plan(
                clean, wpm=self.typing_wpm, min_seconds=self.typing_min_seconds,
                max_seconds=self.typing_max_seconds)
            # Her present state scales the timing: a warm, engaged state types
            # a little brisker and looks the message over quickly; a flat, weary
            # or sad one is slower. Bounded, and never dragged past the ceiling.
            timing = self._timing()
            segments = self._scale_timing(segments, timing.get("typing_scale", 1.0))
            before_send = round(min(
                self.typing_max_seconds,
                before_send * float(timing.get("read_scale", 1.0))), 3)
            with self._send_lock:
                self._reply_in_flight = {
                    "channel": reply_channel, "content": clean,
                    "reference": reference, "mention": mention,
                    "wants_reply": wants_reply, "last_id": last["id"],
                    "before_send": before_send,
                    "typing_seconds": round(sum(segments) + before_send, 3),
                    "source_batch": list(batch),
                }
            self._begin_typing(reply_channel, clean, segments=segments)
            return len(batch)

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

    def check_messages(self) -> int:
        """Explicitly check the channel: fetch and answer anything unhandled.

        This is the "she decides to check her messages" path: it bypasses the
        mention gate (she is looking), answers a ready batch, and delivers any
        typing that has finished. Toggled from config; not called on a normal
        poll. Returns the number of messages handled.
        """
        report = self.poll_once(force=True)
        return int(report.get("handled") or 0)

    def _check_cycle(self) -> int:
        """Autonomous check-once-a-cycle when configured.

        A low-frequency alternative to the mention gate for a single-user
        channel: even without being pinged, she looks once per cycle and answers.
        Off by default, because it makes her attentive again.
        """
        report = self.poll_once(force=True)
        return int(report.get("handled") or 0)

    def flush(self) -> int:
        """Deliver any pending typing and answer any buffered batch (shutdown)."""
        self._join_typing(timeout=5.0)
        flushed = self._flush_pending()
        return flushed + self._answer_batch()

    def run(self) -> None:
        """Poll until stopped. Bounded: one fetch, then sleep.

        Uses exponential backoff when consecutive errors accumulate, so a
        persistent outage does not hammer the API. While a batch is buffering (or
        a reply is being typed) it polls at the debounce cadence so it is
        delivered promptly.
        """
        while not self._stop.is_set():
            try:
                self.poll_once()
                # Optional autonomous check: even without being pinged, look at
                # the channel once per check_interval. Off by default.
                if (self.check_on_cycle and self.check_interval_seconds > 0
                        and (time.monotonic() - self._last_check) >= self.check_interval_seconds):
                    self._last_check = time.monotonic()
                    self._check_cycle()
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
            elif self._batch or self._pending():
                # Work pending: poll no slower than the debounce so it is
                # delivered as soon as the person stops typing.
                self._stop.wait(min(self.poll_seconds, max(0.5, self.batch_seconds)))
            else:
                # Presence: while Roum is here (or before any exchange, so a
                # fresh ping is seen promptly) she watches at the normal cadence;
                # once he has gone quiet she falls back to a slow check and gets
                # on with her own work.
                away = (self._last_exchange is not None
                        and self._presence()["mode"] == presence_mod.AWAY)
                self._stop.wait(max(self.poll_seconds, self.away_poll_seconds)
                                if away else self.poll_seconds)
        self._drain_on_exit()

    def _pending(self) -> bool:
        with self._send_lock:
            return self._reply_in_flight is not None

    # -- diagnostics -----------------------------------------------------
    def diagnostics(self) -> Dict[str, Any]:
        """A truthful transport/delivery snapshot for the admin views."""
        connected = self.last_fetch_ok_at is not None and self.last_error is None
        try:
            presence = self._presence()
        except Exception:
            presence = {}
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
            "pending_reply": self._pending(),
            "typing": self._typing_in_flight(),
            "require_mention": self.require_mention,
            "presence": presence.get("mode"),
            "exchange_open": bool(self._exchange_open),
            "patience_seconds": presence.get("patience"),
            "last_exchange_at": (self._last_exchange.isoformat()
                                 if self._last_exchange else None),
            "timing": self._timing(),
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

    def _modulation_provider() -> Dict[str, Any]:
        """Her current behaviour modulation, read live through the router.

        The adapter uses it to scale typing/patience by her present state. A
        read-only view: fetching it never mutates affect, memory or the store,
        and a missing orchestrator (or any error) degrades to neutral timing.
        """
        orchestrator = getattr(getattr(router, "session", None), "orchestrator", None)
        if orchestrator is None or not callable(
                getattr(orchestrator, "current_modulation", None)):
            return {}
        try:
            return orchestrator.current_modulation() or {}
        except Exception:
            return {}

    # Running defaults: do not answer the channel unless addressed, and reply at
    # a human typing pace. Both are configurable, and both are ON here even
    # though the constructor defaults are off (so direct/test construction stays
    # simple and deterministic).
    return DiscordAdapter(
        transport, router, channel_id=channel_id, data_dir=data_dir,
        bot_user_id=discord_cfg.get("bot_user_id"),
        allowed_author_ids=discord_cfg.get("allowed_author_ids"),
        poll_seconds=float(discord_cfg.get("poll_seconds", 2.0)),
        batch_seconds=float(discord_cfg.get("batch_seconds", 3.0)),
        max_batch=int(discord_cfg.get("max_batch", 8)),
        require_mention=bool(discord_cfg.get("require_mention", True)),
        bot_name=discord_cfg.get("bot_name"),
        typing_wpm=float(discord_cfg.get("typing_wpm")
                         or discord_behavior.DEFAULT_WPM),
        typing_min_seconds=float(discord_cfg.get("typing_min_seconds")
                                 or discord_behavior.MIN_TYPING_SECONDS),
        typing_max_seconds=float(discord_cfg.get("typing_max_seconds")
                                 or discord_behavior.MAX_TYPING_SECONDS),
        typing_simulation=bool(discord_cfg.get("typing_simulation", True)),
        check_on_cycle=bool(discord_cfg.get("check_on_cycle", False)),
        check_interval_seconds=float(discord_cfg.get("check_interval_seconds", 0.0)),
        modulation_provider=_modulation_provider,
        presence_base_patience=float(discord_cfg.get("presence_base_patience")
                                     or presence_mod.DEFAULT_PATIENCE_SECONDS))
