"""Regression tests for the durable incoming-message lifecycle.

These cover the actual failure the audit reported: a mention arriving while she
was busy left *no trace*, so it could neither be answered later nor even shown in
the inspection console. The fix is a durable per-message record
(:class:`astra.discord_adapter.MessageInbox`), a bounded reconciliation pass, and
a recovery pass that re-offers anything unresolved.

The transport and router are real protocol implementations (no mocks at the
level of the code under test); the store is written to a temp dir and re-read
from disk where durability matters.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from astra import discord_adapter as da


class FakeTransport:
    """Implements the DiscordTransport protocol; records sends and fetches."""

    def __init__(self, messages, *, fail_fetch=False):
        self.messages = messages
        self.fail_fetch = fail_fetch
        self.sent = []
        self.typing_calls = []

    def fetch_messages(self, channel_id, *, after=None, limit=20):
        if self.fail_fetch:
            return {"ok": False, "error": "http 503"}
        msgs = list(self.messages)
        if after:
            msgs = [m for m in msgs if str(m["id"]) > str(after)]
        # Discord returns newest-first.
        return {"ok": True, "data": msgs[::-1][:limit]}

    def send_message(self, channel_id, content, *, reference=None,
                     mention_user_id=None):
        self.sent.append({"channel": channel_id, "content": content,
                          "reference": reference, "mention": mention_user_id})
        return {"ok": True, "data": {"id": "srv-1"}}

    def send_typing(self, channel_id):
        self.typing_calls.append(channel_id)
        return {"ok": True, "data": {}}


class FakeRouter:
    def __init__(self, reply="ok", *, raise_error=False):
        self.reply = reply
        self.raise_error = raise_error
        self.calls = []

    def handle(self, text, *, source="cli"):
        self.calls.append((text, source))
        if self.raise_error:
            raise RuntimeError("model unavailable")
        return {"text": self.reply, "handled": True, "source": source}


class FakeClock:
    """A monotonic, mutable clock so tests can advance time deterministically."""

    def __init__(self, start=None):
        self.now = start or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now = self.now + timedelta(seconds=seconds)


def _mention(mid, content="<@42> hi", author="u1"):
    return {"id": mid, "content": content, "author": {"id": author},
            "channel_id": "c1"}


def _addressed(mid, content, author="u1"):
    """A message that addresses Astra (so the mention gate lets it through)."""
    return {"id": mid, "content": f"<@42> {content}", "author": {"id": author},
            "channel_id": "c1"}


def _plain(mid, content="just chatting", author="u1"):
    return {"id": mid, "content": content, "author": {"id": author},
            "channel_id": "c1"}


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.clock = FakeClock()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def adapter(self, transport, router, *, batch_seconds=0.0, **kw):
        kw.setdefault("require_mention", True)
        kw.setdefault("bot_user_id", "42")
        return da.DiscordAdapter(
            transport, router, channel_id="c1", data_dir=self.tmp,
            batch_seconds=batch_seconds, now_fn=self.clock, **kw)


class TestDurableReceipt(_Base):
    def test_mention_is_durably_recorded_before_answering(self):
        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter()
        adapter = self.adapter(transport, router)
        adapter.poll_once()  # baseline
        transport.messages = [_addressed("2", "you there?")]
        report = adapter.poll_once()
        self.assertEqual(report["accepted"], 1)
        # The record survives on disk and reads back as answered.
        fresh = da.MessageInbox(data_dir=self.tmp)
        self.assertEqual(fresh.state_of("2"), da.MSG_RESPONDED)

    def test_pending_message_survives_a_restart_before_it_is_answered(self):
        # She receives a mention but the process dies before answering it. The
        # durable record must still show it as awaiting a reply.
        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter()
        adapter = self.adapter(transport, router, batch_seconds=60.0)
        adapter.poll_once()  # baseline
        transport.messages = [_addressed("2", "important")]
        adapter.poll_once()  # buffered, not yet answered (debounce)
        self.assertNotEqual(adapter.inbox.state_of("2"), da.MSG_RESPONDED)
        # A *new* process (same data_dir) reads the pending record.
        fresh = da.MessageInbox(data_dir=self.tmp)
        record = fresh.get("2")
        self.assertEqual(record["state"], da.MSG_PENDING)
        self.assertIn("important", record["content"])

    def test_recovery_rebuffers_a_pending_message_after_restart(self):
        # Simulate a crash: a message recorded as pending, no in-memory batch.
        inbox = da.MessageInbox(data_dir=self.tmp)
        inbox.receive("9", content="<@42> where are you?", author_id="u1",
                      channel_id="c1", addressed=True)
        inbox.mark("9", da.MSG_PENDING)
        # Age it past the recovery grace window.
        inbox._state["records"]["9"]["received_at"] = (
            self.clock.now - timedelta(minutes=5)).isoformat()
        inbox._persist()

        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter(reply="I'm here.")
        adapter = self.adapter(transport, router, batch_seconds=0.0)
        adapter.poll_once()  # baseline, and the recovery pass re-buffers "9"
        self.assertIn("where are you?", "\n".join(c[0] for c in router.calls))
        fresh = da.MessageInbox(data_dir=self.tmp)
        self.assertEqual(fresh.state_of("9"), da.MSG_RESPONDED)

    def test_seed_baseline_is_not_recorded_as_a_message(self):
        # A fresh baseline must not fill the inbox with old history.
        transport = FakeTransport([_mention("1", "old"), _plain("2", "older")])
        router = FakeRouter()
        adapter = self.adapter(transport, router)
        adapter.poll_once()
        self.assertEqual(adapter.inbox.snapshot()["total"], 0)


class TestIdempotency(_Base):
    def test_a_message_is_never_answered_twice(self):
        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter()
        adapter = self.adapter(transport, router)
        adapter.poll_once()
        transport.messages = [_addressed("2", "hello")]
        adapter.poll_once()
        self.assertEqual(len(router.calls), 1)
        # Force a reconciliation that re-fetches the same message.
        adapter.reconcile()
        self.assertEqual(len(router.calls), 1)
        self.assertEqual(len(transport.sent), 1)

    def test_reconcile_recovers_a_missed_mention_without_duplicating_the_old_one(self):
        transport = FakeTransport([_mention("1", "seen")])
        router = FakeRouter()
        adapter = self.adapter(transport, router)
        adapter.poll_once()
        # Two more mentions appear (as if delivered during an outage).
        transport.messages = [_mention("1", "seen"),
                              _addressed("2", "first missed"),
                              _addressed("3", "second missed")]
        adapter.reconcile()
        answered = "\n".join(c[0] for c in router.calls)
        self.assertIn("first missed", answered)
        self.assertIn("second missed", answered)
        # The already-seen one is not answered again.
        self.assertEqual(len(router.calls), 1)  # one combined turn


class TestReconciliationOnError(_Base):
    def test_a_failed_fetch_arms_reconciliation(self):
        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter()
        adapter = self.adapter(transport, router)
        adapter.poll_once()
        transport.fail_fetch = True
        report = adapter.poll_once()
        self.assertFalse(report["ok"])
        self.assertTrue(adapter.diagnostics()["reconcile_needed"])

    def test_messages_missed_during_an_outage_are_recovered(self):
        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter()
        adapter = self.adapter(transport, router)
        adapter.poll_once()
        # Outage: fetch fails while a mention arrives.
        transport.fail_fetch = True
        transport.messages = [_mention("1", "seed"), _addressed("2", "arrived during outage")]
        adapter.poll_once()
        # Outage ends.
        transport.fail_fetch = False
        adapter.poll_once()
        self.assertIn("arrived during outage",
                      "\n".join(c[0] for c in router.calls))

    def test_lost_cursor_with_existing_inbox_reconciles_not_baselines(self):
        # The corrupt-cursor case from the runtime: the cursor file lost its
        # value but the runtime has been seen before. A mention in the window
        # must be recovered, not treated as history to skip.
        inbox = da.MessageInbox(data_dir=self.tmp)
        inbox.receive("1", content="<@42> earlier", author_id="u1",
                      channel_id="c1", addressed=True)
        inbox.mark("1", da.MSG_RESPONDED)
        transport = FakeTransport([_mention("1", "earlier"),
                                   _addressed("2", "missed while cursor lost")])
        router = FakeRouter()
        adapter = self.adapter(transport, router)
        adapter.poll_once()  # cursor empty but inbox non-empty -> reconcile
        self.assertIn("missed while cursor lost",
                      "\n".join(c[0] for c in router.calls))


class TestDeferral(_Base):
    def test_a_deferred_message_is_reoffered_when_due(self):
        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter(reply="back to you")
        adapter = self.adapter(transport, router, batch_seconds=60.0)
        adapter.poll_once()
        transport.messages = [_addressed("2", "quick question")]
        adapter.poll_once()  # buffered
        adapter.defer("2", reason="reading", revisit_seconds=10.0)
        self.assertEqual(adapter.inbox.state_of("2"), da.MSG_DEFERRED)
        # Before the revisit time: still deferred.
        self.clock.advance(5)
        adapter.poll_once()
        self.assertEqual(adapter.inbox.state_of("2"), da.MSG_DEFERRED)
        # After the revisit time: answered.
        self.clock.advance(10)
        adapter.poll_once()
        self.assertEqual(adapter.inbox.state_of("2"), da.MSG_RESPONDED)
        self.assertIn("quick question", "\n".join(c[0] for c in router.calls))

    def test_a_deferred_message_at_the_retry_limit_is_retired(self):
        inbox = da.MessageInbox(data_dir=self.tmp)
        inbox.receive("5", content="<@42> hard one", author_id="u1",
                      channel_id="c1", addressed=True)
        inbox.mark("5", da.MSG_DEFERRED,
                   revisit_at=self.clock.now.isoformat(), bump_attempt=True)
        # Force it to have exhausted its retries.
        inbox._state["records"]["5"]["attempts"] = da.DEFER_MAX_ATTEMPTS
        inbox._persist()
        adapter = self.adapter(FakeTransport([]), FakeRouter())
        adapter._recover_unresolved(self.clock.now)
        self.assertEqual(adapter.inbox.state_of("5"), da.MSG_FAILED)
        self.assertNotIn("5", [m.get("id") for m in adapter._batch])

    def test_answered_message_cannot_be_deferred(self):
        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter()
        adapter = self.adapter(transport, router)
        adapter.poll_once()
        transport.messages = [_addressed("2", "hi")]
        adapter.poll_once()
        self.assertEqual(adapter.inbox.state_of("2"), da.MSG_RESPONDED)
        adapter.defer("2", revisit_seconds=0.0)
        self.assertEqual(adapter.inbox.state_of("2"), da.MSG_RESPONDED)


class TestHonestOutcomes(_Base):
    def test_generation_failure_is_recorded_as_failed_not_responded(self):
        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter(raise_error=True)
        adapter = self.adapter(transport, router)
        adapter.poll_once()
        transport.messages = [_addressed("2", "hello")]
        adapter.poll_once()
        self.assertEqual(adapter.inbox.state_of("2"), da.MSG_FAILED)
        self.assertEqual(transport.sent, [])

    def test_no_reply_is_recorded_as_dismissed_with_a_reason(self):
        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter(reply="   ")
        adapter = self.adapter(transport, router)
        adapter.poll_once()
        transport.messages = [_addressed("2", "hello")]
        adapter.poll_once()
        record = adapter.inbox.get("2")
        self.assertEqual(record["state"], da.MSG_DISMISSED)
        self.assertTrue(record.get("reason"))

    def test_delivery_failure_is_recorded_as_failed(self):
        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter(reply="answer")
        adapter = self.adapter(transport, router)
        adapter.poll_once()
        transport.messages = [_addressed("2", "hello")]
        transport.fail_send = True

        def failing_send(channel_id, content, *, reference=None,
                         mention_user_id=None):
            return {"ok": False, "error": "http 500"}

        transport.send_message = failing_send
        adapter.poll_once()
        self.assertEqual(adapter.inbox.state_of("2"), da.MSG_FAILED)

    def test_unaddressed_lines_do_not_enter_the_inbox(self):
        transport = FakeTransport([_mention("1", "seed")])
        adapter = self.adapter(transport, FakeRouter())
        adapter.poll_once()
        transport.messages = [_plain("2", "talking to someone else")]
        adapter.poll_once()
        self.assertEqual(adapter.inbox.state_of("2"), None)


class TestDiagnostics(_Base):
    def test_diagnostics_report_the_lifecycle(self):
        transport = FakeTransport([_mention("1", "seed")])
        router = FakeRouter()
        adapter = self.adapter(transport, router)
        adapter.poll_once()
        transport.messages = [_addressed("2", "hi")]
        adapter.poll_once()
        diag = adapter.diagnostics()
        self.assertIn("last_received_at", diag)
        self.assertIn("last_processed_at", diag)
        self.assertIn("pending_messages", diag)
        self.assertIn("oldest_pending_age_seconds", diag)
        self.assertIn("last_reconcile_at", diag)
        self.assertEqual(diag["pending_messages"], 0)
        self.assertIsNotNone(diag["last_processed_at"])

    def test_pending_age_is_reported(self):
        inbox = da.MessageInbox(data_dir=self.tmp)
        inbox.receive("7", content="<@42> waiting", author_id="u1",
                      channel_id="c1", addressed=True)
        snap = inbox.snapshot()
        self.assertEqual(snap["pending"], 1)
        self.assertIsNotNone(snap["oldest_pending_age_seconds"])
        self.assertGreaterEqual(snap["oldest_pending_age_seconds"], 0.0)
        # The adapter's diagnostics surface the same facts, without a poll (which
        # would recover and answer the message).
        adapter = self.adapter(FakeTransport([]), FakeRouter())
        diag = adapter.diagnostics()
        self.assertEqual(diag["pending_messages"], 1)
        self.assertIsNotNone(diag["oldest_pending_age_seconds"])

    def test_diagnostics_never_mutates_the_inbox(self):
        transport = FakeTransport([_mention("1", "seed")])
        adapter = self.adapter(transport, FakeRouter())
        adapter.poll_once()
        transport.messages = [_addressed("2", "hi")]
        adapter.poll_once()
        before = dict(adapter.inbox.snapshot())
        adapter.diagnostics()
        after = dict(adapter.inbox.snapshot())
        self.assertEqual(before, after)


class TestBoundedInbox(_Base):
    def test_only_terminal_records_are_pruned(self):
        inbox = da.MessageInbox(data_dir=self.tmp)
        # A pending message must never be pruned away.
        inbox.receive("pending-1", content="<@42> keep me", author_id="u1",
                      channel_id="c1", addressed=True)
        inbox.mark("pending-1", da.MSG_PENDING)
        for i in range(da.INBOX_MAX_RECORDS + 20):
            inbox.receive(f"done-{i}", content="x", author_id="u1",
                          channel_id="c1", addressed=True)
            inbox.mark(f"done-{i}", da.MSG_RESPONDED)
        snap = inbox.snapshot()
        self.assertLessEqual(snap["total"], da.INBOX_MAX_RECORDS + 1)
        self.assertEqual(inbox.state_of("pending-1"), da.MSG_PENDING)


class TestReportRendering(unittest.TestCase):
    """The admin report must surface the durable lifecycle honestly."""

    def _report(self, diag):
        from astra.application import RuntimeController
        from types import SimpleNamespace

        class FakeDiscord:
            def diagnostics(self_inner):
                return diag

        # discord_report only needs ``_human_age`` from its instance.
        stub = SimpleNamespace(_human_age=RuntimeController._human_age)
        return RuntimeController.discord_report(stub, FakeDiscord())

    def test_report_shows_received_processed_and_pending(self):
        out = self._report({
            "connected": True, "last_fetch_ok_at": "2026-01-01T12:00:00+00:00",
            "last_proactive_at": None, "inbound_delivered": 3, "inbound_failed": 0,
            "last_received_at": "2026-01-01T11:59:00+00:00",
            "last_processed_at": "2026-01-01T11:59:30+00:00",
            "pending_messages": 2, "oldest_pending_age_seconds": 90.0,
            "inbox": {"deferred": 1}, "deliveries": {},
        })
        self.assertIn("last message received:", out)
        self.assertIn("last message processed:", out)
        self.assertIn("pending messages: 2", out)
        self.assertIn("1m", out)          # human age of the oldest pending
        self.assertIn("deferred messages: 1", out)

    def test_report_shows_reconciliation_state(self):
        out = self._report({
            "connected": True, "last_fetch_ok_at": None, "last_proactive_at": None,
            "inbound_delivered": 0, "inbound_failed": 0, "deliveries": {},
            "reconcile_needed": True,
        })
        self.assertIn("reconciliation: RUNNING", out)

    def test_report_shows_reconciliation_error(self):
        out = self._report({
            "connected": False, "last_fetch_ok_at": None, "last_proactive_at": None,
            "inbound_delivered": 0, "inbound_failed": 0, "deliveries": {},
            "last_reconcile_error": "http 503",
        })
        self.assertIn("reconciliation error: http 503", out)

    def test_report_without_lifecycle_fields_still_renders(self):
        # Backwards-compatible: an older diagnostics dict must not crash.
        out = self._report({
            "connected": False, "last_fetch_ok_at": None, "last_proactive_at": None,
            "inbound_delivered": 0, "inbound_failed": 0, "deliveries": {},
        })
        self.assertIn("Discord:", out)


if __name__ == "__main__":
    unittest.main()

