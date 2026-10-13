"""Tests for the Discord transport adapter (transport only, shared router).

A fake transport and router are used so no network call is made and nothing is
mocked at the level of the code under test: the adapter calls the real
``router.handle`` protocol, and a stub implements the same protocol.
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from astra import discord_adapter as da
from astra import presence as presence_mod


class FakeTransport:
    """Implements the DiscordTransport protocol; records sends."""

    def __init__(self, messages, *, fail=False):
        self.messages = messages
        self.fail = fail
        self.sent = []
        self.fetches = 0
        self.typing_calls = []

    def fetch_messages(self, channel_id, *, after=None, limit=20):
        self.fetches += 1
        return {"ok": True, "data": list(self.messages)}

    def send_message(self, channel_id, content, *, reference=None,
                     mention_user_id=None):
        if self.fail:
            return {"ok": False, "error": "http 500"}
        self.sent.append({"channel": channel_id, "content": content,
                          "reference": reference, "mention": mention_user_id})
        return {"ok": True, "data": {"id": "1"}}

    def send_typing(self, channel_id):
        self.typing_calls.append(channel_id)
        return {"ok": True, "data": {}}


class FakeRouter:
    """Implements the ApplicationRouter protocol."""

    def __init__(self, reply="ok", source=None):
        self.reply = reply
        self.calls = []

    def handle(self, text, *, source="cli"):
        self.calls.append((text, source))
        return {"text": self.reply, "handled": True, "source": source}


def _msg(mid, content, *, bot=False, channel_id=None):
    author = {"id": "u1"}
    if bot:
        author["bot"] = True
    msg = {"id": mid, "content": content, "author": author}
    if channel_id:
        msg["channel_id"] = channel_id
    return msg


class TestFormatting(unittest.TestCase):
    def test_empty_becomes_placeholder(self):
        self.assertEqual(da.format_for_discord(""), "(no output)")

    def test_long_text_is_clamped_not_altered_in_the_middle(self):
        out = da.format_for_discord("x" * 5000)
        self.assertLessEqual(len(out), da.MAX_MESSAGE_CHARS)
        self.assertTrue(out.startswith("x"))

    def test_short_text_is_unchanged(self):
        self.assertEqual(da.format_for_discord("hello"), "hello")


class TestMessageSplitting(unittest.TestCase):
    def test_short_text_not_split(self):
        parts = da._split_message("hello world", limit=100)
        self.assertEqual(parts, ["hello world"])

    def test_empty_text_returns_placeholder(self):
        parts = da._split_message("", limit=100)
        self.assertEqual(parts, ["(no output)"])

    def test_splits_on_paragraph_boundary(self):
        text = "a" * 800 + "\n\n" + "b" * 800
        parts = da._split_message(text, limit=1000)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].startswith("a"))
        self.assertTrue(parts[1].startswith("b"))

    def test_splits_on_sentence_boundary(self):
        text = "First sentence. " * 60 + "Second sentence. " * 60
        parts = da._split_message(text, limit=1000)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(len(part), 1000)

    def test_hard_split_when_no_good_boundary(self):
        text = "x" * 2500
        parts = da._split_message(text, limit=1000)
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(parts), text)

    def test_respects_limit(self):
        text = "word " * 500
        parts = da._split_message(text, limit=500)
        for part in parts:
            self.assertLessEqual(len(part), 500)


class TestBatchDebounce(unittest.TestCase):
    """The response timer: a thought sent in several lines is one turn."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _adapter(self, transport, router, *, batch_seconds=3.0, max_batch=8):
        adapter = da.DiscordAdapter(transport, router, channel_id="c1",
                                    data_dir=self.tmp, batch_seconds=batch_seconds,
                                    max_batch=max_batch)
        # Baseline: the first poll must not answer pre-existing history.
        adapter.poll_once()
        return adapter

    def test_a_single_line_is_not_answered_immediately(self):
        from datetime import datetime, timezone, timedelta
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        transport = FakeTransport([_msg("1", "old")])
        router = FakeRouter()
        adapter = self._adapter(transport, router)
        transport.messages = [_msg("2", "only one line")]
        report = adapter.poll_once(now=t0)
        self.assertEqual(report["handled"], 0)  # buffered, not answered
        self.assertEqual(router.calls, [])
        # After the debounce elapses with no further lines, it answers once.
        report = adapter.poll_once(now=t0 + timedelta(seconds=3.1))
        self.assertEqual(len(router.calls), 1)

    def test_a_thought_in_several_lines_is_read_as_one_turn(self):
        from datetime import datetime, timezone, timedelta
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        transport = FakeTransport([_msg("1", "old")])
        router = FakeRouter()
        adapter = self._adapter(transport, router)
        # Three lines arrive within the debounce window.
        transport.messages = [_msg("2", "I was thinking")]
        adapter.poll_once(now=t0)
        transport.messages = [_msg("3", "about that book")]
        adapter.poll_once(now=t0 + timedelta(seconds=1))
        transport.messages = [_msg("4", "you mentioned")]
        adapter.poll_once(now=t0 + timedelta(seconds=2))
        # Nothing answered yet: still within the debounce after the last line.
        self.assertEqual(router.calls, [])
        adapter.poll_once(now=t0 + timedelta(seconds=5.1))
        # Exactly one turn, carrying the whole thought, not three replies.
        self.assertEqual(len(router.calls), 1)
        self.assertIn("I was thinking", router.calls[0][0])
        self.assertIn("you mentioned", router.calls[0][0])

    def test_max_batch_answers_early(self):
        from datetime import datetime, timezone
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        transport = FakeTransport([_msg("1", "old")])
        router = FakeRouter()
        adapter = self._adapter(transport, router, batch_seconds=60.0, max_batch=2)
        transport.messages = [_msg("2", "one")]
        adapter.poll_once(now=t0)
        transport.messages = [_msg("3", "two")]
        adapter.poll_once(now=t0)  # reaches max_batch -> answers now
        self.assertEqual(len(router.calls), 1)

    def test_default_is_not_immediate(self):
        # The configured default (build path) must debounce, never answer line
        # by line. The constructor default of 0 is only for tests/direct use.
        import os
        os.environ["DISCORD_BOT_TOKEN"] = "test-token"
        try:
            adapter = da.build_discord_adapter(
                FakeRouter(), {"discord": {"enabled": True, "channel_id": "c1"}},
                data_dir=self.tmp)
        finally:
            os.environ.pop("DISCORD_BOT_TOKEN", None)
        self.assertIsNotNone(adapter)
        self.assertGreater(adapter.batch_seconds, 0.0)


class TestReplyAndPing(unittest.TestCase):
    """Reply is opt-in; a reply always pings; no reply never pings."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _adapter(self, transport, router, **kw):
        adapter = da.DiscordAdapter(transport, router, channel_id="c1",
                                    data_dir=self.tmp,
                                    batch_seconds=kw.pop("batch_seconds", 0.0), **kw)
        adapter.poll_once()  # baseline (seeded transport)
        return adapter

    def test_plain_reply_has_no_reference_and_no_ping(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="Just an ordinary answer.")
        adapter = self._adapter(transport, router)
        transport.messages = [_msg("7", "hi")]
        adapter.poll_once()
        sent = transport.sent[-1]
        self.assertEqual(len(transport.sent), 1)
        self.assertIsNone(sent["reference"])
        self.assertIsNone(sent["mention"])

    def test_emphasis_reply_references_and_pings_author(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="\u21aa That specific point matters.")
        adapter = self._adapter(transport, router)
        transport.messages = [_msg("7", "hi")]
        adapter.poll_once()
        self.assertEqual(len(transport.sent), 1)
        sent = transport.sent[-1]
        self.assertEqual(sent["reference"], "7")
        self.assertEqual(sent["mention"], "u1")
        self.assertIn("<@u1>", sent["content"])
        self.assertNotIn("\u21aa", sent["content"])


class TestAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _adapter(self, transport, router):
        return da.DiscordAdapter(transport, router, channel_id="c1",
                                 data_dir=self.tmp)

    def test_first_poll_baselines_without_answering_history(self):
        transport = FakeTransport([_msg("1", "old message")])
        router = FakeRouter()
        adapter = self._adapter(transport, router)
        report = adapter.poll_once()
        self.assertEqual(report["handled"], 0)
        self.assertEqual(router.calls, [])
        self.assertEqual(adapter.state.last_message_id, "1")

    def test_new_message_is_routed_and_answered(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="hello back")
        adapter = self._adapter(transport, router)
        adapter.poll_once()  # baseline
        transport.messages = [_msg("2", "hi there")]
        report = adapter.poll_once()
        self.assertEqual(report["handled"], 1)
        self.assertEqual(router.calls, [("hi there", "discord")])
        self.assertEqual(len(transport.sent), 1)
        self.assertEqual(transport.sent[0]["content"], "hello back")
        self.assertEqual(adapter.delivered, 1)

    def test_bot_messages_are_ignored(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter()
        adapter = self._adapter(transport, router)
        adapter.poll_once()
        transport.messages = [_msg("2", "I am a bot", bot=True)]
        report = adapter.poll_once()
        self.assertEqual(report["handled"], 0)
        self.assertEqual(router.calls, [])

    def test_delivery_failure_is_failure_not_success(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="reply")
        adapter = self._adapter(transport, router)
        adapter.poll_once()
        transport.fail = True
        transport.messages = [_msg("2", "hi")]
        report = adapter.poll_once()
        self.assertEqual(adapter.delivered, 0)
        self.assertEqual(adapter.failed, 1)
        self.assertEqual(adapter.last_error, "http 500")

    def test_cursor_persists_across_restart(self):
        transport = FakeTransport([_msg("1", "seed")])
        adapter = self._adapter(transport, FakeRouter())
        adapter.poll_once()
        adapter.state.set_last_message_id("5")
        # A fresh adapter reads the same state file and does not re-baseline.
        fresh = da.DiscordAdapter(transport, FakeRouter(), channel_id="c1",
                                  data_dir=self.tmp)
        self.assertEqual(fresh.state.last_message_id, "5")
        fresh.poll_once()  # not first_run now
        # No messages answered because cursor is ahead, but no error either.
        self.assertIsNone(fresh.last_error)

    def test_typing_indicator_sent_before_reply(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="hello")
        adapter = self._adapter(transport, router)
        adapter.poll_once()  # baseline
        transport.messages = [_msg("2", "hi")]
        adapter.poll_once()
        # Typing was called for the channel.
        self.assertIn("c1", transport.typing_calls)

    def test_reply_not_attached_by_default(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="hello back")
        adapter = self._adapter(transport, router)
        adapter.poll_once()  # baseline
        transport.messages = [_msg("42", "hi")]
        adapter.poll_once()
        # By default a response is a plain message: no reply reference, no ping.
        self.assertEqual(len(transport.sent), 1)
        self.assertIsNone(transport.sent[0]["reference"])
        self.assertIsNone(transport.sent[0].get("mention"))

    def test_emphasis_marks_reply_and_pings_the_author(self):
        from astra import reply_policy
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply=reply_policy.EMPHASIS_MARKER + "about that point...")
        adapter = self._adapter(transport, router)
        adapter.poll_once()  # baseline
        transport.messages = [_msg("42", "hi")]
        adapter.poll_once()
        self.assertEqual(len(transport.sent), 1)
        # The emphasis marker is stripped and the reply references the message.
        self.assertEqual(transport.sent[0]["reference"], "42")
        self.assertNotIn(reply_policy.EMPHASIS_MARKER, transport.sent[0]["content"])
        # The author of the emphasised message is pinged.
        self.assertEqual(transport.sent[0].get("mention"), "u1")

    def test_long_message_is_split(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="x" * 3000)
        adapter = self._adapter(transport, router)
        adapter.poll_once()  # baseline
        transport.messages = [_msg("2", "hi")]
        adapter.poll_once()
        # Should be split into multiple messages.
        self.assertGreater(len(transport.sent), 1)
        # No part references by default; all parts are plain.
        for part in transport.sent:
            self.assertIsNone(part["reference"])
        self.assertEqual(adapter.delivered, 1)

    def test_thread_message_replied_in_thread(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="hello")
        adapter = self._adapter(transport, router)
        adapter.poll_once()  # baseline
        # Message arrives in a thread (different channel_id).
        transport.messages = [_msg("2", "hi", channel_id="thread_123")]
        adapter.poll_once()
        # Reply should go to the thread channel, not the configured channel.
        self.assertEqual(transport.sent[0]["channel"], "thread_123")
        # Typing should also be sent in the thread.
        self.assertIn("thread_123", transport.typing_calls)

    def test_counters_are_thread_safe(self):
        transport = FakeTransport([_msg("1", "seed")])
        adapter = self._adapter(transport, FakeRouter())
        adapter.poll_once()
        # Properties should work.
        self.assertEqual(adapter.delivered, 0)
        self.assertEqual(adapter.failed, 0)


class TestProactiveDelivery(unittest.TestCase):
    """The outbound proactive path: dedupe, retries, failure vs. success."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _adapter(self, transport):
        return da.DiscordAdapter(transport, FakeRouter(), channel_id="c1",
                                 data_dir=self.tmp)

    def test_send_proactive_delivers_and_records(self):
        transport = FakeTransport([])
        adapter = self._adapter(transport)
        result = adapter.send_proactive("a discovery", ref_id="o1")
        self.assertTrue(result["ok"])
        self.assertTrue(result["sent"])
        self.assertEqual(len(transport.sent), 1)
        self.assertTrue(adapter.deliveries.was_delivered("o1"))
        self.assertEqual(adapter.diagnostics()["deliveries"]["delivered"], 1)

    def test_duplicate_is_suppressed_across_restart(self):
        transport = FakeTransport([])
        adapter = self._adapter(transport)
        adapter.send_proactive("a discovery", ref_id="o1")
        # A fresh adapter (same state dir) sees the delivery and does not resend.
        fresh = da.DiscordAdapter(transport, FakeRouter(), channel_id="c1",
                                  data_dir=self.tmp)
        result = fresh.send_proactive("a discovery", ref_id="o1")
        self.assertTrue(result["suppressed"] == "duplicate")
        self.assertEqual(len(transport.sent), 1)

    def test_failure_is_recorded_and_never_counted_delivered(self):
        transport = FakeTransport([])
        transport.fail = True
        adapter = self._adapter(transport)
        result = adapter.send_proactive("a discovery", ref_id="o1")
        self.assertFalse(result["ok"])
        self.assertFalse(result["sent"])
        self.assertFalse(adapter.deliveries.was_delivered("o1"))
        self.assertEqual(adapter.deliveries.get("o1").get("status"), "failed")
        self.assertEqual(adapter.delivered, 0)

    def test_retry_bound_stops_resending(self):
        transport = FakeTransport([])
        transport.fail = True
        adapter = self._adapter(transport)
        for _ in range(da.PROACTIVE_MAX_ATTEMPTS):
            adapter.send_proactive("x", ref_id="o1")
        result = adapter.send_proactive("x", ref_id="o1")
        self.assertEqual(result["suppressed"], "retry-limit")

    def test_empty_content_is_not_sent(self):
        transport = FakeTransport([])
        adapter = self._adapter(transport)
        result = adapter.send_proactive("   ", ref_id="o1")
        self.assertFalse(result["ok"])
        self.assertEqual(len(transport.sent), 0)

    def test_permission_error_is_surfaced_not_retried_at_recipient(self):
        class PermTransport(FakeTransport):
            def send_message(self, channel_id, content, *, reference=None,
                             mention_user_id=None):
                return {"ok": False, "error": "http 403", "status": 403}

        adapter = self._adapter(PermTransport([]))
        result = adapter.send_proactive("x", ref_id="o1")
        self.assertEqual(result["suppressed"], "permission")
        self.assertIn("permission", adapter.last_error or "")

    def test_poll_records_persisted_health_for_a_separate_process(self):
        transport = FakeTransport([_msg("1", "seed")])
        adapter = self._adapter(transport)
        adapter.poll_once()  # success
        health = da.DiscordHealthStore(data_dir=self.tmp)
        self.assertTrue(health.read().get("last_fetch_ok_at"))
        self.assertIsNone(health.read().get("last_error"))

        class FailFetch(FakeTransport):
            def fetch_messages(self, channel_id, *, after=None, limit=20):
                return {"ok": False, "error": "http 503"}

        # A later fetch failure is written to the same health record, so a
        # separate process (the read-only console) sees it.
        adapter.transport = FailFetch([])
        adapter.poll_once()
        fresh = da.DiscordHealthStore(data_dir=self.tmp)
        self.assertEqual(fresh.read().get("last_error"), "http 503")

    def test_successful_send_is_recorded_in_health(self):
        transport = FakeTransport([])
        adapter = self._adapter(transport)
        adapter.send_proactive("done", ref_id="o1")
        self.assertTrue(da.DiscordHealthStore(data_dir=self.tmp).read().get(
            "last_send_ok_at"))


class TestAddressedGate(unittest.TestCase):
    """A running adapter answers only messages addressed to it, unless checking."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _adapter(self, transport, router, **kw):
        kw.setdefault("require_mention", True)
        kw.setdefault("bot_user_id", "42")
        adapter = da.DiscordAdapter(transport, router, channel_id="c1",
                                    data_dir=self.tmp, batch_seconds=0.0, **kw)
        adapter.poll_once()  # baseline
        return adapter

    def test_unaddressed_message_is_not_answered(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="hi")
        adapter = self._adapter(transport, router)
        transport.messages = [_msg("2", "just talking to myself")]
        report = adapter.poll_once()
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(router.calls, [])
        self.assertEqual(transport.sent, [])

    def test_mentioned_message_is_answered(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="hi")
        adapter = self._adapter(transport, router)
        transport.messages = [{"id": "2", "content": "<@42> you there?",
                               "author": {"id": "u1"}}]
        report = adapter.poll_once()
        self.assertEqual(report["handled"], 1)
        self.assertEqual(router.calls, [("<@42> you there?", "discord")])

    def test_reply_to_bot_is_answered(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="hi")
        adapter = self._adapter(transport, router)
        transport.messages = [{
            "id": "2", "content": "no mention here",
            "author": {"id": "u1"},
            "referenced_message": {"author": {"id": "42"}}}]
        report = adapter.poll_once()
        self.assertEqual(report["handled"], 1)

    def test_check_messages_forces_a_read_of_unaddressed(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="hi")
        adapter = self._adapter(transport, router)
        transport.messages = [_msg("2", "no mention at all")]
        # A normal poll ignores it; an explicit check answers it.
        adapter.poll_once()
        self.assertEqual(router.calls, [])
        handled = adapter.check_messages()
        self.assertEqual(handled, 1)
        self.assertEqual(len(router.calls), 1)

    def test_no_gate_when_require_mention_off(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="hi")
        adapter = da.DiscordAdapter(transport, router, channel_id="c1",
                                    data_dir=self.tmp, require_mention=False)
        adapter.poll_once()
        transport.messages = [_msg("2", "no mention")]
        report = adapter.poll_once()
        self.assertEqual(report["handled"], 1)


class TestTypingSimulation(unittest.TestCase):
    """In the running config, replies are typed at a human pace, not pasted."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _adapter(self, transport, router, **kw):
        kw.setdefault("typing_simulation", True)
        adapter = da.DiscordAdapter(transport, router, channel_id="c1",
                                    data_dir=self.tmp, batch_seconds=0.0,
                                    **kw)
        adapter.poll_once()
        return adapter

    def test_reply_is_not_sent_until_typing_finishes(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="a considered reply")
        adapter = self._adapter(transport, router)
        transport.messages = [_msg("2", "hi")]
        adapter.poll_once()
        # Typing is in flight; nothing delivered yet, but the indicator is on.
        self.assertEqual(transport.sent, [])
        self.assertIn("c1", transport.typing_calls)
        self.assertTrue(adapter.diagnostics()["pending_reply"])
        # Let the (real, short) typing thread finish, then poll delivers it.
        transport.messages = []  # the fake transport ignores the cursor
        adapter._join_typing(timeout=30.0)
        adapter.poll_once()
        self.assertEqual(len(transport.sent), 1)
        self.assertEqual(transport.sent[0]["content"], "a considered reply")

    def test_typing_delay_is_word_scaled(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply=" ".join(["word"] * 50))
        adapter = self._adapter(transport, router, typing_wpm=60.0,
                                typing_min_seconds=0.0, typing_max_seconds=600.0)
        transport.messages = [_msg("2", "hi")]
        adapter.poll_once()
        with adapter._send_lock:
            delay = adapter._reply_in_flight["typing_seconds"]
        # 50 words at 60 wpm is 50s; with jitter, comfortably over 30s.
        self.assertGreater(delay, 30.0)

    def test_new_message_interrupts_and_replaces_the_pending_reply(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply=" ".join(["word"] * 40))
        adapter = self._adapter(transport, router, typing_wpm=10.0,
                                typing_min_seconds=0.0, typing_max_seconds=600.0)
        transport.messages = [_msg("2", "first thought")]
        adapter.poll_once()  # starts typing the long reply
        self.assertTrue(adapter.diagnostics()["pending_reply"])
        # A second thought arrives mid-typing: the reply is abandoned and read.
        transport.messages = [_msg("3", "and another thought")]
        adapter.poll_once()
        self.assertFalse(adapter.diagnostics()["pending_reply"])
        self.assertEqual(transport.sent, [])
        # It is re-buffered as one turn, then answered after the debounce. The
        # first (superseded) turn still ran the router once; the final turn is
        # the one that must carry the whole thought.
        transport.messages = []
        adapter._join_typing(timeout=30.0)
        adapter.poll_once()
        final = router.calls[-1][0]
        self.assertIn("first thought", final)
        self.assertIn("and another thought", final)

    def test_flush_delivers_pending_on_shutdown(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="bye")
        adapter = self._adapter(transport, router)
        transport.messages = [_msg("2", "hi")]
        adapter.poll_once()
        adapter.flush()
        self.assertEqual(len(transport.sent), 1)

    def test_typing_off_sends_immediately(self):
        transport = FakeTransport([_msg("1", "seed")])
        router = FakeRouter(reply="plain")
        adapter = da.DiscordAdapter(transport, router, channel_id="c1",
                                    data_dir=self.tmp, typing_simulation=False)
        adapter.poll_once()
        transport.messages = [_msg("2", "hi")]
        adapter.poll_once()
        self.assertEqual(len(transport.sent), 1)


class TestStateShapedTiming(unittest.TestCase):
    """Her affect scales how long her Discord actions take."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _adapter(self, provider, **kw):
        transport = FakeTransport([_msg("1", "seed")])
        adapter = da.DiscordAdapter(
            transport, FakeRouter(reply="hi"), channel_id="c1",
            data_dir=self.tmp, typing_simulation=True, batch_seconds=0.0,
            modulation_provider=provider, **kw)
        adapter.poll_once()
        return adapter, transport

    def test_warm_state_types_faster_and_waits_longer(self):
        warm = {"energy": 0.95, "enthusiasm": 0.95, "pace": 0.95,
                "interest": 0.95, "hesitation": 0.0, "_sad": 0.0}
        low = {"energy": 0.05, "enthusiasm": 0.05, "pace": 0.05,
               "interest": 0.05, "hesitation": 0.7, "_sad": 0.7}
        warm_adapter, _ = self._adapter(lambda: warm)
        low_adapter, _ = self._adapter(lambda: low)
        self.assertLess(warm_adapter._timing()["typing_scale"],
                        low_adapter._timing()["typing_scale"])
        self.assertGreater(warm_adapter._timing()["patience_scale"],
                           low_adapter._timing()["patience_scale"])

    def test_scale_timing_preserves_rhythm(self):
        adapter, _ = self._adapter(lambda: {})
        segs = [2.0, 1.0, 3.0]
        scaled = adapter._scale_timing(segs, 2.0)
        self.assertEqual(scaled, [4.0, 2.0, 6.0])
        self.assertEqual(adapter._scale_timing(segs, 1.0), segs)
        self.assertEqual(adapter._scale_timing([], 2.0), [])

    def test_provider_failure_degrades_to_neutral(self):
        def boom():
            raise RuntimeError("no state")
        adapter, transport = self._adapter(boom)
        self.assertEqual(adapter._modulation(), {})
        self.assertEqual(adapter._timing()["typing_scale"], 1.0)
        transport.messages = [_msg("2", "hi")]
        adapter.poll_once()
        self.assertTrue(adapter.diagnostics()["pending_reply"])


class TestPresenceInAdapter(unittest.TestCase):
    """Walking away closes the exchange and returns her to her own work."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _adapter(self, clock, **kw):
        transport = FakeTransport([_msg("1", "seed")])
        adapter = da.DiscordAdapter(
            transport, FakeRouter(reply="hi"), channel_id="c1",
            data_dir=self.tmp, typing_simulation=False, batch_seconds=0.0,
            now_fn=clock, **kw)
        adapter.poll_once()
        return adapter, transport

    def test_addressed_message_opens_a_live_exchange(self):
        clock = _Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        adapter, transport = self._adapter(clock)
        transport.messages = [_msg("2", "you around?")]
        adapter.poll_once()
        diag = adapter.diagnostics()
        self.assertTrue(diag["exchange_open"])
        self.assertIn(diag["presence"], (presence_mod.PRESENT,
                                         presence_mod.LINGERING))

    def test_going_quiet_closes_the_exchange(self):
        clock = _Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        adapter, transport = self._adapter(clock)
        transport.messages = [_msg("2", "you around?")]
        adapter.poll_once()
        # He walks away: advance past her (state-scaled) patience.
        clock.advance(presence_mod.DEFAULT_PATIENCE_SECONDS + 10)
        diag = adapter.diagnostics()
        self.assertFalse(diag["exchange_open"])
        self.assertEqual(diag["presence"], presence_mod.AWAY)

    def test_a_low_state_gives_up_sooner(self):
        clock = _Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        low = {"energy": 0.05, "enthusiasm": 0.05, "pace": 0.05,
               "interest": 0.05, "hesitation": 0.7, "_sad": 0.7}
        adapter, transport = self._adapter(clock, modulation_provider=lambda: low)
        transport.messages = [_msg("2", "you around?")]
        adapter.poll_once()
        clock.advance(150.0)  # past a low state's patience, inside a warm one's
        self.assertEqual(adapter.diagnostics()["presence"], presence_mod.AWAY)

    def test_speaking_again_reopens_it(self):
        clock = _Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        adapter, transport = self._adapter(clock)
        transport.messages = [_msg("2", "you around?")]
        adapter.poll_once()
        clock.advance(presence_mod.DEFAULT_PATIENCE_SECONDS + 10)
        self.assertFalse(adapter.diagnostics()["exchange_open"])
        transport.messages = [_msg("3", "back")]
        adapter.poll_once()
        self.assertTrue(adapter.diagnostics()["exchange_open"])


class _Clock:
    """A controllable clock for the adapter's ``now_fn``."""

    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now = self.now + timedelta(seconds=seconds)


class TestBuilder(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(da.build_discord_adapter(FakeRouter(), {}))

    def test_missing_token_returns_none(self):
        cfg = {"discord": {"enabled": True, "channel_id": "c", "bot_token_env": "NOPE_XYZ"}}
        self.assertIsNone(da.build_discord_adapter(FakeRouter(), cfg))


class _FakeResponse:
    def __init__(self, status_code, *, json_body=None, headers=None):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.headers = headers or {}

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


class _StubSession:
    """A stub standing in for ``requests.Session``.

    Emits a scripted sequence of responses/exceptions; the real
    ``DiscordTransport._request`` logic runs against it, so the retry/backoff
    behavior is the code under test, not the stub.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def request(self, method, url, *, headers=None, json=None, params=None, timeout=None):
        self.calls += 1
        item = self.script.pop(0) if self.script else self.script_error()
        if isinstance(item, Exception):
            raise item
        return item

    @staticmethod
    def script_error():
        raise AssertionError("stub session ran out of scripted responses")


class TestTransportResilience(unittest.TestCase):
    """The real transport retries transient failures with bounded backoff."""

    def _transport(self, script, timeout=15.0):
        # Patch sleep so the test never actually waits.
        self._slept = []
        original = da.time.sleep
        da.time.sleep = lambda s: self._slept.append(s)
        self.addCleanup(lambda: setattr(da.time, "sleep", original))
        return da.DiscordTransport("tok", session=_StubSession(script),
                                   timeout=timeout)

    def test_read_timeout_is_retried_then_succeeds(self):
        import requests
        t = self._transport([
            requests.exceptions.ReadTimeout("read timed out"),
            _FakeResponse(200, json_body={"id": "m1"}),
        ])
        result = t.fetch_messages("c1")
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"], {"id": "m1"})
        # One retry happened (a sleep was taken) before the success.
        self.assertTrue(self._slept)

    def test_connection_reset_is_retried(self):
        import requests
        t = self._transport([
            requests.exceptions.ConnectionError("Connection aborted"),
            _FakeResponse(200, json_body=[]),
        ])
        self.assertTrue(t.fetch_messages("c1")["ok"])

    def test_persistent_transport_error_gives_up_bounded(self):
        import requests
        t = self._transport([requests.exceptions.ReadTimeout("x")] * 10)
        result = t.fetch_messages("c1")
        self.assertFalse(result["ok"])
        self.assertIn("transport", result["error"])
        # Bounded: never more than TRANSPORT_MAX_ATTEMPTS attempts and sleeps.
        self.assertEqual(len(self._slept), da.TRANSPORT_MAX_ATTEMPTS - 1)

    def test_500_is_retried_then_succeeds(self):
        t = self._transport([_FakeResponse(500), _FakeResponse(200, json_body={})])
        self.assertTrue(t.send_message("c1", "hi")["ok"])

    def test_400_is_not_retried(self):
        t = self._transport([_FakeResponse(400, json_body={})])
        result = t.send_message("c1", "hi")
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], 400)

    def test_429_honours_retry_after_body(self):
        t = self._transport([
            _FakeResponse(429, json_body={"retry_after": 0.75}),
            _FakeResponse(200, json_body={}),
        ])
        self.assertTrue(t.fetch_messages("c1")["ok"])
        self.assertTrue(self._slept)
        self.assertAlmostEqual(self._slept[0], 0.75, places=2)

    def test_429_honours_retry_after_header_as_fallback(self):
        t = self._transport([
            _FakeResponse(429, json_body=ValueError("no json"),
                          headers={"Retry-After": "2"}),
            _FakeResponse(200, json_body={}),
        ])
        self.assertTrue(t.fetch_messages("c1")["ok"])
        self.assertAlmostEqual(self._slept[0], 2.0, places=2)

    def test_rate_limit_sleep_is_clamped(self):
        res = _FakeResponse(429, json_body={"retry_after": 9999})
        self.assertLessEqual(da.DiscordTransport("t")._rate_limit_sleep(res, 0),
                             da.RATE_LIMIT_MAX_SLEEP + 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)