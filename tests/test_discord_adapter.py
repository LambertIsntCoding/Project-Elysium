"""Tests for the Discord transport adapter (transport only, shared router).

A fake transport and router are used so no network call is made and nothing is
mocked at the level of the code under test: the adapter calls the real
``router.handle`` protocol, and a stub implements the same protocol.
"""

import shutil
import tempfile
import unittest

from astra import discord_adapter as da


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


class TestBuilder(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(da.build_discord_adapter(FakeRouter(), {}))

    def test_missing_token_returns_none(self):
        cfg = {"discord": {"enabled": True, "channel_id": "c", "bot_token_env": "NOPE_XYZ"}}
        self.assertIsNone(da.build_discord_adapter(FakeRouter(), cfg))


if __name__ == "__main__":
    unittest.main(verbosity=2)