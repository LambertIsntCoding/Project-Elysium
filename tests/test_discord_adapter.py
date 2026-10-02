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

    def fetch_messages(self, channel_id, *, after=None, limit=20):
        self.fetches += 1
        return {"ok": True, "data": list(self.messages)}

    def send_message(self, channel_id, content):
        if self.fail:
            return {"ok": False, "error": "http 500"}
        self.sent.append(content)
        return {"ok": True, "data": {"id": "1"}}


class FakeRouter:
    """Implements the ApplicationRouter protocol."""

    def __init__(self, reply="ok", source=None):
        self.reply = reply
        self.calls = []

    def handle(self, text, *, source="cli"):
        self.calls.append((text, source))
        return {"text": self.reply, "handled": True, "source": source}


def _msg(mid, content, *, bot=False):
    author = {"id": "u1"}
    if bot:
        author["bot"] = True
    return {"id": mid, "content": content, "author": author}


class TestFormatting(unittest.TestCase):
    def test_empty_becomes_placeholder(self):
        self.assertEqual(da.format_for_discord(""), "(no output)")

    def test_long_text_is_clamped_not_altered_in_the_middle(self):
        out = da.format_for_discord("x" * 5000)
        self.assertLessEqual(len(out), da.MAX_MESSAGE_CHARS)
        self.assertTrue(out.startswith("x"))

    def test_short_text_is_unchanged(self):
        self.assertEqual(da.format_for_discord("hello"), "hello")


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
        self.assertEqual(transport.sent, ["hello back"])
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


class TestBuilder(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(da.build_discord_adapter(FakeRouter(), {}))

    def test_missing_token_returns_none(self):
        cfg = {"discord": {"enabled": True, "channel_id": "c", "bot_token_env": "NOPE_XYZ"}}
        self.assertIsNone(da.build_discord_adapter(FakeRouter(), cfg))


if __name__ == "__main__":
    unittest.main(verbosity=2)
