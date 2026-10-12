"""Tests for the reply-emphasis policy and its use by the Discord adapter."""

import shutil
import tempfile
import unittest

from astra import reply_policy


class TestParse(unittest.TestCase):
    def test_plain_text_is_not_a_reply(self):
        clean, wants = reply_policy.parse("just a normal answer")
        self.assertFalse(wants)
        self.assertEqual(clean, "just a normal answer")

    def test_marker_requests_a_reply_and_is_stripped(self):
        clean, wants = reply_policy.parse(reply_policy.EMPHASIS_MARKER + "answering that")
        self.assertTrue(wants)
        self.assertEqual(clean, "answering that")
        self.assertNotIn(reply_policy.EMPHASIS_MARKER, clean)

    def test_marker_only_at_start(self):
        clean, wants = reply_policy.parse("look " + reply_policy.EMPHASIS_MARKER + "here")
        self.assertFalse(wants)
        self.assertIn(reply_policy.EMPHASIS_MARKER, clean)

    def test_empty_and_none(self):
        self.assertEqual(reply_policy.parse(None)[0], "")
        self.assertFalse(reply_policy.parse(None)[1])


class TestAdapterIntegration(unittest.TestCase):
    """The adapter only reply-attaches (and pings) when the response asks."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _msg(self, mid, content, author="u1"):
        return {"id": mid, "content": content,
                "channel_id": "c1", "author": {"id": author}}

    class _Transport:
        def __init__(self, messages):
            self.messages = messages
            self.sent = []
            self.typing_calls = []

        def fetch_messages(self, channel_id, *, after=None, limit=20):
            return {"ok": True, "data": list(self.messages)}

        def send_message(self, channel_id, content, *, reference=None,
                         mention_user_id=None):
            self.sent.append({"channel": channel_id, "content": content,
                              "reference": reference, "mention": mention_user_id})
            return {"ok": True, "data": {"id": "s1"}}

        def send_typing(self, channel_id):
            self.typing_calls.append(channel_id)
            return {"ok": True, "data": {}}

    class _Router:
        def __init__(self, reply):
            self.reply = reply

        def handle(self, text, *, source="cli"):
            return {"text": self.reply, "handled": True}

    def _adapter(self, transport, router):
        from astra.discord_adapter import DiscordAdapter
        return DiscordAdapter(transport, router, channel_id="c1",
                              data_dir=self.tmp, batch_seconds=0.0)

    def test_default_no_reply_and_no_ping(self):
        transport = self._Transport([self._msg("1", "seed")])
        adapter = self._adapter(transport, self._Router("a plain answer"))
        adapter.poll_once()
        transport.messages = [self._msg("2", "hi")]
        adapter.poll_once()
        self.assertEqual(len(transport.sent), 1)
        self.assertIsNone(transport.sent[0]["reference"])
        self.assertIsNone(transport.sent[0]["mention"])

    def test_emphasis_reply_and_ping(self):
        transport = self._Transport([self._msg("1", "seed")])
        reply = reply_policy.EMPHASIS_MARKER + "this point matters"
        adapter = self._adapter(transport, self._Router(reply))
        adapter.poll_once()
        transport.messages = [self._msg("2", "hi")]
        adapter.poll_once()
        self.assertEqual(len(transport.sent), 1)
        self.assertEqual(transport.sent[0]["reference"], "2")
        self.assertEqual(transport.sent[0]["mention"], "u1")
        self.assertIn("<@u1>", transport.sent[0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
