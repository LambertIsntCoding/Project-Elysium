"""Tests for the pure Discord behaviour helpers (addressing + typing pace).

These are the deterministic rules the adapter leans on: whether a message is
addressed to Astra, and how long a human would take to type a reply. They are
pure functions, so they are tested without a transport or a clock.
"""
import random
import unittest

from astra import discord_behavior as db


class TestAddressing(unittest.TestCase):
    def test_user_mention_addresses(self):
        self.assertTrue(db.is_addressed("<@42> hello there", bot_user_id="42"))

    def test_nickname_mention_form_addresses(self):
        self.assertTrue(db.is_addressed("<@!42> hi", bot_user_id="42"))

    def test_other_users_mention_does_not_address(self):
        self.assertFalse(db.is_addressed("<@99> hello", bot_user_id="42"))

    def test_plain_message_does_not_address(self):
        self.assertFalse(db.is_addressed("just chatting", bot_user_id="42"))

    def test_bot_name_at_addresses(self):
        self.assertTrue(db.is_addressed("@Astra what do you think?",
                                        bot_user_id="42", bot_name="Astra"))

    def test_bot_name_substring_does_not_address(self):
        # "@Astraea" must not count as addressing "@Astra".
        self.assertFalse(db.is_addressed("@Astraea hello",
                                         bot_user_id="42", bot_name="Astra"))

    def test_referenced_bot_addresses_without_a_mention(self):
        self.assertTrue(db.is_addressed("that's interesting",
                                        bot_user_id="42", referenced_bot=True))

    def test_empty_is_not_addressed(self):
        self.assertFalse(db.is_addressed("", bot_user_id="42"))
        self.assertFalse(db.is_addressed(None, bot_user_id="42"))


class TestReferencedBot(unittest.TestCase):
    def test_reply_to_bot_message(self):
        msg = {"referenced_message": {"author": {"id": "42"}}}
        self.assertTrue(db.referenced_bot(msg, "42"))

    def test_reply_to_someone_else(self):
        msg = {"referenced_message": {"author": {"id": "99"}}}
        self.assertFalse(db.referenced_bot(msg, "42"))

    def test_raw_reference_author_id(self):
        self.assertTrue(db.referenced_bot({"referenced_author_id": "42"}, "42"))

    def test_no_reference_no_bot_id(self):
        self.assertFalse(db.referenced_bot({}, None))
        self.assertFalse(db.referenced_bot({"referenced_message": {}}, "42"))


class TestTypingSeconds(unittest.TestCase):
    def test_scales_with_word_count(self):
        rng = random.Random(1)
        short = db.typing_seconds("a few words here", wpm=60.0, rng=rng,
                                  min_seconds=0.0, max_seconds=1000.0)
        long = db.typing_seconds(" ".join(["word"] * 60), wpm=60.0, rng=rng,
                                 min_seconds=0.0, max_seconds=1000.0)
        self.assertGreater(long, short)
        # 60 words at 60 wpm is ~60s; jitter is bounded to +/- a quarter.
        self.assertGreater(long, 40.0)
        self.assertLess(long, 80.0)

    def test_minimum_is_never_instant(self):
        self.assertGreaterEqual(
            db.typing_seconds("hi", wpm=45.0, rng=random.Random(0)),
            db.MIN_TYPING_SECONDS)

    def test_maximum_is_bounded(self):
        huge = " ".join(["word"] * 5000)
        self.assertLessEqual(
            db.typing_seconds(huge, wpm=45.0),
            db.MAX_TYPING_SECONDS)

    def test_is_seeded_and_reproducible(self):
        a = db.typing_seconds("some message", rng=random.Random(7))
        b = db.typing_seconds("some message", rng=random.Random(7))
        self.assertEqual(a, b)


class TestTypingSegments(unittest.TestCase):
    def test_short_reply_is_one_burst(self):
        self.assertEqual(db.typing_segments(2.0), [2.0])

    def test_segments_sum_to_total(self):
        for total in (5.0, 11.5, 30.0):
            segs = db.typing_segments(total, rng=random.Random(3))
            self.assertAlmostEqual(sum(segs), total, places=2)

    def test_long_reply_has_pauses(self):
        segs = db.typing_segments(20.0, rng=random.Random(5))
        self.assertGreater(len(segs), 1)

    def test_zero_returns_empty(self):
        self.assertEqual(db.typing_segments(0.0), [])

    def test_pauses_never_dominate(self):
        segs = db.typing_segments(12.0, rng=random.Random(2),
                                  pause_seconds=(5.0, 6.0))
        # Interleaved (burst, pause, burst, ...): typing bursts outnumber pauses.
        bursts = segs[0::2]
        self.assertGreaterEqual(len(bursts), 2)
        self.assertAlmostEqual(sum(segs), 12.0, places=2)


class TestTypingPlan(unittest.TestCase):
    def _total(self, plan):
        segs, before = plan
        return sum(segs) + before, segs, before

    def test_reproducible_with_seed(self):
        a = db.typing_plan("some reply here", rng=random.Random(11))
        b = db.typing_plan("some reply here", rng=random.Random(11))
        self.assertEqual(a, b)

    def test_total_is_bounded(self):
        for text in ("hi", " ".join(["word"] * 500)):
            total, segs, before = self._total(
                db.typing_plan(text, rng=random.Random(4),
                               min_seconds=1.5, max_seconds=45.0))
            self.assertGreaterEqual(total, 1.5 - 1e-6)
            self.assertLessEqual(total, 45.0 + 1e-6)
            self.assertGreaterEqual(before, 0.0)

    def test_empty_still_takes_the_minimum(self):
        total, _, _ = self._total(db.typing_plan("", rng=random.Random(0)))
        self.assertAlmostEqual(total, db.MIN_TYPING_SECONDS, places=3)

    def test_same_length_varies_widely(self):
        # The whole point: two identical-length messages should not land in a
        # narrow band. Sampling many seeds, the spread must be substantial.
        text = " ".join(["word"] * 20)
        totals = [sum(db.typing_plan(text, rng=random.Random(s))[0])
                  + db.typing_plan(text, rng=random.Random(s))[1]
                  for s in range(40)]
        self.assertGreater(max(totals) - min(totals), 8.0)

    def test_structure_changes_the_total(self):
        # Same word count, but long words + punctuation + breaks cost more.
        plain = " ".join(["cat"] * 24)
        gnarly = "extraordinarily " * 6 + "counterintuitively " * 6 + \
                 "internationalisation " * 6 + "misunderstanding " * 6
        plain_t = self._total(db.typing_plan(plain, rng=random.Random(3)))[0]
        gnarly_t = self._total(db.typing_plan(gnarly, rng=random.Random(3)))[0]
        self.assertGreaterEqual(db.words_in(plain), db.words_in(gnarly))
        self.assertGreater(gnarly_t, plain_t)

    def test_punctuation_adds_pauses(self):
        a = self._total(db.typing_plan("one two three four five six seven",
                                       rng=random.Random(9)))[0]
        b = self._total(db.typing_plan(
            "one. two. three. four. five. six. seven.", rng=random.Random(9)))[0]
        self.assertGreater(b, a)

    def test_before_send_is_a_real_beat_within_bounds(self):
        # With room to spare, the last-look-over pause is a genuine beat and the
        # whole plan still lands inside the ceiling.
        for s in range(20):
            segs, before = db.typing_plan(
                "a reply with several words in it", rng=random.Random(s),
                min_seconds=0.0, max_seconds=1000.0)
            self.assertGreaterEqual(before, 0.4 - 1e-9)
            self.assertLessEqual(before, 2.4 + 1e-9)
            self.assertLessEqual(sum(segs) + before, 1000.0)
            self.assertGreater(sum(segs), 0.0)

    def test_before_send_never_exceeds_the_ceiling(self):
        _, _, before = self._total(db.typing_plan(
            "short", rng=random.Random(1), min_seconds=1.5, max_seconds=1.5))
        # A total pinned to the floor cannot reserve a long before-send pause.
        self.assertLessEqual(before, 1.5)


class TestReadingSeconds(unittest.TestCase):
    def test_bounded_and_positive(self):
        s = db.reading_seconds("a short note")
        self.assertGreaterEqual(s, db.MIN_READ_SECONDS)
        self.assertLessEqual(s, db.MAX_READ_SECONDS)

    def test_longer_text_reads_longer(self):
        self.assertGreater(db.reading_seconds(" ".join(["w"] * 200)),
                           db.reading_seconds("w w"))


if __name__ == "__main__":
    unittest.main()
