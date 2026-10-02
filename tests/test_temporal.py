"""Tests for Astra's sense of time (continuity, not timestamps).

The behaviours the brief makes non-negotiable: elapsed time is *reported* and
never *prescribed*; a gap does not automatically become an emotion; a prompt
build does not erase the gap it is reporting; and no history means no block.
"""
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from astra import temporal
from astra.memory import TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)


class TestSpanPhrase(unittest.TestCase):
    def test_spans_read_as_durations(self):
        self.assertEqual(temporal.elapsed_span_phrase(30), "moments")
        self.assertEqual(temporal.elapsed_span_phrase(6 * 3600), "about 6 hours")
        self.assertEqual(temporal.elapsed_span_phrase(2 * 86400), "about 2 days")
        self.assertEqual(temporal.elapsed_span_phrase(21 * 86400), "about 3 weeks")
        self.assertEqual(temporal.elapsed_span_phrase(300 * 86400), "about 10 months")
        self.assertEqual(temporal.elapsed_span_phrase(400 * 86400), "about 1 year")

    def test_missing_or_negative_is_none(self):
        self.assertIsNone(temporal.elapsed_span_phrase(None))
        self.assertIsNone(temporal.elapsed_span_phrase(-5))


class TestGapLine(unittest.TestCase):
    def test_short_gap_is_not_named(self):
        self.assertIsNone(temporal.gap_line((NOW - timedelta(hours=2)).isoformat(), NOW))

    def test_notable_gap_is_reported_as_a_fact(self):
        line = temporal.gap_line((NOW - timedelta(hours=9)).isoformat(), NOW)
        self.assertIn("about 9 hours", line)
        # Descriptive, not prescribed: it explicitly disclaims a mandated feeling.
        self.assertIn("not a feeling", line)

    def test_missing_history_is_none(self):
        self.assertIsNone(temporal.gap_line(None, NOW))
        self.assertIsNone(temporal.gap_line("not-a-date", NOW))


class TestQuestionsAndProjects(unittest.TestCase):
    def test_only_long_open_questions_appear(self):
        recent = {"content": "Why is the sky blue?", "timestamp": (NOW - timedelta(days=2)).isoformat()}
        old = {"content": "What makes a self?", "timestamp": (NOW - timedelta(days=40)).isoformat()}
        lines = temporal.open_question_lines([recent, old], now=NOW)
        self.assertEqual(len(lines), 1)
        self.assertIn("What makes a self?", lines[0])
        self.assertIn("about 1 month", lines[0])

    def test_opened_at_is_preferred_over_timestamp(self):
        q = {
            "content": "Long question",
            "opened_at": (NOW - timedelta(days=100)).isoformat(),
            "timestamp": NOW.isoformat(),
        }
        self.assertEqual(len(temporal.open_question_lines([q], now=NOW)), 1)

    def test_only_long_projects_appear(self):
        works = [
            {"title": "Fresh", "started_at": (NOW - timedelta(days=1)).isoformat()},
            {"title": "Marathon", "started_at": (NOW - timedelta(days=30)).isoformat()},
        ]
        lines = temporal.project_lines(works, now=NOW)
        self.assertEqual(len(lines), 1)
        self.assertIn("Marathon", lines[0])


class TestTemporalBlock(unittest.TestCase):
    def test_no_history_means_no_block(self):
        self.assertIsNone(temporal.temporal_prompt_block(now=NOW))

    def test_block_is_descriptive_not_prescriptive(self):
        block = temporal.temporal_prompt_block(
            last_seen=(NOW - timedelta(days=3)).isoformat(), now=NOW,
        )
        self.assertIn("THE PASSAGE OF TIME", block)
        self.assertIn("not instructions about how to feel", block)
        self.assertIn("about 3 days", block)

    def test_block_is_bounded(self):
        questions = [
            {"content": f"Q{i}", "timestamp": (NOW - timedelta(days=100 + i)).isoformat()}
            for i in range(10)
        ]
        block = temporal.temporal_prompt_block(
            last_seen=(NOW - timedelta(days=2)).isoformat(),
            open_questions=questions, now=NOW,
        )
        # At most three long questions are listed.
        self.assertEqual(block.count("- \"Q"), 3)


class TestPresencePersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_presence_round_trips(self):
        self.assertIsNone(self.store.last_present())
        stamp = self.store.mark_present()
        self.assertEqual(self.store.last_present(), stamp)

    def test_presence_survives_reload(self):
        stamp = self.store.mark_present()
        reopened = TripleMemoryStore(data_dir=self.tmp)
        self.assertEqual(reopened.last_present(), stamp)

    def test_presence_file_is_plain_json(self):
        self.store.mark_present()
        data = json.load(open(os.path.join(self.tmp, "presence.json")))
        self.assertIn("last_present", data)


class TestImpatience(unittest.TestCase):
    def _question(self, content, stall_days, **extra):
        ts = (NOW - timedelta(days=stall_days)).isoformat()
        mem = {"content": content, "type": "open_question", "timestamp": ts,
               "opened_at": ts, "last_progress_at": ts}
        mem.update(extra)
        return mem

    def _work(self, title, offset, total, stall_days):
        return {"title": title, "total_units": total, "char_offset": offset,
                "last_read_at": (NOW - timedelta(days=stall_days)).isoformat()}

    def test_pull_grows_with_the_stall(self):
        # A longer stall must be a *stronger* pull, not a weaker one.
        short = temporal._stall_strength(5)
        long = temporal._stall_strength(60)
        self.assertGreater(long, short)
        self.assertGreater(long, 0.9)

    def test_no_pull_before_the_threshold(self):
        self.assertEqual(temporal._stall_strength(0.5), 0.0)
        self.assertEqual(temporal.impatience_items(
            [self._question("Fresh question?", 0.5)], [], now=NOW), [])

    def test_a_stalled_question_pulls(self):
        lines = temporal.impatience_lines(
            [self._question("What makes a self?", 30)], [], now=NOW)
        self.assertEqual(len(lines), 1)
        self.assertIn("What makes a self?", lines[0])

    def test_only_a_partly_read_work_can_stall(self):
        stalled = self._work("Kitchen", 400, 1000, 20)
        untouched = self._work("Dune", 0, 1000, 20)
        finished = self._work("Fahrenheit 451", 1000, 1000, 20)
        self.assertEqual(len(temporal.impatience_items([], [stalled], now=NOW)), 1)
        self.assertEqual(temporal.impatience_items([], [untouched], now=NOW), [])
        self.assertEqual(temporal.impatience_items([], [finished], now=NOW), [])

    def test_movement_resets_the_stall(self):
        # A question that just moved has no pull even if it is old overall.
        moved = self._question("Old but active?", 90,
                               last_progress_at=(NOW - timedelta(hours=2)).isoformat())
        self.assertEqual(temporal.impatience_items([moved], [], now=NOW), [])

    def test_block_is_a_pull_not_a_mood(self):
        block = temporal.impatience_prompt_block(
            open_questions=[self._question("What makes a self?", 30)], now=NOW)
        self.assertIn("PULL, NOT A MOOD", block)
        self.assertIn("not a feeling about Roum", block)

    def test_no_stall_means_no_block(self):
        self.assertIsNone(temporal.impatience_prompt_block(
            open_questions=[self._question("Fresh?", 0.5)], now=NOW))
        self.assertIsNone(temporal.impatience_prompt_block(now=NOW))

    def test_impatience_is_not_just_a_session_gap(self):
        # The same gap with nothing stalled must produce no pull at all: this is
        # the line between a grounded signal and an arbitrary timer.
        self.assertIsNone(temporal.impatience_prompt_block(
            open_questions=[self._question("Just started?", 1)], now=NOW))


class TestBoredomIsDeliberatelyAbsent(unittest.TestCase):
    def test_boredom_is_not_implemented(self):
        # Recorded as a decision, not forgotten. Revisit only with a real
        # consumer (something she could do about it).
        self.assertFalse(temporal.BOREDOM_IMPLEMENTED)

    def test_boredom_is_not_in_the_affect_dimensions(self):
        from astra import affect
        self.assertNotIn("boredom", affect.AFFECT_COMPONENTS)

    def test_no_boredom_language_in_any_prompt_block(self):
        block = temporal.temporal_prompt_block(
            last_seen=(NOW - timedelta(days=30)).isoformat(), now=NOW)
        self.assertNotIn("bored", block.casefold())


class TestPromptIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orchestrator = CompanionOrchestrator(self.store, config_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_prompt_building_does_not_record_presence(self):
        # The gap must survive a prompt build, or it could never be sensed.
        self.store.mark_present((datetime.now(timezone.utc) - timedelta(days=4)).isoformat())
        before = self.store.last_present()
        self.orchestrator.build_prompt("hello", [])
        self.orchestrator.build_prompt("hello again", [])
        self.assertEqual(self.store.last_present(), before)

    def test_gap_appears_in_prompt(self):
        self.store.mark_present((datetime.now(timezone.utc) - timedelta(days=4)).isoformat())
        prompt, _ = self.orchestrator.build_prompt_with_diagnostics("hi", [])
        self.assertIn("THE PASSAGE OF TIME", prompt)
        self.assertIn("about 4 days", prompt)

    def test_no_gap_means_no_time_block(self):
        self.store.mark_present()
        prompt, _ = self.orchestrator.build_prompt_with_diagnostics("hi", [])
        self.assertNotIn("THE PASSAGE OF TIME", prompt)

    def test_a_stalled_question_reaches_the_prompt(self):
        qid = self.store.add_question("Why do I keep returning to this?", confidence=0.6)
        old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        with self.store._transaction("self"):
            _, q = self.store._locate("self", qid)
            q["timestamp"] = old
            q["opened_at"] = old
            q["last_progress_at"] = old
        prompt, _ = self.orchestrator.build_prompt_with_diagnostics("hello", [])
        self.assertIn("A LITTLE IMPATIENCE", prompt)
        self.assertIn("Why do I keep returning to this?", prompt)

    def test_no_impatience_block_when_nothing_is_stalled(self):
        self.store.add_question("A brand new question?", confidence=0.6)
        prompt, _ = self.orchestrator.build_prompt_with_diagnostics("hello", [])
        self.assertNotIn("A LITTLE IMPATIENCE", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
