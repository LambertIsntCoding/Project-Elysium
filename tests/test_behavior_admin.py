"""Tests for the admin presentation layer and behaviour modulation.

Covers the pieces added to make Astra's internal state *observable* and to make
it act on behaviour - and, just as importantly, the boundary that none of the
numeric telemetry reaches Astra's prompt or writes state.
"""

import os
import re
import shutil
import tempfile
import unittest

from astra import admin, behavior, telemetry
from astra import inquiry
from astra import affect as affect_mod
from astra import menus as menus_mod
from astra.memory import CommandStore, TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator


class TestTelemetry(unittest.TestCase):
    def test_scale_rounds_magnitude_not_precision(self):
        self.assertEqual(telemetry.to_scale(0.637), 64)
        self.assertEqual(telemetry.to_scale(0.632), 63)
        self.assertEqual(telemetry.to_scale(0.0), 0)
        self.assertEqual(telemetry.to_scale(1.0), 100)
        self.assertEqual(telemetry.to_scale("bad"), 0)

    def test_direction_glyphs(self):
        self.assertEqual(telemetry.direction(0.82, 0.74), telemetry.UP)
        self.assertEqual(telemetry.direction(0.18, 0.25), telemetry.DOWN)
        self.assertEqual(telemetry.direction(0.50, 0.503), telemetry.STEADY)

    def test_format_line_has_number_direction_and_label(self):
        line = telemetry.format_line("Happiness", 0.64, 0.60)
        self.assertIn("Happiness: 64", line)
        self.assertIn(telemetry.UP, line)
        self.assertIn("(High)", line)

    def test_number_is_primary_label_is_supplemental(self):
        line = telemetry.format_line("Interest", 0.82, show_band=False)
        self.assertIn("82", line)


class TestBehavior(unittest.TestCase):
    def test_sadness_lowers_drive(self):
        mod = behavior.from_affect({"sad": 0.8, "weary": 0.6, "engagement": 0.2})
        self.assertLess(mod["energy"], 0.5)
        self.assertLess(mod["initiative"], 0.5)
        self.assertLess(mod["enthusiasm"], 0.5)
        self.assertGreater(mod["brevity"], 0.0)

    def test_positive_affect_raises_drive(self):
        mod = behavior.from_affect({"happy": 0.9, "engagement": 0.8, "curiosity": 0.7})
        self.assertGreater(mod["energy"], 0.5)
        self.assertGreater(mod["initiative"], 0.5)

    def test_neutral_state_says_nothing(self):
        self.assertTrue(behavior.is_neutral(behavior.from_affect({})))
        self.assertIsNone(behavior.prompt_block(behavior.from_affect({})))

    def test_modulation_prompt_block_is_behavioural_not_numeric(self):
        mod = behavior.from_affect({"sad": 0.8, "weary": 0.6})
        block = behavior.prompt_block(mod)
        self.assertIsNotNone(block)
        # No raw numbers: Astra must never receive telemetry as prompt data.
        self.assertNotIn("0.", block)
        for digit in "0123456789":
            self.assertNotIn(digit, block)
        self.assertIn("BEHAVIOUR", block)

    def test_reading_modifiers_track_sad_state(self):
        mod = behavior.from_affect({"sad": 0.9, "weary": 0.7})
        mods = behavior.reading_modifiers(mod)
        self.assertLess(mods["engagement"], 0.5)

    def test_combine_is_bounded(self):
        a = behavior.from_affect({"happy": 1.0})
        b = behavior.from_affect({"sad": 1.0})
        merged = behavior.combine(a, b)
        for key in ("energy", "initiative", "pace"):
            self.assertGreaterEqual(merged[key], 0.0)
            self.assertLessEqual(merged[key], 1.0)


class TestPagination(unittest.TestCase):
    def _records(self, n):
        return [{"id": str(i), "timestamp": f"2026-01-{i:02d}T00:00:00"}
                for i in range(1, n + 1)]

    def test_newest_first_and_page_zero_is_newest(self):
        page = admin.paginate(self._records(10), page=0, page_size=3)
        self.assertEqual([r["id"] for r in page.items], ["10", "9", "8"])
        self.assertFalse(page.has_prev)
        self.assertTrue(page.has_next)

    def test_later_pages_move_backwards_in_time(self):
        page = admin.paginate(self._records(10), page=3, page_size=3)
        self.assertEqual([r["id"] for r in page.items], ["1"])
        self.assertFalse(page.has_next)

    def test_page_beyond_end_clamps(self):
        page = admin.paginate(self._records(4), page=99, page_size=3)
        self.assertEqual(page.page, page.total_pages - 1)

    def test_empty_listing_is_clean(self):
        page = admin.paginate([], page=0, page_size=3)
        self.assertTrue(page.is_empty)
        self.assertIn("nothing", page.nav_line())

    def test_permanent_observations_split(self):
        recs = [{"type": "self_fact"}, {"type": "self_observation"},
                {"type": "self_preference"}]
        settled, observed = admin.split_permanent_and_observations(
            recs, permanent=("self_fact", "self_preference"))
        self.assertEqual(len(settled), 2)
        self.assertEqual(len(observed), 1)


class TestSnapshotStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_bounded_history(self):
        store = admin.SnapshotStore(data_dir=self.tmp, limit=2)
        for i in range(5):
            store.capture("affect", {"values": {"happy": i / 10}})
        self.assertEqual(len(store.recent("affect")), 2)

    def test_persisted_and_reloaded(self):
        store = admin.SnapshotStore(data_dir=self.tmp, limit=3)
        store.capture("affect", {"values": {"happy": 0.5}})
        reloaded = admin.SnapshotStore(data_dir=self.tmp, limit=3)
        self.assertEqual(len(reloaded.recent("affect")), 1)


class TestLiveScreen(unittest.TestCase):
    def test_refresh_loop_with_injected_input(self):
        frames = [{"n": 1}, {"n": 2}]
        state = {"i": 0}

        def render():
            state["i"] += 1
            return [f"frame {state['i']}"]

        screen = admin.LiveScreen("test", render=render,
                                  store=admin.SnapshotStore(data_dir=tempfile.mkdtemp()))
        emitted = []
        keys = iter(["s", "h", "q"])
        screen.run(poll=lambda t: next(keys, "q"), emit=emitted.append,
                   sleep=lambda t: None)
        text = "\n".join(emitted)
        self.assertIn("snapshot captured", text)
        self.assertIn("Left test", text)


class _Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.cmd = CommandStore(data_dir=self.tmp)
        self.config = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        self.orch = CompanionOrchestrator(self.store, config_dir=self.config,
                                          cmd_store=self.cmd)
    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestAffectTelemetryDisplay(_Fixture):
    def test_affect_screen_shows_integers_not_thousandths(self):
        from astra import screens
        self.store.apply_live_affect_event(
            {"kind": affect_mod.LIVE_POSITIVE, "intensity": 0.7, "significance": 0.6})
        lines = screens.affect_render(_fake_session(self.orch))
        text = "\n".join(lines)
        self.assertIn("AFFECT (LIVE)", text)
        # The displayed values are integers on a 0-100 scale, never 0.xyz. Only
        # the "Label: N (descriptor)" value lines are checked: the ISO timestamp
        # in the footer legitimately contains a fractional-seconds "0.".
        value_lines = [ln for ln in lines
                       if re.match(r"^\s*[A-Za-z][A-Za-z ]*:\s*\d+\s*(?:\(|$)", ln)]
        self.assertTrue(value_lines)
        self.assertNotIn("0.", "\n".join(value_lines))
        self.assertIn("Happiness:", text)

    def test_prompt_never_contains_affect_numbers(self):
        self.store.apply_live_affect_event(
            {"kind": affect_mod.LIVE_CRITICISM, "intensity": 0.8, "significance": 0.7})
        prompt = self.orch.build_prompt("how are you?", [])
        # Behavioural modulation is present, but as language, not telemetry.
        self.assertNotIn("Sadness =", prompt)
        self.assertNotIn("Happiness =", prompt)

    def test_relationship_screen_reads_real_state(self):
        from astra import screens
        lines = screens.relationship_render(_fake_session(self.orch))
        text = "\n".join(lines)
        self.assertIn("RELATIONSHIP (LIVE)", text)
        self.assertIn("0-100", text)


class TestMenuRegistry(_Fixture):
    def test_registry_builds_and_word_aliases_work(self):
        from main import ChatSession
        session = ChatSession(self.orch, self.cmd, output_fn=lambda _: None,
                              input_fn=lambda _: "")
        registry = menus_mod.build_registry(session, emit=lambda _: None)
        root = registry.build()
        self.assertIsNotNone(root.find("memories"))
        self.assertIsNotNone(root.find("affect") or root.find("relationships---affect"))
        self.assertEqual(root.items[-1].label, "Exit")

    def test_question_priority_uses_governed_records(self):
        self.store.add_memory("self", "Why does the machine dream?",
                              "open_question", "ai_inference",
                              tags=["qstatus:open"])
        memories = self.store.get_memories("self", status=None)
        rows = inquiry.prioritise_questions(memories)
        self.assertTrue(rows)
        self.assertIn("score", rows[0])
        self.assertTrue(rows[0]["reasons"])


def _fake_session(orch):
    class _S:
        pass
    s = _S()
    s.orchestrator = orch
    s.controller = None
    s.reader = None
    s.store = orch.store
    return s


if __name__ == "__main__":
    unittest.main(verbosity=2)
