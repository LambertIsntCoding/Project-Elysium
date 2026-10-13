"""Tests for state-shaped timing and presence (pure, no I/O, no network).

Two things are covered:

* ``behavior.timing_modifiers`` - how affect becomes *how long* actions take;
* ``astra.presence`` - whether an exchange is still live, and that the same
  elapsed time maps to a different answer depending on her state.
"""

import unittest

from astra import behavior, presence


def _mod(**kw):
    base = {
        "energy": 0.5, "initiative": 0.5, "enthusiasm": 0.5, "pace": 0.5,
        "persistence": 0.5, "interest": 0.5, "brevity": 0.0, "hesitation": 0.0,
    }
    base.update(kw)
    return base


def _warm():
    return _mod(energy=0.95, enthusiasm=0.95, pace=0.95, interest=0.95)


def _low():
    return _mod(energy=0.05, enthusiasm=0.05, pace=0.05, interest=0.05,
                hesitation=0.7, _sad=0.7)


class TestTimingModifiers(unittest.TestCase):
    def test_neutral_is_approximately_unity(self):
        t = behavior.timing_modifiers(_mod())
        self.assertAlmostEqual(t["typing_scale"], 1.0, places=2)
        self.assertAlmostEqual(t["patience_scale"], 1.0, places=2)

    def test_warm_state_types_faster_and_waits_longer(self):
        t = behavior.timing_modifiers(_warm())
        self.assertLess(t["typing_scale"], 1.0)
        self.assertGreater(t["patience_scale"], 1.0)

    def test_low_state_types_slower_and_gives_up_sooner(self):
        t = behavior.timing_modifiers(_low())
        self.assertGreater(t["typing_scale"], 1.0)
        self.assertLess(t["patience_scale"], 1.0)

    def test_bounded_never_instant_or_endless(self):
        for tone in (_low(), _warm()):
            t = behavior.timing_modifiers(tone)
            self.assertGreaterEqual(t["typing_scale"], 0.45)
            self.assertLessEqual(t["typing_scale"], 2.2)
            self.assertGreaterEqual(t["patience_scale"], 0.35)
            self.assertLessEqual(t["patience_scale"], 2.4)

    def test_none_is_neutral(self):
        self.assertEqual(behavior.timing_modifiers(None)["typing_scale"], 1.0)
        self.assertEqual(behavior.timing_modifiers(None)["patience_scale"], 1.0)


class TestPatience(unittest.TestCase):
    def test_warm_state_is_more_patient(self):
        self.assertGreater(presence.patience_seconds(_warm()),
                           presence.patience_seconds(_mod()))

    def test_low_state_is_less_patient(self):
        self.assertLess(presence.patience_seconds(_low()),
                        presence.patience_seconds(_mod()))

    def test_bounded(self):
        for tone in (_low(), _warm()):
            patience = presence.patience_seconds(tone)
            self.assertGreaterEqual(patience, presence.MIN_PATIENCE_SECONDS)
            self.assertLessEqual(patience, presence.MAX_PATIENCE_SECONDS)


class TestAssess(unittest.TestCase):
    def test_no_exchange_is_away(self):
        v = presence.assess(None)
        self.assertEqual(v["mode"], presence.AWAY)
        self.assertTrue(v["resume_background"])

    def test_just_spoke_is_present(self):
        v = presence.assess(5.0)
        self.assertEqual(v["mode"], presence.PRESENT)
        self.assertFalse(v["resume_background"])

    def test_lingering_before_patience_runs_out(self):
        v = presence.assess(presence.DEFAULT_PATIENCE_SECONDS * 0.7)
        self.assertEqual(v["mode"], presence.LINGERING)
        self.assertFalse(v["resume_background"])

    def test_gone_quiet_is_away_and_resumes_background(self):
        v = presence.assess(presence.DEFAULT_PATIENCE_SECONDS + 1)
        self.assertEqual(v["mode"], presence.AWAY)
        self.assertTrue(v["resume_background"])

    def test_the_same_gap_depends_on_her_state(self):
        # The point of the whole feature: identical elapsed time, different
        # answer, because how long she waits is a consequence of how she is.
        gap = 150.0
        warm = presence.assess(gap, modulation=_warm())
        low = presence.assess(gap, modulation=_low())
        self.assertNotEqual(warm["mode"], presence.AWAY)
        self.assertEqual(low["mode"], presence.AWAY)


if __name__ == "__main__":
    unittest.main()
