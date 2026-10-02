"""Tests for the proactive communication policy (gates, cooldowns, suppression)."""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from astra import proactive as P


def _cand(weight=0.5, qid="q1"):
    return {"kind": "unresolved_question", "reason": "x", "question_id": qid,
            "weight": weight}


class TestStrength(unittest.TestCase):
    def test_no_candidate_means_no_strength(self):
        # Strength is derived from a candidate; there is nothing to derive from
        # here, which is the structural guard against time-only messaging.
        self.assertLess(P.candidate_strength({}), P.SURFACE_THRESHOLD)

    def test_time_only_strengthens_an_existing_condition(self):
        base = P.candidate_strength(_cand(weight=0.5))
        stalled = P.candidate_strength(_cand(weight=0.5), unresolved_days=30)
        self.assertGreater(stalled, base)

    def test_strength_saturates(self):
        huge = P.candidate_strength(_cand(weight=0.9), unresolved_days=100000)
        self.assertLessEqual(huge, 1.0)


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.policy = P.ProactivePolicy(data_dir=self.tmp, interrupt_cooldown=100)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_user_active_suppresses(self):
        d = self.policy.evaluate(_cand(0.99), strength=0.99, user_active=True)
        self.assertEqual(d["suppressed_by"], P.SUPPRESS_USER_ACTIVE)
        self.assertFalse(d["interrupt"])
        # Strong candidate is retained for later, not discarded.
        self.assertTrue(d["retain"])

    def test_game_suppresses_interruption_but_retains(self):
        d = self.policy.evaluate(_cand(0.9), strength=0.9, game="dota2.exe")
        self.assertFalse(d["interrupt"])
        self.assertTrue(d["retain"])

    def test_mute_suppresses(self):
        self.policy.set_muted(True)
        d = self.policy.evaluate(_cand(0.9), strength=0.9)
        self.assertEqual(d["suppressed_by"], P.SUPPRESS_MUTED)

    def test_cooldown_suppresses(self):
        self.policy.note_interrupt(now=datetime.now(timezone.utc))
        self.assertTrue(self.policy.in_cooldown())
        d = self.policy.evaluate(_cand(0.9), strength=0.9)
        self.assertEqual(d["suppressed_by"], P.SUPPRESS_COOLDOWN)

    def test_cooldown_expires(self):
        self.policy.note_interrupt(now=datetime.now(timezone.utc) - timedelta(seconds=200))
        self.assertFalse(self.policy.in_cooldown())

    def test_strong_candidate_interrupts_when_permitted(self):
        d = self.policy.evaluate(_cand(1.0), strength=0.99)
        self.assertTrue(d["interrupt"])
        self.assertEqual(d["disposition"], P.DISPOSITION_INTERRUPT)

    def test_weak_candidate_is_dropped_not_surfaced(self):
        d = self.policy.evaluate(_cand(0.1), strength=0.1)
        self.assertFalse(d["surface"])
        self.assertEqual(d["disposition"], P.DISPOSITION_DROP)

    def test_high_priority_bypass_requires_permission(self):
        self.policy.note_interrupt(now=datetime.now(timezone.utc))  # in cooldown
        without = self.policy.evaluate(_cand(0.99), strength=0.99,
                                       high_priority_event=True,
                                       allow_high_priority_bypass=False)
        self.assertFalse(without["interrupt"])
        with_bypass = self.policy.evaluate(_cand(0.99), strength=0.99,
                                           high_priority_event=True,
                                           allow_high_priority_bypass=True)
        self.assertTrue(bypass := with_bypass["interrupt"])
        self.assertTrue(with_bypass["bypassed"])

    def test_bypass_never_overrides_user_active_or_mute(self):
        d = self.policy.evaluate(_cand(0.99), strength=0.99, user_active=True,
                                 high_priority_event=True,
                                 allow_high_priority_bypass=True)
        self.assertFalse(d["interrupt"])
        self.policy.set_muted(True)
        d2 = self.policy.evaluate(_cand(0.99), strength=0.99,
                                  high_priority_event=True,
                                  allow_high_priority_bypass=True)
        self.assertFalse(d2["interrupt"])

    def test_no_spam_same_candidate_not_offered_twice(self):
        self.policy.note_surfaced("q1")
        self.assertTrue(self.policy.was_surfaced("q1"))
        self.assertFalse(self.policy.was_surfaced("q2"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
