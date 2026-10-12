"""Tests for the proactive notification policy: limits, quiet hours, digests."""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from astra import notifications as N


def _utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def _outcome(importance=0.9, ref="o1", summary="something happened"):
    return {"ref_id": ref, "importance": importance, "summary": summary,
            "kind": "self_review"}


class TestDecision(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.policy = N.NotificationPolicy(data_dir=self.tmp, config={
            "timezone": "UTC",
            "interval_seconds": 8 * 3600,
            "max_per_interval": 1,
            "min_gap_seconds": 8 * 3600,
            "max_per_day": 2,
            "min_importance": 0.55,
        })

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_strong_outcome_delivers_when_clear(self):
        d = self.policy.decide(_outcome(), now=_utc("2026-06-01T09:00:00"))
        self.assertEqual(d["disposition"], N.DELIVER)

    def test_low_importance_is_digested_not_delivered(self):
        d = self.policy.decide(_outcome(importance=0.2), now=_utc("2026-06-01T09:00:00"))
        self.assertEqual(d["disposition"], N.DIGEST)
        self.assertEqual(d["suppressed_by"], N.HOLD_LOW_IMPORTANCE)

    def test_missing_content_is_dropped(self):
        d = self.policy.decide({"ref_id": "x", "importance": 0.9, "summary": ""},
                               now=_utc("2026-06-01T09:00:00"))
        self.assertEqual(d["disposition"], N.DROP)

    def test_no_outcome_never_messages(self):
        # With nothing concrete there is no path to a message.
        d = self.policy.decide({"summary": "no ref"}, now=_utc("2026-06-01T09:00:00"))
        self.assertEqual(d["disposition"], N.DROP)

    def test_min_gap_holds_second_message(self):
        policy = N.NotificationPolicy(data_dir=self.tmp, config={
            "timezone": "UTC", "allow_high_priority_bypass": False,
            "interval_seconds": 8 * 3600, "max_per_interval": 1})
        now = _utc("2026-06-01T09:00:00")
        policy.note_delivered("a", now=now)
        d = policy.decide(_outcome(ref="b"), now=now + timedelta(hours=1))
        self.assertEqual(d["disposition"], N.DIGEST)
        self.assertEqual(d["suppressed_by"], N.HOLD_MIN_GAP)

    def test_daily_cap_holds_third_message(self):
        policy = N.NotificationPolicy(data_dir=self.tmp, config={
            "timezone": "UTC", "min_gap_seconds": 0, "interval_seconds": 0,
            "max_per_interval": 10, "max_per_day": 2, "allow_high_priority_bypass": False})
        day = _utc("2026-06-01T09:00:00")
        policy.note_delivered("a", now=day)
        policy.note_delivered("b", now=day + timedelta(hours=1))
        d = policy.decide(_outcome(ref="c"), now=day + timedelta(hours=2))
        self.assertEqual(d["disposition"], N.DIGEST)
        self.assertEqual(d["suppressed_by"], N.HOLD_DAILY_CAP)

    def test_quiet_hours_hold(self):
        policy = N.NotificationPolicy(data_dir=self.tmp, config={
            "timezone": "UTC", "quiet_hours": {"start": "22:00", "end": "07:00"}})
        d = policy.decide(_outcome(), now=_utc("2026-06-01T23:30:00"))
        self.assertEqual(d["disposition"], N.DIGEST)
        self.assertEqual(d["suppressed_by"], N.HOLD_QUIET)

    def test_quiet_hours_are_never_bypassed_by_high_priority(self):
        policy = N.NotificationPolicy(data_dir=self.tmp, config={
            "timezone": "UTC", "quiet_hours": {"start": "22:00", "end": "07:00"},
            "high_priority_importance": 0.5, "allow_high_priority_bypass": True})
        d = policy.decide(_outcome(importance=1.0), now=_utc("2026-06-01T23:30:00"))
        self.assertEqual(d["disposition"], N.DIGEST)

    def test_mute_and_pause_hold(self):
        now = _utc("2026-06-01T09:00:00")
        self.policy.set_muted(True)
        self.assertEqual(self.policy.decide(_outcome(), now=now)["suppressed_by"],
                         N.HOLD_MUTED)
        self.policy.set_muted(False)
        self.policy.set_paused(True)
        self.assertEqual(self.policy.decide(_outcome(), now=now)["suppressed_by"],
                         N.HOLD_PAUSED)

    def test_high_priority_bypass_only_by_config(self):
        now = _utc("2026-06-01T09:00:00")
        policy = N.NotificationPolicy(data_dir=self.tmp, config={
            "timezone": "UTC", "min_gap_seconds": 3600, "high_priority_importance": 0.8,
            "allow_high_priority_bypass": True, "max_per_interval": 1, "max_per_day": 2})
        policy.note_delivered("a", now=now)
        d = policy.decide(_outcome(importance=0.95, ref="b"),
                          now=now + timedelta(minutes=5))
        self.assertEqual(d["disposition"], N.DELIVER)
        self.assertTrue(d["bypassed"])

    def test_duplicate_is_dropped(self):
        now = _utc("2026-06-01T09:00:00")
        self.policy.note_delivered("o1", now=now)
        d = self.policy.decide(_outcome(ref="o1"), now=now + timedelta(days=1))
        self.assertEqual(d["disposition"], N.DROP)
        self.assertEqual(d["suppressed_by"], N.HOLD_DUPLICATE)


class TestDigest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_digest_due_only_when_entries_and_not_quiet(self):
        policy = N.NotificationPolicy(data_dir=self.tmp, config={
            "timezone": "UTC", "digest_enabled": True})
        self.assertFalse(policy.digest_due(now=_utc("2026-06-01T12:00:00")))
        policy.queue_digest(_outcome(importance=0.3, ref="d1", summary="minor"))
        self.assertTrue(policy.digest_due(now=_utc("2026-06-01T12:00:00")))

    def test_digest_counts_toward_daily_cap(self):
        policy = N.NotificationPolicy(data_dir=self.tmp, config={
            "timezone": "UTC", "max_per_day": 1})
        day = _utc("2026-06-01T12:00:00")
        policy.queue_digest(_outcome(importance=0.3, ref="d1", summary="minor"))
        policy.note_delivered("a", now=day)
        self.assertFalse(policy.digest_due(now=day))

    def test_take_digest_consumes(self):
        policy = N.NotificationPolicy(data_dir=self.tmp)
        policy.queue_digest(_outcome(ref="d1", summary="one"))
        entries = policy.take_digest()
        self.assertEqual(len(entries), 1)
        self.assertEqual(policy.pending_digest_count(), 0)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_delivery_and_pause_survive_reload(self):
        policy = N.NotificationPolicy(data_dir=self.tmp, config={"timezone": "UTC"})
        policy.note_delivered("a", now=_utc("2026-06-01T09:00:00"))
        policy.set_paused(True)
        reloaded = N.NotificationPolicy(data_dir=self.tmp, config={"timezone": "UTC"})
        self.assertTrue(reloaded.paused)
        self.assertEqual(reloaded.delivered_on_day(_utc("2026-06-01T10:00:00")), 1)
        self.assertEqual(reloaded.delivered_in_interval(_utc("2026-06-01T10:00:00")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
