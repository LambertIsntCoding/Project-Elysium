"""Tests for the pure, time-aware scheduling arithmetic."""

import unittest
from datetime import datetime, timedelta, timezone

from astra import schedules as S


def _utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


class TestNormalize(unittest.TestCase):
    def test_valid_daily(self):
        spec = S.normalize({"mode": "daily", "at": "09:00"})
        self.assertEqual(spec["mode"], "daily")

    def test_invalid_mode_refused(self):
        self.assertIsNone(S.normalize({"mode": "hourly", "at": "09:00"}))

    def test_daily_needs_a_time(self):
        self.assertIsNone(S.normalize({"mode": "daily"}))

    def test_invalid_catch_up_refused(self):
        self.assertIsNone(S.normalize({"mode": "daily", "at": "09:00",
                                       "catch_up": "whenever"}))

    def test_non_dict_refused(self):
        self.assertIsNone(S.normalize("daily"))


class TestNextDue(unittest.TestCase):
    def test_daily_before_time_is_today(self):
        # now = 06:00 EDT on 2026-06-01, before the 09:00 local time.
        now = _utc("2026-06-01T10:00:00")
        due = S.next_due({"mode": "daily", "at": "09:00",
                          "timezone": "America/New_York"}, now=now)
        self.assertEqual(due, _utc("2026-06-01T13:00:00"))

    def test_daily_after_time_is_tomorrow(self):
        now = _utc("2026-06-01T14:00:00")
        due = S.next_due({"mode": "daily", "at": "09:00",
                          "timezone": "America/New_York"}, now=now)
        self.assertEqual(due, _utc("2026-06-02T13:00:00"))

    def test_daily_window_rolls_to_next_day(self):
        now = _utc("2026-06-01T15:00:00")  # 11:00 EDT
        due = S.next_due({"mode": "daily", "at": "09:00", "start": "08:00",
                          "end": "10:00", "timezone": "America/New_York"}, now=now)
        # Today's window has passed, so the next runnable occurrence is tomorrow.
        self.assertEqual(due, _utc("2026-06-02T13:00:00"))

    def test_dst_spring_forward_keeps_local_hour(self):
        # US DST 2026 begins 2026-03-08. On 03-07 09:00 EST = 14:00 UTC; the
        # next occurrence is 03-08 09:00 EDT = 13:00 UTC (not 14:00).
        now = _utc("2026-03-07T14:00:00")
        due = S.next_due({"mode": "daily", "at": "09:00",
                          "timezone": "America/New_York"}, now=now)
        self.assertEqual(due, _utc("2026-03-08T13:00:00"))

    def test_interval_uses_last_run(self):
        spec = {"mode": "interval", "every_hours": 1, "timezone": "UTC"}
        last = _utc("2026-06-01T10:00:00")
        now = _utc("2026-06-01T10:05:00")
        due = S.next_due(spec, last_run=last, now=now)
        self.assertEqual(due, _utc("2026-06-01T11:00:00"))

    def test_interval_first_time_is_one_interval_away(self):
        spec = {"mode": "interval", "every_seconds": 600, "timezone": "UTC"}
        now = _utc("2026-06-01T10:00:00")
        due = S.next_due(spec, now=now)
        self.assertEqual(due, _utc("2026-06-01T10:10:00"))

    def test_expiry_exhausts_schedule(self):
        now = _utc("2026-06-01T10:00:00")
        due = S.next_due({"mode": "daily", "at": "09:00", "timezone": "UTC",
                          "expires": "2026-06-01T00:00:00"}, now=now)
        self.assertIsNone(due)


class TestQuietHours(unittest.TestCase):
    def test_wraps_midnight(self):
        quiet = {"start": "22:00", "end": "07:00"}
        self.assertTrue(S.in_quiet_hours(_utc("2026-06-01T23:30:00"), quiet))
        self.assertTrue(S.in_quiet_hours(_utc("2026-06-01T03:00:00"), quiet))
        self.assertFalse(S.in_quiet_hours(_utc("2026-06-01T12:00:00"), quiet))

    def test_same_start_end_means_never_quiet(self):
        self.assertFalse(S.in_quiet_hours(
            _utc("2026-06-01T12:00:00"), {"start": "08:00", "end": "08:00"}))

    def test_unparseable_is_never_quiet(self):
        self.assertFalse(S.in_quiet_hours(_utc("2026-06-01T12:00:00"),
                                          {"start": "nope", "end": "07:00"}))
        self.assertFalse(S.in_quiet_hours(_utc("2026-06-01T12:00:00"), None))


class TestMissed(unittest.TestCase):
    def test_future_is_run(self):
        spec = {"mode": "daily", "at": "09:00", "timezone": "UTC"}
        self.assertEqual(S.resolve_missed(spec, _utc("2026-06-02T09:00:00"),
                                          now=_utc("2026-06-01T00:00:00")),
                         S.CATCH_UP_RUN)

    def test_past_deadline_is_skip(self):
        spec = {"mode": "daily", "at": "09:00", "deadline": "10:00",
                "timezone": "UTC"}
        due = _utc("2026-06-01T09:00:00")
        now = _utc("2026-06-01T11:00:00")
        self.assertEqual(S.resolve_missed(spec, due, now=now), S.CATCH_UP_SKIP)

    def test_other_day_honours_policy(self):
        base = {"mode": "daily", "at": "09:00", "timezone": "UTC"}
        due = _utc("2026-06-01T09:00:00")
        now = _utc("2026-06-03T08:00:00")  # a later day, before 09:00
        # Without a deadline the configured catch-up policy decides: a schedule
        # that says skip is skipped, and the default is to still run it late.
        self.assertEqual(S.resolve_missed(dict(base, catch_up="skip"), due, now=now),
                         S.CATCH_UP_SKIP)
        self.assertEqual(S.resolve_missed(base, due, now=now), S.CATCH_UP_RUN)

    def test_other_day_with_deadline_is_skip(self):
        spec = {"mode": "daily", "at": "09:00", "deadline": "10:00",
                "timezone": "UTC"}
        due = _utc("2026-06-01T09:00:00")
        now = _utc("2026-06-03T08:00:00")
        self.assertEqual(S.resolve_missed(spec, due, now=now), S.CATCH_UP_SKIP)

    def test_in_window_honours_policy(self):
        spec = {"mode": "daily", "at": "09:00", "catch_up": "run",
                "timezone": "UTC"}
        due = _utc("2026-06-01T09:00:00")
        now = _utc("2026-06-01T09:30:00")
        self.assertEqual(S.resolve_missed(spec, due, now=now), S.CATCH_UP_RUN)


class TestDependencies(unittest.TestCase):
    def test_missing_dependency_not_met(self):
        self.assertFalse(S.dependencies_met({"depends_on": ["a"]}, lambda _: None))

    def test_terminal_dependency_met(self):
        class T:
            status = "completed"
        self.assertTrue(S.dependencies_met({"depends_on": ["a"]}, lambda _: T()))

    def test_live_dependency_not_met(self):
        class T:
            status = "running"
        self.assertFalse(S.dependencies_met({"depends_on": ["a"]}, lambda _: T()))

    def test_no_dependencies_met(self):
        self.assertTrue(S.dependencies_met({}, lambda _: None))


class TestTzResolution(unittest.TestCase):
    def test_utc(self):
        self.assertEqual(S.resolve_tz("UTC").utcoffset(None), timedelta(0))

    def test_fallback_offset(self):
        # A non-IANA name degrades to a fixed offset rather than crashing.
        tz = S.resolve_tz("UTC+05:30")
        self.assertEqual(tz.utcoffset(None), timedelta(hours=5, minutes=30))

    def test_unknown_name_is_utc(self):
        self.assertEqual(S.resolve_tz("Nowhere/Land").utcoffset(None), timedelta(0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
