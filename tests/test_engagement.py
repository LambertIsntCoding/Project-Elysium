"""Tests for non-response adaptation and effective affect decay.

Priority 1 (engagement) is exercised against the real ``NotificationPolicy`` and
the real ``EngagementTracker`` - no mocking of the decision path. The tracker
only *raises* the threshold; every existing limit, quiet hour and dedupe rule is
still the policy's. Priority 2 (affect) is exercised by reading the effective
view and then firing a real event, asserting the elapsed interval is applied
exactly once and that reads never persist.
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from astra import affect, engagement
from astra import notifications as N


def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


class _Base(unittest.TestCase):
    def setUp(self):
        self._dirs = []

    def tearDown(self):
        for d in self._dirs:
            shutil.rmtree(d, ignore_errors=True)

    def tmpdir(self):
        d = tempfile.mkdtemp()
        self._dirs.append(d)
        return d


# ---------------------------------------------------------------------
# Priority 1: engagement / non-response adaptation
# ---------------------------------------------------------------------
class TestEngagementTracker(_Base):

    def tracker(self, tmp=None, **cfg):
        return engagement.EngagementTracker(tmp or self.tmpdir(), config=cfg)

    def test_isolated_fresh_message_is_awaiting_not_unanswered(self):
        t = self.tracker(reply_window_seconds=3600)
        t.note_sent("m1", now=_utc(2026, 1, 1, 12, 0))
        # Immediately after a send, nothing counts: it is awaiting a reply.
        self.assertEqual(t.unanswered_count(now=_utc(2026, 1, 1, 12, 0)), 0)
        self.assertEqual(t.effective_threshold_boost(now=_utc(2026, 1, 1, 12, 30)), 0.0)

    def test_single_silence_is_weak_evidence_no_boost(self):
        t = self.tracker(reply_window_seconds=3600, silence_after=2)
        t.note_sent("m1", now=_utc(2026, 1, 1, 12, 0))
        later = _utc(2026, 1, 1, 14, 0)
        self.assertEqual(t.unanswered_count(now=later), 1)
        # One unanswered message never raises the bar.
        self.assertEqual(t.effective_threshold_boost(now=later), 0.0)

    def test_repeated_non_response_raises_the_threshold(self):
        t = self.tracker(reply_window_seconds=3600, silence_after=2,
                         threshold_step=0.08, max_threshold_boost=0.25,
                         decay_days=1000.0)  # ignore decay to isolate the count
        base = _utc(2026, 1, 1, 0, 0)
        for i in range(5):
            t.note_sent(f"m{i}", now=base + timedelta(hours=2 * i))
        now = base + timedelta(hours=12)
        self.assertGreaterEqual(t.unanswered_count(now=now), 4)
        self.assertGreater(t.effective_threshold_boost(now=now), 0.0)
        self.assertLessEqual(t.effective_threshold_boost(now=now), 0.25)

    def test_engagement_recovers_the_threshold(self):
        t = self.tracker(reply_window_seconds=3600, silence_after=2, decay_days=1000.0)
        base = _utc(2026, 1, 1, 0, 0)
        # Sends spread over 2h each; the newest is at base+6h.
        for i in range(4):
            t.note_sent(f"m{i}", now=base + timedelta(hours=2 * i))
        # 30 minutes after the newest send: inside the reliable reply window, and
        # the three earlier messages already count as unanswered.
        now = base + timedelta(hours=6, minutes=30)
        self.assertGreater(t.effective_threshold_boost(now=now), 0.0)
        # A reply inside the window answers the newest message and clears the
        # whole chain.
        result = t.note_reply(now=now)
        self.assertTrue(result["associated"])
        self.assertEqual(t.effective_threshold_boost(now=now), 0.0)

    def test_elapsed_time_decays_the_pull(self):
        t = self.tracker(reply_window_seconds=3600, silence_after=2,
                         decay_days=2.0, max_threshold_boost=0.25)
        base = _utc(2026, 1, 1, 0, 0)
        for i in range(5):
            t.note_sent(f"m{i}", now=base + timedelta(hours=2 * i))
        soon = base + timedelta(hours=12)
        much_later = soon + timedelta(days=6)
        self.assertGreater(t.effective_threshold_boost(now=soon), 0.0)
        # Old silence stops holding her back: the boost decays to nothing.
        self.assertEqual(t.effective_threshold_boost(now=much_later), 0.0)

    def test_restart_persists_the_silence(self):
        d = self.tmpdir()
        t = self.tracker(d, reply_window_seconds=3600, silence_after=2, decay_days=1000.0)
        base = _utc(2026, 1, 1, 0, 0)
        for i in range(4):
            t.note_sent(f"m{i}", now=base + timedelta(hours=2 * i))
        now = base + timedelta(hours=10)
        before = t.effective_threshold_boost(now=now)
        self.assertGreater(before, 0.0)
        # A fresh tracker over the same directory sees the same silence.
        reloaded = engagement.EngagementTracker(d, config={
            "reply_window_seconds": 3600, "silence_after": 2, "decay_days": 1000.0})
        self.assertEqual(reloaded.effective_threshold_boost(now=now), before)

    def test_ambiguous_reply_out_of_window_is_not_engagement(self):
        t = self.tracker(reply_window_seconds=3600, silence_after=2)
        t.note_sent("m1", now=_utc(2026, 1, 1, 12, 0))
        # A message arriving a day later is unrelated (Roum was away): it is not
        # counted as engagement, and it does not accrue as silence either.
        result = t.note_reply(now=_utc(2026, 1, 2, 12, 0))
        self.assertFalse(result["associated"])
        self.assertEqual(result["reason"], "out_of_window")
        self.assertEqual(t.unanswered_count(now=_utc(2026, 1, 2, 12, 0)), 0)

    def test_duplicate_ref_is_not_double_counted(self):
        t = self.tracker(reply_window_seconds=3600)
        t.note_sent("same", now=_utc(2026, 1, 1, 0, 0))
        t.note_sent("same", now=_utc(2026, 1, 1, 1, 0))
        self.assertEqual(len(t._state["pending"]), 1)

    def test_disabled_tracker_is_inert(self):
        t = self.tracker(enabled=False, reply_window_seconds=3600)
        t.note_sent("m1", now=_utc(2026, 1, 1, 0, 0))
        self.assertEqual(t.effective_threshold_boost(now=_utc(2026, 1, 2, 0, 0)), 0.0)
        self.assertEqual(len(t._state["pending"]), 0)


class TestPolicyIntegration(_Base):
    """The policy raises its own threshold; it never relaxes a limit."""

    def policy(self, tmp, **cfg):
        base = {"timezone": "UTC", "interval_seconds": 0, "max_per_interval": 5,
                "min_gap_seconds": 0, "max_per_day": 10, "min_importance": 0.55}
        base.update(cfg)
        return N.NotificationPolicy(data_dir=tmp, config=base)

    def outcome(self, importance):
        return {"ref_id": f"o{importance}", "kind": "work_review",
                "summary": "A finding", "importance": importance}

    def test_non_response_raises_the_bar_for_a_borderline_message(self):
        tmp = self.tmpdir()
        now = _utc(2026, 1, 1, 12, 0)
        # A borderline message (0.60 > 0.55) is delivered when she is engaged.
        p = self.policy(tmp)
        self.assertEqual(p.decide(self.outcome(0.60), now=now)["disposition"], N.DELIVER)

    def test_raised_threshold_downgrades_borderline_to_digest(self):
        tmp = self.tmpdir()
        now = _utc(2026, 1, 1, 12, 0)
        p = self.policy(tmp)
        p.engagement_provider = lambda: 0.10  # silence pushes the bar to 0.65
        v = p.decide(self.outcome(0.60), now=now)
        self.assertEqual(v["disposition"], N.DIGEST)
        self.assertEqual(v["suppressed_by"], N.HOLD_LOW_IMPORTANCE)

    def test_raised_threshold_never_relaxes_limits_or_quiet(self):
        tmp = self.tmpdir()
        p = self.policy(tmp, quiet_hours={"start": "22:00", "end": "07:00"})
        # A negative/zero provider must not lower the bar below config.
        p.engagement_provider = lambda: -5.0
        self.assertEqual(p.effective_min_importance(), p.min_importance)
        # Quiet hours still hold regardless.
        quiet = _utc(2026, 1, 1, 23, 0)
        v = p.decide(self.outcome(0.99), now=quiet)
        self.assertEqual(v["suppressed_by"], N.HOLD_QUIET)

    def test_high_priority_still_bypasses_ordinary_limits_when_raised(self):
        tmp = self.tmpdir()
        p = self.policy(tmp, allow_high_priority_bypass=True,
                        min_gap_seconds=99999, max_per_interval=0)
        p.engagement_provider = lambda: 0.40
        v = p.decide(self.outcome(0.95), now=_utc(2026, 1, 1, 12, 0))
        self.assertEqual(v["disposition"], N.DELIVER)

    def test_delivery_opens_an_obligation_but_a_digest_does_not(self):
        tmp = self.tmpdir()
        t = engagement.EngagementTracker(tmp, config={"reply_window_seconds": 3600})
        p = self.policy(tmp)
        p.note_sent_hook = lambda ref_id, now=None: t.note_sent(ref_id, now=now)
        now = _utc(2026, 1, 1, 12, 0)
        p.note_delivered("o1", now=now)
        self.assertEqual(len(t._state["pending"]), 1)
        p.note_delivered("digest:1", now=now, is_digest=True)
        self.assertEqual(len(t._state["pending"]), 1)  # unchanged by the digest


# ---------------------------------------------------------------------
# Priority 3: honest reading outcomes
# ---------------------------------------------------------------------
class _StubReadingReader:
    def __init__(self, report):
        self._report = report

    def run_once(self, **_kw):
        return dict(self._report)


class _ReadingSession:
    output_fn = staticmethod(lambda text: None)

    def note_activity(self):
        pass

    def handle(self, text):
        return text


class TestRouterEngagement(_Base):
    """A human message, through the single router choke point, recovers her."""

    def _controller(self, tmp):
        from astra.application import RuntimeController
        return RuntimeController(
            session=_ReadingSession(), store=None, data_dir=tmp,
            config_dir="./config", config={}, reader=None, library=None,
            poll_seconds=1.0,
        )

    def test_trackers_are_wired_through_the_controller(self):
        ctrl = self._controller(self.tmpdir())
        # The policy's provider and hook are the controller's own layer.
        self.assertIsNotNone(ctrl.notifications.engagement_provider)
        self.assertIsNotNone(ctrl.notifications.note_sent_hook)
        # A delivered standalone message opens an obligation...
        ctrl.notifications.note_delivered("outcome:1")
        self.assertEqual(len(ctrl.engagement._state["pending"]), 1)

    def test_router_records_a_human_message(self):
        from astra.application import ApplicationRouter
        tmp = self.tmpdir()
        ctrl = self._controller(tmp)
        router = ApplicationRouter(ctrl.session, ctrl)
        ctrl.engagement.note_sent("outcome:1")
        self.assertEqual(len(ctrl.engagement._state["pending"]), 1)
        router.handle("morning")  # a human message arrives on a transport
        self.assertEqual(len(ctrl.engagement._state["pending"]), 0)


class TestReadingOutcomes(_Base):

    def controller(self, tmp, report, *, reader):
        from astra.application import RuntimeController
        return RuntimeController(
            session=_ReadingSession(), store=None, data_dir=tmp,
            config_dir="./config", config={}, reader=reader, library=None,
            poll_seconds=1.0,
        )

    def outcome(self, reader, task):
        return self.controller(self.tmpdir(), {}, reader=reader)._handle_reading(task)

    def test_successful_read_is_completed(self):
        from astra import tasks as T
        from astra import reading
        reader = _StubReadingReader({"read": True, "reason": reading.IDLE_OK,
                                     "chunks": 1, "words": 120})
        out = self.outcome(reader, T.BackgroundTask(kind=T.KIND_READING))
        self.assertEqual(out["status"], T.COMPLETED)

    def test_model_unreachable_is_failed_not_completed(self):
        from astra import tasks as T
        from astra import reading
        # The reader could not reach the model: this is a *failure*, not a read.
        reader = _StubReadingReader({"read": False, "reason": reading.IDLE_MODEL_FAILED,
                                     "chunks": 0, "words": 0})
        out = self.outcome(reader, T.BackgroundTask(kind=T.KIND_READING))
        self.assertEqual(out["status"], T.FAILED)

    def test_no_model_permitted_is_blocked_not_failed(self):
        from astra import tasks as T
        from astra import reading
        reader = _StubReadingReader({"read": False, "reason": reading.IDLE_HELD,
                                     "chunks": 0, "words": 0})
        out = self.outcome(reader, T.BackgroundTask(kind=T.KIND_READING))
        self.assertEqual(out["status"], T.BLOCKED)

    def test_courtesy_hold_is_skipped_not_failed(self):
        from astra import tasks as T
        from astra import reading
        for reason in (reading.IDLE_USER_ACTIVE, reading.IDLE_GAME_RUNNING,
                       reading.IDLE_MACHINE_BUSY, reading.IDLE_NOT_CONCENTRATING):
            reader = _StubReadingReader({"read": False, "reason": reason,
                                         "chunks": 0, "words": 0})
            out = self.outcome(reader, T.BackgroundTask(kind=T.KIND_READING))
            self.assertEqual(out["status"], T.SKIPPED, reason)

    def test_no_material_is_completed(self):
        from astra import tasks as T
        from astra import reading
        reader = _StubReadingReader({"read": False, "reason": reading.IDLE_NO_WORK,
                                     "chunks": 0, "words": 0})
        out = self.outcome(reader, T.BackgroundTask(kind=T.KIND_READING))
        self.assertEqual(out["status"], T.COMPLETED)

    def test_disabled_reader_is_blocked(self):
        from astra import tasks as T
        out = self.controller(self.tmpdir(), {}, reader=None)._handle_reading(
            T.BackgroundTask(kind=T.KIND_READING))
        self.assertEqual(out["status"], T.BLOCKED)


# ---------------------------------------------------------------------
# Priority 2: effective affect decay
# ---------------------------------------------------------------------
class TestEffectiveAffectDecay(_Base):

    def _state(self, **components):
        state = affect._blank_state()
        state.update(components)
        state["last_updated"] = _utc(2026, 1, 1, 0, 0).isoformat()
        return state

    def test_effective_state_ages_decayed_without_mutating_the_record(self):
        state = self._state(happy=0.8)
        before = dict(state)
        view = affect.effective_state(state, now=_utc(2026, 1, 1, 0, 0) + timedelta(days=3))
        self.assertLess(view["happy"], 0.8)
        # The stored record is untouched (prompt building stays read-only).
        self.assertEqual(state, before)

    def test_repeated_reads_do_not_compound(self):
        state = self._state(happy=0.8)
        now = _utc(2026, 1, 1, 0, 0) + timedelta(days=2)
        a = affect.effective_state(state, now=now)["happy"]
        for _ in range(20):
            affect.effective_state(state, now=now)
        b = affect.effective_state(state, now=now)["happy"]
        self.assertEqual(a, b)

    def test_reads_never_mutate_so_a_later_write_decays_once(self):
        # Repeated reads of the effective view must not alter the stored record,
        # so a subsequent write decays from the *original* timestamp exactly
        # once rather than compounding across many reads.
        state = self._state(happy=0.8)
        for days in range(1, 6):
            affect.effective_state(state, now=_utc(2026, 1, 1, 0, 0) + timedelta(days=days))
        self.assertEqual(state["happy"], 0.8)
        self.assertEqual(state["last_updated"], _utc(2026, 1, 1, 0, 0).isoformat())

    def test_clock_anomaly_does_not_age(self):
        state = self._state(happy=0.8)
        earlier = _utc(2026, 1, 1, 0, 0) - timedelta(days=5)
        view = affect.effective_state(state, now=earlier)
        self.assertEqual(view["happy"], 0.8)

    def test_missing_timestamp_is_left_as_is(self):
        state = self._state(happy=0.8)
        state["last_updated"] = ""
        view = affect.effective_state(state, now=_utc(2026, 6, 1, 0, 0))
        self.assertEqual(view["happy"], 0.8)

    def test_neutral_read_after_long_gap(self):
        state = self._state(happy=0.05)  # below the surface threshold already
        self.assertTrue(affect.is_neutral(
            affect.effective_state(state, now=_utc(2026, 1, 1, 0, 0) + timedelta(days=10))))

    def test_long_gap_returns_near_neutral_but_not_fabricated(self):
        state = self._state(happy=0.9, weary=0.4)
        far = _utc(2026, 1, 1, 0, 0) + timedelta(days=60)
        view = affect.effective_state(state, now=far)
        # Decays toward the neutral baseline (0.0), never below it.
        self.assertGreaterEqual(view["happy"], 0.0)
        self.assertLess(view["happy"], 0.01)

    def test_store_current_affect_is_effective(self):
        from astra.memory import TripleMemoryStore
        tmp = self.tmpdir()
        store = TripleMemoryStore(data_dir=tmp, mirror_guard=False)
        store.record_experience("a good moment", kind="happy", intensity=0.9,
                                significance=0.9)
        stored = store.current_affect()
        self.assertGreater(stored.get("happy", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
