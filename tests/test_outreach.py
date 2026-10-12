"""Tests for connecting grounded reasons, affect and voice to outreach.

These verify the three changes end to end, without mocking the policy:
Change 1 wires grounded conversation candidates through the *existing*
notification policy; Change 2 lets the existing affect state tilt whether a
candidate is worth saying; Change 3 lets a gated model call phrase a message
that the policy already permitted. The delivery sink is a tiny in-process fake,
so decisions, limits, dedupe and accounting are all exercised for real.
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timezone

from astra import autonomy, notifications as N, proactive, proactive_compose
from astra import schedule_runtime
from astra import tasks as T
from astra.memory import TripleMemoryStore


def _utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


class FakeNotifier:
    def __init__(self, fail=False):
        self.sent = []
        self.delivered_ids = set()
        self.fail = fail

    def __call__(self, text, ref_id, outcome):
        if self.fail:
            return {"ok": False, "error": "http 500"}
        if ref_id and ref_id in self.delivered_ids:
            return {"ok": True, "sent": False, "suppressed": "duplicate"}
        if ref_id:
            self.delivered_ids.add(ref_id)
        self.sent.append({"text": text, "ref_id": ref_id, "outcome": outcome})
        return {"ok": True, "sent": True, "message_id": "m1"}


def _candidate(reason, *, weight=0.7, question_id=None, kind="unresolved_question"):
    return {"kind": kind, "reason": reason, "question_id": question_id,
            "work_id": None, "motive": "understanding", "weight": weight}


class Harness:
    """A plain (non-TestCase) bundle of a real store/ledger/policy + driver."""

    def __init__(self, tmp, *, notifier, notif_config=None, candidates=None,
                 relevance=None, composer=None):
        self.tmp = tmp
        self.holder = {"now": _utc("2026-06-01T12:00:00")}
        self.task_store = T.TaskStore(data_dir=tmp)
        self.ledger = T.TaskLedger(data_dir=tmp)
        self.memory = TripleMemoryStore(data_dir=tmp, mirror_guard=False)
        policy = N.NotificationPolicy(data_dir=tmp, config=notif_config or {
            "timezone": "UTC", "interval_seconds": 0, "max_per_interval": 5,
            "min_gap_seconds": 0, "max_per_day": 10, "min_importance": 0.5,
            "allow_high_priority_bypass": False, "digest_enabled": True})
        self.driver = autonomy.AutonomyDriver(
            schedule_driver=schedule_runtime.ScheduleDriver(
                self.task_store, ledger=self.ledger, now_fn=self.clock),
            planner=None, notifications=policy, scheduled_scheduler=None,
            ledger=self.ledger, notifier=notifier, config={}, now_fn=self.clock)
        self.driver.candidate_provider = (
            (lambda: list(candidates)) if candidates is not None else None)
        self.driver.relevance_provider = (
            (lambda: relevance) if relevance is not None else None)
        self.driver.composer = composer

    def clock(self):
        return self.holder["now"]

    def wake(self, **kw):
        return self.driver.wake(now=self.clock(), **kw)


class _Base(unittest.TestCase):
    def setUp(self):
        self._dirs = []

    def tearDown(self):
        for d in self._dirs:
            shutil.rmtree(d, ignore_errors=True)

    def harness(self, **kw):
        # Each harness gets its own data dir so state (dedupe, delivery counts)
        # never leaks between two harnesses in a single test.
        tmp = tempfile.mkdtemp()
        self._dirs.append(tmp)
        return Harness(tmp, **kw)


class TestCandidateDelivery(_Base):
    """Change 1: grounded non-task reasons reach the existing delivery path."""

    def test_grounded_candidate_can_reach_delivery(self):
        notifier = FakeNotifier()
        h = self.harness(notifier=notifier,
                         candidates=[_candidate("Why did the ritual matter?",
                                                weight=0.75, question_id="q1")])
        report = h.wake()
        self.assertEqual(len(notifier.sent), 1)
        self.assertEqual(report["candidate_notifications"]["delivered"], 1)
        self.assertIn("ritual", notifier.sent[0]["text"])

    def test_unimportant_candidate_does_not_notify(self):
        notifier = FakeNotifier()
        # Digest disabled isolates the *standalone* decision: a low-weight
        # candidate must not become a message merely because it exists.
        cfg = {"timezone": "UTC", "interval_seconds": 0, "max_per_interval": 5,
               "min_gap_seconds": 0, "max_per_day": 10, "min_importance": 0.5,
               "allow_high_priority_bypass": False, "digest_enabled": False}
        h = self.harness(notifier=notifier, notif_config=cfg,
                         candidates=[_candidate("A vague, low-priority musing",
                                                weight=0.2, question_id="q2")])
        report = h.wake()
        self.assertEqual(len(notifier.sent), 0)
        self.assertEqual(report["candidate_notifications"]["delivered"], 0)
        # It is retained for a digest, not dropped silently.
        self.assertEqual(report["candidate_notifications"]["queued"], 1)

    def test_no_candidates_no_message(self):
        notifier = FakeNotifier()
        h = self.harness(notifier=notifier, candidates=[])
        report = h.wake()
        self.assertEqual(len(notifier.sent), 0)
        self.assertEqual(report["candidate_notifications"]["considered"], 0)

    def test_candidate_without_a_reason_is_ignored(self):
        notifier = FakeNotifier()
        h = self.harness(notifier=notifier,
                         candidates=[{"kind": "unresolved_question", "reason": "",
                                      "question_id": "q3", "weight": 1.0}])
        report = h.wake()
        self.assertEqual(len(notifier.sent), 0)
        self.assertEqual(report["candidate_notifications"]["considered"], 0)

    def test_dedupe_holds_across_two_wakes(self):
        notifier = FakeNotifier()
        h = self.harness(notifier=notifier,
                         candidates=[_candidate("A settling question",
                                                weight=0.75, question_id="q9")])
        h.wake()
        h.holder["now"] = _utc("2026-06-01T20:00:00")
        h.wake()
        self.assertEqual(len(notifier.sent), 1)
        self.assertEqual(notifier.sent[0]["ref_id"], "q:q9")

    def test_daily_cap_limits_candidate_messages(self):
        notifier = FakeNotifier()
        cfg = {"timezone": "UTC", "interval_seconds": 0, "max_per_interval": 5,
               "min_gap_seconds": 0, "max_per_day": 1, "min_importance": 0.5,
               "allow_high_priority_bypass": False, "digest_enabled": True}
        h = self.harness(notifier=notifier, notif_config=cfg, candidates=[
            _candidate("First strong reason", weight=0.9, question_id="a"),
            _candidate("Second strong reason", weight=0.9, question_id="b"),
        ])
        report = h.wake()
        self.assertEqual(len(notifier.sent), 1)
        self.assertEqual(report["candidate_notifications"]["delivered"], 1)
        self.assertGreaterEqual(report["candidate_notifications"]["queued"], 1)

    def test_user_active_routes_candidate_to_digest(self):
        notifier = FakeNotifier()
        h = self.harness(notifier=notifier,
                         candidates=[_candidate("A reason to speak",
                                                weight=0.9, question_id="q5")])
        report = h.wake(user_active=True)
        self.assertEqual(len(notifier.sent), 0)
        self.assertGreaterEqual(report["candidate_notifications"]["queued"], 1)


class TestCandidateMapping(unittest.TestCase):
    """Scale / identity semantics are explicit, not assumed."""

    def test_weight_is_discounted_not_treated_as_task_importance(self):
        median = proactive.candidate_outcome(_candidate("median", weight=0.55))
        self.assertLess(median["importance"], 0.55)

    def test_strong_candidate_clears_default_threshold(self):
        strong = proactive.candidate_outcome(_candidate("strong", weight=0.7))
        self.assertGreaterEqual(strong["importance"], 0.55)

    def test_ref_is_content_stable_and_never_index_based(self):
        a = proactive.candidate_ref({"kind": "k", "reason": "same text", "question_id": None})
        b = proactive.candidate_ref({"kind": "k", "reason": "same text", "question_id": None})
        self.assertEqual(a, b)
        self.assertEqual(proactive.candidate_ref({"question_id": "q7"}), "q:q7")

    def test_no_reason_or_identity_yields_no_outcome(self):
        self.assertIsNone(proactive.candidate_outcome({"kind": "k", "weight": 1.0}))
        self.assertIsNone(proactive.candidate_outcome(None))


class TestAffectInfluencesOutreach(_Base):
    """Change 2: existing affect state tilts the decision via anticipation."""

    def _state(self, anticipation):
        return {"anticipation": anticipation}

    def test_neutral_affect_adds_nothing(self):
        self.assertEqual(proactive.outreach_relevance({
            "anticipation": 0.0, "happy": 0.0, "sad": 0.0}), 0.0)

    def test_anticipation_raises_relevance_monotonically(self):
        low = proactive.outreach_relevance(self._state(0.2))
        high = proactive.outreach_relevance(self._state(0.9))
        self.assertGreater(high, low)
        self.assertGreater(low, 0.0)

    def test_unrelated_mood_does_not_move_relevance(self):
        # A strong but non-anticipatory mood (e.g. sadness) must not change the
        # outreach relevance - outreach follows forward pull, not general mood.
        flat = proactive.outreach_relevance({"anticipation": 0.0})
        sad = proactive.outreach_relevance({"anticipation": 0.0, "sad": 0.8})
        self.assertEqual(flat, sad)

    def test_borderline_candidate_becomes_deliverable_with_anticipation(self):
        # Digest disabled isolates the standalone decision from the digest step.
        cfg = {"timezone": "UTC", "interval_seconds": 0, "max_per_interval": 5,
               "min_gap_seconds": 0, "max_per_day": 10, "min_importance": 0.5,
               "allow_high_priority_bypass": False, "digest_enabled": False}
        # Without anticipation it does not clear the bar.
        n1 = FakeNotifier()
        h1 = self.harness(notifier=n1, notif_config=cfg, relevance=0.0,
                          candidates=[_candidate("A work I am close to resolving",
                                                 weight=0.58, question_id="qB")])
        h1.wake()
        self.assertEqual(len(n1.sent), 0)
        # With genuine anticipatory pull it does. (A bound, not indiscriminate:
        # the gain is capped, so it tilts a reason near the line - it cannot lift
        # an arbitrary low-weight candidate into a message.)
        n2 = FakeNotifier()
        h2 = self.harness(
            notifier=n2, notif_config=cfg,
            candidates=[_candidate("A work I am close to resolving",
                                   weight=0.58, question_id="qB")],
            relevance=proactive.outreach_relevance(self._state(0.9)))
        h2.wake()
        self.assertEqual(len(n2.sent), 1)


class TestComposer(_Base):
    """Change 3: the model may phrase, never originate, and fails safe."""

    def _outcome_data(self, reason="A real discovery: the code path is shared."):
        return {"ref_id": "q:x", "kind": "discovery", "status": "candidate",
                "reason": reason, "summary": reason, "importance": 0.9}

    def test_prompt_contains_the_recorded_reason_and_forbids_invention(self):
        prompt = proactive_compose.build_prompt(self._outcome_data(), context="ctx")
        self.assertIn("the code path is shared", prompt)
        self.assertIn("Do not add any discovery", prompt)

    def test_compose_uses_model_output_when_present(self):
        text = proactive_compose.compose(self._outcome_data(), lambda p: "  Astra: hi there  ")
        self.assertEqual(text, "hi there")

    def test_compose_falls_back_to_none_on_model_error(self):
        def boom(_p):
            raise RuntimeError("model down")
        self.assertIsNone(proactive_compose.compose(self._outcome_data(), boom))

    def test_compose_requires_a_recorded_reason(self):
        self.assertIsNone(proactive_compose.compose(
            {"ref_id": "x", "summary": ""}, lambda p: "should not run"))

    def test_composer_only_runs_after_delivery_is_permitted(self):
        # A low-importance candidate is not delivered, so the composer is never
        # consulted - it cannot manufacture a message.
        calls = []
        notifier = FakeNotifier()
        cfg = {"timezone": "UTC", "interval_seconds": 0, "max_per_interval": 5,
               "min_gap_seconds": 0, "max_per_day": 10, "min_importance": 0.5,
               "allow_high_priority_bypass": False, "digest_enabled": False}
        h = self.harness(
            notifier=notifier, notif_config=cfg,
            candidates=[_candidate("weak", weight=0.1, question_id="w")],
            composer=lambda o: calls.append(o) or "composed")
        h.wake()
        self.assertEqual(calls, [])
        self.assertEqual(len(notifier.sent), 0)

    def test_composed_text_is_delivered_and_failure_falls_back(self):
        notifier = FakeNotifier()
        h = self.harness(
            notifier=notifier,
            candidates=[_candidate("important thing", weight=0.9, question_id="c")],
            composer=lambda o: "in my own words")
        h.wake()
        self.assertEqual(notifier.sent[0]["text"], "in my own words")

        # Now a failing composer must not break delivery; the deterministic
        # summary is used and the message still goes out.
        notifier2 = FakeNotifier()
        h2 = self.harness(
            notifier=notifier2,
            candidates=[_candidate("another important thing", weight=0.9,
                                   question_id="c2")],
            composer=lambda o: (_ for _ in ()).throw(RuntimeError("down")))
        h2.wake()
        self.assertEqual(len(notifier2.sent), 1)
        self.assertIn("another important thing", notifier2.sent[0]["text"])


class TestNoEligibleReasonUnchanged(_Base):
    def test_no_reason_and_no_outcome_is_quiet(self):
        notifier = FakeNotifier()
        h = self.harness(notifier=notifier, candidates=[])
        report = h.wake()
        self.assertEqual(len(notifier.sent), 0)
        self.assertEqual(report["notifications"]["considered"], 0)
        self.assertEqual(report["candidate_notifications"]["considered"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
