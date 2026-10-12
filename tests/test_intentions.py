"""Tests for Astra's own plans as intentions (``astra/intentions.py``).

These pin the brief's central rule for this change: Astra talks about what she
has planned as *her own* intentions, drawn from the runtime's real records, and
never as a report about a scheduling system - and what she says cannot drift
from what the systems actually record. Everything runs against real stores (a
``TaskStore``, a ``TaskLedger``, a ``TripleMemoryStore``) and a real
``CompanionOrchestrator`` - no mocks.

Covered:

* recognition: a plans/intentions question renders the block, an ordinary turn
  does not;
* state accuracy: scheduled commitments, self-selected work, the running task,
  postponed work ranks are each reported distinctly, and a possibility
  (planner candidate) is never presented as a decision;
* natural language: the block is first-person and does not name the machinery,
  while still permitting an explicit explanation when asked;
* affect-sensitive framing: a low state does not read the same as a warm one;
* empty and changing schedules: an idle state is reported as idle, and the
  render changes when the underlying tasks change;
* uncertainty: considering/blocked work is labelled as not-yet-decided;
* consistency: the rendered snapshot agrees with the store's own records;
* read-only: building the snapshot and the prompt mutates no task and writes no
  file.
"""

import os
import shutil
import tempfile
import unittest

from astra import intentions
from astra.application import RuntimeController
from astra.memory import TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator
from astra.tasks import BackgroundTask, TaskLedger, TaskStore

CONFIG_DIR = "./config"


def _reading(key="moby", why="unfinished work"):
    return BackgroundTask(
        kind="reading", origin="activity", dedupe_key=key,
        payload={"work_id": key}, progress={"why": why})


def _scheduled(key="nightly-self-review", kind="self_review"):
    return BackgroundTask(
        kind=kind, origin="schedule", dedupe_key=key,
        schedule={"mode": "daily", "at": "03:30"},
        due_at="2026-10-13T07:30:00+00:00")


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.task_store = TaskStore(data_dir=self.tmp)
        self.ledger = TaskLedger(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(self.store, config_dir=CONFIG_DIR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def attach(self):
        """Attach a read-only intentions source, exactly as build_session does."""
        self.orch.set_intentions_source(
            lambda: intentions.build_snapshot(
                self.task_store, ledger=self.ledger))

    def prompt(self, text):
        return self.orch.build_prompt(text, [])


# ---------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------
class TestRecognition(_Case):
    def test_plan_questions_are_recognised(self):
        for text in (
            "What are you planning?",
            "What are your plans?",
            "What are you working on?",
            "Do you have anything planned?",
            "What's next for you?",
            "Are you busy with anything?",
            "What's on your schedule?",
            "What do you intend to do?",
        ):
            self.assertTrue(intentions.asks_intentions(text), text)

    def test_ordinary_turns_are_not_recognised(self):
        for text in (
            "hello there",
            "tell me a joke",
            "I read a book today",
            "what do you think of the weather",
            "did you sleep well",
        ):
            self.assertFalse(intentions.asks_intentions(text), text)

    def test_block_renders_only_when_asked(self):
        self.attach()
        self.task_store.enqueue(_reading())
        self.assertIn("ASTRA'S OWN PLANS", self.prompt("What are you planning?"))
        self.assertNotIn("ASTRA'S OWN PLANS", self.prompt("hello there"))

    def test_no_source_means_no_block(self):
        # Without an attached source the prompt is unchanged, so a bare
        # orchestrator (and every existing test) is unaffected.
        self.assertNotIn("ASTRA'S OWN PLANS", self.prompt("What are you planning?"))


# ---------------------------------------------------------------------
# State accuracy and the category distinctions
# ---------------------------------------------------------------------
class TestStateAccuracy(_Case):
    def test_categories_are_distinct(self):
        self.task_store.enqueue(_scheduled())
        self.task_store.enqueue(_reading())
        snap = intentions.build_snapshot(self.task_store, ledger=self.ledger)
        self.assertEqual(len(snap["commitments"]), 1)
        self.assertEqual(snap["commitments"][0]["kind"], "self_review")
        self.assertEqual(len(snap["selected"]), 1)
        self.assertEqual(snap["selected"][0]["kind"], "reading")

    def test_running_task_is_reported_as_active_not_done(self):
        task = _reading()
        self.task_store.enqueue(task)
        self.task_store.mark(task.id, "running")
        snap = intentions.build_snapshot(self.task_store, ledger=self.ledger)
        self.assertEqual(len(snap["active"]), 1)
        self.assertEqual(snap["active"][0]["status"], "running")
        # It is not also filed as merely selected or finished.
        self.assertEqual(snap["selected"], [])
        self.assertEqual(snap["recent"], [])

    def test_deferred_and_blocked_are_postponed(self):
        a = _reading("a")
        b = _reading("b")
        self.task_store.enqueue(a)
        self.task_store.enqueue(b)
        self.task_store.mark(a.id, "deferred")
        self.task_store.mark(b.id, "blocked")
        snap = intentions.build_snapshot(self.task_store, ledger=self.ledger)
        kinds = {e["kind"] for e in snap["waiting"]}
        self.assertEqual(kinds, {"reading"})
        self.assertEqual(len(snap["waiting"]), 2)

    def test_failed_is_retrying_not_waiting(self):
        task = _reading()
        self.task_store.enqueue(task)
        self.task_store.mark(task.id, "failed")
        snap = intentions.build_snapshot(self.task_store, ledger=self.ledger)
        self.assertEqual(len(snap["retrying"]), 1)
        self.assertEqual(snap["waiting"], [])

    def test_considering_is_a_possibility_not_a_decision(self):
        # With open questions in the store, the planner evidences candidates.
        self.store.add_memory(
            content="An open question about how memory decays.",
            mem_type="open_question", target_model="self",
            tags=["open_question", "qstatus:open"], source="user")
        planner = _FakePlanner()
        snap = intentions.build_snapshot(
            self.task_store, planner=planner, ledger=self.ledger)
        self.assertTrue(snap["considering"])
        # A candidate is only ever in `considering` - never claimed as selected.
        self.assertEqual(snap["selected"], [])
        block = intentions.prompt_block(snap)
        self.assertIn("Only considering, not decided", block)

    def test_completed_appears_in_recent(self):
        self.ledger.record(task_id="t1", kind="memory_review",
                           status="completed", detail="swept")
        snap = intentions.build_snapshot(self.task_store, ledger=self.ledger)
        self.assertEqual(len(snap["recent"]), 1)
        self.assertEqual(snap["recent"][0]["kind"], "memory_review")


class _FakePlanner:
    """A minimal planner exposing the real ``candidates`` interface."""

    def candidates(self):
        return [{"kind": "memory_review", "dedupe_key": "quality",
                 "lane": "maintenance", "reason": "review memory quality"}]


# ---------------------------------------------------------------------
# Natural language vs. the machinery
# ---------------------------------------------------------------------
class TestNaturalExpression(_Case):
    def test_block_is_first_person_and_hides_the_machinery(self):
        self.attach()
        self.task_store.enqueue(_reading())
        self.task_store.enqueue(_scheduled())
        block = intentions.prompt_block(
            intentions.build_snapshot(self.task_store, ledger=self.ledger))
        # The *plan lines* describe her intentions, not a system: the machinery
        # is named only in the instruction that tells her not to describe it.
        plan_lines = [ln for ln in block.splitlines() if ln.startswith("- ")]
        plan_text = "\n".join(plan_lines)
        for forbidden in ("scheduler", "task queue", "task store", "task_store",
                          "dedupe_key", "priority=", "TaskStore"):
            self.assertNotIn(forbidden, plan_text)
        self.assertIn("her own", block.lower())
        self.assertIn("Committed to", block)

    def test_explicit_explanation_is_permitted_when_asked(self):
        block = intentions.prompt_block(
            intentions.build_snapshot(self.task_store, ledger=self.ledger))
        self.assertIn("unless Roum explicitly asks", block)


# ---------------------------------------------------------------------
# Affect-sensitive framing
# ---------------------------------------------------------------------
class TestAffectFraming(_Case):
    def test_low_state_frames_differently_from_warm(self):
        snap = intentions.build_snapshot(self.task_store, ledger=self.ledger)
        low = intentions.prompt_block(
            snap, affect_state={"happy": 0.0, "sad": 0.7, "weary": 0.6,
                                "engagement": 0.1, "curiosity": 0.1,
                                "anticipation": 0.0})
        warm = intentions.prompt_block(
            snap, affect_state={"happy": 0.7, "sad": 0.0, "engagement": 0.8,
                                "curiosity": 0.8, "anticipation": 0.7})
        self.assertNotEqual(low, warm)
        self.assertIn("not eager to take on", low)
        self.assertIn("ready to take things on", warm)

    def test_neutral_state_adds_no_framing(self):
        snap = intentions.build_snapshot(self.task_store, ledger=self.ledger)
        block = intentions.prompt_block(snap, affect_state={})
        self.assertNotIn("not eager to take on", block)
        self.assertNotIn("ready to take things on", block)


# ---------------------------------------------------------------------
# Empty and changing schedules
# ---------------------------------------------------------------------
class TestEmptyAndChanging(_Case):
    def test_idle_is_reported_as_idle(self):
        snap = intentions.build_snapshot(self.task_store, ledger=self.ledger)
        self.assertTrue(intentions.is_idle(snap))
        block = intentions.prompt_block(snap)
        self.assertIn("Nothing is planned or in progress", block)
        self.assertIn("Being idle is allowed", block)

    def test_render_changes_when_the_schedule_changes(self):
        self.attach()
        self.assertIn("Nothing is planned", self.prompt("What are you planning?"))
        self.task_store.enqueue(_reading())
        after = self.prompt("What are you planning?")
        self.assertNotIn("Nothing is planned", after)
        self.assertIn("reading", after)

    def test_removed_task_no_longer_appears(self):
        task = _reading()
        self.task_store.enqueue(task)
        self.assertIn("reading", self.prompt("What are you planning?"))
        self.task_store.mark(task.id, "cancelled")
        after = self.prompt("What are you planning?")
        self.assertNotIn("Chosen for herself: reading", after)


# ---------------------------------------------------------------------
# Uncertainty and honesty
# ---------------------------------------------------------------------
class TestUncertainty(_Case):
    def test_blocked_work_is_labelled_not_asserted(self):
        task = _reading()
        self.task_store.enqueue(task)
        self.task_store.mark(task.id, "blocked")
        block = intentions.prompt_block(
            intentions.build_snapshot(self.task_store, ledger=self.ledger))
        self.assertIn("blocked", block)

    def test_paused_activity_is_stated_without_stopping_work_claim(self):
        snap = intentions.build_snapshot(self.task_store, ledger=self.ledger,
                                         paused=True)
        block = intentions.prompt_block(snap)
        self.assertIn("paused", block)

    def test_must_not_claim_started_or_finished(self):
        block = intentions.prompt_block(
            intentions.build_snapshot(self.task_store, ledger=self.ledger))
        self.assertIn("must not claim to have started", block)
        self.assertIn("Explaining a plan does not begin it", block)


# ---------------------------------------------------------------------
# Consistency between what she says and what is recorded
# ---------------------------------------------------------------------
class TestConsistency(_Case):
    def test_snapshot_agrees_with_the_store_records(self):
        self.task_store.enqueue(_scheduled())
        self.task_store.enqueue(_reading())
        snap = intentions.build_snapshot(self.task_store, ledger=self.ledger)
        live = [t for t in self.task_store.all() if t.is_live()]
        reported = (snap["commitments"] + snap["selected"] + snap["active"]
                    + snap["waiting"] + snap["retrying"])
        self.assertEqual({e["kind"] for e in reported}, {t.kind for t in live})

    def test_controller_snapshot_matches_direct_build(self):
        controller = RuntimeController(
            session=None, store=self.store, data_dir=self.tmp,
            config_dir=CONFIG_DIR,
            config={"autonomy": {"enabled": True, "planning_enabled": True}})
        controller.task_store.enqueue(_reading())
        via_controller = controller.snapshot_intentions()
        self.assertIn("reading", via_controller["selected"][0]["label"])


# ---------------------------------------------------------------------
# Read-only: talking about a plan changes nothing
# ---------------------------------------------------------------------
class TestReadOnly(_Case):
    def _fingerprint(self):
        parts = []
        for name in sorted(os.listdir(self.tmp)):
            path = os.path.join(self.tmp, name)
            if os.path.isfile(path):
                with open(path, "rb") as handle:
                    parts.append((name, handle.read()))
        return parts

    def test_building_snapshot_and_prompt_mutates_no_task(self):
        task = _reading()
        self.task_store.enqueue(task)
        before = self.task_store.get(task.id).status
        intentions.build_snapshot(self.task_store, ledger=self.ledger)
        intentions.prompt_block(
            intentions.build_snapshot(self.task_store, ledger=self.ledger))
        self.assertEqual(self.task_store.get(task.id).status, before)

    def test_talking_about_a_plan_writes_no_file(self):
        self.attach()
        self.task_store.enqueue(_reading())
        before = self._fingerprint()
        self.prompt("What are you planning?")
        self.assertEqual(before, self._fingerprint(),
                         "Talking about a plan must not write to disk.")

    def test_prompt_build_does_not_start_scheduled_work(self):
        task = _scheduled()
        self.task_store.enqueue(task)
        self.attach()
        self.prompt("What are you planning?")
        self.assertEqual(self.task_store.get(task.id).status, "scheduled")


if __name__ == "__main__":
    unittest.main()
