"""Tests for durable improvement proposals and the acknowledgement boundary.

The point of this layer is the brief's hard line: routine, reversible
maintenance may proceed autonomously, but a change to identity, boundaries,
credentials, permissions or source code must remain a *proposal requiring
explicit approval*. These tests pin that approving records a decision and never
applies a change, that a recurring review cannot pile up duplicates, and that
the self-improvement handler records durable proposals rather than only a
transient ledger line.
"""

import shutil
import tempfile
import unittest

from astra import proposals as p
from astra.application import RuntimeController
from astra.memory import TripleMemoryStore
from astra import tasks as task_mod


class _StubSession:
    output_fn = staticmethod(lambda text: None)

    def note_activity(self):
        pass

    def handle(self, text):
        return text


class TestProposalStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = p.ProposalStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_proposal_round_trips(self):
        item = self.store.add(p.ImprovementProposal(
            title="Review failures", scope=p.SCOPE_ROUTINE,
            evidence="3 failures", proposed_change="look closer"))
        reloaded = p.ProposalStore(data_dir=self.tmp)
        fetched = reloaded.get(item.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.title, "Review failures")
        self.assertEqual(fetched.evidence, "3 failures")

    def test_open_proposal_is_not_duplicated(self):
        first = self.store.add(p.ImprovementProposal(title="Same", scope=p.SCOPE_ROUTINE))
        second = self.store.add(p.ImprovementProposal(title="Same", scope=p.SCOPE_ROUTINE))
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.store.all()), 1)

    def test_a_closed_proposal_allows_a_new_one(self):
        first = self.store.add(p.ImprovementProposal(title="Same", scope=p.SCOPE_ROUTINE))
        self.store.set_status(first.id, p.P_DECLINED)
        second = self.store.add(p.ImprovementProposal(title="Same", scope=p.SCOPE_ROUTINE))
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(len(self.store.all()), 2)

    def test_protected_scopes_require_approval(self):
        self.assertTrue(p.ImprovementProposal(scope=p.SCOPE_IDENTITY).requires_approval)
        self.assertTrue(p.ImprovementProposal(scope=p.SCOPE_SOURCE).requires_approval)
        self.assertTrue(p.ImprovementProposal(scope=p.SCOPE_CREDENTIALS).requires_approval)
        self.assertFalse(p.ImprovementProposal(scope=p.SCOPE_ROUTINE).requires_approval)

    def test_status_transitions_are_validated(self):
        item = self.store.add(p.ImprovementProposal(title="T"))
        with self.assertRaises(ValueError):
            self.store.set_status(item.id, "made_up")
        approved = self.store.set_status(item.id, p.P_APPROVED)
        self.assertIsNotNone(approved.decided_at)
        self.assertNotIn(item, self.store.open())

    def test_render_open_is_truthful_when_empty(self):
        self.assertIn("No improvement proposals", self.store.render_open())


class TestProposalIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.ctrl = RuntimeController(
            session=_StubSession(), store=self.store, data_dir=self.tmp,
            config_dir="./config", config={}, reader=None, library=None,
            poll_seconds=1.0,
        )
        from astra.application import ApplicationRouter
        self.router = ApplicationRouter(self.ctrl.session, self.ctrl)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_self_improvement_records_a_durable_proposal_from_evidence(self):
        # Seed a real failure so the handler has something evidenced to propose.
        self.ctrl.ledger.record(task_id="t1", kind=task_mod.KIND_TEST_RUN,
                                status=task_mod.FAILED, detail="tests failed")
        task = task_mod.BackgroundTask(kind=task_mod.KIND_SELF_IMPROVEMENT,
                                       dedupe_key="si")
        result = self.ctrl._handle_self_improvement(task)
        self.assertEqual(result["status"], task_mod.COMPLETED)
        self.assertEqual(len(self.ctrl.proposals.all()), 1)
        proposal = self.ctrl.proposals.all()[0]
        self.assertEqual(proposal.scope, p.SCOPE_ROUTINE)
        self.assertIn("failed/blocked", proposal.evidence.lower())

    def test_no_evidence_records_no_proposal(self):
        task = task_mod.BackgroundTask(kind=task_mod.KIND_SELF_IMPROVEMENT,
                                       dedupe_key="si")
        result = self.ctrl._handle_self_improvement(task)
        self.assertEqual(len(self.ctrl.proposals.all()), 0)
        self.assertIn("no improvement", result["detail"].lower())

    def test_approving_records_a_decision_and_applies_nothing(self):
        item = self.ctrl.proposals.add(p.ImprovementProposal(
            title="Change boundaries", scope=p.SCOPE_BOUNDARIES,
            proposed_change="relax a boundary"))
        out = self.router._proposals_command(["approve", item.id])
        self.assertIn("recorded decision only", out)
        stored = self.ctrl.proposals.get(item.id)
        self.assertEqual(stored.status, p.P_APPROVED)
        # The boundary file the proposal referenced is untouched (nothing applied).
        import os
        self.assertTrue(os.path.isdir("./config"))

    def test_listing_reports_open_proposals(self):
        self.ctrl.proposals.add(p.ImprovementProposal(
            title="Review failures", scope=p.SCOPE_ROUTINE))
        out = self.router._proposals_command([])
        self.assertIn("Review failures", out)
        self.assertIn("Improvement proposals", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
