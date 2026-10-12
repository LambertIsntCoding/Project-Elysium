"""Regression tests for factual grounding in the retrieval -> prompt pipeline.

Two demonstrated gaps, both about *unresolved* conflicts between things Roum
actually said (sourced records):

* ``link_conflict`` records a mutual ``contradicts`` link and weakens neither
  side. Because ``contradiction_count`` is not bumped, the record was rendered
  as an ordinary "User Facts" line and the diagnostics reported
  ``contradicted: False`` - so a genuine, unsettled conflict reached the model
  as if both facts were current and the model could state one as settled.
* ``diagnose`` keyed only on ``contradiction_count``, so a linked unresolved
  conflict was invisible to developer diagnostics.

These tests assert the *behaviour* (the conflict is surfaced, and neither side is
presented as settled) and the diagnostics flag, not the prompt wording. They also
guard the negative cases: a plain one-way reference is not a conflict, and the
ordinary fact-rendering path is unchanged.
"""

import tempfile
import unittest

from astra.memory import TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator, DeterministicLexicalRetriever


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.orchestrator = CompanionOrchestrator(self.store, config_dir="./config")

    def _sourced_fact(self, content):
        return self.store.add_memory("roum", content, "explicit_fact",
                                     "explicit_user_statement")

    def _prompt(self, query="What is my favorite game?"):
        return self.orchestrator.build_prompt_with_diagnostics(query, [])[0]


class TestUnresolvedSourcedConflict(_StoreCase):
    def test_mutually_linked_sourced_facts_are_surfaced_as_a_conflict(self):
        a = self._sourced_fact("Roum's favorite game is Sonic Adventure.")
        b = self._sourced_fact("Roum's favorite game is Final Fantasy VII.")
        self.store.link_conflict(a, b, target="roum", reason="evidence unsettled")
        prompt = self._prompt()
        marker = "CONFLICTING RECORDS"
        self.assertIn(marker, prompt)
        block = prompt[prompt.index(marker):]
        # Both conflicting statements appear in the dedicated conflict block,
        # alongside the instruction not to treat either as settled.
        self.assertIn("Sonic Adventure", block)
        self.assertIn("Final Fantasy VII", block)
        self.assertIn("unresolved", block.lower())

    def test_conflicting_facts_are_not_only_listed_as_settled_user_facts(self):
        """The contradicted pair must not be presented as two plain facts only."""
        a = self._sourced_fact("Roum's favorite game is Sonic Adventure.")
        b = self._sourced_fact("Roum's favorite game is Final Fantasy VII.")
        self.store.link_conflict(a, b, target="roum")
        prompt = self._prompt()
        # The conflict block must exist *and* precede the generic fact list note
        # that says "these are things Roum told you" (which reads as settled).
        self.assertIn("CONFLICTING RECORDS", prompt)
        self.assertLess(prompt.index("CONFLICTING RECORDS"),
                        prompt.index("=== STYLE EXAMPLES ==="))

    def test_one_way_reference_is_not_a_conflict(self):
        """A single (non-mutual) link is not a recorded conflict.

        Exercised through the pure grouping helper, since the store's only
        conflict writer (``link_conflict``) always writes mutual links.
        """
        a = {"id": "a", "content": "Roum's favorite game is Sonic Adventure.",
             "source": "explicit_user_statement", "contradicts": ["b"]}
        b = {"id": "b", "content": "Roum enjoys platform games.",
             "source": "explicit_user_statement", "contradicts": []}
        self.assertEqual(CompanionOrchestrator._conflict_groups([a, b]), [])

    def test_unconflicted_facts_render_without_a_conflict_block(self):
        self._sourced_fact("Roum's favorite game is Sonic Adventure.")
        prompt = self._prompt()
        self.assertIn("User Facts:", prompt)
        self.assertNotIn("CONFLICTING RECORDS", prompt)


class TestDiagnosticsContradictedFlag(_StoreCase):
    def test_linked_unresolved_conflict_flagged_in_diagnostics(self):
        a = self._sourced_fact("Roum's favorite game is Sonic Adventure.")
        b = self._sourced_fact("Roum's favorite game is Final Fantasy VII.")
        self.store.link_conflict(a, b, target="roum")
        _, diag = self.orchestrator.build_prompt_with_diagnostics(
            "What is my favorite game?", [], detailed=True)
        rows = {r["id"]: r for r in diag["candidates"]}
        self.assertTrue(rows[a]["contradicted"])
        self.assertTrue(rows[b]["contradicted"])

    def test_unconflicted_fact_not_flagged(self):
        a = self._sourced_fact("Roum's favorite game is Sonic Adventure.")
        _, diag = self.orchestrator.build_prompt_with_diagnostics(
            "What is my favorite game?", [], detailed=True)
        rows = {r["id"]: r for r in diag["candidates"]}
        self.assertFalse(rows[a]["contradicted"])


class TestConflictGroupingUnit(unittest.TestCase):
    """The grouping helper is pure; exercise it directly."""

    def test_mutual_links_group_and_one_way_does_not(self):
        a = {"id": "a", "content": "x", "source": "explicit_user_statement",
             "contradicts": ["b"]}
        b = {"id": "b", "content": "y", "source": "explicit_user_statement",
             "contradicts": ["a"]}
        c = {"id": "c", "content": "z", "source": "explicit_user_statement",
             "contradicts": ["a"]}  # one-way: a does not point back
        groups = CompanionOrchestrator._conflict_groups([a, b, c])
        self.assertEqual(len(groups), 1)
        self.assertEqual({m["id"] for m in groups[0]}, {"a", "b"})

    def test_unsourced_records_are_not_grouped_as_facts(self):
        a = {"id": "a", "content": "x", "source": "ai_inference",
             "contradicts": ["b"]}
        b = {"id": "b", "content": "y", "source": "ai_inference",
             "contradicts": ["a"]}
        # _append_conflicts filters to sourced records first; the raw grouping
        # still finds the pair, but the append path drops unsourced material.
        prompt_parts = []
        CompanionOrchestrator._conflict_groups([a, b])  # no crash
        self.assertEqual(CompanionOrchestrator._sourced([a, b]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
