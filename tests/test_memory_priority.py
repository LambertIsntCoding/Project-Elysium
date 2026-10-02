"""Tests for the derived memory priority index (authority preserved)."""

import copy
import unittest

from astra import memory_priority as mp


def _mem(**kw):
    base = {
        "id": "m1", "content": "something", "type": "explicit_fact",
        "source": "explicit_user_statement", "confidence": 0.9,
        "status": "active", "timestamp": "2026-01-01T00:00:00+00:00",
        "reinforcement_count": 1, "use_count": 0,
    }
    base.update(kw)
    return base


class TestPriority(unittest.TestCase):
    def test_sourced_outranks_inferred_even_when_inferred_is_heavily_used(self):
        sourced = _mem(id="s", source="explicit_user_statement", use_count=0,
                       reinforcement_count=1, confidence=0.5)
        inferred = _mem(id="i", source="ai_inference", type="interpretation",
                        use_count=999, reinforcement_count=99, confidence=1.0)
        self.assertGreater(mp.score(sourced), mp.score(inferred))

    def test_priority_does_not_mutate_the_record(self):
        mem = _mem()
        before = copy.deepcopy(mem)
        mp.ranked([mem])
        self.assertEqual(mem, before)

    def test_dormant_record_is_penalised_but_kept(self):
        active = _mem(id="a")
        dormant = _mem(id="d", utility="dormant",
                       dormant_at="2026-01-01T00:00:00+00:00")
        self.assertGreater(mp.score(active), mp.score(dormant))
        # Dormant is a soft demotion: the record still produces a score.
        self.assertGreater(mp.score(dormant), 0.0)

    def test_open_question_gets_an_unresolved_bonus(self):
        plain = _mem(id="p", type="observation", source="ai_inference")
        question = _mem(id="q", type="open_question", source="ai_inference",
                        tags=["qstatus:open"])
        self.assertGreater(mp.score(question), mp.score(plain))

    def test_ranking_is_deterministic(self):
        mems = [_mem(id="b", confidence=0.7), _mem(id="a", confidence=0.7)]
        order1 = [m["id"] for m in mp.ranked(mems)]
        order2 = [m["id"] for m in mp.ranked(mems)]
        self.assertEqual(order1, order2)

    def test_work_bonus_scopes_to_active_work(self):
        w1 = _mem(id="w1", tags=["work:book-a"])
        w2 = _mem(id="w2", tags=["work:book-b"])
        ranked = mp.ranked([w1, w2], active_work="book-a")
        self.assertEqual(ranked[0]["id"], "w1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
