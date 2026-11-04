"""
Tests for ir.rerank.

Ordering, depth handling and rank-delta bookkeeping are tested with a stub
that supplies scores directly, so no model is loaded. `TestRealCrossEncoder`
then loads the real model once to verify the behaviour the stage exists for:
promoting a passage that answers the question above one that is merely on the
same topic.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.rerank import (  # noqa: E402
    CrossEncoderReranker,
    IdentityReranker,
    RerankedResult,
    Reranker,
)


class StubReranker(CrossEncoderReranker):
    """Returns scores from a text -> score table instead of running a model."""

    def __init__(self, table: dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.table = table
        self.score_calls = 0

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        self.score_calls += 1
        return [self.table.get(t, 0.0) for t in texts]


# Candidates arrive in a deliberately poor fused order: the best passage is
# last. The reranker must invert this.
TEXTS = {
    'c1': 'weak match',
    'c2': 'medium match',
    'c3': 'the actual answer',
}
CANDIDATES = ['c1', 'c2', 'c3']
SCORES = {'weak match': -2.0, 'medium match': 1.0, 'the actual answer': 8.0}


class TestOrdering(unittest.TestCase):

    def setUp(self):
        self.reranker = StubReranker(SCORES, top_n=10)

    def test_reorders_by_cross_encoder_score(self):
        results = self.reranker.rerank('q', CANDIDATES, TEXTS)
        self.assertEqual([r.doc_id for r in results], ['c3', 'c2', 'c1'])

    def test_assigns_ranks_from_one(self):
        results = self.reranker.rerank('q', CANDIDATES, TEXTS)
        self.assertEqual([r.rank for r in results], [1, 2, 3])

    def test_records_the_incoming_rank(self):
        results = {r.doc_id: r.previous_rank
                   for r in self.reranker.rerank('q', CANDIDATES, TEXTS)}
        self.assertEqual(results, {'c1': 1, 'c2': 2, 'c3': 3})

    def test_rank_delta_is_positive_for_a_promoted_passage(self):
        results = {r.doc_id: r.rank_delta
                   for r in self.reranker.rerank('q', CANDIDATES, TEXTS)}
        self.assertEqual(results['c3'], 2)   # rank 3 -> 1
        self.assertEqual(results['c1'], -2)  # rank 1 -> 3
        self.assertEqual(results['c2'], 0)

    def test_rank_deltas_sum_to_zero(self):
        # A permutation cannot create or destroy rank positions.
        results = self.reranker.rerank('q', CANDIDATES, TEXTS)
        self.assertEqual(sum(r.rank_delta for r in results), 0)

    def test_carries_the_score(self):
        results = {r.doc_id: r.score
                   for r in self.reranker.rerank('q', CANDIDATES, TEXTS)}
        self.assertAlmostEqual(results['c3'], 8.0, delta=1e-9)

    def test_ties_break_deterministically_by_doc_id(self):
        texts = {'zz': 'same', 'aa': 'same'}
        reranker = StubReranker({'same': 1.0})
        results = reranker.rerank('q', ['zz', 'aa'], texts)
        self.assertEqual([r.doc_id for r in results], ['aa', 'zz'])

    def test_marks_reranked_results(self):
        results = self.reranker.rerank('q', CANDIDATES, TEXTS)
        self.assertTrue(all(r.reranked for r in results))


class TestDepth(unittest.TestCase):

    def test_only_the_head_is_reranked(self):
        # top_n=2: c1 and c2 are scored and swapped; c3 keeps its position
        # below them even though it would have scored highest. This is the
        # hard ceiling top_n imposes.
        reranker = StubReranker(SCORES, top_n=2)
        results = reranker.rerank('q', CANDIDATES, TEXTS)
        self.assertEqual([r.doc_id for r in results], ['c2', 'c1', 'c3'])

    def test_tail_is_marked_not_reranked(self):
        reranker = StubReranker(SCORES, top_n=2)
        results = {r.doc_id: r.reranked
                   for r in reranker.rerank('q', CANDIDATES, TEXTS)}
        self.assertTrue(results['c1'])
        self.assertTrue(results['c2'])
        self.assertFalse(results['c3'])

    def test_tail_keeps_relative_order(self):
        reranker = StubReranker(SCORES, top_n=1)
        results = reranker.rerank('q', ['c1', 'c2', 'c3'], TEXTS)
        self.assertEqual([r.doc_id for r in results], ['c1', 'c2', 'c3'])

    def test_top_n_zero_reranks_nothing(self):
        reranker = StubReranker(SCORES, top_n=0)
        results = reranker.rerank('q', CANDIDATES, TEXTS)
        self.assertEqual([r.doc_id for r in results], CANDIDATES)
        self.assertFalse(any(r.reranked for r in results))
        self.assertEqual(reranker.score_calls, 0)

    def test_top_n_beyond_candidate_count_is_safe(self):
        reranker = StubReranker(SCORES, top_n=500)
        results = reranker.rerank('q', CANDIDATES, TEXTS)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.reranked for r in results))

    def test_per_call_top_n_overrides_the_default(self):
        reranker = StubReranker(SCORES, top_n=10)
        results = reranker.rerank('q', CANDIDATES, TEXTS, top_n=2)
        self.assertEqual([r.doc_id for r in results], ['c2', 'c1', 'c3'])

    def test_scores_in_a_single_batched_call(self):
        reranker = StubReranker(SCORES, top_n=10)
        reranker.rerank('q', CANDIDATES, TEXTS)
        self.assertEqual(reranker.score_calls, 1)


class TestEdgeCases(unittest.TestCase):

    def test_empty_candidates(self):
        self.assertEqual(StubReranker(SCORES).rerank('q', [], TEXTS), [])

    def test_missing_text_is_passed_through_not_dropped(self):
        # Losing a candidate because of a lookup gap would silently change
        # recall, so an absent text is scored as empty instead.
        reranker = StubReranker(SCORES, top_n=10)
        results = reranker.rerank('q', ['c3', 'absent'], {'c3': 'the actual answer'})
        self.assertEqual({r.doc_id for r in results}, {'c3', 'absent'})
        self.assertEqual(results[0].doc_id, 'c3')

    def test_empty_text_table(self):
        reranker = StubReranker(SCORES, top_n=10)
        results = reranker.rerank('q', CANDIDATES, {})
        self.assertEqual(len(results), 3)


class TestProbability(unittest.TestCase):

    def test_zero_logit_is_one_half(self):
        result = RerankedResult('d', 0.0, 1, 1)
        self.assertAlmostEqual(result.probability, 0.5, delta=1e-12)

    def test_bounded_in_unit_interval(self):
        for score in (-100.0, -8.0, 0.0, 8.0, 100.0):
            probability = RerankedResult('d', score, 1, 1).probability
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_monotone_in_the_logit(self):
        low = RerankedResult('d', -3.0, 1, 1).probability
        high = RerankedResult('d', 3.0, 1, 1).probability
        self.assertLess(low, high)

    def test_extreme_logits_do_not_overflow(self):
        # math.exp(1000) would raise OverflowError without the clamp.
        self.assertFalse(math.isnan(RerankedResult('d', 1e9, 1, 1).probability))
        self.assertFalse(math.isnan(RerankedResult('d', -1e9, 1, 1).probability))


class TestIdentityReranker(unittest.TestCase):

    def test_preserves_order(self):
        results = IdentityReranker().rerank('q', CANDIDATES, TEXTS)
        self.assertEqual([r.doc_id for r in results], CANDIDATES)

    def test_all_deltas_are_zero(self):
        results = IdentityReranker().rerank('q', CANDIDATES, TEXTS)
        self.assertTrue(all(r.rank_delta == 0 for r in results))

    def test_marks_nothing_as_reranked(self):
        results = IdentityReranker().rerank('q', CANDIDATES, TEXTS)
        self.assertFalse(any(r.reranked for r in results))

    def test_is_a_reranker(self):
        self.assertIsInstance(IdentityReranker(), Reranker)

    def test_empty_candidates(self):
        self.assertEqual(IdentityReranker().rerank('q', [], {}), [])


class TestRealCrossEncoder(unittest.TestCase):
    """
    One real-model test for the behaviour this stage exists to provide.

    Problem #3, reproduced from a failure observed on this project's own
    corpus: lexical retrieval ranked "In the parallel combination, the total
    capacitance C is C1 + C2 + C3" third for the query "electrostatic
    shielding conductor", above the passage that actually defines shielding.
    Both are electrostatics; only one answers the question.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("sentence-transformers not installed")
        cls.reranker = CrossEncoderReranker(device='cpu', top_n=10)

    def test_promotes_the_answering_passage_over_the_topical_one(self):
        texts = {
            'capacitance': 'In the parallel combination, the total '
                           'capacitance C is C = C1 + C2 + C3 where C1, C2 '
                           'and C3 are individual capacitances.',
            'shielding': 'The electric field inside a cavity of any conductor '
                         'is zero. All charges reside only on the outer '
                         'surface of a conductor with a cavity, which is used '
                         'to protect sensitive instruments from external '
                         'electric fields.',
            'unrelated': 'Overgrazing is one of the main reasons for land '
                         'degradation in Gujarat and Rajasthan.',
        }
        # Deliberately hand the reranker the wrong order.
        candidates = ['capacitance', 'unrelated', 'shielding']
        results = self.reranker.rerank(
            'what is electrostatic shielding', candidates, texts)

        self.assertEqual(results[0].doc_id, 'shielding')
        promoted = next(r for r in results if r.doc_id == 'shielding')
        self.assertGreater(promoted.rank_delta, 0)

    def test_scores_an_answer_above_an_off_topic_passage(self):
        scores = self.reranker.score_pairs(
            'what is electrostatic shielding',
            ['The field inside a conductor cavity is zero, shielding its '
             'interior from external fields.',
             'Overgrazing causes land degradation.'])
        self.assertGreater(scores[0], scores[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
