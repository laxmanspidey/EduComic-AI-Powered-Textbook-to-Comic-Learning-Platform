"""
Correctness tests for ir.metrics, checked against values worked out by hand.

Every expected number in this file was computed manually from the formula in
the corresponding docstring, so the suite is independent evidence that the
implementation matches the textbook definitions -- not just that it is
self-consistent.

Run:  venv/Scripts/python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.metrics import (  # noqa: E402
    average_precision,
    dcg_at_k,
    evaluate_run,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    success_at_k,
)

TOL = 1e-4


class TestPrecisionRecall(unittest.TestCase):
    """
    ranking = [r1, n1, r2, n2, n3]
    relevant = {r1, r2, r3, r4}   (r3 and r4 are never retrieved)
    """

    def setUp(self):
        self.ranking = ['r1', 'n1', 'r2', 'n2', 'n3']
        self.qrels = {'r1': 1, 'n1': 0, 'r2': 1, 'n2': 0, 'n3': 0,
                      'r3': 1, 'r4': 1}

    def test_precision_at_1(self):
        # top-1 is r1, relevant -> 1/1
        self.assertAlmostEqual(precision_at_k(self.ranking, self.qrels, 1),
                               1.0, delta=TOL)

    def test_precision_at_5(self):
        # 2 relevant in top 5 -> 2/5
        self.assertAlmostEqual(precision_at_k(self.ranking, self.qrels, 5),
                               0.4, delta=TOL)

    def test_recall_at_5(self):
        # 2 of 4 known relevant retrieved -> 2/4
        self.assertAlmostEqual(recall_at_k(self.ranking, self.qrels, 5),
                               0.5, delta=TOL)

    def test_recall_at_1(self):
        # 1 of 4 -> 0.25
        self.assertAlmostEqual(recall_at_k(self.ranking, self.qrels, 1),
                               0.25, delta=TOL)

    def test_f1_at_5(self):
        # harmonic mean of 0.4 and 0.5 = 2*0.2/0.9 = 0.44444
        self.assertAlmostEqual(f1_at_k(self.ranking, self.qrels, 5),
                               0.444444, delta=TOL)

    def test_precision_denominator_is_k_not_result_count(self):
        # Only one document returned, but k=5. Documented behaviour: the
        # system is charged for the 4 it did not return, so P@5 = 1/5.
        self.assertAlmostEqual(precision_at_k(['r1'], {'r1': 1}, 5),
                               0.2, delta=TOL)

    def test_recall_zero_when_no_relevant_exist(self):
        self.assertEqual(recall_at_k(['a', 'b'], {'a': 0, 'b': 0}, 5), 0.0)

    def test_unjudged_documents_count_as_non_relevant(self):
        # 'x' and 'y' are absent from qrels -> treated as grade 0.
        self.assertAlmostEqual(precision_at_k(['x', 'y', 'r1'], {'r1': 1}, 3),
                               1 / 3, delta=TOL)


class TestGradedThreshold(unittest.TestCase):
    """rel_threshold collapses graded judgements for the binary measures."""

    def setUp(self):
        self.ranking = ['d1', 'd2']
        self.qrels = {'d1': 1, 'd2': 2}

    def test_threshold_1_counts_both(self):
        self.assertAlmostEqual(precision_at_k(self.ranking, self.qrels, 2,
                                              rel_threshold=1), 1.0, delta=TOL)

    def test_threshold_2_counts_only_highly_relevant(self):
        self.assertAlmostEqual(precision_at_k(self.ranking, self.qrels, 2,
                                              rel_threshold=2), 0.5, delta=TOL)

    def test_threshold_shifts_first_relevant_rank(self):
        # At threshold 1 the first relevant doc is at rank 1 -> RR = 1.0
        self.assertAlmostEqual(reciprocal_rank(self.ranking, self.qrels,
                                               rel_threshold=1), 1.0, delta=TOL)
        # At threshold 2 it moves to rank 2 -> RR = 0.5
        self.assertAlmostEqual(reciprocal_rank(self.ranking, self.qrels,
                                               rel_threshold=2), 0.5, delta=TOL)


class TestReciprocalRank(unittest.TestCase):

    def test_first_relevant_at_rank_2(self):
        self.assertAlmostEqual(reciprocal_rank(['a', 'b', 'c'], {'b': 1}),
                               0.5, delta=TOL)

    def test_no_relevant_found_is_zero(self):
        self.assertEqual(reciprocal_rank(['a', 'b', 'c'], {'z': 1}), 0.0)

    def test_empty_qrels_is_zero(self):
        self.assertEqual(reciprocal_rank(['a'], {}), 0.0)


class TestSuccessAtK(unittest.TestCase):

    def test_relevant_outside_cutoff(self):
        self.assertEqual(success_at_k(['n', 'n', 'r'], {'r': 1}, 2), 0.0)

    def test_relevant_inside_cutoff(self):
        self.assertEqual(success_at_k(['n', 'n', 'r'], {'r': 1}, 3), 1.0)


class TestAveragePrecision(unittest.TestCase):
    """
    ranking = [d1, d2, d3], relevant at ranks 1 and 3.

        P@1 = 1/1 = 1.0
        P@3 = 2/3 = 0.666667
        AP  = (1.0 + 0.666667) / R
    """

    def test_ap_all_relevant_retrieved(self):
        # R = 2  ->  AP = 1.666667 / 2 = 0.833333
        qrels = {'d1': 1, 'd2': 0, 'd3': 1}
        self.assertAlmostEqual(average_precision(['d1', 'd2', 'd3'], qrels),
                               0.833333, delta=TOL)

    def test_ap_penalises_unretrieved_relevant(self):
        # Same ranking, but a third relevant document exists and is never
        # returned. R = 3  ->  AP = 1.666667 / 3 = 0.555556.
        # This is the TREC convention: AP rewards recall, not just precision.
        qrels = {'d1': 1, 'd2': 0, 'd3': 1, 'd99': 1}
        self.assertAlmostEqual(average_precision(['d1', 'd2', 'd3'], qrels),
                               0.555556, delta=TOL)

    def test_ap_is_one_for_perfect_ranking(self):
        qrels = {'d1': 1, 'd2': 1}
        self.assertAlmostEqual(average_precision(['d1', 'd2', 'd3'], qrels),
                               1.0, delta=TOL)

    def test_ap_zero_when_no_relevant(self):
        self.assertEqual(average_precision(['d1'], {'d1': 0}), 0.0)


class TestDCG(unittest.TestCase):
    """
    ranking = [d1, d2, d3] with grades 0, 2, 1.

    exponential gain:
        DCG@3 = (2^0-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
              = 0 + 3/1.5849625 + 1/2
              = 1.8927893 + 0.5 = 2.3927893
    """

    def setUp(self):
        self.ranking = ['d1', 'd2', 'd3']
        self.qrels = {'d1': 0, 'd2': 2, 'd3': 1}

    def test_dcg_exponential(self):
        expected = 3 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(dcg_at_k(self.ranking, self.qrels, 3),
                               expected, delta=TOL)
        self.assertAlmostEqual(dcg_at_k(self.ranking, self.qrels, 3),
                               2.3927893, delta=TOL)

    def test_dcg_linear(self):
        # DCG = 0 + 2/log2(3) + 1/log2(4) = 1.2618595 + 0.5 = 1.7618595
        self.assertAlmostEqual(dcg_at_k(self.ranking, self.qrels, 3,
                                        gain='linear'),
                               1.7618595, delta=TOL)

    def test_dcg_rejects_unknown_gain(self):
        with self.assertRaises(ValueError):
            dcg_at_k(self.ranking, self.qrels, 3, gain='quadratic')


class TestNDCG(unittest.TestCase):
    """
    Same data as TestDCG. Ideal ranking uses grades [2, 1]:

        IDCG@3 = 3/log2(2) + 1/log2(3) = 3 + 0.6309298 = 3.6309298
        nDCG@3 = 2.3927893 / 3.6309298 = 0.6590018
    """

    def setUp(self):
        self.ranking = ['d1', 'd2', 'd3']
        self.qrels = {'d1': 0, 'd2': 2, 'd3': 1}

    def test_ndcg_exponential(self):
        self.assertAlmostEqual(ndcg_at_k(self.ranking, self.qrels, 3),
                               0.6590018, delta=TOL)

    def test_ndcg_linear_differs_from_exponential(self):
        # 1.7618595 / 2.6309298 = 0.6696718
        self.assertAlmostEqual(ndcg_at_k(self.ranking, self.qrels, 3,
                                         gain='linear'),
                               0.6696718, delta=TOL)
        self.assertNotAlmostEqual(
            ndcg_at_k(self.ranking, self.qrels, 3, gain='linear'),
            ndcg_at_k(self.ranking, self.qrels, 3, gain='exponential'),
            delta=TOL)

    def test_ndcg_is_one_for_ideal_ranking(self):
        # Highest grade first, then the lower one.
        self.assertAlmostEqual(ndcg_at_k(['d2', 'd3', 'd1'], self.qrels, 3),
                               1.0, delta=TOL)

    def test_ndcg_zero_when_nothing_relevant_exists(self):
        self.assertEqual(ndcg_at_k(['d1'], {'d1': 0}, 3), 0.0)

    def test_ndcg_bounded_by_one(self):
        for ranking in (['d1', 'd2', 'd3'], ['d3', 'd2', 'd1'],
                        ['d2', 'd1', 'd3'], ['d2', 'd3', 'd1']):
            self.assertLessEqual(ndcg_at_k(ranking, self.qrels, 3), 1.0 + TOL)


class TestEvaluateRun(unittest.TestCase):

    def test_skips_queries_with_no_relevant_documents(self):
        run = {'q1': ['d1'], 'q2': ['d2']}
        qrels = {'q1': {'d1': 1}, 'q2': {'d2': 0}}
        means, per_query = evaluate_run(run, qrels)
        self.assertEqual(set(per_query), {'q1'})

    def test_keeps_unjudged_queries_when_asked(self):
        run = {'q1': ['d1'], 'q2': ['d2']}
        qrels = {'q1': {'d1': 1}, 'q2': {'d2': 0}}
        _, per_query = evaluate_run(run, qrels, skip_unjudged_queries=False)
        self.assertEqual(set(per_query), {'q1', 'q2'})

    def test_reports_map_not_ap(self):
        run = {'q1': ['d1', 'd2']}
        qrels = {'q1': {'d1': 1, 'd2': 1}}
        means, _ = evaluate_run(run, qrels)
        self.assertIn('MAP', means)
        self.assertNotIn('AP', means)

    def test_mean_is_average_of_per_query_values(self):
        # q1 perfect (RR=1.0), q2 first relevant at rank 2 (RR=0.5).
        run = {'q1': ['a', 'b'], 'q2': ['x', 'y']}
        qrels = {'q1': {'a': 1}, 'q2': {'y': 1}}
        means, per_query = evaluate_run(run, qrels)
        self.assertAlmostEqual(per_query['q1']['MRR'], 1.0, delta=TOL)
        self.assertAlmostEqual(per_query['q2']['MRR'], 0.5, delta=TOL)
        self.assertAlmostEqual(means['MRR'], 0.75, delta=TOL)

    def test_empty_run_returns_empty_results(self):
        means, per_query = evaluate_run({}, {})
        self.assertEqual(means, {})
        self.assertEqual(per_query, {})

    def test_per_query_contains_all_requested_cutoffs(self):
        run = {'q1': ['a', 'b', 'c']}
        qrels = {'q1': {'a': 1}}
        _, per_query = evaluate_run(run, qrels, cutoffs=(1, 3))
        keys = set(per_query['q1'])
        for name in ('P@1', 'R@1', 'nDCG@1', 'Success@1',
                     'P@3', 'R@3', 'nDCG@3', 'Success@3', 'MRR', 'AP'):
            self.assertIn(name, keys)


if __name__ == '__main__':
    unittest.main(verbosity=2)
