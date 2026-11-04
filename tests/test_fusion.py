"""
Tests for ir.fusion.

The RRF fixture is the worked example from the project notes:

    BM25  : D7, D3, D10, D2
    Dense : D3, D7, D8, D10

With k = 60 the fused scores are, by hand:

    d7  = 1/61 + 1/62 = 0.0163934426 + 0.0161290323 = 0.0325224749
    d3  = 1/62 + 1/61 = 0.0161290323 + 0.0163934426 = 0.0325224749   (tie)
    d10 = 1/63 + 1/64 = 0.0158730159 + 0.0156250000 = 0.0314980159
    d8  = 1/63                                      = 0.0158730159
    d2  = 1/64                                      = 0.0156250000

so the fused order is d3, d7 (tie broken by id), d10, d8, d2.

Note d8 outranks d2 even though both were returned by only one retriever:
rank 3 in one list beats rank 4 in another. That is RRF working on positions
alone, with no score comparison anywhere.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.fusion import (  # noqa: E402
    RRF_K,
    fuse,
    normalise,
    reciprocal_rank_fusion,
    score_fusion,
)

TOL = 1e-12

# The notes' example. Scores are arbitrary and deliberately on different
# scales, to prove RRF never looks at them.
RUNS = {
    'bm25': [('d7', 18.4), ('d3', 12.1), ('d10', 9.7), ('d2', 4.2)],
    'dense': [('d3', 0.91), ('d7', 0.88), ('d8', 0.80), ('d10', 0.74)],
}

RRF_D7 = 1 / 61 + 1 / 62
RRF_D10 = 1 / 63 + 1 / 64


class TestNormalise(unittest.TestCase):

    def test_minmax_maps_to_unit_range(self):
        self.assertEqual(normalise([10.0, 5.0, 0.0], 'minmax'), [1.0, 0.5, 0.0])

    def test_minmax_identical_scores_give_neutral_weight(self):
        # No spread means no evidence to rank on; 0.5 for all beats dividing
        # by zero or silently returning 1.0 for everything.
        self.assertEqual(normalise([5.0, 5.0, 5.0], 'minmax'), [0.5, 0.5, 0.5])

    def test_zscore_centres_and_scales(self):
        # mean 2, population stdev sqrt(2/3) = 0.81649658
        result = normalise([1.0, 2.0, 3.0], 'zscore')
        self.assertAlmostEqual(result[0], -1.2247449, delta=1e-6)
        self.assertAlmostEqual(result[1], 0.0, delta=1e-12)
        self.assertAlmostEqual(result[2], 1.2247449, delta=1e-6)

    def test_zscore_identical_scores_give_neutral_weight(self):
        self.assertEqual(normalise([2.0, 2.0], 'zscore'), [0.5, 0.5])

    def test_none_passes_through(self):
        self.assertEqual(normalise([3.0, 1.0], 'none'), [3.0, 1.0])

    def test_empty_input(self):
        self.assertEqual(normalise([], 'minmax'), [])

    def test_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            normalise([1.0], 'softmax')


class TestReciprocalRankFusion(unittest.TestCase):

    def test_fused_order_matches_hand_computation(self):
        results = reciprocal_rank_fusion(RUNS)
        self.assertEqual([r.doc_id for r in results],
                         ['d3', 'd7', 'd10', 'd8', 'd2'])

    def test_fused_scores_match_hand_computation(self):
        by_id = {r.doc_id: r.score for r in reciprocal_rank_fusion(RUNS)}
        self.assertAlmostEqual(by_id['d7'], RRF_D7, delta=TOL)
        self.assertAlmostEqual(by_id['d3'], RRF_D7, delta=TOL)
        self.assertAlmostEqual(by_id['d10'], RRF_D10, delta=TOL)
        self.assertAlmostEqual(by_id['d8'], 1 / 63, delta=TOL)
        self.assertAlmostEqual(by_id['d2'], 1 / 64, delta=TOL)

    def test_single_retriever_rank_three_beats_other_rank_four(self):
        # d8 appears only in the dense run at rank 3; d2 only in bm25 at
        # rank 4. Position alone decides.
        results = {r.doc_id: r.rank for r in reciprocal_rank_fusion(RUNS)}
        self.assertLess(results['d8'], results['d2'])

    def test_ignores_score_magnitude_entirely(self):
        # Multiply every BM25 score by 1000 and shrink the dense ones: the
        # fused ranking must not move, because only ranks are used. This is
        # the property that makes normalisation unnecessary.
        inflated = {
            'bm25': [(d, s * 1000) for d, s in RUNS['bm25']],
            'dense': [(d, s * 0.001) for d, s in RUNS['dense']],
        }
        self.assertEqual([r.doc_id for r in reciprocal_rank_fusion(RUNS)],
                         [r.doc_id for r in reciprocal_rank_fusion(inflated)])

    def test_ranks_are_assigned_from_one(self):
        results = reciprocal_rank_fusion(RUNS)
        self.assertEqual([r.rank for r in results], [1, 2, 3, 4, 5])

    def test_ties_break_deterministically_by_doc_id(self):
        first = reciprocal_rank_fusion(RUNS)
        second = reciprocal_rank_fusion(RUNS)
        self.assertEqual([r.doc_id for r in first], [r.doc_id for r in second])
        # d3 precedes d7 on the tie purely by id ordering.
        self.assertEqual(first[0].doc_id, 'd3')

    def test_top_k_truncates(self):
        self.assertEqual(len(reciprocal_rank_fusion(RUNS, top_k=2)), 2)

    def test_weights_can_favour_one_retriever(self):
        # Doubling the lexical weight: d7 = 2/61 + 1/62 = 0.0489159,
        # d3 = 2/62 + 1/61 = 0.0486515, so d7 overtakes d3.
        results = reciprocal_rank_fusion(
            RUNS, weights={'bm25': 2.0, 'dense': 1.0})
        self.assertEqual(results[0].doc_id, 'd7')
        self.assertAlmostEqual(results[0].score, 2 / 61 + 1 / 62, delta=TOL)

    def test_default_k_is_sixty(self):
        self.assertEqual(RRF_K, 60)
        explicit = reciprocal_rank_fusion(RUNS, k=60)
        default = reciprocal_rank_fusion(RUNS)
        self.assertEqual([r.score for r in explicit], [r.score for r in default])

    def test_smaller_k_sharpens_top_rank_influence(self):
        # With k=1, rank 1 contributes 1/2 and rank 4 only 1/5: a much wider
        # spread than k=60 gives (1/61 vs 1/64).
        sharp = reciprocal_rank_fusion(RUNS, k=1)
        flat = reciprocal_rank_fusion(RUNS, k=60)
        sharp_spread = sharp[0].score - sharp[-1].score
        flat_spread = flat[0].score - flat[-1].score
        self.assertGreater(sharp_spread, flat_spread)

    def test_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(RUNS, k=0)

    def test_empty_runs(self):
        self.assertEqual(reciprocal_rank_fusion({}), [])

    def test_single_retriever_preserves_its_order(self):
        results = reciprocal_rank_fusion({'bm25': RUNS['bm25']})
        self.assertEqual([r.doc_id for r in results],
                         ['d7', 'd3', 'd10', 'd2'])


class TestContributions(unittest.TestCase):

    def test_records_one_contribution_per_retriever(self):
        d7 = next(r for r in reciprocal_rank_fusion(RUNS) if r.doc_id == 'd7')
        self.assertEqual(d7.n_retrievers, 2)
        self.assertEqual({c.retriever for c in d7.contributions},
                         {'bm25', 'dense'})

    def test_records_the_rank_each_retriever_gave(self):
        d7 = next(r for r in reciprocal_rank_fusion(RUNS) if r.doc_id == 'd7')
        ranks = {c.retriever: c.rank for c in d7.contributions}
        self.assertEqual(ranks, {'bm25': 1, 'dense': 2})

    def test_carries_raw_scores_for_the_trace(self):
        d7 = next(r for r in reciprocal_rank_fusion(RUNS) if r.doc_id == 'd7')
        raw = {c.retriever: c.raw_score for c in d7.contributions}
        self.assertAlmostEqual(raw['bm25'], 18.4, delta=1e-9)
        self.assertAlmostEqual(raw['dense'], 0.88, delta=1e-9)

    def test_contributions_sum_to_the_fused_score(self):
        for r in reciprocal_rank_fusion(RUNS):
            self.assertAlmostEqual(sum(c.contribution for c in r.contributions),
                                   r.score, delta=TOL)

    def test_shares_sum_to_one(self):
        d7 = next(r for r in reciprocal_rank_fusion(RUNS) if r.doc_id == 'd7')
        shares = d7.contribution_shares()
        self.assertAlmostEqual(sum(shares.values()), 1.0, delta=1e-12)
        # Rank 1 in bm25 beats rank 2 in dense, so lexical holds the larger share.
        self.assertGreater(shares['bm25'], shares['dense'])

    def test_single_retriever_doc_has_full_share(self):
        d8 = next(r for r in reciprocal_rank_fusion(RUNS) if r.doc_id == 'd8')
        self.assertEqual(d8.contribution_shares(), {'dense': 1.0})


class TestScoreFusion(unittest.TestCase):
    """
    bm25  : d1=10.0, d2=5.0, d3=0.0   -> minmax 1.0, 0.5, 0.0
    dense : d3=0.9,  d1=0.8, d2=0.7   -> minmax 1.0, 0.5, 0.0

    CombSUM:  d1 = 1.0 + 0.5 = 1.5
              d3 = 0.0 + 1.0 = 1.0
              d2 = 0.5 + 0.0 = 0.5
    """

    def setUp(self):
        self.runs = {
            'bm25': [('d1', 10.0), ('d2', 5.0), ('d3', 0.0)],
            'dense': [('d3', 0.9), ('d1', 0.8), ('d2', 0.7)],
        }

    def test_combsum_matches_hand_computation(self):
        results = score_fusion(self.runs, method='combsum',
                               normalisation='minmax')
        by_id = {r.doc_id: r.score for r in results}
        self.assertAlmostEqual(by_id['d1'], 1.5, delta=1e-12)
        self.assertAlmostEqual(by_id['d3'], 1.0, delta=1e-12)
        self.assertAlmostEqual(by_id['d2'], 0.5, delta=1e-12)
        self.assertEqual([r.doc_id for r in results], ['d1', 'd3', 'd2'])

    def test_combmnz_rewards_cross_retriever_agreement(self):
        # d1 is found by both retrievers but scores lower in total than d2,
        # which only one retriever found. CombSUM ranks d2 first; CombMNZ
        # doubles d1's score for the agreement and flips the order.
        runs = {'bm25': [('d1', 0.5)],
                'dense': [('d2', 1.0), ('d1', 0.4)]}
        combsum = score_fusion(runs, method='combsum', normalisation='none')
        combmnz = score_fusion(runs, method='combmnz', normalisation='none')
        self.assertEqual(combsum[0].doc_id, 'd2')
        self.assertEqual(combmnz[0].doc_id, 'd1')

    def test_combmnz_multiplies_by_retriever_count(self):
        runs = {'bm25': [('d1', 1.0)], 'dense': [('d1', 0.9)]}
        combsum = score_fusion(runs, method='combsum', normalisation='none')
        combmnz = score_fusion(runs, method='combmnz', normalisation='none')
        self.assertAlmostEqual(combmnz[0].score, combsum[0].score * 2,
                               delta=1e-12)

    def test_score_fusion_is_sensitive_to_scale_but_rrf_is_not(self):
        """
        The core argument for RRF, demonstrated side by side.

            bm25  : d1=1.0, d2=0.5, d3=0.2
            dense : d2=1.0, d3=0.6, d1=0.1

        Un-normalised CombSUM:  d2=1.5, d1=1.1, d3=0.8  -> d2, d1, d3
        Multiply BM25 by 100:   d1=100.1, d2=51.0, d3=20.6 -> d1, d2, d3

        The ranking flipped although neither retriever changed its mind about
        anything -- only the units changed. RRF sees identical ranks in both
        cases and so returns identical output.
        """
        runs = {'bm25': [('d1', 1.0), ('d2', 0.5), ('d3', 0.2)],
                'dense': [('d2', 1.0), ('d3', 0.6), ('d1', 0.1)]}
        inflated = {'bm25': [(d, s * 100) for d, s in runs['bm25']],
                    'dense': runs['dense']}

        combsum_balanced = [r.doc_id for r in score_fusion(
            runs, method='combsum', normalisation='none')]
        combsum_inflated = [r.doc_id for r in score_fusion(
            inflated, method='combsum', normalisation='none')]
        self.assertEqual(combsum_balanced, ['d2', 'd1', 'd3'])
        self.assertEqual(combsum_inflated, ['d1', 'd2', 'd3'])
        self.assertNotEqual(combsum_balanced, combsum_inflated)

        rrf_balanced = [r.doc_id for r in reciprocal_rank_fusion(runs)]
        rrf_inflated = [r.doc_id for r in reciprocal_rank_fusion(inflated)]
        self.assertEqual(rrf_balanced, ['d2', 'd1', 'd3'])
        self.assertEqual(rrf_balanced, rrf_inflated)

    def test_minmax_normalisation_mitigates_but_distorts(self):
        # Min-max rescues the scale problem, at the cost of forcing every
        # retriever's top hit to exactly 1.0 -- so a retriever that found
        # nothing good looks as confident as one that found a perfect match.
        runs = {'bm25': [('d1', 100.0), ('d2', 99.0)],
                'dense': [('d3', 0.02), ('d4', 0.01)]}
        results = score_fusion(runs, method='combsum', normalisation='minmax')
        top = {r.doc_id: r.score for r in results}
        self.assertAlmostEqual(top['d1'], top['d3'], delta=1e-12)

    def test_weights_apply(self):
        runs = {'bm25': [('d1', 1.0)], 'dense': [('d2', 1.0)]}
        results = score_fusion(runs, method='combsum', normalisation='none',
                               weights={'bm25': 3.0, 'dense': 1.0})
        self.assertEqual(results[0].doc_id, 'd1')
        self.assertAlmostEqual(results[0].score, 3.0, delta=1e-12)

    def test_records_normalised_scores_in_contributions(self):
        results = score_fusion(self.runs, method='combsum',
                               normalisation='minmax')
        d1 = next(r for r in results if r.doc_id == 'd1')
        norm = {c.retriever: c.normalised_score for c in d1.contributions}
        self.assertAlmostEqual(norm['bm25'], 1.0, delta=1e-12)
        self.assertAlmostEqual(norm['dense'], 0.5, delta=1e-12)

    def test_skips_empty_runs(self):
        results = score_fusion({'bm25': [('d1', 1.0)], 'dense': []},
                               method='combsum', normalisation='none')
        self.assertEqual([r.doc_id for r in results], ['d1'])

    def test_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            score_fusion(self.runs, method='borda')


class TestFuseDispatcher(unittest.TestCase):

    def test_rrf_is_the_default(self):
        self.assertEqual([r.doc_id for r in fuse(RUNS)],
                         [r.doc_id for r in reciprocal_rank_fusion(RUNS)])

    def test_dispatches_to_combsum(self):
        runs = {'bm25': [('d1', 1.0)], 'dense': [('d1', 0.5)]}
        self.assertEqual([r.doc_id for r in fuse(runs, method='combsum')],
                         [r.doc_id for r in score_fusion(runs, method='combsum')])

    def test_dispatches_to_combmnz(self):
        runs = {'bm25': [('d1', 1.0)], 'dense': [('d2', 0.5)]}
        results = fuse(runs, method='combmnz')
        self.assertEqual(len(results), 2)

    def test_single_run_passes_through_with_contributions(self):
        # A BM25-only pipeline must produce the same result shape as a hybrid
        # one, so the explanation UI needs no special case.
        results = fuse({'bm25': RUNS['bm25']})
        self.assertEqual([r.doc_id for r in results],
                         ['d7', 'd3', 'd10', 'd2'])
        self.assertTrue(all(r.n_retrievers == 1 for r in results))

    def test_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            fuse(RUNS, method='magic')

    def test_top_k_applies_to_both_families(self):
        self.assertEqual(len(fuse(RUNS, method='rrf', top_k=3)), 3)
        self.assertEqual(len(fuse(RUNS, method='combsum', top_k=3)), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
