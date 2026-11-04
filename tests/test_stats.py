"""
Tests for ir.stats.

The randomisation test is checked on cases where the correct answer is known
by construction:

  * identical systems       -> every permutation is as extreme as the observed
                               zero difference, so p must be exactly 1.0
  * one system dominating   -> only a handful of the 2^n sign assignments can
                               match the observed mean, so p must be small
  * fixed seed              -> p must reproduce exactly, since reported
                               p-values have to be stable across runs
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.stats import (  # noqa: E402
    bonferroni_correction,
    compare_against_baseline,
    holm_correction,
    randomisation_test,
)


class TestRandomisationTest(unittest.TestCase):

    def test_identical_systems_give_p_of_one(self):
        # All per-query differences are 0, so |permuted mean| >= 0 always.
        a = {f'q{i}': 0.5 for i in range(10)}
        result = randomisation_test(a, dict(a), n_permutations=1000)
        self.assertEqual(result['delta'], 0.0)
        self.assertEqual(result['p_value'], 1.0)

    def test_strong_consistent_difference_is_significant(self):
        # A beats B on all 20 queries by a wide margin.
        a = {f'q{i}': 0.9 for i in range(20)}
        b = {f'q{i}': 0.1 for i in range(20)}
        result = randomisation_test(a, b, n_permutations=10_000)
        self.assertAlmostEqual(result['delta'], 0.8, delta=1e-9)
        self.assertLess(result['p_value'], 0.01)

    def test_noisy_difference_is_not_significant(self):
        # A wins on half the queries and loses on the other half by the same
        # amount: the means are equal, so nothing should be detected.
        a, b = {}, {}
        for i in range(20):
            a[f'q{i}'], b[f'q{i}'] = (0.8, 0.2) if i % 2 == 0 else (0.2, 0.8)
        result = randomisation_test(a, b, n_permutations=10_000)
        self.assertAlmostEqual(result['delta'], 0.0, delta=1e-9)
        self.assertGreater(result['p_value'], 0.05)

    def test_reports_means_and_counts(self):
        a = {'q1': 1.0, 'q2': 0.0}
        b = {'q1': 0.0, 'q2': 0.0}
        result = randomisation_test(a, b, n_permutations=500)
        self.assertAlmostEqual(result['mean_a'], 0.5, delta=1e-9)
        self.assertAlmostEqual(result['mean_b'], 0.0, delta=1e-9)
        self.assertEqual(result['n_queries'], 2.0)
        self.assertEqual(result['n_permutations'], 500.0)

    def test_p_value_is_reproducible_with_fixed_seed(self):
        a = {f'q{i}': i / 10 for i in range(15)}
        b = {f'q{i}': (i / 10) * 0.8 for i in range(15)}
        first = randomisation_test(a, b, n_permutations=2000, seed=7)
        second = randomisation_test(a, b, n_permutations=2000, seed=7)
        self.assertEqual(first['p_value'], second['p_value'])

    def test_different_seeds_give_similar_p_values(self):
        a = {f'q{i}': 0.9 for i in range(20)}
        b = {f'q{i}': 0.4 for i in range(20)}
        p1 = randomisation_test(a, b, n_permutations=10_000, seed=1)['p_value']
        p2 = randomisation_test(a, b, n_permutations=10_000, seed=99)['p_value']
        self.assertLess(abs(p1 - p2), 0.02)

    def test_p_value_can_never_be_zero(self):
        # The (hits + 1) / (permutations + 1) estimator is honest: a finite
        # permutation sample cannot prove p = 0.
        a = {f'q{i}': 1.0 for i in range(30)}
        b = {f'q{i}': 0.0 for i in range(30)}
        self.assertGreater(randomisation_test(a, b, n_permutations=100)['p_value'], 0.0)

    def test_uses_only_shared_queries(self):
        a = {'q1': 1.0, 'q2': 1.0, 'extra': 99.0}
        b = {'q1': 0.0, 'q2': 0.0}
        result = randomisation_test(a, b, n_permutations=500)
        self.assertEqual(result['n_queries'], 2.0)

    def test_raises_with_no_shared_queries(self):
        with self.assertRaises(ValueError):
            randomisation_test({'a': 1.0}, {'b': 1.0})


class TestBonferroni(unittest.TestCase):

    def test_divides_alpha_by_number_of_tests(self):
        result = bonferroni_correction({'a': 0.01, 'b': 0.04, 'c': 0.5},
                                       alpha=0.05)
        for row in result.values():
            self.assertAlmostEqual(row['adjusted_alpha'], 0.05 / 3, delta=1e-12)

    def test_significance_uses_adjusted_threshold(self):
        # 0.04 would pass at alpha=0.05 alone, but not at 0.05/3 = 0.0167.
        result = bonferroni_correction({'a': 0.01, 'b': 0.04, 'c': 0.5})
        self.assertEqual(result['a']['significant'], 1.0)
        self.assertEqual(result['b']['significant'], 0.0)
        self.assertEqual(result['c']['significant'], 0.0)

    def test_empty_input(self):
        self.assertEqual(bonferroni_correction({}), {})


class TestHolm(unittest.TestCase):

    def test_thresholds_step_down(self):
        # m=3: thresholds are alpha/3, alpha/2, alpha/1 by ascending p.
        result = holm_correction({'a': 0.001, 'b': 0.02, 'c': 0.4}, alpha=0.05)
        self.assertAlmostEqual(result['a']['adjusted_alpha'], 0.05 / 3, delta=1e-12)
        self.assertAlmostEqual(result['b']['adjusted_alpha'], 0.05 / 2, delta=1e-12)
        self.assertAlmostEqual(result['c']['adjusted_alpha'], 0.05 / 1, delta=1e-12)

    def test_ranks_by_ascending_p_value(self):
        result = holm_correction({'big': 0.4, 'small': 0.001, 'mid': 0.02})
        self.assertEqual(result['small']['rank'], 1.0)
        self.assertEqual(result['mid']['rank'], 2.0)
        self.assertEqual(result['big']['rank'], 3.0)

    def test_stops_at_first_non_rejection(self):
        # 0.001 <= 0.0167 passes. 0.03 > 0.025 fails, which must also block
        # the third test even though 0.04 <= 0.05.
        result = holm_correction({'a': 0.001, 'b': 0.03, 'c': 0.04}, alpha=0.05)
        self.assertEqual(result['a']['significant'], 1.0)
        self.assertEqual(result['b']['significant'], 0.0)
        self.assertEqual(result['c']['significant'], 0.0)

    def test_is_no_more_conservative_than_bonferroni(self):
        p_values = {'a': 0.001, 'b': 0.02, 'c': 0.03}
        holm = holm_correction(p_values)
        bonf = bonferroni_correction(p_values)
        for name in p_values:
            self.assertGreaterEqual(holm[name]['significant'],
                                    bonf[name]['significant'])

    def test_empty_input(self):
        self.assertEqual(holm_correction({}), {})


class TestCompareAgainstBaseline(unittest.TestCase):

    def setUp(self):
        # 20 queries. 'strong' beats the baseline everywhere; 'same' matches it.
        self.per_query = {
            'baseline': {f'q{i}': {'nDCG@10': 0.40} for i in range(20)},
            'strong': {f'q{i}': {'nDCG@10': 0.80} for i in range(20)},
            'same': {f'q{i}': {'nDCG@10': 0.40} for i in range(20)},
        }

    def test_omits_the_baseline_itself(self):
        result = compare_against_baseline(self.per_query, 'baseline')
        self.assertEqual(set(result), {'strong', 'same'})

    def test_detects_a_real_improvement(self):
        result = compare_against_baseline(self.per_query, 'baseline')
        self.assertAlmostEqual(result['strong']['delta'], 0.40, delta=1e-9)
        self.assertEqual(result['strong']['significant'], 1.0)

    def test_does_not_flag_an_identical_system(self):
        result = compare_against_baseline(self.per_query, 'baseline')
        self.assertAlmostEqual(result['same']['delta'], 0.0, delta=1e-9)
        self.assertEqual(result['same']['significant'], 0.0)

    def test_reports_both_means(self):
        result = compare_against_baseline(self.per_query, 'baseline')
        self.assertAlmostEqual(result['strong']['mean'], 0.80, delta=1e-9)
        self.assertAlmostEqual(result['strong']['baseline_mean'], 0.40, delta=1e-9)

    def test_applies_holm_correction(self):
        result = compare_against_baseline(self.per_query, 'baseline')
        self.assertIn('adjusted_alpha', result['strong'])

    def test_raises_on_unknown_baseline(self):
        with self.assertRaises(KeyError):
            compare_against_baseline(self.per_query, 'nonexistent')

    def test_respects_requested_measure(self):
        per_query = {
            'baseline': {f'q{i}': {'MRR': 0.2, 'nDCG@10': 0.9} for i in range(20)},
            'variant': {f'q{i}': {'MRR': 0.9, 'nDCG@10': 0.9} for i in range(20)},
        }
        mrr = compare_against_baseline(per_query, 'baseline', measure='MRR')
        ndcg = compare_against_baseline(per_query, 'baseline', measure='nDCG@10')
        self.assertEqual(mrr['variant']['significant'], 1.0)
        self.assertEqual(ndcg['variant']['significant'], 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
