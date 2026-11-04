"""
Correctness tests for ir.indexes.bm25.

Three layers of evidence:

  1. Hand-computed scores on a 3-document corpus, worked out from the BM25
     formula with a calculator. Verifies the implementation against the
     definition.
  2. Parameter behaviour: k1 = 0 must collapse to binary presence, b = 0 must
     switch off length normalisation. These are structural properties of the
     formula, so they catch sign and placement errors that a single numeric
     check can miss.
  3. Exact agreement with rank_bm25 on the 'okapi' IDF variant, including its
     negative-IDF flooring. Skipped automatically if rank_bm25 is absent.

The hand-computed section uses `fit_tokenized` so no stemming or stopword
decision can affect the arithmetic.

Run:  venv/Scripts/python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.indexes.bm25 import BM25Index  # noqa: E402

TOL = 1e-6

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture
#
#   d1: cat sat mat      (len 3)
#   d2: dog sat          (len 2)
#   d3: cat dog cat      (len 3)
#
#   N = 3, avgdl = 8/3 = 2.6666667
#   df: cat 2, sat 2, dog 2, mat 1
#
#   lucene idf = ln(1 + (N - df + 0.5) / (df + 0.5))
#       df=2 -> ln(1 + 1.5/2.5) = ln(1.6)      = 0.47000363
#       df=1 -> ln(1 + 2.5/1.5) = ln(2.6666667) = 0.98082925
# ──────────────────────────────────────────────────────────────────────────────
CORPUS = {
    'd1': ['cat', 'sat', 'mat'],
    'd2': ['dog', 'sat'],
    'd3': ['cat', 'dog', 'cat'],
}
IDF_DF2 = math.log(1.6)
IDF_DF1 = math.log(8 / 3)


def bm25_reference(idf: float, tf: int, doc_len: int, avgdl: float = 8 / 3,
                   k1: float = 1.5, b: float = 0.75) -> float:
    """
    The BM25 term score transcribed straight from the formula.

    Used alongside the decimal literals in the tests below. The literal pins
    the value so a silent regression is caught; this function documents where
    that literal comes from and guards against a typo in it. Written out here
    rather than imported, so it is genuinely independent of the code
    under test.
    """
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))


class TestIndexStructure(unittest.TestCase):

    def setUp(self):
        self.ix = BM25Index().fit_tokenized(CORPUS)

    def test_document_count(self):
        self.assertEqual(len(self.ix), 3)

    def test_document_lengths(self):
        self.assertEqual(self.ix.doc_len, [3, 2, 3])

    def test_average_document_length(self):
        self.assertAlmostEqual(self.ix.avgdl, 8 / 3, delta=TOL)

    def test_document_frequencies(self):
        self.assertEqual(self.ix.df,
                         {'cat': 2, 'sat': 2, 'mat': 1, 'dog': 2})

    def test_vocabulary_size(self):
        self.assertEqual(self.ix.vocabulary_size, 4)

    def test_postings_record_term_frequency(self):
        # 'cat' appears once in d1 (index 0) and twice in d3 (index 2).
        self.assertEqual(sorted(self.ix.postings['cat']), [(0, 1), (2, 2)])

    def test_lucene_idf_values(self):
        self.assertAlmostEqual(self.ix.idf['cat'], IDF_DF2, delta=TOL)
        self.assertAlmostEqual(self.ix.idf['mat'], IDF_DF1, delta=TOL)

    def test_lucene_idf_always_positive(self):
        # A term in every document still gets idf > 0 under the lucene form.
        ix = BM25Index().fit_tokenized({'a': ['x'], 'b': ['x'], 'c': ['x']})
        self.assertGreater(ix.idf['x'], 0.0)


class TestHandComputedScores(unittest.TestCase):
    """
    k1 = 1.5, b = 0.75, avgdl = 2.6666667

    length_norm(d) = k1 * (1 - b + b * |d| / avgdl)
        |d| = 3 -> 1.5 * (0.25 + 0.75 * 1.125) = 1.5 * 1.09375 = 1.640625
        |d| = 2 -> 1.5 * (0.25 + 0.75 * 0.75)  = 1.5 * 0.8125  = 1.21875

    score = idf * tf * (k1 + 1) / (tf + length_norm)
    """

    def setUp(self):
        self.ix = BM25Index(k1=1.5, b=0.75).fit_tokenized(CORPUS)

    def test_score_single_occurrence(self):
        # d1, 'cat': tf=1, |d|=3
        #   = 0.470003629 * 1 * 2.5 / (1 + 1.640625)
        #   = 1.175009073 / 2.640625 = 0.4449738502
        scores = self.ix.score_terms(['cat'])
        self.assertAlmostEqual(scores[0], 0.4449738502, delta=1e-9)
        self.assertAlmostEqual(scores[0], bm25_reference(IDF_DF2, 1, 3),
                               delta=1e-12)

    def test_score_double_occurrence(self):
        # d3, 'cat': tf=2, |d|=3
        #   = 0.470003629 * 2 * 2.5 / (2 + 1.640625)
        #   = 2.350018146 / 3.640625 = 0.6454985466
        scores = self.ix.score_terms(['cat'])
        self.assertAlmostEqual(scores[2], 0.6454985466, delta=1e-9)
        self.assertAlmostEqual(scores[2], bm25_reference(IDF_DF2, 2, 3),
                               delta=1e-12)

    def test_term_frequency_saturates(self):
        # Doubling tf from 1 to 2 raises the score by a factor of ~1.45,
        # not 2. That sub-linearity is the whole purpose of k1.
        scores = self.ix.score_terms(['cat'])
        ratio = scores[2] / scores[0]
        self.assertLess(ratio, 2.0)
        self.assertAlmostEqual(ratio, 1.4506437768, delta=1e-9)

    def test_shorter_document_scores_higher_at_equal_tf(self):
        # 'sat' occurs once in d1 (len 3) and once in d2 (len 2).
        #   d1 = 1.175009073 / 2.640625 = 0.4449738502
        #   d2 = 1.175009073 / 2.21875  = 0.5295815541
        scores = self.ix.score_terms(['sat'])
        self.assertAlmostEqual(scores[0], 0.4449738502, delta=1e-9)
        self.assertAlmostEqual(scores[1], 0.5295815541, delta=1e-9)
        self.assertAlmostEqual(scores[1], bm25_reference(IDF_DF2, 1, 2),
                               delta=1e-12)
        self.assertGreater(scores[1], scores[0])

    def test_only_matching_documents_are_scored(self):
        # 'mat' appears only in d1, so d2 and d3 are absent from the result
        # rather than present with score 0. This is the inverted index doing
        # its job.
        scores = self.ix.score_terms(['mat'])
        self.assertEqual(set(scores), {0})

    def test_multi_term_query_sums_contributions(self):
        cat = self.ix.score_terms(['cat'])
        sat = self.ix.score_terms(['sat'])
        both = self.ix.score_terms(['cat', 'sat'])
        self.assertAlmostEqual(both[0], cat[0] + sat[0], delta=TOL)

    def test_repeated_query_term_counts_twice(self):
        once = self.ix.score_terms(['cat'])
        twice = self.ix.score_terms(['cat', 'cat'])
        self.assertAlmostEqual(twice[0], 2 * once[0], delta=TOL)

    def test_unknown_term_scores_nothing(self):
        self.assertEqual(self.ix.score_terms(['quantum']), {})


class TestParameterBehaviour(unittest.TestCase):

    def test_k1_zero_collapses_to_binary_presence(self):
        # With k1 = 0: score = idf * tf * 1 / (tf + 0) = idf, for any tf.
        # So d1 (tf=1) and d3 (tf=2) must tie exactly.
        ix = BM25Index(k1=0.0).fit_tokenized(CORPUS)
        scores = ix.score_terms(['cat'])
        self.assertAlmostEqual(scores[0], IDF_DF2, delta=TOL)
        self.assertAlmostEqual(scores[2], IDF_DF2, delta=TOL)

    def test_b_zero_disables_length_normalisation(self):
        # With b = 0: length_norm = k1 for every document, so 'sat' with tf=1
        # scores identically in d1 (len 3) and d2 (len 2).
        ix = BM25Index(b=0.0).fit_tokenized(CORPUS)
        scores = ix.score_terms(['sat'])
        self.assertAlmostEqual(scores[0], scores[1], delta=TOL)

    def test_b_one_normalises_fully(self):
        # Sanity: with b = 1 the shorter document's advantage is largest.
        ix0 = BM25Index(b=0.0).fit_tokenized(CORPUS)
        ix1 = BM25Index(b=1.0).fit_tokenized(CORPUS)
        gap0 = ix0.score_terms(['sat'])[1] - ix0.score_terms(['sat'])[0]
        gap1 = ix1.score_terms(['sat'])[1] - ix1.score_terms(['sat'])[0]
        self.assertAlmostEqual(gap0, 0.0, delta=TOL)
        self.assertGreater(gap1, gap0)

    def test_higher_k1_saturates_later(self):
        # Larger k1 means tf keeps mattering for longer, so the tf=2 vs tf=1
        # ratio should grow with k1.
        ratios = []
        for k1 in (0.5, 1.5, 3.0):
            s = BM25Index(k1=k1).fit_tokenized(CORPUS).score_terms(['cat'])
            ratios.append(s[2] / s[0])
        self.assertLess(ratios[0], ratios[1])
        self.assertLess(ratios[1], ratios[2])

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            BM25Index(idf_variant='made-up')
        with self.assertRaises(ValueError):
            BM25Index(k1=-1.0)
        with self.assertRaises(ValueError):
            BM25Index(b=1.5)


class TestSearchAndExplain(unittest.TestCase):

    def setUp(self):
        self.ix = BM25Index().fit_tokenized(CORPUS)

    def test_search_orders_by_descending_score(self):
        results = self.ix.search_terms(['cat'], top_k=10)
        self.assertEqual([doc for doc, _ in results], ['d3', 'd1'])
        self.assertGreater(results[0][1], results[1][1])

    def test_search_respects_top_k(self):
        self.assertEqual(len(self.ix.search_terms(['cat', 'sat'], top_k=1)), 1)

    def test_ties_break_deterministically_by_doc_id(self):
        # k1 = 0 makes d1 and d3 tie on 'cat'; the id ordering must be stable
        # so repeated experiment runs produce identical rankings.
        ix = BM25Index(k1=0.0).fit_tokenized(CORPUS)
        first = ix.search_terms(['cat'], top_k=10)
        self.assertEqual([d for d, _ in first], ['d1', 'd3'])
        self.assertEqual(first, ix.search_terms(['cat'], top_k=10))

    def test_explain_total_matches_score(self):
        ix = BM25Index(remove_stopwords=False, stem=False).fit(
            {'d1': 'cat sat mat', 'd2': 'dog sat', 'd3': 'cat dog cat'})
        breakdown = ix.explain('cat sat', 'd1')
        direct = dict(ix.search('cat sat', top_k=10))
        self.assertAlmostEqual(breakdown['total'], direct['d1'], delta=TOL)

    def test_explain_lists_terms_by_contribution(self):
        ix = BM25Index(remove_stopwords=False, stem=False).fit(
            {'d1': 'cat sat mat', 'd2': 'dog sat', 'd3': 'cat dog cat'})
        rows = ix.explain('cat mat', 'd1')['terms']
        contributions = [r['contribution'] for r in rows]
        self.assertEqual(contributions, sorted(contributions, reverse=True))
        # 'mat' is rarer (df=1) so it must contribute more than 'cat' (df=2).
        self.assertEqual(rows[0]['term'], 'mat')

    def test_explain_reports_zero_for_absent_term(self):
        ix = BM25Index(remove_stopwords=False, stem=False).fit(
            {'d1': 'cat sat mat', 'd2': 'dog sat', 'd3': 'cat dog cat'})
        rows = {r['term']: r for r in ix.explain('cat dog', 'd1')['terms']}
        self.assertEqual(rows['dog']['tf'], 0)
        self.assertAlmostEqual(rows['dog']['contribution'], 0.0, delta=TOL)

    def test_explain_rejects_unknown_document(self):
        with self.assertRaises(KeyError):
            self.ix.explain('cat', 'nonexistent')


class TestEdgeCases(unittest.TestCase):

    def test_empty_corpus(self):
        ix = BM25Index().fit_tokenized({})
        self.assertEqual(len(ix), 0)
        self.assertEqual(ix.avgdl, 0.0)
        self.assertEqual(ix.search_terms(['cat']), [])

    def test_empty_document(self):
        ix = BM25Index().fit_tokenized({'d1': [], 'd2': ['cat']})
        self.assertEqual(ix.doc_len, [0, 1])
        self.assertEqual([d for d, _ in ix.search_terms(['cat'])], ['d2'])

    def test_empty_query(self):
        ix = BM25Index().fit_tokenized(CORPUS)
        self.assertEqual(ix.search_terms([]), [])

    def test_fit_applies_tokenizer(self):
        # Stemming should conflate 'capacitors' with 'capacitor'.
        ix = BM25Index().fit({'d1': 'Capacitors store charge',
                              'd2': 'Resistors dissipate power'})
        self.assertEqual([d for d, _ in ix.search('capacitor', top_k=5)], ['d1'])


class TestRankBM25Equivalence(unittest.TestCase):
    """
    Exact agreement with rank_bm25.BM25Okapi.

    rank_bm25 uses the 'okapi' IDF form with negative values floored at
    epsilon * mean_idf, so we configure our index identically. On the 3-document
    fixture every df=2 term has a negative raw IDF, which means this test also
    exercises the flooring path -- a stricter check than a corpus where all
    IDFs happen to be positive.
    """

    def setUp(self):
        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
        except ImportError:
            self.skipTest("rank_bm25 not installed")

    def _compare(self, corpus: dict[str, list[str]], query: list[str]):
        from rank_bm25 import BM25Okapi

        doc_ids = list(corpus)
        reference = BM25Okapi([corpus[d] for d in doc_ids],
                              k1=1.5, b=0.75, epsilon=0.25)
        expected = reference.get_scores(query)

        mine = BM25Index(k1=1.5, b=0.75, idf_variant='okapi',
                         epsilon=0.25).fit_tokenized(corpus)
        actual = mine.score_terms(query)

        for i, doc_id in enumerate(doc_ids):
            self.assertAlmostEqual(
                actual.get(i, 0.0), float(expected[i]), delta=1e-9,
                msg=f"score mismatch on {doc_id} for query {query}")

    def test_matches_on_small_corpus(self):
        self._compare(CORPUS, ['cat'])
        self._compare(CORPUS, ['sat'])
        self._compare(CORPUS, ['cat', 'dog'])
        self._compare(CORPUS, ['mat'])

    def test_matches_with_repeated_query_term(self):
        self._compare(CORPUS, ['cat', 'cat', 'dog'])

    def test_matches_on_larger_varied_corpus(self):
        import random
        rng = random.Random(7)
        vocab = [f'term{i}' for i in range(40)]
        corpus = {
            f'doc{j}': [rng.choice(vocab) for _ in range(rng.randint(5, 60))]
            for j in range(50)
        }
        for _ in range(10):
            query = [rng.choice(vocab) for _ in range(rng.randint(1, 4))]
            self._compare(corpus, query)

    def test_idf_values_match(self):
        from rank_bm25 import BM25Okapi
        doc_ids = list(CORPUS)
        reference = BM25Okapi([CORPUS[d] for d in doc_ids],
                              k1=1.5, b=0.75, epsilon=0.25)
        mine = BM25Index(idf_variant='okapi', epsilon=0.25).fit_tokenized(CORPUS)
        for term in mine.df:
            self.assertAlmostEqual(mine.idf[term], reference.idf[term],
                                   delta=1e-9, msg=f"idf mismatch on {term!r}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
