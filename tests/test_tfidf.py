"""
Tests for ir.indexes.tfidf, including the properties that make the TF-IDF vs
BM25 comparison a controlled one.

The important structural test is
`TestComparabilityWithBM25.test_both_indexes_see_the_same_vocabulary`: if the
two indexes tokenised differently, any measured gap between them would be
partly a preprocessing artefact rather than a weighting difference, and the
classical-vs-neural comparison would not mean what the write-up claims.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.indexes.base import Retriever  # noqa: E402
from ir.indexes.bm25 import BM25Index  # noqa: E402
from ir.indexes.tfidf import TFIDFIndex  # noqa: E402

DOCS = {
    'p1': 'Capacitors store electric charge on conducting plates.',
    'p2': 'A conductor shields its interior from external electric fields.',
    'p3': 'Resistors dissipate electrical power as heat.',
    'p4': 'The electric field inside a conductor is zero at equilibrium.',
}


class TestFitAndSearch(unittest.TestCase):

    def setUp(self):
        self.ix = TFIDFIndex().fit(DOCS)

    def test_indexes_every_document(self):
        self.assertEqual(len(self.ix), 4)

    def test_builds_a_vocabulary(self):
        self.assertGreater(self.ix.vocabulary_size, 0)

    def test_finds_the_relevant_document(self):
        results = self.ix.search('capacitor charge', top_k=5)
        self.assertEqual(results[0][0], 'p1')

    def test_respects_top_k(self):
        self.assertLessEqual(len(self.ix.search('electric', top_k=2)), 2)

    def test_cosine_scores_lie_in_unit_interval(self):
        # TF-IDF weights are non-negative and rows are L2-normalised, so the
        # cosine cannot leave [0, 1].
        for _, score in self.ix.search('electric conductor field', top_k=10):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0 + 1e-9)

    def test_excludes_documents_sharing_no_term(self):
        # 'resistor' appears only in p3, so nothing else may be returned.
        results = self.ix.search('resistor', top_k=10)
        self.assertEqual([d for d, _ in results], ['p3'])

    def test_identical_query_and_document_scores_near_one(self):
        ix = TFIDFIndex().fit({'only': 'entropy increases spontaneously'})
        score = ix.search('entropy increases spontaneously', top_k=1)[0][1]
        self.assertAlmostEqual(score, 1.0, delta=1e-6)

    def test_ties_break_deterministically(self):
        # Two documents with identical content must order by id, stably.
        ix = TFIDFIndex().fit({'pb': 'same words here', 'pa': 'same words here'})
        first = ix.search('same words here', top_k=2)
        self.assertEqual([d for d, _ in first], ['pa', 'pb'])
        self.assertEqual(first, ix.search('same words here', top_k=2))

    def test_rarer_term_outranks_common_term(self):
        # 'electric' is in 3 of 4 documents; 'resistor' in 1. The document
        # matching the rarer term should win on a query containing both.
        results = self.ix.search('electric resistor', top_k=4)
        self.assertEqual(results[0][0], 'p3')


class TestEdgeCases(unittest.TestCase):

    def test_empty_corpus_returns_nothing(self):
        ix = TFIDFIndex().fit({})
        self.assertEqual(len(ix), 0)
        self.assertEqual(ix.vocabulary_size, 0)
        self.assertEqual(ix.search('anything'), [])

    def test_empty_query_returns_nothing(self):
        self.assertEqual(TFIDFIndex().fit(DOCS).search(''), [])

    def test_out_of_vocabulary_query_returns_nothing(self):
        # Nothing in the corpus relates to this; an empty list is the honest
        # answer rather than arbitrary low-scoring padding.
        self.assertEqual(TFIDFIndex().fit(DOCS).search('photosynthesis'), [])

    def test_stopword_only_query_returns_nothing(self):
        self.assertEqual(TFIDFIndex().fit(DOCS).search('the and of'), [])

    def test_search_before_fit_returns_nothing(self):
        self.assertEqual(TFIDFIndex().search('electric'), [])


class TestExplain(unittest.TestCase):

    def setUp(self):
        self.ix = TFIDFIndex().fit(DOCS)

    def test_contributions_sum_to_the_cosine_score(self):
        direct = dict(self.ix.search('electric conductor', top_k=10))
        breakdown = self.ix.explain('electric conductor', 'p2')
        self.assertAlmostEqual(breakdown['total'], direct['p2'], delta=1e-9)

    def test_terms_ordered_by_contribution(self):
        rows = self.ix.explain('electric conductor', 'p2')['terms']
        contributions = [r['contribution'] for r in rows]
        self.assertEqual(contributions, sorted(contributions, reverse=True))

    def test_reports_both_weights_per_term(self):
        rows = self.ix.explain('conductor', 'p2')['terms']
        self.assertTrue(rows)
        for row in rows:
            self.assertIn('query_weight', row)
            self.assertIn('doc_weight', row)

    def test_term_absent_from_document_contributes_zero(self):
        rows = {r['term']: r for r in
                self.ix.explain('electric resistor', 'p2')['terms']}
        # 'resistor' (stemmed) is not in p2.
        absent = [r for t, r in rows.items() if r['doc_weight'] == 0.0]
        self.assertTrue(absent)
        for row in absent:
            self.assertAlmostEqual(row['contribution'], 0.0, delta=1e-12)

    def test_rejects_unknown_document(self):
        with self.assertRaises(KeyError):
            self.ix.explain('electric', 'nope')

    def test_raises_before_fit(self):
        with self.assertRaises(RuntimeError):
            TFIDFIndex().explain('electric', 'p1')


class TestSublinearTF(unittest.TestCase):

    #: Both documents contain the same six distinct terms, so every IDF is
    #: equal and term frequency is the only thing separating them. 'many'
    #: repeats 'charge' six times.
    DOCS = {
        'once': 'charge electric field conductor plate equilibrium',
        'many': ('charge charge charge charge charge charge '
                 'electric field conductor plate equilibrium'),
    }

    def test_sublinear_tf_damps_repetition(self):
        # With raw tf, 'charge' gets weight 6 in the repetitive document;
        # with sublinear tf it gets 1 + ln(6) = 2.79. After L2 normalisation
        # the repetitive document's advantage on a 'charge' query shrinks from
        # ~2.30x to ~1.91x. This is TF-IDF's weaker analogue of BM25's k1
        # saturation -- weaker because the damping is fixed, not tunable.
        linear = TFIDFIndex(sublinear_tf=False).fit(self.DOCS)
        sublinear = TFIDFIndex(sublinear_tf=True).fit(self.DOCS)

        def ratio(ix):
            scores = dict(ix.search('charge', top_k=2))
            return scores['many'] / scores['once']

        self.assertGreater(ratio(linear), 2.0)
        self.assertLess(ratio(sublinear), ratio(linear))

    def test_l2_normalisation_erases_tf_in_single_term_documents(self):
        """
        A property worth knowing before interpreting TF-IDF results.

        A document consisting of one repeated term becomes a one-dimensional
        vector, and L2-normalising it yields exactly 1.0 whatever the term
        frequency was. So 'charge' x1 and 'charge' x8 score identically, and
        sublinear_tf makes no difference at all. BM25 has no such blind spot,
        because it normalises by document length rather than by vector norm.
        """
        docs = {'one': 'charge', 'eight': 'charge charge charge charge '
                                          'charge charge charge charge'}
        for sublinear in (False, True):
            scores = dict(TFIDFIndex(sublinear_tf=sublinear)
                          .fit(docs).search('charge', top_k=2))
            self.assertAlmostEqual(scores['one'], scores['eight'], delta=1e-9)

        # BM25 does distinguish them.
        bm25 = dict(BM25Index().fit(docs).search('charge', top_k=2))
        self.assertGreater(bm25['eight'], bm25['one'])


class TestComparabilityWithBM25(unittest.TestCase):
    """
    Both indexes must consume the identical token stream, or the measured gap
    between them is contaminated by preprocessing.
    """

    def test_both_indexes_see_the_same_vocabulary(self):
        tfidf = TFIDFIndex(remove_stopwords=True, stem=True).fit(DOCS)
        bm25 = BM25Index(remove_stopwords=True, stem=True).fit(DOCS)
        self.assertEqual(set(tfidf.vectorizer.get_feature_names_out()),
                         set(bm25.postings.keys()))

    def test_both_indexes_agree_on_document_count(self):
        self.assertEqual(len(TFIDFIndex().fit(DOCS)), len(BM25Index().fit(DOCS)))

    def test_both_retrieve_the_obviously_relevant_document(self):
        tfidf = TFIDFIndex().fit(DOCS)
        bm25 = BM25Index().fit(DOCS)
        self.assertEqual(tfidf.search('resistor power heat', top_k=1)[0][0],
                         bm25.search('resistor power heat', top_k=1)[0][0])

    def test_scores_are_on_different_scales(self):
        # The reason RRF exists: TF-IDF cosine is bounded by 1, BM25 is not.
        # Summing them directly would let BM25 dominate arbitrarily.
        query = 'electric conductor field'
        tfidf_top = TFIDFIndex().fit(DOCS).search(query, top_k=1)[0][1]
        bm25_top = BM25Index().fit(DOCS).search(query, top_k=1)[0][1]
        self.assertLessEqual(tfidf_top, 1.0 + 1e-9)
        self.assertGreater(bm25_top, 1.0)


class TestRetrieverInterface(unittest.TestCase):

    def test_both_indexes_are_retrievers(self):
        self.assertIsInstance(TFIDFIndex(), Retriever)
        self.assertIsInstance(BM25Index(), Retriever)

    def test_names_are_distinct_and_stable(self):
        # These strings end up as keys in saved run files, so they must not
        # drift between versions.
        self.assertEqual(TFIDFIndex.name, 'tfidf')
        self.assertEqual(BM25Index.name, 'bm25')

    def test_search_signature_is_interchangeable(self):
        for index in (TFIDFIndex().fit(DOCS), BM25Index().fit(DOCS)):
            results = index.search('electric conductor', top_k=3)
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 3)
            for doc_id, score in results:
                self.assertIsInstance(doc_id, str)
                self.assertIsInstance(score, float)


if __name__ == '__main__':
    unittest.main(verbosity=2)
