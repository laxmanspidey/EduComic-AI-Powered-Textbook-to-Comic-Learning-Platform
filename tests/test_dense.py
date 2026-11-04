"""
Tests for ir.indexes.dense.

Almost everything here runs against a deterministic 3-dimensional fake
embedder, so the suite needs no model download, no GPU and no network, and the
expected cosine values can be worked out by hand. Only
`TestRealModelSemantics` loads all-MiniLM-L6-v2, and it exists to verify the
one property that justifies having a dense retriever at all: matching meaning
across completely disjoint vocabulary.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.indexes.base import Retriever  # noqa: E402
from ir.indexes.dense import (  # noqa: E402
    DenseIndex,
    Embedder,
    _l2_normalise,
)

ROOT2 = 2 ** 0.5


class FakeEmbedder(Embedder):
    """
    Maps text onto three orthogonal axes by keyword presence.

        'physics'            -> [1, 0, 0]
        'physics biology'    -> [1, 1, 0] / sqrt(2)
        'nothing relevant'   -> [0, 0, 0]

    Cosines are therefore exact and hand-checkable, and `calls` records how
    many times documents were actually embedded so the cache can be tested.
    """

    model_id = 'fake:3d'
    AXES = ('physics', 'biology', 'chemistry')

    def __init__(self, model_id: str = 'fake:3d'):
        self.model_id = model_id
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if axis in lowered else 0.0 for axis in self.AXES]

    def encode_documents(self, texts: list[str]):
        self.calls += 1
        import numpy as np
        if not texts:
            return np.zeros((0, 3), dtype='float32')
        return _l2_normalise([self._vector(t) for t in texts])

    def encode_query(self, text: str):
        return _l2_normalise(self._vector(text))


DOCS = {
    'd_phys': 'physics',
    'd_both': 'physics biology',
    'd_bio': 'biology',
    'd_none': 'completely unrelated text',
}


def fresh_index(tmp: str, embedder: Embedder | None = None,
                use_cache: bool = False) -> DenseIndex:
    return DenseIndex(embedder=embedder or FakeEmbedder(),
                      cache_dir=Path(tmp), use_cache=use_cache)


class TestL2Normalise(unittest.TestCase):

    def test_scales_rows_to_unit_length(self):
        import numpy as np
        result = _l2_normalise([[3.0, 4.0]])
        self.assertAlmostEqual(float(np.linalg.norm(result[0])), 1.0, delta=1e-6)
        self.assertAlmostEqual(float(result[0][0]), 0.6, delta=1e-6)

    def test_handles_one_dimensional_input(self):
        import numpy as np
        result = _l2_normalise([0.0, 5.0])
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, delta=1e-6)

    def test_leaves_zero_vectors_alone_instead_of_producing_nan(self):
        import numpy as np
        result = _l2_normalise([[0.0, 0.0], [3.0, 4.0]])
        self.assertFalse(bool(np.isnan(result).any()))
        self.assertEqual(list(result[0]), [0.0, 0.0])

    def test_zero_one_dimensional_vector(self):
        import numpy as np
        self.assertFalse(bool(np.isnan(_l2_normalise([0.0, 0.0])).any()))

    def test_returns_float32(self):
        self.assertEqual(_l2_normalise([[1.0, 1.0]]).dtype.name, 'float32')


class TestSearch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ix = fresh_index(self.tmp.name).fit(DOCS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_indexes_every_document(self):
        self.assertEqual(len(self.ix), 4)

    def test_reports_dimensionality(self):
        self.assertEqual(self.ix.dimensions, 3)

    def test_exact_axis_match_scores_one(self):
        scores = dict(self.ix.search('physics', top_k=4))
        self.assertAlmostEqual(scores['d_phys'], 1.0, delta=1e-6)

    def test_partial_match_scores_one_over_root_two(self):
        # query [1,0,0] vs doc [1,1,0]/sqrt(2) -> cosine = 1/sqrt(2)
        scores = dict(self.ix.search('physics', top_k=4))
        self.assertAlmostEqual(scores['d_both'], 1 / ROOT2, delta=1e-6)

    def test_orthogonal_document_is_excluded(self):
        # 'biology' is orthogonal to 'physics' -> cosine 0 -> dropped.
        returned = {d for d, _ in self.ix.search('physics', top_k=4)}
        self.assertNotIn('d_bio', returned)
        self.assertNotIn('d_none', returned)

    def test_ranks_exact_above_partial(self):
        results = self.ix.search('physics', top_k=4)
        self.assertEqual([d for d, _ in results], ['d_phys', 'd_both'])

    def test_respects_top_k(self):
        self.assertLessEqual(len(self.ix.search('physics biology', top_k=1)), 1)

    def test_top_k_larger_than_corpus_is_safe(self):
        self.assertLessEqual(len(self.ix.search('physics', top_k=999)), 4)

    def test_ties_break_deterministically_by_doc_id(self):
        ix = fresh_index(self.tmp.name).fit(
            {'zz': 'physics', 'aa': 'physics'})
        results = ix.search('physics', top_k=2)
        self.assertEqual([d for d, _ in results], ['aa', 'zz'])
        self.assertEqual(results, ix.search('physics', top_k=2))

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.ix.search('   '), [])

    def test_empty_corpus_returns_nothing(self):
        ix = fresh_index(self.tmp.name).fit({})
        self.assertEqual(len(ix), 0)
        self.assertEqual(ix.dimensions, 0)
        self.assertEqual(ix.search('physics'), [])

    def test_search_before_fit_returns_nothing(self):
        self.assertEqual(fresh_index(self.tmp.name).search('physics'), [])

    def test_scores_every_document_unlike_sparse_indexes(self):
        # A dense retriever always has an opinion: a query matching both axes
        # scores all three on-axis documents, where BM25 would return only
        # those sharing a token.
        returned = {d for d, _ in self.ix.search('physics biology', top_k=4)}
        self.assertEqual(returned, {'d_phys', 'd_both', 'd_bio'})


class TestSimilarity(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ix = fresh_index(self.tmp.name).fit(DOCS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_matches_the_search_score(self):
        from_search = dict(self.ix.search('physics', top_k=4))['d_both']
        self.assertAlmostEqual(self.ix.similarity('physics', 'd_both'),
                               from_search, delta=1e-6)

    def test_orthogonal_pair_is_zero(self):
        self.assertAlmostEqual(self.ix.similarity('physics', 'd_bio'),
                               0.0, delta=1e-6)

    def test_unknown_document_is_zero(self):
        self.assertEqual(self.ix.similarity('physics', 'nope'), 0.0)


class TestEmbeddingCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_second_fit_reuses_the_cache(self):
        embedder = FakeEmbedder()
        DenseIndex(embedder, cache_dir=Path(self.tmp.name)).fit(DOCS)
        self.assertEqual(embedder.calls, 1)

        # A fresh index over identical content must not re-embed.
        second = DenseIndex(embedder, cache_dir=Path(self.tmp.name)).fit(DOCS)
        self.assertEqual(embedder.calls, 1)
        self.assertEqual(len(second), 4)

    def test_cached_index_returns_the_same_scores(self):
        embedder = FakeEmbedder()
        first = DenseIndex(embedder, cache_dir=Path(self.tmp.name)).fit(DOCS)
        expected = first.search('physics', top_k=4)
        second = DenseIndex(embedder, cache_dir=Path(self.tmp.name)).fit(DOCS)
        self.assertEqual(second.search('physics', top_k=4), expected)

    def test_changing_passage_text_invalidates_the_cache(self):
        # The cache key hashes content, not just ids. Reusing stale vectors
        # against edited passages would corrupt every downstream measurement
        # invisibly, so this is the most important cache test here.
        embedder = FakeEmbedder()
        DenseIndex(embedder, cache_dir=Path(self.tmp.name)).fit(DOCS)
        edited = dict(DOCS, d_phys='chemistry')
        DenseIndex(embedder, cache_dir=Path(self.tmp.name)).fit(edited)
        self.assertEqual(embedder.calls, 2)

    def test_adding_a_passage_invalidates_the_cache(self):
        embedder = FakeEmbedder()
        DenseIndex(embedder, cache_dir=Path(self.tmp.name)).fit(DOCS)
        DenseIndex(embedder, cache_dir=Path(self.tmp.name)).fit(
            dict(DOCS, d_new='chemistry'))
        self.assertEqual(embedder.calls, 2)

    def test_different_model_does_not_share_a_cache(self):
        first = FakeEmbedder(model_id='fake:A')
        second = FakeEmbedder(model_id='fake:B')
        DenseIndex(first, cache_dir=Path(self.tmp.name)).fit(DOCS)
        DenseIndex(second, cache_dir=Path(self.tmp.name)).fit(DOCS)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)

    def test_use_cache_false_always_recomputes(self):
        embedder = FakeEmbedder()
        for _ in range(3):
            DenseIndex(embedder, cache_dir=Path(self.tmp.name),
                       use_cache=False).fit(DOCS)
        self.assertEqual(embedder.calls, 3)

    def test_corrupt_cache_falls_back_to_recomputing(self):
        embedder = FakeEmbedder()
        index = DenseIndex(embedder, cache_dir=Path(self.tmp.name))
        index.fit(DOCS)

        for npy in Path(self.tmp.name).glob('*.npy'):
            npy.write_bytes(b'not a numpy file')

        rebuilt = DenseIndex(embedder, cache_dir=Path(self.tmp.name)).fit(DOCS)
        self.assertEqual(embedder.calls, 2)
        self.assertEqual(len(rebuilt), 4)


class TestRetrieverInterface(unittest.TestCase):

    def test_is_a_retriever(self):
        self.assertIsInstance(DenseIndex(FakeEmbedder()), Retriever)

    def test_name_is_stable(self):
        self.assertEqual(DenseIndex.name, 'dense')

    def test_search_returns_the_shared_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = fresh_index(tmp).fit(DOCS)
            for doc_id, score in index.search('physics', top_k=2):
                self.assertIsInstance(doc_id, str)
                self.assertIsInstance(score, float)


class TestRealModelSemantics(unittest.TestCase):
    """
    The property that justifies dense retrieval: matching meaning where the
    vocabulary does not overlap at all.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("sentence-transformers not installed")

    DOCS = {
        'target': 'Overfitting occurs when a model learns the training '
                  'data too closely and consequently exhibits poor '
                  'generalization on unseen samples.',
        'decoy1': 'The transport layer provides end to end communication '
                  'between hosts using port numbers.',
        'decoy2': 'Photosynthesis converts light energy into chemical '
                  'energy stored in glucose molecules.',
    }

    def test_dense_reaches_target_when_no_stem_is_shared(self):
        """
        Problem #1 (terminology mismatch), reproduced exactly.

        'why does a classifier do poorly on fresh examples' shares *no* stem
        with the target passage: classifier/model, poorly/poor,
        fresh/unseen, examples/samples all differ after stemming. BM25
        therefore returns nothing at all, while the dense retriever still
        finds the passage at cosine 0.397.
        """
        from ir.indexes.bm25 import BM25Index
        from ir.indexes.dense import SentenceTransformerEmbedder
        from ir.text import tokenize

        query = 'why does a classifier do poorly on fresh examples'

        # Establish the premise rather than assuming it.
        self.assertEqual(set(tokenize(query)) & set(tokenize(self.DOCS['target'])),
                         set())

        self.assertEqual(BM25Index().fit(self.DOCS).search(query, top_k=3), [])

        with tempfile.TemporaryDirectory() as tmp:
            dense = DenseIndex(SentenceTransformerEmbedder(device='cpu'),
                               cache_dir=Path(tmp)).fit(self.DOCS)
            results = dense.search(query, top_k=3)
        self.assertEqual(results[0][0], 'target')
        self.assertGreater(results[0][1], 0.3)

    def test_lexical_still_works_when_generic_terms_happen_to_overlap(self):
        """
        The honest counterpart, and a correction to how this example is
        usually told.

        'why does a model perform badly on new data' is described as a
        vocabulary mismatch because the student never writes "overfitting",
        "generalization" or "unseen samples". But it does share "model" and
        "data", and those two generic terms are enough for BM25 to retrieve
        the passage. So terminology mismatch is not all-or-nothing: it bites
        only when no term survives, which is why the hybrid keeps both arms
        instead of replacing lexical matching with semantics.
        """
        from ir.indexes.bm25 import BM25Index
        from ir.text import tokenize

        query = 'why does a model perform badly on new data'
        shared = set(tokenize(query)) & set(tokenize(self.DOCS['target']))
        self.assertEqual(shared, {'model', 'data'})

        bm25_top = BM25Index().fit(self.DOCS).search(query, top_k=3)
        self.assertEqual(bm25_top[0][0], 'target')


if __name__ == '__main__':
    unittest.main(verbosity=2)
