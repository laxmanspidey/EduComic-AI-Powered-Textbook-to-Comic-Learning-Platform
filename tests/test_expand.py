"""
Tests for ir.query.expand.

RM3 is checked through its invariants rather than a single hand-computed
number, because the relevance model sums over every term of ten feedback
passages. The strongest of those invariants is the weight split: original
query terms must hold exactly `alpha` of the total mass and expansion terms
exactly `1 - alpha`. That one assertion catches almost any error in the
interpolation step.

LLMExpander is tested without Ollama: `_parse` is pure, and the rest runs
against a subclass that overrides the model call.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.indexes.bm25 import BM25Index  # noqa: E402
from ir.query.expand import (  # noqa: E402
    ExpandedQuery,
    LLMExpander,
    RM3Expander,
    identity_expansion,
)

TOL = 1e-9

# A small collection where the vocabulary gap is deliberate: the documents say
# "degradation", "erosion" and "wasteland"; a student says "useless farmland".
DOCS = {
    'd1': 'Land degradation is caused by soil erosion and overgrazing in '
          'arid regions. Degradation reduces fertility.',
    'd2': 'Soil erosion removes the topsoil layer. Erosion is accelerated by '
          'deforestation and overgrazing on sloped land.',
    'd3': 'Wasteland includes barren rocky land unsuitable for cultivation '
          'after severe degradation of the soil.',
    'd4': 'Capacitors store electric charge between two conducting plates '
          'separated by a dielectric material.',
}


class TestExpandedQuery(unittest.TestCase):

    def test_as_text_appends_added_terms(self):
        expanded = ExpandedQuery(original='soil', added_terms=['erosion'])
        self.assertEqual(expanded.as_text(), 'soil erosion')

    def test_as_text_without_additions_is_the_original(self):
        self.assertEqual(ExpandedQuery(original='soil').as_text(), 'soil')

    def test_truthiness_reflects_whether_terms_were_added(self):
        self.assertFalse(bool(ExpandedQuery(original='soil')))
        self.assertTrue(bool(ExpandedQuery(original='soil',
                                           added_terms=['erosion'])))

    def test_original_terms_are_tokenised(self):
        expanded = ExpandedQuery(original='The Soil Erosion')
        self.assertEqual(expanded.original_terms, ['soil', 'erosion'])


class TestIdentityExpansion(unittest.TestCase):

    def test_adds_nothing(self):
        self.assertEqual(identity_expansion('soil erosion').added_terms, [])

    def test_weights_are_raw_term_counts(self):
        weights = identity_expansion('soil soil erosion').weights
        self.assertAlmostEqual(weights['soil'], 2.0, delta=TOL)
        self.assertAlmostEqual(weights['erosion'], 1.0, delta=TOL)

    def test_method_is_none(self):
        self.assertEqual(identity_expansion('soil').method, 'none')


class TestRM3(unittest.TestCase):

    def setUp(self):
        self.index = BM25Index().fit(DOCS)
        self.expander = RM3Expander(self.index, DOCS, fb_docs=3, fb_terms=5,
                                    alpha=0.6)

    def test_adds_terms_from_the_feedback_passages(self):
        expanded = self.expander.expand('soil')
        self.assertTrue(expanded.added_terms)
        # Everything added must actually occur in the collection -- RM3 can
        # only ever propose terms it harvested from real passages.
        for term in expanded.added_terms:
            self.assertIn(term, self.index.postings)

    def test_never_re_adds_an_original_query_term(self):
        expanded = self.expander.expand('soil erosion')
        for term in expanded.original_terms:
            self.assertNotIn(term, expanded.added_terms)

    def test_original_terms_hold_alpha_of_the_weight(self):
        expanded = self.expander.expand('soil erosion')
        originals = set(expanded.original_terms)
        original_mass = sum(w for t, w in expanded.weights.items()
                            if t in originals)
        self.assertAlmostEqual(original_mass, 0.6, delta=TOL)

    def test_expansion_terms_hold_one_minus_alpha(self):
        expanded = self.expander.expand('soil erosion')
        added_mass = sum(expanded.weights[t] for t in expanded.added_terms)
        self.assertAlmostEqual(added_mass, 0.4, delta=TOL)

    def test_total_weight_is_one(self):
        expanded = self.expander.expand('soil erosion')
        self.assertAlmostEqual(sum(expanded.weights.values()), 1.0, delta=TOL)

    def test_alpha_one_disables_expansion(self):
        expander = RM3Expander(self.index, DOCS, alpha=1.0)
        self.assertEqual(expander.expand('soil').added_terms, [])

    def test_zero_feedback_terms_disables_expansion(self):
        expander = RM3Expander(self.index, DOCS, fb_terms=0)
        self.assertEqual(expander.expand('soil').added_terms, [])

    def test_respects_the_feedback_term_cap(self):
        expander = RM3Expander(self.index, DOCS, fb_terms=3)
        self.assertLessEqual(len(expander.expand('soil').added_terms), 3)

    def test_query_matching_nothing_cannot_be_expanded(self):
        # The documented limitation: RM3 learns from the first retrieval, so
        # when that retrieval is empty there is nothing to learn from. This is
        # exactly the 'farmland becoming useless' case.
        expanded = self.expander.expand('photosynthesis chlorophyll')
        self.assertEqual(expanded.added_terms, [])

    def test_empty_query(self):
        expanded = self.expander.expand('')
        self.assertEqual(expanded.added_terms, [])
        self.assertEqual(expanded.weights, {})

    def test_stopword_only_query(self):
        self.assertEqual(self.expander.expand('the and of').added_terms, [])

    def test_expansion_improves_reach_on_a_vocabulary_gap(self):
        # 'overgrazing' retrieves d1 and d2. Expansion harvests 'degradation'
        # and 'erosion' from them, which brings in d3 -- a passage the
        # original query could never reach, since it contains no query term.
        original = {d for d, _ in self.index.search('overgrazing', top_k=10)}
        expanded = self.expander.expand('overgrazing')
        widened = {d for d, _ in
                   self.index.search_weighted(expanded.weights, top_k=10)}
        self.assertNotIn('d3', original)
        self.assertIn('d3', widened)

    def test_use_idf_changes_which_terms_are_selected(self):
        with_idf = RM3Expander(self.index, DOCS, use_idf=True).expand('soil')
        without = RM3Expander(self.index, DOCS, use_idf=False).expand('soil')
        self.assertNotEqual(with_idf.added_terms, without.added_terms)

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            RM3Expander(self.index, DOCS, alpha=1.5)
        with self.assertRaises(ValueError):
            RM3Expander(self.index, DOCS, fb_docs=0)
        with self.assertRaises(ValueError):
            RM3Expander(self.index, DOCS, fb_terms=-1)


class TestLLMParsing(unittest.TestCase):
    """`_parse` must survive everything a small local model emits."""

    def test_bare_json_array(self):
        self.assertEqual(LLMExpander._parse('["erosion", "degradation"]'),
                         ['erosion', 'degradation'])

    def test_fenced_json_array(self):
        content = '```json\n["erosion", "wasteland"]\n```'
        self.assertEqual(LLMExpander._parse(content), ['erosion', 'wasteland'])

    def test_unlabelled_fence(self):
        self.assertEqual(LLMExpander._parse('```\n["erosion"]\n```'),
                         ['erosion'])

    def test_strips_leaked_reasoning_block(self):
        content = ('<think>The student wants soil terms, so I should list '
                   'them.</think>\n["erosion", "degradation"]')
        self.assertEqual(LLMExpander._parse(content),
                         ['erosion', 'degradation'])

    def test_array_embedded_in_prose(self):
        content = 'Here are the terms: ["erosion", "barren"] Hope that helps!'
        self.assertEqual(LLMExpander._parse(content), ['erosion', 'barren'])

    def test_drops_empty_entries(self):
        self.assertEqual(LLMExpander._parse('["erosion", "", "  "]'),
                         ['erosion'])

    def test_unparseable_output_returns_empty(self):
        self.assertEqual(LLMExpander._parse('I am not going to answer that.'),
                         [])

    def test_json_object_is_not_accepted_as_a_term_list(self):
        self.assertEqual(LLMExpander._parse('{"terms": ["erosion"]}'), [])


class StubLLMExpander(LLMExpander):
    """Returns a fixed term list instead of calling Ollama."""

    def __init__(self, terms: list[str], fail: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.terms = terms
        self.fail = fail
        self.calls = 0

    def _call(self, query: str) -> list[str]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("ollama unreachable")
        return list(self.terms)


class TestLLMExpander(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_adds_the_returned_terms(self):
        expander = StubLLMExpander(['land degradation', 'desertification'],
                                   cache_dir=self.cache)
        expanded = expander.expand('why is farmland becoming useless')
        self.assertIn('degradation', expanded.added_terms)
        self.assertIn('desertification', expanded.added_terms)

    def test_splits_multi_word_phrases_into_terms(self):
        expander = StubLLMExpander(['soil erosion'], cache_dir=self.cache)
        expanded = expander.expand('bad land')
        self.assertIn('soil', expanded.added_terms)
        self.assertIn('erosion', expanded.added_terms)

    def test_drops_terms_already_in_the_query(self):
        expander = StubLLMExpander(['soil', 'erosion'], cache_dir=self.cache)
        expanded = expander.expand('soil problems')
        self.assertNotIn('soil', expanded.added_terms)

    def test_deduplicates_added_terms(self):
        expander = StubLLMExpander(['erosion', 'erosion', 'soil erosion'],
                                   cache_dir=self.cache)
        added = expander.expand('land').added_terms
        self.assertEqual(len(added), len(set(added)))

    def test_respects_the_term_cap(self):
        expander = StubLLMExpander([f'term{i}' for i in range(30)],
                                   max_terms=4, cache_dir=self.cache)
        self.assertLessEqual(len(expander.expand('land').added_terms), 4)

    def test_weight_split_matches_the_configured_weight(self):
        expander = StubLLMExpander(['erosion', 'degradation'], weight=0.4,
                                   cache_dir=self.cache)
        expanded = expander.expand('useless farmland')
        added_mass = sum(expanded.weights[t] for t in expanded.added_terms)
        self.assertAlmostEqual(added_mass, 0.4, delta=TOL)
        self.assertAlmostEqual(sum(expanded.weights.values()), 1.0, delta=TOL)

    def test_can_expand_a_query_rm3_cannot(self):
        # The complement to RM3's limitation: the LLM's knowledge comes from
        # outside the collection, so a query matching nothing is still
        # expandable.
        expander = StubLLMExpander(['land degradation', 'wasteland'],
                                   cache_dir=self.cache)
        self.assertTrue(expander.expand('useless farmland').added_terms)

    def test_falls_back_to_the_original_query_when_ollama_fails(self):
        expander = StubLLMExpander([], fail=True, cache_dir=self.cache)
        expanded = expander.expand('soil erosion')
        self.assertEqual(expanded.added_terms, [])
        self.assertAlmostEqual(expanded.weights['soil'], 1.0, delta=TOL)

    def test_empty_model_response_leaves_query_unchanged(self):
        expander = StubLLMExpander([], cache_dir=self.cache)
        self.assertEqual(expander.expand('soil').added_terms, [])

    def test_second_call_is_served_from_cache(self):
        expander = StubLLMExpander(['erosion'], cache_dir=self.cache)
        expander.expand('soil problems')
        expander.expand('soil problems')
        self.assertEqual(expander.calls, 1)

    def test_cached_result_is_identical(self):
        expander = StubLLMExpander(['erosion'], cache_dir=self.cache)
        first = expander.expand('soil problems')
        second = expander.expand('soil problems')
        self.assertEqual(first.added_terms, second.added_terms)
        self.assertEqual(first.weights, second.weights)

    def test_different_query_is_not_served_from_cache(self):
        expander = StubLLMExpander(['erosion'], cache_dir=self.cache)
        expander.expand('soil problems')
        expander.expand('water problems')
        self.assertEqual(expander.calls, 2)

    def test_cache_can_be_disabled(self):
        expander = StubLLMExpander(['erosion'], cache_dir=self.cache,
                                   use_cache=False)
        expander.expand('soil problems')
        expander.expand('soil problems')
        self.assertEqual(expander.calls, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
