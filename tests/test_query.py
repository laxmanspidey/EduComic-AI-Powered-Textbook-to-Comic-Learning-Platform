"""
Tests for ir.query.intent and ir.query.rewrite.

The rewriter's rule gate gets the most attention here, because it decides
whether the LLM runs at all. A gate that is too eager wastes a model call per
query and risks corrupting self-contained questions; one that is too shy
leaves elliptical follow-ups unresolved. Both directions are tested.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.query.intent import (  # noqa: E402
    Intent,
    analyse,
    classify,
    comparison_entities,
    policy_for,
)
from ir.query.rewrite import (  # noqa: E402
    ConversationTurn,
    ConversationalRewriter,
    NoOpRewriter,
    looks_dependent,
)


class TestClassify(unittest.TestCase):

    def test_definition(self):
        for query in ('what is electrostatic shielding',
                      'define entropy',
                      'what does CAP stand for',
                      'meaning of inertia'):
            self.assertEqual(classify(query), Intent.DEFINITION, query)

    def test_comparison(self):
        for query in ('difference between TCP and UDP',
                      'compare mitosis and meiosis',
                      'TCP vs UDP',
                      'how do they differ'):
            self.assertEqual(classify(query), Intent.COMPARISON, query)

    def test_causal(self):
        for query in ('why does entropy increase',
                      'what causes soil erosion',
                      'what happens when the bus stops suddenly'):
            self.assertEqual(classify(query), Intent.CAUSAL, query)

    def test_process(self):
        for query in ('how does RSA key generation work',
                      'steps of photosynthesis',
                      'what is the procedure for titration'):
            self.assertEqual(classify(query), Intent.PROCESS, query)

    def test_numerical(self):
        for query in ('formula for capacitance in parallel',
                      'calculate the electric potential',
                      'how much energy is stored'):
            self.assertEqual(classify(query), Intent.NUMERICAL, query)

    def test_example(self):
        for query in ('examples of land degradation',
                      'types of soil',
                      'give examples of supervised learning'):
            self.assertEqual(classify(query), Intent.EXAMPLE, query)

    def test_unmatched_falls_back_to_factual(self):
        # Falling back is deliberate: a wrong policy is worse than a neutral
        # one, because it misroutes the query away from the arm that would
        # have answered it.
        self.assertEqual(classify('electrostatic shielding conductor'),
                         Intent.FACTUAL)

    def test_empty_query(self):
        self.assertEqual(classify(''), Intent.FACTUAL)
        self.assertEqual(classify('   '), Intent.FACTUAL)

    def test_comparison_wins_over_definition(self):
        # "what is the difference between X and Y" matches both patterns;
        # the comparison reading is the more specific one.
        self.assertEqual(classify('what is the difference between TCP and UDP'),
                         Intent.COMPARISON)


class TestPolicies(unittest.TestCase):

    def test_definition_leans_lexical(self):
        weights = policy_for(Intent.DEFINITION).retriever_weights
        self.assertGreater(weights['bm25'], weights['dense'])

    def test_causal_leans_semantic(self):
        weights = policy_for(Intent.CAUSAL).retriever_weights
        self.assertGreater(weights['dense'], weights['bm25'])

    def test_numerical_leans_most_lexical(self):
        numerical = policy_for(Intent.NUMERICAL).retriever_weights
        definition = policy_for(Intent.DEFINITION).retriever_weights
        self.assertGreater(numerical['bm25'], definition['bm25'])

    def test_definition_skips_expansion(self):
        # The term itself is the anchor; expansion only dilutes it.
        self.assertFalse(policy_for(Intent.DEFINITION).expand)

    def test_causal_uses_expansion(self):
        self.assertTrue(policy_for(Intent.CAUSAL).expand)

    def test_process_widens_the_context_window(self):
        self.assertGreater(policy_for(Intent.PROCESS).context_window,
                           policy_for(Intent.DEFINITION).context_window)

    def test_comparison_splits_entities(self):
        self.assertTrue(policy_for(Intent.COMPARISON).split_entities)

    def test_every_intent_has_a_policy(self):
        for intent in Intent:
            self.assertIsNotNone(policy_for(intent).retriever_weights)


class TestComparisonEntities(unittest.TestCase):

    def test_between_x_and_y(self):
        self.assertEqual(
            comparison_entities('difference between consistency and availability'),
            ['consistency', 'availability'])

    def test_strips_articles(self):
        self.assertEqual(
            comparison_entities('the difference between the cat and the dog'),
            ['cat', 'dog'])

    def test_bare_versus(self):
        self.assertEqual(comparison_entities('TCP vs UDP'), ['TCP', 'UDP'])

    def test_question_mark_is_trimmed(self):
        self.assertEqual(
            comparison_entities('what is the difference between mitosis and meiosis?'),
            ['mitosis', 'meiosis'])

    def test_unextractable_comparison_returns_empty(self):
        # "compare these two protocols" names nothing. An empty list means
        # "do not split", which is a fine fallback, not an error.
        self.assertEqual(comparison_entities('compare these two protocols'), [])


class TestAnalyse(unittest.TestCase):

    def test_attaches_intent_and_policy(self):
        analysed = analyse('why does entropy increase')
        self.assertEqual(analysed.intent, Intent.CAUSAL)
        self.assertTrue(analysed.policy.expand)

    def test_splittable_comparison_yields_subqueries(self):
        analysed = analyse('difference between consistency and availability')
        self.assertTrue(analysed.should_split)
        self.assertEqual(analysed.subqueries(),
                         ['difference between consistency and availability',
                          'consistency', 'availability'])

    def test_original_query_is_kept_as_a_subquery(self):
        # A passage that compares both directly is the ideal answer, so the
        # combined query must still run alongside the per-entity ones.
        analysed = analyse('TCP vs UDP')
        self.assertEqual(analysed.subqueries()[0], 'TCP vs UDP')

    def test_unsplittable_comparison_does_not_split(self):
        analysed = analyse('compare these two protocols')
        self.assertFalse(analysed.should_split)
        self.assertEqual(len(analysed.subqueries()), 1)

    def test_non_comparison_has_no_entities(self):
        self.assertEqual(analyse('what is entropy').entities, [])


class TestDependencyGate(unittest.TestCase):

    def test_pronoun_marks_dependence(self):
        dependent, reason = looks_dependent('why does it increase?')
        self.assertTrue(dependent)
        self.assertIn('it', reason)

    def test_demonstrative_marks_dependence(self):
        self.assertTrue(looks_dependent('how does that work')[0])

    def test_short_interrogative_marks_dependence(self):
        self.assertTrue(looks_dependent('why?')[0])
        self.assertTrue(looks_dependent('and then?')[0])

    def test_very_short_query_marks_dependence(self):
        self.assertTrue(looks_dependent('more detail')[0])

    def test_self_contained_query_does_not(self):
        for query in ('what is electrostatic shielding',
                      'difference between consistency and availability',
                      'formula for capacitance in a parallel combination'):
            self.assertFalse(looks_dependent(query)[0], query)

    def test_empty_query_is_not_dependent(self):
        self.assertFalse(looks_dependent('')[0])

    def test_reason_is_always_given(self):
        for query in ('why does it increase?', 'what is entropy', ''):
            self.assertTrue(looks_dependent(query)[1])


class TestRewriteParsing(unittest.TestCase):

    def test_plain_question(self):
        self.assertEqual(ConversationalRewriter._clean('why does entropy increase?'),
                         'why does entropy increase?')

    def test_strips_quotes(self):
        self.assertEqual(ConversationalRewriter._clean('"why does entropy increase?"'),
                         'why does entropy increase?')

    def test_strips_preamble(self):
        self.assertEqual(
            ConversationalRewriter._clean('Rewritten question: why does entropy increase?'),
            'why does entropy increase?')

    def test_strips_thinking_block(self):
        content = ('<think>The pronoun refers to entropy.</think>\n'
                   'why does entropy increase?')
        self.assertEqual(ConversationalRewriter._clean(content),
                         'why does entropy increase?')

    def test_takes_the_first_non_empty_line(self):
        self.assertEqual(
            ConversationalRewriter._clean('\n\nwhy does entropy increase?\nI hope that helps.'),
            'why does entropy increase?')

    def test_empty_content(self):
        self.assertEqual(ConversationalRewriter._clean(''), '')


class StubRewriter(ConversationalRewriter):
    """Returns a fixed rewrite instead of calling Ollama."""

    def __init__(self, rewritten: str, fail: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.rewritten = rewritten
        self.fail = fail
        self.calls = 0

    def _call(self, query: str, history_text: str) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError('ollama unreachable')
        return self.rewritten


HISTORY = [ConversationTurn('what is entropy',
                            'Entropy is a measure of disorder.')]


class TestConversationalRewriter(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolves_a_pronoun(self):
        rewriter = StubRewriter('why does entropy increase?', cache_dir=self.cache)
        result = rewriter.rewrite('why does it increase?', HISTORY)
        self.assertEqual(result.rewritten, 'why does entropy increase?')
        self.assertTrue(result.changed)

    def test_no_history_means_no_rewrite(self):
        rewriter = StubRewriter('anything', cache_dir=self.cache)
        result = rewriter.rewrite('why does it increase?', [])
        self.assertEqual(result.rewritten, 'why does it increase?')
        self.assertEqual(rewriter.calls, 0)

    def test_self_contained_query_skips_the_model(self):
        # The gate's main purpose: no call, and no risk of the model
        # narrowing a general question to the previous turn's topic.
        rewriter = StubRewriter('corrupted', cache_dir=self.cache)
        result = rewriter.rewrite('what is electrostatic shielding', HISTORY)
        self.assertEqual(result.rewritten, 'what is electrostatic shielding')
        self.assertEqual(rewriter.calls, 0)

    def test_unchanged_rewrite_is_not_marked_changed(self):
        rewriter = StubRewriter('why does it increase?', cache_dir=self.cache)
        self.assertFalse(rewriter.rewrite('why does it increase?', HISTORY).changed)

    def test_falls_back_when_the_model_fails(self):
        rewriter = StubRewriter('', fail=True, cache_dir=self.cache)
        result = rewriter.rewrite('why does it increase?', HISTORY)
        self.assertEqual(result.rewritten, 'why does it increase?')
        self.assertIn('unavailable', result.reason)

    def test_empty_model_response_falls_back(self):
        rewriter = StubRewriter('', cache_dir=self.cache)
        result = rewriter.rewrite('why does it increase?', HISTORY)
        self.assertEqual(result.rewritten, 'why does it increase?')

    def test_overlong_rewrite_is_rejected(self):
        # A rewrite should resolve referents, not write an essay. An
        # unbounded expansion would change what was asked.
        rewriter = StubRewriter(' '.join(['word'] * 200), cache_dir=self.cache)
        result = rewriter.rewrite('why does it increase?', HISTORY)
        self.assertEqual(result.rewritten, 'why does it increase?')
        self.assertIn('too long', result.reason)

    def test_records_the_history_it_used(self):
        rewriter = StubRewriter('why does entropy increase?', cache_dir=self.cache)
        result = rewriter.rewrite('why does it increase?', HISTORY)
        self.assertEqual(result.history_used, ['what is entropy'])

    def test_second_call_is_cached(self):
        rewriter = StubRewriter('why does entropy increase?', cache_dir=self.cache)
        rewriter.rewrite('why does it increase?', HISTORY)
        rewriter.rewrite('why does it increase?', HISTORY)
        self.assertEqual(rewriter.calls, 1)

    def test_different_history_is_not_cached_together(self):
        rewriter = StubRewriter('x', cache_dir=self.cache)
        rewriter.rewrite('why does it increase?', HISTORY)
        rewriter.rewrite('why does it increase?',
                         [ConversationTurn('what is enthalpy')])
        self.assertEqual(rewriter.calls, 2)

    def test_history_window_is_respected(self):
        rewriter = StubRewriter('x', max_history=1, cache_dir=self.cache)
        history = [ConversationTurn('first'), ConversationTurn('second')]
        rendered = rewriter._format_history(history)
        self.assertNotIn('first', rendered)
        self.assertIn('second', rendered)


class TestNoOpRewriter(unittest.TestCase):

    def test_passes_through(self):
        result = NoOpRewriter().rewrite('why does it increase?', HISTORY)
        self.assertEqual(result.rewritten, 'why does it increase?')
        self.assertFalse(result.changed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
