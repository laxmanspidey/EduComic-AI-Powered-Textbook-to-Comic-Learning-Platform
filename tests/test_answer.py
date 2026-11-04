"""
Tests for ir.answer.

The important ones are the attribution checks. A generator that always
produces fluent text will look fine while hiding retrieval failures, so the
tests assert that abstention survives, that invented citation markers are
dropped, and that a model failure degrades to an ungrounded answer rather than
an exception.

No Ollama needed: a stub supplies the model reply.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.answer import Answer, AnswerGenerator, Citation  # noqa: E402


@dataclass
class FakeResult:
    """Stands in for pipeline.ScoredPassage."""
    passage_id: str
    text: str
    citation: str = 'Physics | p19'
    breadcrumb: str = 'Physics > Electrostatics'
    rank: int = 1
    matched_terms: list = field(default_factory=list)


RESULTS = [
    FakeResult('phys::p25', 'Electrostatic shielding: the field inside a '
                            'cavity of a conductor is zero.'),
    FakeResult('phys::p26', 'All charges reside on the outer surface of a '
                            'conductor with a cavity.'),
    FakeResult('phys::p30', 'A capacitor stores charge between two plates.'),
]


class StubGenerator(AnswerGenerator):
    """Returns a fixed model reply instead of calling Ollama."""

    def __init__(self, reply: str, fail: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.reply = reply
        self.fail = fail
        self.calls = 0

    def _call(self, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        if self.fail:
            raise RuntimeError('ollama unreachable')
        return self.reply


def make(reply: str, **kwargs) -> StubGenerator:
    tmp = kwargs.pop('cache_dir', None)
    return StubGenerator(reply, cache_dir=tmp, **kwargs)


GOOD = ('{"answerable": true, "answer": "The field inside a conductor cavity '
        'is zero [1], so charges sit on the outer surface [2].", '
        '"citations": [1, 2]}')


class TestParsing(unittest.TestCase):

    def test_bare_json_object(self):
        parsed = AnswerGenerator._parse(GOOD)
        self.assertTrue(parsed['answerable'])
        self.assertIn('[1]', parsed['answer'])

    def test_fenced_json(self):
        parsed = AnswerGenerator._parse(f'```json\n{GOOD}\n```')
        self.assertIn('answer', parsed)

    def test_strips_thinking_block(self):
        parsed = AnswerGenerator._parse(f'<think>reasoning</think>{GOOD}')
        self.assertIn('answer', parsed)

    def test_json_embedded_in_prose(self):
        parsed = AnswerGenerator._parse(f'Here you go: {GOOD} Hope it helps.')
        self.assertIn('answer', parsed)

    def test_plain_prose_falls_back_to_prose(self):
        # A usable answer in the wrong envelope is still usable; it just
        # loses its explicit abstention flag.
        parsed = AnswerGenerator._parse('The field inside a cavity is zero.')
        self.assertEqual(parsed['answer'], 'The field inside a cavity is zero.')
        self.assertTrue(parsed['answerable'])

    def test_empty_reply(self):
        self.assertEqual(AnswerGenerator._parse(''), {})

    def test_marker_extraction(self):
        self.assertEqual(AnswerGenerator._markers_in('a [1] b [3] c [1]'),
                         [1, 3, 1])

    def test_marker_extraction_ignores_long_numbers(self):
        self.assertEqual(AnswerGenerator._markers_in('see [1234]'), [])


class TestGeneration(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_produces_a_grounded_answer(self):
        answer = make(GOOD, cache_dir=self.cache).generate('what is shielding',
                                                           RESULTS)
        self.assertTrue(answer.grounded)
        self.assertIn('cavity', answer.text)
        self.assertEqual(answer.latency_ms > 0, True)

    def test_resolves_citations_to_passage_ids(self):
        answer = make(GOOD, cache_dir=self.cache).generate('q', RESULTS)
        self.assertEqual(answer.cited_passage_ids, ['phys::p25', 'phys::p26'])

    def test_citation_carries_provenance(self):
        answer = make(GOOD, cache_dir=self.cache).generate('q', RESULTS)
        first = answer.citations[0]
        self.assertEqual(first.marker, 1)
        self.assertIn('Electrostatics', first.breadcrumb)
        self.assertTrue(first.snippet)

    def test_drops_citations_for_passages_never_supplied(self):
        # The model cites [9], which was never in the prompt. A fabricated
        # citation looks verified, so it must not be shown.
        reply = ('{"answerable": true, "answer": "Something [1] and more [9].",'
                 ' "citations": [1, 9]}')
        answer = make(reply, cache_dir=self.cache).generate('q', RESULTS)
        self.assertEqual(answer.cited_passage_ids, ['phys::p25'])

    def test_markers_in_text_count_even_if_citations_field_is_missing(self):
        reply = '{"answerable": true, "answer": "Because of this [2]."}'
        answer = make(reply, cache_dir=self.cache).generate('q', RESULTS)
        self.assertEqual(answer.cited_passage_ids, ['phys::p26'])

    def test_abstention_is_preserved(self):
        # The key safety property: a generator that always answers would hide
        # retrieval failures behind fluent text.
        reply = ('{"answerable": false, "answer": "This textbook does not '
                 'cover photosynthesis.", "citations": []}')
        answer = make(reply, cache_dir=self.cache).generate('q', RESULTS)
        self.assertFalse(answer.grounded)
        self.assertIn('does not cover', answer.text)

    def test_grounded_answer_without_citations_is_flagged(self):
        reply = '{"answerable": true, "answer": "Some claim.", "citations": []}'
        answer = make(reply, cache_dir=self.cache).generate('q', RESULTS)
        self.assertTrue(answer.uncited)

    def test_cited_answer_is_not_flagged(self):
        answer = make(GOOD, cache_dir=self.cache).generate('q', RESULTS)
        self.assertFalse(answer.uncited)

    def test_respects_the_passage_limit(self):
        generator = make(GOOD, n_passages=2, cache_dir=self.cache)
        generator.generate('q', RESULTS)
        self.assertIn('[2]', generator.last_prompt)
        self.assertNotIn('[3]', generator.last_prompt)

    def test_context_word_budget_truncates(self):
        long_results = [FakeResult('d::p0', ' '.join(['word'] * 5000))]
        generator = make(GOOD, context_words=100, cache_dir=self.cache)
        generator.generate('q', long_results)
        # The prompt holds the question and instructions too, so just check
        # the passage body was cut well below its original length.
        self.assertLess(generator.last_prompt.count('word'), 400)

    def test_records_the_passages_it_used(self):
        answer = make(GOOD, n_passages=2, cache_dir=self.cache).generate(
            'q', RESULTS)
        self.assertEqual(answer.context_passages,
                         ['phys::p25', 'phys::p26'])


class TestFailureModes(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_results_returns_ungrounded_answer(self):
        answer = make(GOOD, cache_dir=self.cache).generate('q', [])
        self.assertFalse(answer.grounded)
        self.assertIn('Nothing', answer.text)

    def test_no_results_does_not_call_the_model(self):
        generator = make(GOOD, cache_dir=self.cache)
        generator.generate('q', [])
        self.assertEqual(generator.calls, 0)

    def test_model_failure_degrades_instead_of_raising(self):
        answer = make('', fail=True, cache_dir=self.cache).generate('q', RESULTS)
        self.assertFalse(answer.grounded)
        self.assertIn('unavailable', answer.text)
        self.assertIn('ollama', answer.error)

    def test_empty_model_reply_is_ungrounded(self):
        answer = make('', cache_dir=self.cache).generate('q', RESULTS)
        self.assertFalse(answer.grounded)

    def test_whitespace_only_answer_is_ungrounded(self):
        reply = '{"answerable": true, "answer": "   ", "citations": []}'
        answer = make(reply, cache_dir=self.cache).generate('q', RESULTS)
        self.assertFalse(answer.grounded)


class TestCaching(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_second_identical_call_is_cached(self):
        generator = make(GOOD, cache_dir=self.cache)
        generator.generate('what is shielding', RESULTS)
        generator.generate('what is shielding', RESULTS)
        self.assertEqual(generator.calls, 1)

    def test_different_passages_invalidate_the_cache(self):
        # A change in retrieval must produce a fresh answer, not a stale one
        # generated from passages that are no longer being shown.
        generator = make(GOOD, cache_dir=self.cache)
        generator.generate('q', RESULTS)
        generator.generate('q', RESULTS[1:])
        self.assertEqual(generator.calls, 2)

    def test_different_query_invalidates_the_cache(self):
        generator = make(GOOD, cache_dir=self.cache)
        generator.generate('q1', RESULTS)
        generator.generate('q2', RESULTS)
        self.assertEqual(generator.calls, 2)

    def test_cache_can_be_disabled(self):
        generator = make(GOOD, cache_dir=self.cache, use_cache=False)
        generator.generate('q', RESULTS)
        generator.generate('q', RESULTS)
        self.assertEqual(generator.calls, 2)

    def test_cached_answer_matches_the_original(self):
        generator = make(GOOD, cache_dir=self.cache)
        first = generator.generate('q', RESULTS)
        second = generator.generate('q', RESULTS)
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.cited_passage_ids, second.cited_passage_ids)


class TestSerialisation(unittest.TestCase):

    def test_to_dict_is_json_safe(self):
        import json
        answer = Answer(text='x', query='q',
                        citations=[Citation(1, 'p::0', 'cite', 'crumb', 'snip')])
        json.dumps(answer.to_dict())


if __name__ == '__main__':
    unittest.main(verbosity=2)
