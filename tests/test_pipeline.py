"""
Tests for ir.context, ir.pipeline and ir.eval.

The pipeline tests use a fake dense retriever so no model loads; the point
being verified is that stages compose, that every stage records what it did,
and that an un-configured stage is skipped rather than silently defaulted.

`TestDeOverlapping` covers the one piece of non-obvious logic in context
assembly: adjacent passages share 80 words by design, so joining them naively
repeats text.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.context import Breadcrumb, ContextAssembler  # noqa: E402
from ir.corpus import build_from_texts  # noqa: E402
from ir.eval.collection import (  # noqa: E402
    EvalQuery,
    Judgement,
    SyntheticQueryGenerator,
    TestCollection,
    manual_collection,
)
from ir.eval.experiments import (  # noqa: E402
    SystemResult,
    ablation_table,
    comparison_table,
    run_system,
)
from ir.indexes.base import Retriever  # noqa: E402
from ir.indexes.bm25 import BM25Index  # noqa: E402
from ir.pipeline import Pipeline  # noqa: E402
from ir.query.expand import RM3Expander  # noqa: E402
from ir.rerank import IdentityReranker, Reranker, RerankedResult  # noqa: E402

TOL = 1e-9


def make_corpus():
    """Three documents of 900 words each -> several overlapping passages."""
    return build_from_texts({
        'physics': ' '.join(
            ['electrostatic shielding conductor cavity field zero charge'] * 130),
        'soil': ' '.join(
            ['land degradation soil erosion overgrazing wasteland barren'] * 130),
        'systems': ' '.join(
            ['consistency availability partition tolerance distributed cap'] * 130),
    }, chunk_size=400, overlap=80)


class FakeDense(Retriever):
    """Scores by a fixed passage-id order, no embeddings."""

    name = 'dense'

    def __init__(self, order: list[str]):
        self.order = order

    def search(self, query: str, top_k: int = 10):
        return [(pid, 1.0 - i * 0.1) for i, pid in enumerate(self.order[:top_k])]


class StubReranker(Reranker):
    """Promotes a chosen passage to the top."""

    name = 'stub'

    def __init__(self, favour: str):
        self.favour = favour

    def rerank(self, query, candidates, texts, top_n=None):
        ordered = ([self.favour] + [c for c in candidates if c != self.favour]
                   if self.favour in candidates else list(candidates))
        previous = {c: i for i, c in enumerate(candidates, start=1)}
        return [RerankedResult(doc_id=c, score=float(-i), rank=i,
                               previous_rank=previous[c], reranked=True)
                for i, c in enumerate(ordered, start=1)]


# ──────────────────────────────────────────────────────────────────────────────
# Context assembly
# ──────────────────────────────────────────────────────────────────────────────

class TestBreadcrumb(unittest.TestCase):

    def test_path_omits_empty_levels(self):
        crumb = Breadcrumb(textbook='Physics', concept='Shielding', pages='p19')
        self.assertEqual(crumb.path(), ['Physics', 'Shielding'])

    def test_string_includes_pages(self):
        crumb = Breadcrumb(textbook='Physics', pages='p19-20')
        self.assertEqual(str(crumb), 'Physics (p19-20)')

    def test_empty_breadcrumb(self):
        self.assertEqual(Breadcrumb().path(), [])


class TestDeOverlapping(unittest.TestCase):
    """Adjacent passages share the configured overlap; joining must not repeat it."""

    def test_removes_the_shared_span(self):
        corpus = build_from_texts(
            {'d': ' '.join(f'w{i}' for i in range(1000))},
            chunk_size=400, overlap=80)
        assembler = ContextAssembler(corpus, window=1)
        first = corpus.by_document('d')[1]
        context = assembler.assemble(first.passage_id).context_text
        words = context.split()
        self.assertEqual(len(words), len(set(words)))

    def test_context_is_longer_than_the_passage_alone(self):
        corpus = make_corpus()
        assembler = ContextAssembler(corpus, window=1)
        passage = corpus.by_document('physics')[1]
        assembled = assembler.assemble(passage.passage_id)
        self.assertGreater(len(assembled.context_text),
                           len(assembled.passage_text))
        self.assertTrue(assembled.expanded)

    def test_window_zero_returns_only_the_passage(self):
        corpus = make_corpus()
        assembler = ContextAssembler(corpus, window=0)
        passage = corpus.by_document('physics')[1]
        assembled = assembler.assemble(passage.passage_id)
        self.assertEqual(assembled.neighbour_ids, [passage.passage_id])
        self.assertFalse(assembled.expanded)

    def test_parent_mode_groups_by_parent_id(self):
        corpus = make_corpus()
        assembler = ContextAssembler(corpus, use_parent=True)
        passage = corpus.by_document('physics')[0]
        assembled = assembler.assemble(passage.passage_id)
        for neighbour_id in assembled.neighbour_ids:
            self.assertEqual(corpus.get(neighbour_id).parent_id,
                             passage.parent_id)

    def test_scored_text_is_always_the_passage_not_the_context(self):
        # A score must never be reported against text the ranking never saw.
        corpus = make_corpus()
        assembler = ContextAssembler(corpus, window=2)
        passage = corpus.by_document('soil')[1]
        assembled = assembler.assemble(passage.passage_id)
        self.assertEqual(assembled.passage_text, passage.text)

    def test_respects_the_character_budget(self):
        corpus = make_corpus()
        assembler = ContextAssembler(corpus, window=3, max_chars=300)
        assembled = assembler.assemble(corpus.by_document('physics')[1].passage_id)
        self.assertLessEqual(len(assembled.context_text), 300)
        self.assertTrue(assembled.truncated)

    def test_unknown_passage_returns_none(self):
        self.assertIsNone(ContextAssembler(make_corpus()).assemble('nope'))

    def test_concept_lookup_populates_the_breadcrumb(self):
        corpus = make_corpus()
        passage = corpus.by_document('physics')[0]
        assembler = ContextAssembler(
            corpus, concept_lookup={passage.passage_id: ('Shielding', 'c1')})
        crumb = assembler.assemble(passage.passage_id).breadcrumb
        self.assertEqual(crumb.concept, 'Shielding')
        self.assertEqual(crumb.concept_id, 'c1')

    def test_rejects_negative_window(self):
        with self.assertRaises(ValueError):
            ContextAssembler(make_corpus(), window=-1)

    def test_generation_context_includes_citations(self):
        corpus = make_corpus()
        assembler = ContextAssembler(corpus, window=1)
        ids = [p.passage_id for p in corpus.by_document('physics')[:2]]
        context = assembler.build_generation_context(ids, max_words=300)
        self.assertIn('[', context)
        self.assertLessEqual(len(context.split()), 320)

    def test_generation_context_does_not_repeat_neighbours(self):
        # Adjacent hits expand into overlapping windows; without dedup the
        # same block would be included twice.
        corpus = make_corpus()
        assembler = ContextAssembler(corpus, window=1)
        passages = corpus.by_document('physics')
        ids = [passages[0].passage_id, passages[1].passage_id]
        context = assembler.build_generation_context(ids, max_words=2000)
        self.assertEqual(context.count('['), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineComposition(unittest.TestCase):

    def setUp(self):
        self.corpus = make_corpus()
        self.docs = self.corpus.documents()
        self.bm25 = BM25Index().fit(self.docs)
        self.ids = list(self.docs)

    def test_requires_a_retriever(self):
        with self.assertRaises(ValueError):
            Pipeline(retrievers=[], corpus=self.corpus)

    def test_single_retriever_runs(self):
        pipeline = Pipeline([self.bm25], self.corpus, top_k=5)
        trace = pipeline.search('electrostatic shielding')
        self.assertLessEqual(len(trace.results), 5)
        self.assertEqual(trace.retrievers_used, ['bm25'])

    def test_single_retriever_reports_no_fusion(self):
        # Fusion across one list is meaningless, and claiming a 100% share
        # would imply a comparison that never happened.
        pipeline = Pipeline([self.bm25], self.corpus)
        trace = pipeline.search('electrostatic shielding')
        self.assertEqual(trace.fusion_method, 'none')
        self.assertTrue(all(r.fusion is None for r in trace.results))

    def test_hybrid_records_fusion_shares(self):
        dense = FakeDense(self.ids)
        pipeline = Pipeline([self.bm25, dense], self.corpus)
        trace = pipeline.search('electrostatic shielding')
        self.assertEqual(trace.fusion_method, 'rrf')
        top = trace.results[0]
        self.assertIsNotNone(top.fusion)
        self.assertAlmostEqual(sum(top.contribution_shares().values()),
                               1.0, delta=1e-9)

    def test_names_describe_the_configuration(self):
        self.assertEqual(Pipeline([self.bm25], self.corpus).name, 'bm25')
        hybrid = Pipeline([self.bm25, FakeDense(self.ids)], self.corpus)
        self.assertEqual(hybrid.name, 'bm25+dense_rrf')
        with_rerank = Pipeline([self.bm25], self.corpus,
                               reranker=IdentityReranker())
        self.assertIn('identity', with_rerank.name)

    def test_describe_reports_every_component(self):
        pipeline = Pipeline([self.bm25], self.corpus,
                            reranker=IdentityReranker())
        described = pipeline.describe()
        self.assertEqual(described['retrievers'], ['bm25'])
        self.assertEqual(described['reranker'], 'identity')
        self.assertIsNone(described['expander'])

    def test_top_k_limits_results(self):
        pipeline = Pipeline([self.bm25], self.corpus, top_k=2)
        self.assertLessEqual(len(pipeline.search('soil erosion').results), 2)

    def test_ranks_are_sequential_from_one(self):
        pipeline = Pipeline([self.bm25], self.corpus, top_k=5)
        trace = pipeline.search('electrostatic shielding')
        self.assertEqual([r.rank for r in trace.results],
                         list(range(1, len(trace.results) + 1)))

    def test_empty_query_produces_no_results(self):
        pipeline = Pipeline([self.bm25], self.corpus)
        self.assertEqual(pipeline.search('').results, [])


class TestPipelineStages(unittest.TestCase):

    def setUp(self):
        self.corpus = make_corpus()
        self.docs = self.corpus.documents()
        self.bm25 = BM25Index().fit(self.docs)

    def test_reranker_reorders_and_records_the_delta(self):
        query = 'electrostatic shielding conductor'
        # Pick a target the retriever actually returns, otherwise the stub has
        # nothing to promote and the test would pass vacuously.
        candidates = [pid for pid, _ in self.bm25.search(query, top_k=5)]
        self.assertGreater(len(candidates), 1)
        target = candidates[-1]

        pipeline = Pipeline([self.bm25], self.corpus,
                            reranker=StubReranker(target), top_k=5)
        trace = pipeline.search(query)
        self.assertEqual(trace.results[0].passage_id, target)
        self.assertIsNotNone(trace.results[0].rerank)
        self.assertGreater(trace.results[0].rerank['rank_delta'], 0)
        self.assertEqual(trace.reranker_used, 'stub')

    def test_no_reranker_leaves_rerank_unset(self):
        pipeline = Pipeline([self.bm25], self.corpus)
        trace = pipeline.search('electrostatic shielding')
        self.assertEqual(trace.reranker_used, '')
        self.assertTrue(all(r.rerank is None for r in trace.results))
        self.assertNotIn('rerank', trace.timings_ms)

    def test_expansion_records_added_terms(self):
        expander = RM3Expander(self.bm25, self.docs, fb_terms=5)
        pipeline = Pipeline([self.bm25], self.corpus, expander=expander)
        trace = pipeline.search('shielding')
        self.assertEqual(trace.expansion_method, 'rm3')
        self.assertIsInstance(trace.expansion_terms, list)

    def test_matched_terms_come_from_the_lexical_index(self):
        pipeline = Pipeline([self.bm25], self.corpus, top_k=3)
        trace = pipeline.search('electrostatic shielding')
        self.assertTrue(trace.results)
        self.assertTrue(any(r.matched_terms for r in trace.results))

    def test_dense_only_pipeline_reports_no_matched_terms(self):
        # A dense score does not decompose into attributable terms, so none
        # are invented.
        pipeline = Pipeline([FakeDense(list(self.docs))], self.corpus, top_k=3)
        trace = pipeline.search('electrostatic shielding')
        self.assertTrue(all(r.matched_terms == [] for r in trace.results))

    def test_intent_aware_records_the_policy(self):
        pipeline = Pipeline([self.bm25], self.corpus, intent_aware=True)
        trace = pipeline.search('why does soil erode')
        self.assertEqual(trace.intent, 'causal')
        self.assertIn('weights', trace.intent_policy)

    def test_intent_aware_splits_a_comparison(self):
        pipeline = Pipeline([self.bm25], self.corpus, intent_aware=True)
        trace = pipeline.search('difference between consistency and availability')
        self.assertEqual(trace.intent, 'comparison')
        self.assertEqual(len(trace.subqueries), 3)

    def test_intent_policy_can_suppress_expansion(self):
        # DEFINITION policy sets expand=False, so a configured expander is
        # skipped and labelled as such rather than looking unconfigured.
        expander = RM3Expander(self.bm25, self.docs)
        pipeline = Pipeline([self.bm25], self.corpus, expander=expander,
                            intent_aware=True)
        trace = pipeline.search('what is electrostatic shielding')
        self.assertIn('skipped', trace.expansion_method)

    def test_assembler_attaches_context_and_breadcrumb(self):
        assembler = ContextAssembler(self.corpus, window=1)
        pipeline = Pipeline([self.bm25], self.corpus, assembler=assembler,
                            top_k=3)
        trace = pipeline.search('electrostatic shielding')
        self.assertTrue(trace.results[0].context_text)
        self.assertTrue(trace.results[0].breadcrumb)

    def test_timings_recorded_only_for_stages_that_ran(self):
        pipeline = Pipeline([self.bm25], self.corpus)
        trace = pipeline.search('soil')
        self.assertIn('retrieve', trace.timings_ms)
        self.assertNotIn('rewrite', trace.timings_ms)
        self.assertNotIn('expand', trace.timings_ms)
        self.assertGreater(trace.total_ms, 0)

    def test_trace_serialises(self):
        pipeline = Pipeline([self.bm25], self.corpus, top_k=2)
        payload = pipeline.search('soil erosion').to_dict()
        self.assertIn('results', payload)
        self.assertIn('timings_ms', payload)
        import json
        json.dumps(payload)   # must not raise

    def test_ranking_returns_ids_in_order(self):
        pipeline = Pipeline([self.bm25], self.corpus, top_k=3)
        trace = pipeline.search('soil erosion')
        self.assertEqual(trace.ranking(),
                         [r.passage_id for r in trace.results])


# ──────────────────────────────────────────────────────────────────────────────
# Test collection
# ──────────────────────────────────────────────────────────────────────────────

class TestTestCollection(unittest.TestCase):

    def setUp(self):
        self.collection = TestCollection(
            queries=[
                EvalQuery('q1', 'what is shielding', style='literal',
                          source_passage_id='physics::p0'),
                EvalQuery('q2', 'why no electricity in a metal box',
                          style='paraphrased',
                          source_passage_id='physics::p0'),
            ],
            judgements=[
                Judgement('q1', 'physics::p0', 2),
                Judgement('q1', 'physics::p1', 1, source='overlap'),
                Judgement('q2', 'physics::p0', 2),
            ])

    def test_qrels_shape_matches_the_metrics_contract(self):
        qrels = self.collection.qrels()
        self.assertEqual(qrels['q1']['physics::p0'], 2)
        self.assertEqual(qrels['q1']['physics::p1'], 1)

    def test_by_style_filters_queries_and_judgements(self):
        literal = self.collection.by_style('literal')
        self.assertEqual(len(literal), 1)
        self.assertEqual(set(literal.qrels()), {'q1'})

    def test_stats_counts_styles_and_grades(self):
        stats = self.collection.stats()
        self.assertEqual(stats.n_queries, 2)
        self.assertEqual(stats.by_style['literal'], 1)
        self.assertEqual(stats.by_style['paraphrased'], 1)

    def test_report_states_exact_coincidence_when_all_single_relevant(self):
        # Every query has exactly one relevant passage, so MAP == MRR and
        # Recall@k == Success@k *exactly*. The report must say so, otherwise
        # the results table gets built on duplicate columns.
        collection = TestCollection(
            queries=[EvalQuery('q1', 'a'), EvalQuery('q2', 'b')],
            judgements=[Judgement('q1', 'p0', 2), Judgement('q2', 'p1', 2)])
        report = collection.stats().report()
        self.assertIn('exactly', report)
        self.assertIn('MRR', report)

    def test_report_warns_when_most_queries_are_single_relevant(self):
        # Two of three single-relevant: the measures no longer coincide
        # exactly, so the wording must soften to "highly correlated".
        collection = TestCollection(
            queries=[EvalQuery(f'q{i}', 'x') for i in range(3)],
            judgements=[Judgement('q0', 'p0', 2),
                        Judgement('q1', 'p1', 2),
                        Judgement('q2', 'p2', 2), Judgement('q2', 'p3', 1)])
        report = collection.stats().report()
        self.assertIn('correlated', report)
        self.assertIn('67%', report)

    def test_report_is_silent_when_judgements_are_rich(self):
        collection = TestCollection(
            queries=[EvalQuery('q1', 'a')],
            judgements=[Judgement('q1', f'p{i}', 2) for i in range(4)])
        report = collection.stats().report()
        self.assertNotIn('NOTE', report)

    def test_stats_track_the_relevant_distribution(self):
        stats = self.collection.stats()
        self.assertEqual(stats.max_relevant_per_query, 2)
        self.assertEqual(stats.single_relevant_queries, 1)
        self.assertAlmostEqual(stats.mean_relevant_per_query, 1.5, delta=TOL)

    def test_merge_keeps_the_higher_grade(self):
        self.collection.merge_judgements(
            [Judgement('q1', 'physics::p1', 2, source='human')])
        self.assertEqual(self.collection.qrels()['q1']['physics::p1'], 2)

    def test_merge_adds_new_judgements(self):
        self.collection.merge_judgements(
            [Judgement('q2', 'physics::p9', 1, source='llm')])
        self.assertEqual(self.collection.qrels()['q2']['physics::p9'], 1)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.collection.save(tmp)
            loaded = TestCollection.load(tmp)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded.qrels(), self.collection.qrels())

    def test_load_from_missing_directory_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = TestCollection.load(Path(tmp) / 'absent')
        self.assertEqual(len(loaded), 0)


class TestSyntheticGeneration(unittest.TestCase):

    def test_parses_a_json_object(self):
        parsed = SyntheticQueryGenerator._parse(
            '{"literal": "what is entropy", "paraphrased": "what is disorder"}')
        self.assertEqual(parsed['literal'], 'what is entropy')

    def test_parses_a_fenced_object(self):
        parsed = SyntheticQueryGenerator._parse(
            '```json\n{"literal": "a", "paraphrased": "b"}\n```')
        self.assertEqual(parsed['paraphrased'], 'b')

    def test_strips_a_thinking_block(self):
        parsed = SyntheticQueryGenerator._parse(
            '<think>hmm</think>{"literal": "a", "paraphrased": "b"}')
        self.assertEqual(parsed['literal'], 'a')

    def test_unparseable_returns_empty(self):
        self.assertEqual(SyntheticQueryGenerator._parse('no json here'), {})

    def test_sampling_is_spread_across_documents(self):
        # A long document must not dominate the query set, or the evaluation
        # becomes mostly about one textbook.
        corpus = make_corpus()
        generator = SyntheticQueryGenerator(corpus, n_passages=6, min_tokens=10)
        sampled = generator.sample_passages()
        self.assertGreaterEqual(len({p.doc_id for p in sampled}), 2)

    def test_sampling_is_reproducible(self):
        corpus = make_corpus()
        first = SyntheticQueryGenerator(corpus, n_passages=5, seed=7,
                                        min_tokens=10).sample_passages()
        second = SyntheticQueryGenerator(corpus, n_passages=5, seed=7,
                                         min_tokens=10).sample_passages()
        self.assertEqual([p.passage_id for p in first],
                         [p.passage_id for p in second])

    def test_overlapping_neighbour_receives_partial_credit(self):
        # The p25/p26 case: adjacent passages share the configured overlap and
        # frequently contain the same sentence, so returning the neighbour is
        # not a miss.
        corpus = make_corpus()
        generator = SyntheticQueryGenerator(corpus)
        source = corpus.by_document('physics')[1]
        judgements = generator._judgements_for('q0', source)
        grades = {j.passage_id: j.grade for j in judgements}
        self.assertEqual(grades[source.passage_id], 2)
        self.assertTrue(any(g == 1 for pid, g in grades.items()
                            if pid != source.passage_id))

    def test_manual_collection_builds_judgements(self):
        corpus = make_corpus()
        passage_id = corpus.by_document('soil')[0].passage_id
        collection = manual_collection(
            corpus, [('why is farmland becoming useless', passage_id)])
        self.assertEqual(len(collection), 1)
        self.assertEqual(collection.qrels()['m000'][passage_id], 2)

    def test_manual_collection_rejects_unknown_passage(self):
        with self.assertRaises(KeyError):
            manual_collection(make_corpus(), [('q', 'nope')])


# ──────────────────────────────────────────────────────────────────────────────
# Experiments
# ──────────────────────────────────────────────────────────────────────────────

class TestExperiments(unittest.TestCase):

    def setUp(self):
        self.corpus = make_corpus()
        self.docs = self.corpus.documents()
        self.bm25 = BM25Index().fit(self.docs)
        passage_id = self.corpus.by_document('physics')[0].passage_id
        self.collection = manual_collection(
            self.corpus, [('electrostatic shielding conductor', passage_id)])

    def test_run_system_produces_metrics_and_latency(self):
        pipeline = Pipeline([self.bm25], self.corpus, top_k=10)
        result, traces = run_system(pipeline, self.collection, progress=False)
        self.assertEqual(result.n_queries, 1)
        self.assertIn('nDCG@10', result.means)
        self.assertGreater(result.mean_latency_ms, 0)
        self.assertEqual(traces, [])

    def test_traces_captured_only_when_requested(self):
        pipeline = Pipeline([self.bm25], self.corpus)
        _, traces = run_system(pipeline, self.collection, progress=False,
                               capture_traces=True)
        self.assertEqual(len(traces), 1)

    def test_comparison_table_renders(self):
        pipeline = Pipeline([self.bm25], self.corpus)
        result, _ = run_system(pipeline, self.collection, progress=False)
        table = comparison_table({result.name: result})
        self.assertIn('nDCG@10', table)
        self.assertIn(result.name, table)

    def test_comparison_table_omits_redundant_measures(self):
        # MAP equals MRR when there is one relevant passage per query, so
        # printing both would show duplicate columns.
        pipeline = Pipeline([self.bm25], self.corpus)
        result, _ = run_system(pipeline, self.collection, progress=False)
        table = comparison_table({result.name: result})
        self.assertNotIn('MAP', table)

    def test_comparison_table_marks_significance(self):
        result = SystemResult('sys', {}, {'nDCG@10': 0.8}, {}, n_queries=20)
        table = comparison_table({'sys': result},
                                 significance={'sys': {'significant': 1.0}})
        self.assertIn('†', table)

    def test_ablation_table_shows_deltas(self):
        full = SystemResult('full', {}, {'nDCG@10': 0.80}, {})
        without = SystemResult('no_rerank', {}, {'nDCG@10': 0.70}, {})
        table = ablation_table({'full': full, 'no_rerank': without}, 'full')
        self.assertIn('-0.100', table)

    def test_ablation_table_handles_unknown_system(self):
        self.assertIn('unknown', ablation_table({}, 'absent'))

    def test_empty_results_table(self):
        self.assertIn('no results', comparison_table({}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
