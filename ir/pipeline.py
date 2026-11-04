"""
The retrieval pipeline: one composable object that runs every stage and
records what each one did.

Two requirements shaped this module.

First, every system in the results table must be the *same code* with
different components, or the comparison measures implementation differences
rather than method differences. So a pipeline is configuration:

    Pipeline(retrievers=[bm25])                          # BM25 alone
    Pipeline(retrievers=[bm25, dense])                   # hybrid, RRF
    Pipeline(retrievers=[bm25, dense], reranker=ce)      # + reranking
    Pipeline(retrievers=[bm25, dense], reranker=ce,
             expander=rm3)                               # + expansion

and an ablation is removing one argument.

Second, "why was this retrieved?" has to be answerable with real numbers, and
that is only possible if every stage records its contribution as it runs.
Retrofitting it later is not feasible -- by the time a ranking is a list of
ids, the information is gone. So `search()` returns a `RetrievalTrace`, not a
list.

Stage order:

    rewrite -> classify -> expand -> retrieve -> fuse -> rerank -> assemble

Each stage is optional. Absent stages are skipped and recorded as skipped, so
the trace of a BM25-only run has the same shape as the trace of the full
pipeline and the UI needs no special cases.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .fusion import fuse
from .query.expand import ExpandedQuery, identity_expansion
from .query.intent import Intent, analyse
from .rerank import RerankedResult

#: Retrievers that benefit from query expansion. Adding synonyms to a bag of
#: words helps when a term is simply absent; adding them to a sentence before
#: embedding it can blur the vector instead. Expansion is therefore applied to
#: the sparse arm only unless overridden -- and the experiment can measure
#: whether that default is right.
SPARSE_RETRIEVERS = frozenset({'bm25', 'tfidf'})


@dataclass
class ScoredPassage:
    """
    One passage in the final ranking, with its full provenance.

    Every stage that touched this passage leaves a dict here. A stage that did
    not run leaves None, which is meaningfully different from a stage that ran
    and scored zero.
    """
    passage_id: str
    score: float
    rank: int

    retriever_scores: dict[str, dict] = field(default_factory=dict)
    fusion: dict | None = None
    rerank: dict | None = None

    matched_terms: list[str] = field(default_factory=list)
    breadcrumb: str = ''
    citation: str = ''
    text: str = ''
    context_text: str = ''

    def contribution_shares(self) -> dict[str, float]:
        """
        Each retriever's share of the fused score, for the explanation panel.

        Empty when fusion did not run (a single-retriever pipeline), because
        attributing 100% to the only retriever would imply a comparison that
        was never made.
        """
        if not self.fusion:
            return {}
        return self.fusion.get('shares', {})

    @property
    def found_by(self) -> list[str]:
        return sorted(self.retriever_scores)


@dataclass
class RetrievalTrace:
    """
    Everything that happened during one search.

    This is the object the explanation UI, the latency column of the analytics
    dashboard, and the debugging workflow all read from. It is also what gets
    serialised into a run file, so the fields are plain data.
    """
    query: str
    results: list[ScoredPassage] = field(default_factory=list)

    query_rewritten: str = ''
    rewrite_changed: bool = False
    rewrite_reason: str = ''

    intent: str = ''
    intent_policy: dict = field(default_factory=dict)
    subqueries: list[str] = field(default_factory=list)

    expansion_method: str = 'none'
    expansion_terms: list[str] = field(default_factory=list)

    retrievers_used: list[str] = field(default_factory=list)
    fusion_method: str = ''
    reranker_used: str = ''
    candidate_count: int = 0

    timings_ms: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def effective_query(self) -> str:
        """The query actually sent to the retrievers."""
        return self.query_rewritten or self.query

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())

    def ranking(self) -> list[str]:
        """Passage ids in rank order -- the format the evaluator consumes."""
        return [r.passage_id for r in self.results]

    def to_dict(self) -> dict:
        """JSON-serialisable form, for run files and the trace viewer."""
        return {
            'query': self.query,
            'query_rewritten': self.query_rewritten,
            'rewrite_changed': self.rewrite_changed,
            'rewrite_reason': self.rewrite_reason,
            'intent': self.intent,
            'subqueries': self.subqueries,
            'expansion_method': self.expansion_method,
            'expansion_terms': self.expansion_terms,
            'retrievers_used': self.retrievers_used,
            'fusion_method': self.fusion_method,
            'reranker_used': self.reranker_used,
            'candidate_count': self.candidate_count,
            'timings_ms': self.timings_ms,
            'config': self.config,
            'results': [
                {
                    'passage_id': r.passage_id,
                    'score': r.score,
                    'rank': r.rank,
                    'retriever_scores': r.retriever_scores,
                    'fusion': r.fusion,
                    'rerank': r.rerank,
                    'matched_terms': r.matched_terms,
                    'breadcrumb': r.breadcrumb,
                    'citation': r.citation,
                }
                for r in self.results
            ],
        }


class Pipeline:
    """
    A configured retrieval system.

    Args:
        retrievers:       one or more fitted `Retriever` objects. Order is
                          irrelevant; they are keyed by `name`.
        corpus:           an `ir.corpus.Corpus`, for passage text and metadata.
        fusion_method:    'rrf', 'combsum' or 'combmnz'.
        rrf_k:            RRF damping constant.
        expander:         object with `.expand(query) -> ExpandedQuery`.
        rewriter:         object with `.rewrite(query, history)`.
        reranker:         object with `.rerank(query, candidates, texts)`.
        assembler:        an `ir.context.ContextAssembler`.
        intent_aware:     let the classified intent set fusion weights, rerank
                          depth and whether to expand. Off by default so the
                          fixed-configuration baseline is the default and
                          intent routing has to earn its place.
        candidate_depth:  how many results each retriever returns into fusion.
                          Must exceed the reranker's depth, or reranking has
                          nothing to reorder.
        top_k:            final result count.
        expand_retrievers: which retrievers receive the expanded query.
                          Defaults to the sparse ones.
        name:             identifier for results tables and run files.
    """

    def __init__(self, retrievers: list, corpus, fusion_method: str = 'rrf',
                 rrf_k: int = 60, expander=None, rewriter=None, reranker=None,
                 assembler=None, intent_aware: bool = False,
                 candidate_depth: int = 50, top_k: int = 10,
                 expand_retrievers: set[str] | None = None,
                 name: str | None = None):
        if not retrievers:
            raise ValueError("a pipeline needs at least one retriever")

        self.retrievers = list(retrievers)
        self.corpus = corpus
        self.documents = corpus.documents()
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.expander = expander
        self.rewriter = rewriter
        self.reranker = reranker
        self.assembler = assembler
        self.intent_aware = intent_aware
        self.candidate_depth = candidate_depth
        self.top_k = top_k
        self.expand_retrievers = (SPARSE_RETRIEVERS if expand_retrievers is None
                                  else set(expand_retrievers))
        self.name = name or self._default_name()

    def _default_name(self) -> str:
        """A readable identifier describing the configuration."""
        parts = ['+'.join(sorted(r.name for r in self.retrievers))]
        if len(self.retrievers) > 1:
            parts.append(self.fusion_method)
        if self.expander is not None:
            parts.append(getattr(self.expander, 'method', 'expand'))
        if self.reranker is not None:
            parts.append(getattr(self.reranker, 'name', 'rerank'))
        if self.intent_aware:
            parts.append('intent')
        return '_'.join(parts)

    def describe(self) -> dict:
        """Configuration summary, recorded in every trace and run file."""
        return {
            'name': self.name,
            'retrievers': sorted(r.name for r in self.retrievers),
            'fusion': self.fusion_method if len(self.retrievers) > 1 else None,
            'rrf_k': self.rrf_k if self.fusion_method == 'rrf' else None,
            'expander': getattr(self.expander, 'method', None),
            'rewriter': getattr(self.rewriter, 'name', None),
            'reranker': getattr(self.reranker, 'name', None),
            'intent_aware': self.intent_aware,
            'candidate_depth': self.candidate_depth,
            'top_k': self.top_k,
        }

    # ── stages ────────────────────────────────────────────────────────────
    def _retrieve_one(self, retriever, query: str,
                      expanded: ExpandedQuery | None) -> list[tuple[str, float]]:
        """
        Run one retriever, using the expanded query if it should receive one.

        BM25 gets the weighted form, which is exact; anything else gets the
        expanded text, which loses the weights. That difference is why
        expansion is restricted to the sparse arm by default.
        """
        use_expansion = (expanded is not None
                         and bool(expanded.added_terms)
                         and retriever.name in self.expand_retrievers)

        if use_expansion and hasattr(retriever, 'search_weighted'):
            return retriever.search_weighted(expanded.weights,
                                             top_k=self.candidate_depth)
        if use_expansion:
            return retriever.search(expanded.as_text(),
                                    top_k=self.candidate_depth)
        return retriever.search(query, top_k=self.candidate_depth)

    def _matched_terms(self, query: str, passage_id: str) -> list[str]:
        """
        Terms present in both query and passage, from the lexical index.

        Only BM25 can answer this. A dense score is a single inner product in
        a learned space and does not decompose into attributable terms, so a
        dense-only pipeline reports no matched terms rather than inventing
        them. That asymmetry is stated plainly in the UI.
        """
        for retriever in self.retrievers:
            if retriever.name == 'bm25' and hasattr(retriever, 'explain'):
                try:
                    breakdown = retriever.explain(query, passage_id)
                except KeyError:
                    return []
                return [row['term'] for row in breakdown['terms']
                        if row['tf'] > 0]
        return []

    # ── search ────────────────────────────────────────────────────────────
    def search(self, query: str, history: list | None = None) -> RetrievalTrace:
        """
        Run the full pipeline and return a trace.

        Stages that are not configured are skipped; their timing entries are
        omitted rather than recorded as zero, so the dashboard shows which
        stages a system actually ran.
        """
        trace = RetrievalTrace(query=query, config=self.describe())
        timer = time.perf_counter

        # 1. Rewrite -------------------------------------------------------
        effective = query
        if self.rewriter is not None:
            start = timer()
            rewritten = self.rewriter.rewrite(query, history or [])
            trace.timings_ms['rewrite'] = (timer() - start) * 1000
            effective = rewritten.rewritten
            trace.query_rewritten = rewritten.rewritten
            trace.rewrite_changed = rewritten.changed
            trace.rewrite_reason = rewritten.reason
        else:
            trace.query_rewritten = query

        # 2. Classify ------------------------------------------------------
        policy = None
        subqueries = [effective]
        if self.intent_aware:
            start = timer()
            classified = analyse(effective)
            trace.timings_ms['classify'] = (timer() - start) * 1000
            policy = classified.policy
            subqueries = classified.subqueries()
            trace.intent = classified.intent.value
            trace.intent_policy = {
                'weights': policy.retriever_weights,
                'expand': policy.expand,
                'rerank_depth': policy.rerank_depth,
                'split_entities': classified.should_split,
            }
        else:
            trace.intent = Intent.FACTUAL.value
        trace.subqueries = subqueries

        # 3. Expand --------------------------------------------------------
        expanded: ExpandedQuery | None = None
        should_expand = self.expander is not None and (
            policy is None or policy.expand)
        if should_expand:
            start = timer()
            expanded = self.expander.expand(effective)
            trace.timings_ms['expand'] = (timer() - start) * 1000
            trace.expansion_method = expanded.method
            trace.expansion_terms = list(expanded.added_terms)
        elif self.expander is not None:
            # Configured but suppressed by the intent policy; say so rather
            # than leaving it looking unconfigured.
            trace.expansion_method = f"{getattr(self.expander, 'method', '?')}(skipped)"
            expanded = identity_expansion(effective)

        # 4. Retrieve ------------------------------------------------------
        start = timer()
        runs: dict[str, list[tuple[str, float]]] = {}
        raw_scores: dict[str, dict[str, float]] = {}
        for retriever in self.retrievers:
            for index, subquery in enumerate(subqueries):
                sub_expanded = expanded if index == 0 else None
                results = self._retrieve_one(retriever, subquery, sub_expanded)
                key = (retriever.name if len(subqueries) == 1
                       else f"{retriever.name}#{index}")
                runs[key] = results
                scores = raw_scores.setdefault(retriever.name, {})
                for passage_id, score in results:
                    # Across subqueries keep the best score per passage.
                    if score > scores.get(passage_id, float('-inf')):
                        scores[passage_id] = score
        trace.timings_ms['retrieve'] = (timer() - start) * 1000
        trace.retrievers_used = sorted(r.name for r in self.retrievers)

        # 5. Fuse ----------------------------------------------------------
        start = timer()
        weights = None
        if policy is not None and policy.retriever_weights:
            weights = {}
            for key in runs:
                base = key.split('#')[0]
                weights[key] = policy.retriever_weights.get(base, 1.0)
        fused = fuse(runs, method=self.fusion_method, k=self.rrf_k,
                     weights=weights)
        trace.timings_ms['fuse'] = (timer() - start) * 1000
        trace.fusion_method = (self.fusion_method if len(runs) > 1 else 'none')
        trace.candidate_count = len(fused)

        candidate_ids = [r.doc_id for r in fused]
        fusion_by_id = {r.doc_id: r for r in fused}

        # 6. Rerank --------------------------------------------------------
        reranked: dict[str, RerankedResult] = {}
        final_order = candidate_ids
        if self.reranker is not None and candidate_ids:
            depth = policy.rerank_depth if policy is not None else None
            start = timer()
            results = self.reranker.rerank(effective, candidate_ids,
                                           self.documents, top_n=depth)
            trace.timings_ms['rerank'] = (timer() - start) * 1000
            reranked = {r.doc_id: r for r in results}
            final_order = [r.doc_id for r in results]
            trace.reranker_used = getattr(self.reranker, 'name', 'reranker')

        final_order = final_order[:self.top_k]

        # 7. Assemble ------------------------------------------------------
        start = timer()
        scored: list[ScoredPassage] = []
        for rank, passage_id in enumerate(final_order, start=1):
            passage = self.corpus.get(passage_id)
            fusion_result = fusion_by_id.get(passage_id)

            retriever_scores = {}
            for name, scores in raw_scores.items():
                if passage_id in scores:
                    retriever_scores[name] = {'score': scores[passage_id]}
            if fusion_result:
                for contribution in fusion_result.contributions:
                    base = contribution.retriever.split('#')[0]
                    if base in retriever_scores:
                        retriever_scores[base].setdefault(
                            'rank', contribution.rank)

            fusion_info = None
            if fusion_result and len(runs) > 1:
                fusion_info = {
                    'score': fusion_result.score,
                    'rank': fusion_result.rank,
                    'shares': fusion_result.contribution_shares(),
                    'n_retrievers': fusion_result.n_retrievers,
                }

            rerank_info = None
            rerank_result = reranked.get(passage_id)
            if rerank_result is not None:
                rerank_info = {
                    'score': rerank_result.score,
                    'rank': rerank_result.rank,
                    'previous_rank': rerank_result.previous_rank,
                    'rank_delta': rerank_result.rank_delta,
                    'reranked': rerank_result.reranked,
                    'probability': rerank_result.probability,
                }

            if rerank_result is not None and rerank_result.reranked:
                final_score = rerank_result.score
            elif fusion_result is not None:
                final_score = fusion_result.score
            else:
                final_score = 0.0

            entry = ScoredPassage(
                passage_id=passage_id,
                score=final_score,
                rank=rank,
                retriever_scores=retriever_scores,
                fusion=fusion_info,
                rerank=rerank_info,
                matched_terms=self._matched_terms(effective, passage_id),
                text=passage.text if passage else '',
                citation=passage.citation if passage else '',
            )

            if self.assembler is not None:
                assembled = self.assembler.assemble(passage_id)
                if assembled is not None:
                    entry.context_text = assembled.context_text
                    entry.breadcrumb = str(assembled.breadcrumb)

            scored.append(entry)

        trace.timings_ms['assemble'] = (timer() - start) * 1000
        trace.results = scored
        return trace

    def __repr__(self) -> str:
        return f"Pipeline(name={self.name!r}, top_k={self.top_k})"
