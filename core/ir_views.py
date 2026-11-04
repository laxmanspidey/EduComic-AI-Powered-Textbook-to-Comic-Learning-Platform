"""
Views for the retrieval layer: search, system comparison, analytics, judging.

Kept separate from `core/views.py`, which serves the comic application. The
two share models but nothing else, and the retrieval views are the part a
reader of this project will want to find.
"""
from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_POST

from .models import (
    RelevanceJudgement,
    SearchQuery,
    SearchSession,
    Textbook,
)

#: Selectable retrieval systems. These names match the rows of the results
#: table, so the application and the experiments run identical configurations
#: -- which is what makes a measured number mean something about the app.
SYSTEMS = (
    ('tfidf', 'TF-IDF'),
    ('bm25', 'BM25'),
    ('dense', 'Dense'),
    ('hybrid', 'Hybrid (RRF)'),
    ('hybrid_rerank', 'Hybrid + Reranker'),
    ('full', 'Full pipeline'),
)
SYSTEM_KEYS = frozenset(key for key, _ in SYSTEMS)


@login_required
def textbook_search(request, pk):
    """
    Search one textbook.

    Scoped to a single textbook deliberately. Measured on this corpus, a
    whole-library index answers "what does CAP mean" with three physics
    capacitor passages in the top four, because the embedding model and the
    cross-encoder both conflate CAP with capacit-. Restricting the index to
    the textbook being read removes those passages from consideration
    entirely, which is a structural fix rather than a ranking tweak.
    """
    from . import ir_service

    textbook = get_object_or_404(Textbook, pk=pk, user=request.user)
    query = (request.GET.get('q') or '').strip()
    system = request.GET.get('system', 'full')
    if system not in SYSTEM_KEYS:
        system = 'full'
    conversational = request.GET.get('conversational') == '1'
    # Answer generation costs an Ollama call on top of retrieval, so it is
    # switchable. Default on: a student asked a question and wants an answer,
    # not a list of passages.
    #
    # An unchecked checkbox sends nothing, so its absence is only meaningful
    # once the form has actually been submitted. The presence of 'q' is what
    # distinguishes a submitted form from a first visit -- relying on a hidden
    # companion field instead would make the result depend on parameter order.
    want_answer = (request.GET.get('answer') == '1'
                   if 'q' in request.GET else True)

    context = {
        'textbook': textbook,
        'query': query,
        'system': system,
        'systems': SYSTEMS,
        'conversational': conversational,
        'want_answer': want_answer,
        'trace': None,
        'answer': None,
        'engine_stats': None,
        'build_error': None,
        'session': None,
    }

    try:
        engine = ir_service.get_engine(textbook)
        context['engine_stats'] = engine.stats()
    except (FileNotFoundError, ValueError) as error:
        # A scanned PDF produces no passages. Reporting that is far more
        # useful than an empty result list, which looks like a ranking failure
        # when it is really an ingestion one.
        context['build_error'] = str(error)
        return render(request, 'core/search.html', context)

    if not query:
        return render(request, 'core/search.html', context)

    session = None
    history = []
    if conversational:
        session, _ = SearchSession.objects.get_or_create(
            user=request.user, textbook=textbook)
        history = session.history()
    context['session'] = session

    trace = ir_service.search(textbook, query, system=system, top_k=10,
                              history=history or None)
    context['trace'] = trace

    answer = None
    if want_answer:
        answer = ir_service.answer_question(textbook, trace)
        context['answer'] = answer

    payload = trace.to_dict()
    if answer is not None:
        # Stored inside the trace rather than as its own column: like the rest
        # of the trace it is write-once diagnostic data whose shape will change
        # as the generator does, and it is only ever read back for the query it
        # belongs to.
        payload['answer'] = answer.to_dict()

    SearchQuery.objects.create(
        session=session, user=request.user, textbook=textbook,
        query_text=query, rewritten_text=trace.query_rewritten,
        intent=trace.intent, system=system,
        expansion_terms=trace.expansion_terms, trace=payload,
        result_count=len(trace.results),
        # The answer is the better conversation turn: a follow-up resolves
        # against what the student was told, not against a raw passage.
        top_snippet=(answer.text[:300] if answer and answer.grounded
                     else (trace.results[0].text[:300] if trace.results else '')),
        total_ms=trace.total_ms + (answer.latency_ms if answer else 0.0),
    )
    return render(request, 'core/search.html', context)


@login_required
def search_compare(request, pk):
    """
    Run every retrieval system on one query, side by side.

    The demo view for Problem #7, classical versus neural. It is a
    qualitative illustration only: a single query cannot establish that one
    method beats another, and the template says so explicitly. The evidence
    is the measured table produced by tools/run_experiments.py.
    """
    from . import ir_service

    textbook = get_object_or_404(Textbook, pk=pk, user=request.user)
    query = (request.GET.get('q') or '').strip()
    context = {'textbook': textbook, 'query': query, 'columns': [],
               'build_error': None, 'agreement': None}

    if not query:
        return render(request, 'core/search_compare.html', context)

    try:
        engine = ir_service.get_engine(textbook)
    except (FileNotFoundError, ValueError) as error:
        context['build_error'] = str(error)
        return render(request, 'core/search_compare.html', context)

    for system, label in SYSTEMS:
        if system == 'hybrid_rerank':
            continue   # 'full' already covers reranking; keep five columns
        trace = engine.pipeline(system=system, top_k=5).search(query)
        context['columns'].append({
            'system': system, 'label': label,
            'total_ms': trace.total_ms, 'results': trace.results,
        })

    # Set overlap across the systems' top-5. The quickest way to see whether
    # the methods genuinely disagree on this query: when they agree
    # completely, neither fusion nor reranking can change anything, and the
    # query tells you nothing about their relative quality.
    sets = [{r.passage_id for r in column['results']}
            for column in context['columns'] if column['results']]
    if sets:
        union = set().union(*sets)
        shared = set.intersection(*sets) if len(sets) > 1 else sets[0]
        context['agreement'] = {
            'shared': len(shared), 'union': len(union),
            'ratio': len(shared) / len(union) if union else 0.0,
        }
    return render(request, 'core/search_compare.html', context)


@login_required
def ir_dashboard(request):
    """
    Retrieval analytics.

    Two halves, and the distinction is the point. The upper half reports the
    offline experiment: measured effectiveness per system on the test
    collection, with significance tests. The lower half reports the live
    search log: what was asked, how it was classified, where time went.

    Only the upper half is evidence about retrieval quality. The template
    labels them separately so a demo cannot accidentally present usage
    statistics as effectiveness results.
    """
    from ir.eval.collection import TestCollection
    from ir.eval.experiments import (
        PRIMARY_MEASURE,
        comparison_table,
        latest_results,
        significance_table,
    )

    loaded = latest_results()
    results, significance = loaded if loaded else ({}, {})

    rows = []
    for name, result in sorted(
            results.items(), key=lambda kv: -kv[1].measure(PRIMARY_MEASURE)):
        entry = significance.get(name, {})
        rows.append({
            'name': name,
            'ndcg10': result.measure('nDCG@10'),
            'ndcg5': result.measure('nDCG@5'),
            'mrr': result.measure('MRR'),
            'p1': result.measure('P@1'),
            'latency': result.mean_latency_ms,
            'n_queries': result.n_queries,
            'delta': entry.get('delta'),
            'p_value': entry.get('p_value'),
            'significant': bool(entry.get('significant')),
        })

    best = max((row['ndcg10'] for row in rows), default=0.0) or 1.0
    for row in rows:
        row['bar_pct'] = round(100 * row['ndcg10'] / best, 1)

    collection = TestCollection.load()
    queries = list(SearchQuery.objects.filter(user=request.user)[:50])

    intent_counts: dict[str, int] = {}
    stage_totals: dict[str, float] = {}
    for entry in queries:
        key = entry.intent or 'unknown'
        intent_counts[key] = intent_counts.get(key, 0) + 1
        for stage, value in entry.stage_timings.items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + value

    stage_means = ({stage: total / len(queries)
                    for stage, total in stage_totals.items()}
                   if queries else {})
    slowest = max(stage_means.values(), default=0.0) or 1.0

    return render(request, 'core/ir_dashboard.html', {
        'rows': rows,
        'primary_measure': PRIMARY_MEASURE,
        'has_results': bool(results),
        'comparison_markdown': comparison_table(results, significance),
        'significance_markdown': significance_table(significance),
        'collection_stats': collection.stats() if len(collection) else None,
        'recent_queries': queries[:15],
        'query_total': len(queries),
        'intent_counts': sorted(intent_counts.items(), key=lambda kv: -kv[1]),
        'stage_means': [
            (stage, value, round(100 * value / slowest, 1))
            for stage, value in sorted(stage_means.items(),
                                        key=lambda kv: -kv[1])
        ],
        'rewritten_count': sum(1 for q in queries if q.was_rewritten),
    })


@login_required
@require_POST
def judge_passage(request):
    """
    Record a human relevance judgement.

    Used to verify a subset of the synthetic test collection. The generated
    collection assumes the passage a question was written from is the relevant
    one; that assumption needs checking, and the human-versus-synthetic
    agreement rate is itself a reportable result.
    """
    query_id = (request.POST.get('query_id') or '').strip()
    passage_id = (request.POST.get('passage_id') or '').strip()
    if not query_id or not passage_id:
        return JsonResponse({'error': 'query_id and passage_id are required'},
                            status=400)
    try:
        grade = int(request.POST.get('grade', 0))
    except (TypeError, ValueError):
        return JsonResponse({'error': 'grade must be 0, 1 or 2'}, status=400)
    if grade not in (0, 1, 2):
        return JsonResponse({'error': 'grade must be 0, 1 or 2'}, status=400)

    _, created = RelevanceJudgement.objects.update_or_create(
        user=request.user, query_id=query_id, passage_id=passage_id,
        defaults={
            'grade': grade,
            'query_text': (request.POST.get('query_text') or '')[:2000],
            'passage_text': (request.POST.get('passage_text') or '')[:4000],
        })
    return JsonResponse({'saved': True, 'created': created, 'grade': grade})
