"""
Information Retrieval effectiveness measures.

All functions share one contract:

    ranking : list[str]
        Document ids returned by a system, best-first (rank 1 = index 0).
    qrels   : dict[str, int]
        Relevance judgements for the query. Maps document id -> graded
        relevance (0 = non-relevant, 1 = relevant, 2 = highly relevant, ...).
        Documents absent from `qrels` are treated as relevance 0. This is the
        standard TREC assumption: unjudged documents count as non-relevant.

Binary measures (precision, recall, MRR, MAP) collapse graded judgements using
`rel_threshold`: a document counts as relevant when `grade >= rel_threshold`.
Graded measures (DCG, nDCG) use the grades directly.

References
----------
Jarvelin & Kekalainen (2002), "Cumulated gain-based evaluation of IR
    techniques", ACM TOIS 20(4) -- the origin of (n)DCG.
Manning, Raghavan & Schutze (2008), "Introduction to Information Retrieval",
    ch. 8 -- precision, recall, MAP and the averaging conventions.
"""
from __future__ import annotations

import math

# ──────────────────────────────────────────────────────────────────────────────
# Set-based measures
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(ranking: list[str], qrels: dict[str, int], k: int,
                   rel_threshold: int = 1) -> float:
    """
    Fraction of the top-k retrieved documents that are relevant.

        P@k = |{relevant} and {top-k retrieved}| / k

    The denominator is always `k`, not `len(ranking[:k])`. A system returning
    only 3 documents at k=10 is charged for the 7 it did not return, which is
    the behaviour we want when comparing systems at a fixed cutoff.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for d in ranking[:k] if qrels.get(d, 0) >= rel_threshold)
    return hits / k


def recall_at_k(ranking: list[str], qrels: dict[str, int], k: int,
                rel_threshold: int = 1) -> float:
    """
    Fraction of all known relevant documents appearing in the top k.

        R@k = |{relevant} and {top-k retrieved}| / |{relevant}|

    Returns 0.0 when the query has no relevant documents, so such queries do
    not silently inflate the mean. Filter them out upstream instead.
    """
    total_relevant = sum(1 for g in qrels.values() if g >= rel_threshold)
    if total_relevant == 0:
        return 0.0
    hits = sum(1 for d in ranking[:k] if qrels.get(d, 0) >= rel_threshold)
    return hits / total_relevant


def f1_at_k(ranking: list[str], qrels: dict[str, int], k: int,
            rel_threshold: int = 1) -> float:
    """Harmonic mean of P@k and R@k."""
    p = precision_at_k(ranking, qrels, k, rel_threshold)
    r = recall_at_k(ranking, qrels, k, rel_threshold)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def success_at_k(ranking: list[str], qrels: dict[str, int], k: int,
                 rel_threshold: int = 1) -> float:
    """
    1.0 if at least one relevant document is in the top k, else 0.0.

    Also called hit rate. Appropriate for question answering, where a single
    good passage is enough to answer the question.
    """
    return 1.0 if any(qrels.get(d, 0) >= rel_threshold for d in ranking[:k]) else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Rank-based measures
# ──────────────────────────────────────────────────────────────────────────────

def reciprocal_rank(ranking: list[str], qrels: dict[str, int],
                    k: int | None = None, rel_threshold: int = 1) -> float:
    """
    Reciprocal of the rank of the first relevant document.

        RR = 1 / rank_of_first_relevant        (0 if none found)

    Averaged over queries this gives MRR. It measures only how quickly the
    first good result arrives and is blind to everything below it, which is
    why it suits single-answer tasks rather than recall-oriented ones.
    """
    pool = ranking[:k] if k else ranking
    for i, d in enumerate(pool, start=1):
        if qrels.get(d, 0) >= rel_threshold:
            return 1.0 / i
    return 0.0


def average_precision(ranking: list[str], qrels: dict[str, int],
                      k: int | None = None, rel_threshold: int = 1) -> float:
    """
    Mean of the precision values taken at each relevant document's rank.

        AP = (1/R) * sum over relevant ranks i of P@i

    R is the total number of relevant documents in the judgements. Dividing by
    R rather than by the number actually retrieved is the TREC convention: it
    charges the system for relevant documents it never returned, so AP rewards
    recall as well as precision.

    Averaged over queries this gives MAP.
    """
    total_relevant = sum(1 for g in qrels.values() if g >= rel_threshold)
    if total_relevant == 0:
        return 0.0

    pool = ranking[:k] if k else ranking
    hits = 0
    precision_sum = 0.0
    for i, d in enumerate(pool, start=1):
        if qrels.get(d, 0) >= rel_threshold:
            hits += 1
            precision_sum += hits / i
    return precision_sum / total_relevant


# ──────────────────────────────────────────────────────────────────────────────
# Graded measures
# ──────────────────────────────────────────────────────────────────────────────

def dcg_at_k(ranking: list[str], qrels: dict[str, int], k: int,
             gain: str = 'exponential') -> float:
    """
    Discounted Cumulative Gain: graded relevance, discounted logarithmically
    by rank so a good document found late is worth less.

        exponential:  DCG@k = sum over i=1..k of (2^rel_i - 1) / log2(i + 1)
        linear:       DCG@k = sum over i=1..k of  rel_i        / log2(i + 1)

    `exponential` is the modern default (Burges et al. 2005) and sharply
    rewards highly relevant documents. `linear` is the original formulation of
    Jarvelin & Kekalainen (2002). The choice changes the numbers, so it must be
    stated whenever nDCG is reported.
    """
    if gain not in ('exponential', 'linear'):
        raise ValueError(f"gain must be 'exponential' or 'linear', got {gain!r}")

    total = 0.0
    for i, d in enumerate(ranking[:k], start=1):
        rel = qrels.get(d, 0)
        numerator = (2 ** rel - 1) if gain == 'exponential' else rel
        total += numerator / math.log2(i + 1)
    return total


def ndcg_at_k(ranking: list[str], qrels: dict[str, int], k: int,
              gain: str = 'exponential') -> float:
    """
    DCG@k normalised by the best achievable DCG@k for this query.

        nDCG@k = DCG@k / IDCG@k

    IDCG comes from the ideal ranking: all judged documents sorted by
    descending grade. Normalising makes the measure comparable across queries
    with different numbers of relevant documents, the property that precision
    and raw DCG both lack.
    """
    actual = dcg_at_k(ranking, qrels, k, gain)

    ideal_grades = sorted((g for g in qrels.values() if g > 0), reverse=True)
    ideal = 0.0
    for i, rel in enumerate(ideal_grades[:k], start=1):
        numerator = (2 ** rel - 1) if gain == 'exponential' else rel
        ideal += numerator / math.log2(i + 1)

    if ideal == 0.0:
        return 0.0
    return actual / ideal


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation over a query set
# ──────────────────────────────────────────────────────────────────────────────

#: Default cutoffs reported in the experiment tables.
DEFAULT_CUTOFFS = (1, 3, 5, 10)


def evaluate_query(ranking: list[str], qrels: dict[str, int],
                   cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
                   rel_threshold: int = 1,
                   gain: str = 'exponential') -> dict[str, float]:
    """
    Compute the full measure suite for a single query.

    Returns a flat dict, e.g. {'P@5': 0.4, 'nDCG@10': 0.66, 'MRR': 0.5, ...}.
    Per-query dicts are what `stats.py` needs for significance testing, so
    always retain them rather than only their means.
    """
    out: dict[str, float] = {}
    for k in cutoffs:
        out[f'P@{k}'] = precision_at_k(ranking, qrels, k, rel_threshold)
        out[f'R@{k}'] = recall_at_k(ranking, qrels, k, rel_threshold)
        out[f'nDCG@{k}'] = ndcg_at_k(ranking, qrels, k, gain)
        out[f'Success@{k}'] = success_at_k(ranking, qrels, k, rel_threshold)
    out['MRR'] = reciprocal_rank(ranking, qrels, rel_threshold=rel_threshold)
    out['AP'] = average_precision(ranking, qrels, rel_threshold=rel_threshold)
    return out


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]],
                 cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
                 rel_threshold: int = 1,
                 gain: str = 'exponential',
                 skip_unjudged_queries: bool = True
                 ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Evaluate a whole run.

    Args:
        run:   query_id -> ranked list of document ids.
        qrels: query_id -> {document id -> grade}.

    Returns:
        (means, per_query). `means` maps measure name -> mean over queries;
        `per_query` maps query_id -> that query's measure dict.

    Queries with no relevant document in the judgements are skipped by
    default. Keeping them would drag every system's mean toward zero by an
    identical amount, flattening real differences for no benefit.

    MAP and MRR are the means of AP and RR, and appear in `means` under those
    conventional names.
    """
    per_query: dict[str, dict[str, float]] = {}

    for qid, ranking in run.items():
        q = qrels.get(qid, {})
        if skip_unjudged_queries and not any(g >= rel_threshold for g in q.values()):
            continue
        per_query[qid] = evaluate_query(ranking, q, cutoffs, rel_threshold, gain)

    if not per_query:
        return {}, {}

    measures = list(next(iter(per_query.values())).keys())
    means = {m: sum(v[m] for v in per_query.values()) / len(per_query)
             for m in measures}

    # Conventional names for the averaged forms.
    means['MAP'] = means.pop('AP')
    return means, per_query
