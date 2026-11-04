"""
Fusion of several ranked result lists into one.

The central problem: BM25 and cosine similarity are not comparable. BM25 is
unbounded above and depends on corpus statistics; cosine similarity sits in
[-1, 1]. Adding them directly lets whichever has the larger variance dominate
the result, and the mix changes from query to query. So naive score addition is
not a baseline worth shipping -- it is the mistake the alternatives exist to
avoid.

Two families are provided.

Rank-based (preferred)
    Reciprocal Rank Fusion ignores scores entirely and uses only positions,
    which makes it scale-free by construction. Nothing needs normalising
    because nothing is compared across systems except rank.

        RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))

    k damps the influence of top ranks; k = 60 is the value from the original
    paper and the de facto standard. A document missing from a retriever's
    list simply contributes nothing for that retriever.

    Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods", SIGIR.

Score-based (for comparison)
    Normalise each retriever's scores to a common range, then combine. Works,
    but the normalisation is query-dependent: min-max over a result list means
    the top document always scores 1.0 whether it was excellent or mediocre.
    Included so "RRF vs normalised score fusion" is a measured row in the
    results table rather than an assertion.

        CombSUM(d)  = sum of normalised scores
        CombMNZ(d)  = CombSUM(d) * (number of retrievers that returned d)

    Fox & Shaw (1994), "Combination of Multiple Searches", TREC-2. CombMNZ
    multiplies by the hit count, so agreement between retrievers is rewarded
    explicitly rather than implicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: The k constant from Cormack et al. (2009). Larger k flattens the weight
#: given to top ranks; smaller k makes rank 1 dominate.
RRF_K = 60

#: Supported score normalisation schemes for the score-based family.
NORMALISATIONS = ('minmax', 'zscore', 'none')

#: Supported fusion methods.
FUSION_METHODS = ('rrf', 'combsum', 'combmnz')


@dataclass
class Contribution:
    """What one retriever contributed to one document's fused score."""
    retriever: str
    rank: int           # 1-based rank in that retriever's list
    raw_score: float
    normalised_score: float
    contribution: float  # the term this retriever added to the fused score


@dataclass
class FusedResult:
    """
    One document in the fused ranking, carrying its provenance.

    `contributions` is what lets the UI answer "this was ranked 2nd because
    BM25 put it 1st and the dense retriever put it 7th" instead of only
    reporting a final number.
    """
    doc_id: str
    score: float
    rank: int = 0
    contributions: list[Contribution] = field(default_factory=list)

    @property
    def n_retrievers(self) -> int:
        """How many retrievers returned this document at all."""
        return len(self.contributions)

    def contribution_shares(self) -> dict[str, float]:
        """
        Each retriever's share of the fused score, summing to 1.0.

        This is the number behind the "Lexical 35% / Semantic 40%" breakdown
        in the retrieval-explanation panel.
        """
        total = sum(c.contribution for c in self.contributions)
        if total == 0:
            return {c.retriever: 0.0 for c in self.contributions}
        return {c.retriever: c.contribution / total for c in self.contributions}


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalise(scores: list[float], method: str = 'minmax') -> list[float]:
    """
    Rescale one retriever's scores.

    'minmax'  -> (s - min) / (max - min), mapping the list onto [0, 1].
                 Note the distortion this introduces: the best document in
                 every list becomes exactly 1.0, so a retriever that found
                 nothing good looks as confident as one that found a perfect
                 match. This is the main argument for preferring RRF.
    'zscore'  -> (s - mean) / stdev. Preserves relative spacing, so a clear
                 winner stays a clear winner, but the output is unbounded.
    'none'    -> returned unchanged, for ablating normalisation away.

    Degenerate inputs (empty, or every score identical) return 0.5 for each
    element rather than dividing by zero: with no spread there is no evidence
    to rank on, so every document gets the same neutral weight.
    """
    if method not in NORMALISATIONS:
        raise ValueError(f"method must be one of {NORMALISATIONS}, got {method!r}")
    if not scores:
        return []
    if method == 'none':
        return list(scores)

    if method == 'minmax':
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [0.5] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    stdev = variance ** 0.5
    if stdev == 0:
        return [0.5] * len(scores)
    return [(s - mean) / stdev for s in scores]


# ──────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    runs: dict[str, list[tuple[str, float]]],
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedResult]:
    """
    Fuse ranked lists by reciprocal rank.

    Args:
        runs:    retriever name -> [(doc_id, score), ...] in rank order. The
                 scores are carried through for the trace but play no part in
                 the fused value, which is the point.
        k:       RRF damping constant.
        weights: optional per-retriever multiplier, defaulting to 1.0. Use this
                 to lean on the lexical side for acronym-heavy queries, which
                 is how intent-aware routing is expressed at this layer.
        top_k:   truncate the fused ranking.

    Returns:
        FusedResult list, descending by score, ties broken by doc_id so the
        output is deterministic across runs.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    weights = weights or {}
    accumulated: dict[str, float] = {}
    contributions: dict[str, list[Contribution]] = {}

    for retriever, ranked in runs.items():
        weight = weights.get(retriever, 1.0)
        for position, (doc_id, raw_score) in enumerate(ranked, start=1):
            term = weight / (k + position)
            accumulated[doc_id] = accumulated.get(doc_id, 0.0) + term
            contributions.setdefault(doc_id, []).append(
                Contribution(
                    retriever=retriever,
                    rank=position,
                    raw_score=raw_score,
                    normalised_score=1.0 / (k + position),
                    contribution=term,
                )
            )

    results = [
        FusedResult(doc_id=doc_id, score=score,
                    contributions=contributions[doc_id])
        for doc_id, score in accumulated.items()
    ]
    results.sort(key=lambda r: (-r.score, r.doc_id))
    for i, r in enumerate(results, start=1):
        r.rank = i
    return results[:top_k] if top_k else results


# ──────────────────────────────────────────────────────────────────────────────
# Score-based fusion
# ──────────────────────────────────────────────────────────────────────────────

def score_fusion(
    runs: dict[str, list[tuple[str, float]]],
    method: str = 'combsum',
    normalisation: str = 'minmax',
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedResult]:
    """
    Fuse ranked lists by combining normalised scores.

    Args:
        runs:          retriever name -> [(doc_id, score), ...].
        method:        'combsum' sums normalised scores; 'combmnz' multiplies
                       that sum by the number of retrievers that returned the
                       document, rewarding cross-retriever agreement.
        normalisation: see `normalise`.
        weights:       optional per-retriever multiplier.
        top_k:         truncate the fused ranking.

    A document absent from a retriever's list contributes 0 from that
    retriever -- the conventional treatment, though note it is an assumption:
    absence means "not in the top-n returned", not "scored zero".
    """
    if method not in ('combsum', 'combmnz'):
        raise ValueError(f"method must be 'combsum' or 'combmnz', got {method!r}")

    weights = weights or {}
    accumulated: dict[str, float] = {}
    contributions: dict[str, list[Contribution]] = {}

    for retriever, ranked in runs.items():
        if not ranked:
            continue
        weight = weights.get(retriever, 1.0)
        raw = [s for _, s in ranked]
        normalised = normalise(raw, normalisation)

        for position, ((doc_id, raw_score), norm_score) in enumerate(
                zip(ranked, normalised), start=1):
            term = weight * norm_score
            accumulated[doc_id] = accumulated.get(doc_id, 0.0) + term
            contributions.setdefault(doc_id, []).append(
                Contribution(
                    retriever=retriever,
                    rank=position,
                    raw_score=raw_score,
                    normalised_score=norm_score,
                    contribution=term,
                )
            )

    results = []
    for doc_id, score in accumulated.items():
        contribs = contributions[doc_id]
        if method == 'combmnz':
            score = score * len(contribs)
        results.append(FusedResult(doc_id=doc_id, score=score,
                                   contributions=contribs))

    results.sort(key=lambda r: (-r.score, r.doc_id))
    for i, r in enumerate(results, start=1):
        r.rank = i
    return results[:top_k] if top_k else results


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def fuse(
    runs: dict[str, list[tuple[str, float]]],
    method: str = 'rrf',
    k: int = RRF_K,
    normalisation: str = 'minmax',
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedResult]:
    """
    Single entry point, so a pipeline can name its fusion strategy in config.

    `method` is one of FUSION_METHODS. 'rrf' ignores `normalisation`; the
    score-based methods ignore `k`.

    A single input run is passed through in its original order, with
    contributions still recorded -- so a BM25-only pipeline and a hybrid
    pipeline produce the same result shape and the explanation UI needs no
    special case.
    """
    if method not in FUSION_METHODS:
        raise ValueError(f"method must be one of {FUSION_METHODS}, got {method!r}")
    if method == 'rrf':
        return reciprocal_rank_fusion(runs, k=k, weights=weights, top_k=top_k)
    return score_fusion(runs, method=method, normalisation=normalisation,
                        weights=weights, top_k=top_k)
