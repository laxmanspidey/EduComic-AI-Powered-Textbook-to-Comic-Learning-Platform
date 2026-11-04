"""
Cross-encoder reranking.

The retrievers in Layer 3 are all bi-encoders or term-matchers: they score a
query and a passage *independently* and then compare the results. BM25 sums
per-term weights; dense retrieval compares two vectors that were computed
without reference to each other. Neither ever looks at a query and a passage
together.

A cross-encoder does. It concatenates them into one input and runs full
attention across both, so it can judge "does this passage answer *this*
question" rather than "are these two things broadly about the same topic".
That is why it fixes the semantic-overranking failure: a passage about
capacitance in parallel and a question about electrostatic shielding are
topically adjacent but not an answer, and only a joint model can tell.

The cost is that it cannot index anything. Scoring is O(candidates) model
calls per query, so it can only ever be a second stage over a small candidate
pool -- which is exactly how it is used here: retrieve broadly and cheaply,
then rerank the head precisely.

Measured on this machine: 52ms per (query, passage) pair on CPU, so a top-20
pool costs about 1.04s per query. CPU is deliberate; the GPU is already shared
between Ollama and ComfyUI.

Scores are raw relevance logits, not probabilities. They are unbounded and
only comparable *within* one query's candidate list -- never across queries,
and never against BM25 or cosine scores. `probability()` applies a sigmoid for
display purposes only.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

#: Default cross-encoder. Small (~22M parameters), trained on MS MARCO
#: passage ranking, and the standard choice for this stage.
DEFAULT_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

#: How many candidates to rerank by default. Beyond ~40 the latency grows
#: faster than the quality does, because fusion rarely buries a good passage
#: that deep.
DEFAULT_TOP_N = 20


@dataclass
class RerankedResult:
    """
    One passage after reranking, carrying how far the reranker moved it.

    `rank_delta` is what makes the reranker's effect visible in the trace and
    in the UI: a large positive delta is the reranker rescuing a passage that
    fusion had buried, which is the whole point of the stage.
    """
    doc_id: str
    score: float
    rank: int
    previous_rank: int
    reranked: bool = True

    @property
    def rank_delta(self) -> int:
        """Positive means the reranker promoted this passage."""
        return self.previous_rank - self.rank

    @property
    def probability(self) -> float:
        """
        Sigmoid of the logit, for display only.

        The model was trained with a ranking objective, not a calibrated
        probabilistic one, so this is a monotone rescaling into (0, 1) and
        not a trustworthy confidence. Useful for a progress bar in the UI,
        not for thresholding.
        """
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, self.score))))


class Reranker(ABC):
    """Reorders a candidate list using the query and passage text together."""

    name: str = 'reranker'

    @abstractmethod
    def rerank(self, query: str, candidates: list[str],
               texts: dict[str, str],
               top_n: int | None = None) -> list[RerankedResult]:
        """
        Args:
            query:      the (possibly rewritten) query string.
            candidates: passage ids in their incoming fused order, best-first.
            texts:      passage id -> text, for the candidates at minimum.
            top_n:      rerank only this many from the head of `candidates`.

        Returns:
            The full list reordered, every element carrying its previous rank.
        """
        raise NotImplementedError


class IdentityReranker(Reranker):
    """
    Passes the candidate order through unchanged.

    Exists so "pipeline without reranking" is a configuration rather than a
    code branch, which keeps the ablation table honest: the no-rerank row runs
    the identical code path with one component swapped.
    """

    name = 'identity'

    def rerank(self, query: str, candidates: list[str],
               texts: dict[str, str],
               top_n: int | None = None) -> list[RerankedResult]:
        return [
            RerankedResult(doc_id=doc_id, score=float(-position),
                           rank=position, previous_rank=position,
                           reranked=False)
            for position, doc_id in enumerate(candidates, start=1)
        ]


class CrossEncoderReranker(Reranker):
    """
    Reranking with a sentence-transformers CrossEncoder.

    Only the first `top_n` candidates are scored. The remainder keep their
    fused order and are appended below the reranked head -- standard practice,
    and worth being explicit about: a tail passage is never given the chance
    to outrank a reranked one, so `top_n` is a hard ceiling on how far the
    reranker can reach. Raise it to let the reranker rescue deeper passages,
    at linear cost.
    """

    name = 'cross_encoder'

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = 'cpu',
                 top_n: int = DEFAULT_TOP_N, batch_size: int = 32,
                 max_length: int = 512):
        """
        Args:
            model_name:  HuggingFace cross-encoder id.
            device:      'cpu' by default, to leave the 8GB of VRAM to Ollama
                         and ComfyUI.
            top_n:       default candidate depth to rerank.
            batch_size:  pairs per forward pass.
            max_length:  token cap per pair. A 400-word passage plus a query
                         fits in 512 tokens; longer passages are truncated
                         from the end, so the start of a passage matters most.
        """
        self.model_name = model_name
        self.device = device
        self.top_n = top_n
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load(self):
        """Load lazily: constructing a pipeline must not cost 30s."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device,
                                       max_length=self.max_length)
        return self._model

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """
        Raw logits for (query, text) pairs, in input order.

        Batched in one call: the measured 52ms/pair already reflects batching,
        and scoring pairs one at a time would be several times slower.
        """
        if not texts:
            return []
        model = self._load()
        scores = model.predict([(query, t) for t in texts],
                               batch_size=self.batch_size,
                               show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(self, query: str, candidates: list[str],
               texts: dict[str, str],
               top_n: int | None = None) -> list[RerankedResult]:
        if not candidates:
            return []

        depth = self.top_n if top_n is None else top_n
        depth = max(0, min(depth, len(candidates)))

        head = candidates[:depth]
        tail = candidates[depth:]

        # A candidate whose text is missing is passed through rather than
        # dropped: losing a passage because of a lookup gap would silently
        # change recall.
        head_texts = [texts.get(doc_id, '') for doc_id in head]
        scores = self.score_pairs(query, head_texts) if head else []

        previous = {doc_id: i for i, doc_id in enumerate(candidates, start=1)}

        scored = sorted(
            zip(head, scores),
            key=lambda pair: (-pair[1], pair[0]),
        )

        results: list[RerankedResult] = []
        for position, (doc_id, score) in enumerate(scored, start=1):
            results.append(RerankedResult(
                doc_id=doc_id, score=score, rank=position,
                previous_rank=previous[doc_id], reranked=True))

        for offset, doc_id in enumerate(tail):
            position = len(scored) + offset + 1
            results.append(RerankedResult(
                doc_id=doc_id, score=float('-inf'), rank=position,
                previous_rank=previous[doc_id], reranked=False))

        return results

    def __repr__(self) -> str:
        return (f"CrossEncoderReranker(model={self.model_name!r}, "
                f"device={self.device!r}, top_n={self.top_n})")
