"""
BM25 over an explicit inverted index.

Implemented directly rather than taken from a library so the three mechanisms
that make BM25 work are visible and tunable:

  1. Term frequency saturation (k1)
     Raw tf is a bad signal: a passage containing "capacitor" 20 times is not
     20x more about capacitors than one containing it once. BM25 feeds tf
     through tf / (tf + k1 * ...), which rises steeply at first and then
     flattens, approaching an asymptote of (k1 + 1). Lower k1 saturates
     sooner. k1 = 0 collapses the term to a pure binary presence test.

  2. Document length normalisation (b)
     A long passage accumulates term occurrences by being long, not by being
     relevant. The factor (1 - b + b * |d| / avgdl) divides out that
     advantage. b = 1 normalises fully by length; b = 0 disables it entirely.

  3. Inverse document frequency
     A term occurring in most passages discriminates poorly. Two standard
     forms are provided; see `IDF_VARIANTS`.

Scoring:

    score(q, d) = sum over t in q of
                      idf(t) * ( tf(t,d) * (k1 + 1) )
                             / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )

The inverted index is what makes this cheap: only documents that actually
contain a query term are ever touched, so cost scales with postings-list
length rather than with collection size.

Reference
---------
Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and
Beyond", Foundations and Trends in Information Retrieval 3(4).
"""
from __future__ import annotations

import math
from collections import defaultdict

from .base import Retriever
from ..text import tokenize

#: Supported IDF formulations.
#:
#: 'lucene' : ln(1 + (N - df + 0.5) / (df + 0.5))
#:     Always positive, because the +1 inside the log keeps the argument above
#:     1. This is what Lucene and Elasticsearch use, and the sensible default.
#:
#: 'okapi'  : ln(N - df + 0.5) - ln(df + 0.5)
#:     The original Robertson-Sparck Jones form. Goes *negative* once a term
#:     appears in more than about half the collection, which lets a very
#:     common term subtract from a document's score. Implementations patch this
#:     by flooring negative values at `epsilon * mean_idf`. Provided mainly so
#:     the implementation can be verified against rank_bm25, which uses this
#:     form -- see tests/test_bm25.py.
IDF_VARIANTS = ('lucene', 'okapi')


class BM25Index(Retriever):
    """
    An in-memory BM25 index.

    Attributes:
        doc_ids:  external document ids, positionally aligned with internal
                  integer indices.
        postings: term -> list of (doc_index, term_frequency).
        doc_len:  internal doc index -> token count.
        df:       term -> number of documents containing it.
        avgdl:    mean document length in tokens.
    """

    name = 'bm25'

    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 idf_variant: str = 'lucene', epsilon: float = 0.25,
                 remove_stopwords: bool = True, stem: bool = True):
        """
        Args:
            k1:          term-frequency saturation. Typical range 1.2-2.0.
            b:           length normalisation, 0 to 1. Typical 0.75.
            idf_variant: one of IDF_VARIANTS.
            epsilon:     floor multiplier for negative IDFs, 'okapi' only.
            remove_stopwords, stem: passed to `ir.text.tokenize`.
        """
        if idf_variant not in IDF_VARIANTS:
            raise ValueError(
                f"idf_variant must be one of {IDF_VARIANTS}, got {idf_variant!r}")
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")

        self.k1 = k1
        self.b = b
        self.idf_variant = idf_variant
        self.epsilon = epsilon
        self.remove_stopwords = remove_stopwords
        self.stem = stem

        self.doc_ids: list[str] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: list[int] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, documents: dict[str, str]) -> 'BM25Index':
        """
        Build the index from {document id -> text}.

        Returns self so construction can be chained.
        """
        tokenized = {
            doc_id: tokenize(text, self.remove_stopwords, self.stem)
            for doc_id, text in documents.items()
        }
        return self.fit_tokenized(tokenized)

    def fit_tokenized(self, tokenized: dict[str, list[str]]) -> 'BM25Index':
        """
        Build the index from pre-tokenised documents.

        Separate from `fit` so an experiment can hold the token stream fixed
        while varying k1 and b, and so the equivalence test can feed the exact
        token lists it also gives to rank_bm25.
        """
        self.doc_ids = list(tokenized.keys())
        self.doc_len = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for idx, doc_id in enumerate(self.doc_ids):
            tokens = tokenized[doc_id]
            self.doc_len.append(len(tokens))

            counts: dict[str, int] = defaultdict(int)
            for t in tokens:
                counts[t] += 1
            for term, tf in counts.items():
                postings[term].append((idx, tf))

        self.postings = dict(postings)
        self.df = {term: len(plist) for term, plist in self.postings.items()}
        n_docs = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / n_docs) if n_docs else 0.0
        self._compute_idf()
        return self

    def _compute_idf(self) -> None:
        """Precompute IDF for every indexed term."""
        n = len(self.doc_ids)
        self.idf = {}
        if n == 0:
            return

        if self.idf_variant == 'lucene':
            for term, df in self.df.items():
                self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            return

        # 'okapi': compute raw values, then floor the negative ones at
        # epsilon * mean_idf, matching the conventional patch for this form.
        negative: list[str] = []
        total = 0.0
        for term, df in self.df.items():
            value = math.log(n - df + 0.5) - math.log(df + 0.5)
            self.idf[term] = value
            total += value
            if value < 0:
                negative.append(term)

        if self.idf:
            floor = self.epsilon * (total / len(self.idf))
            for term in negative:
                self.idf[term] = floor

    # ──────────────────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────────────────

    def _length_norm(self, doc_index: int) -> float:
        """The k1 * (1 - b + b * |d| / avgdl) denominator component."""
        if self.avgdl == 0:
            return self.k1
        return self.k1 * (1.0 - self.b
                          + self.b * self.doc_len[doc_index] / self.avgdl)

    def score_terms(self, query_terms: list[str]) -> dict[int, float]:
        """
        Score every document containing at least one query term.

        Query terms are NOT deduplicated: a term repeated in the query
        contributes once per occurrence, which is the conventional behaviour
        and matters for expanded queries where a term may be added back.

        Returns {internal doc index -> score}. Documents sharing no term with
        the query are absent rather than scored 0, which is the whole point of
        the inverted index.
        """
        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_index, tf in plist:
                denom = tf + self._length_norm(doc_index)
                if denom == 0:
                    continue
                scores[doc_index] += idf * (tf * (self.k1 + 1.0)) / denom
        return dict(scores)

    def score_weighted_terms(self, weights: dict[str, float]) -> dict[int, float]:
        """
        Score documents against a weighted query.

        Generalises `score_terms`: each term's BM25 contribution is multiplied
        by its weight, so

            score(q, d) = sum over t of  w_t * idf(t) * tf*(k1+1) / (tf + norm)

        `score_terms(['a', 'a', 'b'])` is the special case where weights are
        raw query term counts. Weighted queries are what query expansion
        produces -- an expansion term should not count as much as a term the
        student actually typed, and only a weight can express that.

        Negative weights are allowed and subtract, which is how a Rocchio-style
        negative feedback term would be applied.
        """
        scores: dict[int, float] = defaultdict(float)
        for term, weight in weights.items():
            if weight == 0:
                continue
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_index, tf in plist:
                denom = tf + self._length_norm(doc_index)
                if denom == 0:
                    continue
                scores[doc_index] += weight * idf * (tf * (self.k1 + 1.0)) / denom
        return dict(scores)

    def search_weighted(self, weights: dict[str, float],
                        top_k: int = 10) -> list[tuple[str, float]]:
        """`search_terms` for a weighted query. Ties break by passage id."""
        scored = self.score_weighted_terms(weights)
        ranked = sorted(scored.items(),
                        key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))
        return [(self.doc_ids[i], s) for i, s in ranked[:top_k]]

    def document_terms(self, doc_id: str) -> dict[str, int]:
        """
        Term frequencies for one document.

        Reconstructed from the postings lists rather than kept as a second
        forward index, because the only caller is pseudo-relevance feedback,
        which needs this for about ten documents per query. Paying O(vocab)
        there is cheaper than doubling the index's memory for every document.
        """
        if doc_id not in self.doc_ids:
            raise KeyError(f"unknown document id {doc_id!r}")
        target = self.doc_ids.index(doc_id)
        return {term: tf
                for term, plist in self.postings.items()
                for index, tf in plist if index == target}

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Retrieve for a raw query string.

        Returns [(document id, score), ...] sorted by descending score, then
        by document id so ties break deterministically -- otherwise two runs
        of the same experiment could report different rankings.
        """
        terms = tokenize(query, self.remove_stopwords, self.stem)
        return self.search_terms(terms, top_k)

    def search_terms(self, query_terms: list[str],
                     top_k: int = 10) -> list[tuple[str, float]]:
        """`search` for an already-tokenised (or expanded) query."""
        scored = self.score_terms(query_terms)
        ranked = sorted(scored.items(),
                        key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))
        return [(self.doc_ids[i], s) for i, s in ranked[:top_k]]

    # ──────────────────────────────────────────────────────────────────────
    # Introspection, used by the retrieval-explanation UI
    # ──────────────────────────────────────────────────────────────────────

    def explain(self, query: str, doc_id: str) -> dict:
        """
        Per-term breakdown of one document's score.

        Returns {'doc_id', 'total', 'doc_length', 'avgdl', 'k1', 'b',
        'terms': [{'term', 'tf', 'df', 'idf', 'contribution'}, ...]} with
        terms ordered by descending contribution.

        This is what lets the UI answer "why was this retrieved?" with real
        numbers rather than a guess, and it is also the quickest way to sanity
        check a suspicious ranking.
        """
        if doc_id not in self.doc_ids:
            raise KeyError(f"unknown document id {doc_id!r}")
        doc_index = self.doc_ids.index(doc_id)
        terms = tokenize(query, self.remove_stopwords, self.stem)

        rows = []
        total = 0.0
        for term in terms:
            plist = self.postings.get(term, [])
            tf = next((f for i, f in plist if i == doc_index), 0)
            idf = self.idf.get(term, 0.0)
            if tf:
                contribution = idf * (tf * (self.k1 + 1.0)) / (
                    tf + self._length_norm(doc_index))
            else:
                contribution = 0.0
            total += contribution
            rows.append({
                'term': term,
                'tf': tf,
                'df': self.df.get(term, 0),
                'idf': idf,
                'contribution': contribution,
            })

        rows.sort(key=lambda r: -r['contribution'])
        return {
            'doc_id': doc_id,
            'total': total,
            'doc_length': self.doc_len[doc_index],
            'avgdl': self.avgdl,
            'k1': self.k1,
            'b': self.b,
            'terms': rows,
        }

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __repr__(self) -> str:
        return (f"BM25Index(docs={len(self.doc_ids)}, "
                f"vocab={self.vocabulary_size}, k1={self.k1}, b={self.b}, "
                f"idf={self.idf_variant!r})")
