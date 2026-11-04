"""
TF-IDF retrieval over the vector space model.

The classical baseline, and the one BM25 is an improvement on. Including it is
not box-ticking: the gap between TF-IDF and BM25 isolates exactly what BM25's
two extra mechanisms buy, because both see the identical token stream from
`ir.text.tokenize`.

How this differs from BM25, which is the comparison worth making explicit in a
write-up:

  Term frequency
      TF-IDF    weight grows with log(1 + tf) or raw tf -- unbounded.
      BM25      tf / (tf + k1 * ...) saturates toward an asymptote, so the
                20th occurrence of a word adds almost nothing.

  Document length
      TF-IDF    L2-normalises the whole document vector. Every document
                becomes a unit vector, so length is divided out uniformly.
      BM25      pivoted normalisation via b, tunable, and applied per term
                rather than to the vector as a whole.

  IDF
      TF-IDF    (scikit-learn, smooth_idf) ln((1 + N) / (1 + df)) + 1 --
                always positive, the +1 guaranteeing a term never loses all
                weight.
      BM25      ln(1 + (N - df + 0.5) / (df + 0.5)) in the lucene form.

  Scoring
      TF-IDF    cosine similarity between query and document vectors, which
                is symmetric and bounded in [0, 1] for non-negative weights.
      BM25      an asymmetric sum over query terms, unbounded above.

Backed by scikit-learn's TfidfVectorizer, with tokenisation overridden so the
comparison stays controlled.
"""
from __future__ import annotations

from .base import Retriever
from ..text import tokenize


class TFIDFIndex(Retriever):
    """
    TF-IDF index with cosine similarity scoring.

    Attributes:
        doc_ids:   passage ids, aligned with the matrix rows.
        matrix:    sparse document-term matrix, L2-normalised rows.
        vectorizer: the fitted scikit-learn vectorizer.
    """

    name = 'tfidf'

    def __init__(self, sublinear_tf: bool = True, remove_stopwords: bool = True,
                 stem: bool = True, min_df: int = 1):
        """
        Args:
            sublinear_tf:     use 1 + log(tf) instead of raw tf. On by default
                              because raw tf badly over-weights repetition --
                              a softer version of the problem BM25 solves with
                              k1. Turn it off to show that effect.
            remove_stopwords, stem: forwarded to `ir.text.tokenize`, and must
                              match the BM25 index's settings for the
                              comparison to be meaningful.
            min_df:           drop terms appearing in fewer than this many
                              documents. 1 keeps everything, which is right
                              for a small collection where a term occurring
                              once may be the only answer to some query.
        """
        self.sublinear_tf = sublinear_tf
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.min_df = min_df

        self.doc_ids: list[str] = []
        self.matrix = None
        self.vectorizer = None

    # ──────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────

    def _analyzer(self, text: str) -> list[str]:
        """
        Hand scikit-learn our own tokens.

        Passed as `analyzer=`, which bypasses its built-in lowercasing,
        pattern matching and stopword handling entirely, so BM25 and TF-IDF
        are guaranteed to index the same terms.
        """
        return tokenize(text, self.remove_stopwords, self.stem)

    def fit(self, documents: dict[str, str]) -> 'TFIDFIndex':
        """Build the index from {passage_id -> text}. Returns self."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.doc_ids = list(documents.keys())
        if not self.doc_ids:
            self.matrix = None
            self.vectorizer = None
            return self

        self.vectorizer = TfidfVectorizer(
            analyzer=self._analyzer,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            norm='l2',        # unit-length rows, so dot product == cosine
            smooth_idf=True,  # ln((1 + N) / (1 + df)) + 1
            use_idf=True,
        )
        self.matrix = self.vectorizer.fit_transform(
            [documents[d] for d in self.doc_ids])
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Cosine similarity between the query vector and every document.

        Because both sides are L2-normalised, the dot product *is* the cosine,
        so no separate division is needed. Documents scoring exactly 0 share no
        term with the query and are dropped rather than returned as padding --
        matching BM25Index, so `top_k` means the same thing for both and
        precision at a fixed cutoff stays comparable.
        """
        if self.matrix is None or not self.doc_ids:
            return []

        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            # No query term survived tokenisation or is in the vocabulary.
            return []

        scores = (self.matrix @ query_vector.T).toarray().ravel()

        ranked = sorted(
            ((self.doc_ids[i], float(s)) for i, s in enumerate(scores) if s > 0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:top_k]

    def explain(self, query: str, doc_id: str) -> dict:
        """
        Per-term breakdown of one document's cosine score.

        Returns {'doc_id', 'total', 'terms': [{'term', 'query_weight',
        'doc_weight', 'contribution'}, ...]} ordered by contribution. Each
        contribution is the product of the two L2-normalised weights, so they
        sum to the cosine similarity.
        """
        if self.matrix is None:
            raise RuntimeError("index is empty; call fit() first")
        if doc_id not in self.doc_ids:
            raise KeyError(f"unknown passage id {doc_id!r}")

        row = self.doc_ids.index(doc_id)
        query_vector = self.vectorizer.transform([query])
        doc_vector = self.matrix[row]
        vocabulary = self.vectorizer.get_feature_names_out()

        doc_weights = dict(zip(doc_vector.indices, doc_vector.data))

        rows = []
        total = 0.0
        for column, query_weight in zip(query_vector.indices, query_vector.data):
            doc_weight = doc_weights.get(column, 0.0)
            contribution = float(query_weight) * float(doc_weight)
            total += contribution
            rows.append({
                'term': str(vocabulary[column]),
                'query_weight': float(query_weight),
                'doc_weight': float(doc_weight),
                'contribution': contribution,
            })

        rows.sort(key=lambda r: -r['contribution'])
        return {'doc_id': doc_id, 'total': total, 'terms': rows}

    @property
    def vocabulary_size(self) -> int:
        if self.vectorizer is None:
            return 0
        return len(self.vectorizer.get_feature_names_out())

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __repr__(self) -> str:
        return (f"TFIDFIndex(docs={len(self.doc_ids)}, "
                f"vocab={self.vocabulary_size}, "
                f"sublinear_tf={self.sublinear_tf})")
