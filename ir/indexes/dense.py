"""
Dense retrieval over sentence embeddings.

Where BM25 matches tokens, this matches meaning: query and passage are mapped
into one vector space and compared by cosine similarity, so "why does a model
do badly on new data" can reach a passage that says "poor generalization on
unseen samples" without sharing a single content word. That is the half of the
hybrid BM25 cannot do -- and the converse also holds, which is why neither is
used alone here.

Embedding is pluggable. Two backends:

  SentenceTransformerEmbedder   all-MiniLM-L6-v2, 384-d, runs in-process on
                                CPU. Measured on this machine: 55 passages/s,
                                so a 500-passage corpus embeds in ~9s, once.

  OllamaEmbedder                nomic-embed-text, 768-d, served by the Ollama
                                daemon already running for the comic pipeline.
                                Retrieval-tuned and generally stronger than
                                MiniLM, at the cost of contending for VRAM.

Both default to CPU/local and are interchangeable, which makes "does a
retrieval-tuned 768-d embedder beat a general-purpose 384-d one?" a row in the
results table rather than an architectural commitment.

Embeddings are cached to disk keyed by (model identity + corpus content), so
re-running an experiment never re-embeds, and changing the corpus
automatically invalidates the cache rather than silently reusing stale vectors.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

from .base import Retriever

#: Where embedding caches live.
CACHE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'embeddings'


# ──────────────────────────────────────────────────────────────────────────────
# Embedder interface
# ──────────────────────────────────────────────────────────────────────────────

class Embedder(ABC):
    """
    Maps text to vectors.

    Documents and queries get separate methods because asymmetric models
    require different prefixes for each -- nomic-embed-text wants
    "search_document: " on passages and "search_query: " on queries, and
    omitting them measurably degrades retrieval. A single `encode` would hide
    that distinction and quietly cost accuracy.
    """

    #: Stable identifier; part of the embedding cache key.
    model_id: str = 'embedder'

    #: Vector dimensionality, for reporting.
    dimensions: int = 0

    @abstractmethod
    def encode_documents(self, texts: list[str]):
        """Return an (n, d) float32 array of L2-normalised row vectors."""
        raise NotImplementedError

    @abstractmethod
    def encode_query(self, text: str):
        """Return a (d,) float32 L2-normalised vector."""
        raise NotImplementedError


def _l2_normalise(matrix):
    """
    Scale each row to unit length so a dot product equals the cosine.

    Zero rows are left alone rather than producing NaN: an empty or
    unencodable passage should score 0 against everything, not poison the
    whole similarity matrix.
    """
    import numpy as np

    matrix = np.asarray(matrix, dtype='float32')
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix if norm == 0 else (matrix / norm).astype('float32')

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype('float32')


class SentenceTransformerEmbedder(Embedder):
    """
    sentence-transformers, in-process.

    Defaults to CPU deliberately. This machine has 8 GB of VRAM already shared
    between Ollama and ComfyUI; adding a third tenant causes paging rather than
    speed, and at 55 passages/s the CPU path is not the bottleneck anyway.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2',
                 device: str = 'cpu', batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model_id = f"st:{model_name}"
        self._model = None

    def _load(self):
        """Load lazily, so constructing an index costs nothing."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dimensions = self._model.get_sentence_embedding_dimension()
        return self._model

    def encode_documents(self, texts: list[str]):
        import numpy as np
        if not texts:
            return np.zeros((0, 0), dtype='float32')
        vectors = self._load().encode(
            texts, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True)
        return _l2_normalise(vectors)

    def encode_query(self, text: str):
        vectors = self._load().encode([text], show_progress_bar=False,
                                      convert_to_numpy=True)
        return _l2_normalise(vectors[0])


class OllamaEmbedder(Embedder):
    """
    Embeddings from the local Ollama daemon.

    nomic-embed-text is asymmetric and expects task prefixes; they are applied
    here rather than left to the caller so no call site can forget them.
    """

    def __init__(self, model_name: str = 'nomic-embed-text',
                 base_url: str = 'http://127.0.0.1:11434',
                 document_prefix: str = 'search_document: ',
                 query_prefix: str = 'search_query: ',
                 batch_size: int = 16, timeout: int = 120):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.timeout = timeout
        self.model_id = f"ollama:{model_name}"

    def _embed(self, inputs: list[str]):
        """
        Call Ollama's embedding endpoint.

        Prefers the batch /api/embed endpoint and falls back to the older
        single-input /api/embeddings, so this works across Ollama versions
        without the caller caring which is available.
        """
        import numpy as np
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={'model': self.model_name, 'input': inputs},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                payload = response.json()
                if 'embeddings' in payload:
                    return np.asarray(payload['embeddings'], dtype='float32')
        except requests.RequestException:
            pass

        vectors = []
        for text in inputs:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={'model': self.model_name, 'prompt': text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            vectors.append(response.json()['embedding'])
        return np.asarray(vectors, dtype='float32')

    def encode_documents(self, texts: list[str]):
        import numpy as np
        if not texts:
            return np.zeros((0, 0), dtype='float32')

        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            chunks.append(self._embed([self.document_prefix + t for t in batch]))
        vectors = np.vstack(chunks)
        self.dimensions = int(vectors.shape[1])
        return _l2_normalise(vectors)

    def encode_query(self, text: str):
        vectors = self._embed([self.query_prefix + text])
        return _l2_normalise(vectors[0])


# ──────────────────────────────────────────────────────────────────────────────
# Dense index
# ──────────────────────────────────────────────────────────────────────────────

class DenseIndex(Retriever):
    """
    Brute-force exact nearest-neighbour search over normalised embeddings.

    Exact, not approximate: at a few thousand passages a full matrix product
    takes under a millisecond, so an ANN structure such as HNSW would add
    recall loss and tuning surface for no measurable gain. That trade flips
    somewhere around a million vectors, far above anything this collection
    will reach.

    Attributes:
        doc_ids:  passage ids aligned with matrix rows.
        matrix:   (n, d) L2-normalised float32 embeddings.
    """

    name = 'dense'

    def __init__(self, embedder: Embedder | None = None,
                 cache_dir: Path | None = None, use_cache: bool = True):
        """
        Args:
            embedder:  defaults to SentenceTransformerEmbedder on CPU.
            cache_dir: where to persist embeddings; defaults to CACHE_DIR.
            use_cache: set False to force recomputation.
        """
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.use_cache = use_cache

        self.doc_ids: list[str] = []
        self.matrix = None

    # ── caching ───────────────────────────────────────────────────────────
    def _cache_key(self, documents: dict[str, str]) -> str:
        """
        Fingerprint the model plus the exact corpus content.

        Hashing the text, not just the ids, means re-chunking or editing a
        passage invalidates the cache automatically. Reusing stale vectors
        against a changed corpus would silently corrupt every measurement
        downstream, and that failure is invisible -- so it is prevented here
        rather than documented as a caveat.
        """
        digest = hashlib.sha256()
        digest.update(self.embedder.model_id.encode('utf-8'))
        for doc_id in sorted(documents):
            digest.update(doc_id.encode('utf-8'))
            digest.update(documents[doc_id].encode('utf-8'))
        return digest.hexdigest()[:16]

    def _cache_paths(self, key: str) -> tuple[Path, Path]:
        return (self.cache_dir / f"{key}.npy",
                self.cache_dir / f"{key}.ids.json")

    def _load_cache(self, key: str) -> bool:
        import numpy as np
        vectors_path, ids_path = self._cache_paths(key)
        if not (vectors_path.exists() and ids_path.exists()):
            return False
        try:
            self.matrix = np.load(vectors_path)
            self.doc_ids = json.loads(ids_path.read_text(encoding='utf-8'))
            return len(self.doc_ids) == self.matrix.shape[0]
        except Exception:
            # A corrupt or truncated cache must not be fatal; recompute.
            self.matrix, self.doc_ids = None, []
            return False

    def _save_cache(self, key: str) -> None:
        import numpy as np
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        vectors_path, ids_path = self._cache_paths(key)
        np.save(vectors_path, self.matrix)
        ids_path.write_text(json.dumps(self.doc_ids), encoding='utf-8')

    # ── construction ──────────────────────────────────────────────────────
    def fit(self, documents: dict[str, str]) -> 'DenseIndex':
        """Embed every passage, using the disk cache when it is valid."""
        import numpy as np

        if not documents:
            self.doc_ids, self.matrix = [], None
            return self

        key = self._cache_key(documents)
        if self.use_cache and self._load_cache(key):
            return self

        self.doc_ids = list(documents.keys())
        self.matrix = self.embedder.encode_documents(
            [documents[d] for d in self.doc_ids])
        self.matrix = np.asarray(self.matrix, dtype='float32')

        if self.use_cache:
            self._save_cache(key)
        return self

    # ── scoring ───────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Cosine similarity against every passage.

        Both sides are L2-normalised, so the matrix-vector product is the
        cosine directly. Unlike the sparse indexes, every passage receives a
        score -- a dense retriever always has an opinion about everything,
        which is exactly why it can rank a loosely related passage above an
        exact term match, and why the reranker stage exists.

        Negative similarities are dropped: for sentence embeddings they
        indicate opposition rather than weak relevance, and returning them
        would only pad the result list.
        """
        import numpy as np

        if self.matrix is None or not self.doc_ids:
            return []
        if not query.strip():
            return []

        query_vector = np.asarray(self.embedder.encode_query(query),
                                  dtype='float32')
        scores = self.matrix @ query_vector

        count = min(top_k, len(self.doc_ids))
        # argpartition finds the top-k without sorting all n, then we sort
        # just those k.
        candidates = np.argpartition(-scores, count - 1)[:count]
        ranked = sorted(
            ((self.doc_ids[i], float(scores[i])) for i in candidates),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return [(d, s) for d, s in ranked if s > 0.0]

    def similarity(self, query: str, doc_id: str) -> float:
        """
        Cosine similarity for one passage, for the explanation panel.

        There is no per-term breakdown to offer: a dense score is a single
        inner product in a learned space and does not decompose into
        attributable terms. That asymmetry with BM25 is worth stating plainly
        in the UI rather than papering over -- it is the honest reason
        lexical matching remains useful for explainability.
        """
        import numpy as np

        if self.matrix is None or doc_id not in self.doc_ids:
            return 0.0
        row = self.doc_ids.index(doc_id)
        query_vector = np.asarray(self.embedder.encode_query(query),
                                  dtype='float32')
        return float(self.matrix[row] @ query_vector)

    @property
    def dimensions(self) -> int:
        if self.matrix is None:
            return 0
        return int(self.matrix.shape[1])

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __repr__(self) -> str:
        return (f"DenseIndex(docs={len(self.doc_ids)}, "
                f"dims={self.dimensions}, model={self.embedder.model_id!r})")
