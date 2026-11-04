"""
Bridge between Django and the `ir` package.

The `ir` package has no Django dependency, so the evaluation harness can run as
a plain script without Django startup and the library stays testable in
isolation. Everything Django-specific lives here: locating a textbook's PDF,
mapping passages onto `Concept` rows, and caching built indexes.

Scoping
-------
One engine per textbook, not one engine over the whole library. This is a
retrieval-quality decision, not just a tidiness one. Measured on this corpus:
the query "what does CAP mean" against a whole-library index put three physics
*capacitor* passages in the top four, because both the embedding model and the
cross-encoder conflate `CAP` with `capacit-`. Scoping the index to the one
textbook being read drops that to zero off-domain results, because the
colliding passages are not in the index at all.

Caching
-------
Building an engine costs a corpus parse (fast) plus embedding every passage
(~55 passages/second on CPU). Both are cached:

  * embeddings       on disk by `DenseIndex`, keyed by passage content
  * built engines    in memory here, keyed by textbook id and invalidated when
                     the textbook's `updated_at` changes

So the first search after an upload pays a few seconds; every later search is
milliseconds.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

from ir.context import ContextAssembler
from ir.corpus import Corpus, build_from_pdfs
from ir.indexes.bm25 import BM25Index
from ir.indexes.dense import DenseIndex, SentenceTransformerEmbedder
from ir.indexes.tfidf import TFIDFIndex
from ir.pipeline import Pipeline
from ir.query.expand import RM3Expander
from ir.query.rewrite import ConversationalRewriter
from ir.rerank import CrossEncoderReranker

#: Built engines, keyed by textbook id. Guarded by `_LOCK` because Django's
#: dev server and the Django-Q worker both touch this from several threads.
_ENGINES: dict[str, 'TextbookEngine'] = {}
_LOCK = threading.Lock()

#: Shared across textbooks: the model weights are identical, and loading
#: sentence-transformers or a cross-encoder repeatedly costs seconds each time
#: (32s for the cross-encoder, measured). Both are CPU-bound and stateless
#: between calls, so one instance serves every textbook.
_EMBEDDER: SentenceTransformerEmbedder | None = None
_RERANKER: CrossEncoderReranker | None = None
_ANSWERER = None


def _embedder() -> SentenceTransformerEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformerEmbedder(device='cpu')
    return _EMBEDDER


def _reranker() -> CrossEncoderReranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoderReranker(device='cpu', top_n=20)
    return _RERANKER


def _answerer():
    """
    Shared answer generator.

    Unlike the embedder and reranker this holds no model weights -- it talks
    to Ollama over HTTP -- but it does own the answer cache, so one instance
    keeps cache behaviour consistent across requests.
    """
    global _ANSWERER
    if _ANSWERER is None:
        from ir.answer import AnswerGenerator

        _ANSWERER = AnswerGenerator(
            model=getattr(settings, 'OLLAMA_MODEL', 'qwen3.5:4b'),
            base_url=getattr(settings, 'OLLAMA_URL', 'http://127.0.0.1:11434'),
        )
    return _ANSWERER


@dataclass
class TextbookEngine:
    """Everything needed to search one textbook."""
    textbook_id: str
    corpus: Corpus
    bm25: BM25Index
    tfidf: TFIDFIndex
    dense: DenseIndex
    assembler: ContextAssembler
    version: str = ''

    def pipeline(self, system: str = 'full', top_k: int = 10,
                 with_rewriter: bool = False) -> Pipeline:
        """
        Build a named pipeline over this textbook's indexes.

        The names match the rows of the results table, so the application and
        the experiments run the identical configurations -- which is what
        makes the measured numbers mean anything about the app.
        """
        documents = self.corpus.documents()

        if system == 'tfidf':
            return Pipeline([self.tfidf], self.corpus, top_k=top_k,
                            assembler=self.assembler, name='tfidf')
        if system == 'bm25':
            return Pipeline([self.bm25], self.corpus, top_k=top_k,
                            assembler=self.assembler, name='bm25')
        if system == 'dense':
            return Pipeline([self.dense], self.corpus, top_k=top_k,
                            assembler=self.assembler, name='dense')
        if system == 'hybrid':
            return Pipeline([self.bm25, self.dense], self.corpus, top_k=top_k,
                            assembler=self.assembler, name='hybrid_rrf')
        if system == 'hybrid_rerank':
            return Pipeline([self.bm25, self.dense], self.corpus, top_k=top_k,
                            reranker=_reranker(), assembler=self.assembler,
                            name='hybrid_rrf_rerank')

        # 'full' -- everything the application uses by default.
        return Pipeline(
            [self.bm25, self.dense], self.corpus, top_k=top_k,
            expander=RM3Expander(self.bm25, documents),
            rewriter=ConversationalRewriter() if with_rewriter else None,
            reranker=_reranker(),
            assembler=self.assembler,
            intent_aware=True,
            name='full',
        )

    def context_for(self, query: str, max_words: int = 600,
                    top_k: int = 5) -> str:
        """
        Context string for the comic and quiz agents.

        Reuses this engine, so a task generating content for twenty concepts
        builds the indexes once rather than twenty times. Uses the reranked
        hybrid rather than the full pipeline: query expansion and
        conversational rewriting are aimed at human phrasing, and these
        queries are concept titles produced by another agent.
        """
        trace = self.pipeline(system='hybrid_rerank', top_k=top_k).search(query)
        return self.assembler.build_generation_context(
            trace.ranking(), max_words=max_words)

    def stats(self) -> dict:
        return {
            'passages': len(self.corpus),
            'vocabulary': self.bm25.vocabulary_size,
            'dimensions': self.dense.dimensions,
            'mean_passage_tokens': self.corpus.stats.mean_passage_tokens,
            'documents': self.corpus.stats.n_documents,
            'empty_documents': self.corpus.stats.empty_documents,
        }


def _concept_lookup(textbook, corpus: Corpus) -> dict[str, tuple[str, str]]:
    """
    Map each passage to its most similar `Concept`.

    The breadcrumb deliberately uses concepts rather than detected chapters.
    Chapter detection on these PDFs produces titles like ", Table 6.1.]" and
    overlapping single-page ranges, so it is not a trustworthy structural
    signal. Concepts are LLM-extracted subtopics, and they are also the unit
    the application already builds comics and quizzes around -- so a search
    result can offer "read the comic for this subtopic" for free.

    Assignment is by embedding similarity between the concept's title plus
    description and the passage. Returns an empty mapping when the textbook
    has no concepts yet, in which case breadcrumbs simply omit that level.
    """
    from .models import Concept

    concepts = list(
        Concept.objects.filter(chapter__textbook=textbook)
        .values_list('id', 'title', 'description'))
    if not concepts or not len(corpus):
        return {}

    import numpy as np

    embedder = _embedder()
    concept_vectors = embedder.encode_documents(
        [f"{title}. {description}" for _, title, description in concepts])
    passages = list(corpus)
    passage_vectors = embedder.encode_documents([p.text for p in passages])

    # Both sides are L2-normalised, so this product is the cosine.
    similarity = np.asarray(passage_vectors) @ np.asarray(concept_vectors).T
    best = similarity.argmax(axis=1)

    return {
        passage.passage_id: (concepts[best[i]][1], str(concepts[best[i]][0]))
        for i, passage in enumerate(passages)
    }


def build_engine(textbook) -> TextbookEngine:
    """
    Build an engine for one textbook from its PDF.

    Raises FileNotFoundError when the PDF is missing, and ValueError when the
    PDF yields no passages -- which happens with scanned documents. That is
    surfaced as an error rather than an empty index, because an empty index
    silently returns no results for every query and looks like a ranking
    problem instead of an ingestion one.
    """
    pdf_path = Path(textbook.pdf_file.path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF missing for textbook {textbook.id}: {pdf_path}")

    chapters = [
        {'number': c.number, 'title': c.title,
         'page_start': c.page_start, 'page_end': c.page_end}
        for c in textbook.chapters.all()
    ]

    corpus = build_from_pdfs(
        [pdf_path],
        titles={str(pdf_path): textbook.title},
        chapters={pdf_path.stem: chapters},
    )

    if not len(corpus):
        raise ValueError(
            f"'{textbook.title}' produced no searchable text. "
            f"It is most likely a scanned PDF and needs OCR."
        )

    documents = corpus.documents()
    engine = TextbookEngine(
        textbook_id=str(textbook.id),
        corpus=corpus,
        bm25=BM25Index().fit(documents),
        tfidf=TFIDFIndex().fit(documents),
        dense=DenseIndex(_embedder()).fit(documents),
        assembler=ContextAssembler(
            corpus, window=1,
            concept_lookup=_concept_lookup(textbook, corpus)),
        version=str(textbook.updated_at),
    )
    return engine


def get_engine(textbook, rebuild: bool = False) -> TextbookEngine:
    """
    Return a cached engine for `textbook`, building it if needed.

    The cache is invalidated when the textbook's `updated_at` changes, so
    re-processing a textbook produces a fresh index rather than serving a
    stale one.
    """
    key = str(textbook.id)
    version = str(textbook.updated_at)

    with _LOCK:
        cached = _ENGINES.get(key)
        if cached is not None and cached.version == version and not rebuild:
            return cached

    # Built outside the lock: embedding a corpus takes seconds and must not
    # block searches against other textbooks.
    engine = build_engine(textbook)

    with _LOCK:
        _ENGINES[key] = engine
    return engine


def invalidate(textbook_id: str) -> None:
    """Drop a cached engine, e.g. after re-processing a textbook."""
    with _LOCK:
        _ENGINES.pop(str(textbook_id), None)


def search(textbook, query: str, system: str = 'full', top_k: int = 10,
           history: list | None = None):
    """
    Search one textbook and return a `RetrievalTrace`.

    `history` is a list of `ir.query.rewrite.ConversationTurn`; passing any
    enables the conversational rewriter, so a follow-up such as "why does it
    increase?" is resolved against the previous turn before retrieval.
    """
    engine = get_engine(textbook)
    pipeline = engine.pipeline(system=system, top_k=top_k,
                              with_rewriter=bool(history))
    return pipeline.search(query, history=history)


def answer_question(textbook, trace, query: str | None = None):
    """
    Generate a grounded answer from a completed retrieval.

    Takes the trace rather than re-running retrieval, so the answer is built
    from exactly the passages the page displays -- the citations then point at
    results the student can see and check, which is the whole value of
    showing them.

    Uses `trace.effective_query` by default, so a rewritten follow-up
    ("why does it increase?" -> "why does entropy increase?") is answered as
    the resolved question rather than the elliptical one.
    """
    engine = get_engine(textbook)
    return _answerer().generate(
        query or trace.effective_query, trace.results, engine.assembler)


def retrieve_context(textbook, query: str, max_words: int = 600,
                     top_k: int = 5) -> str:
    """
    Context string for the comic and quiz agents.

    This is the replacement for `rag.retriever.retrieve` plus
    `build_context`. Two differences that matter for generated content:
    retrieval is hybrid and reranked rather than dense-only, and each block
    carries its citation, so a generated caption can be traced back to a page.
    """
    engine = get_engine(textbook)
    trace = engine.pipeline(system='hybrid_rerank', top_k=top_k).search(query)
    return engine.assembler.build_generation_context(
        trace.ranking(), max_words=max_words)
