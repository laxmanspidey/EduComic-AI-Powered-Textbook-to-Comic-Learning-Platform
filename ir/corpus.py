"""
Document processing: PDF files -> a flat collection of retrievable passages.

This module deliberately replaces the indexing path in `tasks/comic_tasks.py`,
which chunked text *inside* the per-chapter loop:

    for ch in detected_chapters:
        chunks = chunk_pages(pages, ch['page_start'], ch['page_end'])

That makes the corpus a function of chapter detection. When detection
misfires -- and on real textbooks it does, producing overlapping or
single-page ranges -- pages outside every detected range are never indexed,
while pages inside two ranges are indexed twice. The corpus silently shrinks
and duplicates, with nothing reported.

Here the order is inverted:

    1. Extract every page.
    2. Chunk the entire document.
    3. Attach chapter/section metadata to each passage by page lookup.

Chapter detection becomes advisory metadata rather than a gate. Every page with
text is indexed exactly once, whatever the detector does, and `CorpusStats`
reports any document that yielded nothing instead of failing quietly.

Chunking has two modes, so the choice is measurable rather than assumed:

  'continuous'   (default) Concatenate the document's pages into one stream and
                 chunk across page boundaries, recording the page span each
                 passage covers. A paragraph split across a page break stays in
                 one passage.
  'page_bounded' Chunk each page independently, as the original code did. Kept
                 so "does page-bounded chunking hurt retrieval?" can be an
                 ablation row rather than a guess.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Passage:
    """
    One retrievable unit.

    `passage_id` is the document id for every index, qrels file and run file in
    the package, so it must be stable across rebuilds. It is derived from the
    document id plus the passage's ordinal position, not from a hash of the
    text, so that re-chunking with different parameters produces a clearly
    different id set rather than a confusing partial overlap.
    """
    passage_id: str
    text: str
    doc_id: str
    doc_title: str
    page_start: int
    page_end: int
    position: int
    token_count: int
    chapter_number: int | None = None
    chapter_title: str = ''
    parent_id: str = ''

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p{self.page_start}"
        return f"p{self.page_start}-{self.page_end}"

    @property
    def citation(self) -> str:
        """Human-readable provenance, shown in the retrieval UI."""
        parts = [self.doc_title, self.page_label]
        if self.chapter_title:
            parts.insert(1, f"ch{self.chapter_number}: {self.chapter_title}")
        return " | ".join(p for p in parts if p)


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class CorpusStats:
    """Build report. `empty_documents` is the signal the old pipeline lacked."""
    n_documents: int = 0
    n_passages: int = 0
    n_pages: int = 0
    n_pages_with_text: int = 0
    total_tokens: int = 0
    per_document: dict[str, int] = field(default_factory=dict)
    empty_documents: list[str] = field(default_factory=list)
    duplicate_documents: list[str] = field(default_factory=list)

    @property
    def mean_passage_tokens(self) -> float:
        return self.total_tokens / self.n_passages if self.n_passages else 0.0

    def report(self) -> str:
        lines = [
            "Corpus build report",
            "-" * 58,
            f"documents indexed      {self.n_documents}",
            f"pages seen             {self.n_pages}",
            f"pages with text        {self.n_pages_with_text}",
            f"passages built         {self.n_passages}",
            f"mean passage tokens    {self.mean_passage_tokens:.1f}",
        ]
        if self.per_document:
            lines.append("")
            lines.append("passages per document:")
            for doc, n in sorted(self.per_document.items(),
                                 key=lambda kv: -kv[1]):
                lines.append(f"  {doc[:44]:46s} {n:6d}")
        if self.duplicate_documents:
            lines.append("")
            lines.append("SKIPPED as byte-identical duplicates:")
            for d in self.duplicate_documents:
                lines.append(f"  {d}")
        if self.empty_documents:
            lines.append("")
            lines.append("WARNING - produced zero passages (no extractable "
                         "text; likely a scanned PDF needing OCR):")
            for d in self.empty_documents:
                lines.append(f"  {d}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────────────

#: A page with fewer than this many characters is treated as having no usable
#: text. Catches page-number-only pages and the blank pages that scanned PDFs
#: produce for every single page.
MIN_PAGE_CHARS = 30


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """
    Extract plain text from every page of a PDF.

    Returns one PageText per page, including pages whose text is empty, so the
    caller can report how much of the document was unreadable rather than
    silently seeing a short document.
    """
    import fitz  # imported lazily: the metrics and index modules do not need it

    pages: list[PageText] = []
    doc = fitz.open(str(pdf_path))
    try:
        for i in range(len(doc)):
            pages.append(PageText(page=i + 1, text=doc[i].get_text("text").strip()))
    finally:
        doc.close()
    return pages


def clean_text(text: str) -> str:
    """
    Normalise whitespace and repair the artefacts PDF extraction introduces.

    Specifically: de-hyphenate words broken across lines ("capac-\\nitor" ->
    "capacitor"), collapse runs of whitespace, and drop isolated page-number
    lines. These matter for retrieval because a hyphen-split token is simply
    unmatchable -- no query will ever contain "capac".
    """
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────

CHUNK_MODES = ('continuous', 'page_bounded')


def _window(words: list, size: int, overlap: int):
    """Yield (start, end) index pairs for a sliding window over `words`."""
    if size <= 0:
        raise ValueError(f"chunk size must be > 0, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size), got {overlap}")
    stride = size - overlap
    start = 0
    n = len(words)
    while start < n:
        yield start, min(start + size, n)
        if start + size >= n:
            break
        start += stride


def chunk_continuous(pages: list[PageText], chunk_size: int = 400,
                     overlap: int = 80) -> list[tuple[str, int, int]]:
    """
    Chunk a document as one continuous word stream, across page boundaries.

    Returns [(text, page_start, page_end), ...]. Each word carries the page it
    came from, so a passage spanning a page break reports the full span and
    provenance survives.
    """
    words: list[str] = []
    word_pages: list[int] = []
    for page in pages:
        cleaned = clean_text(page.text)
        if len(cleaned) < MIN_PAGE_CHARS:
            continue
        for w in cleaned.split():
            words.append(w)
            word_pages.append(page.page)

    out: list[tuple[str, int, int]] = []
    for start, end in _window(words, chunk_size, overlap):
        span = word_pages[start:end]
        out.append((' '.join(words[start:end]), min(span), max(span)))
    return out


def chunk_page_bounded(pages: list[PageText], chunk_size: int = 400,
                       overlap: int = 80) -> list[tuple[str, int, int]]:
    """
    Chunk each page independently -- the original pipeline's behaviour.

    Provided for ablation. A passage never spans pages, which means any
    paragraph crossing a page break is cut in two and both halves lose context.
    """
    out: list[tuple[str, int, int]] = []
    for page in pages:
        cleaned = clean_text(page.text)
        if len(cleaned) < MIN_PAGE_CHARS:
            continue
        words = cleaned.split()
        for start, end in _window(words, chunk_size, overlap):
            out.append((' '.join(words[start:end]), page.page, page.page))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Structure attachment
# ──────────────────────────────────────────────────────────────────────────────

def attach_chapters(passages: list[Passage],
                    chapters: list[dict]) -> list[Passage]:
    """
    Label each passage with the chapter covering its starting page.

    `chapters` is a list of {'number', 'title', 'page_start', 'page_end'} --
    the same shape `rag.extractor.detect_chapters` returns, so an existing
    detector can be reused as a metadata source.

    Overlapping ranges are resolved by preferring the narrowest range that
    contains the page, which is the most specific available label. A passage on
    a page no chapter covers keeps `chapter_number = None` and is still
    retrievable -- the point of this module.
    """
    if not chapters:
        return passages

    ordered = sorted(chapters,
                     key=lambda c: (c['page_end'] - c['page_start'],
                                    c['page_start']))
    for p in passages:
        for ch in ordered:
            if ch['page_start'] <= p.page_start <= ch['page_end']:
                p.chapter_number = ch['number']
                p.chapter_title = str(ch.get('title', ''))[:200]
                break
    return passages


def assign_parents(passages: list[Passage], group_size: int = 3) -> list[Passage]:
    """
    Group consecutive passages into parent blocks for context expansion.

    Parent-child retrieval searches over small passages, which embed and score
    well, then hands the surrounding parent block to the generator, which needs
    context. `group_size` consecutive passages of one document share a
    `parent_id`.

    Grouping by position rather than by detected section is intentional: it
    does not inherit the chapter detector's errors, and it guarantees every
    passage has a parent.
    """
    for p in passages:
        p.parent_id = f"{p.doc_id}::parent{p.position // group_size}"
    return passages


# ──────────────────────────────────────────────────────────────────────────────
# Corpus
# ──────────────────────────────────────────────────────────────────────────────

class Corpus:
    """
    An ordered collection of passages, with the lookups the indexes need.

    Serialises to JSONL so an experiment can be re-run against a frozen corpus
    without re-parsing PDFs -- which also means a reported result stays
    reproducible after the PDFs change.
    """

    def __init__(self, passages: list[Passage] | None = None,
                 stats: CorpusStats | None = None):
        self.passages: list[Passage] = passages or []
        self.stats: CorpusStats = stats or CorpusStats()
        self._by_id: dict[str, Passage] = {p.passage_id: p for p in self.passages}

    # ── lookups ───────────────────────────────────────────────────────────
    def documents(self) -> dict[str, str]:
        """{passage_id -> text}, the input format every index takes."""
        return {p.passage_id: p.text for p in self.passages}

    def get(self, passage_id: str) -> Passage | None:
        return self._by_id.get(passage_id)

    def parent_text(self, passage_id: str) -> str:
        """
        Concatenated text of every passage sharing this one's parent block.

        Overlapping windows mean adjacent passages repeat words; that is
        acceptable for a generation context and is why this is not used for
        scoring.
        """
        p = self._by_id.get(passage_id)
        if p is None:
            return ''
        return ' '.join(q.text for q in self.passages if q.parent_id == p.parent_id)

    def by_document(self, doc_id: str) -> list[Passage]:
        return [p for p in self.passages if p.doc_id == doc_id]

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as fh:
            fh.write(json.dumps({'_stats': asdict(self.stats)}) + '\n')
            for p in self.passages:
                fh.write(json.dumps(asdict(p), ensure_ascii=False) + '\n')

    @classmethod
    def load(cls, path: str | Path) -> 'Corpus':
        passages: list[Passage] = []
        stats = CorpusStats()
        with Path(path).open('r', encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if '_stats' in obj:
                    stats = CorpusStats(**obj['_stats'])
                    continue
                passages.append(Passage(**obj))
        return cls(passages, stats)

    def __len__(self) -> int:
        return len(self.passages)

    def __iter__(self):
        return iter(self.passages)

    def __repr__(self) -> str:
        return (f"Corpus(passages={len(self.passages)}, "
                f"documents={self.stats.n_documents})")


# ──────────────────────────────────────────────────────────────────────────────
# Build entry points
# ──────────────────────────────────────────────────────────────────────────────

def build_from_pdfs(
    pdf_paths: list[str | Path],
    titles: dict[str, str] | None = None,
    chapters: dict[str, list[dict]] | None = None,
    chunk_size: int = 400,
    overlap: int = 80,
    mode: str = 'continuous',
    parent_group_size: int = 3,
    skip_duplicates: bool = True,
) -> Corpus:
    """
    Build a corpus from PDF files on disk.

    Args:
        pdf_paths:         files to index.
        titles:            path (as str) -> display title. Defaults to filename.
        chapters:          doc_id -> chapter dicts, for metadata only.
        chunk_size:        words per passage.
        overlap:           words shared between adjacent passages.
        mode:              one of CHUNK_MODES.
        parent_group_size: passages per parent block.
        skip_duplicates:   skip byte-identical files. Your media/textbooks
                           currently holds the same 400-page PDF twice under
                           two names, which would otherwise double-count every
                           passage in it and inflate every measured score.

    Returns:
        A Corpus. Inspect `.stats.report()` before trusting it -- that is where
        a scanned PDF shows up as an explicit warning.
    """
    if mode not in CHUNK_MODES:
        raise ValueError(f"mode must be one of {CHUNK_MODES}, got {mode!r}")

    chunker = chunk_continuous if mode == 'continuous' else chunk_page_bounded
    titles = titles or {}
    chapters = chapters or {}

    passages: list[Passage] = []
    stats = CorpusStats()
    seen_hashes: dict[str, str] = {}

    for path in pdf_paths:
        path = Path(path)
        doc_id = path.stem
        doc_title = titles.get(str(path), path.stem)

        if skip_duplicates:
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                stats.duplicate_documents.append(
                    f"{path.name} (identical to {seen_hashes[digest]})")
                continue
            seen_hashes[digest] = path.name

        pages = extract_pages(path)
        stats.n_pages += len(pages)
        stats.n_pages_with_text += sum(
            1 for pg in pages if len(clean_text(pg.text)) >= MIN_PAGE_CHARS)

        chunks = chunker(pages, chunk_size, overlap)
        if not chunks:
            stats.empty_documents.append(f"{path.name} ({len(pages)} pages)")
            stats.per_document[doc_id] = 0
            stats.n_documents += 1
            continue

        doc_passages = [
            Passage(
                passage_id=f"{doc_id}::p{i}",
                text=text,
                doc_id=doc_id,
                doc_title=doc_title,
                page_start=page_start,
                page_end=page_end,
                position=i,
                token_count=len(text.split()),
            )
            for i, (text, page_start, page_end) in enumerate(chunks)
        ]

        attach_chapters(doc_passages, chapters.get(doc_id, []))
        assign_parents(doc_passages, parent_group_size)

        passages.extend(doc_passages)
        stats.per_document[doc_id] = len(doc_passages)
        stats.n_documents += 1

    stats.n_passages = len(passages)
    stats.total_tokens = sum(p.token_count for p in passages)
    return Corpus(passages, stats)


def build_from_texts(texts: dict[str, str], chunk_size: int = 400,
                     overlap: int = 80) -> Corpus:
    """
    Build a corpus from in-memory text, keyed by document id.

    Used by the tests and for quick experiments, so no part of the retrieval
    stack needs a PDF on disk to be exercised.
    """
    passages: list[Passage] = []
    stats = CorpusStats()

    for doc_id, text in texts.items():
        pages = [PageText(page=1, text=text)]
        chunks = chunk_continuous(pages, chunk_size, overlap)
        doc_passages = [
            Passage(
                passage_id=f"{doc_id}::p{i}",
                text=chunk_text,
                doc_id=doc_id,
                doc_title=doc_id,
                page_start=ps,
                page_end=pe,
                position=i,
                token_count=len(chunk_text.split()),
            )
            for i, (chunk_text, ps, pe) in enumerate(chunks)
        ]
        assign_parents(doc_passages)
        passages.extend(doc_passages)
        stats.per_document[doc_id] = len(doc_passages)
        stats.n_documents += 1
        if not doc_passages:
            stats.empty_documents.append(doc_id)

    stats.n_passages = len(passages)
    stats.n_pages = stats.n_documents
    stats.n_pages_with_text = stats.n_documents
    stats.total_tokens = sum(p.token_count for p in passages)
    return Corpus(passages, stats)
