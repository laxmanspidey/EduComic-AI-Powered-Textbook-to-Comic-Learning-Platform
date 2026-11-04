"""
Context assembly: turning a retrieved passage into usable evidence.

The failure this addresses is specific to chunked retrieval. A passage reads:

    "This process increases the value by 20%."

Retrieved on its own it is useless -- *what* process, *what* value. The
referent lives in the preceding passage, which was never retrieved because it
does not match the query. The ranking was correct; the unit of delivery was
too small.

Two mechanisms, deliberately separate:

  Parent-child expansion
      Search over small passages, deliver the larger block around them.
      Small passages rank better -- a 400-word chunk mentioning the query term
      once is diluted, and embeddings of long text blur. But generation and
      reading want the surrounding context. So retrieval and delivery use
      different granularities, which is the whole point.

  Hierarchical breadcrumb
      Attach where the passage sits: textbook -> subtopic -> page. This is
      what lets a result say "covered in subtopic Electrostatic Shielding"
      and link to the comic and quiz that already exist for it.

Note on the hierarchy: the breadcrumb uses the `Concept` layer rather than
detected chapters. Chapter detection on these PDFs produces titles like
", Table 6.1.]" and three chapters all claiming page 4, so it is not a
trustworthy structural signal. Concepts are LLM-extracted subtopics and are
both more reliable and more useful, since they are the unit the application
already builds comics and quizzes around.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Breadcrumb:
    """Where a passage sits in the textbook's structure."""
    textbook: str = ''
    chapter: str = ''
    concept: str = ''
    pages: str = ''
    concept_id: str = ''

    def path(self) -> list[str]:
        """Non-empty levels, coarsest first, for rendering as a trail."""
        return [part for part in (self.textbook, self.chapter, self.concept)
                if part]

    def __str__(self) -> str:
        trail = ' > '.join(self.path())
        return f"{trail} ({self.pages})" if self.pages else trail


@dataclass
class AssembledContext:
    """
    One retrieved passage plus everything needed to present or generate from it.

    Attributes:
        passage_id:    the passage that was actually retrieved and scored.
        passage_text:  that passage alone. Always the scored unit, so a score
                       is never reported against text the ranking never saw.
        context_text:  the expanded block handed to a generator.
        neighbour_ids: passages included in the expansion, in reading order.
        breadcrumb:    structural position.
        truncated:     whether `context_text` hit the character budget.
    """
    passage_id: str
    passage_text: str
    context_text: str
    neighbour_ids: list[str] = field(default_factory=list)
    breadcrumb: Breadcrumb = field(default_factory=Breadcrumb)
    truncated: bool = False

    @property
    def expanded(self) -> bool:
        return len(self.neighbour_ids) > 1


class ContextAssembler:
    """
    Builds AssembledContext objects from a corpus.

    Args:
        corpus:         an `ir.corpus.Corpus`.
        window:         neighbouring passages to include on each side by
                        position within the document. 0 disables expansion.
        use_parent:     expand to the whole parent block instead of a
                        positional window. Parent blocks are fixed groupings
                        assigned at index time, so every passage in a block
                        yields identical context -- better for caching, less
                        precise than a window centred on the hit.
        max_chars:      budget for `context_text`. Guards the downstream LLM
                        prompt, since an unbounded context can exceed the
                        model's window and silently drop the passage that
                        was actually retrieved.
        concept_lookup: passage_id -> (concept_title, concept_id), supplied by
                        the Django layer. Kept injectable so this module has
                        no Django dependency and stays testable standalone.
    """

    def __init__(self, corpus, window: int = 1, use_parent: bool = False,
                 max_chars: int = 4000,
                 concept_lookup: dict[str, tuple[str, str]] | None = None):
        if window < 0:
            raise ValueError(f"window must be >= 0, got {window}")
        self.corpus = corpus
        self.window = window
        self.use_parent = use_parent
        self.max_chars = max_chars
        self.concept_lookup = concept_lookup or {}

        # position index per document, so neighbours are O(1) to find.
        self._by_document: dict[str, list] = {}
        for passage in corpus:
            self._by_document.setdefault(passage.doc_id, []).append(passage)
        for passages in self._by_document.values():
            passages.sort(key=lambda p: p.position)

    # ── neighbours ────────────────────────────────────────────────────────
    def _neighbours(self, passage) -> list:
        """Passages to include alongside `passage`, in reading order."""
        siblings = self._by_document.get(passage.doc_id, [])
        if not siblings:
            return [passage]

        if self.use_parent:
            return [p for p in siblings if p.parent_id == passage.parent_id] \
                   or [passage]

        if self.window == 0:
            return [passage]

        index = next((i for i, p in enumerate(siblings)
                      if p.passage_id == passage.passage_id), None)
        if index is None:
            return [passage]
        low = max(0, index - self.window)
        high = min(len(siblings), index + self.window + 1)
        return siblings[low:high]

    @staticmethod
    def _join(passages: list) -> str:
        """
        Concatenate passages, dropping the overlap between adjacent ones.

        Passages overlap by design (80 words by default), so naive joining
        repeats text. Repetition in a generation prompt wastes budget and
        makes the model treat the duplicated span as emphasised, so the
        overlap is removed by matching the tail of one passage against the
        head of the next.
        """
        if not passages:
            return ''

        parts = [passages[0].text]
        for previous, current in zip(passages, passages[1:]):
            previous_words = previous.text.split()
            current_words = current.text.split()

            overlap = 0
            limit = min(len(previous_words), len(current_words))
            # Longest suffix of `previous` that prefixes `current`.
            for size in range(limit, 0, -1):
                if previous_words[-size:] == current_words[:size]:
                    overlap = size
                    break
            parts.append(' '.join(current_words[overlap:]))

        return ' '.join(part for part in parts if part).strip()

    # ── breadcrumb ────────────────────────────────────────────────────────
    def breadcrumb_for(self, passage) -> Breadcrumb:
        """Structural position, preferring the concept layer over chapters."""
        concept_title, concept_id = self.concept_lookup.get(
            passage.passage_id, ('', ''))
        return Breadcrumb(
            textbook=passage.doc_title or passage.doc_id,
            chapter=(f"Ch {passage.chapter_number}: {passage.chapter_title}"
                     if passage.chapter_number and passage.chapter_title else ''),
            concept=concept_title,
            pages=passage.page_label,
            concept_id=concept_id,
        )

    # ── assembly ──────────────────────────────────────────────────────────
    def assemble(self, passage_id: str) -> AssembledContext | None:
        """Assemble context for one passage, or None if it is unknown."""
        passage = self.corpus.get(passage_id)
        if passage is None:
            return None

        neighbours = self._neighbours(passage)
        context = self._join(neighbours)

        truncated = False
        if self.max_chars and len(context) > self.max_chars:
            context = context[:self.max_chars].rsplit(' ', 1)[0]
            truncated = True

        return AssembledContext(
            passage_id=passage_id,
            passage_text=passage.text,
            context_text=context,
            neighbour_ids=[p.passage_id for p in neighbours],
            breadcrumb=self.breadcrumb_for(passage),
            truncated=truncated,
        )

    def assemble_many(self, passage_ids: list[str]) -> list[AssembledContext]:
        """Assemble several, skipping unknown ids."""
        out = []
        for passage_id in passage_ids:
            assembled = self.assemble(passage_id)
            if assembled is not None:
                out.append(assembled)
        return out

    def build_generation_context(self, passage_ids: list[str],
                                 max_words: int = 600) -> str:
        """
        One context string for the comic and quiz agents.

        Replaces `rag.retriever.build_context`, with two differences that
        matter: passages are de-overlapped before joining, and each block
        carries its citation so a generated caption can be traced back to a
        page.

        Deduplication works on the union of every hit's expansion window, not
        on each window in turn. Two adjacent hits expand into windows that
        share most of their passages -- emitting both would repeat whole
        paragraphs, and emitting only the second's new passage would strand it
        without its context. So the union is collected first, then split into
        contiguous runs of passage positions, and each run is de-overlapped
        into one block. Adjacent hits therefore produce a single merged block
        rather than two overlapping ones.
        """
        selected: dict[str, list] = {}
        order: list[str] = []

        for passage_id in passage_ids:
            assembled = self.assemble(passage_id)
            if assembled is None:
                continue
            passage = self.corpus.get(passage_id)
            if passage.doc_id not in order:
                order.append(passage.doc_id)
            bucket = selected.setdefault(passage.doc_id, [])
            for neighbour_id in assembled.neighbour_ids:
                neighbour = self.corpus.get(neighbour_id)
                if neighbour is not None and neighbour not in bucket:
                    bucket.append(neighbour)

        blocks: list[str] = []
        words = 0

        for doc_id in order:
            passages = sorted(selected[doc_id], key=lambda p: p.position)

            # Split into contiguous runs so a gap between two hits in the same
            # document does not get joined into one misleading block.
            runs: list[list] = []
            for passage in passages:
                if runs and passage.position == runs[-1][-1].position + 1:
                    runs[-1].append(passage)
                else:
                    runs.append([passage])

            for run in runs:
                text = self._join(run)
                block_words = text.split()
                if not block_words:
                    continue
                if words + len(block_words) > max_words:
                    remaining = max_words - words
                    if remaining < 30:
                        return '\n\n'.join(blocks)
                    block_words = block_words[:remaining]
                    text = ' '.join(block_words)

                blocks.append(f"[{self.breadcrumb_for(run[0])}]\n{text}")
                words += len(block_words)
                if words >= max_words:
                    return '\n\n'.join(blocks)

        return '\n\n'.join(blocks)
