"""
Test collection construction.

A retrieval system cannot be claimed to work without queries and relevance
judgements, and hand-annotating those is the expensive part of any IR project.
Queries here are generated synthetically: an LLM reads a passage and writes a
question that passage answers, so the source passage is relevant *by
construction* and needs no separate judging.

Two query styles are generated per passage, which is what turns the collection
into an experiment rather than just a benchmark:

  literal     Phrased using the passage's own terminology. Lexical retrieval
              should do well.
  paraphrased Phrased the way a student who has not learned the terminology
              would ask, deliberately avoiding the passage's key terms.
              Lexical retrieval should struggle; semantic retrieval and query
              expansion should be the methods that recover it.

Reporting nDCG separately by style measures the terminology-mismatch problem
directly, instead of asserting it from a hand-picked example.

Graded judgements, and why adjacent passages get partial credit
---------------------------------------------------------------
Passages overlap by design -- 80 of 400 words with the default chunking. Two
adjacent passages frequently contain the same sentence, so a system returning
the neighbour of the source passage has very likely returned the answer. In
this corpus, `physics_txt_book::p25` and `::p26` share 80 words and *both*
contain the electrostatic shielding definition.

Marking only the source passage relevant would therefore punish a system for
returning an equally good answer. Judgements are:

    2  the source passage the question was generated from
    1  a passage whose word overlap with the source exceeds
       `OVERLAP_THRESHOLD`, i.e. one that probably contains the same text
    0  everything else

This is a stated assumption, not a measurement: the overlapping region might
not contain the answering sentence. It is recorded in each judgement's
`source` field so the effect can be isolated -- evaluating with
`rel_threshold=2` ignores the partial credit entirely.

Known limitation
----------------
Synthetic judgements are incomplete. A genuinely relevant passage that the
generator never saw counts as non-relevant, which penalises exactly the
systems good enough to find it. Pooled judging (run every system, judge the
union of their top-k) is the standard fix; `qrels` carries a `source` field on
every judgement so pooled entries can be merged in later without rebuilding
the collection.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'collection'

#: Word-overlap ratio above which a passage receives partial credit.
OVERLAP_THRESHOLD = 0.15

#: Judgement grades.
GRADE_SOURCE = 2
GRADE_OVERLAP = 1


@dataclass
class EvalQuery:
    """One test query."""
    query_id: str
    text: str
    source: str = 'synthetic'        # synthetic | manual | quiz
    style: str = 'literal'           # literal | paraphrased
    source_passage_id: str = ''
    doc_id: str = ''


@dataclass
class Judgement:
    """One relevance judgement, with its provenance."""
    query_id: str
    passage_id: str
    grade: int
    source: str = 'synthetic'        # synthetic | overlap | llm | human


@dataclass
class CollectionStats:
    n_queries: int = 0
    n_judgements: int = 0
    by_style: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    by_grade: dict[str, int] = field(default_factory=dict)
    mean_relevant_per_query: float = 0.0
    max_relevant_per_query: int = 0
    single_relevant_queries: int = 0

    def report(self) -> str:
        lines = [
            "Test collection",
            "-" * 52,
            f"queries                 {self.n_queries}",
            f"judgements              {self.n_judgements}",
            f"mean relevant / query   {self.mean_relevant_per_query:.2f}",
        ]
        for label, mapping in (('style', self.by_style),
                               ('query source', self.by_source),
                               ('grade', self.by_grade)):
            if mapping:
                lines.append('')
                lines.append(f"by {label}:")
                for key, count in sorted(mapping.items()):
                    lines.append(f"  {str(key):22s} {count}")
        # Be precise about when measures actually coincide rather than
        # warning on a rounded average: MAP equals MRR only when *every*
        # query has exactly one relevant passage.
        if self.n_queries and self.max_relevant_per_query <= 1:
            lines.append('')
            lines.append("NOTE: every query has exactly one relevant passage, "
                         "so Recall@k equals")
            lines.append("      Success@k and MAP equals MRR exactly. Report "
                         "MRR and nDCG@k only.")
        elif self.n_queries and self.single_relevant_queries / self.n_queries > 0.5:
            share = self.single_relevant_queries / self.n_queries
            lines.append('')
            lines.append(f"NOTE: {share:.0%} of queries have a single relevant "
                         f"passage, so MAP and MRR")
            lines.append("      will be highly correlated. Prefer MRR and "
                         "nDCG@k in the results table.")
        return '\n'.join(lines)


class TestCollection:
    """Queries plus judgements, with the lookups the evaluator needs."""

    def __init__(self, queries: list[EvalQuery] | None = None,
                 judgements: list[Judgement] | None = None):
        self.queries: list[EvalQuery] = queries or []
        self.judgements: list[Judgement] = judgements or []

    # ── views ─────────────────────────────────────────────────────────────
    def qrels(self) -> dict[str, dict[str, int]]:
        """{query_id -> {passage_id -> grade}}, as `ir.metrics` expects."""
        out: dict[str, dict[str, int]] = {}
        for judgement in self.judgements:
            out.setdefault(judgement.query_id, {})[judgement.passage_id] = \
                judgement.grade
        return out

    def query_texts(self) -> dict[str, str]:
        return {q.query_id: q.text for q in self.queries}

    def by_style(self, style: str) -> 'TestCollection':
        """
        A sub-collection of one query style.

        This is how the literal-vs-paraphrased comparison is run: evaluate the
        same systems on each sub-collection and report both columns.
        """
        kept = [q for q in self.queries if q.style == style]
        keep_ids = {q.query_id for q in kept}
        return TestCollection(
            kept, [j for j in self.judgements if j.query_id in keep_ids])

    def stats(self) -> CollectionStats:
        stats = CollectionStats(n_queries=len(self.queries),
                                n_judgements=len(self.judgements))
        for query in self.queries:
            stats.by_style[query.style] = stats.by_style.get(query.style, 0) + 1
            stats.by_source[query.source] = \
                stats.by_source.get(query.source, 0) + 1
        for judgement in self.judgements:
            key = f"grade {judgement.grade} ({judgement.source})"
            stats.by_grade[key] = stats.by_grade.get(key, 0) + 1
        if self.queries:
            per_query: dict[str, int] = {q.query_id: 0 for q in self.queries}
            for judgement in self.judgements:
                if judgement.grade >= 1 and judgement.query_id in per_query:
                    per_query[judgement.query_id] += 1
            counts = list(per_query.values())
            stats.mean_relevant_per_query = sum(counts) / len(counts)
            stats.max_relevant_per_query = max(counts)
            stats.single_relevant_queries = sum(1 for c in counts if c == 1)
        return stats

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, directory: str | Path = DATA_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'queries.jsonl').open('w', encoding='utf-8') as fh:
            for query in self.queries:
                fh.write(json.dumps(asdict(query), ensure_ascii=False) + '\n')
        with (directory / 'qrels.jsonl').open('w', encoding='utf-8') as fh:
            for judgement in self.judgements:
                fh.write(json.dumps(asdict(judgement)) + '\n')

    @classmethod
    def load(cls, directory: str | Path = DATA_DIR) -> 'TestCollection':
        directory = Path(directory)
        queries, judgements = [], []
        queries_path = directory / 'queries.jsonl'
        qrels_path = directory / 'qrels.jsonl'
        if queries_path.exists():
            for line in queries_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    queries.append(EvalQuery(**json.loads(line)))
        if qrels_path.exists():
            for line in qrels_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    judgements.append(Judgement(**json.loads(line)))
        return cls(queries, judgements)

    def merge_judgements(self, extra: list[Judgement]) -> None:
        """
        Add judgements, keeping the highest grade per (query, passage).

        This is the hook for pooled judging: a later pooling run merges in
        without rebuilding the collection, and a human judgement can override
        a synthetic one by carrying a higher grade.
        """
        best: dict[tuple[str, str], Judgement] = {
            (j.query_id, j.passage_id): j for j in self.judgements}
        for judgement in extra:
            key = (judgement.query_id, judgement.passage_id)
            current = best.get(key)
            if current is None or judgement.grade > current.grade:
                best[key] = judgement
        self.judgements = list(best.values())

    def __len__(self) -> int:
        return len(self.queries)

    def __repr__(self) -> str:
        return (f"TestCollection(queries={len(self.queries)}, "
                f"judgements={len(self.judgements)})")


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic query generation
# ──────────────────────────────────────────────────────────────────────────────

GENERATION_PROMPT = """You are building a test set for a textbook search \
engine. Read this textbook passage:

\"\"\"{passage}\"\"\"

Write exactly two questions a student might search for, both answerable by \
this passage.

Question 1 (literal): use the passage's own technical terms.
Question 2 (paraphrased): ask the same thing in everyday words, deliberately \
AVOIDING the passage's technical terms. A student who has not learned the \
terminology yet would phrase it this way.

Rules:
- Each question must be answerable from this passage alone
- 5 to 15 words each
- Natural search phrasing, not exam phrasing
- Question 2 must not reuse the key technical nouns from question 1

Return ONLY a JSON object:
{{"literal": "...", "paraphrased": "..."}}"""


def _word_overlap(a: str, b: str) -> float:
    """Jaccard-style overlap of word sets, relative to the smaller passage."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


class SyntheticQueryGenerator:
    """
    Generates a test collection from a corpus using a local LLM.

    Args:
        corpus:    an `ir.corpus.Corpus`.
        model:     Ollama model name.
        n_passages: how many passages to sample. Two queries are produced per
                   passage, so the collection is roughly twice this size.
        seed:      fixes the sample, so a rebuilt collection is the same
                   collection and results stay comparable.
        min_tokens: skip passages shorter than this; a stub passage yields a
                   question with no real answer in it.
    """

    def __init__(self, corpus, model: str = 'qwen3.5:4b',
                 base_url: str = 'http://127.0.0.1:11434',
                 n_passages: int = 100, seed: int = 42,
                 min_tokens: int = 80, temperature: float = 0.3,
                 timeout: int = 120):
        self.corpus = corpus
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.n_passages = n_passages
        self.seed = seed
        self.min_tokens = min_tokens
        self.temperature = temperature
        self.timeout = timeout

    # ── model ─────────────────────────────────────────────────────────────
    def _call(self, passage_text: str) -> dict:
        import requests

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                'model': self.model,
                'messages': [{'role': 'user', 'content':
                              GENERATION_PROMPT.format(passage=passage_text[:2500])}],
                'stream': False,
                'think': False,
                'options': {'temperature': self.temperature,
                            'num_predict': 250},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json().get('message', {}).get('content', '')
        return self._parse(content)

    @staticmethod
    def _parse(content: str) -> dict:
        """Extract {'literal': ..., 'paraphrased': ...} defensively."""
        content = re.sub(r'<think>[\s\S]*?</think>', '', content,
                         flags=re.IGNORECASE).strip()
        fenced = re.search(r'```(?:json)?\s*([\s\S]+?)```', content,
                           re.IGNORECASE)
        candidates = [fenced.group(1).strip()] if fenced else []
        candidates.append(content)
        match = re.search(r'(\{[\s\S]+?\})', content)
        if match:
            candidates.append(match.group(1))

        for text in candidates:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                literal = str(parsed.get('literal', '')).strip()
                paraphrased = str(parsed.get('paraphrased', '')).strip()
                if literal or paraphrased:
                    return {'literal': literal, 'paraphrased': paraphrased}
        return {}

    # ── judgements ────────────────────────────────────────────────────────
    def _judgements_for(self, query_id: str, source_passage) -> list[Judgement]:
        """Grade 2 for the source, grade 1 for passages that overlap it."""
        out = [Judgement(query_id=query_id,
                         passage_id=source_passage.passage_id,
                         grade=GRADE_SOURCE, source='synthetic')]

        for other in self.corpus.by_document(source_passage.doc_id):
            if other.passage_id == source_passage.passage_id:
                continue
            if abs(other.position - source_passage.position) > 1:
                continue
            if _word_overlap(source_passage.text, other.text) >= OVERLAP_THRESHOLD:
                out.append(Judgement(query_id=query_id,
                                     passage_id=other.passage_id,
                                     grade=GRADE_OVERLAP, source='overlap'))
        return out

    # ── generation ────────────────────────────────────────────────────────
    def sample_passages(self) -> list:
        """
        Choose passages to generate from.

        Sampled across documents proportionally rather than uniformly over all
        passages, so a long document cannot dominate the query set and make
        the evaluation mostly about one textbook.
        """
        by_document: dict[str, list] = {}
        for passage in self.corpus:
            if passage.token_count >= self.min_tokens:
                by_document.setdefault(passage.doc_id, []).append(passage)
        if not by_document:
            return []

        rng = random.Random(self.seed)
        budget = min(self.n_passages,
                     sum(len(v) for v in by_document.values()))
        per_document = max(1, budget // len(by_document))

        chosen = []
        for doc_id in sorted(by_document):
            passages = sorted(by_document[doc_id], key=lambda p: p.position)
            chosen.extend(rng.sample(passages,
                                     min(per_document, len(passages))))

        rng.shuffle(chosen)
        return chosen[:budget]

    def generate(self, progress: bool = True) -> TestCollection:
        """
        Build the collection. Passages whose generation fails are skipped.

        One model call per passage. At the ~250ms warm latency measured on
        this machine, 100 passages takes a few minutes, and the result is
        saved so it is paid once.
        """
        passages = self.sample_passages()
        queries: list[EvalQuery] = []
        judgements: list[Judgement] = []
        failures = 0

        for index, passage in enumerate(passages, start=1):
            if progress and index % 10 == 0:
                print(f"  [collection] {index}/{len(passages)} passages, "
                      f"{len(queries)} queries, {failures} failures")
            try:
                generated = self._call(passage.text)
            except Exception as error:
                failures += 1
                if progress:
                    print(f"  [collection] call failed on "
                          f"{passage.passage_id}: {error}")
                continue

            if not generated:
                failures += 1
                continue

            for style in ('literal', 'paraphrased'):
                text = generated.get(style, '').strip()
                if len(text.split()) < 3:
                    continue
                query_id = f"q{len(queries):04d}_{style[:3]}"
                queries.append(EvalQuery(
                    query_id=query_id, text=text, source='synthetic',
                    style=style, source_passage_id=passage.passage_id,
                    doc_id=passage.doc_id))
                judgements.extend(self._judgements_for(query_id, passage))

        if progress:
            print(f"  [collection] done: {len(queries)} queries from "
                  f"{len(passages) - failures}/{len(passages)} passages")
        return TestCollection(queries, judgements)


def manual_collection(corpus, pairs: list[tuple[str, str]]) -> TestCollection:
    """
    Build a small collection from hand-written (query, passage_id) pairs.

    For sanity checks and for the hand-verified subset that validates the
    synthetic collection. Overlap partial credit is applied the same way, so
    manual and synthetic judgements are directly comparable.
    """
    generator = SyntheticQueryGenerator(corpus)
    queries, judgements = [], []
    for index, (text, passage_id) in enumerate(pairs):
        passage = corpus.get(passage_id)
        if passage is None:
            raise KeyError(f"unknown passage id {passage_id!r}")
        query_id = f"m{index:03d}"
        queries.append(EvalQuery(query_id=query_id, text=text, source='manual',
                                 style='manual',
                                 source_passage_id=passage_id,
                                 doc_id=passage.doc_id))
        judgements.extend(generator._judgements_for(query_id, passage))
    return TestCollection(queries, judgements)
