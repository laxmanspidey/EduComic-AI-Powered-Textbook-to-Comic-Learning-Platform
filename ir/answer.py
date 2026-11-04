"""
Answer generation over retrieved passages.

Retrieval returns ranked evidence; a student asked a question. This module
closes that gap by passing the top passages to a local LLM and asking for an
answer grounded in them.

Three properties matter here, and they are what separate this from a chatbot
bolted onto a search box.

Grounding
    The model is given numbered passages and told to answer from those alone.
    Every claim should trace to a passage, and the answer carries inline [n]
    markers that resolve back to passage ids. Without that link an answer is
    unfalsifiable, and the retrieval quality measured elsewhere would say
    nothing about what the student actually reads.

Abstention
    If the retrieved passages do not answer the question, the correct output
    is "this textbook does not cover that", not a fluent guess. A generator
    that always answers hides retrieval failures -- the system looks fine
    while returning invented content, which is worse than returning nothing.
    `answerable` is therefore part of the model's response, not inferred.

Attribution is checked, not trusted
    Citation markers the model emits are validated against the passages it was
    actually given. A marker pointing at a passage that was not supplied is
    dropped rather than displayed, because a fabricated citation is more
    damaging than no citation.

The context comes from `ir.context.ContextAssembler`, so the model sees
de-overlapped passages with their surrounding block rather than isolated
fragments -- the "This process increases the value by 20%" problem applies to
generation at least as much as to reading.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / 'data' / 'answers'

#: Passages fed to the generator. Beyond about five the context dilutes and a
#: small model starts answering from the wrong one; below three it misses
#: answers split across adjacent passages.
DEFAULT_N_PASSAGES = 4

#: Word budget for the assembled context. qwen3.5:4b handles this comfortably
#: and it leaves room for the question and the answer.
DEFAULT_CONTEXT_WORDS = 900


@dataclass
class Citation:
    """One passage the answer drew on."""
    marker: int
    passage_id: str
    citation: str = ''
    breadcrumb: str = ''
    snippet: str = ''
    concept_id: str = ''


@dataclass
class Answer:
    """
    A generated answer with its provenance.

    `grounded` is False when the model abstained, when generation failed, or
    when nothing was retrieved. The UI must distinguish that from a confident
    answer, so the field is explicit rather than inferred from the text.
    """
    text: str
    query: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    model: str = ''
    latency_ms: float = 0.0
    context_passages: list[str] = field(default_factory=list)
    error: str = ''

    @property
    def cited_passage_ids(self) -> list[str]:
        return [c.passage_id for c in self.citations]

    @property
    def uncited(self) -> bool:
        """
        A grounded answer with no citations is suspicious.

        It means the model produced prose without pointing at evidence, which
        is the shape a hallucination takes even when the passages were
        relevant. Worth surfacing in the UI rather than hiding.
        """
        return self.grounded and not self.citations

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'query': self.query,
            'grounded': self.grounded,
            'model': self.model,
            'latency_ms': self.latency_ms,
            'error': self.error,
            'citations': [
                {'marker': c.marker, 'passage_id': c.passage_id,
                 'citation': c.citation, 'breadcrumb': c.breadcrumb,
                 'snippet': c.snippet, 'concept_id': c.concept_id}
                for c in self.citations
            ],
            'context_passages': self.context_passages,
        }


ANSWER_PROMPT = """You are helping a student understand their textbook. \
Answer their question using ONLY the numbered passages below.

{passages}

Question: {query}

Rules:
- Answer only from the passages above. Do not add outside knowledge.
- Cite the passages you used with inline markers like [1] or [2].
- If the passages do not contain the answer, set "answerable" to false and \
explain in one sentence what is missing. Do not guess.
- Write 2 to 5 sentences, in plain language a student can follow.
- Do not mention "the passages" or "the context" in your answer; just explain \
the topic.

Return ONLY a JSON object:
{{"answerable": true, "answer": "Your explanation with [1] markers.", \
"citations": [1, 2]}}"""


class AnswerGenerator:
    """
    Generates grounded answers with a local Ollama model.

    Args:
        model:          Ollama model name.
        n_passages:     how many retrieved passages to supply.
        context_words:  word budget for the assembled context.
        temperature:    low by default; this is an extraction-and-explanation
                        task, not a creative one, and a higher temperature
                        mostly buys ungrounded elaboration.
        cache_dir:      answers are cached by (model, query, passage ids), so
                        re-running a demo or an evaluation does not pay for
                        the same generation twice. The passage ids are part of
                        the key, so a change in retrieval produces a fresh
                        answer rather than a stale one.
    """

    def __init__(self, model: str = 'qwen3.5:4b',
                 base_url: str = 'http://127.0.0.1:11434',
                 n_passages: int = DEFAULT_N_PASSAGES,
                 context_words: int = DEFAULT_CONTEXT_WORDS,
                 temperature: float = 0.2, timeout: int = 120,
                 cache_dir: Path | None = None, use_cache: bool = True,
                 max_tokens: int = 400):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.n_passages = n_passages
        self.context_words = context_words
        self.temperature = temperature
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.use_cache = use_cache
        self.max_tokens = max_tokens

    # ── cache ─────────────────────────────────────────────────────────────
    def _cache_path(self, query: str, passage_ids: list[str]) -> Path:
        digest = hashlib.sha256(
            f"{self.model}|{query}|{'|'.join(passage_ids)}".encode('utf-8')
        ).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, query: str, passage_ids: list[str]) -> dict | None:
        if not self.use_cache:
            return None
        path = self._cache_path(query, passage_ids)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            return payload if payload.get('query') == query else None
        except Exception:
            return None

    def _write_cache(self, query: str, passage_ids: list[str],
                     payload: dict) -> None:
        if not self.use_cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(query, passage_ids).write_text(
            json.dumps(payload), encoding='utf-8')

    # ── model ─────────────────────────────────────────────────────────────
    def _call(self, prompt: str) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False,
                'think': False,
                'options': {'temperature': self.temperature,
                            'num_predict': self.max_tokens},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get('message', {}).get('content', '')

    @staticmethod
    def _parse(content: str) -> dict:
        """
        Extract the answer object from the model's reply.

        Same defensive ladder as the rest of the package: strip any leaked
        reasoning block, try a fenced object, then a bare object, then the
        outermost braces. Falls back to treating the whole reply as prose,
        because a usable answer in the wrong envelope is still a usable
        answer -- it just loses its explicit abstention flag, so it is marked
        answerable and left for the citation check to assess.
        """
        content = re.sub(r'<think>[\s\S]*?</think>', '', content,
                         flags=re.IGNORECASE).strip()
        if not content:
            return {}

        fenced = re.search(r'```(?:json)?\s*([\s\S]+?)```', content,
                           re.IGNORECASE)
        candidates = []
        if fenced:
            candidates.append(fenced.group(1).strip())
        candidates.append(content)
        brace = re.search(r'(\{[\s\S]+\})', content)
        if brace:
            candidates.append(brace.group(1))

        for text in candidates:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and 'answer' in parsed:
                return parsed

        # Unparseable JSON but non-empty prose: keep the prose.
        return {'answerable': True, 'answer': content, 'citations': []}

    # ── prompt construction ───────────────────────────────────────────────
    @staticmethod
    def _format_passages(blocks: list[tuple[int, str, str]]) -> str:
        """Render numbered passages, each with its source line."""
        parts = []
        for marker, citation, text in blocks:
            header = f"[{marker}]" + (f" ({citation})" if citation else "")
            parts.append(f"{header}\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _markers_in(text: str) -> list[int]:
        """Citation markers actually present in the answer text."""
        return [int(m) for m in re.findall(r'\[(\d{1,2})\]', text)]

    # ── generation ────────────────────────────────────────────────────────
    def generate(self, query: str, results: list, assembler=None) -> Answer:
        """
        Generate an answer from retrieval results.

        Args:
            query:     the question, already rewritten if rewriting is on.
            results:   `ScoredPassage` objects from a `RetrievalTrace`.
            assembler: optional `ContextAssembler`. When given, each passage is
                       expanded to its surrounding block so the model sees
                       whole paragraphs; without it the scored passage text is
                       used as-is.

        Never raises: a model failure returns an ungrounded Answer carrying
        the error, so a dead Ollama degrades the page to search-only rather
        than breaking it.
        """
        start = time.perf_counter()

        if not results:
            return Answer(
                text="Nothing in this textbook matched your question, so "
                     "there is no evidence to answer from.",
                query=query, grounded=False, model=self.model,
                latency_ms=(time.perf_counter() - start) * 1000)

        selected = results[:self.n_passages]
        passage_ids = [r.passage_id for r in selected]

        # Build the numbered context, respecting the word budget.
        blocks: list[tuple[int, str, str]] = []
        lookup: dict[int, object] = {}
        used_words = 0
        for marker, result in enumerate(selected, start=1):
            text = ''
            if assembler is not None:
                assembled = assembler.assemble(result.passage_id)
                if assembled is not None:
                    text = assembled.context_text
            text = text or result.text
            words = text.split()
            remaining = self.context_words - used_words
            if remaining <= 40:
                break
            if len(words) > remaining:
                text = ' '.join(words[:remaining])
            used_words += min(len(words), remaining)
            blocks.append((marker, result.breadcrumb or result.citation, text))
            lookup[marker] = result

        if not blocks:
            return Answer(text="No usable passage text was retrieved.",
                          query=query, grounded=False, model=self.model,
                          latency_ms=(time.perf_counter() - start) * 1000)

        cached = self._read_cache(query, passage_ids)
        if cached is not None:
            payload = cached.get('payload', {})
        else:
            prompt = ANSWER_PROMPT.format(
                passages=self._format_passages(blocks), query=query)
            try:
                payload = self._parse(self._call(prompt))
            except Exception as error:
                return Answer(
                    text="The answer model is unavailable, so only the "
                         "retrieved passages are shown below.",
                    query=query, grounded=False, model=self.model,
                    error=str(error), context_passages=passage_ids,
                    latency_ms=(time.perf_counter() - start) * 1000)
            self._write_cache(query, passage_ids,
                              {'query': query, 'payload': payload})

        answer_text = str(payload.get('answer', '')).strip()
        if not answer_text:
            return Answer(
                text="The answer model returned nothing usable. The retrieved "
                     "passages are shown below.",
                query=query, grounded=False, model=self.model,
                context_passages=passage_ids,
                latency_ms=(time.perf_counter() - start) * 1000)

        answerable = bool(payload.get('answerable', True))

        # Validate citations against what the model was actually given. A
        # marker for a passage that was never supplied is dropped: a
        # fabricated citation is worse than none, because it looks verified.
        claimed = payload.get('citations')
        markers = [int(m) for m in claimed if str(m).isdigit()] \
            if isinstance(claimed, list) else []
        markers += self._markers_in(answer_text)

        citations: list[Citation] = []
        for marker in sorted(set(markers)):
            result = lookup.get(marker)
            if result is None:
                continue
            citations.append(Citation(
                marker=marker,
                passage_id=result.passage_id,
                citation=result.citation,
                breadcrumb=result.breadcrumb,
                snippet=result.text[:220],
            ))

        return Answer(
            text=answer_text, query=query, citations=citations,
            grounded=answerable, model=self.model,
            context_passages=passage_ids,
            latency_ms=(time.perf_counter() - start) * 1000)
