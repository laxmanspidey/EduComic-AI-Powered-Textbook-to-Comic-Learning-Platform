"""
Conversational query rewriting.

The failure: a student asks "what is entropy", gets an answer, then asks "why
does it increase?". Sent to a retriever on its own, the second query carries
almost no information -- "increase" and a pronoun. Increase *what*? The
conversation holds the referent; the query does not.

Rewriting resolves the query against the history into a standalone form:

    history : "what is entropy"
    current : "why does it increase?"
    rewritten: "why does entropy increase?"

Design decision worth noting: a rule-based gate runs *before* the model.
Rewriting is only attempted when the query actually looks dependent -- it
contains a pronoun, is very short, or begins with a bare interrogative. This
matters for two reasons beyond the 250ms saved per query:

  1. Rewriting a self-contained query can corrupt it. A model asked to
     "rewrite using context" will often inject context that was not needed,
     narrowing a general question into whatever the previous turn was about.
  2. Most queries in a session are self-contained, so gating turns the LLM
     call from per-query into occasional.

The gate is conservative in the safe direction: a false positive costs one
model call and usually returns the query unchanged, while a false negative
leaves a dependent query unresolved. So it errs toward rewriting.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'rewrite'

#: Pronouns and demonstratives that usually point at a previous turn.
_DEPENDENT_MARKERS = re.compile(
    r'\b(it|its|it\'s|this|that|these|those|they|them|their|he|she|his|her|'
    r'the former|the latter|the same|such|one)\b', re.IGNORECASE)

#: A query starting with a bare interrogative and little else is elliptical:
#: "why?", "how so", "and then?".
_BARE_INTERROGATIVE = re.compile(
    r'^\s*(why|how|when|where|what about|and|so|then|but)\b', re.IGNORECASE)

#: Below this many tokens a follow-up is almost certainly elliptical.
SHORT_QUERY_TOKENS = 5


@dataclass
class ConversationTurn:
    """One exchange. `answer` is optional but improves referent resolution."""
    query: str
    answer: str = ''


@dataclass
class RewrittenQuery:
    """
    The result of a rewrite attempt.

    `rewritten` always holds a usable query: the original when no rewrite was
    attempted or the attempt failed. `changed` says whether it differs, which
    is what the trace and the UI display.
    """
    original: str
    rewritten: str
    changed: bool = False
    reason: str = ''
    history_used: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.rewritten


def looks_dependent(query: str) -> tuple[bool, str]:
    """
    Decide whether `query` appears to depend on conversation history.

    Returns (dependent, reason). The reason is recorded in the trace so a
    rewrite is always explainable -- "rewritten because it contains the
    pronoun 'it'" is auditable in a way "the model decided to" is not.
    """
    stripped = (query or '').strip()
    if not stripped:
        return False, 'empty query'

    tokens = stripped.split()

    marker = _DEPENDENT_MARKERS.search(stripped)
    if marker:
        return True, f"contains referent {marker.group(0)!r}"

    if len(tokens) <= SHORT_QUERY_TOKENS and _BARE_INTERROGATIVE.match(stripped):
        return True, f"short interrogative ({len(tokens)} tokens)"

    if len(tokens) <= 2:
        return True, f"very short ({len(tokens)} tokens)"

    return False, 'appears self-contained'


REWRITE_PROMPT = """Rewrite the student's latest question so it stands alone, \
using the conversation for context.

Conversation so far:
{history}

Latest question: {query}

Rules:
- Replace pronouns and references with what they actually refer to
- Keep the student's intent and question type exactly
- Add nothing that was not asked
- If the question already stands alone, return it unchanged
- Return only the rewritten question, no explanation, no quotes"""


class ConversationalRewriter:
    """
    LLM query rewriting behind a rule-based gate.

    Args:
        model:       Ollama model name.
        max_history: how many previous turns to include. Two is usually
                     enough: referents almost always point at the immediately
                     preceding turn, and a longer window invites the model to
                     drag in stale topics.
        include_answers: feed previous answers as well as questions. Helps
                     resolve a referent introduced by the answer rather than
                     the question, at the cost of a longer prompt.
    """

    name = 'conversational'

    def __init__(self, model: str = 'qwen3.5:4b',
                 base_url: str = 'http://127.0.0.1:11434',
                 max_history: int = 2, include_answers: bool = True,
                 answer_chars: int = 240, temperature: float = 0.1,
                 timeout: int = 60, cache_dir: Path | None = None,
                 use_cache: bool = True):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.max_history = max_history
        self.include_answers = include_answers
        self.answer_chars = answer_chars
        self.temperature = temperature
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.use_cache = use_cache

    # ── history rendering ─────────────────────────────────────────────────
    def _format_history(self, history: list[ConversationTurn]) -> str:
        recent = history[-self.max_history:] if self.max_history else []
        lines = []
        for turn in recent:
            lines.append(f"Student: {turn.query}")
            if self.include_answers and turn.answer:
                lines.append(f"System: {turn.answer[:self.answer_chars]}")
        return '\n'.join(lines)

    # ── cache ─────────────────────────────────────────────────────────────
    def _cache_path(self, query: str, history_text: str) -> Path:
        digest = hashlib.sha256(
            f"{self.model}|{history_text}|{query}".encode('utf-8')
        ).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, query: str, history_text: str) -> str | None:
        if not self.use_cache:
            return None
        path = self._cache_path(query, history_text)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload.get('query') == query:
                return payload.get('rewritten')
        except Exception:
            return None
        return None

    def _write_cache(self, query: str, history_text: str, rewritten: str) -> None:
        if not self.use_cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(query, history_text).write_text(
            json.dumps({'query': query, 'rewritten': rewritten,
                        'model': self.model}), encoding='utf-8')

    # ── model call ────────────────────────────────────────────────────────
    def _call(self, query: str, history_text: str) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                'model': self.model,
                'messages': [{'role': 'user', 'content': REWRITE_PROMPT.format(
                    history=history_text, query=query)}],
                'stream': False,
                'think': False,
                'options': {'temperature': self.temperature,
                            'num_predict': 120},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json().get('message', {}).get('content', '')
        return self._clean(content)

    @staticmethod
    def _clean(content: str) -> str:
        """
        Extract a bare question from the model's reply.

        Strips leaked reasoning blocks, surrounding quotes, and any
        "Rewritten question:" preamble a small model tends to add. Takes the
        first non-empty line, since extra lines are commentary rather than
        query.
        """
        content = re.sub(r'<think>[\s\S]*?</think>', '', content,
                         flags=re.IGNORECASE).strip()
        content = re.sub(r'^(rewritten|standalone)\s*(question|query)?\s*:\s*',
                         '', content, flags=re.IGNORECASE).strip()
        for line in content.splitlines():
            line = line.strip().strip('"').strip("'").strip()
            if line:
                return line
        return ''

    # ── rewriting ─────────────────────────────────────────────────────────
    def rewrite(self, query: str,
                history: list[ConversationTurn] | None = None
                ) -> RewrittenQuery:
        """
        Rewrite `query` against `history` if it looks dependent.

        Never raises: a model failure returns the original query, so a dead
        Ollama degrades retrieval quality rather than breaking search.
        """
        history = history or []

        if not history:
            return RewrittenQuery(original=query, rewritten=query,
                                  reason='no conversation history')

        dependent, reason = looks_dependent(query)
        if not dependent:
            return RewrittenQuery(original=query, rewritten=query,
                                  reason=reason)

        history_text = self._format_history(history)
        if not history_text:
            return RewrittenQuery(original=query, rewritten=query,
                                  reason='no usable history')

        rewritten = self._read_cache(query, history_text)
        if rewritten is None:
            try:
                rewritten = self._call(query, history_text)
                self._write_cache(query, history_text, rewritten)
            except Exception:
                return RewrittenQuery(
                    original=query, rewritten=query,
                    reason=f'{reason}; rewrite unavailable')

        if not rewritten:
            return RewrittenQuery(original=query, rewritten=query,
                                  reason=f'{reason}; model returned nothing')

        # Guard against a model that returns something unusably long: a
        # rewrite should resolve referents, not write an essay.
        if len(rewritten.split()) > max(40, len(query.split()) * 6):
            return RewrittenQuery(original=query, rewritten=query,
                                  reason=f'{reason}; rewrite rejected as too long')

        changed = rewritten.strip().lower() != query.strip().lower()
        return RewrittenQuery(
            original=query, rewritten=rewritten, changed=changed,
            reason=reason,
            history_used=[t.query for t in history[-self.max_history:]])


class NoOpRewriter:
    """
    Passes queries through untouched.

    Lets "pipeline without rewriting" be a configuration rather than a branch,
    so the ablation row runs the same code path.
    """

    name = 'none'

    def rewrite(self, query: str,
                history: list[ConversationTurn] | None = None
                ) -> RewrittenQuery:
        return RewrittenQuery(original=query, rewritten=query,
                              reason='rewriting disabled')
