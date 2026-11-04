"""
Query expansion: adding terms a student did not type.

The failure this addresses, observed on this project's own corpus: the query
"why is farmland becoming useless" retrieves almost nothing lexically, because
the textbook says *land degradation*, *wasteland* and *barren* and never says
"farmland" or "useless". No amount of better ranking helps when the terms are
simply absent -- the query itself has to change.

Two expanders, deliberately from different traditions:

  RM3Expander   Pseudo-relevance feedback. Retrieve once, assume the top few
                passages are relevant, harvest their distinctive terms, and
                search again with those added. No LLM, no network, no training
                -- just term statistics. Microseconds per query, and it is the
                canonical IR answer to this problem.

                Lavrenko & Croft (2001), "Relevance-Based Language Models",
                SIGIR; RM3's interpolated form from Abdul-Jaleel et al. (2004),
                "UMass at TREC 2004".

  LLMExpander   Ask a language model for the terminology a textbook would use.
                Has knowledge RM3 cannot derive from the collection, but costs
                a model call and can hallucinate terms that appear nowhere.

Having both turns "does an LLM beat classical PRF for terminology mismatch?"
into a measurable comparison rather than an assumption. RM3 is the honest
baseline the LLM has to beat.

A caution that matters for interpreting results: expansion helps *lexical*
retrieval, where an absent term is fatal. Appending synonyms to a sentence
before embedding it can just as easily blur the dense vector and hurt. The
pipeline therefore decides per-retriever which query to send, and the
experiment should report the two separately.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..text import tokenize

#: Where LLM expansion results are cached.
CACHE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'expansion'


@dataclass
class ExpandedQuery:
    """
    The result of expanding a query.

    Carries both representations because the retrievers need different things:
    `weights` for BM25, which can score a weighted query exactly, and
    `as_text()` for anything that only accepts a string.
    """
    original: str
    weights: dict[str, float] = field(default_factory=dict)
    added_terms: list[str] = field(default_factory=list)
    method: str = 'none'

    def as_text(self) -> str:
        """
        The expanded query as a plain string.

        Term weights are lost here -- a string has no way to carry them. Use
        `weights` with `BM25Index.search_weighted` whenever weighting matters.
        """
        return ' '.join([self.original, *self.added_terms]).strip()

    @property
    def original_terms(self) -> list[str]:
        return tokenize(self.original)

    def __bool__(self) -> bool:
        """True when expansion actually added something."""
        return bool(self.added_terms)


def identity_expansion(query: str) -> ExpandedQuery:
    """
    No expansion: the original query, weighted by its own term counts.

    Used for the un-expanded rows of the results table, so that path runs the
    same weighted-scoring code as the expanded one and any difference is
    attributable to the added terms alone.
    """
    counts: dict[str, float] = {}
    for term in tokenize(query):
        counts[term] = counts.get(term, 0.0) + 1.0
    return ExpandedQuery(original=query, weights=counts, method='none')


# ──────────────────────────────────────────────────────────────────────────────
# RM3 pseudo-relevance feedback
# ──────────────────────────────────────────────────────────────────────────────

class RM3Expander:
    """
    RM3 relevance-model expansion over a BM25 index.

    The procedure:

      1. Run the original query, take the top `fb_docs` passages. These are
         *assumed* relevant -- nobody judged them. That assumption is the
         method's single weakness: if the first retrieval is bad, expansion
         amplifies the error. This is called query drift, and it is why
         `alpha` keeps most of the weight on what the student actually typed.

      2. Estimate a relevance model over terms:

             p(w | R) = sum over feedback docs d of  p(d | q) * p(w | d)

         with p(w|d) = tf(w,d) / |d| and p(d|q) obtained by normalising the
         retrieval scores so they sum to 1.

      3. Interpolate with the original query model:

             p_final(w) = alpha * p(w | q) + (1 - alpha) * p(w | R)

         which is the "3" in RM3 -- RM1 omits this step and drifts badly.

    Args:
        index:     a fitted BM25Index, used both to retrieve feedback
                   passages and for its term statistics.
        documents: passage id -> text, needed to read feedback passage
                   contents.
        fb_docs:   how many top passages to treat as relevant. 10 is standard;
                   fewer is noisier, more drifts further off-topic.
        fb_terms:  how many expansion terms to keep.
        alpha:     weight retained by the original query, in [0, 1]. 1.0
                   disables expansion entirely.
        use_idf:   multiply relevance-model weights by IDF, so terms that are
                   frequent in the feedback set *and* rare in the collection
                   are preferred. Without it, expansion mostly adds common
                   words that were never discriminative.
    """

    method = 'rm3'

    def __init__(self, index, documents: dict[str, str], fb_docs: int = 10,
                 fb_terms: int = 10, alpha: float = 0.6, use_idf: bool = True,
                 min_term_length: int = 3):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if fb_docs < 1:
            raise ValueError(f"fb_docs must be >= 1, got {fb_docs}")
        if fb_terms < 0:
            raise ValueError(f"fb_terms must be >= 0, got {fb_terms}")

        self.index = index
        self.documents = documents
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms
        self.alpha = alpha
        self.use_idf = use_idf
        self.min_term_length = min_term_length

    def _relevance_model(self, feedback: list[tuple[str, float]]
                         ) -> dict[str, float]:
        """Estimate p(w | R) from the feedback passages."""
        total = sum(score for _, score in feedback)
        if total <= 0:
            return {}

        model: dict[str, float] = {}
        for doc_id, score in feedback:
            doc_probability = score / total
            text = self.documents.get(doc_id, '')
            terms = tokenize(text, self.index.remove_stopwords, self.index.stem)
            if not terms:
                continue

            counts: dict[str, int] = {}
            for term in terms:
                counts[term] = counts.get(term, 0) + 1

            length = len(terms)
            for term, tf in counts.items():
                if len(term) < self.min_term_length:
                    continue
                model[term] = model.get(term, 0.0) + doc_probability * (tf / length)

        if self.use_idf:
            for term in list(model):
                model[term] *= self.index.idf.get(term, 0.0)

        return model

    def expand(self, query: str) -> ExpandedQuery:
        """
        Expand `query`. Returns the original unchanged if nothing is retrieved
        or `alpha` is 1.0.
        """
        original = tokenize(query, self.index.remove_stopwords, self.index.stem)
        if not original:
            return ExpandedQuery(original=query, weights={}, method=self.method)

        original_counts: dict[str, float] = {}
        for term in original:
            original_counts[term] = original_counts.get(term, 0.0) + 1.0
        query_model = {t: c / len(original) for t, c in original_counts.items()}

        if self.alpha >= 1.0 or self.fb_terms == 0:
            return ExpandedQuery(original=query, weights=original_counts,
                                 method=self.method)

        feedback = self.index.search_terms(original, top_k=self.fb_docs)
        if not feedback:
            # Nothing retrieved means nothing to learn from. Returning the
            # original query is the correct behaviour -- and note this is
            # exactly the "farmland" case, where lexical retrieval found
            # almost nothing, so RM3 cannot rescue it either. Only the LLM
            # expander can, because its knowledge comes from outside the
            # collection.
            return ExpandedQuery(original=query, weights=original_counts,
                                 method=self.method)

        model = self._relevance_model(feedback)

        # Keep the best expansion terms that are not already in the query.
        candidates = sorted(
            ((term, weight) for term, weight in model.items()
             if term not in original_counts and weight > 0),
            key=lambda pair: (-pair[1], pair[0]),
        )[:self.fb_terms]

        if not candidates:
            return ExpandedQuery(original=query, weights=original_counts,
                                 method=self.method)

        expansion_mass = sum(weight for _, weight in candidates)
        weights = {term: self.alpha * probability
                   for term, probability in query_model.items()}
        for term, weight in candidates:
            weights[term] = (1.0 - self.alpha) * (weight / expansion_mass)

        return ExpandedQuery(
            original=query,
            weights=weights,
            added_terms=[term for term, _ in candidates],
            method=self.method,
        )


# ──────────────────────────────────────────────────────────────────────────────
# LLM expansion
# ──────────────────────────────────────────────────────────────────────────────

#: Asks for terminology, not a rewritten question. A rewritten question would
#: be a different operation (see query/rewrite.py) and would not help a bag-of
#: -words index at all.
EXPANSION_PROMPT = """A student searching a textbook typed this query:

"{query}"

List the technical terms, synonyms and domain vocabulary that the textbook
itself would most likely use for this topic, which the student did not write.
For example, for "why is farmland becoming useless" a textbook would say:
land degradation, soil erosion, desertification, wasteland, barren.

Rules:
- Return 5 to 10 single words or short phrases
- Only terminology an actual textbook would contain
- Do not repeat words already in the query
- No explanations

Return ONLY a JSON array of strings, nothing else."""


class LLMExpander:
    """
    Expansion using a local LLM through Ollama.

    Unlike RM3 this does not depend on the first retrieval succeeding, so it
    can rescue a query whose terms are entirely absent from the collection --
    the one case RM3 provably cannot handle. The trade is a model call, the
    possibility of terms that appear nowhere in the corpus (harmless, they
    simply score nothing), and non-determinism unless temperature is low.

    Results are cached to disk keyed by model and query. Expansion is called
    repeatedly across experiment runs for the same query set, and a warm
    Ollama call costs ~250ms on this machine -- cheap once, wasteful a
    thousand times.
    """

    method = 'llm'

    def __init__(self, model: str = 'qwen3.5:4b',
                 base_url: str = 'http://127.0.0.1:11434',
                 max_terms: int = 10, temperature: float = 0.2,
                 timeout: int = 60, cache_dir: Path | None = None,
                 use_cache: bool = True, weight: float = 0.4):
        """
        Args:
            max_terms:   cap on added terms.
            temperature: low, because expansion should be reproducible across
                         experiment runs.
            weight:      total mass given to expansion terms, mirroring
                         (1 - alpha) in RM3 so the two are comparable.
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.max_terms = max_terms
        self.temperature = temperature
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.use_cache = use_cache
        self.weight = weight

    # ── cache ─────────────────────────────────────────────────────────────
    def _cache_path(self, query: str) -> Path:
        digest = hashlib.sha256(
            f"{self.model}|{self.max_terms}|{query}".encode('utf-8')
        ).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, query: str) -> list[str] | None:
        if not self.use_cache:
            return None
        path = self._cache_path(query)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload.get('query') == query:
                return list(payload.get('terms', []))
        except Exception:
            return None
        return None

    def _write_cache(self, query: str, terms: list[str]) -> None:
        if not self.use_cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(query).write_text(
            json.dumps({'query': query, 'model': self.model, 'terms': terms}),
            encoding='utf-8')

    # ── model call ────────────────────────────────────────────────────────
    def _call(self, query: str) -> list[str]:
        import requests

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                'model': self.model,
                'messages': [{'role': 'user',
                              'content': EXPANSION_PROMPT.format(query=query)}],
                'stream': False,
                'think': False,
                'options': {'temperature': self.temperature,
                            'num_predict': 300},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json().get('message', {}).get('content', '')
        return self._parse(content)

    @staticmethod
    def _parse(content: str) -> list[str]:
        """
        Pull a term list out of the model's reply.

        Reuses the defensive approach from agents/llm_client.py: strip any
        leaked reasoning block, then try a fenced JSON array, then a bare
        array. Returns [] rather than raising, because a failed expansion
        should degrade to the un-expanded query and not break retrieval.
        """
        content = re.sub(r'<think>[\s\S]*?</think>', '', content,
                         flags=re.IGNORECASE).strip()

        fenced = re.search(r'```(?:json)?\s*([\s\S]+?)```', content,
                           re.IGNORECASE)
        candidate = fenced.group(1).strip() if fenced else content

        for text in (candidate, content):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r'(\[[\s\S]+?\])', text)
                if not match:
                    continue
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        return []

    # ── expansion ─────────────────────────────────────────────────────────
    def expand(self, query: str) -> ExpandedQuery:
        original = tokenize(query)
        original_counts: dict[str, float] = {}
        for term in original:
            original_counts[term] = original_counts.get(term, 0.0) + 1.0

        raw_terms = self._read_cache(query)
        if raw_terms is None:
            try:
                raw_terms = self._call(query)
                self._write_cache(query, raw_terms)
            except Exception:
                # Ollama down, timeout, or unparseable output: fall back to
                # the original query so retrieval still works.
                return ExpandedQuery(original=query, weights=original_counts,
                                     method=self.method)

        # A returned phrase may tokenise to several terms; keep them all, and
        # drop anything already in the query.
        added: list[str] = []
        for phrase in raw_terms:
            for term in tokenize(phrase):
                if term not in original_counts and term not in added:
                    added.append(term)
        added = added[:self.max_terms]

        if not added:
            return ExpandedQuery(original=query, weights=original_counts,
                                 method=self.method)

        if original:
            weights = {term: (1.0 - self.weight) * (count / len(original))
                       for term, count in original_counts.items()}
        else:
            weights = {}
        per_term = self.weight / len(added)
        for term in added:
            weights[term] = per_term

        return ExpandedQuery(original=query, weights=weights,
                             added_terms=added, method=self.method)
