"""
Query intent classification, and the retrieval policy each intent implies.

Rule-based on purpose. Educational questions are highly templated -- "what
is", "difference between", "how does ... work", "why does" -- so surface
patterns classify them accurately, and the whole thing costs microseconds. An
LLM classifier here would add 250ms per query and a failure mode, to replace
regexes that already match the question forms a textbook search actually
receives.

Intent matters because different question types want different retrieval:

  DEFINITION   "what is electrostatic shielding"
               The exact term is the answer's anchor. Lexical matching is
               strong here, and a semantically-similar passage that never
               names the term is usually worse than one that does.
               -> lean lexical.

  COMPARISON   "difference between consistency and availability"
               Needs evidence about *both* entities, which a single query may
               not find: one entity often dominates the ranking. Handled by
               splitting into per-entity queries and fusing, so each side is
               guaranteed representation.
               -> split, then fuse.

  CAUSAL       "why does entropy increase"
               The answer is explanatory prose which rarely repeats the
               question's words. This is where vocabulary mismatch bites
               hardest.
               -> lean semantic, expand the query.

  PROCESS      "how does RSA key generation work"
               The answer spans consecutive passages -- step 1 here, step 2
               in the next chunk.
               -> widen the context window.

  NUMERICAL    "formula for capacitance in parallel"
               Symbols and identifiers are exact tokens an embedding blurs.
               -> lean lexical.

  EXAMPLE      "give examples of land degradation"
               -> lean semantic.

  FACTUAL      anything unmatched. Balanced defaults.

The weights below are a starting policy, not a tuned result. They are
configuration precisely so the experiment can measure whether intent-aware
routing beats a single fixed configuration -- which is the actual research
question, and may well come out negative.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    DEFINITION = 'definition'
    COMPARISON = 'comparison'
    CAUSAL = 'causal'
    PROCESS = 'process'
    NUMERICAL = 'numerical'
    EXAMPLE = 'example'
    FACTUAL = 'factual'


@dataclass
class RoutingPolicy:
    """
    What an intent implies for the pipeline.

    Attributes:
        retriever_weights: per-retriever multiplier handed to fusion. Values
                           above 1.0 favour that retriever.
        expand:            whether query expansion is worth its cost here.
        rerank_depth:      how many candidates to rerank. Deeper costs more
                           (52ms per pair) but lets the reranker rescue
                           passages fusion buried.
        context_window:    how many neighbouring passages to attach.
        split_entities:    run one query per detected entity and fuse.
    """
    retriever_weights: dict[str, float] = field(default_factory=dict)
    expand: bool = True
    rerank_depth: int = 20
    context_window: int = 1
    split_entities: bool = False


#: Default policy per intent. Tunable, and deliberately so.
POLICIES: dict[Intent, RoutingPolicy] = {
    Intent.DEFINITION: RoutingPolicy({'bm25': 1.5, 'dense': 1.0},
                                     expand=False, rerank_depth=20),
    Intent.COMPARISON: RoutingPolicy({'bm25': 1.0, 'dense': 1.0},
                                     expand=True, rerank_depth=30,
                                     split_entities=True),
    Intent.CAUSAL: RoutingPolicy({'bm25': 1.0, 'dense': 1.5},
                                 expand=True, rerank_depth=20),
    Intent.PROCESS: RoutingPolicy({'bm25': 1.0, 'dense': 1.2},
                                  expand=True, rerank_depth=20,
                                  context_window=2),
    Intent.NUMERICAL: RoutingPolicy({'bm25': 2.0, 'dense': 1.0},
                                    expand=False, rerank_depth=10),
    Intent.EXAMPLE: RoutingPolicy({'bm25': 1.0, 'dense': 1.3},
                                  expand=True, rerank_depth=20),
    Intent.FACTUAL: RoutingPolicy({'bm25': 1.0, 'dense': 1.0},
                                  expand=True, rerank_depth=20),
}

# ──────────────────────────────────────────────────────────────────────────────
# Patterns
#
# Order matters: COMPARISON is tested before DEFINITION because "what is the
# difference between X and Y" matches both, and the comparison reading is the
# more specific one.
# ──────────────────────────────────────────────────────────────────────────────

_COMPARISON = re.compile(
    r'\b(difference|differences|differ|compare|comparison|contrast|'
    r'versus|vs\.?|distinguish|distinction|better than|'
    r'similarities|similarity|advantages? over)\b', re.IGNORECASE)

_COMPARISON_SPLIT = re.compile(
    r'\b(?:between|among)\s+(.+?)\s+(?:and|or|vs\.?|versus)\s+(.+?)'
    r'(?:\s*[?.]|$)', re.IGNORECASE)

_COMPARISON_SPLIT_BARE = re.compile(
    r'^(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\s*[?.]|$)', re.IGNORECASE)

_NUMERICAL = re.compile(
    r'\b(formula|equation|calculate|calculation|compute|how much|how many|'
    r'value of|derive|derivation|units? of|expression for|solve)\b',
    re.IGNORECASE)

_PROCESS = re.compile(
    r'\b(how (?:does|do|is|are|can|to)|steps?|procedure|process|mechanism|'
    r'algorithm|stages?|work(?:s|ing)?\b.*\?|method for)\b', re.IGNORECASE)

_CAUSAL = re.compile(
    r'\b(why|reason|reasons|because|cause|causes|caused by|leads? to|'
    r'results? in|due to|explain why|what happens (?:when|if))\b',
    re.IGNORECASE)

_EXAMPLE = re.compile(
    r'\b(examples?|instances?|such as|for instance|illustrate|'
    r'give.*examples?|list.*examples?|types? of|kinds? of)\b', re.IGNORECASE)

_DEFINITION = re.compile(
    r'\b(what (?:is|are|was|were)|define|definition|meaning of|'
    r'what does .* mean|who (?:is|was)|terminology|refers? to|'
    r'stands? for)\b', re.IGNORECASE)

#: Words stripped from an extracted comparison entity.
_ENTITY_NOISE = re.compile(
    r'^(?:the|a|an|is|are|was|were|of|in|on|for|to|and|or)\s+|'
    r'\s+(?:the|a|an|is|are|was|were|of|in|on|for|to|and|or)$',
    re.IGNORECASE)


def classify(query: str) -> Intent:
    """
    Classify a query by surface form.

    Falls back to FACTUAL rather than guessing, because a wrong policy is
    worse than a neutral one: routing a definition query to the semantic arm
    costs exactly the precision the lexical arm was there to provide.
    """
    if not query or not query.strip():
        return Intent.FACTUAL

    # Most specific first.
    if _COMPARISON.search(query):
        return Intent.COMPARISON
    if _NUMERICAL.search(query):
        return Intent.NUMERICAL
    if _EXAMPLE.search(query):
        return Intent.EXAMPLE
    if _PROCESS.search(query):
        return Intent.PROCESS
    if _CAUSAL.search(query):
        return Intent.CAUSAL
    if _DEFINITION.search(query):
        return Intent.DEFINITION
    return Intent.FACTUAL


def policy_for(intent: Intent) -> RoutingPolicy:
    """The routing policy for an intent, FACTUAL's if unknown."""
    return POLICIES.get(intent, POLICIES[Intent.FACTUAL])


def _clean_entity(text: str) -> str:
    """Trim articles and trailing function words from an extracted entity."""
    cleaned = text.strip().strip('?.,;:')
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = _ENTITY_NOISE.sub('', cleaned).strip()
    return cleaned


def comparison_entities(query: str) -> list[str]:
    """
    Pull the two things being compared out of a comparison query.

    "difference between consistency and availability"
        -> ['consistency', 'availability']

    Returns [] when the pattern does not match, which is common: "compare
    these two protocols" names nothing extractable. The caller must treat an
    empty list as "do not split" rather than as an error, since a single
    combined query is a perfectly good fallback.
    """
    for pattern in (_COMPARISON_SPLIT, _COMPARISON_SPLIT_BARE):
        match = pattern.search(query)
        if match:
            entities = [_clean_entity(match.group(1)),
                        _clean_entity(match.group(2))]
            entities = [e for e in entities if len(e) >= 2]
            if len(entities) == 2:
                return entities
    return []


@dataclass
class ClassifiedQuery:
    """A query with its intent, policy, and any extracted entities."""
    query: str
    intent: Intent
    policy: RoutingPolicy
    entities: list[str] = field(default_factory=list)

    @property
    def should_split(self) -> bool:
        """Split only when the policy asks *and* entities were found."""
        return self.policy.split_entities and len(self.entities) >= 2

    def subqueries(self) -> list[str]:
        """
        The queries to actually run.

        For a splittable comparison this is the original plus one query per
        entity: the original can still find a passage that compares them
        directly, which is the ideal answer, while the per-entity queries
        guarantee each side is represented even if no such passage exists.
        """
        if not self.should_split:
            return [self.query]
        return [self.query, *self.entities]


def analyse(query: str) -> ClassifiedQuery:
    """Classify a query and attach its policy and entities."""
    intent = classify(query)
    entities = comparison_entities(query) if intent is Intent.COMPARISON else []
    return ClassifiedQuery(query=query, intent=intent,
                           policy=policy_for(intent), entities=entities)
