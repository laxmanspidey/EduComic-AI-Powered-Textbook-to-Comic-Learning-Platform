"""
Text normalisation shared by the sparse indexes.

Keeping tokenisation in one place means TF-IDF and BM25 see exactly the same
token stream, so a difference between them is a difference in *weighting*, not
in preprocessing. It also makes preprocessing itself ablatable: stopword
removal and stemming are flags, so "BM25 without stemming" is a row in the
results table rather than a separate code path.
"""
from __future__ import annotations

import re

#: A compact English stopword list. Deliberately short: aggressive stopping
#: hurts phrase-like textbook queries ("law of inertia" -> "law inertia"), and
#: BM25's IDF term already discounts frequent words. Kept as a flag so the
#: effect can be measured instead of assumed.
STOPWORDS: frozenset[str] = frozenset("""
a an the and or but if then than that this these those of in on at to for from
by with about into over after before between out against during without within
is are was were be been being am do does did doing have has had having
it its itself he she they them their his her our your my we you i as so such
no nor not only own same too very can will just should now what which who whom
when where why how all any both each few more most other some
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")


def _light_stem(token: str) -> str:
    """
    Minimal suffix stripping: plurals and common verb endings.

    This is NOT a Porter stemmer. It handles the cases that matter most for
    matching a student's query against textbook prose ("capacitors" ->
    "capacitor", "charges" -> "charge") without the aggressive conflation that
    makes Porter output unreadable in a retrieval-explanation UI. Swap in
    nltk.PorterStemmer here if a standard algorithm is wanted; the signature
    is all the rest of the package depends on.
    """
    if len(token) <= 3:
        return token
    for suffix, keep in (('ies', 'y'), ('sses', 'ss'), ('ing', ''),
                         ('ed', ''), ('es', ''), ('s', '')):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[:len(token) - len(suffix)] + keep
            # Avoid producing a bare consonant cluster, e.g. "bring" -> "br".
            if any(c in 'aeiou' for c in stem):
                return stem
            return token
    return token


def tokenize(text: str, remove_stopwords: bool = True,
             stem: bool = True, min_length: int = 2) -> list[str]:
    """
    Lowercase, split into alphanumeric tokens, then optionally stop and stem.

    Args:
        text:             raw passage or query text.
        remove_stopwords: drop tokens in STOPWORDS.
        stem:             apply `_light_stem`.
        min_length:       discard tokens shorter than this (after stemming).
                          2 keeps useful short tokens such as "pv" and "ac"
                          while dropping stray single characters from PDF
                          extraction noise.

    Returns:
        List of tokens, in order, with duplicates retained. Term frequency is
        the point of a sparse index, so the caller counts; we do not dedupe.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [_light_stem(t) for t in tokens]
    return [t for t in tokens if len(t) >= min_length]
