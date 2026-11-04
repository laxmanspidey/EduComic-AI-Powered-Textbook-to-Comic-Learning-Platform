"""
Significance testing for retrieval system comparisons.

Reporting "nDCG@10 went from 0.84 to 0.87" says nothing on its own: with 50
queries that difference can easily be noise. Retrieval experiments are paired
-- every system runs the same queries -- so the right tests are paired tests
over the per-query measure values that `metrics.evaluate_run` returns.

Two tests are provided:

  randomisation_test   (primary)
      A paired permutation test. Under the null hypothesis the two systems are
      equivalent, so for each query the labelling of its two scores is
      arbitrary. We repeatedly flip those labels at random and count how often
      the resulting mean difference is at least as extreme as the observed
      one. It assumes nothing about the distribution of scores, which matters
      because IR measures are bounded, skewed and frequently zero-inflated --
      exactly where the t-test's normality assumption is weakest.

      Smucker, Allan & Carterette (2007), "A comparison of statistical
      significance tests for information retrieval evaluation", CIKM.

  paired_t_test        (secondary, requires scipy)
      Reported alongside for familiarity. Kept optional so this module has no
      hard third-party dependency.

Multiple comparisons: comparing one baseline against five variants means five
tests, and at alpha = 0.05 the chance of at least one false positive is about
23%. Use `holm_correction` (preferred) or `bonferroni_correction` and say in
the write-up which one was applied.
"""
from __future__ import annotations

import random

# ──────────────────────────────────────────────────────────────────────────────
# Paired randomisation (permutation) test
# ──────────────────────────────────────────────────────────────────────────────

def randomisation_test(scores_a: dict[str, float], scores_b: dict[str, float],
                       n_permutations: int = 10_000,
                       seed: int | None = 42,
                       two_tailed: bool = True) -> dict[str, float]:
    """
    Paired two-sided randomisation test on per-query scores.

    Args:
        scores_a, scores_b: query_id -> measure value, for the two systems.
                            Only query ids present in both are used.
        n_permutations:     number of random label swaps. 10,000 gives a
                            p-value resolution of 1e-4, which is ample for a
                            0.05 or 0.01 threshold.
        seed:               fixed by default so reported p-values reproduce.
        two_tailed:         compare |difference|; set False for a directional
                            hypothesis.

    Returns:
        {'mean_a', 'mean_b', 'delta', 'p_value', 'n_queries', 'n_permutations'}
        where `delta` is mean_a - mean_b.

    The p-value uses the standard (hits + 1) / (permutations + 1) estimator,
    which cannot return exactly 0 -- honest, since a finite sample of
    permutations can never prove p = 0.
    """
    shared = sorted(set(scores_a) & set(scores_b))
    n = len(shared)
    if n == 0:
        raise ValueError("no query ids in common between the two runs")

    diffs = [scores_a[q] - scores_b[q] for q in shared]
    observed = sum(diffs) / n

    rng = random.Random(seed)
    target = abs(observed) if two_tailed else observed

    hits = 0
    for _ in range(n_permutations):
        total = 0.0
        for d in diffs:
            # Swapping the two systems' labels for a query flips the sign of
            # that query's difference.
            total += d if rng.random() < 0.5 else -d
        permuted = total / n
        if (abs(permuted) if two_tailed else permuted) >= target:
            hits += 1

    return {
        'mean_a': sum(scores_a[q] for q in shared) / n,
        'mean_b': sum(scores_b[q] for q in shared) / n,
        'delta': observed,
        'p_value': (hits + 1) / (n_permutations + 1),
        'n_queries': float(n),
        'n_permutations': float(n_permutations),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Paired t-test (optional, scipy)
# ──────────────────────────────────────────────────────────────────────────────

def paired_t_test(scores_a: dict[str, float],
                  scores_b: dict[str, float]) -> dict[str, float]:
    """
    Two-tailed paired t-test on per-query scores.

    Requires scipy. Raises RuntimeError if it is unavailable, so the caller can
    fall back to `randomisation_test` alone.

    Returns {'mean_a', 'mean_b', 'delta', 't_statistic', 'p_value', 'n_queries'}.
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "paired_t_test requires scipy; use randomisation_test instead"
        ) from exc

    shared = sorted(set(scores_a) & set(scores_b))
    if len(shared) < 2:
        raise ValueError("need at least 2 shared queries for a t-test")

    a = [scores_a[q] for q in shared]
    b = [scores_b[q] for q in shared]
    t_stat, p_value = scipy_stats.ttest_rel(a, b)

    return {
        'mean_a': sum(a) / len(a),
        'mean_b': sum(b) / len(b),
        'delta': (sum(a) - sum(b)) / len(a),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'n_queries': float(len(shared)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multiple-comparison corrections
# ──────────────────────────────────────────────────────────────────────────────

def bonferroni_correction(p_values: dict[str, float],
                          alpha: float = 0.05) -> dict[str, dict[str, float]]:
    """
    Bonferroni: reject when p <= alpha / m, for m tests.

    Simple and always valid, but conservative -- it costs real statistical
    power when m is large. Prefer `holm_correction`.

    Returns comparison name -> {'p_value', 'adjusted_alpha', 'significant'}.
    """
    m = len(p_values)
    if m == 0:
        return {}
    adjusted = alpha / m
    return {
        name: {
            'p_value': p,
            'adjusted_alpha': adjusted,
            'significant': float(p <= adjusted),
        }
        for name, p in p_values.items()
    }


def holm_correction(p_values: dict[str, float],
                    alpha: float = 0.05) -> dict[str, dict[str, float]]:
    """
    Holm-Bonferroni step-down correction.

    Sort the m p-values ascending and compare the i-th (1-indexed) against
    alpha / (m - i + 1). Stop at the first non-rejection; everything after it
    is also non-significant. Uniformly more powerful than Bonferroni while
    controlling the same family-wise error rate.

    Returns comparison name -> {'p_value', 'adjusted_alpha', 'rank',
    'significant'}.
    """
    m = len(p_values)
    if m == 0:
        return {}

    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    out: dict[str, dict[str, float]] = {}
    still_rejecting = True

    for i, (name, p) in enumerate(ordered, start=1):
        threshold = alpha / (m - i + 1)
        if still_rejecting and p > threshold:
            # First failure ends the step-down; nothing below can be rejected.
            still_rejecting = False
        out[name] = {
            'p_value': p,
            'adjusted_alpha': threshold,
            'rank': float(i),
            'significant': float(still_rejecting),
        }

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: compare many systems against one baseline
# ──────────────────────────────────────────────────────────────────────────────

def compare_against_baseline(
    per_query: dict[str, dict[str, dict[str, float]]],
    baseline: str,
    measure: str = 'nDCG@10',
    alpha: float = 0.05,
    n_permutations: int = 10_000,
    seed: int | None = 42,
) -> dict[str, dict[str, float]]:
    """
    Run a randomisation test for every system against one baseline, then apply
    the Holm correction across that family of tests.

    Args:
        per_query: system name -> query_id -> measure dict, i.e. the second
                   element of `metrics.evaluate_run` collected per system.
        baseline:  key in `per_query` to compare everything else against.
        measure:   which measure to test, e.g. 'nDCG@10' or 'MRR'.

    Returns:
        system name -> {'mean', 'baseline_mean', 'delta', 'p_value',
        'adjusted_alpha', 'significant'}. The baseline itself is omitted.

    Testing one measure chosen in advance is deliberate. Running the tests on
    every measure and reporting whichever reaches significance is p-hacking;
    name the primary measure before looking at the results.
    """
    if baseline not in per_query:
        raise KeyError(f"baseline {baseline!r} not in per_query: "
                       f"{sorted(per_query)}")

    def column(system: str) -> dict[str, float]:
        return {qid: m[measure] for qid, m in per_query[system].items()}

    base_col = column(baseline)

    results: dict[str, dict[str, float]] = {}
    raw_p: dict[str, float] = {}
    for system in per_query:
        if system == baseline:
            continue
        r = randomisation_test(column(system), base_col,
                               n_permutations=n_permutations, seed=seed)
        results[system] = {
            'mean': r['mean_a'],
            'baseline_mean': r['mean_b'],
            'delta': r['delta'],
            'p_value': r['p_value'],
        }
        raw_p[system] = r['p_value']

    corrected = holm_correction(raw_p, alpha=alpha)
    for system, c in corrected.items():
        results[system]['adjusted_alpha'] = c['adjusted_alpha']
        results[system]['significant'] = c['significant']

    return results
