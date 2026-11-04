"""
Running retrieval experiments and reporting them.

An experiment is: a set of pipelines, one test collection, one results table
with significance tests. Because every pipeline is the same code with
different components, a row of the table and an ablation of the full system
are the same operation -- which is what keeps the comparison honest.

What gets reported, and what does not
-------------------------------------
With a synthetic collection there is roughly one relevant passage per query.
Under that condition several standard measures become mathematically
identical: Recall@k equals Success@k, and MAP equals MRR (with one relevant
document, AP = 1/rank = RR). Printing them as separate columns would show
duplicate numbers and look like a bug, so `PRIMARY_MEASURES` reports MRR and
nDCG@k and omits the redundant ones. `metrics.evaluate_run` still computes
everything, and the full set is written to the run file.

Significance
------------
A difference in mean nDCG is not a result. Every comparison against the
baseline runs a paired randomisation test over per-query scores, with Holm
correction across the family of comparisons, and the table marks what
survived. The primary measure is fixed in advance rather than chosen after
looking at the numbers.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..metrics import evaluate_run
from ..stats import compare_against_baseline

RUNS_DIR = Path(__file__).resolve().parent.parent / 'data' / 'runs'

#: Measures shown in the comparison table. Deliberately short: see module
#: docstring on why Recall@k and MAP are omitted for a one-relevant-passage
#: collection.
PRIMARY_MEASURES = ('nDCG@10', 'nDCG@5', 'MRR', 'P@1', 'Success@5')

#: The measure significance is tested on. Chosen before running anything.
PRIMARY_MEASURE = 'nDCG@10'


@dataclass
class SystemResult:
    """One pipeline's results on one collection."""
    name: str
    config: dict
    means: dict[str, float]
    per_query: dict[str, dict[str, float]]
    ranking: dict[str, list[str]] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    n_queries: int = 0

    @property
    def mean_latency_ms(self) -> float:
        if not self.latency_ms:
            return 0.0
        return sum(self.latency_ms.values()) / len(self.latency_ms)

    def measure(self, name: str) -> float:
        return self.means.get(name, 0.0)


def run_system(pipeline, collection, progress: bool = True,
               capture_traces: bool = False
               ) -> tuple[SystemResult, list]:
    """
    Run one pipeline over every query in `collection`.

    Returns (SystemResult, traces). Traces are only retained when asked for,
    because a trace holds passage text and keeping hundreds of them costs
    far more memory than the numbers do.
    """
    qrels = collection.qrels()
    run: dict[str, list[str]] = {}
    latency: dict[str, float] = {}
    traces = []

    total = len(collection.queries)
    for index, query in enumerate(collection.queries, start=1):
        if progress and index % 25 == 0:
            print(f"    [{pipeline.name}] {index}/{total}")
        start = time.perf_counter()
        trace = pipeline.search(query.text)
        latency[query.query_id] = (time.perf_counter() - start) * 1000
        run[query.query_id] = trace.ranking()
        if capture_traces:
            traces.append(trace)

    means, per_query = evaluate_run(run, qrels)
    result = SystemResult(
        name=pipeline.name, config=pipeline.describe(), means=means,
        per_query=per_query, ranking=run, latency_ms=latency,
        n_queries=len(per_query))
    return result, traces


def run_experiment(pipelines: list, collection, baseline: str | None = None,
                   progress: bool = True
                   ) -> tuple[dict[str, SystemResult], dict]:
    """
    Run several pipelines and test each against a baseline.

    Args:
        pipelines: configured `Pipeline` objects.
        collection: a `TestCollection`.
        baseline:  which system to compare against. Defaults to the first,
                   which should therefore be the weakest -- usually TF-IDF or
                   BM25, so improvements are measured against a classical
                   method rather than against another neural one.

    Returns:
        (results, significance). `significance` maps system name to the
        randomisation-test outcome on PRIMARY_MEASURE, Holm-corrected.
    """
    if not pipelines:
        raise ValueError("no pipelines to run")

    results: dict[str, SystemResult] = {}
    for pipeline in pipelines:
        if progress:
            print(f"  running {pipeline.name}")
        result, _ = run_system(pipeline, collection, progress=progress)
        results[pipeline.name] = result

    baseline = baseline or pipelines[0].name
    significance: dict = {}
    if baseline in results and len(results) > 1:
        per_query = {name: r.per_query for name, r in results.items()}
        try:
            significance = compare_against_baseline(
                per_query, baseline, measure=PRIMARY_MEASURE)
        except (KeyError, ValueError):
            significance = {}

    return results, significance


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def comparison_table(results: dict[str, SystemResult],
                     significance: dict | None = None,
                     measures: tuple[str, ...] = PRIMARY_MEASURES,
                     baseline: str | None = None) -> str:
    """
    Markdown comparison table, ready to paste into a report.

    A dagger marks a system whose improvement over the baseline survived the
    Holm-corrected randomisation test on PRIMARY_MEASURE. Systems that
    improved the mean but did not reach significance are shown without one --
    which is the honest presentation, since with a small collection most
    small gains will not.
    """
    if not results:
        return "_no results_"

    significance = significance or {}
    names = list(results)
    header = ['System', *measures, 'ms/query']
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']

    for name in names:
        result = results[name]
        cells = [name]
        for measure in measures:
            value = result.measure(measure)
            cell = f"{value:.3f}"
            if measure == PRIMARY_MEASURE:
                entry = significance.get(name)
                if entry and entry.get('significant'):
                    cell += ' †'
                if baseline and name == baseline:
                    cell += ' *'
            cells.append(cell)
        cells.append(f"{result.mean_latency_ms:.0f}")
        lines.append('| ' + ' | '.join(cells) + ' |')

    lines.append('')
    lines.append(f"n = {results[names[0]].n_queries} queries. "
                 f"† significant vs baseline on {PRIMARY_MEASURE} "
                 f"(paired randomisation test, Holm-corrected, alpha 0.05).")
    if baseline:
        lines.append(f"* = baseline ({baseline}).")
    return '\n'.join(lines)


def significance_table(significance: dict,
                       measure: str = PRIMARY_MEASURE) -> str:
    """Per-comparison detail: delta, p-value, corrected threshold."""
    if not significance:
        return "_no comparisons_"

    lines = [f'| System | {measure} | baseline | delta | p | alpha* | sig |',
             '|---|---|---|---|---|---|---|']
    for name, entry in sorted(significance.items(),
                              key=lambda kv: -kv[1].get('delta', 0)):
        lines.append(
            f"| {name} | {entry.get('mean', 0):.3f} | "
            f"{entry.get('baseline_mean', 0):.3f} | "
            f"{entry.get('delta', 0):+.3f} | "
            f"{entry.get('p_value', 1):.4f} | "
            f"{entry.get('adjusted_alpha', 0):.4f} | "
            f"{'yes' if entry.get('significant') else 'no'} |")
    return '\n'.join(lines)


def ablation_table(results: dict[str, SystemResult], full_system: str,
                   measure: str = PRIMARY_MEASURE) -> str:
    """
    Ablation view: each row's loss relative to the full system.

    Reads from the same results dict as the comparison table, because an
    ablation *is* a comparison -- the rows just happen to be the full system
    minus one component each.
    """
    if full_system not in results:
        return f"_unknown system {full_system!r}_"

    reference = results[full_system].measure(measure)
    lines = [f'| Configuration | {measure} | vs full |',
             '|---|---|---|']
    lines.append(f"| {full_system} (full) | {reference:.3f} | — |")

    for name, result in results.items():
        if name == full_system:
            continue
        value = result.measure(measure)
        delta = value - reference
        lines.append(f"| {name} | {value:.3f} | {delta:+.3f} |")
    return '\n'.join(lines)


def by_style_table(results_by_style: dict[str, dict[str, SystemResult]],
                   measure: str = PRIMARY_MEASURE) -> str:
    """
    Systems by query style -- the terminology-mismatch result.

    Expects {'literal': {...}, 'paraphrased': {...}}. The gap between the two
    columns is the measurement of how much each method depends on the student
    already knowing the textbook's vocabulary. A lexical system should show a
    wide gap and a semantic one a narrow gap; if the gaps are equal, the
    terminology-mismatch premise does not hold on this collection and the
    write-up should say so.
    """
    styles = list(results_by_style)
    if not styles:
        return "_no results_"

    systems: list[str] = []
    for style in styles:
        for name in results_by_style[style]:
            if name not in systems:
                systems.append(name)

    lines = ['| System | ' + ' | '.join(styles) + ' | gap |',
             '|' + '|'.join(['---'] * (len(styles) + 2)) + '|']
    for name in systems:
        values = []
        for style in styles:
            result = results_by_style[style].get(name)
            values.append(result.measure(measure) if result else 0.0)
        gap = (values[0] - values[-1]) if len(values) > 1 else 0.0
        lines.append(f"| {name} | " +
                     ' | '.join(f"{v:.3f}" for v in values) +
                     f" | {gap:+.3f} |")
    lines.append('')
    lines.append(f"{measure} by query style. Gap = first column minus last; "
                 f"a large positive gap means the system depends on the "
                 f"student using the textbook's own terminology.")
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_results(results: dict[str, SystemResult], significance: dict,
                 directory: str | Path = RUNS_DIR,
                 label: str = '') -> Path:
    """
    Write a run to disk: one JSON file per experiment.

    Holds per-query scores and full rankings, not just means, so a reported
    number can be re-derived and a significance test re-run without
    re-executing the pipelines.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f"{label + '_' if label else ''}{stamp}.json"
    path = directory / name

    payload = {
        'timestamp': stamp,
        'label': label,
        'primary_measure': PRIMARY_MEASURE,
        'systems': {
            system_name: {
                'config': result.config,
                'means': result.means,
                'per_query': result.per_query,
                'ranking': result.ranking,
                'latency_ms': result.latency_ms,
                'n_queries': result.n_queries,
            }
            for system_name, result in results.items()
        },
        'significance': significance,
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


def load_results(path: str | Path) -> tuple[dict[str, SystemResult], dict]:
    """Read a saved run back into SystemResult objects."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    results = {
        name: SystemResult(
            name=name, config=entry.get('config', {}),
            means=entry.get('means', {}),
            per_query=entry.get('per_query', {}),
            ranking=entry.get('ranking', {}),
            latency_ms=entry.get('latency_ms', {}),
            n_queries=entry.get('n_queries', 0))
        for name, entry in payload.get('systems', {}).items()
    }
    return results, payload.get('significance', {})


def latest_results(directory: str | Path = RUNS_DIR
                   ) -> tuple[dict[str, SystemResult], dict] | None:
    """
    Most recent saved run, or None. Used by the analytics dashboard.

    Ordered by the timestamp inside the payload, falling back to file mtime.
    Sorting filenames would order by the run's label first -- a run labelled
    'demo' would outrank a later one labelled 'clean' -- and silently show
    stale numbers on the dashboard.
    """
    directory = Path(directory)
    if not directory.exists():
        return None

    files = list(directory.glob('*.json'))
    if not files:
        return None

    def recency(path: Path) -> tuple[str, float]:
        try:
            stamp = json.loads(path.read_text(encoding='utf-8')).get('timestamp', '')
        except Exception:
            stamp = ''
        return (str(stamp), path.stat().st_mtime)

    return load_results(max(files, key=recency))
