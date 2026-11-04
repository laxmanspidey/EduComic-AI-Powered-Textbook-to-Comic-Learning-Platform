"""
Run the retrieval experiment and print the results tables.

Deliberately Django-free: it builds the corpus straight from the PDFs, so it
starts instantly and can be re-run without a database. The pipelines it
constructs are the same ones `core/ir_service.py` serves to the application,
so the numbers here describe the system the app actually uses.

Usage
-----
    # first run: generate the test collection, then evaluate (needs Ollama)
    venv/Scripts/python.exe tools/run_experiments.py --generate --passages 60

    # later runs: reuse the saved collection (no Ollama needed)
    venv/Scripts/python.exe tools/run_experiments.py

    # skip the slow systems while iterating
    venv/Scripts/python.exe tools/run_experiments.py --no-rerank

Output goes to the terminal and to ir/data/runs/<label>_<timestamp>.json,
which the IR analytics page reads.

Systems evaluated
-----------------
    tfidf                     classical vector space baseline
    bm25                      classical probabilistic baseline
    dense                     neural bi-encoder
    hybrid_rrf                BM25 + dense, reciprocal rank fusion
    hybrid_combsum            BM25 + dense, normalised score fusion
    hybrid_rrf_rm3            + pseudo-relevance feedback expansion
    hybrid_rrf_rerank         + cross-encoder reranking
    full                      + expansion + reranking + intent routing

The baseline is TF-IDF, so every improvement is measured against a classical
method rather than against another neural one.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.corpus import build_from_pdfs                       # noqa: E402
from ir.eval.collection import SyntheticQueryGenerator, TestCollection  # noqa: E402
from ir.eval.experiments import (                           # noqa: E402
    PRIMARY_MEASURE,
    ablation_table,
    by_style_table,
    comparison_table,
    run_experiment,
    run_system,
    save_results,
    significance_table,
)
from ir.indexes.bm25 import BM25Index                       # noqa: E402
from ir.indexes.dense import DenseIndex, SentenceTransformerEmbedder  # noqa: E402
from ir.indexes.tfidf import TFIDFIndex                     # noqa: E402
from ir.pipeline import Pipeline                            # noqa: E402
from ir.query.expand import RM3Expander                     # noqa: E402
from ir.rerank import CrossEncoderReranker                  # noqa: E402

PDF_GLOB = 'media/textbooks/*.pdf'


def build_pipelines(corpus, with_rerank: bool = True) -> list[Pipeline]:
    """Construct every system to be compared, baseline first."""
    documents = corpus.documents()
    tfidf = TFIDFIndex().fit(documents)
    bm25 = BM25Index().fit(documents)
    dense = DenseIndex(SentenceTransformerEmbedder(device='cpu')).fit(documents)

    pipelines = [
        Pipeline([tfidf], corpus, name='tfidf'),
        Pipeline([bm25], corpus, name='bm25'),
        Pipeline([dense], corpus, name='dense'),
        Pipeline([bm25, dense], corpus, fusion_method='rrf',
                 name='hybrid_rrf'),
        Pipeline([bm25, dense], corpus, fusion_method='combsum',
                 name='hybrid_combsum'),
        Pipeline([bm25, dense], corpus, fusion_method='rrf',
                 expander=RM3Expander(bm25, documents),
                 name='hybrid_rrf_rm3'),
    ]

    if with_rerank:
        reranker = CrossEncoderReranker(device='cpu', top_n=20)
        pipelines.append(
            Pipeline([bm25, dense], corpus, fusion_method='rrf',
                     reranker=reranker, name='hybrid_rrf_rerank'))
        pipelines.append(
            Pipeline([bm25, dense], corpus, fusion_method='rrf',
                     expander=RM3Expander(bm25, documents),
                     reranker=reranker, intent_aware=True, name='full'))

    return pipelines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true',
                        help='generate a fresh test collection (needs Ollama)')
    parser.add_argument('--passages', type=int, default=60,
                        help='passages to generate queries from (2 each)')
    parser.add_argument('--model', default='qwen3.5:4b',
                        help='Ollama model for query generation')
    parser.add_argument('--no-rerank', action='store_true',
                        help='skip the cross-encoder systems (much faster)')
    parser.add_argument('--label', default='experiment',
                        help='label for the saved run file')
    args = parser.parse_args(argv)

    # ── corpus ────────────────────────────────────────────────────────────
    pdfs = sorted(glob.glob(PDF_GLOB))
    if not pdfs:
        print(f"No PDFs found at {PDF_GLOB}")
        return 1

    print("Building corpus...")
    corpus = build_from_pdfs(pdfs)
    print(corpus.stats.report())
    print()

    if not len(corpus):
        print("Corpus is empty; nothing to evaluate.")
        return 1

    if len(corpus) < 200:
        print(f"  NOTE: {len(corpus)} passages is a small collection. A top-10 "
              f"result is {10 / len(corpus):.0%} of it, so systems will be "
              f"hard to\n        separate and significance tests will rarely "
              f"fire. Add more short PDFs to\n        media/textbooks/ and "
              f"re-run. Check them first with tools/check_pdfs.py.\n")

    # ── collection ────────────────────────────────────────────────────────
    if args.generate:
        print(f"Generating queries from {args.passages} passages "
              f"using {args.model}...")
        generator = SyntheticQueryGenerator(
            corpus, model=args.model, n_passages=args.passages)
        collection = generator.generate()
        collection.save()
        print(f"Saved to ir/data/collection/\n")
    else:
        collection = TestCollection.load()
        if not len(collection):
            print("No test collection found. Run once with --generate "
                  "(requires Ollama).")
            return 1

    print(collection.stats().report())
    print()

    # ── run ───────────────────────────────────────────────────────────────
    pipelines = build_pipelines(corpus, with_rerank=not args.no_rerank)

    # Warm every pipeline before timing anything.
    #
    # sentence-transformers and the cross-encoder load lazily on first use, so
    # without this the first neural system absorbs the whole model-load cost
    # and reports a latency several times its real one. Measured before adding
    # this: 'dense' showed 313 ms/query while 'hybrid_rrf' -- which also
    # embeds the query -- showed 7 ms, purely because 'dense' ran first.
    print("Warming models (excluded from reported latency)...")
    for pipeline in pipelines:
        pipeline.search('warmup query for model loading')

    print(f"Evaluating {len(pipelines)} systems on {len(collection)} "
          f"queries...")
    results, significance = run_experiment(
        pipelines, collection, baseline='tfidf')

    # ── report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("SYSTEM COMPARISON")
    print("=" * 74)
    print(comparison_table(results, significance, baseline='tfidf'))

    print("\n" + "=" * 74)
    print("SIGNIFICANCE vs TF-IDF baseline")
    print("=" * 74)
    print(significance_table(significance))

    full_name = 'full' if 'full' in results else 'hybrid_rrf'
    print("\n" + "=" * 74)
    print(f"ABLATION (relative to {full_name})")
    print("=" * 74)
    print(ablation_table(results, full_name))

    # Literal vs paraphrased: the terminology-mismatch measurement.
    styles = {q.style for q in collection.queries}
    if {'literal', 'paraphrased'} <= styles:
        print("\n" + "=" * 74)
        print("BY QUERY STYLE  (terminology mismatch)")
        print("=" * 74)
        by_style: dict[str, dict] = {}
        for style in ('literal', 'paraphrased'):
            subset = collection.by_style(style)
            by_style[style] = {}
            for pipeline in pipelines:
                result, _ = run_system(pipeline, subset, progress=False)
                by_style[style][pipeline.name] = result
        print(by_style_table(by_style))
        print("\nA wide positive gap means the system depends on the student "
              "using the\ntextbook's own terminology. Narrow gaps are the "
              "point of semantic retrieval\nand query expansion.")

    path = save_results(results, significance, label=args.label)
    print(f"\nSaved: {path}")
    print("View it in the app at /retrieval/")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
