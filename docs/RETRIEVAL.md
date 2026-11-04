# EduRetrieve — the information retrieval layer

A multi-stage hybrid retrieval framework over textbook PDFs. The comic and
quiz generators are applications built on top of it.

## Why this exists

The original pipeline had exactly one retrieval component: cosine similarity
in ChromaDB with a `distance < 0.85` cutoff. That is a single dense retriever
with no evaluation, which means there was no way to answer "does retrieval
work?" other than by looking at generated output and forming an impression.

Eight concrete failures motivated the eight components. Each one below was
observed on this project's own corpus, not taken from a paper.

| Problem | Observed | Component |
|---|---|---|
| 1. Terminology mismatch | `"why does a classifier do poorly on fresh examples"` shares no stem with the target passage; BM25 returns **nothing** | Query expansion (RM3 + LLM) |
| 2. Keyword blindness | `"what does CAP mean"` — the dense retriever returned three physics **capacitor** passages in the top 4 | BM25, hybrid fusion |
| 3. Semantic overranking | Topically adjacent passages outrank answering ones | Cross-encoder reranking |
| 4. Context loss | A passage reading *"This process increases the value by 20%"* is useless alone | Parent-child expansion, breadcrumbs |
| 5. Ambiguous follow-ups | `"why does it increase?"` carries no referent | Conversational rewriting |
| 6. No evaluation | No way to answer "how do you know?" | Test collection, metrics, significance tests |
| 7. No classical baseline | Dense-only; nothing to compare against | TF-IDF, BM25, hybrid, ablations |
| 8. Unexplainable results | Only "it had the highest similarity" | `RetrievalTrace`, per-term breakdowns |

## Architecture

```
PDF → passages                       ir/corpus.py
   ↓
indexes:  TF-IDF · BM25 · Dense      ir/indexes/
   ↓
query understanding                  ir/query/
   rewrite → classify → expand
   ↓
retrieve → fuse (RRF)                ir/fusion.py
   ↓
cross-encoder rerank                 ir/rerank.py
   ↓
context assembly                     ir/context.py
   ↓
Ask-the-textbook · Comic · Quiz
```

Composed by `ir/pipeline.py`. Every system is the *same code* with different
components, so a results-table row and an ablation are the same operation:

```python
Pipeline([bm25], corpus)                                    # BM25
Pipeline([bm25, dense], corpus)                             # hybrid, RRF
Pipeline([bm25, dense], corpus, reranker=ce)                # + reranking
Pipeline([bm25, dense], corpus, reranker=ce, expander=rm3)  # full
```

## Key design decisions

**Indexing is decoupled from chapter detection.** The original pipeline
chunked *inside* detected chapter ranges, so pages no chapter covered were
never indexed. On the Class 10 Social Science PDF, detection produced three
chapters all claiming page 4, and chapter titles like `", Table 6.1.]"`. The
corpus silently shrank and duplicated. Now the whole document is chunked and
chapter labels are attached afterwards as advisory metadata.

**The `Concept` model is the section layer, not `Chapter`.** Chapter detection
is unreliable; concepts are LLM-extracted subtopics and are also the unit the
app already builds comics and quizzes around. So a search result can say
"covered in subtopic X" and link straight to its comic.

**Search is scoped to one textbook.** Measured: a whole-library index answers
`"what does CAP mean"` with 2 physics passages in the top 5; scoping to the
one textbook gives 0. Cross-domain acronym collision becomes structurally
impossible rather than something ranking has to fix.

**Chunking spans page boundaries.** Page-bounded chunking cuts paragraphs in
half at every page break. Both modes are implemented so it is an ablation
(`mode='continuous'` vs `'page_bounded'`), not an assumption.

**RRF rather than score addition.** BM25 is unbounded; cosine is in [-1,1].
Adding them lets whichever has larger variance dominate, and the mix shifts
per query. RRF fuses *ranks*, so it is scale-free. `test_fusion.py`
demonstrates the failure: multiplying BM25 scores by 100 reorders CombSUM and
leaves RRF untouched.

**Everything runs on CPU.** The GPU is already shared between Ollama and
ComfyUI on an 8 GB card. Embedding runs at 55 passages/s and the
cross-encoder at 52 ms/pair, so CPU is not the bottleneck.

## Measured costs

On an RTX 4060 Laptop (8 GB) with 16 GB RAM, all CPU:

| Operation | Cost |
|---|---|
| PDF → passages | <1s |
| BM25 index build | <0.1s |
| BM25 query | ~1ms |
| Embed corpus | 55 passages/s, cached by content |
| Dense query | ~18ms |
| Cross-encoder | 52ms/pair → ~1s at top-20 |
| Ollama warm call | ~250ms (50s cold start — batch LLM work) |

A full sweep of 8 systems over 200 queries runs in minutes, dominated by the
reranker.

## Running the experiment

```bash
# first run: generate the test collection (needs Ollama), then evaluate
venv/Scripts/python.exe tools/run_experiments.py --generate --passages 60

# later runs reuse the saved collection, no Ollama needed
venv/Scripts/python.exe tools/run_experiments.py

# iterate faster by skipping the cross-encoder systems
venv/Scripts/python.exe tools/run_experiments.py --no-rerank
```

Results print to the terminal and save to `ir/data/runs/`, which the
`/retrieval/` page reads.

## The test collection

Queries are generated synthetically: an LLM reads a passage and writes a
question it answers, so the source passage is relevant by construction.

**Two styles per passage**, which is what makes it an experiment:

- `literal` — uses the passage's own terminology
- `paraphrased` — deliberately avoids it, the way a student who has not
  learned the vocabulary would ask

Reporting nDCG by style **measures** the terminology-mismatch problem instead
of asserting it. A lexical system should show a wide gap; semantic retrieval
and expansion should narrow it.

**Graded judgements.** Passages overlap by 80 of 400 words, so adjacent
passages often contain the same sentence — `physics_txt_book::p25` and `::p26`
both contain the electrostatic shielding definition. Marking only the source
relevant would punish a system for returning an equally good answer. So:

- `2` — the source passage
- `1` — a passage overlapping the source above threshold
- `0` — everything else

**Stated limitations.** Synthetic judgements are incomplete: a genuinely
relevant passage the generator never saw counts as non-relevant, which
penalises the systems good enough to find it. Pooled judging is the standard
fix and `qrels.jsonl` carries a `source` field on every judgement so pooled or
human entries merge in without rebuilding. Human judgements can be recorded
in the admin and exported with the *Export selected judgements* action.

Also: with roughly one relevant passage per query, `Recall@k` equals
`Success@k` and `MAP` equals `MRR`. The results table reports MRR and nDCG@k
and omits the redundant columns rather than printing duplicate numbers.

## Significance testing

A difference in mean nDCG is not a result. Every comparison against the
baseline runs a **paired randomisation test** over per-query scores with
**Holm correction** across the family of comparisons. The primary measure
(`nDCG@10`) is fixed in advance rather than chosen after seeing the numbers.

The randomisation test is preferred over the t-test because IR measures are
bounded, skewed and often zero-inflated — exactly where normality assumptions
are weakest (Smucker, Allan & Carterette, CIKM 2007).

## A methodological note worth reading

During development, two confident claims about retrieval quality were made
from eyeballed examples, and **both were wrong**:

1. "BM25 fails on this overfitting query" — it succeeded, because the query
   shared the generic terms `model` and `data` with the passage.
2. "The reranker demoted the correct passage" — the promoted passage actually
   contained the section heading *"6. Electrostatic shielding"*. The judgement
   had been made by reading the first 66 characters of a 400-word passage.

Same root cause both times: **inspecting a handful of top-1 snippets is not
evaluation.** Relevance judged by eye on a truncated snippet is itself
unreliable, so a single worked example is weak evidence about a retrieval
system. That is the real argument for the measurement apparatus, and it is
stronger than "we should measure things".

## Verification

```bash
venv/Scripts/python.exe -m unittest discover -s tests   # 420 tests
venv/Scripts/python.exe tools/smoke_views.py            # renders every page
venv/Scripts/python.exe tools/check_pdfs.py media/textbooks
```

The BM25 implementation is hand-written over an explicit inverted index and
**asserted equal to `rank_bm25.BM25Okapi` to 1e-9** — on the fixture, on
repeated query terms, on a 50-document random corpus across 10 random
queries, and on the IDF values including the negative-flooring path. Metrics
are checked against hand-computed values, including the TREC recall penalty in
AP and both DCG gain formulations.

## Reference points

- Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25
  and Beyond* — BM25, `k1`, `b`
- Cormack, Clarke & Buettcher (2009), SIGIR — Reciprocal Rank Fusion, `k=60`
- Järvelin & Kekäläinen (2002), ACM TOIS — (n)DCG
- Lavrenko & Croft (2001), SIGIR; Abdul-Jaleel et al. (2004) — relevance
  models, RM3
- Fox & Shaw (1994), TREC-2 — CombSUM, CombMNZ
- Smucker, Allan & Carterette (2007), CIKM — significance tests for IR
- Thakur et al. (2021), NeurIPS — BEIR, for systematic retrieval evaluation

Verify each one before citing it; do not cite anything you have not opened.
