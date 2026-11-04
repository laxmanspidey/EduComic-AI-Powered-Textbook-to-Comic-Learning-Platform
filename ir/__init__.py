"""
EduRetrieve — a multi-stage hybrid information retrieval framework
for educational textbook corpora.

Package layout
--------------
corpus.py      Document processing: PDF -> passages with structural metadata.
metrics.py     IR effectiveness measures (P@k, R@k, MRR, MAP, nDCG@k).
stats.py       Significance testing for system comparisons.
indexes/       Sparse (TF-IDF, BM25) and dense (embedding) indexes.
fusion.py      Rank- and score-level fusion of multiple result lists.
rerank.py      Cross-encoder neural reranking.
expansion.py   Query expansion (classical RM3 PRF, and LLM-based).
retrievers.py  Composable retrieval pipelines sharing one interface.
qrels.py       Test-collection construction via TREC-style pooling.
experiments.py Experiment runner producing comparison and ablation tables.
"""
__all__ = []
