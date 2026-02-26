"""lyrics_reco.retrieval

Retrieval utilities shared by baseline and proposed methods.

Included:
- cosine.py: cosine similarity Top-M/Top-K retrieval for dense/sparse matrices
- filters.py: metadata-based candidate filtering (exclude self/artist, year window)
- mmr.py: MMR re-ranking to trade off relevance vs diversity
- results.py: build a tidy recommendation table (CSV-friendly)
"""

from .cosine import topk_cosine, batch_topk_cosine
from .filters import FilterConfig, filter_candidates
from .mmr import mmr_rerank
from .results import build_recommendations_table
