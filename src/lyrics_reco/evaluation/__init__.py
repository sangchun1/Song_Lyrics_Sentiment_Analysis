"""lyrics_reco.evaluation

Evaluation utilities for the project.

Implements metrics from your research plan / configs/eval.yaml:
- Recall@K
- NDCG@K (graded relevance from pseudo ground-truth)
- Emotion Consistency@K (cosine similarity between query and rec emotion vectors)
- ILD@K (intra-list diversity; average pairwise distance)

Core modules:
- pseudo_gt.py: pseudo ground-truth builder (genre + year window proxy)
- metrics.py: metric implementations
- runner.py: helpers to compute per-query + aggregate results and save CSV
"""

from .pseudo_gt import build_pseudo_ground_truth, PseudoGTConfig, GenreYearIndex
from .metrics import (
    recall_at_k,
    ndcg_at_k,
    emotion_consistency_at_k,
    ild_at_k,
    aggregate_metrics_table,
)
from .runner import evaluate_from_rec_table, group_rec_table
