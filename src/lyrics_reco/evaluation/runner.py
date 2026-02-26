"""
lyrics_reco.evaluation.runner

Glue utilities to evaluate recommendation outputs.

Supports evaluating from a "tidy recommendation table" (CSV-friendly):
  query_index, rec_index, rank, score, ...

This keeps baseline/proposed outputs compatible.

Key functions:
- group_rec_table: convert table -> (query_indices, rec_indices_list, rec_scores_list)
- evaluate_from_rec_table: compute per-query metrics + aggregate table
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..common.io import save_csv
from ..common.paths import PATHS, ProjectPaths
from .pseudo_gt import PseudoGTConfig, GenreYearIndex, build_pseudo_ground_truth
from .metrics import (
    recall_at_k,
    ndcg_at_k,
    emotion_consistency_at_k,
    ild_at_k,
    aggregate_metrics_table,
)

ArrayLike = Union[np.ndarray, "scipy.sparse.spmatrix"]  # type: ignore


def group_rec_table(
    rec_df: pd.DataFrame,
    *,
    query_index_col: str = "query_index",
    rec_index_col: str = "rec_index",
    score_col: str = "score",
    rank_col: str = "rank",
) -> Tuple[np.ndarray, Sequence[np.ndarray], Sequence[np.ndarray]]:
    """
    Group a tidy rec table into per-query arrays, sorted by rank ascending.
    """
    needed = {query_index_col, rec_index_col, score_col, rank_col}
    missing = [c for c in needed if c not in rec_df.columns]
    if missing:
        raise ValueError(f"rec_df missing columns: {missing}")

    qs = rec_df[query_index_col].astype(int).unique()
    qs = np.sort(qs)

    rec_indices_list = []
    rec_scores_list = []

    for qi in qs.tolist():
        sub = rec_df[rec_df[query_index_col].astype(int) == int(qi)].copy()
        sub = sub.sort_values(rank_col, ascending=True)
        rec_indices_list.append(sub[rec_index_col].astype(int).to_numpy())
        rec_scores_list.append(pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0).astype(float).to_numpy())

    return qs.astype(int), rec_indices_list, rec_scores_list


@dataclass(frozen=True)
class EvalConfig:
    k_values: Tuple[int, ...] = (5, 10, 20)
    # which metrics to compute
    do_recall: bool = True
    do_ndcg: bool = True
    do_ec: bool = True
    do_ild: bool = True
    # ild distance
    ild_distance: str = "1-cosine"


def evaluate_from_rec_table(
    meta_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    *,
    eval_cfg: EvalConfig = EvalConfig(),
    pseudo_cfg: PseudoGTConfig = PseudoGTConfig(),
    emotion_vectors: Optional[ArrayLike] = None,
    item_vectors: Optional[ArrayLike] = None,
    paths: ProjectPaths = PATHS,
    save_dir: Optional[Union[str, "pathlib.Path"]] = None,  # type: ignore
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate recommendations from a tidy rec_df.

    Returns:
      - per_query_df
      - agg_df (metric,k,mean,std,n)

    If save_dir is provided, saves:
      - per_query_metrics.csv
      - summary_metrics.csv
    """
    q_idx, rec_indices_list, rec_scores_list = group_rec_table(rec_df)

    # pseudo ground truth
    idxer = GenreYearIndex(meta_df, year_col=pseudo_cfg.year_col, genre_col=pseudo_cfg.genre_col)
    relevant_map, grade_map = build_pseudo_ground_truth(meta_df, q_idx, cfg=pseudo_cfg, index=idxer)

    rows = []
    for qi, rec_idx in zip(q_idx.tolist(), rec_indices_list):
        row = {"query_index": int(qi)}
        for k in eval_cfg.k_values:
            k = int(k)
            if eval_cfg.do_recall:
                row[f"recall@{k}"] = recall_at_k(rec_idx, relevant_map[int(qi)], k)
            if eval_cfg.do_ndcg:
                row[f"ndcg@{k}"] = ndcg_at_k(rec_idx, grade_map[int(qi)], k)
            if eval_cfg.do_ec:
                if emotion_vectors is None:
                    row[f"ec@{k}"] = np.nan
                else:
                    row[f"ec@{k}"] = emotion_consistency_at_k(emotion_vectors, int(qi), rec_idx, k=k)
            if eval_cfg.do_ild:
                if item_vectors is None:
                    row[f"ild@{k}"] = np.nan
                else:
                    row[f"ild@{k}"] = ild_at_k(item_vectors, rec_idx, k=k, distance=eval_cfg.ild_distance)
        rows.append(row)

    per_query_df = pd.DataFrame(rows)
    agg_df = aggregate_metrics_table(per_query_df, k_values=eval_cfg.k_values)

    if save_dir is not None:
        from pathlib import Path
        sd = Path(save_dir)
        if not sd.is_absolute():
            sd = (paths.root / sd).resolve()
        sd.mkdir(parents=True, exist_ok=True)
        save_csv(per_query_df, sd / "per_query_metrics.csv", index=False)
        save_csv(agg_df, sd / "summary_metrics.csv", index=False)

    return per_query_df, agg_df
