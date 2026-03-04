"""
lyrics_reco.pipeline.run_baseline

Baseline retrieval + evaluation pipeline (lexicon ratio vectors).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..common.config import load_yaml, dump_run_config
from ..common.io import save_csv
from ..common.logging import setup_run_logger
from ..common.seed import set_seed
from ..common.paths import PATHS
from ..lexicon.load import load_lexicons_from_cfg
from ..baseline.emotion_features import build_lexicon_feature_table
from ..retrieval.cosine import topk_cosine, l2_normalize_rows
from ..retrieval.filters import FilterConfig, filter_candidates
from ..retrieval.mmr import mmr_rerank
from ..retrieval.results import build_recommendations_table
from ..evaluation.runner import evaluate_from_rec_table, EvalConfig
from ..evaluation.pseudo_gt import PseudoGTConfig
from .utils import cfg_get, sample_queries, make_run_dirs


def _load_configs(eval_path: str, retrieval_path: str, emotion_path: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if eval_path:
        cfg.update(load_yaml(eval_path))
    if retrieval_path:
        cfg.update(load_yaml(retrieval_path))
    if emotion_path:
        cfg.update(load_yaml(emotion_path))
    return cfg


def _build_vectors(
    meta_df: pd.DataFrame,
    emotion_cfg: Dict[str, Any],
    *,
    include_intensity: bool,
    include_vad: bool,
    logger,
):
    bundle = load_lexicons_from_cfg(emotion_cfg)
    emotions = cfg_get(emotion_cfg, ["emotion", "emotions"], None)

    feats = build_lexicon_feature_table(
        meta_df,
        bundle,
        song_id_col="song_id",
        text_col="lyrics_clean",
        emotions=emotions,
        include_intensity=include_intensity,
        include_vad=include_vad,
        intensity_aggregation=cfg_get(emotion_cfg, ["intensity", "aggregation"], "mean"),
        vad_aggregation=cfg_get(emotion_cfg, ["vad", "aggregation"], "mean"),
    )

    ratio_cols = sorted([c for c in feats.columns if c.startswith("ratio_")]) if emotions is None else [f"ratio_{e.lower()}" for e in emotions]
    X = feats[ratio_cols].astype(float).to_numpy()
    Xn = l2_normalize_rows(X)
    logger.info("Built baseline vectors: N=%d, D=%d", Xn.shape[0], Xn.shape[1])
    return feats, Xn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed CSV (from preprocess)")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--emotion-config", default="configs/emotion_context.yaml")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save-vectors-csv", action="store_true", default=False)
    ap.add_argument("--include-intensity", action="store_true", default=False)  # baseline에서는 기본 OFF
    ap.add_argument("--include-vad", action="store_true", default=False)        # baseline에서는 기본 OFF

    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--top-m", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--mmr-lambda", type=float, default=-1.0)
    ap.add_argument("--disable-mmr", action="store_true", default=False)

    args = ap.parse_args()

    set_seed(args.seed)
    cfg = _load_configs(args.eval_config, args.retrieval_config, args.emotion_config)

    run_cfg = {"pipeline": "run_baseline", "params": vars(args), "merged_cfg": cfg}
    art_meta = dump_run_config(run_cfg, prefix="baseline")
    logger = setup_run_logger(art_meta.run_id, name="lyrics_reco", also_to_reports=True)
    art_dir, _ = make_run_dirs(art_meta.run_id)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (PATHS.root / data_path).resolve()
    meta_df = pd.read_csv(data_path)
    logger.info("Loaded processed data: rows=%d cols=%d", len(meta_df), meta_df.shape[1])

    logger.info("Building baseline vectors (lexicon ratio)...")
    feats_df, X = _build_vectors(
        meta_df,
        cfg,
        include_intensity=args.include_intensity,
        include_vad=args.include_vad,
        logger=logger,
    )
    if args.save_vectors_csv:
        save_csv(feats_df, art_dir / "baseline_lexicon_features.csv", index=False)

    # query sampling
    eval_seed = int(cfg_get(cfg, ["eval", "seed"], args.seed))
    n_queries = int(cfg_get(cfg, ["eval", "n_queries"], 500))
    if args.n_queries and args.n_queries > 0:
        n_queries = args.n_queries

    stratify_by = cfg_get(cfg, ["eval", "query_sampling", "stratify_by"], [])
    min_per_stratum = int(cfg_get(cfg, ["eval", "query_sampling", "min_per_stratum"], 0))
    q_idx = sample_queries(meta_df, n_queries=n_queries, seed=eval_seed, stratify_by=stratify_by, min_per_stratum=min_per_stratum)

    # retrieval settings
    top_m = int(cfg_get(cfg, ["retrieval", "top_m"], 200))
    top_k = int(cfg_get(cfg, ["retrieval", "top_k"], 20))
    if args.top_m and args.top_m > 0:
        top_m = args.top_m
    if args.top_k and args.top_k > 0:
        top_k = args.top_k

    mmr_enabled = bool(cfg_get(cfg, ["retrieval", "mmr", "enabled"], True))
    if args.disable_mmr:
        mmr_enabled = False
    mmr_lambda = float(cfg_get(cfg, ["retrieval", "mmr", "lambda"], 0.7))
    if args.mmr_lambda >= 0.0:
        mmr_lambda = float(args.mmr_lambda)

    fcfg = FilterConfig(
        exclude_self=bool(cfg_get(cfg, ["filters", "exclude_same_song"], True)),
        exclude_same_artist=bool(cfg_get(cfg, ["filters", "exclude_same_artist"], False)),
        year_window=cfg_get(cfg, ["filters", "year_window"], None),
        song_id_col="song_id",
        artist_col="artist",
        year_col="year",
    )

    rec_indices_list = []
    rec_scores_list = []

    for t_i, qi in enumerate(q_idx.tolist(), start=1):
        progress_every = 25
        if t_i == 1 or (t_i % progress_every) == 0:
            logger.info("Retrieval progress: %d/%d queries", t_i, len(q_idx))
        cand_idx, cand_sc = topk_cosine(X, int(qi), top_k=top_m, exclude_self=False, normalize=False)
        cand_idx, cand_sc = filter_candidates(meta_df, query_index=int(qi), cand_indices=cand_idx, cand_scores=cand_sc, cfg=fcfg)

        if cand_idx.size == 0:
            rec_indices_list.append(np.array([], dtype=int))
            rec_scores_list.append(np.array([], dtype=float))
            continue

        if mmr_enabled:
            sel_idx, sel_sc = mmr_rerank(X, int(qi), cand_idx, cand_sc, top_k=top_k, lambda_=mmr_lambda)
        else:
            sel_idx, sel_sc = cand_idx[:top_k], cand_sc[:top_k]

        rec_indices_list.append(sel_idx.astype(int))
        rec_scores_list.append(sel_sc.astype(float))

    rec_df = build_recommendations_table(meta_df, q_idx, rec_indices_list, rec_scores_list)
    save_csv(rec_df, art_dir / "baseline_recommendations.csv", index=False)

    # evaluation
    k_values = tuple(int(x) for x in cfg_get(cfg, ["eval", "k_values"], [5, 10, 20]))
    eval_cfg = EvalConfig(k_values=k_values)

    pseudo_cfg = PseudoGTConfig(
        year_window=cfg_get(cfg, ["pseudo_ground_truth", "year_window"], 10),
        require_same_genre=bool(cfg_get(cfg, ["pseudo_ground_truth", "require_same_genre"], True)),
        exclude_self=bool(cfg_get(cfg, ["pseudo_ground_truth", "exclude_same_song"], True)),
        exclude_same_artist=bool(cfg_get(cfg, ["pseudo_ground_truth", "exclude_same_artist"], True)),
        graded_enabled=bool(cfg_get(cfg, ["pseudo_ground_truth", "graded_relevance", "enabled"], True)),
        grade_if_same_genre_and_within_year=int(cfg_get(cfg, ["pseudo_ground_truth", "graded_relevance", "grade_if_same_genre_and_within_year"], 2)),
        grade_if_same_genre_only=int(cfg_get(cfg, ["pseudo_ground_truth", "graded_relevance", "grade_if_same_genre_only"], 0)),
        max_grade1_per_query=int(cfg_get(cfg, ["pseudo_ground_truth", "graded_relevance", "max_grade1_per_query"], 0)),
        song_id_col="song_id",
        artist_col="artist",
        year_col="year",
        genre_col="genre",
    )

    evaluate_from_rec_table(
        meta_df,
        rec_df,
        eval_cfg=eval_cfg,
        pseudo_cfg=pseudo_cfg,
        emotion_vectors=X,
        item_vectors=X,
        save_dir=art_dir,
    )

    logger.info("Done. Artifacts in %s", art_dir)


if __name__ == "__main__":
    main()