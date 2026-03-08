"""
lyrics_reco.pipeline.run_baseline

Baseline retrieval + evaluation pipeline (lexicon ratio vectors).

Main change in this version:
- keeps the per-run vector CSV under artifacts/runs/<run_id>/ when requested
- maintains a central demo-friendly baseline vector artifact under artifacts/vectors/
  (NPZ by default, optional CSV/both)
- also saves baseline_song_ids.npy so demo can align rows safely
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..baseline.emotion_features import build_lexicon_feature_table
from ..common.config import dump_run_config, load_yaml
from ..common.io import save_csv
from ..common.logging import setup_run_logger
from ..common.paths import PATHS
from ..common.seed import set_seed
from ..common.vector_store import (
    copy_vector_csv,
    save_central_vectors,
    save_dense_vectors_npz,
    save_song_ids,
)
from ..evaluation.pseudo_gt import PseudoGTConfig
from ..evaluation.runner import EvalConfig, evaluate_from_rec_table
from ..lexicon.load import load_lexicons_from_cfg
from ..retrieval.cosine import l2_normalize_rows, topk_cosine
from ..retrieval.filters import FilterConfig, filter_candidates
from ..retrieval.mmr import mmr_rerank
from ..retrieval.results import build_recommendations_table
from .utils import cfg_get, make_run_dirs, sample_queries



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
    ratio_cols = sorted([c for c in feats.columns if c.startswith("ratio_")])
    if emotions is not None:
        ratio_cols = [f"ratio_{str(e).lower()}" for e in emotions]
    X = feats[ratio_cols].astype(float).to_numpy()
    Xn = l2_normalize_rows(X)
    logger.info("Built baseline vectors: N=%d, D=%d", Xn.shape[0], Xn.shape[1])
    return feats, Xn



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed CSV (from preprocess)")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--emotion-config", default="configs/emotion_context.yaml")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save-vectors-csv", action="store_true", default=False)
    ap.add_argument("--include-intensity", action="store_true", default=False)
    ap.add_argument("--include-vad", action="store_true", default=False)
    ap.add_argument(
        "--save-central-vectors",
        dest="save_central_vectors",
        action="store_true",
        default=True,
        help="Save a central demo-friendly baseline vector artifact under artifacts/vectors/",
    )
    ap.add_argument(
        "--no-save-central-vectors",
        dest="save_central_vectors",
        action="store_false",
    )
    ap.add_argument(
        "--central-vectors-out",
        default="",
        help="Optional override path for the central baseline vectors artifact. If omitted, a default under artifacts/vectors/ is used.",
    )
    ap.add_argument(
        "--central-song-ids-out",
        default="",
        help="Optional override path for baseline song_id mapping (.npy).",
    )
    ap.add_argument(
        "--central-format",
        choices=["npz", "csv", "both"],
        default="npz",
        help="Format for the central baseline vectors artifact.",
    )

    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--top-m", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--mmr-lambda", type=float, default=-1.0)
    ap.add_argument("--disable-mmr", action="store_true", default=False)
    return ap.parse_args()



def main() -> None:
    args = parse_args()

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

    run_vec_path = None
    if args.save_vectors_csv:
        run_vec_path = save_csv(feats_df, art_dir / "baseline_lexicon_features.csv", index=False)
        logger.info("Saved run baseline vectors CSV: %s", run_vec_path)

    if args.save_central_vectors:
        if args.central_format in {"csv", "both"}:
            central_out_csv = args.central_vectors_out if args.central_vectors_out.lower().endswith(".csv") else None
            if run_vec_path is not None and central_out_csv is not None:
                central_csv_path = copy_vector_csv(run_vec_path, "baseline", out_path=central_out_csv, paths=PATHS)
            else:
                central_csv_path = save_central_vectors(
                    feats_df,
                    "baseline",
                    out_path=central_out_csv,
                    paths=PATHS,
                )
            logger.info("Saved central baseline vectors CSV: %s", central_csv_path)

        if args.central_format in {"npz", "both"}:
            central_npz_out = args.central_vectors_out if args.central_vectors_out.lower().endswith(".npz") else None
            central_npz_path = save_dense_vectors_npz(X, "baseline", out_path=central_npz_out, paths=PATHS)
            logger.info("Saved central baseline vectors NPZ: %s", central_npz_path)

        song_ids_path = save_song_ids(
            feats_df["song_id"].astype(str).to_numpy(),
            "baseline",
            out_path=args.central_song_ids_out or None,
            paths=PATHS,
        )
        logger.info("Saved central baseline song ids: %s", song_ids_path)

    eval_seed = int(cfg_get(cfg, ["eval", "seed"], args.seed))
    n_queries = int(cfg_get(cfg, ["eval", "n_queries"], 500))
    if args.n_queries and args.n_queries > 0:
        n_queries = args.n_queries

    stratify_by = cfg_get(cfg, ["eval", "query_sampling", "stratify_by"], [])
    min_per_stratum = int(cfg_get(cfg, ["eval", "query_sampling", "min_per_stratum"], 0))
    q_idx = sample_queries(
        meta_df,
        n_queries=n_queries,
        seed=eval_seed,
        stratify_by=stratify_by,
        min_per_stratum=min_per_stratum,
    )

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
        if t_i == 1 or (t_i % 25) == 0:
            logger.info("Retrieval progress: %d/%d queries", t_i, len(q_idx))

        cand_idx, cand_sc = topk_cosine(X, int(qi), top_k=top_m, exclude_self=False, normalize=False)
        cand_idx, cand_sc = filter_candidates(
            meta_df,
            query_index=int(qi),
            cand_indices=cand_idx,
            cand_scores=cand_sc,
            cfg=fcfg,
        )

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

    k_values = tuple(int(x) for x in cfg_get(cfg, ["eval", "k_values"], [5, 10, 20]))
    eval_cfg = EvalConfig(k_values=k_values)
    pseudo_cfg = PseudoGTConfig(
        year_window=cfg_get(cfg, ["pseudo_ground_truth", "year_window"], 10),
        require_same_genre=bool(cfg_get(cfg, ["pseudo_ground_truth", "require_same_genre"], True)),
        exclude_self=bool(cfg_get(cfg, ["pseudo_ground_truth", "exclude_same_song"], True)),
        exclude_same_artist=bool(cfg_get(cfg, ["pseudo_ground_truth", "exclude_same_artist"], True)),
        graded_enabled=bool(cfg_get(cfg, ["pseudo_ground_truth", "graded_relevance", "enabled"], True)),
        grade_if_same_genre_and_within_year=int(
            cfg_get(cfg, ["pseudo_ground_truth", "graded_relevance", "grade_if_same_genre_and_within_year"], 2)
        ),
        grade_if_same_genre_only=int(
            cfg_get(cfg, ["pseudo_ground_truth", "graded_relevance", "grade_if_same_genre_only"], 0)
        ),
        max_grade1_per_query=int(
            cfg_get(cfg, ["pseudo_ground_truth", "graded_relevance", "max_grade1_per_query"], 0)
        ),
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
