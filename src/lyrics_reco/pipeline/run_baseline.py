"""
lyrics_reco.pipeline.run_baseline

Baseline retrieval + evaluation pipeline (song-conditioned TF-IDF baseline).

Storage behavior in this version:
- per-run artifacts are still written under artifacts/runs/<run_id>/
- canonical central demo/fusion artifacts are refreshed under artifacts/vectors/
  * baseline_vectors.npz
  * baseline_song_ids.npy
- optional CSV sidecar can also be written when requested
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize as sk_normalize

from ..baseline.emotion_features import build_lexicon_feature_table
from ..baseline.tfidf import (
    apply_song_conditioned_emotion_weights,
    build_song_emotion_distribution,
    build_tfidf,
    build_vocab_emotion_matrix,
    save_vocab_idf_csv,
)
from ..common.config import dump_run_config, load_yaml
from ..common.io import save_csv
from ..common.logging import setup_run_logger
from ..common.paths import PATHS
from ..common.seed import set_seed
from ..common.vector_store import save_central_vectors, save_dense_vectors_npz, save_song_ids
from ..evaluation.pseudo_gt import PseudoGTConfig
from ..evaluation.runner import EvalConfig, evaluate_from_rec_table
from ..lexicon.load import load_lexicons_from_cfg
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


def _l2_normalize_sparse(X: sparse.csr_matrix) -> sparse.csr_matrix:
    return sk_normalize(X, norm="l2", axis=1, copy=True)


def _topk_cosine_sparse(
    Xn: sparse.csr_matrix,
    query_index: int,
    top_k: int,
    *,
    exclude_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    q = Xn.getrow(int(query_index))
    scores = (Xn @ q.T).toarray().ravel().astype(np.float32)

    if exclude_self:
        scores[int(query_index)] = -np.inf

    k = min(int(top_k), scores.shape[0])
    if k <= 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(int), scores[idx].astype(np.float32)


def _build_tfidf_baseline(
    meta_df: pd.DataFrame,
    emotion_cfg: Dict[str, Any],
    logger,
):
    """
    Research-plan baseline:
        v(s) = x(s) ⊙ (A p(s))
    where
        x(s): TF-IDF vector of song s
        A: vocab-emotion projection matrix
        p(s): song-level emotion distribution
    """
    text_col = "lyrics_dedup" if "lyrics_dedup" in meta_df.columns else "lyrics_clean"
    lex_col = "lyrics_clean" if "lyrics_clean" in meta_df.columns else text_col

    tfidf_art = build_tfidf(
        meta_df,
        text_col=text_col,
        max_features=200_000,
        ngram_range=(1, 1),
        min_df=3,
        max_df=0.95,
    )

    bundle = load_lexicons_from_cfg(emotion_cfg)
    emotions = cfg_get(
        emotion_cfg,
        ["emotion", "emotions"],
        ["anger", "fear", "joy", "sadness", "disgust", "trust"],
    )

    P_song = build_song_emotion_distribution(
        meta_df,
        text_col=lex_col,
        bundle=bundle,
        emotions=emotions,
    )
    A_vocab = build_vocab_emotion_matrix(tfidf_art.vocab, bundle, emotions)

    X_weighted = apply_song_conditioned_emotion_weights(
        tfidf_art.X,
        P_song,
        A_vocab,
        floor_weight=0.0,
    ).tocsr()
    Xn = _l2_normalize_sparse(X_weighted)

    logger.info(
        "Built TF-IDF baseline: N=%d, D=%d, nnz=%d",
        Xn.shape[0],
        Xn.shape[1],
        Xn.nnz,
    )

    aux = {
        "P_song": P_song,
        "A_vocab": A_vocab,
        "emotions": [str(e).lower() for e in emotions],
    }
    return tfidf_art, aux, Xn


def _build_eval_emotion_vectors(
    meta_df: pd.DataFrame,
    emotion_cfg: Dict[str, Any],
) -> np.ndarray:
    bundle = load_lexicons_from_cfg(emotion_cfg)
    emotions = cfg_get(emotion_cfg, ["emotion", "emotions"], None)

    feats = build_lexicon_feature_table(
        meta_df,
        bundle,
        song_id_col="song_id",
        text_col="lyrics_clean",
        emotions=emotions,
        include_intensity=True,
        include_vad=True,
        intensity_aggregation=cfg_get(emotion_cfg, ["intensity", "aggregation"], "mean"),
        vad_aggregation=cfg_get(emotion_cfg, ["vad", "aggregation"], "mean"),
    )

    ratio_cols = [f"ratio_{str(e).lower()}" for e in emotions]
    vad_cols = [c for c in ["valence", "arousal", "dominance"] if c in feats.columns]
    emotion_cols = ratio_cols + vad_cols

    E = feats[emotion_cols].astype(np.float32).to_numpy()
    E = sk_normalize(E, norm="l2", axis=1, copy=True)
    return E


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
        help="Save canonical baseline artifacts under artifacts/vectors/",
    )
    ap.add_argument(
        "--no-save-central-vectors",
        dest="save_central_vectors",
        action="store_false",
    )
    ap.add_argument(
        "--central-vectors-out",
        default="",
        help="Optional override path for the central baseline vectors artifact (.npz expected).",
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
        help="NPZ is canonical for baseline. CSV is an optional auxiliary export.",
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
    logger.info("Building baseline vectors (song-conditioned TF-IDF baseline)...")
    tfidf_art, baseline_aux, X = _build_tfidf_baseline(meta_df, cfg, logger)
    E_eval = _build_eval_emotion_vectors(meta_df, cfg)

    # per-run artifacts -------------------------------------------------------
    sparse.save_npz(art_dir / "baseline_tfidf_weighted.npz", X)
    np.save(art_dir / "baseline_song_ids.npy", meta_df["song_id"].astype(str).to_numpy(dtype=object))

    save_vocab_idf_csv(
        tfidf_art.vocab,
        tfidf_art.idf,
        art_dir / "baseline_tfidf_vocab_idf.csv",
    )

    emotions = baseline_aux["emotions"]
    ratio_df = pd.DataFrame(
        baseline_aux["P_song"],
        columns=[f"ratio_{e}" for e in emotions],
    )
    ratio_df.insert(0, "song_id", meta_df["song_id"].astype(str).tolist())
    save_csv(ratio_df, art_dir / "baseline_song_emotion_distribution.csv", index=False)

    vocab_rows = [{"term": t, "index": int(i)} for t, i in tfidf_art.vocab.items()]
    vocab_df = pd.DataFrame(vocab_rows).sort_values("index").reset_index(drop=True)
    np.save(art_dir / "baseline_vocab_emotion_matrix.npy", baseline_aux["A_vocab"])
    save_csv(vocab_df, art_dir / "baseline_vocab_index.csv", index=False)

    # canonical central artifacts --------------------------------------------
    if args.save_central_vectors:
        fmt = str(args.central_format).lower()
        central_npz_path = None
        if fmt in {"npz", "both", "csv"}:
            central_npz_path = save_dense_vectors_npz(
                X,
                "baseline",
                out_path=(args.central_vectors_out or None),
                paths=PATHS,
            )
            logger.info("Saved central baseline vectors (npz): %s", central_npz_path)

        save_song_ids(
            meta_df["song_id"].astype(str).to_numpy(),
            "baseline",
            out_path=(args.central_song_ids_out or None),
            paths=PATHS,
        )
        logger.info("Saved central baseline song ids")

        if fmt in {"csv", "both"}:
            csv_path = save_central_vectors(ratio_df, "baseline", paths=PATHS)
            logger.info(
                "Saved auxiliary central baseline CSV: %s (stores song-level emotion ratios, not sparse TF-IDF)",
                csv_path,
            )

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

        cand_idx, cand_sc = _topk_cosine_sparse(
            X,
            int(qi),
            top_k=top_m,
            exclude_self=False,
        )
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
        emotion_vectors=E_eval,
        item_vectors=X,
        save_dir=art_dir,
    )
    logger.info("Done. Artifacts in %s", art_dir)


if __name__ == "__main__":
    main()
