"""lyrics_reco.pipeline.run_fusion

Late-fusion runner for the revised research design.

Fusion score:
    score_fusion(q, s) = lambda * score_proposed(q, s) + (1-lambda) * score_baseline(q, s)

MMR redundancy is computed in proposed z-space.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from ..common.config import dump_run_config, load_yaml
from ..common.io import save_csv
from ..common.logging import setup_run_logger
from ..common.paths import PATHS
from ..common.seed import set_seed
from ..common.vector_store import default_song_ids_path, default_vector_path
from ..evaluation.pseudo_gt import PseudoGTConfig
from ..evaluation.runner import EvalConfig, evaluate_from_rec_table
from ..pipeline.utils import cfg_get, make_run_dirs, sample_queries
from ..retrieval.cosine import topk_cosine
from ..retrieval.filters import FilterConfig, filter_candidates
from ..retrieval.mmr import mmr_rerank
from ..retrieval.results import build_recommendations_table


def _load_configs(*paths: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for p in paths:
        if p:
            cfg.update(load_yaml(p))
    return cfg


def _resolve_existing_path(path_str: str | None, fallback: Path | None) -> Path:
    if path_str:
        p = Path(path_str)
        if not p.is_absolute():
            p = (PATHS.root / p).resolve()
        return p
    if fallback is None:
        raise FileNotFoundError("Could not resolve required artifact path")
    return fallback.resolve()


def _vector_cols(df: pd.DataFrame, prefix: str = "z_") -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No vector columns found with prefix '{prefix}'")
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def _topk_cosine_sparse(Xn: sparse.csr_matrix, query_index: int, top_k: int, *, exclude_self: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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


def _split_z_components(Z: np.ndarray, emotions: Sequence[str], vector_layout: str) -> Dict[str, np.ndarray]:
    emo_dim = len(list(emotions))
    if vector_layout == "embedding_ratio_vad":
        tail = emo_dim + 3
        emb_dim = Z.shape[1] - tail
        return {
            "embedding": Z[:, :emb_dim].astype(np.float32, copy=False),
            "emotion_ratio": Z[:, emb_dim: emb_dim + emo_dim].astype(np.float32, copy=False),
            "vad": Z[:, emb_dim + emo_dim: emb_dim + emo_dim + 3].astype(np.float32, copy=False),
        }
    if vector_layout == "embedding_ratio_intensity_vad":
        tail = emo_dim + emo_dim + 3
        emb_dim = Z.shape[1] - tail
        pos = emb_dim
        out = {
            "embedding": Z[:, :emb_dim].astype(np.float32, copy=False),
            "emotion_ratio": Z[:, pos: pos + emo_dim].astype(np.float32, copy=False),
        }
        pos += emo_dim
        out["intensity"] = Z[:, pos: pos + emo_dim].astype(np.float32, copy=False)
        pos += emo_dim
        out["vad"] = Z[:, pos: pos + 3].astype(np.float32, copy=False)
        return out
    raise ValueError(f"Unknown vector layout: {vector_layout}")


def _normalize_minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi - lo <= 1e-12:
        return np.ones_like(scores, dtype=np.float32)
    return (scores - lo) / (hi - lo)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--emotion-config", default="configs/emotion_context.yaml")
    ap.add_argument("--baseline-npz", default="")
    ap.add_argument("--baseline-song-ids", default="")
    ap.add_argument("--proposed-csv", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fusion-lambda", type=float, default=0.6)
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--top-m", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--use-mmr", action="store_true", default=False)
    ap.add_argument("--mmr-lambda", type=float, default=0.7)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = _load_configs(args.eval_config, args.retrieval_config, args.emotion_config)
    run_cfg = {"pipeline": "run_fusion", "params": vars(args), "merged_cfg": cfg}
    art_meta = dump_run_config(run_cfg, prefix="fusion")
    logger = setup_run_logger(art_meta.run_id, name="lyrics_reco", also_to_reports=True)
    art_dir, _ = make_run_dirs(art_meta.run_id)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (PATHS.root / data_path).resolve()
    meta_df = pd.read_csv(data_path)
    meta_df = meta_df.drop_duplicates(subset=["song_id"], keep="first").reset_index(drop=True)

    baseline_npz = _resolve_existing_path(args.baseline_npz, default_vector_path("baseline", paths=PATHS))
    baseline_song_ids = _resolve_existing_path(args.baseline_song_ids, default_song_ids_path("baseline", paths=PATHS))
    proposed_csv = _resolve_existing_path(args.proposed_csv, default_vector_path("proposed", paths=PATHS))

    logger.info("Using baseline vectors: %s", baseline_npz)
    logger.info("Using baseline song ids: %s", baseline_song_ids)
    logger.info("Using proposed vectors: %s", proposed_csv)

    X_base = sparse.load_npz(baseline_npz).tocsr()
    song_ids = np.load(baseline_song_ids, allow_pickle=True).astype(str)
    if len(song_ids) != X_base.shape[0]:
        raise ValueError(
            f"baseline song_ids length ({len(song_ids)}) does not match vector rows ({X_base.shape[0]})"
        )
    row_map = {sid: i for i, sid in enumerate(song_ids.tolist())}
    order = np.asarray([row_map[str(sid)] for sid in meta_df["song_id"].astype(str).tolist()], dtype=np.int64)
    X_base = X_base[order]

    vectors_df = pd.read_csv(proposed_csv)
    vectors_df = vectors_df.drop_duplicates(subset=["song_id"], keep="first").reset_index(drop=True)
    vec_cols = _vector_cols(vectors_df, prefix="z_")
    lookup = vectors_df.set_index("song_id", drop=False)
    aligned = lookup.loc[meta_df["song_id"].astype(str).tolist()]
    Z = aligned[vec_cols].to_numpy(dtype=np.float32)

    emotions = [e.lower() for e in cfg_get(cfg, ["emotion", "emotions"], ["anger", "fear", "joy", "sadness", "disgust", "trust"])]
    vector_layout = str(cfg_get(cfg, ["aggregation", "vector_layout"], "embedding_ratio_vad"))
    comps = _split_z_components(Z, emotions, vector_layout)
    emotion_vectors = np.concatenate([comps["emotion_ratio"], comps.get("vad", np.zeros((len(Z), 0), dtype=np.float32))], axis=1)
    item_vectors = Z

    eval_seed = int(cfg_get(cfg, ["eval", "seed"], args.seed))
    n_queries = int(cfg_get(cfg, ["eval", "n_queries"], 300))
    if args.n_queries > 0:
        n_queries = args.n_queries
    q_idx = sample_queries(
        meta_df,
        n_queries=n_queries,
        seed=eval_seed,
        stratify_by=cfg_get(cfg, ["eval", "query_sampling", "stratify_by"], []),
        min_per_stratum=int(cfg_get(cfg, ["eval", "query_sampling", "min_per_stratum"], 0)),
    )

    top_m = int(cfg_get(cfg, ["retrieval", "top_m"], 200))
    top_k = int(cfg_get(cfg, ["retrieval", "top_k"], 20))
    if args.top_m > 0:
        top_m = args.top_m
    if args.top_k > 0:
        top_k = args.top_k

    fcfg = FilterConfig(
        exclude_self=bool(cfg_get(cfg, ["filters", "exclude_same_song"], True)),
        exclude_same_artist=bool(cfg_get(cfg, ["filters", "exclude_same_artist"], False)),
        year_window=cfg_get(cfg, ["filters", "year_window"], None),
        song_id_col="song_id",
        artist_col="artist",
        year_col="year",
    )

    rec_indices_list: List[np.ndarray] = []
    rec_scores_list: List[np.ndarray] = []
    for qi in q_idx.tolist():
        qi = int(qi)
        base_idx, base_sc = _topk_cosine_sparse(X_base, qi, top_m, exclude_self=False)
        prop_idx, prop_sc = topk_cosine(Z, qi, top_k=top_m, exclude_self=False, normalize=True)

        cand_union = sorted(set(base_idx.tolist()) | set(prop_idx.tolist()))
        if not cand_union:
            rec_indices_list.append(np.array([], dtype=int))
            rec_scores_list.append(np.array([], dtype=float))
            continue

        cand_union_np = np.asarray(cand_union, dtype=int)
        base_floor = float(np.nanmin(base_sc)) if base_sc.size else 0.0
        prop_floor = float(np.nanmin(prop_sc)) if prop_sc.size else 0.0
        base_map = {int(i): float(s) for i, s in zip(base_idx.tolist(), base_sc.tolist())}
        prop_map = {int(i): float(s) for i, s in zip(prop_idx.tolist(), prop_sc.tolist())}
        base_scores = np.asarray([base_map.get(int(i), base_floor) for i in cand_union_np], dtype=np.float32)
        prop_scores = np.asarray([prop_map.get(int(i), prop_floor) for i in cand_union_np], dtype=np.float32)
        fused_scores = args.fusion_lambda * _normalize_minmax(prop_scores) + (1.0 - args.fusion_lambda) * _normalize_minmax(base_scores)

        cand_union_np, fused_scores = filter_candidates(
            meta_df,
            query_index=qi,
            cand_indices=cand_union_np,
            cand_scores=fused_scores,
            cfg=fcfg,
        )
        if cand_union_np.size == 0:
            rec_indices_list.append(np.array([], dtype=int))
            rec_scores_list.append(np.array([], dtype=float))
            continue

        order_idx = np.argsort(-fused_scores)
        cand_union_np = cand_union_np[order_idx]
        fused_scores = fused_scores[order_idx]

        if args.use_mmr and cand_union_np.size > top_k:
            sel_idx, sel_sc = mmr_rerank(Z, qi, cand_union_np, fused_scores, top_k=top_k, lambda_=float(args.mmr_lambda))
        else:
            sel_idx, sel_sc = cand_union_np[:top_k], fused_scores[:top_k]

        rec_indices_list.append(np.asarray(sel_idx, dtype=int))
        rec_scores_list.append(np.asarray(sel_sc, dtype=float))

    rec_df = build_recommendations_table(meta_df, q_idx, rec_indices_list, rec_scores_list)
    save_csv(rec_df, art_dir / "fusion_recommendations.csv", index=False)

    eval_cfg = EvalConfig(k_values=tuple(int(x) for x in cfg_get(cfg, ["eval", "k_values"], [5, 10, 20])))
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
        emotion_vectors=emotion_vectors,
        item_vectors=item_vectors,
        save_dir=art_dir,
    )
    logger.info("Done. Artifacts in %s", art_dir)


if __name__ == "__main__":
    main()
