"""
lyrics_reco.pipeline.run_proposed

End-to-end proposed pipeline:
processed.csv -> emotion_context z(s) -> vectordb upsert -> retrieval -> evaluation

Main change in this version:
- keeps the per-run vector CSV under artifacts/runs/<run_id>/ as before
- additionally maintains a central copy under artifacts/vectors/proposed_vectors.csv
  so demo / quickstart code can load it directly
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..common.config import dump_run_config, load_yaml
from ..common.io import save_csv
from ..common.logging import setup_run_logger
from ..common.paths import PATHS
from ..common.seed import set_seed
from ..common.vector_store import copy_vector_csv, save_central_vectors
from ..emotion_context.builder import build_song_vectors_from_df
from ..evaluation.pseudo_gt import PseudoGTConfig
from ..evaluation.runner import EvalConfig, evaluate_from_rec_table
from ..pipeline.utils import cfg_get, make_run_dirs, sample_queries
from ..retrieval.filters import FilterConfig, filter_candidates
from ..retrieval.mmr import mmr_rerank
from ..retrieval.results import build_recommendations_table
from ..vectordb.factory import open_vectordb_from_cfg



def _safe_dataclass_init(cls, **kwargs):
    fields = getattr(cls, "__dataclass_fields__", {}) or {}
    return cls(**{k: v for k, v in kwargs.items() if k in fields})



def _load_configs(*paths: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for p in paths:
        if not p:
            continue
        cfg.update(load_yaml(p))
    return cfg



def _vector_cols(df: pd.DataFrame, prefix: str = "z_") -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No vector columns found with prefix '{prefix}'")

    def _key(c: str) -> int:
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return 10**9

    return sorted(cols, key=_key)



def _dist_to_score(dist: float, metric: str) -> float:
    m = metric.lower()
    if m == "cosine":
        return float(1.0 - dist)
    if m in ("l2", "euclidean"):
        return float(-dist)
    return float(dist)



def _to_python_scalar(v: Any) -> Any:
    if pd.isna(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)



def _split_z_components(
    Z: np.ndarray,
    *,
    emotions: Sequence[str],
    intensity_enabled: bool,
    vad_enabled: bool,
) -> Dict[str, np.ndarray]:
    """
    Z layout (per emotion_context.concat_song_vector):
    [embedding | emotion_ratio | intensity? | vad?]
    """
    emo_dim = len(list(emotions))
    tail = emo_dim + (emo_dim if intensity_enabled else 0) + (3 if vad_enabled else 0)
    d = Z.shape[1]
    if d < tail:
        raise ValueError(f"Vector dim too small: D={d}, expected at least tail={tail}")

    emb_dim = d - tail
    out: Dict[str, np.ndarray] = {}
    out["embedding"] = Z[:, :emb_dim].astype(np.float32, copy=False)
    out["emotion_ratio"] = Z[:, emb_dim : emb_dim + emo_dim].astype(np.float32, copy=False)
    pos = emb_dim + emo_dim
    if intensity_enabled:
        out["intensity"] = Z[:, pos : pos + emo_dim].astype(np.float32, copy=False)
        pos += emo_dim
    if vad_enabled:
        out["vad"] = Z[:, pos : pos + 3].astype(np.float32, copy=False)
    return out



def _upsert_vectors_batched(
    db,
    vectors_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    id_col: str = "song_id",
    vector_prefix: str = "z_",
    metadata_cols: Sequence[str] = ("title", "artist", "year", "genre"),
    batch_size: int = 5000,
    logger=None,
) -> None:
    vec_cols = _vector_cols(vectors_df, prefix=vector_prefix)
    if id_col not in vectors_df.columns:
        raise ValueError(f"vectors_df missing id_col '{id_col}'")
    if id_col not in meta_df.columns:
        raise ValueError(f"meta_df missing id_col '{id_col}'")

    meta_lookup = meta_df.set_index(id_col, drop=False)
    n = len(vectors_df)
    bs = int(batch_size) if int(batch_size) > 0 else 5000

    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        batch = vectors_df.iloc[i0:i1]
        ids = batch[id_col].astype(str).tolist()
        X = batch[vec_cols].to_numpy(dtype=np.float32)
        metadatas = []
        for sid in ids:
            if sid in meta_lookup.index:
                row = meta_lookup.loc[sid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                md = {}
                for c in metadata_cols:
                    if c in meta_lookup.columns:
                        md[c] = _to_python_scalar(row[c])
                metadatas.append(md)
            else:
                metadatas.append({})
        db.upsert(ids, X, metadatas=metadatas)
        if logger and (i0 // bs) % 20 == 0:
            logger.info("Index upsert: %d/%d", i1, n)



def _local_mmr_on_candidates(
    q_vec: np.ndarray,
    cand_vecs: np.ndarray,
    cand_scores: np.ndarray,
    *,
    top_k: int,
    lambda_: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if cand_vecs.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    X_local = np.vstack([q_vec[None, :], cand_vecs]).astype(np.float32, copy=False)
    cand_idx_local = np.arange(1, cand_vecs.shape[0] + 1, dtype=int)
    sel_local, sel_sc = mmr_rerank(
        X_local,
        query_index=0,
        cand_indices=cand_idx_local,
        cand_scores=cand_scores.astype(float, copy=False),
        top_k=int(top_k),
        lambda_=float(lambda_),
        normalize=True,
    )
    sel_pos = (sel_local - 1).astype(int)
    return sel_pos, sel_sc



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed CSV (from preprocess)")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--emotion-config", default="configs/emotion_context.yaml")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--vectors-csv", default="", help="Load precomputed vectors instead of building")
    ap.add_argument("--save-vectors", action="store_true", default=True)
    ap.add_argument("--no-save-vectors", dest="save_vectors", action="store_false")
    ap.add_argument(
        "--save-central-vectors",
        dest="save_central_vectors",
        action="store_true",
        default=True,
        help="Save a central demo-friendly copy under artifacts/vectors/proposed_vectors.csv",
    )
    ap.add_argument(
        "--no-save-central-vectors",
        dest="save_central_vectors",
        action="store_false",
    )
    ap.add_argument(
        "--central-vectors-out",
        default="",
        help="Optional override path for the central proposed vectors CSV",
    )

    ap.add_argument("--rebuild-index", action="store_true", default=False)
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--top-m", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--disable-mmr", action="store_true", default=False)
    ap.add_argument("--mmr-lambda", type=float, default=-1.0)
    ap.add_argument("--ild-space", choices=["emotion_ratio", "embedding", "z"], default="emotion_ratio")
    return ap.parse_args()



def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    cfg = _load_configs(args.eval_config, args.retrieval_config, args.emotion_config)
    run_cfg = {"pipeline": "run_proposed", "params": vars(args), "merged_cfg": cfg}

    art_meta = dump_run_config(run_cfg, prefix="proposed")
    run_id = getattr(art_meta, "run_id", None)
    if run_id is None and isinstance(art_meta, dict):
        run_id = art_meta.get("run_id")
    if run_id is None:
        raise RuntimeError("dump_run_config must return an object/dict with run_id")

    try:
        logger = setup_run_logger(run_id, name="lyrics_reco", also_to_reports=True)
    except TypeError:
        logger = setup_run_logger(run_id)

    art_dir, _ = make_run_dirs(run_id)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (PATHS.root / data_path).resolve()

    meta_df = pd.read_csv(data_path)
    if "song_id" not in meta_df.columns:
        raise ValueError("processed CSV must contain 'song_id' (did preprocess run?)")
    if "genre" not in meta_df.columns:
        meta_df["genre"] = "unknown"

    meta_before = len(meta_df)
    meta_df = meta_df.drop_duplicates(subset=["song_id"], keep="first").reset_index(drop=True)
    meta_after = len(meta_df)
    if meta_after != meta_before:
        logger.warning("Deduplicated meta_df by song_id: %d -> %d", meta_before, meta_after)

    logger.info("Loaded processed data: %s | rows=%d cols=%d", data_path, len(meta_df), len(meta_df.columns))

    # --- Build or load emotion-context vectors ---
    if args.vectors_csv:
        vec_path = Path(args.vectors_csv)
        if not vec_path.is_absolute():
            vec_path = (PATHS.root / vec_path).resolve()
        logger.info("Loading vectors from: %s", vec_path)
        vectors_df = pd.read_csv(vec_path)
    else:
        logger.info("Building emotion-context vectors from cfg: %s", args.emotion_config)
        out_vec = art_dir / "emotion_context_vectors.csv" if args.save_vectors else None
        vectors_df = build_song_vectors_from_df(meta_df, cfg, out_csv=out_vec, paths=PATHS, logger=logger)
        vec_path = out_vec if out_vec is not None else None

    if args.save_central_vectors:
        central_out = args.central_vectors_out or None
        if args.vectors_csv:
            central_path = copy_vector_csv(vec_path, "proposed", out_path=central_out, paths=PATHS)
        else:
            central_path = save_central_vectors(vectors_df, "proposed", out_path=central_out, paths=PATHS)
        logger.info("Saved central proposed vectors: %s", central_path)

    if "song_id" not in vectors_df.columns:
        raise ValueError("vectors_df missing 'song_id'")

    vec_before = len(vectors_df)
    vectors_df = vectors_df.drop_duplicates(subset=["song_id"], keep="first").reset_index(drop=True)
    vec_after = len(vectors_df)
    if vec_after != vec_before:
        logger.warning("Deduplicated vectors_df by song_id: %d -> %d", vec_before, vec_after)

    vec_cols = _vector_cols(vectors_df, prefix="z_")
    logger.info("Vectors ready: rows=%d dim=%d", len(vectors_df), len(vec_cols))

    have = set(vectors_df["song_id"].astype(str).tolist())
    before = len(meta_df)
    meta_df = meta_df[meta_df["song_id"].astype(str).isin(have)].reset_index(drop=True)
    after = len(meta_df)
    if after != before:
        logger.warning("Filtered meta_df to songs with vectors: %d -> %d", before, after)

    vectors_lookup = vectors_df.set_index("song_id", drop=False)
    aligned = vectors_lookup.loc[meta_df["song_id"].astype(str).tolist()]
    Z = aligned[vec_cols].to_numpy(dtype=np.float32)

    emotions = [
        e.lower()
        for e in cfg_get(cfg, ["emotion", "emotions"], ["anger", "fear", "joy", "sadness", "disgust", "trust"])
    ]
    intensity_enabled = bool(cfg_get(cfg, ["intensity", "enabled"], True))
    vad_enabled = bool(cfg_get(cfg, ["vad", "enabled"], True))
    comps = _split_z_components(Z, emotions=emotions, intensity_enabled=intensity_enabled, vad_enabled=vad_enabled)
    emotion_vectors = comps["emotion_ratio"]
    if args.ild_space == "embedding":
        item_vectors = comps["embedding"]
    elif args.ild_space == "z":
        item_vectors = Z
    else:
        item_vectors = comps["emotion_ratio"]

    # --- Open / build VectorDB index ---
    if args.rebuild_index:
        cfg = dict(cfg)
        cfg["index"] = dict(cfg.get("index", {}))
        cfg["index"]["rebuild"] = True

    db = open_vectordb_from_cfg(cfg)
    do_ingest = bool(cfg_get(cfg, ["index", "rebuild"], False)) or (db.count() == 0)
    if do_ingest:
        logger.info("Upserting vectors into VectorDB ...")
        batch_size = int(cfg_get(cfg, ["index", "batch_size"], 5000))
        _upsert_vectors_batched(
            db,
            vectors_df,
            meta_df,
            id_col="song_id",
            vector_prefix="z_",
            metadata_cols=("title", "artist", "year", "genre"),
            batch_size=batch_size,
            logger=logger,
        )
        logger.info("VectorDB upsert done. count=%d", db.count())
    else:
        logger.info("VectorDB already has entries. count=%d (skip upsert)", db.count())

    # --- Query sampling ---
    eval_seed = int(cfg_get(cfg, ["eval", "seed"], args.seed))
    n_queries = int(cfg_get(cfg, ["eval", "n_queries"], 300))
    if args.n_queries and args.n_queries > 0:
        n_queries = int(args.n_queries)
    stratify_by = cfg_get(cfg, ["eval", "query_sampling", "stratify_by"], [])
    min_per_stratum = int(cfg_get(cfg, ["eval", "query_sampling", "min_per_stratum"], 0))
    q_idx = sample_queries(
        meta_df,
        n_queries=n_queries,
        seed=eval_seed,
        stratify_by=stratify_by,
        min_per_stratum=min_per_stratum,
    )
    logger.info("Sampled queries: n=%d", len(q_idx))

    # --- Retrieval settings ---
    top_m = int(cfg_get(cfg, ["retrieval", "top_m"], 200))
    top_k = int(cfg_get(cfg, ["retrieval", "top_k"], 20))
    if args.top_m and args.top_m > 0:
        top_m = int(args.top_m)
    if args.top_k and args.top_k > 0:
        top_k = int(args.top_k)

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
    metric = str(cfg_get(cfg, ["index", "metric"], "cosine"))

    sid_to_index = {str(sid): i for i, sid in enumerate(meta_df["song_id"].astype(str).tolist())}

    # --- Retrieval loop ---
    rec_indices_list: List[np.ndarray] = []
    rec_scores_list: List[np.ndarray] = []
    for t, qi in enumerate(q_idx.tolist(), start=1):
        qi = int(qi)
        q_vec = Z[qi].astype(np.float32, copy=False)

        res = db.query(q_vec, top_k=top_m, where=None, include=["distances"])
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]

        cand_indices: List[int] = []
        cand_scores: List[float] = []
        for sid, dist in zip(ids, dists):
            sid = str(sid)
            if sid not in sid_to_index:
                continue
            cand_indices.append(int(sid_to_index[sid]))
            cand_scores.append(_dist_to_score(float(dist), metric))

        cand_indices_np = np.asarray(cand_indices, dtype=int)
        cand_scores_np = np.asarray(cand_scores, dtype=float)
        cand_indices_np, cand_scores_np = filter_candidates(
            meta_df,
            query_index=qi,
            cand_indices=cand_indices_np,
            cand_scores=cand_scores_np,
            cfg=fcfg,
        )

        if cand_indices_np.size == 0:
            rec_indices_list.append(np.array([], dtype=int))
            rec_scores_list.append(np.array([], dtype=float))
            continue

        cand_song_ids_f = [str(meta_df.iloc[int(i)]["song_id"]) for i in cand_indices_np.tolist()]
        if mmr_enabled and cand_indices_np.size > top_k:
            cand_vecs = np.asarray(vectors_lookup.loc[cand_song_ids_f][vec_cols].to_numpy(dtype=np.float32))
            sel_pos, sel_sc = _local_mmr_on_candidates(
                q_vec,
                cand_vecs,
                cand_scores_np,
                top_k=top_k,
                lambda_=mmr_lambda,
            )
            sel_indices = cand_indices_np[sel_pos]
            sel_scores = sel_sc
        else:
            sel_indices = cand_indices_np[:top_k]
            sel_scores = cand_scores_np[:top_k]

        rec_indices_list.append(sel_indices.astype(int))
        rec_scores_list.append(np.asarray(sel_scores, dtype=float))

        if t % 25 == 0 or t == len(q_idx):
            logger.info("Retrieval progress: %d/%d queries", t, len(q_idx))

    rec_df = build_recommendations_table(meta_df, q_idx, rec_indices_list, rec_scores_list)
    save_csv(rec_df, art_dir / "proposed_recommendations.csv", index=False)

    # --- Evaluation ---
    k_values = tuple(int(x) for x in cfg_get(cfg, ["eval", "k_values"], [5, 10, 20]))
    eval_cfg = _safe_dataclass_init(EvalConfig, k_values=k_values)

    pseudo_kwargs = dict(
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
    pseudo_cfg = _safe_dataclass_init(PseudoGTConfig, **pseudo_kwargs)

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
