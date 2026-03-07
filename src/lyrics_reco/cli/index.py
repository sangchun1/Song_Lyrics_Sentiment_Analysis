"""Build/load emotion-context vectors and upsert them into the VectorDB.

This command is useful when you want to prepare the retrieval index once,
then reuse it later from `demo.py` or another query script.

Examples
--------
Build vectors from processed data and rebuild the index:

python -m lyrics_reco.cli index \
    --data data/processed/genius_processed.csv \
    --emotion-config configs/emotion_context.yaml \
    --retrieval-config configs/retrieval.yaml \
    --rebuild-index

Reuse an existing vectors CSV and only upsert to the DB:

python -m lyrics_reco.cli index \
    --data data/processed/genius_processed.csv \
    --vectors-csv artifacts/runs/proposed_xxx/emotion_context_vectors.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from ..common.config import dump_run_config, load_yaml
from ..common.logging import setup_run_logger
from ..common.paths import PATHS
from ..common.seed import set_seed
from ..emotion_context.builder import build_song_vectors_from_df
from ..pipeline.utils import make_run_dirs
from ..vectordb.factory import open_vectordb_from_cfg


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

    def _key(col: str) -> int:
        try:
            return int(col.split("_", 1)[1])
        except Exception:
            return 10**9

    return sorted(cols, key=_key)


def _to_python_scalar(v: Any) -> Any:
    if pd.isna(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


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
                md = {c: _to_python_scalar(row[c]) for c in metadata_cols if c in meta_lookup.columns}
                metadatas.append(md)
            else:
                metadatas.append({})

        db.upsert(ids, X, metadatas=metadatas)
        if logger:
            logger.info("Index upsert: %d/%d", i1, n)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed CSV (from preprocess)")
    ap.add_argument("--emotion-config", default="configs/emotion_context.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--vectors-csv",
        default="",
        help="if provided, load precomputed vectors instead of rebuilding them",
    )
    ap.add_argument("--save-vectors", action="store_true", default=True)
    ap.add_argument("--no-save-vectors", dest="save_vectors", action="store_false")
    ap.add_argument("--rebuild-index", action="store_true", default=False)
    args = ap.parse_args(list(argv) if argv is not None else None)

    set_seed(args.seed)
    cfg = _load_configs(args.retrieval_config, args.emotion_config)
    run_cfg = {"pipeline": "cli.index", "params": vars(args), "merged_cfg": cfg}
    art_meta = dump_run_config(run_cfg, prefix="index")
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
        raise ValueError("processed CSV must contain 'song_id'")
    if "genre" not in meta_df.columns:
        meta_df["genre"] = "unknown"

    before = len(meta_df)
    meta_df = meta_df.drop_duplicates(subset=["song_id"], keep="first").reset_index(drop=True)
    after = len(meta_df)
    if before != after:
        logger.warning("Deduplicated meta_df by song_id: %d -> %d", before, after)

    logger.info("Loaded processed data: %s | rows=%d cols=%d", data_path, len(meta_df), len(meta_df.columns))

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

    if "song_id" not in vectors_df.columns:
        raise ValueError("vectors_df missing 'song_id'")

    vec_before = len(vectors_df)
    vectors_df = vectors_df.drop_duplicates(subset=["song_id"], keep="first").reset_index(drop=True)
    vec_after = len(vectors_df)
    if vec_before != vec_after:
        logger.warning("Deduplicated vectors_df by song_id: %d -> %d", vec_before, vec_after)

    vec_cols = _vector_cols(vectors_df)
    logger.info("Vectors ready: rows=%d dim=%d", len(vectors_df), len(vec_cols))

    if args.rebuild_index:
        cfg = dict(cfg)
        cfg["index"] = dict(cfg.get("index", {}))
        cfg["index"]["rebuild"] = True

    db = open_vectordb_from_cfg(cfg)
    do_ingest = bool(cfg.get("index", {}).get("rebuild", False)) or (db.count() == 0)

    if do_ingest:
        logger.info("Upserting vectors into VectorDB ...")
        batch_size = int(cfg.get("index", {}).get("batch_size", 5000))
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

    logger.info("Done. Artifacts in %s", art_dir)


if __name__ == "__main__":
    main()
