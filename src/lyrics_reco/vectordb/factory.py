"""
lyrics_reco.vectordb.factory

Open a vector DB using configs/retrieval.yaml 'index' section.

Expected retrieval.yaml:
  index:
    backend: chroma
    persist_dir: artifacts/indexes/chroma
    collection_name: lyrics_reco
    metric: cosine
    rebuild: false
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping

from ..common.paths import PATHS, ProjectPaths
from .config import IndexConfig
from .utils import cfg_get
from .chroma import ChromaVectorDB


def open_vectordb_from_cfg(cfg: Mapping[str, Any], *, paths: ProjectPaths = PATHS) -> ChromaVectorDB:
    backend = str(cfg_get(cfg, ["index", "backend"], "chroma")).lower()

    persist_dir = cfg_get(cfg, ["index", "persist_dir"], str(paths.art_indexes / "chroma"))
    p = Path(str(persist_dir))
    if not p.is_absolute():
        p = (paths.root / p).resolve()

    icfg = IndexConfig(
        backend=backend,
        persist_dir=str(p),
        collection_name=str(cfg_get(cfg, ["index", "collection_name"], "lyrics_reco")),
        metric=str(cfg_get(cfg, ["index", "metric"], "cosine")),
        rebuild=bool(cfg_get(cfg, ["index", "rebuild"], False)),
        batch_size=int(cfg_get(cfg, ["index", "batch_size"], 5000)),
        normalize_before_upsert=bool(cfg_get(cfg, ["index", "normalize_before_upsert"], True)),
        id_col=str(cfg_get(cfg, ["index", "id_col"], "song_id")),
    )

    if backend != "chroma":
        raise ValueError(f"Unsupported index backend: {backend}")

    return ChromaVectorDB(icfg)