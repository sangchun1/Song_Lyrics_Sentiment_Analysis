"""
lyrics_reco.vectordb.query

Convenience query helpers.

Distances -> scores:
- cosine: score = 1 - distance (higher is better)
- l2:     score = -distance
- ip:     score = distance (backend dependent)
"""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .factory import open_vectordb_from_cfg
from .utils import cfg_get


def _dist_to_score(dist: float, metric: str) -> float:
    m = metric.lower()
    if m == "cosine":
        return float(1.0 - dist)
    if m in ("l2", "euclidean"):
        return float(-dist)
    return float(dist)


def query_by_vector(
    cfg: Mapping[str, Any],
    query_vectors: np.ndarray,
    *,
    top_k: int = 20,
    where: Optional[Dict[str, Any]] = None,
    include_metadatas: bool = True,
) -> List[Dict[str, Any]]:
    db = open_vectordb_from_cfg(cfg)
    metric = str(cfg_get(cfg, ["index", "metric"], "cosine"))

    include = ["distances"]
    if include_metadatas:
        include.append("metadatas")

    res = db.query(query_vectors, top_k=top_k, where=where, include=include)

    out: List[Dict[str, Any]] = []
    ids_list = res.get("ids", [])
    dists_list = res.get("distances", [])
    meta_list = res.get("metadatas", None)

    for qi in range(len(ids_list)):
        ids = ids_list[qi]
        dists = dists_list[qi] if qi < len(dists_list) else []
        metas = meta_list[qi] if meta_list is not None and qi < len(meta_list) else [None] * len(ids)
        for sid, dist, md in zip(ids, dists, metas):
            out.append(
                {
                    "query_i": qi,
                    "song_id": sid,
                    "distance": float(dist),
                    "score": _dist_to_score(float(dist), metric),
                    "metadata": md,
                }
            )
    return out


def query_by_id(
    cfg: Mapping[str, Any],
    song_id: str,
    *,
    top_k: int = 20,
    where: Optional[Dict[str, Any]] = None,
    include_metadatas: bool = True,
) -> List[Dict[str, Any]]:
    db = open_vectordb_from_cfg(cfg)
    got = db.get([song_id], include=["embeddings"])
    emb = got.get("embeddings", [])
    if not emb:
        return []
    q = np.asarray(emb[0], dtype=np.float32)
    return query_by_vector(cfg, q, top_k=top_k, where=where, include_metadatas=include_metadatas)