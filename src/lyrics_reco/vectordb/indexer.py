"""
lyrics_reco.vectordb.indexer

Build / upsert a vector DB from:
  - vectors_df: [song_id, z_0, z_1, ...]
  - meta_df: processed songs with metadata columns
"""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .factory import open_vectordb_from_cfg


def _extract_vectors(
    vectors_df: pd.DataFrame,
    *,
    id_col: str = "song_id",
    prefix: str = "z_",
) -> Tuple[List[str], np.ndarray]:
    if id_col not in vectors_df.columns:
        raise ValueError(f"vectors_df missing id_col: {id_col}")

    vec_cols = [c for c in vectors_df.columns if c.startswith(prefix)]
    if not vec_cols:
        raise ValueError(f"No vector columns found with prefix '{prefix}'")

    ids = vectors_df[id_col].astype(str).tolist()
    X = vectors_df[vec_cols].to_numpy(dtype=np.float32)
    return ids, X


def _build_metadatas(
    meta_df: pd.DataFrame,
    ids: Sequence[str],
    *,
    id_col: str = "song_id",
    keep_cols: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    if keep_cols is None:
        keep_cols = ["title", "artist", "year", "genre"]
    keep_cols = [c for c in keep_cols if c in meta_df.columns]

    m = meta_df.set_index(id_col, drop=False)
    out: List[Dict[str, Any]] = []
    for sid in ids:
        if sid in m.index:
            row = m.loc[sid]
            md = {c: (None if pd.isna(row[c]) else row[c]) for c in keep_cols}
        else:
            md = {}
        out.append(md)
    return out


def build_index_from_frames(
    vectors_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    id_col: str = "song_id",
    vector_prefix: str = "z_",
    metadata_cols: Optional[Sequence[str]] = None,
    documents_col: Optional[str] = None,
):
    """
    Build / upsert into Chroma.

    - vectors_df: output of EmotionContextBuilder (song_id + z_* columns)
    - meta_df: processed genius csv (song_id + metadata columns)
    """
    db = open_vectordb_from_cfg(cfg)
    ids, X = _extract_vectors(vectors_df, id_col=id_col, prefix=vector_prefix)

    metadatas = _build_metadatas(meta_df, ids, id_col=id_col, keep_cols=metadata_cols)

    documents = None
    if documents_col is not None and documents_col in meta_df.columns:
        m = meta_df.set_index(id_col, drop=False)
        documents = [str(m.loc[sid][documents_col]) if sid in m.index else "" for sid in ids]

    db.upsert_batched(ids, X, metadatas=metadatas, documents=documents)
    return db