"""
lyrics_reco.vectordb.chroma

Thin wrapper around ChromaDB (optional dependency).

Install:
  pip install chromadb

Notes:
- cosine metric이면 normalize_before_upsert=True로 저장 권장
- query 결과는 distances(낮을수록 유사)로 오므로, 필요하면 score=1-distance로 변환해서 씀
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import IndexConfig
from .utils import l2_normalize_rows, batched_indices


class ChromaVectorDB:
    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg
        try:
            import chromadb
        except Exception as e:
            raise ImportError(
                "chromadb is required for the 'chroma' backend. Install: pip install chromadb"
            ) from e

        self.persist_dir = Path(cfg.persist_dir)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        if cfg.rebuild:
            try:
                self.client.delete_collection(name=cfg.collection_name)
            except Exception:
                pass

        md = None
        m = cfg.metric.lower()
        if m in ("cosine", "l2", "ip"):
            md = {"hnsw:space": ("cosine" if m == "cosine" else ("l2" if m == "l2" else "ip"))}

        self.collection = self.client.get_or_create_collection(
            name=cfg.collection_name,
            metadata=md,
        )

    def count(self) -> int:
        return int(self.collection.count())

    def upsert(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        *,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        documents: Optional[Sequence[str]] = None,
    ) -> None:
        ids = [str(x) for x in ids]
        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] != len(ids):
            raise ValueError("vectors must be shape (N, D) and align to ids")

        if self.cfg.metric.lower() == "cosine" and self.cfg.normalize_before_upsert:
            X = l2_normalize_rows(X)

        self.collection.upsert(
            ids=ids,
            embeddings=X.tolist(),  # Chroma expects list[list[float]]
            metadatas=list(metadatas) if metadatas is not None else None,
            documents=list(documents) if documents is not None else None,
        )

    def upsert_batched(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        *,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        documents: Optional[Sequence[str]] = None,
    ) -> None:
        n = len(ids)
        for rg in batched_indices(n, int(self.cfg.batch_size)):
            i0, i1 = rg.start, rg.stop
            md = metadatas[i0:i1] if metadatas is not None else None
            docs = documents[i0:i1] if documents is not None else None
            self.upsert(ids[i0:i1], vectors[i0:i1], metadatas=md, documents=docs)

    def query(
        self,
        query_vectors: np.ndarray,
        *,
        top_k: int = 20,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        Q = np.asarray(query_vectors, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[None, :]

        if self.cfg.metric.lower() == "cosine" and self.cfg.normalize_before_upsert:
            Q = l2_normalize_rows(Q)

        include = include or ["metadatas", "distances"]
        return self.collection.query(
            query_embeddings=Q.tolist(),
            n_results=int(top_k),
            where=where,
            include=include,
        )

    def get(self, ids: Sequence[str], *, include: Optional[List[str]] = None) -> Dict[str, Any]:
        include = include or ["metadatas", "documents", "embeddings"]
        return self.collection.get(ids=[str(x) for x in ids], include=include)

    def delete(self, ids: Sequence[str]) -> None:
        self.collection.delete(ids=[str(x) for x in ids])