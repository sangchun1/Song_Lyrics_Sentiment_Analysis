"""
lyrics_reco.evaluation.metrics

Metric implementations:
- Recall@K
- NDCG@K (graded relevance)
- Emotion Consistency@K
- ILD@K (intra-list diversity)

All functions are designed to work per-query and aggregate later.

Important:
- For Emotion Consistency and ILD you need vectors:
  - emotion vectors for EC (e.g., emotion ratios or context vectors)
  - item vectors for ILD (same space as retrieval, typically)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse


ArrayLike = Union[np.ndarray, sparse.spmatrix]


def recall_at_k(rec_indices: np.ndarray, relevant_set: np.ndarray, k: int) -> float:
    """
    Recall@K = |rec[:K] ∩ relevant| / |relevant|
    If |relevant|==0 => 0.0
    """
    rel = set(int(x) for x in relevant_set.tolist())
    if len(rel) == 0:
        return 0.0
    top = rec_indices[: int(k)].tolist()
    hit = sum(1 for x in top if int(x) in rel)
    return float(hit) / float(len(rel))


def _dcg(rels: Sequence[int]) -> float:
    # standard DCG with exponential gains
    dcg = 0.0
    for i, r in enumerate(rels, start=1):
        if r <= 0:
            continue
        dcg += (2.0 ** float(r) - 1.0) / np.log2(i + 1.0)
    return float(dcg)


def ndcg_at_k(rec_indices: np.ndarray, grade_map: Dict[int, int], k: int) -> float:
    """
    NDCG@K for a single query.
    grade_map: cand_index -> grade (0/1/2..)
    """
    k = int(k)
    rec_k = rec_indices[:k].tolist()
    rels = [int(grade_map.get(int(i), 0)) for i in rec_k]
    dcg = _dcg(rels)

    # Ideal: sort all grades descending and take top-k
    if not grade_map:
        return 0.0
    all_grades = sorted((int(g) for g in grade_map.values() if int(g) > 0), reverse=True)
    idcg = _dcg(all_grades[:k]) if all_grades else 0.0
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def _l2_normalize_rows_dense(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms


def _cosine_dense(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))


def _cosine_sparse(u: sparse.spmatrix, v: sparse.spmatrix) -> float:
    num = float(u @ v.T)
    nu = float(np.sqrt(u.multiply(u).sum()))
    nv = float(np.sqrt(v.multiply(v).sum()))
    return num / max(nu * nv, 1e-12)


def emotion_consistency_at_k(
    emotion_vectors: ArrayLike,
    query_index: int,
    rec_indices: np.ndarray,
    *,
    k: int,
) -> float:
    """
    Emotion Consistency@K as average cosine similarity between query emotion vector and each rec emotion vector.
    """
    k = int(k)
    rec_k = rec_indices[:k].astype(int)
    if rec_k.size == 0:
        return 0.0

    if sparse.issparse(emotion_vectors):
        qv = emotion_vectors[int(query_index)]
        sims = []
        for ri in rec_k.tolist():
            sims.append(_cosine_sparse(qv, emotion_vectors[int(ri)]))
        return float(np.mean(sims)) if sims else 0.0

    X = np.asarray(emotion_vectors, dtype=float)
    q = X[int(query_index)]
    sims = []
    for ri in rec_k.tolist():
        sims.append(_cosine_dense(q, X[int(ri)]))
    return float(np.mean(sims)) if sims else 0.0


def ild_at_k(
    item_vectors: ArrayLike,
    rec_indices: np.ndarray,
    *,
    k: int,
    distance: str = "1-cosine",  # 1-cosine | l2
) -> float:
    """
    Intra-List Diversity@K.

    For K items, compute average pairwise distance among the recommended set.
    - distance="1-cosine": 1 - cosine similarity
    - distance="l2": Euclidean distance
    """
    k = int(k)
    idx = rec_indices[:k].astype(int)
    if idx.size <= 1:
        return 0.0

    dist = distance.lower()
    # extract sub-matrix
    if sparse.issparse(item_vectors):
        V = item_vectors[idx].tocsr()
        # normalize for cosine distance
        if dist in {"1-cosine", "cosine"}:
            norms = np.sqrt(V.multiply(V).sum(axis=1)).A1
            norms = np.maximum(norms, 1e-12)
            Vn = sparse.diags(1.0 / norms) @ V
            # pairwise sims: (KxK)
            S = (Vn @ Vn.T).toarray()
            # average upper triangle (excluding diagonal)
            K = S.shape[0]
            tri = np.triu_indices(K, k=1)
            sims = S[tri]
            return float(np.mean(1.0 - sims))
        else:
            # L2 pairwise distances (small K => dense ok)
            A = V.toarray()
            K = A.shape[0]
            dists = []
            for i in range(K):
                for j in range(i + 1, K):
                    dists.append(float(np.linalg.norm(A[i] - A[j])))
            return float(np.mean(dists)) if dists else 0.0

    # dense
    A = np.asarray(item_vectors, dtype=float)[idx]
    K = A.shape[0]
    dists = []
    if dist in {"1-cosine", "cosine"}:
        An = _l2_normalize_rows_dense(A)
        for i in range(K):
            for j in range(i + 1, K):
                dists.append(1.0 - float(np.dot(An[i], An[j])))
        return float(np.mean(dists)) if dists else 0.0
    else:
        for i in range(K):
            for j in range(i + 1, K):
                dists.append(float(np.linalg.norm(A[i] - A[j])))
        return float(np.mean(dists)) if dists else 0.0


def aggregate_metrics_table(per_query: pd.DataFrame, *, k_values: Sequence[int]) -> pd.DataFrame:
    """
    Aggregate per-query metrics into a single-row-per-metric-per-k table.

    Expected per_query columns:
      - query_index
      - recall@{k}, ndcg@{k}, ec@{k}, ild@{k}
    """
    rows = []
    for k in k_values:
        k = int(k)
        for m in ["recall", "ndcg", "ec", "ild"]:
            col = f"{m}@{k}"
            if col not in per_query.columns:
                continue
            vals = pd.to_numeric(per_query[col], errors="coerce").dropna().to_numpy()
            rows.append({
                "metric": m,
                "k": k,
                "mean": float(np.mean(vals)) if vals.size else 0.0,
                "std": float(np.std(vals)) if vals.size else 0.0,
                "n": int(vals.size),
            })
    return pd.DataFrame(rows)
