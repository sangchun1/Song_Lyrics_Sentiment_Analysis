"""
lyrics_reco.retrieval.cosine

Cosine Top-K retrieval for:
- dense numpy arrays (N, D)
- sparse CSR matrices (N, V) e.g., TF-IDF

Design:
- No full similarity matrix.
- Compute similarities for one query or a batch of queries.
- Optional row-normalization for true cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np
from scipy import sparse


ArrayLike = Union[np.ndarray, sparse.spmatrix]


def l2_normalize_rows(X: ArrayLike) -> ArrayLike:
    """Row-wise L2 normalization."""
    if sparse.issparse(X):
        X = X.tocsr()
        norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
        norms = np.maximum(norms, 1e-12)
        inv = 1.0 / norms
        return sparse.diags(inv) @ X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms


def topk_cosine(
    X: ArrayLike,
    query_index: int,
    *,
    top_k: int = 20,
    exclude_self: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (topk_indices, topk_scores) for a single query row.

    scores are cosine similarities if normalize=True else dot products.
    """
    Xn = l2_normalize_rows(X) if normalize else X

    if sparse.issparse(Xn):
        q = Xn[query_index]  # (1, V)
        sims = (Xn @ q.T).toarray().ravel()
    else:
        q = Xn[query_index]  # (D,)
        sims = Xn @ q

    if exclude_self and 0 <= query_index < sims.shape[0]:
        sims[query_index] = -np.inf

    k = min(int(top_k), sims.shape[0])
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.astype(int), sims[idx].astype(float)


def batch_topk_cosine(
    X: ArrayLike,
    query_indices: Sequence[int],
    *,
    top_k: int = 20,
    exclude_self: bool = True,
    normalize: bool = True,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Batch version: list of (query_index, topk_indices, topk_scores)."""
    out: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for qi in query_indices:
        idx, sc = topk_cosine(X, qi, top_k=top_k, exclude_self=exclude_self, normalize=normalize)
        out.append((int(qi), idx, sc))
    return out
