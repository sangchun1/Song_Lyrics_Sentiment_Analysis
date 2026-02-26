"""
lyrics_reco.retrieval.mmr

Maximal Marginal Relevance (MMR) re-ranking.

Given:
- query vector q
- candidate vectors C (Top-M)
MMR selects items iteratively:
    argmax_{c in remaining}  lambda * sim(q, c) - (1-lambda) * max_{s in selected} sim(c, s)

This encourages diversity while keeping relevance.

Implementation supports:
- dense numpy vectors
- sparse CSR rows (works but can be slower)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import sparse


ArrayLike = Union[np.ndarray, sparse.spmatrix]


def _row_norm(x: ArrayLike) -> float:
    if sparse.issparse(x):
        return float(np.sqrt(x.multiply(x).sum()))
    return float(np.linalg.norm(x))


def _cosine(u: ArrayLike, v: ArrayLike, *, eps: float = 1e-12) -> float:
    # u and v are single vectors (1D dense or 1xD sparse)
    if sparse.issparse(u) or sparse.issparse(v):
        if not sparse.issparse(u):
            u = sparse.csr_matrix(u)
        if not sparse.issparse(v):
            v = sparse.csr_matrix(v)
        num = float(u @ v.T)
        den = max(_row_norm(u) * _row_norm(v), eps)
        return num / den
    num = float(np.dot(u, v))
    den = max(float(np.linalg.norm(u)) * float(np.linalg.norm(v)), eps)
    return num / den


def mmr_rerank(
    X: ArrayLike,
    query_index: int,
    cand_indices: np.ndarray,
    cand_scores: Optional[np.ndarray] = None,
    *,
    top_k: int = 20,
    lambda_: float = 0.7,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MMR re-rank candidates into Top-K.

    Parameters
    ----------
    X : (N, D) dense or sparse
    query_index : int
    cand_indices : array of candidate row indices (Top-M)
    cand_scores : optional array of sim(query, cand) aligned to cand_indices.
                  If not provided, will compute.
    top_k : number of items to select
    lambda_ : relevance/diversity tradeoff
    normalize : if True, cosine similarity is used (by normalizing inside sim),
                otherwise dot product is used (still via cosine helper for stability).

    Returns
    -------
    (selected_indices, selected_scores)
    - selected_scores are similarities to the query (not the MMR objective).
    """
    if cand_indices.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    lam = float(lambda_)
    lam = min(max(lam, 0.0), 1.0)

    M = cand_indices.astype(int)
    K = min(int(top_k), M.size)

    # Prepare query vector
    q = X[int(query_index)]
    if sparse.issparse(X):
        qv = q
    else:
        qv = np.asarray(q)

    # Precompute sim(q, c) if not provided
    if cand_scores is None:
        sims_q = np.array([_cosine(qv, X[i]) for i in M], dtype=float)
    else:
        sims_q = np.asarray(cand_scores, dtype=float)
        if sims_q.shape[0] != M.shape[0]:
            raise ValueError("cand_scores must align with cand_indices")

    selected: List[int] = []
    selected_qs: List[float] = []

    remaining = list(range(M.size))  # positions into M/sims_q

    # For speed, if X is dense, cache candidate vectors
    if not sparse.issparse(X):
        cand_vecs = X[M]
    else:
        cand_vecs = None  # access by X[M[pos]]

    while remaining and len(selected) < K:
        best_pos = None
        best_obj = -1e18

        for pos in remaining:
            rel = sims_q[pos]

            if not selected:
                div = 0.0
            else:
                # max similarity to already selected items
                if cand_vecs is not None:
                    v = cand_vecs[pos]
                else:
                    v = X[M[pos]]

                max_sim = -1e18
                for sel_pos in selected:
                    if cand_vecs is not None:
                        svec = cand_vecs[sel_pos]
                    else:
                        svec = X[M[sel_pos]]
                    sim_cs = _cosine(v, svec)
                    if sim_cs > max_sim:
                        max_sim = sim_cs
                div = max_sim

            obj = lam * rel - (1.0 - lam) * div
            if obj > best_obj:
                best_obj = obj
                best_pos = pos

        assert best_pos is not None
        selected.append(best_pos)
        selected_qs.append(float(sims_q[best_pos]))
        remaining.remove(best_pos)

    sel_indices = M[np.array(selected, dtype=int)]
    sel_scores = np.array(selected_qs, dtype=float)
    return sel_indices, sel_scores
