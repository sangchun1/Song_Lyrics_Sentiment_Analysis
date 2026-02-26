"""
lyrics_reco.baseline.similarity

Compatibility wrapper around lyrics_reco.retrieval.cosine.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from ..retrieval.cosine import topk_cosine as _topk
from ..retrieval.cosine import batch_topk_cosine as _batch

ArrayLike = Union[np.ndarray, sparse.spmatrix]


def topk_cosine_for_index(
    X: ArrayLike,
    query_index: int,
    *,
    top_k: int = 20,
    exclude_self: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    return _topk(X, query_index, top_k=top_k, exclude_self=exclude_self, normalize=normalize)


def batch_topk_cosine(
    X: ArrayLike,
    query_indices: Sequence[int],
    *,
    top_k: int = 20,
    exclude_self: bool = True,
    normalize: bool = True,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    return _batch(X, query_indices, top_k=top_k, exclude_self=exclude_self, normalize=normalize)