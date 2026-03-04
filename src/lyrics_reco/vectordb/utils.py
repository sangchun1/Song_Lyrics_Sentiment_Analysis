"""
lyrics_reco.vectordb.utils
"""

from __future__ import annotations
from typing import Any, Mapping, Sequence, List
import numpy as np


def cfg_get(cfg: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def batched_indices(n: int, batch_size: int) -> List[range]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [range(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (X / norms).astype(np.float32)