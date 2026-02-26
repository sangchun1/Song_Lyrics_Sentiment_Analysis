"""
lyrics_reco.pipeline.utils

Small helpers used by pipeline scripts:
- nested config access with defaults
- query sampling (random or stratified)
- building run output directories
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..common.paths import PATHS, ProjectPaths, ensure_dir


def cfg_get(cfg: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """Safely get nested config value."""
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def make_run_dirs(run_id: str, *, paths: ProjectPaths = PATHS) -> Tuple[Path, Path]:
    """Return (artifacts_run_dir, reports_run_dir)."""
    art = ensure_dir(paths.art_runs / run_id)
    rep = ensure_dir(paths.rep_runs / run_id)
    return art, rep


def sample_queries(
    meta_df: pd.DataFrame,
    *,
    n_queries: int,
    seed: int = 42,
    stratify_by: Optional[Sequence[str]] = None,
    min_per_stratum: int = 0,
) -> np.ndarray:
    """
    Sample query indices.

    If stratify_by is provided (e.g., ["genre","decade"]), do:
    - groupby those cols
    - sample up to min_per_stratum from each group
    - then fill remaining quota with random samples from the rest
    """
    rng = np.random.default_rng(int(seed))
    N = len(meta_df)
    n_queries = min(int(n_queries), N)
    if n_queries <= 0:
        return np.array([], dtype=int)

    if not stratify_by:
        return rng.choice(np.arange(N), size=n_queries, replace=False).astype(int)

    cols = [c for c in stratify_by if c in meta_df.columns]
    if not cols:
        return rng.choice(np.arange(N), size=n_queries, replace=False).astype(int)

    # Create a stratum key (avoid NaNs)
    tmp = meta_df[cols].copy()
    for c in cols:
        tmp[c] = tmp[c].astype(str).fillna("")
    key = tmp.apply(lambda r: "||".join(r.values.tolist()), axis=1)

    # Sample per stratum
    selected = []
    used = np.zeros(N, dtype=bool)

    if min_per_stratum and min_per_stratum > 0:
        for k, idx in key.groupby(key).groups.items():
            idx = np.array(list(idx), dtype=int)
            take = min(int(min_per_stratum), idx.size)
            if take <= 0:
                continue
            pick = rng.choice(idx, size=take, replace=False)
            selected.extend(pick.tolist())
            used[pick] = True

    # Fill remaining
    remaining = n_queries - len(selected)
    if remaining > 0:
        pool = np.where(~used)[0]
        if pool.size > 0:
            pick = rng.choice(pool, size=min(remaining, pool.size), replace=False)
            selected.extend(pick.tolist())

    # If overshoot, trim
    if len(selected) > n_queries:
        selected = rng.choice(np.array(selected), size=n_queries, replace=False).tolist()

    return np.array(selected, dtype=int)
