"""
lyrics_reco.retrieval.filters

Metadata-based candidate filtering.

This is intentionally lightweight and generic:
- exclude_self
- exclude_same_artist
- year_window (|year_q - year_c| <= window)

Use this after retrieving a candidate pool (Top-M),
then optionally apply MMR to select Top-K from filtered candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FilterConfig:
    exclude_self: bool = True
    exclude_same_artist: bool = False
    year_window: Optional[int] = None  # e.g., 10 means +/-10 years; None disables

    # Column names in meta_df
    song_id_col: str = "song_id"
    artist_col: str = "artist"
    year_col: str = "year"


def filter_candidates(
    meta_df: pd.DataFrame,
    *,
    query_index: int,
    cand_indices: np.ndarray,
    cand_scores: np.ndarray,
    cfg: FilterConfig = FilterConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter candidate indices/scores based on cfg and meta_df.

    Returns:
        (filtered_indices, filtered_scores) aligned.
    """
    if len(cand_indices) != len(cand_scores):
        raise ValueError("cand_indices and cand_scores must have same length")

    keep = np.ones(len(cand_indices), dtype=bool)

    if cfg.exclude_self:
        keep &= (cand_indices != int(query_index))

    if cfg.exclude_same_artist and cfg.artist_col in meta_df.columns:
        q_artist = meta_df.iloc[int(query_index)][cfg.artist_col]
        # handle NaN as string mismatch-safe
        keep &= meta_df.iloc[cand_indices][cfg.artist_col].astype(str).values != str(q_artist)

    if cfg.year_window is not None and cfg.year_col in meta_df.columns:
        q_year = meta_df.iloc[int(query_index)][cfg.year_col]
        try:
            qy = int(q_year)
            yrs = pd.to_numeric(meta_df.iloc[cand_indices][cfg.year_col], errors="coerce").fillna(-10**9).astype(int).values
            keep &= (np.abs(yrs - qy) <= int(cfg.year_window))
        except Exception:
            # if query year invalid, skip year filtering
            pass

    return cand_indices[keep], cand_scores[keep]
