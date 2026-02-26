"""
lyrics_reco.evaluation.pseudo_gt

Pseudo ground-truth with safeguards for graded relevance.

- grade 2: same genre AND within year window
- grade 1: same genre only (OPTIONAL; set grade_if_same_genre_only>0)
  You can cap grade1 with max_grade1_per_query to avoid huge dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PseudoGTConfig:
    year_window: Optional[int] = 10
    require_same_genre: bool = True
    exclude_self: bool = True
    exclude_same_artist: bool = True

    graded_enabled: bool = True
    grade_if_same_genre_and_within_year: int = 2
    grade_if_same_genre_only: int = 0  # default OFF
    max_grade1_per_query: int = 0      # 0=no cap

    song_id_col: str = "song_id"
    artist_col: str = "artist"
    year_col: str = "year"
    genre_col: str = "genre"


class GenreYearIndex:
    def __init__(self, meta_df: pd.DataFrame, *, year_col: str = "year", genre_col: str = "genre"):
        if year_col not in meta_df.columns or genre_col not in meta_df.columns:
            raise ValueError(f"meta_df must contain '{year_col}' and '{genre_col}'")

        years = pd.to_numeric(meta_df[year_col], errors="coerce").fillna(-10**9).astype(int).to_numpy()
        genres = meta_df[genre_col].astype(str).fillna("").to_numpy()

        self._map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for g in np.unique(genres):
            if g == "" or g.lower() == "nan":
                continue
            idx = np.where(genres == g)[0]
            if idx.size == 0:
                continue
            ys = years[idx]
            order = np.argsort(ys)
            self._map[g] = (ys[order], idx[order])

        order_all = np.argsort(years)
        self._all = (years[order_all], order_all)

    def slice(self, genre: Optional[str], year_min: Optional[int], year_max: Optional[int]) -> np.ndarray:
        if genre is None:
            ys, idxs = self._all
        else:
            pair = self._map.get(str(genre))
            if pair is None:
                return np.array([], dtype=int)
            ys, idxs = pair

        if year_min is None or year_max is None:
            return idxs.copy()

        lo = np.searchsorted(ys, int(year_min), side="left")
        hi = np.searchsorted(ys, int(year_max), side="right")
        return idxs[lo:hi].copy()


def _get_int(meta_df: pd.DataFrame, idx: int, col: str) -> Optional[int]:
    try:
        v = meta_df.iloc[int(idx)][col]
        if pd.isna(v):
            return None
        return int(v)
    except Exception:
        return None


def _get_str(meta_df: pd.DataFrame, idx: int, col: str) -> str:
    try:
        v = meta_df.iloc[int(idx)][col]
        return "" if pd.isna(v) else str(v)
    except Exception:
        return ""


def build_pseudo_ground_truth(
    meta_df: pd.DataFrame,
    query_indices: Sequence[int],
    *,
    cfg: PseudoGTConfig = PseudoGTConfig(),
    index: Optional[GenreYearIndex] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
    needed = {cfg.year_col, cfg.genre_col}
    missing = [c for c in needed if c not in meta_df.columns]
    if missing:
        raise ValueError(f"meta_df missing columns: {missing}")

    idxer = index or GenreYearIndex(meta_df, year_col=cfg.year_col, genre_col=cfg.genre_col)

    relevant_map: Dict[int, np.ndarray] = {}
    grade_map: Dict[int, Dict[int, int]] = {}

    for qi in query_indices:
        qi = int(qi)
        q_genre = _get_str(meta_df, qi, cfg.genre_col) if cfg.require_same_genre else None
        q_year = _get_int(meta_df, qi, cfg.year_col)

        if cfg.year_window is None or q_year is None:
            cand = idxer.slice(q_genre, None, None)
        else:
            w = int(cfg.year_window)
            cand = idxer.slice(q_genre, q_year - w, q_year + w)

        if cfg.exclude_self:
            cand = cand[cand != qi]

        if cfg.exclude_same_artist and cfg.artist_col in meta_df.columns:
            q_artist = _get_str(meta_df, qi, cfg.artist_col)
            if q_artist:
                artists = meta_df.iloc[cand][cfg.artist_col].astype(str).to_numpy()
                cand = cand[artists != q_artist]

        gm: Dict[int, int] = {}
        if cfg.graded_enabled:
            for ci in cand.tolist():
                gm[int(ci)] = int(cfg.grade_if_same_genre_and_within_year)

            # OPTIONAL grade1 (same genre but outside window)
            if (
                cfg.require_same_genre
                and cfg.year_window is not None
                and q_year is not None
                and int(cfg.grade_if_same_genre_only) > 0
            ):
                all_same_genre = idxer.slice(q_genre, None, None)

                if cfg.exclude_self:
                    all_same_genre = all_same_genre[all_same_genre != qi]

                if cfg.exclude_same_artist and cfg.artist_col in meta_df.columns:
                    q_artist = _get_str(meta_df, qi, cfg.artist_col)
                    if q_artist:
                        artists2 = meta_df.iloc[all_same_genre][cfg.artist_col].astype(str).to_numpy()
                        all_same_genre = all_same_genre[artists2 != q_artist]

                years2 = pd.to_numeric(meta_df.iloc[all_same_genre][cfg.year_col], errors="coerce").fillna(-10**9).astype(int).to_numpy()
                w = int(cfg.year_window)
                out_idx = all_same_genre[np.abs(years2 - int(q_year)) > w]

                cap = int(cfg.max_grade1_per_query) if int(cfg.max_grade1_per_query) > 0 else None
                if cap is not None:
                    out_idx = out_idx[:cap]

                for ci in out_idx.tolist():
                    if int(ci) not in gm:
                        gm[int(ci)] = int(cfg.grade_if_same_genre_only)

        strong = (
            np.array([ci for ci, gr in gm.items() if gr >= int(cfg.grade_if_same_genre_and_within_year)], dtype=int)
            if cfg.graded_enabled
            else cand.astype(int)
        )

        relevant_map[qi] = strong
        grade_map[qi] = gm

    return relevant_map, grade_map