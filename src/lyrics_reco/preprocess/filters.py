"""
lyrics_reco.preprocess.filters

Row-level filters and transformations for Genius Song Lyrics dataset.

Key functions:
- filter_year_range: enforce plausible year bounds (default 1950~2022)
- drop_translation_pages: remove "Genius English Translations" artist pages
- expand_multi_artist_rows: optional split "A & B" into multiple rows
- dedup_title_artist: deduplicate by (title, artist)
- coerce_views: coerce views to non-negative int
- top_n_global: global Top-N by views
- top_n_per_year: per-year Top-N by views (balanced sampling)
- top_n_global_plus_year_floor: Option A
    -> global Top-N + ensure recent-year minimum per year
"""

from __future__ import annotations

import re
from typing import List

import pandas as pd


def filter_year_range(
    df: pd.DataFrame,
    *,
    year_col: str = "year",
    start: int = 1950,
    end: int = 2022,
) -> pd.DataFrame:
    """
    Keep rows with year in [start, end] (inclusive).

    - Coerces year to numeric; invalid years are dropped.
    - Converts year to int.
    """
    if year_col not in df.columns:
        return df
    y = pd.to_numeric(df[year_col], errors="coerce")
    mask = y.notna() & (y >= start) & (y <= end)
    out = df.loc[mask].copy()
    out[year_col] = y.loc[mask].astype(int)
    return out


def drop_translation_pages(
    df: pd.DataFrame,
    *,
    artist_col: str = "artist",
    translation_artist: str = "Genius English Translations",
) -> pd.DataFrame:
    """Drop Genius translation pages (common noise)."""
    if artist_col not in df.columns:
        return df
    return df[df[artist_col].astype(str) != translation_artist].copy()


_MULTI_ARTIST_SPLIT_RE = re.compile(r"\s*&\s*")


def expand_multi_artist_rows(df: pd.DataFrame, *, artist_col: str = "artist") -> pd.DataFrame:
    """
    Expand rows where artist is "A & B" into multiple rows.

    Note:
    - Only splits on '&' with spaces like "A & B".
    - Optional because this increases dataset size.
    """
    if artist_col not in df.columns:
        return df

    rows = []
    for _, r in df.iterrows():
        artist = str(r[artist_col])
        if " & " not in artist:
            rows.append(r)
            continue

        parts = [p.strip() for p in _MULTI_ARTIST_SPLIT_RE.split(artist) if p.strip()]
        if not parts:
            rows.append(r)
            continue

        for p in parts:
            rr = r.copy()
            rr[artist_col] = p
            rows.append(rr)

    return pd.DataFrame(rows).reset_index(drop=True)


def dedup_title_artist(df: pd.DataFrame, *, title_col: str = "title", artist_col: str = "artist") -> pd.DataFrame:
    """Deduplicate by (title, artist)."""
    if title_col not in df.columns or artist_col not in df.columns:
        return df
    return df.drop_duplicates(subset=[title_col, artist_col]).reset_index(drop=True)


def coerce_views(df: pd.DataFrame, *, views_col: str = "views") -> pd.DataFrame:
    """Ensure views exists and is non-negative int."""
    df = df.copy()
    if views_col not in df.columns:
        df[views_col] = 0
        return df
    df[views_col] = pd.to_numeric(df[views_col], errors="coerce").fillna(0)
    df[views_col] = df[views_col].where(df[views_col] >= 0, 0).astype(int)
    return df


def top_n_global(df: pd.DataFrame, *, n: int, sort_col: str = "views") -> pd.DataFrame:
    """Global Top-N by sort_col (views)."""
    if n <= 0:
        return df
    if sort_col not in df.columns:
        return df.head(n).copy()
    return df.nlargest(n, sort_col).reset_index(drop=True)


def top_n_per_year(
    df: pd.DataFrame,
    *,
    per_year: int,
    year_col: str = "year",
    sort_col: str = "views",
) -> pd.DataFrame:
    """Keep Top-N within each year (balanced sampling)."""
    if per_year <= 0:
        return df
    if year_col not in df.columns:
        return top_n_global(df, n=per_year, sort_col=sort_col)
    if sort_col not in df.columns:
        return df.groupby(year_col, group_keys=False).head(per_year).reset_index(drop=True)
    return (
        df.sort_values([year_col, sort_col], ascending=[True, False])
          .groupby(year_col, group_keys=False)
          .head(per_year)
          .reset_index(drop=True)
    )


def top_n_global_plus_year_floor(
    df: pd.DataFrame,
    *,
    n_global: int,
    year_start: int,
    year_end: int,
    min_per_year: int,
    year_col: str = "year",
    sort_col: str = "views",
) -> pd.DataFrame:
    """
    Option A:
    1) global top-N by views
    2) for each year in [year_start, year_end], ensure at least min_per_year rows
       by adding more rows from that year (also sorted by views)

    Returns a DF that can be larger than n_global (because of added rows).
    """
    if n_global <= 0:
        base = df.copy()
    else:
        if sort_col not in df.columns:
            base = df.head(n_global).copy()
        else:
            base = df.nlargest(n_global, sort_col).copy()

    if min_per_year <= 0 or year_col not in df.columns:
        return base.reset_index(drop=True)

    # Track original row indices already included
    base_idx = set(base.index)
    extras: List[pd.DataFrame] = []

    for y in range(int(year_start), int(year_end) + 1):
        cur = int((base[year_col] == y).sum())
        need = max(0, int(min_per_year) - cur)
        if need == 0:
            continue

        cand = df[df[year_col] == y].copy()
        if cand.empty:
            continue

        if sort_col in cand.columns:
            cand = cand.sort_values(sort_col, ascending=False)

        # remove already selected rows by original index
        cand = cand.loc[~cand.index.isin(base_idx)]
        if cand.empty:
            continue

        extras.append(cand.head(need))

    if not extras:
        return base.reset_index(drop=True)

    out = pd.concat([base] + extras, axis=0)

    # Ensure uniqueness by original index
    out = out[~out.index.duplicated(keep="first")]

    return out.reset_index(drop=True)
