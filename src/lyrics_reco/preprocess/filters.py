"""
lyrics_reco.preprocess.filters

Row-level filters and transformations for Genius Song Lyrics dataset.

Key functions:
- process_genius_translations: convert "Genius English Translations" pages into normal rows
- filter_year_range: enforce plausible year bounds (default 1950~2022)
- expand_multi_artist_rows: optional split multi-artist strings into multiple rows
- dedup_title_artist: deduplicate by (title, artist)
- coerce_views: coerce views to non-negative int
- top_n_global: global Top-N by views
- top_n_per_year: per-year Top-N by views
- top_n_global_plus_year_floor: Option A (global top + recent-year floor)
"""

from __future__ import annotations

import re
from typing import List

import pandas as pd


def process_genius_translations(
    df: pd.DataFrame,
    *,
    artist_col: str = "artist",
    title_col: str = "title",
    translation_artist: str = "Genius English Translations",
) -> pd.DataFrame:
    """
    preprocessing.py behavior:
    Convert translation pages into normal (artist,title) rows (does NOT drop).

    If artist == "Genius English Translations":
      - artist <- title.split(" - ")[0]
      - title  <- remove "English Translation"
      - title  <- remove leading "{artist} - "
    """
    if artist_col not in df.columns or title_col not in df.columns:
        return df

    out = df.copy()
    mask = out[artist_col].astype(str) == translation_artist
    if not mask.any():
        return out

    out.loc[mask, artist_col] = out.loc[mask, title_col].astype(str).str.split(" - ").str[0]
    out.loc[mask, title_col] = out.loc[mask, title_col].astype(str).str.replace(r"English Translation", "", regex=True)

    for idx in out[mask].index.tolist():
        art = str(out.at[idx, artist_col])
        artist_pattern = re.escape(art) + r"\s*-\s*"
        out.at[idx, title_col] = re.sub(r"^" + artist_pattern, "", str(out.at[idx, title_col])).strip()

    return out


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


def expand_multi_artist_rows(df: pd.DataFrame, *, artist_col: str = "artist") -> pd.DataFrame:
    """
    Split multi-artist rows into multiple rows (preprocessing.py regex).

    Split on:
      &, , , feat., featuring, X/x
    """
    if artist_col not in df.columns:
        return df

    split_re = re.compile(r"\s*(?:&|,|feat\.|Feat\.|FEAT\.|featuring|Featuring| X | x )\s*")
    expanded_rows = []

    for _, row in df.iterrows():
        artists = split_re.split(str(row[artist_col]))
        artists = [a.strip() for a in artists if a.strip()]

        if len(artists) > 1:
            for a in artists:
                new_row = row.copy()
                new_row[artist_col] = a
                expanded_rows.append(new_row)

    if not expanded_rows:
        return df.reset_index(drop=True)

    expanded_df = pd.DataFrame(expanded_rows)
    base = df.copy()
    base[artist_col] = base[artist_col].astype(str)

    # preprocessing.py heuristic: remove rows containing " & " after expansion
    df_cleaned = base[~base[artist_col].str.contains(" & ", na=False)]
    final_df = pd.concat([df_cleaned, expanded_df], ignore_index=True)
    return final_df.reset_index(drop=True)


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
        base = df.nlargest(n_global, sort_col).copy() if sort_col in df.columns else df.head(n_global).copy()

    if min_per_year <= 0 or year_col not in df.columns:
        return base.reset_index(drop=True)

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
        cand = cand.sort_values(sort_col, ascending=False) if sort_col in cand.columns else cand
        cand = cand.loc[~cand.index.isin(base_idx)]
        if cand.empty:
            continue
        extras.append(cand.head(need))

    if not extras:
        return base.reset_index(drop=True)

    out = pd.concat([base] + extras, axis=0)
    out = out[~out.index.duplicated(keep="first")]
    return out.reset_index(drop=True)
