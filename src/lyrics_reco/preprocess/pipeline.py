"""
lyrics_reco.preprocess.pipeline

End-to-end preprocessing pipeline for Genius Song Lyrics.csv (CSV-first).

Default policy (vectordb-friendly, Option A):
- Filter years to 1950~2022 (remove weird future years etc.)
- English filter: language column + optional fastText validation
- Drop Genius English Translations pages
- Dedup (title, artist)
- Trim size with:
    global Top-N by views + ensure recent-year minimum per year
- Produce both lyrics_clean and lyrics_dedup
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ..common.io import save_csv
from ..common.paths import PATHS, ProjectPaths
from .schema import SCHEMA, add_song_id, ensure_columns
from .filters import (
    coerce_views,
    dedup_title_artist,
    drop_translation_pages,
    process_genius_translations,
    expand_multi_artist_rows,
    filter_year_range,
    top_n_global,
    top_n_global_plus_year_floor,
)
from .language import english_filter
from .text_cleaning import clean_lyrics


PathLike = Union[str, Path]


@dataclass(frozen=True)
class PreprocessConfig:
    # I/O
    input_csv: PathLike
    output_csv: PathLike = "data/processed/genius_processed.csv"

    # Filters
    start_year: int = 1950
    end_year: int = 2022
    drop_translations: bool = True
    expand_multi_artist: bool = False

    # English filter
    use_fasttext: bool = True
    fasttext_model_path: PathLike = "assets/lid/lid.176.bin"
    fasttext_threshold: float = 0.5

    # Trimming (Option A)
    top_global: int = 500_000

    recent_year_start: int = 2020
    recent_year_end: int = 2022
    recent_min_per_year: int = 20_000  # adjust 10k~30k depending on desired coverage

    # Lyrics cleaning
    strip_brackets: bool = True
    remove_tail: bool = True
    remove_repeat_blocks: bool = True

    # Slang replacement
    apply_slang: bool = True
    slang_map_path: PathLike = "assets/slang_map.json"


def load_genius_minimal(input_csv: PathLike, *, chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    Load Genius CSV. If chunksize is provided, we still return a single DataFrame
    (trimming happens later), but chunked reading can reduce peak memory during parsing.
    """
    if chunksize is None:
        return pd.read_csv(input_csv)

    parts = []
    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def preprocess_genius(df: pd.DataFrame, cfg: PreprocessConfig, *, paths: ProjectPaths = PATHS) -> pd.DataFrame:
    # 1) Keep/rename relevant columns
    df = ensure_columns(df)

    # 2) Drop missing essentials
    df = df.dropna(subset=["title", "artist", "lyrics"]).copy()

    # 3) Year range
    df = filter_year_range(df, start=cfg.start_year, end=cfg.end_year)

    # 4) Views numeric (for trimming)
    df = coerce_views(df)

    # 5) English filter
    lang_res = english_filter(
        df,
        text_col="lyrics",
        lang_col="language",
        use_fasttext=cfg.use_fasttext,
        fasttext_model_path=cfg.fasttext_model_path,
        fasttext_threshold=cfg.fasttext_threshold,
        paths=paths,
    )
    df = df.loc[lang_res.mask].copy()

    # 6) Optional: drop translation pages
    if cfg.drop_translations:
        df = drop_translation_pages(df)

    # 7) Optional: expand multi-artist
    if cfg.expand_multi_artist:
        df = expand_multi_artist_rows(df)

    # 8) Dedup (title, artist)
    df = dedup_title_artist(df)

    # 9) Trimming (Option A: global top + recent-year floor)
    if cfg.top_global and cfg.top_global > 0:
        rs = int(cfg.recent_year_start)
        re = int(min(cfg.recent_year_end, cfg.end_year))

        if cfg.recent_min_per_year and cfg.recent_min_per_year > 0:
            df = top_n_global_plus_year_floor(
                df,
                n_global=int(cfg.top_global),
                year_start=rs,
                year_end=re,
                min_per_year=int(cfg.recent_min_per_year),
                year_col="year",
                sort_col="views",
            )
        else:
            df = top_n_global(df, n=int(cfg.top_global), sort_col="views")

    # 10) Clean lyrics (two versions)
    df = df.copy()
    df[SCHEMA.lyrics_clean_col] = df["lyrics"].astype(str).map(
        lambda t: clean_lyrics(
            t,
            strip_brackets=cfg.strip_brackets,
            remove_tail=cfg.remove_tail,
            keep_newlines=True,
            remove_repeat_blocks=False,  # clean only
        )
    )
    df[SCHEMA.lyrics_dedup_col] = df["lyrics"].astype(str).map(
        lambda t: clean_lyrics(
            t,
            strip_brackets=cfg.strip_brackets,
            remove_tail=cfg.remove_tail,
            keep_newlines=True,
            remove_repeat_blocks=cfg.remove_repeat_blocks,  # embedding-friendly
        )
    )

    # 11) Add song_id
    df = add_song_id(df)

    # 12) Final columns
    keep_cols = [
        SCHEMA.id_col,
        "title",
        "artist",
        "genre",
        "year",
        "views",
        SCHEMA.lyrics_clean_col,
        SCHEMA.lyrics_dedup_col,
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


def run_preprocess(cfg: PreprocessConfig, *, paths: ProjectPaths = PATHS, chunksize: Optional[int] = None) -> Path:
    """
    Load Genius CSV -> preprocess -> save processed CSV.

    Returns:
        output path
    """
    inp = Path(cfg.input_csv)
    if not inp.is_absolute():
        inp = (paths.root / inp).resolve()

    out = Path(cfg.output_csv)
    if not out.is_absolute():
        out = (paths.root / out).resolve()

    df_raw = load_genius_minimal(inp, chunksize=chunksize)
    df_out = preprocess_genius(df_raw, cfg, paths=paths)

    return save_csv(df_out, out, index=False, atomic=True)
