"""
lyrics_reco.preprocess.pipeline

End-to-end preprocessing pipeline for Genius Song Lyrics.csv (CSV-first).

IMPORTANT: This module is designed for large CSVs.
- If chunksize is provided, we do TRUE streaming selection:
    * read chunks
    * apply lightweight filters
    * maintain global Top-N by views
    * maintain recent-year per-year minimum (Option A)
  then run heavier cleaning only on the selected subset.

Default policy (vectordb-friendly, Option A):
- Filter years to 1950~2022
- English filter: language column + optional fastText validation
- (Optional) process Genius English Translations pages into normal rows
- (Optional) expand multi-artist rows
- Dedup (title, artist)
- Trim:
    global Top-N by views + ensure recent-year minimum per year
- Produce both lyrics_clean and lyrics_dedup
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from ..common.io import save_csv
from ..common.paths import PATHS, ProjectPaths
from .schema import SCHEMA, add_song_id, ensure_columns
from .filters import (
    coerce_views,
    dedup_title_artist,
    expand_multi_artist_rows,
    filter_year_range,
    process_genius_translations,
)
from .language import english_filter
from .text_cleaning import clean_lyrics


PathLike = Union[str, Path]
logger = logging.getLogger("lyrics_reco")


@dataclass(frozen=True)
class PreprocessConfig:
    # I/O
    input_csv: PathLike
    output_csv: PathLike = "data/processed/genius_processed.csv"

    # Filters
    start_year: int = 1950
    end_year: int = 2022
    process_genius_translations: bool = True
    expand_multi_artist: bool = False

    # English filter
    use_fasttext: bool = True
    fasttext_model_path: PathLike = "assets/lid/lid.176.bin"
    fasttext_threshold: float = 0.5

    # Trimming (Option A)
    top_global: int = 500_000
    recent_year_start: int = 2020
    recent_year_end: int = 2022
    recent_min_per_year: int = 20_000

    # Lyrics cleaning
    strip_brackets: bool = True
    remove_tail: bool = True
    remove_repeat_blocks: bool = True

    # Slang replacement
    apply_slang: bool = True
    slang_map_path: PathLike = "assets/slang_map.json"


def _resolve_path(p: PathLike, *, paths: ProjectPaths = PATHS) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = (paths.root / pp).resolve()
    return pp


def _detect_usecols(csv_path: Path) -> Optional[List[str]]:
    """
    Read only columns we need if they exist, to speed up parsing.
    """
    try:
        cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        wanted = ["title", "artist", "lyrics", "tag", "genre", "views", "year", "language"]
        usecols = [c for c in wanted if c in cols]
        return usecols if usecols else None
    except Exception:
        return None


def _process_chunk_light(chunk: pd.DataFrame, cfg: PreprocessConfig, *, paths: ProjectPaths = PATHS) -> pd.DataFrame:
    """
    Lightweight per-chunk processing (safe for streaming):
    - ensure/rename columns
    - drop missing essentials
    - year filter
    - views coercion
    - english filter
    - optional translations processing
    - optional expand multi-artist
    - light dedup within chunk
    """
    chunk = ensure_columns(chunk)
    chunk = chunk.dropna(subset=["title", "artist", "lyrics"]).copy()

    chunk = filter_year_range(chunk, start=cfg.start_year, end=cfg.end_year)
    chunk = coerce_views(chunk)

    lang_res = english_filter(
        chunk,
        text_col="lyrics",
        lang_col="language",
        use_fasttext=cfg.use_fasttext,
        fasttext_model_path=cfg.fasttext_model_path,
        fasttext_threshold=cfg.fasttext_threshold,
        paths=paths,
    )
    chunk = chunk.loc[lang_res.mask].copy()

    if cfg.process_genius_translations:
        chunk = process_genius_translations(chunk)

    if cfg.expand_multi_artist:
        chunk = expand_multi_artist_rows(chunk)

    chunk = dedup_title_artist(chunk)

    keep = [c for c in ["title", "artist", "genre", "year", "views", "lyrics"] if c in chunk.columns]
    return chunk[keep].copy()


def _update_top_keep(top_keep: Optional[pd.DataFrame], cur: pd.DataFrame, *, n: int) -> Optional[pd.DataFrame]:
    if n <= 0 or cur.empty:
        return top_keep
    if top_keep is None:
        return cur.nlargest(n, "views").copy()
    comb = pd.concat([top_keep, cur], ignore_index=True)
    return comb.nlargest(n, "views").copy()


def _update_year_keep(year_keep: Dict[int, pd.DataFrame], cur: pd.DataFrame, *, year: int, n: int) -> None:
    if n <= 0:
        return
    sub = cur[cur["year"] == int(year)]
    if sub.empty:
        return
    prev = year_keep.get(int(year))
    if prev is None:
        year_keep[int(year)] = sub.nlargest(n, "views").copy()
    else:
        comb = pd.concat([prev, sub], ignore_index=True)
        year_keep[int(year)] = comb.nlargest(n, "views").copy()


def select_streaming(
    input_csv: Path,
    cfg: PreprocessConfig,
    *,
    chunksize: int,
    paths: ProjectPaths = PATHS,
    log_every_chunks: int = 5,
) -> pd.DataFrame:
    """
    Streaming selection (Option A):
    - global Top-N by views
    - ensure recent-year per-year floor

    Returns a selected subset DF (still raw lyrics; cleaning happens after selection).
    """
    usecols = _detect_usecols(input_csv)

    top_keep: Optional[pd.DataFrame] = None
    year_keep: Dict[int, pd.DataFrame] = {}

    rs = int(cfg.recent_year_start)
    re_ = int(min(cfg.recent_year_end, cfg.end_year))
    years_floor = list(range(rs, re_ + 1))

    t0 = time.time()
    total_rows_read = 0
    n_chunks = 0

    logger.info("Streaming read: chunksize=%d", int(chunksize))
    logger.info("Option A: top_global=%d, year_floor=%s..%s min_per_year=%d",
                int(cfg.top_global), rs, re_, int(cfg.recent_min_per_year))

    for chunk in pd.read_csv(input_csv, chunksize=int(chunksize), usecols=usecols):
        n_chunks += 1
        total_rows_read += len(chunk)

        cur = _process_chunk_light(chunk, cfg, paths=paths)
        if not cur.empty:
            top_keep = _update_top_keep(top_keep, cur, n=int(cfg.top_global))
            if cfg.recent_min_per_year and cfg.recent_min_per_year > 0:
                for y in years_floor:
                    _update_year_keep(year_keep, cur, year=y, n=int(cfg.recent_min_per_year))

        if (n_chunks % max(1, int(log_every_chunks))) == 0:
            elapsed = max(time.time() - t0, 1e-9)
            rate = total_rows_read / elapsed
            top_n = 0 if top_keep is None else len(top_keep)
            yr_sizes = {y: (len(year_keep.get(y, [])) if y in year_keep else 0) for y in years_floor}
            logger.info(
                "chunks=%d | rows_read=%d | rate=%.0f rows/s | top_keep=%d | year_floor=%s",
                n_chunks, total_rows_read, rate, top_n, yr_sizes
            )

    parts: List[pd.DataFrame] = []
    if top_keep is not None and not top_keep.empty:
        parts.append(top_keep)
    for y in sorted(year_keep.keys()):
        parts.append(year_keep[y])

    if not parts:
        return pd.DataFrame(columns=["title", "artist", "genre", "year", "views", "lyrics"])

    selected = pd.concat(parts, ignore_index=True)

    # Final dedup (title, artist) to match previous policy
    selected = dedup_title_artist(selected)

    logger.info("Selection done: selected_rows=%d", len(selected))
    return selected


def _load_slang_map(cfg: PreprocessConfig, *, paths: ProjectPaths = PATHS) -> Optional[dict]:
    if not getattr(cfg, "apply_slang", False):
        return None
    try:
        p = _resolve_path(getattr(cfg, "slang_map_path", "assets/slang_map.json"), paths=paths)
        slang_map = json.loads(p.read_text(encoding="utf-8"))
        return {str(k).lower(): str(v).lower() for k, v in slang_map.items()}
    except Exception:
        return None


def _clean_series_with_progress(
    series: pd.Series,
    *,
    cfg: PreprocessConfig,
    slang_map: Optional[dict],
    remove_repeat_blocks: bool,
    remove_repeat_lines_anywhere: bool,
    log_every: int = 5000,
    label: str = "",
) -> List[str]:
    out: List[str] = []
    n = len(series)
    t0 = time.time()

    for i, t in enumerate(series.astype(str).tolist(), start=1):
        out.append(
            clean_lyrics(
                t,
                strip_brackets=cfg.strip_brackets,
                remove_tail=cfg.remove_tail,
                keep_newlines=True,
                remove_repeat_blocks=remove_repeat_blocks,
                slang_map=slang_map,
                lowercase=True,
                remove_non_alpha=True,
                reduce_repeat_chars=True,
                max_token_len=25,
                remove_repeat_lines_anywhere=remove_repeat_lines_anywhere,
            )
        )

        if (i % max(1, int(log_every))) == 0 or i == n:
            elapsed = max(time.time() - t0, 1e-9)
            logger.info("%s cleaning: %d/%d (%.0f items/s)", label, i, n, i / elapsed)

    return out


def preprocess_selected(df: pd.DataFrame, cfg: PreprocessConfig, *, paths: ProjectPaths = PATHS) -> pd.DataFrame:
    """
    Heavier processing on selected subset:
    - lyrics_clean / lyrics_dedup
    - song_id
    - final columns
    """
    df = df.copy()

    slang_map = _load_slang_map(cfg, paths=paths)
    if slang_map is None and getattr(cfg, "apply_slang", False):
        logger.warning("slang_map enabled but failed to load: %s", cfg.slang_map_path)

    logger.info("Cleaning lyrics on selected subset: N=%d", len(df))

    df[SCHEMA.lyrics_clean_col] = _clean_series_with_progress(
        df["lyrics"],
        cfg=cfg,
        slang_map=slang_map,
        remove_repeat_blocks=False,
        remove_repeat_lines_anywhere=False,
        log_every=5000,
        label="lyrics_clean",
    )

    df[SCHEMA.lyrics_dedup_col] = _clean_series_with_progress(
        df["lyrics"],
        cfg=cfg,
        slang_map=slang_map,
        remove_repeat_blocks=cfg.remove_repeat_blocks,
        remove_repeat_lines_anywhere=True,
        log_every=5000,
        label="lyrics_dedup",
    )

    df = add_song_id(df)

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
    return df[keep_cols].reset_index(drop=True)


def run_preprocess(cfg: PreprocessConfig, *, paths: ProjectPaths = PATHS, chunksize: Optional[int] = None) -> Path:
    """
    Load Genius CSV -> preprocess -> save processed CSV.

    If chunksize is provided (>0), uses streaming selection with progress logs.
    """
    inp = _resolve_path(cfg.input_csv, paths=paths)
    out = _resolve_path(cfg.output_csv, paths=paths)

    logger.info("Input: %s", inp)
    logger.info("Output: %s", out)
    logger.info("Config: year=[%d,%d], fasttext=%s, process_translations=%s, expand_multi_artist=%s",
                cfg.start_year, cfg.end_year, cfg.use_fasttext, cfg.process_genius_translations, cfg.expand_multi_artist)

    if chunksize is None or int(chunksize) <= 0:
        logger.warning("chunksize disabled -> reading full CSV into memory (may be slow/large).")
        df_raw = pd.read_csv(inp)
        df_sel = _process_chunk_light(df_raw, cfg, paths=paths)
    else:
        df_sel = select_streaming(inp, cfg, chunksize=int(chunksize), paths=paths, log_every_chunks=1)

    df_out = preprocess_selected(df_sel, cfg, paths=paths)

    logger.info("Saving processed CSV: rows=%d", len(df_out))
    return save_csv(df_out, out, index=False, atomic=True)
