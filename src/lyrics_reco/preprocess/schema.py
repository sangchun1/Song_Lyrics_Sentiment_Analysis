"""
lyrics_reco.preprocess.schema

Column standardization for Genius Song Lyrics.csv.

We keep a minimal, stable schema across the project:
- song_id: stable identifier (hash)
- title, artist, genre, year, views
- lyrics_clean: cleaned lyrics text (keeps newlines)
- lyrics_dedup: cleaned lyrics with repeated blocks removed (for embedding)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd


GENIUS_RENAME_MAP: Dict[str, str] = {
    "tag": "genre",
}


REQUIRED_RAW_COLS = ["title", "artist", "lyrics"]
OPTIONAL_RAW_COLS = ["tag", "genre", "views", "year", "language"]


@dataclass(frozen=True)
class StandardSchema:
    id_col: str = "song_id"
    title_col: str = "title"
    artist_col: str = "artist"
    genre_col: str = "genre"
    year_col: str = "year"
    views_col: str = "views"
    lyrics_clean_col: str = "lyrics_clean"
    lyrics_dedup_col: str = "lyrics_dedup"


SCHEMA = StandardSchema()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common columns and keep only known ones if present."""
    df = df.rename(columns=GENIUS_RENAME_MAP)
    keep = [c for c in ["title", "artist", "genre", "tag", "views", "year", "lyrics", "language"] if c in df.columns]
    df = df[keep].copy()
    if "tag" in df.columns and "genre" not in df.columns:
        df = df.rename(columns={"tag": "genre"})
    return df


def make_song_id(title: str, artist: str, year: Optional[int]) -> str:
    """Stable id (sha1) based on title/artist/year."""
    y = "" if year is None else str(int(year))
    s = f"{title}\u241E{artist}\u241E{y}".strip().lower()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def add_song_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    years = df["year"] if "year" in df.columns else pd.Series([None] * len(df))
    df[SCHEMA.id_col] = [
        make_song_id(t, a, (None if pd.isna(y) else int(y)))
        for t, a, y in zip(df["title"].astype(str), df["artist"].astype(str), years)
    ]
    return df
