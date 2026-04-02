"""lyrics_reco.emotion_context.splitter

Line splitting utilities for lyrics.

We assume preprocess already produced:
- lyrics_clean (cleaned, may still contain repeats)
- lyrics_dedup (embedding-friendly version with repeats reduced)

This revision explicitly supports two streams:
- embedding stream: dedup lines
- lexicon stream: original lines kept
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd

from ..preprocess.text_cleaning import normalize_whitespace

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MULTI_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SplitConfig:
    line_split: str = "newline"  # newline | sentence
    strip_brackets: bool = True
    min_line_chars: int = 3
    max_lines_per_song: int = 250
    dedup_lines: bool = True


def normalize_line_key(text: str) -> str:
    s = normalize_whitespace(str(text or ""), keep_newlines=False).strip().lower()
    return _MULTI_SPACE_RE.sub(" ", s)


def _strip_bracket_only_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]") and len(s) <= 40:
            continue
        out.append(ln)
    return out


def split_lyrics_to_lines(text: str, cfg: SplitConfig) -> List[str]:
    if text is None:
        return []
    t = str(text)

    if cfg.line_split == "sentence":
        parts = _SENT_SPLIT_RE.split(t.replace("\n", " ").strip())
    else:
        parts = t.split("\n")

    lines: List[str] = []
    for p in parts:
        s = normalize_whitespace(p, keep_newlines=False).strip()
        if len(s) < int(cfg.min_line_chars):
            continue
        lines.append(s)

    if cfg.strip_brackets:
        lines = _strip_bracket_only_lines(lines)

    if cfg.dedup_lines:
        seen = set()
        uniq = []
        for ln in lines:
            key = normalize_line_key(ln)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(ln)
        lines = uniq

    if cfg.max_lines_per_song and len(lines) > int(cfg.max_lines_per_song):
        lines = lines[: int(cfg.max_lines_per_song)]

    return lines


def explode_songs_to_line_table(
    song_ids: Sequence[str],
    lyrics_list: Sequence[str],
    cfg: SplitConfig,
    *,
    dedup_override: bool | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    use_dedup = cfg.dedup_lines if dedup_override is None else bool(dedup_override)
    local_cfg = SplitConfig(
        line_split=cfg.line_split,
        strip_brackets=cfg.strip_brackets,
        min_line_chars=cfg.min_line_chars,
        max_lines_per_song=cfg.max_lines_per_song,
        dedup_lines=use_dedup,
    )
    for si, (song_id, txt) in enumerate(zip(song_ids, lyrics_list)):
        lines = split_lyrics_to_lines(txt, local_cfg)
        for li, ln in enumerate(lines):
            rows.append(
                {
                    "song_id": str(song_id),
                    "song_index": int(si),
                    "line_index": int(li),
                    "line_text": ln,
                    "line_key": normalize_line_key(ln),
                }
            )
    return pd.DataFrame(rows, columns=["song_id", "song_index", "line_index", "line_text", "line_key"])


def explode_songs_to_lines(
    song_ids: Sequence[str],
    lyrics_list: Sequence[str],
    cfg: SplitConfig,
) -> Tuple[List[str], List[int], List[int]]:
    tbl = explode_songs_to_line_table(song_ids, lyrics_list, cfg)
    return (
        tbl["line_text"].tolist(),
        tbl["song_index"].astype(int).tolist(),
        tbl["line_index"].astype(int).tolist(),
    )
