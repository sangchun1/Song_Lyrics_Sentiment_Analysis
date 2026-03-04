"""lyrics_reco.emotion_context.splitter

Line splitting utilities for lyrics.

We assume preprocess already produced:
- lyrics_clean (cleaned, may still contain repeats)
- lyrics_dedup (embedding-friendly version with repeats reduced)

This module:
- splits by newline (default) or sentence-ish separators
- filters too-short lines
- optional dedup of identical lines per song
- cap max_lines_per_song to keep memory bounded
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional

from ..preprocess.text_cleaning import normalize_whitespace

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

@dataclass(frozen=True)
class SplitConfig:
    line_split: str = "newline"  # newline | sentence
    strip_brackets: bool = True
    min_line_chars: int = 3
    max_lines_per_song: int = 250
    dedup_lines: bool = True

def _strip_bracket_only_lines(lines: List[str]) -> List[str]:
    # If preprocess already removed tags, this usually no-ops.
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # lines that are only bracket tags like "[Chorus]"
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

    # normalize whitespace per line
    lines = []
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
            if ln in seen:
                continue
            seen.add(ln)
            uniq.append(ln)
        lines = uniq

    if cfg.max_lines_per_song and len(lines) > int(cfg.max_lines_per_song):
        lines = lines[: int(cfg.max_lines_per_song)]

    return lines

def explode_songs_to_lines(
    song_ids: Sequence[str],
    lyrics_list: Sequence[str],
    cfg: SplitConfig,
) -> Tuple[List[str], List[int], List[int]]:
    """Explode songs into a flat list of lines.

    Returns:
        lines: flat list of line strings
        song_index: per-line song index (0..len(song_ids)-1)
        line_index: per-line line index (within song)
    """
    all_lines: List[str] = []
    song_index: List[int] = []
    line_index: List[int] = []

    for si, txt in enumerate(lyrics_list):
        lines = split_lyrics_to_lines(txt, cfg)
        for li, ln in enumerate(lines):
            all_lines.append(ln)
            song_index.append(si)
            line_index.append(li)

    return all_lines, song_index, line_index
