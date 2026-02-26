"""
lyrics_reco.preprocess.text_cleaning

Lyrics cleaning utilities.

Key behaviors (inspired by your original notebook):
- Remove section tags like [Chorus], [Verse], [Bridge] (optional)
- Remove common Genius tail artifacts: "Embed", "You might also like", etc.
- Normalize whitespace but keep newlines (for line-based processing)
- Optional: remove repeated blocks (chorus repeats) by deduplicating stanza blocks
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional


_BRACKET_TAG_RE = re.compile(r"\[(.*?)\]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_EMBED_RE = re.compile(r"(\d+)?Embed\b", re.IGNORECASE)
_YOU_MIGHT_ALSO_LIKE_RE = re.compile(r"You might also like", re.IGNORECASE)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_bracket_tags(text: str) -> str:
    # Replace bracketed tags with newline to preserve stanza boundaries
    return _BRACKET_TAG_RE.sub("\n", text)


def remove_genius_tail(text: str) -> str:
    # Remove common tail artifacts
    text = _YOU_MIGHT_ALSO_LIKE_RE.split(text)[0]
    text = _EMBED_RE.sub("", text)
    return text


def normalize_whitespace(text: str, *, keep_newlines: bool = True) -> str:
    text = _normalize_newlines(text)

    if keep_newlines:
        # normalize spaces within lines
        lines = [ _MULTI_SPACE_RE.sub(" ", ln).strip() for ln in text.split("\n") ]
        # collapse excessive blank lines
        out_lines: List[str] = []
        blank = 0
        for ln in lines:
            if ln == "":
                blank += 1
                if blank <= 1:
                    out_lines.append("")
            else:
                blank = 0
                out_lines.append(ln)
        return "\n".join(out_lines).strip()
    else:
        text = _MULTI_SPACE_RE.sub(" ", text)
        return text.strip()


def dedup_consecutive_lines(text: str) -> str:
    lines = [ln.strip() for ln in _normalize_newlines(text).split("\n")]
    out: List[str] = []
    prev = None
    for ln in lines:
        if ln == "" and (out and out[-1] == ""):
            continue
        if prev is not None and ln != "" and ln == prev:
            continue
        out.append(ln)
        prev = ln
    return "\n".join(out).strip()


def dedup_repeated_blocks(text: str) -> str:
    """
    Remove repeated stanza blocks while preserving order.

    Heuristic:
    - Split by blank lines into blocks
    - Keep first occurrence of identical normalized blocks
    """
    t = normalize_whitespace(text, keep_newlines=True)
    blocks = [b.strip() for b in t.split("\n\n") if b.strip()]
    seen = set()
    kept: List[str] = []
    for b in blocks:
        key = "\n".join([ln.strip().lower() for ln in b.split("\n") if ln.strip()])
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(b)
    return "\n\n".join(kept).strip()


def clean_lyrics(
    text: str,
    *,
    strip_brackets: bool = True,
    remove_tail: bool = True,
    keep_newlines: bool = True,
    remove_repeat_blocks: bool = False,
) -> str:
    if text is None:
        return ""
    t = str(text)
    t = _normalize_newlines(t)

    if strip_brackets:
        t = strip_bracket_tags(t)
    if remove_tail:
        t = remove_genius_tail(t)

    t = normalize_whitespace(t, keep_newlines=keep_newlines)
    t = dedup_consecutive_lines(t)

    if remove_repeat_blocks:
        t = dedup_repeated_blocks(t)

    return t
