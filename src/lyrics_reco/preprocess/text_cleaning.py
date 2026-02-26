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


def _normalize_quotes_and_ing(text: str) -> str:
    # normalize curly apostrophes and fix "walkin'" -> "walking"
    t = text.replace("’", "'").replace("‘", "'")
    t = re.sub(r"(\w+)in'", r"\1ing", t)
    return t


def _remove_non_alpha(text: str, *, keep_newlines: bool) -> str:
    # keep apostrophes
    if keep_newlines:
        # allow newline
        return re.sub(r"[^a-zA-Z\s'\n]", " ", text)
    return re.sub(r"[^a-zA-Z\s']", " ", text)


def _apply_slang_map(text: str, slang_map: Optional[dict]) -> str:
    if not slang_map:
        return text
    t = text
    # assume lowercase input
    for short, full in slang_map.items():
        t = re.sub(r"\b" + re.escape(short) + r"\b", full, t)
    return t


def _reduce_repeat_chars(text: str) -> str:
    # aaa -> a, coool -> col
    return re.sub(r"(\w)\1{2,}", r"\1", text)


def _remove_long_words(text: str, *, max_len: int = 25) -> str:
    return " ".join([w for w in text.split() if len(w) < int(max_len)])


def _remove_duplicate_lines_anywhere(text: str) -> str:
    lines = _normalize_newlines(text).split("\n")
    seen = {}
    uniq = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        seen[s] = seen.get(s, 0) + 1
        if seen[s] <= 1:
            uniq.append(s)
    return "\n".join(uniq).strip()


def clean_lyrics(
    text: str,
    *,
    strip_brackets: bool = True,
    remove_tail: bool = True,
    keep_newlines: bool = True,
    remove_repeat_blocks: bool = False,
    # additions inspired by old preprocess_lyrics
    lowercase: bool = True,
    remove_non_alpha: bool = True,
    slang_map: Optional[dict] = None,
    reduce_repeat_chars: bool = True,
    max_token_len: int = 25,
    remove_repeat_lines_anywhere: bool = False,
) -> str:
    """
    Clean Genius lyrics text.

    Original behaviors:
    - Remove [Chorus]/[Verse] tags (optional)
    - Remove Genius tail artifacts (optional)
    - Normalize whitespace (keep newlines by default)
    - Deduplicate consecutive lines
    - Optional: remove repeated stanza blocks

    Added (from your old preprocessing.py):
    - normalize quotes & "in'" -> "ing"
    - lowercase
    - optional slang replacement via slang_map
    - optional remove non-alpha characters (keep apostrophes)
    - optional reduce repeated characters (e.g., 'soooo' -> 'so')
    - optional remove very long tokens
    - optional remove duplicate lines anywhere (stronger than consecutive dedup)
    """
    if text is None:
        return ""

    t = str(text)
    t = _normalize_newlines(t)
    t = _normalize_quotes_and_ing(t)

    if strip_brackets:
        t = strip_bracket_tags(t)
    if remove_tail:
        t = remove_genius_tail(t)

    # normalize whitespace early
    t = normalize_whitespace(t, keep_newlines=keep_newlines)
    t = dedup_consecutive_lines(t)

    if remove_repeat_blocks:
        t = dedup_repeated_blocks(t)

    if remove_repeat_lines_anywhere:
        t = _remove_duplicate_lines_anywhere(t)

    if lowercase:
        t = t.lower()

    # slang replacement (expects lowercase)
    t = _apply_slang_map(t, slang_map)

    if remove_non_alpha:
        t = _remove_non_alpha(t, keep_newlines=keep_newlines)

    if reduce_repeat_chars:
        t = _reduce_repeat_chars(t)

    # collapse whitespace again
    t = normalize_whitespace(t, keep_newlines=keep_newlines)

    if max_token_len and int(max_token_len) > 0:
        # if keeping newlines, process per line
        if keep_newlines:
            lines = []
            for ln in t.split("\n"):
                lines.append(_remove_long_words(ln, max_len=max_token_len))
            t = "\n".join(lines).strip()
        else:
            t = _remove_long_words(t, max_len=max_token_len)

    return t
