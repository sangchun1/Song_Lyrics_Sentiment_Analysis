"""
lyrics_reco.baseline.tokenize

Lightweight tokenization & stopword filtering for baseline methods.

Notes:
- Preprocess already produces lyrics_clean / lyrics_dedup, but baseline still needs:
  - consistent lowercasing
  - simple word tokenization
  - optional stopword / filler removal
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Set, Union

import pandas as pd

from ..common.paths import PATHS, ProjectPaths

PathLike = Union[str, Path]

# Keep apostrophes inside words, e.g. "don't"
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def simple_tokenize(text: str, *, lowercase: bool = True, min_len: int = 2) -> List[str]:
    if text is None:
        return []
    s = str(text)
    if lowercase:
        s = s.lower()
    toks = _TOKEN_RE.findall(s)
    if min_len > 1:
        toks = [t for t in toks if len(t) >= min_len]
    return toks

def load_word_set(path: PathLike, *, paths: ProjectPaths = PATHS, lowercase: bool = True) -> Set[str]:
    """
    Load newline-delimited words into a set.
    Supports both relative (from repo root) and absolute paths.
    """
    p = Path(path)
    if not p.is_absolute():
        p = (paths.root / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"word list not found: {p}")

    words = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if not w:
            continue
        words.add(w.lower() if lowercase else w)
    return words

def load_slang_map(path: PathLike, *, paths: ProjectPaths = PATHS) -> dict:
    """
    Load a slang map JSON (token->token). Optional utility.
    If you already normalized slang in preprocessing, you can skip this.
    """
    p = Path(path)
    if not p.is_absolute():
        p = (paths.root / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"slang_map not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def apply_slang(tokens: Sequence[str], slang_map: dict) -> List[str]:
    return [slang_map.get(t, t) for t in tokens]

def filter_tokens(
    tokens: Sequence[str],
    *,
    stopwords: Optional[Set[str]] = None,
    filler_words: Optional[Set[str]] = None,
) -> List[str]:
    out = []
    for t in tokens:
        if stopwords is not None and t in stopwords:
            continue
        if filler_words is not None and t in filler_words:
            continue
        out.append(t)
    return out

def prepare_text(
    text: str,
    *,
    stopwords: Optional[Set[str]] = None,
    filler_words: Optional[Set[str]] = None,
    slang_map: Optional[dict] = None,
) -> str:
    """Tokenize -> slang(optional) -> stop/filler filter(optional) -> join for vectorizer input."""
    toks = simple_tokenize(text, lowercase=True)
    if slang_map is not None:
        toks = apply_slang(toks, slang_map)
    toks = filter_tokens(toks, stopwords=stopwords, filler_words=filler_words)
    return " ".join(toks)

def prepare_text_series(
    s: pd.Series,
    *,
    stopwords: Optional[Set[str]] = None,
    filler_words: Optional[Set[str]] = None,
    slang_map: Optional[dict] = None,
) -> pd.Series:
    return s.astype(str).map(
        lambda x: prepare_text(x, stopwords=stopwords, filler_words=filler_words, slang_map=slang_map)
    )