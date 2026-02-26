"""
lyrics_reco.preprocess.tokenize

Tokenization / normalization helper inspired by your old tokenize_lemmatize().

Why a new module?
- Some experiments (TF-IDF baseline, analysis) may want token lists.
- spaCy lemmatization is heavy and sometimes hard to install; we provide a safe fallback.

Behavior:
- regex tokenization (letters + internal apostrophes)
- optional stopword removal (from assets/stopwords.txt)
- optional stemming (PorterStemmer) as a lemmatization-lite fallback

NOTE:
- This is optional and not required for the lexicon-ratio baseline.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd

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


def tokenize_lemmatize(
    texts: Sequence[str],
    *,
    stopwords: Optional[Set[str]] = None,
    stem: bool = False,
) -> List[List[str]]:
    """
    Tokenize + (optional) stopword removal + (optional) stemming.

    This is a practical replacement for spaCy lemmatization in your old code.
    """
    from nltk.stem import PorterStemmer
    ps = PorterStemmer() if stem else None

    out: List[List[str]] = []
    for t in texts:
        toks = simple_tokenize(t, lowercase=True)
        if stopwords is not None:
            toks = [x for x in toks if x not in stopwords]
        if ps is not None:
            toks = [ps.stem(x) for x in toks]
        out.append(toks)
    return out
