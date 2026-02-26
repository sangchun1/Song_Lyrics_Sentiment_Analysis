"""
lyrics_reco.preprocess.language

English filtering utilities.

Two-stage approach (like your original notebook):
1) Use existing 'language' column if present (fast).
2) Optionally validate via fastText lid.176.bin (more reliable).

This module is robust:
- If fasttext is not installed or model file missing, it will fall back
  to the 'language' column only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

from ..common.paths import PATHS, ProjectPaths


PathLike = Union[str, Path]


@dataclass(frozen=True)
class LangFilterResult:
    mask: pd.Series
    method: str  # "column" | "fasttext" | "column+fasttext"


def filter_english_by_language_column(df: pd.DataFrame, *, lang_col: str = "language") -> pd.Series:
    if lang_col not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    return df[lang_col].astype(str).str.lower().eq("en")


def _load_fasttext_model(model_path: Path):
    import fasttext  # fasttext-wheel

    return fasttext.load_model(str(model_path))


def fasttext_is_english(
    texts: Sequence[str],
    *,
    model_path: PathLike = "assets/lid/lid.176.bin",
    paths: ProjectPaths = PATHS,
    threshold: float = 0.5,
) -> pd.Series:
    """
    Return boolean Series indicating English by fastText.

    threshold: probability threshold for __label__en.
    """
    mp = Path(model_path)
    if not mp.is_absolute():
        mp = (paths.root / mp).resolve()

    model = _load_fasttext_model(mp)

    labels, probs = model.predict(list(texts), k=1)
    out = []
    for lab, pr in zip(labels, probs):
        lab0 = lab[0] if isinstance(lab, (list, tuple)) else lab
        pr0 = pr[0] if isinstance(pr, (list, tuple)) else pr
        out.append((lab0 == "__label__en") and (float(pr0) >= float(threshold)))
    return pd.Series(out)


def english_filter(
    df: pd.DataFrame,
    *,
    text_col: str = "lyrics",
    lang_col: str = "language",
    use_fasttext: bool = True,
    fasttext_model_path: PathLike = "assets/lid/lid.176.bin",
    fasttext_threshold: float = 0.5,
    paths: ProjectPaths = PATHS,
) -> LangFilterResult:
    """
    Produce an English mask.

    If use_fasttext=True, fastText is attempted; if unavailable, fall back.
    """
    col_mask = filter_english_by_language_column(df, lang_col=lang_col)

    if not use_fasttext:
        return LangFilterResult(mask=col_mask, method="column")

    try:
        ft_mask = fasttext_is_english(
            df[text_col].astype(str).tolist(),
            model_path=fasttext_model_path,
            paths=paths,
            threshold=fasttext_threshold,
        )
        mask = col_mask & ft_mask if lang_col in df.columns else ft_mask
        method = "column+fasttext" if lang_col in df.columns else "fasttext"
        return LangFilterResult(mask=mask, method=method)
    except Exception:
        # fallback
        return LangFilterResult(mask=col_mask, method="column")
