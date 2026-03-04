"""lyrics_reco.emotion_context.line_features

Vectorized lexicon feature computation for a batch of lyric lines.

This is intentionally fast:
- CountVectorizer restricted to lexicon vocab
- sparse matrix multiplications

Outputs a DataFrame aligned to input lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from ..lexicon.load import LexiconsBundle

_TOKEN_PATTERN = r"(?u)\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b"

@dataclass(frozen=True)
class LineFeatureConfig:
    emotions: Optional[Sequence[str]] = None
    intensity_enabled: bool = True
    vad_enabled: bool = True
    intensity_aggregation: str = "mean"  # mean|sum
    vad_aggregation: str = "mean"        # mean|sum

def _count_tokens(texts: Sequence[str]) -> np.ndarray:
    ser = pd.Series(list(texts), dtype="string")
    return ser.fillna("").str.count(_TOKEN_PATTERN).astype(int).to_numpy()

def compute_line_lexicon_features(
    lines: Sequence[str],
    bundle: LexiconsBundle,
    cfg: LineFeatureConfig,
) -> pd.DataFrame:
    lines = list(lines)
    n = len(lines)
    if n == 0:
        return pd.DataFrame()

    nrc_df = bundle.nrc.df.copy()  # index word, cols emotions
    if cfg.emotions is not None:
        emo_cols = [e.lower() for e in cfg.emotions]
        for e in emo_cols:
            if e not in nrc_df.columns:
                nrc_df[e] = 0
        nrc_df = nrc_df[emo_cols]
    emo_cols = list(nrc_df.columns)

    vocab_words = nrc_df.index.astype(str).tolist()
    cv = CountVectorizer(vocabulary=vocab_words, lowercase=True, token_pattern=_TOKEN_PATTERN)
    Xw = cv.fit_transform(lines)  # (N,V)

    W = sparse.csr_matrix(nrc_df.values.astype(np.float32))  # (V,E)
    emo_counts = (Xw @ W).astype(np.float32)
    emo_counts_dense = np.asarray(emo_counts.todense(), dtype=np.float32)

    emotion_word_count = np.asarray(Xw.sum(axis=1)).ravel().astype(np.int32)
    total_tokens = _count_tokens(lines)

    emo_ratios = np.zeros_like(emo_counts_dense, dtype=np.float32)
    nz = emotion_word_count > 0
    emo_ratios[nz] = emo_counts_dense[nz] / emotion_word_count[nz, None]

    out = pd.DataFrame(index=range(n))
    for j, e in enumerate(emo_cols):
        out[f"ratio_{e}"] = emo_ratios[:, j].astype(float)
        out[f"count_{e}"] = emo_counts_dense[:, j].astype(int)
    out["emotion_word_count"] = emotion_word_count.astype(int)
    out["total_tokens"] = total_tokens.astype(int)

    # intensity (optional)
    if cfg.intensity_enabled and bundle.intensity is not None:
        inten_df = bundle.intensity.df.copy()
        inten_df = inten_df.reindex(index=nrc_df.index, columns=emo_cols, fill_value=0.0).astype(np.float32)
        I = sparse.csr_matrix(inten_df.values)  # (V,E)

        inten_sum = (Xw @ I).astype(np.float32)
        inten_sum_dense = np.asarray(inten_sum.todense(), dtype=np.float32)

        if cfg.intensity_aggregation.lower() == "sum":
            inten_out = inten_sum_dense
        else:
            mask = (inten_df.values > 0).astype(np.float32)
            M = sparse.csr_matrix(mask)
            inten_cnt = (Xw @ M).astype(np.float32)
            inten_cnt_dense = np.asarray(inten_cnt.todense(), dtype=np.float32)
            den = np.maximum(inten_cnt_dense, 1e-12)
            inten_out = inten_sum_dense / den

        for j, e in enumerate(emo_cols):
            out[f"intensity_{e}"] = inten_out[:, j].astype(float)

    # VAD (optional)
    if cfg.vad_enabled and bundle.vad is not None:
        vad_df = bundle.vad.df.copy()
        vad_words = vad_df.index.astype(str).tolist()

        cv_vad = CountVectorizer(vocabulary=vad_words, lowercase=True, token_pattern=_TOKEN_PATTERN)
        Xv = cv_vad.fit_transform(lines)  # (N, Vv)

        V = vad_df[["valence", "arousal", "dominance"]].astype(np.float32).to_numpy()
        Vmat = sparse.csr_matrix(V)  # (Vv,3)

        vad_sum = (Xv @ Vmat).astype(np.float32)
        vad_sum_dense = np.asarray(vad_sum.todense(), dtype=np.float32)
        vad_cnt = np.asarray(Xv.sum(axis=1)).ravel().astype(np.float32)

        if cfg.vad_aggregation.lower() == "sum":
            vad_out = vad_sum_dense
        else:
            den = np.maximum(vad_cnt, 1e-12)
            vad_out = vad_sum_dense / den[:, None]

        out["valence"] = vad_out[:, 0].astype(float)
        out["arousal"] = vad_out[:, 1].astype(float)
        out["dominance"] = vad_out[:, 2].astype(float)
        out["vad_word_count"] = vad_cnt.astype(int)

    return out
