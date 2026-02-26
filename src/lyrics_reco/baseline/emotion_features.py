"""
lyrics_reco.baseline.emotion_features

Lexicon-based feature tables (CSV-friendly), optimized for large N (e.g., 500k).

- Uses CountVectorizer restricted to NRC vocab (sparse)
- Computes emotion counts/ratios via matrix multiplication (Xw @ W)
- Optional intensity/VAD are OFF by default (turn on only if needed)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from ..lexicon.load import LexiconsBundle

_TOKEN_PATTERN = r"(?u)\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b"


def _count_tokens_series(texts: pd.Series) -> np.ndarray:
    return texts.astype(str).str.count(_TOKEN_PATTERN).fillna(0).astype(int).to_numpy()


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.maximum(den, 1e-12)
    return num / den


def build_lexicon_feature_table(
    df: pd.DataFrame,
    bundle: LexiconsBundle,
    *,
    song_id_col: str = "song_id",
    text_col: str = "lyrics_clean",
    emotions: Optional[Sequence[str]] = None,
    include_intensity: bool = False,
    include_vad: bool = False,
    intensity_aggregation: str = "mean",  # mean|sum (max는 비싸서 미지원/근사)
    vad_aggregation: str = "mean",        # mean|sum
) -> pd.DataFrame:
    if song_id_col not in df.columns:
        raise ValueError(f"missing song_id_col: {song_id_col}")
    if text_col not in df.columns:
        raise ValueError(f"missing text_col: {text_col}")

    texts = df[text_col].astype(str)
    song_ids = df[song_id_col].astype(str).to_numpy()

    # ----- NRC emotion counts -----
    nrc_df = bundle.nrc.df.copy()  # index=word, cols=emotions
    if emotions is not None:
        emo_cols = [e.lower() for e in emotions]
        for e in emo_cols:
            if e not in nrc_df.columns:
                nrc_df[e] = 0
        nrc_df = nrc_df[emo_cols]
    emo_cols = list(nrc_df.columns)

    vocab_words = nrc_df.index.astype(str).tolist()
    cv = CountVectorizer(vocabulary=vocab_words, lowercase=True, token_pattern=_TOKEN_PATTERN)
    Xw = cv.fit_transform(texts.tolist())  # (N, V) counts for lexicon words only

    W = sparse.csr_matrix(nrc_df.values.astype(np.float32))  # (V, E)
    emo_counts = (Xw @ W).astype(np.float32)                 # (N, E)
    emo_counts_dense = np.asarray(emo_counts.todense(), dtype=np.float32)

    emotion_word_count = np.asarray(Xw.sum(axis=1)).ravel().astype(np.int32)
    total_tokens = _count_tokens_series(texts)

    emo_ratios = np.zeros_like(emo_counts_dense, dtype=np.float32)
    nz = emotion_word_count > 0
    emo_ratios[nz] = emo_counts_dense[nz] / emotion_word_count[nz, None]

    out = pd.DataFrame({"song_id": song_ids})
    for j, e in enumerate(emo_cols):
        out[f"count_{e}"] = emo_counts_dense[:, j].astype(int)
        out[f"ratio_{e}"] = emo_ratios[:, j].astype(float)

    out["emotion_word_count"] = emotion_word_count.astype(int)
    out["total_tokens"] = total_tokens.astype(int)

    # ----- Intensity (optional) -----
    if include_intensity and bundle.intensity is not None:
        inten_df = bundle.intensity.df.copy()
        inten_df = inten_df.reindex(index=nrc_df.index, columns=emo_cols, fill_value=0.0).astype(np.float32)
        I = sparse.csr_matrix(inten_df.values)  # (V, E)

        inten_sum = (Xw @ I).astype(np.float32)
        inten_sum_dense = np.asarray(inten_sum.todense(), dtype=np.float32)

        if intensity_aggregation.lower() == "sum":
            inten_out = inten_sum_dense
        else:
            # mean (default)
            mask = (inten_df.values > 0).astype(np.float32)
            M = sparse.csr_matrix(mask)
            inten_cnt = (Xw @ M).astype(np.float32)
            inten_cnt_dense = np.asarray(inten_cnt.todense(), dtype=np.float32)
            inten_out = _safe_divide(inten_sum_dense, inten_cnt_dense)

        for j, e in enumerate(emo_cols):
            out[f"intensity_{e}"] = inten_out[:, j].astype(float)

    # ----- VAD (optional) -----
    if include_vad and bundle.vad is not None:
        vad_df = bundle.vad.df.copy()  # index=word, cols=valence/arousal/dominance
        vad_words = vad_df.index.astype(str).tolist()

        cv_vad = CountVectorizer(vocabulary=vad_words, lowercase=True, token_pattern=_TOKEN_PATTERN)
        Xv = cv_vad.fit_transform(texts.tolist())  # (N, Vv)

        V = vad_df[["valence", "arousal", "dominance"]].astype(np.float32).to_numpy()  # (Vv, 3)
        Vmat = sparse.csr_matrix(V)

        vad_sum = (Xv @ Vmat).astype(np.float32)
        vad_sum_dense = np.asarray(vad_sum.todense(), dtype=np.float32)
        vad_cnt = np.asarray(Xv.sum(axis=1)).ravel().astype(np.float32)

        if vad_aggregation.lower() == "sum":
            vad_out = vad_sum_dense
        else:
            vad_out = _safe_divide(vad_sum_dense, vad_cnt[:, None])

        out["valence"] = vad_out[:, 0].astype(float)
        out["arousal"] = vad_out[:, 1].astype(float)
        out["dominance"] = vad_out[:, 2].astype(float)
        out["vad_word_count"] = vad_cnt.astype(int)

    return out