"""lyrics_reco.emotion_context.line_features

Vectorized lexicon feature computation for a batch of lyric lines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from ..lexicon.load import LexiconsBundle

logger = logging.getLogger(__name__)

# NOTE:
# Use a normal word-boundary token pattern. The previous pattern contained a
# broken boundary character, which can silently corrupt lexicon matching.
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


def _log_line_feature_qc(out: pd.DataFrame) -> None:
    """Emit lightweight QA stats for line-level lexicon features."""
    if out.empty:
        return

    emotion_nonzero_ratio = float((out["emotion_word_count"] > 0).mean())
    vad_nonzero_ratio = float((out["vad_word_count"] > 0).mean()) if "vad_word_count" in out.columns else 0.0
    intensity_mean_avg = float(out["line_intensity_mean"].mean()) if "line_intensity_mean" in out.columns else 0.0

    logger.info(
        "line feature QA | lines=%d | emotion_nonzero_ratio=%.4f | vad_nonzero_ratio=%.4f | avg_line_intensity_mean=%.4f",
        len(out),
        emotion_nonzero_ratio,
        vad_nonzero_ratio,
        intensity_mean_avg,
    )


def compute_line_lexicon_features(
    lines: Sequence[str],
    bundle: LexiconsBundle,
    cfg: LineFeatureConfig,
) -> pd.DataFrame:
    lines = list(lines)
    n = len(lines)
    if n == 0:
        return pd.DataFrame()

    nrc_df = bundle.nrc.df.copy()
    if cfg.emotions is not None:
        emo_cols = [e.lower() for e in cfg.emotions]
        for e in emo_cols:
            if e not in nrc_df.columns:
                nrc_df[e] = 0
        nrc_df = nrc_df[emo_cols]
    emo_cols = list(nrc_df.columns)

    vocab_words = nrc_df.index.astype(str).tolist()
    cv = CountVectorizer(vocabulary=vocab_words, lowercase=True, token_pattern=_TOKEN_PATTERN)
    Xw = cv.fit_transform(lines)

    W = sparse.csr_matrix(nrc_df.values.astype(np.float32))
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

    if cfg.intensity_enabled and bundle.intensity is not None:
        inten_df = bundle.intensity.df.copy()
        inten_df = inten_df.reindex(index=nrc_df.index, columns=emo_cols, fill_value=0.0).astype(np.float32)
        I = sparse.csr_matrix(inten_df.values)

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

    if cfg.vad_enabled and bundle.vad is not None:
        vad_df = bundle.vad.df.copy()
        vad_words = vad_df.index.astype(str).tolist()

        cv_vad = CountVectorizer(vocabulary=vad_words, lowercase=True, token_pattern=_TOKEN_PATTERN)
        Xv = cv_vad.fit_transform(lines)

        V = vad_df[["valence", "arousal", "dominance"]].astype(np.float32).to_numpy()
        Vmat = sparse.csr_matrix(V)

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

    count_cols = [c for c in out.columns if c.startswith("count_")]
    inten_cols = [c for c in out.columns if c.startswith("intensity_")]
    out["line_emotion_mass"] = out[count_cols].sum(axis=1).astype(float) if count_cols else 0.0
    out["line_has_emotion"] = (out["emotion_word_count"] > 0).astype(int)

    if inten_cols:
        tmp = out[inten_cols].replace(0, np.nan)
        out["line_intensity_mean"] = tmp.mean(axis=1).fillna(0.0).astype(float)

        intensity_mass = np.zeros(len(out), dtype=np.float32)
        for c in count_cols:
            emo = c.replace("count_", "")
            ic = f"intensity_{emo}"
            if ic in out.columns:
                intensity_mass += (
                    out[c].to_numpy(dtype=np.float32)
                    * out[ic].to_numpy(dtype=np.float32)
                )
        out["line_intensity_mass"] = intensity_mass.astype(float)
    else:
        out["line_intensity_mean"] = 0.0
        out["line_intensity_mass"] = 0.0

    if "vad_word_count" not in out.columns:
        out["vad_word_count"] = 0
        out["valence"] = 0.0
        out["arousal"] = 0.0
        out["dominance"] = 0.0

    _log_line_feature_qc(out)
    return out
