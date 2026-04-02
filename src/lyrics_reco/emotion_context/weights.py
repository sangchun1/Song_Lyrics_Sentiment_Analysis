"""lyrics_reco.emotion_context.weights

Research-plan line weighting with explicit intensity/arousal control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeightConfig:
    alpha_emotion: float = 1.0
    beta_intensity: float = 0.5
    gamma_arousal: float = 0.25
    normalize: str = "softmax"   # none | l1 | softmax
    softmax_temperature: float = 1.0
    eps: float = 1e-8


def _softmax_group(values: np.ndarray, groups: np.ndarray, temperature: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    t = max(float(temperature), 1e-6)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        x = values[idx].astype(np.float64) / t
        x = x - np.max(x)
        ex = np.exp(x)
        s = ex.sum()
        if s <= 0:
            out[idx] = np.full(len(idx), 1.0 / max(len(idx), 1), dtype=np.float32)
        else:
            out[idx] = (ex / s).astype(np.float32)
    return out


def _l1_group(values: np.ndarray, groups: np.ndarray, eps: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        v = np.clip(values[idx].astype(np.float32), 0.0, None)
        s = float(v.sum())
        if s <= eps:
            out[idx] = np.full(len(idx), 1.0 / max(len(idx), 1), dtype=np.float32)
        else:
            out[idx] = v / s
    return out


def _row_normalize_nonnegative(X: np.ndarray, eps: float) -> np.ndarray:
    X = np.clip(X.astype(np.float32), 0.0, None)
    denom = np.maximum(X.sum(axis=1, keepdims=True), eps)
    return X / denom


def compute_line_weights(
    line_feat_df: pd.DataFrame,
    song_index: Sequence[int],
    emotions: Sequence[str],
    cfg: WeightConfig,
    *,
    use_intensity: bool = True,
    use_vad: bool = True,
    return_details: bool = False,
):
    n = len(line_feat_df)
    if n == 0:
        empty = np.zeros(0, dtype=np.float32)
        return {"weights": empty, "raw": empty, "align": empty, "alpha_tilde": empty, "arousal_tilde": empty} if return_details else empty

    groups = np.asarray(song_index, dtype=int)
    ratio_cols = [f"ratio_{str(e).lower()}" for e in emotions if f"ratio_{str(e).lower()}" in line_feat_df.columns]
    if not ratio_cols:
        raise ValueError("No ratio_* columns found in line_feat_df for the given emotions.")

    raw = np.zeros(n, dtype=np.float32)
    align_all = np.zeros(n, dtype=np.float32)
    alpha_all = np.zeros(n, dtype=np.float32)
    arousal_all = np.ones(n, dtype=np.float32)

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        sub = line_feat_df.iloc[idx]

        E = sub[ratio_cols].fillna(0.0).astype(np.float32).to_numpy()
        E_tilde = _row_normalize_nonnegative(E, cfg.eps)

        if use_intensity and "line_intensity_mean" in sub.columns:
            alpha_tilde = np.clip(sub["line_intensity_mean"].to_numpy(dtype=np.float32), 0.0, None)
        else:
            alpha_tilde = (E.sum(axis=1) > 0).astype(np.float32)

        denom = max(float(alpha_tilde.sum()), cfg.eps)
        p_song = (alpha_tilde[:, None] * E_tilde).sum(axis=0) / denom
        p_song = np.clip(p_song.astype(np.float32), 0.0, None)
        p_song = p_song / max(float(p_song.sum()), cfg.eps)

        align = (E_tilde * p_song[None, :]).sum(axis=1).astype(np.float32)

        if use_vad and "arousal" in sub.columns:
            a = sub["arousal"].fillna(0.0).to_numpy(dtype=np.float32)
            a_min, a_max = float(a.min()), float(a.max())
            if a_max - a_min > cfg.eps:
                arousal_tilde = (a - a_min) / (a_max - a_min)
            else:
                arousal_tilde = np.ones_like(a, dtype=np.float32)
        else:
            arousal_tilde = np.ones(len(sub), dtype=np.float32)

        raw_g = cfg.alpha_emotion * np.clip(align, cfg.eps, None)
        if use_intensity and cfg.beta_intensity != 0:
            raw_g *= np.power(np.clip(alpha_tilde, cfg.eps, None), cfg.beta_intensity)
        if use_vad and cfg.gamma_arousal != 0:
            raw_g *= np.power(np.clip(arousal_tilde, cfg.eps, None), cfg.gamma_arousal)

        raw[idx] = raw_g.astype(np.float32)
        align_all[idx] = align
        alpha_all[idx] = alpha_tilde
        arousal_all[idx] = arousal_tilde

    norm = str(cfg.normalize).lower().strip()
    if norm == "none":
        weights = raw.astype(np.float32)
    elif norm == "l1":
        weights = _l1_group(raw, groups, cfg.eps)
    elif norm == "softmax":
        weights = _softmax_group(raw, groups, cfg.softmax_temperature)
    else:
        raise ValueError(f"Unknown normalize mode: {cfg.normalize}")

    if return_details:
        return {
            "weights": weights.astype(np.float32),
            "raw": raw.astype(np.float32),
            "align": align_all.astype(np.float32),
            "alpha_tilde": alpha_all.astype(np.float32),
            "arousal_tilde": arousal_all.astype(np.float32),
        }
    return weights.astype(np.float32)
