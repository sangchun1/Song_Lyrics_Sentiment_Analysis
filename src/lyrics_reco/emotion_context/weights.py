"""
lyrics_reco.emotion_context.weights

Research-plan line weighting:

For each line i in a song:
- e_tilde_i : normalized 6D line emotion distribution
- alpha_tilde_i : average line emotion intensity
- p(s) : song-level emotion distribution

Weight:
    w_i = alpha_tilde_i * (e_tilde_i^T p(s))

Then normalize per song:
- none
- l1
- softmax
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeightConfig:
    # config key names are kept for backward compatibility with current yaml
    alpha_emotion: float = 1.0   # global multiplier on plan weight
    beta_intensity: float = 1.0  # kept only for config compatibility (unused)
    gamma_arousal: float = 0.0   # kept only for config compatibility (unused)
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
    denom = X.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return X / denom


def compute_line_weights(
    line_feat_df: pd.DataFrame,
    song_index: Sequence[int],
    emotions: Sequence[str],
    cfg: WeightConfig,
    *,
    use_intensity: bool = True,
    use_vad: bool = True,  # kept for signature compatibility; not used in plan weight
) -> np.ndarray:
    n = len(line_feat_df)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    groups = np.asarray(song_index, dtype=int)

    ratio_cols = [f"ratio_{str(e).lower()}" for e in emotions if f"ratio_{str(e).lower()}" in line_feat_df.columns]
    intensity_cols = [f"intensity_{str(e).lower()}" for e in emotions if f"intensity_{str(e).lower()}" in line_feat_df.columns]

    if not ratio_cols:
        raise ValueError("No ratio_* columns found in line_feat_df for the given emotions.")

    raw = np.zeros(n, dtype=np.float32)

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        sub = line_feat_df.iloc[idx]

        # e_tilde_i: normalized 6D line emotion distribution
        E = sub[ratio_cols].fillna(0.0).astype(np.float32).to_numpy()
        E_tilde = _row_normalize_nonnegative(E, cfg.eps)

        # alpha_tilde_i: average line intensity
        if use_intensity and intensity_cols:
            A = sub[intensity_cols].fillna(0.0).astype(np.float32).to_numpy()
            alpha_tilde = np.clip(A.mean(axis=1), 0.0, None).astype(np.float32)
        else:
            # if intensity is disabled, use binary presence of emotion words
            alpha_tilde = (E.sum(axis=1) > 0).astype(np.float32)

        # p(s): song-level emotion distribution, intensity-weighted
        denom = float(alpha_tilde.sum())
        if denom > cfg.eps:
            p_song = (alpha_tilde[:, None] * E_tilde).sum(axis=0) / denom
        else:
            p_song = E_tilde.mean(axis=0)

        p_song = np.clip(p_song.astype(np.float32), 0.0, None)
        p_song = p_song / max(float(p_song.sum()), cfg.eps)

        # alignment term: e_tilde_i^T p(s)
        align = (E_tilde * p_song[None, :]).sum(axis=1).astype(np.float32)

        # research-plan raw weight
        raw[idx] = (cfg.alpha_emotion * alpha_tilde * align).astype(np.float32)

    norm = str(cfg.normalize).lower().strip()

    if norm == "none":
        return raw.astype(np.float32)

    if norm == "l1":
        return _l1_group(raw, groups, cfg.eps)

    if norm == "softmax":
        return _softmax_group(raw, groups, cfg.softmax_temperature)

    raise ValueError(f"Unknown normalize mode: {cfg.normalize}")