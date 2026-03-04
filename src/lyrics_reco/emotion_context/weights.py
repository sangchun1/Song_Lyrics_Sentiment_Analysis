"""lyrics_reco.emotion_context.weights

Compute per-line weights from lexicon features.

Default heuristic (configurable):
  raw = alpha * ||emotion_ratio||_2
      + beta  * mean(intensity_*)
      + gamma * abs(arousal)

Then normalize per song:
  - none: use raw as-is
  - l1: raw / sum(raw)
  - softmax: softmax(raw / T)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class WeightConfig:
    alpha_emotion: float = 1.0
    beta_intensity: float = 1.0
    gamma_arousal: float = 1.0

    normalize: str = "softmax"   # none | l1 | softmax
    softmax_temperature: float = 1.0

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
            out[idx] = 0.0
        else:
            out[idx] = (ex / s).astype(np.float32)
    return out

def compute_line_weights(
    line_feat_df: pd.DataFrame,
    song_index: Sequence[int],
    emotions: Sequence[str],
    cfg: WeightConfig,
    *,
    use_intensity: bool = True,
    use_vad: bool = True,
) -> np.ndarray:
    song_index = np.asarray(song_index, dtype=int)
    n = len(song_index)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    emo_cols = [f"ratio_{e.lower()}" for e in emotions if f"ratio_{e.lower()}" in line_feat_df.columns]
    emo = line_feat_df[emo_cols].to_numpy(dtype=np.float32) if emo_cols else np.zeros((n, 0), dtype=np.float32)
    emo_strength = np.linalg.norm(emo, axis=1).astype(np.float32)

    intensity_strength = np.zeros((n,), dtype=np.float32)
    if use_intensity:
        inten_cols = [c for c in line_feat_df.columns if c.startswith("intensity_")]
        if inten_cols:
            inten = line_feat_df[inten_cols].to_numpy(dtype=np.float32)
            intensity_strength = np.mean(inten, axis=1).astype(np.float32)

    arousal_strength = np.zeros((n,), dtype=np.float32)
    if use_vad and "arousal" in line_feat_df.columns:
        arousal_strength = np.abs(line_feat_df["arousal"].to_numpy(dtype=np.float32))

    raw = (
        float(cfg.alpha_emotion) * emo_strength
        + float(cfg.beta_intensity) * intensity_strength
        + float(cfg.gamma_arousal) * arousal_strength
    )
    raw = np.clip(raw, 0.0, None).astype(np.float32)

    norm = str(cfg.normalize).lower()
    if norm == "none":
        return raw

    if norm == "l1":
        out = np.zeros_like(raw, dtype=np.float32)
        for g in np.unique(song_index):
            idx = np.where(song_index == g)[0]
            s = float(raw[idx].sum())
            if s <= 1e-12:
                out[idx] = 0.0
            else:
                out[idx] = raw[idx] / s
        return out

    return _softmax_group(raw, song_index, temperature=float(cfg.softmax_temperature))
