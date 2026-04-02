"""lyrics_reco.emotion_context.aggregate

Aggregate per-line embeddings/features into song-level vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AggregateConfig:
    method: str = "weighted_mean"   # weighted_mean | mean


def _group_weighted_mean(X: np.ndarray, groups: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    groups = np.asarray(groups, dtype=int)
    uniq = np.unique(groups)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    d = X.shape[1]
    out = np.zeros((len(uniq), d), dtype=np.float32)
    for gi, g in enumerate(uniq):
        idx = np.where(groups == g)[0]
        Xi = X[idx]
        if len(idx) == 0:
            continue
        if w is None:
            out[gi] = Xi.mean(axis=0)
        else:
            wi = np.asarray(w[idx], dtype=np.float64)
            s = float(wi.sum())
            if s <= 1e-12:
                out[gi] = Xi.mean(axis=0)
            else:
                out[gi] = (Xi * wi[:, None]).sum(axis=0) / s
    return out


def aggregate_song_embedding(
    line_embeddings: np.ndarray,
    song_index: Sequence[int],
    weights: np.ndarray | None,
    *,
    agg_cfg: AggregateConfig = AggregateConfig(),
) -> np.ndarray:
    groups = np.asarray(song_index, dtype=int)
    w = None if agg_cfg.method == "mean" else weights
    return _group_weighted_mean(np.asarray(line_embeddings, dtype=np.float32), groups, w)


def aggregate_song_emotion_tail(
    line_feat_df: pd.DataFrame,
    song_index: Sequence[int],
    emotions: Sequence[str],
    *,
    include_vad: bool = True,
    include_intensity: bool = False,
    song_feature_weight: str = "emotion_word_count",
) -> Dict[str, np.ndarray]:
    groups = np.asarray(song_index, dtype=int)
    uniq = np.unique(groups)
    out: Dict[str, np.ndarray] = {}

    ratio_cols = [f"ratio_{e.lower()}" for e in emotions if f"ratio_{e.lower()}" in line_feat_df.columns]
    ratio_mat = line_feat_df[ratio_cols].to_numpy(dtype=np.float32) if ratio_cols else np.zeros((len(groups), 0), dtype=np.float32)
    weight_col = song_feature_weight if song_feature_weight in line_feat_df.columns else "emotion_word_count"
    w = line_feat_df[weight_col].fillna(0).to_numpy(dtype=np.float32)
    out["emotion_ratio"] = _group_weighted_mean(ratio_mat, groups, w) if ratio_mat.shape[1] else np.zeros((len(uniq), 0), dtype=np.float32)

    if include_intensity:
        inten_cols = [c for c in line_feat_df.columns if c.startswith("intensity_")]
        inten_mat = line_feat_df[inten_cols].to_numpy(dtype=np.float32) if inten_cols else np.zeros((len(groups), 0), dtype=np.float32)
        iw = line_feat_df.get("line_emotion_mass", pd.Series(np.ones(len(line_feat_df)))).fillna(0).to_numpy(dtype=np.float32)
        out["intensity"] = _group_weighted_mean(inten_mat, groups, iw) if inten_mat.shape[1] else np.zeros((len(uniq), 0), dtype=np.float32)

    if include_vad:
        vad_cols = [c for c in ["valence", "arousal", "dominance"] if c in line_feat_df.columns]
        vad_mat = line_feat_df[vad_cols].to_numpy(dtype=np.float32) if vad_cols else np.zeros((len(groups), 0), dtype=np.float32)
        vw = line_feat_df.get("vad_word_count", pd.Series(np.ones(len(line_feat_df)))).fillna(0).to_numpy(dtype=np.float32)
        out["vad"] = _group_weighted_mean(vad_mat, groups, vw) if vad_mat.shape[1] else np.zeros((len(uniq), 0), dtype=np.float32)

    return out


def concat_song_vector(components: Dict[str, np.ndarray], *, layout: str = "embedding_ratio_vad") -> np.ndarray:
    if layout == "embedding_ratio_vad":
        keys = ["embedding", "emotion_ratio", "vad"]
    elif layout == "embedding_ratio_intensity_vad":
        keys = ["embedding", "emotion_ratio", "intensity", "vad"]
    else:
        raise ValueError(f"Unknown vector layout: {layout}")

    parts = []
    for key in keys:
        if key in components and components[key].size:
            parts.append(components[key])
    if not parts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)
