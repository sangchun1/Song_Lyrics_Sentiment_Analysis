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

def _group_weighted_mean(
    X: np.ndarray,
    groups: np.ndarray,
    w: Optional[np.ndarray],
) -> np.ndarray:
    groups = np.asarray(groups, dtype=int)
    uniq = np.unique(groups)
    d = X.shape[1]
    out = np.zeros((len(uniq), d), dtype=np.float32)

    for gi, g in enumerate(uniq):
        idx = np.where(groups == g)[0]
        Xi = X[idx]
        if w is None:
            out[gi] = Xi.mean(axis=0)
        else:
            wi = w[idx].astype(np.float64)
            s = wi.sum()
            if s <= 1e-12:
                out[gi] = Xi.mean(axis=0)
            else:
                out[gi] = (Xi * wi[:, None]).sum(axis=0) / s
    return out

def aggregate_song_components(
    line_embeddings: np.ndarray,
    line_feat_df: pd.DataFrame,
    song_index: Sequence[int],
    weights: Optional[np.ndarray],
    emotions: Sequence[str],
    *,
    include_intensity: bool,
    include_vad: bool,
    agg_cfg: AggregateConfig = AggregateConfig(),
) -> Dict[str, np.ndarray]:
    song_index = np.asarray(song_index, dtype=int)
    uniq = np.unique(song_index)

    w = None if agg_cfg.method == "mean" else weights

    emb = _group_weighted_mean(line_embeddings, song_index, w)

    emo_cols = [f"ratio_{e.lower()}" for e in emotions if f"ratio_{e.lower()}" in line_feat_df.columns]
    emo_mat = line_feat_df[emo_cols].to_numpy(dtype=np.float32) if emo_cols else np.zeros((len(song_index), 0), dtype=np.float32)
    emo_agg = _group_weighted_mean(emo_mat, song_index, w) if emo_mat.shape[1] else np.zeros((len(uniq), 0), dtype=np.float32)

    out: Dict[str, np.ndarray] = {"embedding": emb, "emotion_ratio": emo_agg}

    if include_intensity:
        inten_cols = [c for c in line_feat_df.columns if c.startswith("intensity_")]
        if inten_cols:
            inten_mat = line_feat_df[inten_cols].to_numpy(dtype=np.float32)
            out["intensity"] = _group_weighted_mean(inten_mat, song_index, w)
        else:
            out["intensity"] = np.zeros((len(uniq), 0), dtype=np.float32)

    if include_vad:
        cols = [c for c in ["valence", "arousal", "dominance"] if c in line_feat_df.columns]
        if cols:
            vad_mat = line_feat_df[cols].to_numpy(dtype=np.float32)
            out["vad"] = _group_weighted_mean(vad_mat, song_index, w)
        else:
            out["vad"] = np.zeros((len(uniq), 0), dtype=np.float32)

    return out

def concat_song_vector(components: Dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for key in ["embedding", "emotion_ratio", "intensity", "vad"]:
        if key in components and components[key].size:
            parts.append(components[key])
    if not parts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)
