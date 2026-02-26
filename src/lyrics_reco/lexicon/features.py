"""
lyrics_reco.lexicon.features

Lexicon-based feature extraction utilities.

Do you *need* this file?
- Not strictly required (you could compute features inside each pipeline),
  but it's strongly recommended because the same lexicon features are reused across:
  - baseline (emotion-weighted representations)
  - emotion_context (line weighting via emotion/intensity/VAD)
  - evaluation (Emotion Consistency@K; query/reco emotion profiles)

This module provides:
- Emotion profile (counts/ratios) from NRC Emotion Lexicon
- Emotion intensity aggregation from NRC Emotion Intensity Lexicon
- VAD aggregation from NRC-VAD Lexicon
- A unified FeatureResult container with stable ordering

Assumptions:
- Input is a list of tokens/words (already tokenized).
- We do lightweight normalization (lowercase + strip) here.
- No pickle usage; outputs are Python dicts / numpy arrays / DataFrames.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .load import LexiconsBundle, NrcEmotionLexicon, NrcIntensityLexicon, VadLexicon


_WORD_CLEAN_RE = re.compile(r"^[\W_]+|[\W_]+$")  # trim non-word chars at ends


def normalize_token(token: str, *, lowercase: bool = True) -> str:
    """Lightweight token normalization for lexicon lookup."""
    t = token.strip()
    t = _WORD_CLEAN_RE.sub("", t)
    if lowercase:
        t = t.lower()
    return t


def normalize_tokens(tokens: Sequence[str], *, lowercase: bool = True) -> List[str]:
    return [normalize_token(t, lowercase=lowercase) for t in tokens if t and normalize_token(t, lowercase=lowercase)]


# -----------------------------
# Feature container
# -----------------------------
@dataclass(frozen=True)
class FeatureResult:
    emotions: Tuple[str, ...]                 # stable order
    emotion_counts: np.ndarray                # shape (E,)
    emotion_ratios: np.ndarray                # shape (E,)
    emotion_word_count: int                   # total matched emotion tokens
    total_tokens: int                         # total tokens (after normalization)

    # Optional extras
    intensity: Optional[np.ndarray] = None    # shape (E,) mean intensity per emotion
    vad: Optional[Tuple[float, float, float]] = None  # (valence, arousal, dominance)
    vad_word_count: int = 0                   # matched VAD tokens

    def to_dict(self, *, prefix: str = "") -> Dict[str, Union[int, float]]:
        """Flatten to a dict suitable for DataFrame rows / CSV logging."""
        p = prefix
        out: Dict[str, Union[int, float]] = {
            f"{p}emotion_word_count": int(self.emotion_word_count),
            f"{p}total_tokens": int(self.total_tokens),
            f"{p}vad_word_count": int(self.vad_word_count),
        }
        for i, e in enumerate(self.emotions):
            out[f"{p}count_{e}"] = int(self.emotion_counts[i])
            out[f"{p}ratio_{e}"] = float(self.emotion_ratios[i])

        if self.intensity is not None:
            for i, e in enumerate(self.emotions):
                out[f"{p}intensity_{e}"] = float(self.intensity[i])

        if self.vad is not None:
            v, a, d = self.vad
            out[f"{p}valence"] = float(v)
            out[f"{p}arousal"] = float(a)
            out[f"{p}dominance"] = float(d)

        return out


# -----------------------------
# Emotion profile
# -----------------------------
def compute_emotion_profile(
    tokens: Sequence[str],
    nrc: NrcEmotionLexicon,
    *,
    emotions: Optional[Sequence[str]] = None,
    lowercase: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, int, Tuple[str, ...]]:
    """
    Compute emotion counts/ratios using NRC Emotion Lexicon.

    Returns:
        counts (E,), ratios (E,), emotion_word_count, total_tokens, emotions_order
    """
    toks = normalize_tokens(tokens, lowercase=lowercase)
    total = len(toks)

    emo_order = tuple([e.lower() for e in (emotions if emotions is not None else list(nrc.df.columns))])
    emo_index = {e: i for i, e in enumerate(emo_order)}

    counts = np.zeros(len(emo_order), dtype=np.int32)
    matched = 0

    # Fast lookup: nrc.lookup maps word -> set(emotions)
    for w in toks:
        emos = nrc.lookup.get(w)
        if not emos:
            continue
        matched += 1
        for e in emos:
            if e in emo_index:
                counts[emo_index[e]] += 1

    if matched > 0:
        ratios = counts.astype(np.float64) / float(matched)
    else:
        ratios = np.zeros_like(counts, dtype=np.float64)

    return counts, ratios, matched, total, emo_order


# -----------------------------
# Intensity profile
# -----------------------------
def compute_intensity_profile(
    tokens: Sequence[str],
    intensity_lex: NrcIntensityLexicon,
    *,
    emotions: Sequence[str],
    lowercase: bool = True,
    aggregation: str = "mean",  # mean | sum | max
) -> np.ndarray:
    """
    Aggregate emotion intensity scores per emotion for the given tokens.

    Returns:
        intensity_vec (E,)
    """
    toks = normalize_tokens(tokens, lowercase=lowercase)
    emo_order = [e.lower() for e in emotions]
    emo_index = {e: i for i, e in enumerate(emo_order)}

    # accumulators
    sums = np.zeros(len(emo_order), dtype=np.float64)
    cnts = np.zeros(len(emo_order), dtype=np.int32)
    maxs = np.zeros(len(emo_order), dtype=np.float64)

    lookup = intensity_lex.lookup  # word -> {emotion: score}

    for w in toks:
        d = lookup.get(w)
        if not d:
            continue
        for e, s in d.items():
            if e not in emo_index:
                continue
            i = emo_index[e]
            sums[i] += float(s)
            cnts[i] += 1
            if float(s) > maxs[i]:
                maxs[i] = float(s)

    agg = aggregation.lower()
    if agg == "sum":
        return sums
    if agg == "max":
        return maxs
    # mean (default)
    out = np.zeros(len(emo_order), dtype=np.float64)
    mask = cnts > 0
    out[mask] = sums[mask] / cnts[mask]
    return out


# -----------------------------
# VAD profile
# -----------------------------
def compute_vad(
    tokens: Sequence[str],
    vad_lex: VadLexicon,
    *,
    lowercase: bool = True,
    aggregation: str = "mean",  # mean | sum | max (max is per-dimension)
) -> Tuple[Tuple[float, float, float], int]:
    """
    Aggregate VAD over matched tokens.

    Returns:
        (valence, arousal, dominance), matched_count
    """
    toks = normalize_tokens(tokens, lowercase=lowercase)
    lookup = vad_lex.lookup  # word -> (v,a,d)

    vals: List[Tuple[float, float, float]] = []
    for w in toks:
        v = lookup.get(w)
        if v is not None:
            vals.append(v)

    if not vals:
        return (0.0, 0.0, 0.0), 0

    arr = np.array(vals, dtype=np.float64)  # (N, 3)
    agg = aggregation.lower()
    if agg == "sum":
        v, a, d = arr.sum(axis=0).tolist()
        return (float(v), float(a), float(d)), arr.shape[0]
    if agg == "max":
        v, a, d = arr.max(axis=0).tolist()
        return (float(v), float(a), float(d)), arr.shape[0]

    # mean
    v, a, d = arr.mean(axis=0).tolist()
    return (float(v), float(a), float(d)), arr.shape[0]


# -----------------------------
# Unified interface
# -----------------------------
def compute_lexicon_features(
    tokens: Sequence[str],
    bundle: LexiconsBundle,
    *,
    emotions: Optional[Sequence[str]] = None,
    lowercase: bool = True,
    intensity_aggregation: str = "mean",
    vad_aggregation: str = "mean",
) -> FeatureResult:
    """
    Compute (emotion counts/ratios) + optional (intensity) + optional (VAD).

    This is the function you typically call from pipelines.
    """
    nrc = bundle.nrc
    counts, ratios, emo_matched, total, emo_order = compute_emotion_profile(
        tokens, nrc, emotions=emotions, lowercase=lowercase
    )

    intensity_vec = None
    if bundle.intensity is not None:
        intensity_vec = compute_intensity_profile(
            tokens,
            bundle.intensity,
            emotions=emo_order,
            lowercase=lowercase,
            aggregation=intensity_aggregation,
        )

    vad_tuple = None
    vad_cnt = 0
    if bundle.vad is not None:
        vad_tuple, vad_cnt = compute_vad(
            tokens,
            bundle.vad,
            lowercase=lowercase,
            aggregation=vad_aggregation,
        )

    return FeatureResult(
        emotions=tuple(emo_order),
        emotion_counts=counts,
        emotion_ratios=ratios,
        emotion_word_count=int(emo_matched),
        total_tokens=int(total),
        intensity=intensity_vec,
        vad=vad_tuple,
        vad_word_count=int(vad_cnt),
    )


def batch_compute_features(
    token_lists: Sequence[Sequence[str]],
    bundle: LexiconsBundle,
    *,
    emotions: Optional[Sequence[str]] = None,
    lowercase: bool = True,
    intensity_aggregation: str = "mean",
    vad_aggregation: str = "mean",
    prefix: str = "",
) -> pd.DataFrame:
    """
    Batch helper: compute features for many items and return a DataFrame.

    token_lists: list of token lists (e.g., songs or lines)
    """
    rows = []
    for toks in token_lists:
        fr = compute_lexicon_features(
            toks,
            bundle,
            emotions=emotions,
            lowercase=lowercase,
            intensity_aggregation=intensity_aggregation,
            vad_aggregation=vad_aggregation,
        )
        rows.append(fr.to_dict(prefix=prefix))
    return pd.DataFrame(rows)
