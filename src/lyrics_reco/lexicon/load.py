"""
lyrics_reco.lexicon.load

Lexicon loaders (assets-based):
- NRC Emotion Lexicon (word, emotion, association 0/1)
- NRC Emotion Intensity Lexicon (word, emotion, score)
- NRC-VAD Lexicon (word, valence, arousal, dominance)

Design:
- Robust to common file formats (TSV with/without header; CSV fallback).
- Uses project paths (PATHS.root) so config can pass relative paths like:
    assets/lexicons/nrc_lexicon.txt
- Returns both:
  1) a DataFrame (easy to inspect/save as CSV), and
  2) a compact lookup dict (fast at runtime).

No pickle usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..common.paths import PATHS, ProjectPaths


PathLike = Union[str, Path]


# -----------------------------
# Utilities
# -----------------------------
def resolve_asset_path(path: PathLike, *, paths: ProjectPaths = PATHS) -> Path:
    """
    Resolve an asset path.

    If `path` is absolute -> return as is.
    If relative -> resolve from project root (PATHS.root).
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (paths.root / p).resolve()


def _read_table_guess(
    file_path: Path,
    *,
    sep_primary: str = "\t",
    encoding: str = "utf-8",
    header: Optional[int] = None,
) -> pd.DataFrame:
    """
    Try reading TSV first, then CSV as a fallback.
    Keeps things forgiving across slightly different NRC file variants.
    """
    try:
        return pd.read_csv(file_path, sep=sep_primary, header=header, encoding=encoding)
    except Exception:
        return pd.read_csv(file_path, sep=",", header=header, encoding=encoding)


def _lower_if(s: str, lower: bool) -> str:
    return s.lower() if lower else s


# -----------------------------
# NRC Emotion Lexicon
# -----------------------------
@dataclass(frozen=True)
class NrcEmotionLexicon:
    """
    NRC Emotion Lexicon container.

    df:
      index: word
      columns: emotions
      values: 0/1 (int)
    lookup:
      word -> set(emotions_with_1)
    """
    df: pd.DataFrame
    lookup: Dict[str, set]


def load_nrc_emotion_lexicon(
    path: PathLike,
    *,
    emotions: Optional[Sequence[str]] = None,
    lowercase: bool = True,
    paths: ProjectPaths = PATHS,
) -> NrcEmotionLexicon:
    """
    Load NRC Emotion Lexicon.

    Expected canonical format (TSV, no header):
        word <tab> emotion <tab> association(0/1)

    Many distributions include emotions like:
      anger, anticipation, disgust, fear, joy, sadness, surprise, trust
    but you can pass a subset via `emotions`.
    """
    fp = resolve_asset_path(path, paths=paths)
    raw = _read_table_guess(fp, header=None)
    if raw.shape[1] < 3:
        raise ValueError(f"Unexpected NRC Emotion Lexicon format: {fp} (need 3 columns)")

    raw = raw.iloc[:, :3].copy()
    raw.columns = ["word", "emotion", "association"]

    raw["word"] = raw["word"].astype(str).map(lambda x: _lower_if(x.strip(), lowercase))
    raw["emotion"] = raw["emotion"].astype(str).str.strip().str.lower()
    raw["association"] = pd.to_numeric(raw["association"], errors="coerce").fillna(0).astype(int)

    if emotions is not None:
        emo_set = {e.lower() for e in emotions}
        raw = raw[raw["emotion"].isin(emo_set)]

    # Pivot to wide: word x emotion
    df = (
        raw.pivot_table(index="word", columns="emotion", values="association", aggfunc="max", fill_value=0)
        .astype(int)
        .sort_index()
    )

    # Ensure all requested emotions appear as columns
    if emotions is not None:
        for e in [e.lower() for e in emotions]:
            if e not in df.columns:
                df[e] = 0
        df = df[[e.lower() for e in emotions]]

    # Build lookup: word -> set(emotions)
    lookup: Dict[str, set] = {}
    for w, row in df.iterrows():
        pos = set(row.index[row.values == 1].tolist())
        if pos:
            lookup[w] = pos

    return NrcEmotionLexicon(df=df, lookup=lookup)


# -----------------------------
# NRC Emotion Intensity Lexicon
# -----------------------------
@dataclass(frozen=True)
class NrcIntensityLexicon:
    """
    NRC Emotion Intensity Lexicon container.

    df:
      index: word
      columns: emotions
      values: intensity score (float)
    lookup:
      word -> dict(emotion -> score)
    """
    df: pd.DataFrame
    lookup: Dict[str, Dict[str, float]]


def load_nrc_intensity_lexicon(
    path: PathLike,
    *,
    emotions: Optional[Sequence[str]] = None,
    lowercase: bool = True,
    paths: ProjectPaths = PATHS,
) -> NrcIntensityLexicon:
    """
    Load NRC Emotion Intensity Lexicon.

    Expected canonical format (TSV, no header):
        word <tab> emotion <tab> score(float)
    """
    fp = resolve_asset_path(path, paths=paths)
    raw = _read_table_guess(fp, header=None)
    if raw.shape[1] < 3:
        raise ValueError(f"Unexpected NRC Intensity Lexicon format: {fp} (need 3 columns)")

    raw = raw.iloc[:, :3].copy()
    raw.columns = ["word", "emotion", "score"]

    raw["word"] = raw["word"].astype(str).map(lambda x: _lower_if(x.strip(), lowercase))
    raw["emotion"] = raw["emotion"].astype(str).str.strip().str.lower()
    raw["score"] = pd.to_numeric(raw["score"], errors="coerce").fillna(0.0).astype(float)

    if emotions is not None:
        emo_set = {e.lower() for e in emotions}
        raw = raw[raw["emotion"].isin(emo_set)]

    df = (
        raw.pivot_table(index="word", columns="emotion", values="score", aggfunc="mean", fill_value=0.0)
        .astype(float)
        .sort_index()
    )

    if emotions is not None:
        for e in [e.lower() for e in emotions]:
            if e not in df.columns:
                df[e] = 0.0
        df = df[[e.lower() for e in emotions]]

    lookup: Dict[str, Dict[str, float]] = {}
    for w, row in df.iterrows():
        d = {emo: float(val) for emo, val in row.to_dict().items() if float(val) > 0.0}
        if d:
            lookup[w] = d

    return NrcIntensityLexicon(df=df, lookup=lookup)


# -----------------------------
# NRC-VAD Lexicon
# -----------------------------
@dataclass(frozen=True)
class VadLexicon:
    """
    NRC-VAD Lexicon container.

    df:
      index: word
      columns: valence, arousal, dominance (float)
    lookup:
      word -> (valence, arousal, dominance)
    """
    df: pd.DataFrame
    lookup: Dict[str, Tuple[float, float, float]]


def load_vad_lexicon(
    path: PathLike,
    *,
    lowercase: bool = True,
    paths: ProjectPaths = PATHS,
) -> VadLexicon:
    """
    Load NRC-VAD Lexicon.

    Common formats:
    - TSV/CSV with header:
        Word, Valence, Arousal, Dominance
      or
        word <tab> valence <tab> arousal <tab> dominance
    """
    fp = resolve_asset_path(path, paths=paths)

    # Try with header first (common for VAD files)
    try:
        raw = _read_table_guess(fp, header=0)
        cols = [c.strip().lower() for c in raw.columns]
        raw.columns = cols
    except Exception:
        raw = _read_table_guess(fp, header=None)
        if raw.shape[1] < 4:
            raise ValueError(f"Unexpected VAD Lexicon format: {fp} (need 4 columns)")
        raw = raw.iloc[:, :4].copy()
        raw.columns = ["word", "valence", "arousal", "dominance"]

    # Map likely header variants
    col_map = {}
    for c in raw.columns:
        lc = str(c).strip().lower()
        if lc in {"word", "term"}:
            col_map[c] = "word"
        elif lc.startswith("val"):
            col_map[c] = "valence"
        elif lc.startswith("aro"):
            col_map[c] = "arousal"
        elif lc.startswith("dom"):
            col_map[c] = "dominance"
    raw = raw.rename(columns=col_map)

    needed = {"word", "valence", "arousal", "dominance"}
    if not needed.issubset(set(raw.columns)):
        raise ValueError(f"VAD Lexicon missing required columns {needed} in file: {fp}")

    df = raw[["word", "valence", "arousal", "dominance"]].copy()
    df["word"] = df["word"].astype(str).map(lambda x: _lower_if(x.strip(), lowercase))
    for c in ["valence", "arousal", "dominance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    df = df.drop_duplicates(subset=["word"]).set_index("word").sort_index()

    lookup: Dict[str, Tuple[float, float, float]] = {
        w: (float(r["valence"]), float(r["arousal"]), float(r["dominance"])) for w, r in df.iterrows()
    }

    return VadLexicon(df=df, lookup=lookup)


# -----------------------------
# Convenience: load all
# -----------------------------
@dataclass(frozen=True)
class LexiconsBundle:
    nrc: NrcEmotionLexicon
    intensity: Optional[NrcIntensityLexicon]
    vad: Optional[VadLexicon]


def load_lexicons_from_cfg(
    cfg: Mapping[str, Any],
    *,
    paths: ProjectPaths = PATHS,
) -> LexiconsBundle:
    """
    Convenience loader for configs/emotion_context.yaml style configs.

    Expected keys (typical):
      cfg["emotion"]["nrc_lexicon_file"]
      cfg["intensity"]["enabled"], cfg["intensity"]["nrc_intensity_file"]
      cfg["vad"]["enabled"], cfg["vad"]["vad_lexicon_file"]
      cfg["emotion"]["emotions"] (optional subset)
    """
    # Avoid importing Any at top-level in case user wants minimal typing elsewhere
    emotions = None
    if "emotion" in cfg and isinstance(cfg["emotion"], Mapping):
        emotions = cfg["emotion"].get("emotions", None)

    nrc_path = cfg["emotion"]["nrc_lexicon_file"]
    nrc = load_nrc_emotion_lexicon(nrc_path, emotions=emotions, paths=paths)

    intensity = None
    if cfg.get("intensity", {}).get("enabled", False):
        intensity_path = cfg["intensity"]["nrc_intensity_file"]
        intensity = load_nrc_intensity_lexicon(intensity_path, emotions=emotions, paths=paths)

    vad = None
    if cfg.get("vad", {}).get("enabled", False):
        vad_path = cfg["vad"]["vad_lexicon_file"]
        vad = load_vad_lexicon(vad_path, paths=paths)

    return LexiconsBundle(nrc=nrc, intensity=intensity, vad=vad)
