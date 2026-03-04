"""lyrics_reco.emotion_context.builder

Batched builder for emotion-context vectors.

Output:
  - song vectors z(s) as DataFrame (CSV-friendly)
  - optional line feature table (huge; off by default)

This module focuses on correctness + reproducibility + reasonable performance.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..common.paths import PATHS, ProjectPaths
from ..common.io import save_csv
from ..lexicon.load import load_lexicons_from_cfg
from .utils import cfg_get
from .splitter import SplitConfig, explode_songs_to_lines
from .embedder import EmbedderConfig, SentenceTransformerEmbedder
from .line_features import LineFeatureConfig, compute_line_lexicon_features
from .weights import WeightConfig, compute_line_weights
from .aggregate import AggregateConfig, aggregate_song_components, concat_song_vector

PathLike = Union[str, os.PathLike, Path]

def _vector_df(song_ids: Sequence[str], Z: np.ndarray, *, prefix: str = "z_") -> pd.DataFrame:
    cols = [f"{prefix}{i}" for i in range(Z.shape[1])]
    out = pd.DataFrame(Z, columns=cols)
    out.insert(0, "song_id", list(song_ids))
    return out

def _iter_batches(n: int, batch_size: int):
    for i in range(0, n, batch_size):
        yield i, min(i + batch_size, n)

@dataclass(frozen=True)
class BuilderConfig:
    text_col: str = "lyrics_dedup"
    fallback_text_col: str = "lyrics_clean"
    song_batch_size: int = 512
    save_line_features: bool = False

class EmotionContextBuilder:
    def __init__(self, cfg: Mapping[str, Any], *, paths: ProjectPaths = PATHS, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.paths = paths
        self.logger = logger or logging.getLogger("lyrics_reco.emotion_context")

        e_cfg = EmbedderConfig(
            model_name=cfg_get(cfg, ["embedder", "model_name"], "sentence-transformers/all-MiniLM-L6-v2"),
            device=cfg_get(cfg, ["embedder", "device"], "auto"),
            batch_size=int(cfg_get(cfg, ["embedder", "batch_size"], 64)),
            normalize_embeddings=bool(cfg_get(cfg, ["embedder", "normalize_embeddings"], True)),
            max_length=int(cfg_get(cfg, ["embedder", "max_length"], 256)),
        )
        self.embedder = SentenceTransformerEmbedder(e_cfg)

        self.lex_bundle = load_lexicons_from_cfg(cfg)

        self.emotions = [e.lower() for e in cfg_get(cfg, ["emotion", "emotions"], ["anger","fear","joy","sadness","disgust","trust"])]

        self.split_cfg = SplitConfig(
            line_split=cfg_get(cfg, ["text", "line_split"], "newline"),
            strip_brackets=bool(cfg_get(cfg, ["text", "strip_brackets"], True)),
            min_line_chars=int(cfg_get(cfg, ["text", "min_line_chars"], 3)),
            max_lines_per_song=int(cfg_get(cfg, ["text", "max_lines_per_song"], 250)),
            dedup_lines=bool(cfg_get(cfg, ["text", "dedup_for_embedding"], True)),
        )

        self.line_feat_cfg = LineFeatureConfig(
            emotions=self.emotions,
            intensity_enabled=bool(cfg_get(cfg, ["intensity", "enabled"], True)),
            vad_enabled=bool(cfg_get(cfg, ["vad", "enabled"], True)),
            intensity_aggregation=cfg_get(cfg, ["intensity", "aggregation"], "mean"),
            vad_aggregation=cfg_get(cfg, ["vad", "aggregation"], "mean"),
        )

        self.weight_cfg = WeightConfig(
            alpha_emotion=float(cfg_get(cfg, ["line_weighting", "alpha_emotion"], 1.0)),
            beta_intensity=float(cfg_get(cfg, ["line_weighting", "beta_intensity"], 1.0)),
            gamma_arousal=float(cfg_get(cfg, ["line_weighting", "gamma_arousal"], 1.0)),
            normalize=cfg_get(cfg, ["line_weighting", "normalize"], "softmax"),
            softmax_temperature=float(cfg_get(cfg, ["line_weighting", "softmax_temperature"], 1.0)),
        )

        self.agg_cfg = AggregateConfig(method=cfg_get(cfg, ["aggregation", "method"], "weighted_mean"))

        self.builder_cfg = BuilderConfig(
            text_col=cfg_get(cfg, ["text", "text_col"], "lyrics_dedup"),
            fallback_text_col=cfg_get(cfg, ["text", "fallback_text_col"], "lyrics_clean"),
            song_batch_size=int(cfg_get(cfg, ["runtime", "song_batch_size"], 512)),
            save_line_features=bool(cfg_get(cfg, ["outputs", "save_line_features"], False)),
        )

    def build_from_df(
        self,
        df: pd.DataFrame,
        *,
        out_csv: Optional[PathLike] = None,
        line_feat_csv: Optional[PathLike] = None,
    ) -> pd.DataFrame:
        song_ids = df["song_id"].astype(str).tolist()
        text_col = self.builder_cfg.text_col if self.builder_cfg.text_col in df.columns else self.builder_cfg.fallback_text_col
        lyrics = df[text_col].astype(str).tolist()

        n = len(song_ids)
        self.logger.info("EmotionContext: songs=%d, text_col=%s, batch=%d", n, text_col, self.builder_cfg.song_batch_size)

        vec_rows: List[pd.DataFrame] = []
        line_rows: List[pd.DataFrame] = []

        for s, e in _iter_batches(n, self.builder_cfg.song_batch_size):
            batch_song_ids = song_ids[s:e]
            batch_lyrics = lyrics[s:e]

            lines, song_index, line_index = explode_songs_to_lines(batch_song_ids, batch_lyrics, self.split_cfg)
            if len(lines) == 0:
                continue

            E = self.embedder.encode(lines)
            if E.ndim != 2 or E.shape[0] != len(lines):
                raise RuntimeError("Embedder returned unexpected shape")

            lf = compute_line_lexicon_features(lines, self.lex_bundle, self.line_feat_cfg)
            lf.insert(0, "song_index", song_index)
            lf.insert(1, "line_index", line_index)

            w = compute_line_weights(
                lf,
                song_index=song_index,
                emotions=self.emotions,
                cfg=self.weight_cfg,
                use_intensity=self.line_feat_cfg.intensity_enabled,
                use_vad=self.line_feat_cfg.vad_enabled,
            )

            comps = aggregate_song_components(
                E,
                lf,
                song_index=song_index,
                weights=w,
                emotions=self.emotions,
                include_intensity=self.line_feat_cfg.intensity_enabled,
                include_vad=self.line_feat_cfg.vad_enabled,
                agg_cfg=self.agg_cfg,
            )
            Z = concat_song_vector(comps)

            uniq = np.unique(np.asarray(song_index, dtype=int))
            batch_out_ids = [batch_song_ids[i] for i in uniq.tolist()]
            vec_df = _vector_df(batch_out_ids, Z, prefix="z_")
            vec_rows.append(vec_df)

            if self.builder_cfg.save_line_features:
                lf2 = lf.copy()
                lf2.insert(0, "song_id", [batch_song_ids[i] for i in lf2["song_index"].astype(int).tolist()])
                line_rows.append(lf2)

            if (s // self.builder_cfg.song_batch_size) % 10 == 0:
                self.logger.info("  processed songs=%d/%d | lines=%d", e, n, len(lines))

        out_df = pd.concat(vec_rows, ignore_index=True) if vec_rows else pd.DataFrame()

        if out_csv is not None:
            out_path = Path(out_csv)
            if not out_path.is_absolute():
                out_path = (self.paths.root / out_path).resolve()
            save_csv(out_df, out_path, index=False, atomic=True)
            self.logger.info("Saved song vectors: %s", out_path)

        if self.builder_cfg.save_line_features and line_feat_csv is not None and line_rows:
            lf_df = pd.concat(line_rows, ignore_index=True)
            lf_path = Path(line_feat_csv)
            if not lf_path.is_absolute():
                lf_path = (self.paths.root / lf_path).resolve()
            save_csv(lf_df, lf_path, index=False, atomic=True)
            self.logger.info("Saved line features: %s", lf_path)

        return out_df

def build_song_vectors_from_df(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    out_csv: Optional[PathLike] = None,
    line_feat_csv: Optional[PathLike] = None,
    paths: ProjectPaths = PATHS,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    builder = EmotionContextBuilder(cfg, paths=paths, logger=logger)
    return builder.build_from_df(df, out_csv=out_csv, line_feat_csv=line_feat_csv)
