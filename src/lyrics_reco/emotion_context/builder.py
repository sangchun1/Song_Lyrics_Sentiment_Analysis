"""lyrics_reco.emotion_context.builder

Batched builder for emotion-context vectors.
This revision uses a dual-stream design:
- embedding stream: dedup lines
- lexicon stream: original lines for emotion/intensity/VAD + weighting
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..common.io import save_csv
from ..common.paths import PATHS, ProjectPaths
from ..lexicon.load import load_lexicons_from_cfg
from .aggregate import AggregateConfig, aggregate_song_embedding, aggregate_song_emotion_tail, concat_song_vector
from .embedder import EmbedderConfig, SentenceTransformerEmbedder
from .line_features import LineFeatureConfig, compute_line_lexicon_features
from .splitter import SplitConfig, explode_songs_to_line_table
from .utils import cfg_get
from .weights import WeightConfig, compute_line_weights

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
    embedding_text_col: str = "lyrics_dedup"
    lexicon_text_col: str = "lyrics_clean"
    fallback_embedding_text_col: str = "lyrics_clean"
    fallback_lexicon_text_col: str = "lyrics_dedup"
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
        self.emotions = [e.lower() for e in cfg_get(cfg, ["emotion", "emotions"], ["anger", "fear", "joy", "sadness", "disgust", "trust"])]

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
            beta_intensity=float(cfg_get(cfg, ["line_weighting", "beta_intensity"], 0.5)),
            gamma_arousal=float(cfg_get(cfg, ["line_weighting", "gamma_arousal"], 0.25)),
            normalize=cfg_get(cfg, ["line_weighting", "normalize"], "softmax"),
            softmax_temperature=float(cfg_get(cfg, ["line_weighting", "softmax_temperature"], 1.0)),
        )

        self.agg_cfg = AggregateConfig(method=cfg_get(cfg, ["aggregation", "method"], "weighted_mean"))
        self.vector_layout = cfg_get(cfg, ["aggregation", "vector_layout"], "embedding_ratio_vad")
        self.song_feature_weight = cfg_get(cfg, ["aggregation", "song_feature_weight"], "emotion_word_count")

        self.builder_cfg = BuilderConfig(
            embedding_text_col=cfg_get(cfg, ["text", "embedding_text_col"], "lyrics_dedup"),
            lexicon_text_col=cfg_get(cfg, ["text", "lexicon_text_col"], "lyrics_clean"),
            fallback_embedding_text_col=cfg_get(cfg, ["text", "fallback_embedding_text_col"], "lyrics_clean"),
            fallback_lexicon_text_col=cfg_get(cfg, ["text", "fallback_lexicon_text_col"], "lyrics_dedup"),
            song_batch_size=int(cfg_get(cfg, ["runtime", "song_batch_size"], 512)),
            save_line_features=bool(cfg_get(cfg, ["outputs", "save_line_features"], False)),
        )

    @staticmethod
    def _collapse_lex_weights_to_embed(embed_tbl: pd.DataFrame, lex_feat_df: pd.DataFrame, *, weight_col: str = "norm_weight") -> np.ndarray:
        out = np.zeros(len(embed_tbl), dtype=np.float32)
        embed_lookup = {(int(r.song_index), str(r.line_key)): i for i, r in enumerate(embed_tbl.itertuples(index=False))}
        grouped = lex_feat_df.groupby(["song_index", "line_key"], as_index=False)[weight_col].sum()
        for r in grouped.itertuples(index=False):
            key = (int(r.song_index), str(r.line_key))
            if key in embed_lookup:
                out[embed_lookup[key]] = float(getattr(r, weight_col))

        song_idx = embed_tbl["song_index"].to_numpy(dtype=int)
        for g in np.unique(song_idx):
            idx = np.where(song_idx == g)[0]
            s = float(out[idx].sum())
            if s > 0:
                out[idx] /= s
            else:
                out[idx] = 1.0 / max(len(idx), 1)
        return out

    def build_from_df(self, df: pd.DataFrame, *, out_csv: Optional[PathLike] = None, line_feat_csv: Optional[PathLike] = None) -> pd.DataFrame:
        song_ids = df["song_id"].astype(str).tolist()
        embed_col = self.builder_cfg.embedding_text_col if self.builder_cfg.embedding_text_col in df.columns else self.builder_cfg.fallback_embedding_text_col
        lex_col = self.builder_cfg.lexicon_text_col if self.builder_cfg.lexicon_text_col in df.columns else self.builder_cfg.fallback_lexicon_text_col
        embed_lyrics = df[embed_col].astype(str).tolist()
        lex_lyrics = df[lex_col].astype(str).tolist()

        n = len(song_ids)
        self.logger.info("EmotionContext: songs=%d, embed_col=%s, lex_col=%s, batch=%d", n, embed_col, lex_col, self.builder_cfg.song_batch_size)

        vec_rows: List[pd.DataFrame] = []
        line_rows: List[pd.DataFrame] = []

        for s, e in _iter_batches(n, self.builder_cfg.song_batch_size):
            batch_song_ids = song_ids[s:e]
            batch_embed_lyrics = embed_lyrics[s:e]
            batch_lex_lyrics = lex_lyrics[s:e]

            embed_tbl = explode_songs_to_line_table(batch_song_ids, batch_embed_lyrics, self.split_cfg, dedup_override=True)
            lex_tbl = explode_songs_to_line_table(batch_song_ids, batch_lex_lyrics, self.split_cfg, dedup_override=False)

            if len(embed_tbl) == 0 or len(lex_tbl) == 0:
                continue

            E = self.embedder.encode(embed_tbl["line_text"].tolist())
            if E.ndim != 2 or E.shape[0] != len(embed_tbl):
                raise RuntimeError("Embedder returned unexpected shape")

            lf = compute_line_lexicon_features(lex_tbl["line_text"].tolist(), self.lex_bundle, self.line_feat_cfg)
            lf.insert(0, "song_id", lex_tbl["song_id"].tolist())
            lf.insert(1, "song_index", lex_tbl["song_index"].astype(int).tolist())
            lf.insert(2, "line_index", lex_tbl["line_index"].astype(int).tolist())
            lf.insert(3, "line_text", lex_tbl["line_text"].tolist())
            lf.insert(4, "line_key", lex_tbl["line_key"].tolist())

            weight_info = compute_line_weights(
                lf,
                song_index=lf["song_index"].tolist(),
                emotions=self.emotions,
                cfg=self.weight_cfg,
                use_intensity=self.line_feat_cfg.intensity_enabled,
                use_vad=self.line_feat_cfg.vad_enabled,
                return_details=True,
            )
            lf["raw_weight"] = weight_info["raw"]
            lf["norm_weight"] = weight_info["weights"]
            lf["align_score"] = weight_info["align"]
            lf["alpha_tilde"] = weight_info["alpha_tilde"]
            lf["arousal_tilde"] = weight_info["arousal_tilde"]

            embed_weights = self._collapse_lex_weights_to_embed(embed_tbl, lf, weight_col="norm_weight")
            song_emb = aggregate_song_embedding(E, embed_tbl["song_index"].tolist(), embed_weights, agg_cfg=self.agg_cfg)
            song_tail = aggregate_song_emotion_tail(
                lf,
                lf["song_index"].tolist(),
                self.emotions,
                include_vad=bool(cfg_get(self.cfg, ["aggregation", "include_vad"], True)),
                include_intensity=(self.vector_layout == "embedding_ratio_intensity_vad"),
                song_feature_weight=self.song_feature_weight,
            )

            comps = {"embedding": song_emb, "emotion_ratio": song_tail["emotion_ratio"]}
            if "vad" in song_tail:
                comps["vad"] = song_tail["vad"]
            if "intensity" in song_tail:
                comps["intensity"] = song_tail["intensity"]
            Z = concat_song_vector(comps, layout=self.vector_layout)

            uniq = np.unique(np.asarray(embed_tbl["song_index"], dtype=int))
            batch_out_ids = [batch_song_ids[i] for i in uniq.tolist()]
            vec_rows.append(_vector_df(batch_out_ids, Z, prefix="z_"))

            if self.builder_cfg.save_line_features:
                embed_map = set((int(r.song_index), str(r.line_key)) for r in embed_tbl.itertuples(index=False))
                lf2 = lf.copy()
                lf2["mapped_to_embed"] = [int((int(si), str(lk)) in embed_map) for si, lk in zip(lf2["song_index"], lf2["line_key"])]
                line_rows.append(lf2)

            if (s // self.builder_cfg.song_batch_size) % 10 == 0:
                self.logger.info("  processed songs=%d/%d | embed_lines=%d | lex_lines=%d", e, n, len(embed_tbl), len(lex_tbl))

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
