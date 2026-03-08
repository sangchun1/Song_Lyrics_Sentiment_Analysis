from __future__ import annotations

"""
Export central vector CSVs for the demo.

This script is useful when you already finished the pipeline runs but did not save
song-level vectors in a stable location yet.

Outputs
-------
- artifacts/vectors/baseline_vectors.csv
- artifacts/vectors/proposed_vectors.csv

Examples
--------
python -m lyrics_reco.cli.export_vectors --data data/processed/genius_processed.csv
python -m lyrics_reco.cli.export_vectors --baseline-only
python -m lyrics_reco.cli.export_vectors --proposed-only
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..baseline.emotion_features import build_lexicon_feature_table
from ..common.config import load_yaml
from ..common.paths import PATHS
from ..common.vector_store import save_central_vectors
from ..emotion_context.builder import build_song_vectors_from_df
from ..lexicon.load import load_lexicons_from_cfg
from ..pipeline.utils import cfg_get



def _resolve_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (PATHS.root / p).resolve()



def _deduplicate_by_song_id(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if "song_id" not in df.columns:
        raise ValueError("Processed CSV must contain 'song_id'")
    before = len(df)
    out = df.drop_duplicates(subset=["song_id"], keep="first").reset_index(drop=True)
    after = len(out)
    if after != before:
        logger.warning("Deduplicated processed data by song_id: %d -> %d", before, after)
    return out



def _build_baseline_vectors(
    meta_df: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    include_intensity: bool,
    include_vad: bool,
) -> pd.DataFrame:
    bundle = load_lexicons_from_cfg(cfg)
    emotions = cfg_get(cfg, ["emotion", "emotions"], None)
    feats = build_lexicon_feature_table(
        meta_df,
        bundle,
        song_id_col="song_id",
        text_col="lyrics_clean",
        emotions=emotions,
        include_intensity=include_intensity,
        include_vad=include_vad,
        intensity_aggregation=cfg_get(cfg, ["intensity", "aggregation"], "mean"),
        vad_aggregation=cfg_get(cfg, ["vad", "aggregation"], "mean"),
    )
    return feats



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export central vector CSVs for demo.py")
    ap.add_argument("--data", default="data/processed/genius_processed.csv", help="Processed CSV path")
    ap.add_argument("--emotion-config", default="configs/emotion_context.yaml", help="Emotion config YAML")

    ap.add_argument("--baseline-out", default="", help="Override output path for baseline vector CSV")
    ap.add_argument("--proposed-out", default="", help="Override output path for proposed vector CSV")

    ap.add_argument("--baseline-only", action="store_true", default=False, help="Export only baseline vectors")
    ap.add_argument("--proposed-only", action="store_true", default=False, help="Export only proposed vectors")

    ap.add_argument("--include-intensity", action="store_true", default=False, help="Include baseline intensity features")
    ap.add_argument("--include-vad", action="store_true", default=False, help="Include baseline VAD features")
    return ap.parse_args()



def main() -> None:
    args = parse_args()
    if args.baseline_only and args.proposed_only:
        raise ValueError("Use only one of --baseline-only or --proposed-only")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("lyrics_reco.export_vectors")

    data_path = _resolve_path(args.data)
    emotion_cfg_path = _resolve_path(args.emotion_config)

    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not emotion_cfg_path.exists():
        raise FileNotFoundError(emotion_cfg_path)

    cfg = load_yaml(emotion_cfg_path)
    meta_df = pd.read_csv(data_path)
    logger.info("Loaded processed data: %s | rows=%d cols=%d", data_path, len(meta_df), len(meta_df.columns))
    meta_df = _deduplicate_by_song_id(meta_df, logger)

    do_baseline = not args.proposed_only
    do_proposed = not args.baseline_only

    if do_baseline:
        logger.info("Building baseline vectors...")
        baseline_df = _build_baseline_vectors(
            meta_df,
            cfg,
            include_intensity=bool(args.include_intensity),
            include_vad=bool(args.include_vad),
        )
        baseline_path = save_central_vectors(
            baseline_df,
            "baseline",
            out_path=(args.baseline_out or None),
            paths=PATHS,
        )
        logger.info("Saved baseline vectors: %s", baseline_path)

    if do_proposed:
        logger.info("Building proposed vectors...")
        proposed_df = build_song_vectors_from_df(meta_df, cfg, out_csv=None, paths=PATHS, logger=logger)
        proposed_path = save_central_vectors(
            proposed_df,
            "proposed",
            out_path=(args.proposed_out or None),
            paths=PATHS,
        )
        logger.info("Saved proposed vectors: %s", proposed_path)

    print("\nDone.")
    if do_baseline:
        print(f"Baseline: {baseline_path}")
    if do_proposed:
        print(f"Proposed: {proposed_path}")
    print("Now run: python -m lyrics_reco.cli.demo --title \"Hello\" --artist \"Adele\" --k 10")


if __name__ == "__main__":
    main()
