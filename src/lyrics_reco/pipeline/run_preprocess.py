"""
lyrics_reco.pipeline.run_preprocess

CLI to run preprocessing for Genius Song Lyrics.csv.
"""

from __future__ import annotations

import argparse

from ..common.config import dump_run_config
from ..common.logging import setup_run_logger
from ..common.seed import set_seed
from ..preprocess.pipeline import PreprocessConfig, run_preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/processed/genius_processed.csv")

    ap.add_argument("--start-year", type=int, default=1950)
    ap.add_argument("--end-year", type=int, default=2022)

    ap.add_argument("--process-genius-translations", dest="process_genius_translations", action="store_true")
    ap.add_argument("--no-process-genius-translations", dest="process_genius_translations", action="store_false")
    ap.set_defaults(process_genius_translations=True)

    ap.add_argument("--expand-multi-artist", action="store_true", default=False)

    ap.add_argument("--use-fasttext", dest="use_fasttext", action="store_true")
    ap.add_argument("--no-fasttext", dest="use_fasttext", action="store_false")
    ap.set_defaults(use_fasttext=True)

    ap.add_argument("--fasttext-threshold", type=float, default=0.5)
    ap.add_argument("--fasttext-model", default="assets/lid/lid.176.bin")

    ap.add_argument("--top-global", type=int, default=500_000)
    ap.add_argument("--recent-year-start", type=int, default=2020)
    ap.add_argument("--recent-year-end", type=int, default=2022)
    ap.add_argument("--recent-min-per-year", type=int, default=20_000)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunksize", type=int, default=200_000, help="0 disables streaming")

    args = ap.parse_args()

    cfg = PreprocessConfig(
        input_csv=args.input,
        output_csv=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        process_genius_translations=args.process_genius_translations,
        expand_multi_artist=args.expand_multi_artist,
        use_fasttext=args.use_fasttext,
        fasttext_model_path=args.fasttext_model,
        fasttext_threshold=args.fasttext_threshold,
        top_global=args.top_global,
        recent_year_start=args.recent_year_start,
        recent_year_end=args.recent_year_end,
        recent_min_per_year=args.recent_min_per_year,
    )

    run_cfg = {"pipeline": "run_preprocess", "params": vars(args)}
    art = dump_run_config(run_cfg, prefix="preprocess")
    logger = setup_run_logger(art.run_id, name="lyrics_reco", also_to_reports=True, reset_handlers=True, use_rich_console=False)
    set_seed(args.seed)

    ch = None if args.chunksize <= 0 else int(args.chunksize)
    logger.info("Starting preprocessing...")
    out = run_preprocess(cfg, chunksize=ch)
    logger.info("Preprocess done. Output: %s", out)


if __name__ == "__main__":
    main()