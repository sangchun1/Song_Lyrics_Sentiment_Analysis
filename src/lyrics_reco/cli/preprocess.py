"""CLI wrapper for the preprocessing pipeline.

Examples
--------
python -m lyrics_reco.cli preprocess \
    --input data/raw/lyrics.csv \
    --output data/processed/genius_processed.csv
"""

from __future__ import annotations

from typing import Sequence

from ._compat import run_with_argv
from ..pipeline.run_preprocess import main as _pipeline_main


def main(argv: Sequence[str] | None = None) -> None:
    run_with_argv(_pipeline_main, argv, prog="lyrics_reco.cli preprocess")


if __name__ == "__main__":
    main()
