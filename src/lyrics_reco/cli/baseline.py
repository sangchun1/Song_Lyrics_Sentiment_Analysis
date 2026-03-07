"""CLI wrapper for the baseline retrieval pipeline.

Examples
--------
python -m lyrics_reco.cli baseline \
    --data data/processed/genius_processed.csv \
    --eval-config configs/eval.yaml \
    --retrieval-config configs/retrieval.yaml \
    --emotion-config configs/emotion_context.yaml
"""

from __future__ import annotations

from typing import Sequence

from ._compat import run_with_argv
from ..pipeline.run_baseline import main as _pipeline_main


def main(argv: Sequence[str] | None = None) -> None:
    run_with_argv(_pipeline_main, argv, prog="lyrics_reco.cli baseline")


if __name__ == "__main__":
    main()
