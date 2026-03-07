"""Top-level command dispatcher for `python -m lyrics_reco.cli`.

Supported subcommands
---------------------
- preprocess
- baseline
- proposed
- index
- demo
- help

Examples
--------
python -m lyrics_reco.cli preprocess --input data/raw/lyrics.csv
python -m lyrics_reco.cli baseline --data data/processed/genius_processed.csv
python -m lyrics_reco.cli proposed --data data/processed/genius_processed.csv --rebuild-index
python -m lyrics_reco.cli index --data data/processed/genius_processed.csv --rebuild-index
python -m lyrics_reco.cli demo --title "Hello" --artist "Adele" --k 10
"""

from __future__ import annotations

import sys
from textwrap import dedent
from typing import Callable, Dict, Sequence

from . import baseline, demo, index, preprocess, proposed
from ._compat import run_with_argv


_COMMANDS: Dict[str, Callable[[Sequence[str] | None], None]] = {
    "preprocess": preprocess.main,
    "baseline": baseline.main,
    "proposed": proposed.main,
    "index": index.main,
    "demo": lambda argv=None: run_with_argv(demo.main, argv, prog="lyrics_reco.cli demo"),
}


def _print_help() -> None:
    msg = dedent(
        """
        Usage:
          python -m lyrics_reco.cli <command> [args ...]

        Commands:
          preprocess   Run preprocessing pipeline
          baseline     Run baseline retrieval + evaluation
          proposed     Run proposed retrieval + evaluation
          index        Build/load vectors and upsert them to VectorDB
          demo         Compare baseline vs proposed recommendations for one query song

        Examples:
          python -m lyrics_reco.cli preprocess --input data/raw/lyrics.csv
          python -m lyrics_reco.cli baseline --data data/processed/genius_processed.csv
          python -m lyrics_reco.cli proposed --data data/processed/genius_processed.csv --rebuild-index
          python -m lyrics_reco.cli index --data data/processed/genius_processed.csv --rebuild-index
          python -m lyrics_reco.cli demo --title "Hello" --artist "Adele" --k 10
        """
    ).strip()
    print(msg)


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    command = argv[0].strip().lower()
    rest = argv[1:]

    if command not in _COMMANDS:
        print(f"Unknown command: {command}\n")
        _print_help()
        raise SystemExit(2)

    _COMMANDS[command](rest)


if __name__ == "__main__":
    main()
