"""Command-line entry points for the lyrics_reco package.

This package provides a thin execution layer on top of the pipeline modules.
The goal is to expose stable, user-facing commands such as:

    python -m lyrics_reco.cli preprocess ...
    python -m lyrics_reco.cli baseline ...
    python -m lyrics_reco.cli proposed ...
    python -m lyrics_reco.cli index ...
    python -m lyrics_reco.cli demo ...

The `demo` command compares baseline vs proposed recommendations for a
single query song using saved vector artifacts.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
