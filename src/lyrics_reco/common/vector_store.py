"""
Helpers for a central vector store under ``artifacts/vectors``.

This module keeps demo / quickstart code independent from a specific run_id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from .io import save_csv
from .paths import PATHS, ProjectPaths, ensure_dir

VectorKind = Literal["baseline", "proposed"]



def ensure_vectors_dir(paths: ProjectPaths = PATHS) -> Path:
    return ensure_dir(paths.art_vectors)



def central_vector_path(kind: VectorKind, *, paths: ProjectPaths = PATHS) -> Path:
    ensure_vectors_dir(paths)
    if kind == "baseline":
        return paths.baseline_vectors_csv()
    if kind == "proposed":
        return paths.proposed_vectors_csv()
    raise ValueError(f"Unsupported vector kind: {kind}")



def latest_run_vector_path(kind: VectorKind, *, paths: ProjectPaths = PATHS) -> Optional[Path]:
    ensure_vectors_dir(paths)
    pattern = {
        "baseline": "*/baseline_lexicon_features.csv",
        "proposed": "*/emotion_context_vectors.csv",
    }[kind]
    candidates = sorted(paths.art_runs.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1].resolve() if candidates else None



def default_vector_path(kind: VectorKind, *, paths: ProjectPaths = PATHS) -> Optional[Path]:
    central = central_vector_path(kind, paths=paths)
    if central.exists():
        return central.resolve()
    return latest_run_vector_path(kind, paths=paths)



def save_central_vectors(
    df: pd.DataFrame,
    kind: VectorKind,
    *,
    out_path: str | Path | None = None,
    paths: ProjectPaths = PATHS,
) -> Path:
    dest = Path(out_path).expanduser().resolve() if out_path else central_vector_path(kind, paths=paths)
    save_csv(df, dest, index=False)
    return dest



def copy_vector_csv(
    src: str | Path,
    kind: VectorKind,
    *,
    out_path: str | Path | None = None,
    paths: ProjectPaths = PATHS,
) -> Path:
    src_path = Path(src).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    df = pd.read_csv(src_path)
    return save_central_vectors(df, kind, out_path=out_path, paths=paths)
