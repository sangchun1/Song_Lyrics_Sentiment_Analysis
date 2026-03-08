"""
Helpers for a central vector store under ``artifacts/vectors``.

This module keeps demo / quickstart code independent from a specific run_id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .io import save_csv
from .paths import PATHS, ProjectPaths, ensure_dir, ensure_parent_dir

VectorKind = Literal["baseline", "proposed"]



def ensure_vectors_dir(paths: ProjectPaths = PATHS) -> Path:
    return ensure_dir(paths.art_vectors)



def central_vector_path(kind: VectorKind, *, fmt: str = "csv", paths: ProjectPaths = PATHS) -> Path:
    ensure_vectors_dir(paths)
    fmt = fmt.lower().lstrip(".")
    if kind == "baseline":
        if fmt == "csv":
            return paths.baseline_vectors_csv()
        if fmt == "npz":
            return paths.baseline_vectors_npz()
    if kind == "proposed":
        if fmt == "csv":
            return paths.proposed_vectors_csv()
        if fmt == "npz":
            return paths.proposed_vectors_npz()
    raise ValueError(f"Unsupported vector kind/format: {kind}/{fmt}")



def central_song_ids_path(kind: VectorKind, *, paths: ProjectPaths = PATHS) -> Path:
    ensure_vectors_dir(paths)
    if kind == "baseline":
        return paths.baseline_song_ids_npy()
    if kind == "proposed":
        return paths.proposed_song_ids_npy()
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
    for fmt in ("npz", "csv"):
        central = central_vector_path(kind, fmt=fmt, paths=paths)
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
    dest = Path(out_path).expanduser().resolve() if out_path else central_vector_path(kind, fmt="csv", paths=paths)
    save_csv(df, dest, index=False)
    return dest



def save_song_ids(
    song_ids: Sequence[str] | np.ndarray | pd.Series,
    kind: VectorKind,
    *,
    out_path: str | Path | None = None,
    paths: ProjectPaths = PATHS,
) -> Path:
    dest = Path(out_path).expanduser().resolve() if out_path else central_song_ids_path(kind, paths=paths)
    ensure_parent_dir(dest)
    arr = np.asarray(song_ids, dtype=object)
    np.save(dest, arr)
    return dest



def save_dense_vectors_npz(
    X: np.ndarray | sparse.spmatrix,
    kind: VectorKind,
    *,
    out_path: str | Path | None = None,
    paths: ProjectPaths = PATHS,
) -> Path:
    dest = Path(out_path).expanduser().resolve() if out_path else central_vector_path(kind, fmt="npz", paths=paths)
    ensure_parent_dir(dest)

    if sparse.issparse(X):
        sparse.save_npz(dest, X)
        return dest

    arr = np.asarray(X, dtype=np.float32)
    np.savez_compressed(dest, X=arr)
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
