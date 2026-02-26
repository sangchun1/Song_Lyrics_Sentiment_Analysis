"""
lyrics_reco.common.io

CSV-first I/O utilities for the project.

Design goals:
- Prefer CSV for tabular artifacts (no pickle assumptions).
- Keep file writes safe: auto-create parent dirs, optional atomic write.
- Be explicit about encoding defaults (utf-8-sig is convenient on Windows Excel).

Public API (typical):
- load_csv(path, **kwargs) -> pd.DataFrame
- save_csv(df, path, index=False, atomic=True, **kwargs) -> Path
- save_records_csv(records, path, ...) -> Path
- load_json / save_json (for small configs/metrics)
- load_text / save_text (for logs, prompts)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd

from .paths import ensure_parent_dir, with_suffix_csv


PathLike = Union[str, os.PathLike, Path]


# -----------------------------
# CSV (tabular) I/O
# -----------------------------
def load_csv(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a CSV into a DataFrame.

    Notes:
    - Default encoding is utf-8 (read side). If you saved with utf-8-sig and
      read with utf-8, it still works because pandas handles BOM fine.
    """
    p = Path(path)
    return pd.read_csv(p, encoding=encoding, **kwargs)


def save_csv(
    df: pd.DataFrame,
    path: PathLike,
    *,
    index: bool = False,
    encoding: str = "utf-8-sig",
    atomic: bool = True,
    **kwargs: Any,
) -> Path:
    """
    Save a DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
    path : PathLike
        Output path ('.csv' will be enforced).
    index : bool
        Whether to write row index.
    encoding : str
        Default 'utf-8-sig' for Excel-friendly CSV on Windows.
    atomic : bool
        If True, write to a temp file then replace (safer against partial writes).
    kwargs : Any
        Passed to df.to_csv (e.g., sep=',', quoting=..., float_format=...).

    Returns
    -------
    Path
        Final output path.
    """
    out = with_suffix_csv(Path(path))
    ensure_parent_dir(out)

    if not atomic:
        df.to_csv(out, index=index, encoding=encoding, **kwargs)
        return out

    # Atomic write: temp file in same directory for best cross-device reliability
    tmp_dir = out.parent
    fd, tmp_name = tempfile.mkstemp(prefix=out.stem + "_", suffix=".tmp", dir=tmp_dir)
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        df.to_csv(tmp_path, index=index, encoding=encoding, **kwargs)
        tmp_path.replace(out)
    finally:
        # If something went wrong before replace, clean up tmp
        if tmp_path.exists() and tmp_path != out:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return out


def save_records_csv(
    records: Sequence[Mapping[str, Any]],
    path: PathLike,
    *,
    index: bool = False,
    encoding: str = "utf-8-sig",
    atomic: bool = True,
    **kwargs: Any,
) -> Path:
    """
    Save a list of dict records to CSV (common for metrics logs).

    Example:
        save_records_csv([{"k": 10, "recall": 0.12}, ...], "reports/tables/metrics.csv")
    """
    df = pd.DataFrame.from_records(records)
    return save_csv(df, path, index=index, encoding=encoding, atomic=atomic, **kwargs)


# -----------------------------
# JSON (small artifacts)
# -----------------------------
def load_json(path: PathLike, *, encoding: str = "utf-8") -> Any:
    """Load a JSON file (for small configs/metadata)."""
    p = Path(path)
    with p.open("r", encoding=encoding) as f:
        return json.load(f)


def save_json(
    obj: Any,
    path: PathLike,
    *,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
    atomic: bool = True,
) -> Path:
    """Save an object to JSON (for small configs/metadata)."""
    out = Path(path)
    ensure_parent_dir(out)

    if not atomic:
        with out.open("w", encoding=encoding) as f:
            json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)
        return out

    tmp_dir = out.parent
    fd, tmp_name = tempfile.mkstemp(prefix=out.stem + "_", suffix=".tmp", dir=tmp_dir)
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with tmp_path.open("w", encoding=encoding) as f:
            json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)
        tmp_path.replace(out)
    finally:
        if tmp_path.exists() and tmp_path != out:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return out


# -----------------------------
# Text (logs/prompts)
# -----------------------------
def load_text(path: PathLike, *, encoding: str = "utf-8") -> str:
    p = Path(path)
    return p.read_text(encoding=encoding)


def save_text(
    text: str,
    path: PathLike,
    *,
    encoding: str = "utf-8",
    atomic: bool = True,
) -> Path:
    out = Path(path)
    ensure_parent_dir(out)

    if not atomic:
        out.write_text(text, encoding=encoding)
        return out

    tmp_dir = out.parent
    fd, tmp_name = tempfile.mkstemp(prefix=out.stem + "_", suffix=".tmp", dir=tmp_dir)
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        tmp_path.write_text(text, encoding=encoding)
        tmp_path.replace(out)
    finally:
        if tmp_path.exists() and tmp_path != out:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return out
