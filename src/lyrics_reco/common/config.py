"""
lyrics_reco.common.config

Experiment configuration utilities.

Key features:
- Load one or more YAML config files from ./configs
- Apply (recursive) overrides safely
- Generate a reproducible run_id (timestamp + short hash of config)
- Dump config for reproducibility (CSV-first; also writes JSON for convenience)

Why CSV-first?
- Your project standard is "no pickle; artifacts mostly as CSV".
- Config isn't naturally tabular, so we dump a flattened 2-col CSV:
    key,value
    model.name,sentence-transformers/all-MiniLM-L6-v2
    retrieval.top_k,10
  and also save JSON (optional but very practical).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import yaml

from .io import save_csv, save_json, save_text
from .paths import PATHS, ProjectPaths, ensure_dir


PathLike = Union[str, Path]


# -----------------------------
# YAML loading
# -----------------------------
def load_yaml(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {p}")
    return data


def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge updates into base (mutates base and also returns it).
    - If both base[k] and updates[k] are dicts -> recurse
    - Else -> overwrite
    """
    for k, v in updates.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, Mapping):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


# -----------------------------
# Flatten to CSV
# -----------------------------
def _flatten_items(obj: Any, prefix: str = "", sep: str = ".") -> List[Tuple[str, str]]:
    """
    Flatten nested dict/list into key-path pairs for CSV dump.

    - dict: key paths joined by sep
    - list/tuple: index-based keys like "layers[0]"
    """
    items: List[Tuple[str, str]] = []

    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            items.extend(_flatten_items(v, key, sep))
        return items

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            items.extend(_flatten_items(v, key, sep))
        return items

    items.append((prefix, "" if obj is None else str(obj)))
    return items


def config_to_flat_csv_rows(config: Mapping[str, Any]) -> List[Dict[str, str]]:
    rows = [{"key": k, "value": v} for k, v in _flatten_items(config)]
    rows.sort(key=lambda r: r["key"])  # stable ordering
    return rows


# -----------------------------
# run_id
# -----------------------------
def _now_kst() -> datetime:
    """Return current time in Asia/Seoul if zoneinfo is available; else local time."""
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        return datetime.now(ZoneInfo("Asia/Seoul"))
    except Exception:
        return datetime.now()


def make_run_id(
    config: Mapping[str, Any],
    *,
    prefix: str = "run",
    ts_format: str = "%Y%m%d_%H%M%S",
    hash_len: int = 8,
) -> str:
    """
    Create a run_id like:
        run_20260225_134501_a1b2c3d4
    where hash is based on canonical JSON of config.
    """
    ts = _now_kst().strftime(ts_format)
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:hash_len]
    return f"{prefix}_{ts}_{h}"


# -----------------------------
# Build config from files
# -----------------------------
def resolve_config(
    config_files: Sequence[PathLike],
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge config YAMLs in order (later wins), then apply overrides.

    Example:
        cfg = resolve_config(
            ["configs/data.yaml", "configs/baseline.yaml"],
            overrides={"retrieval": {"top_k": 20}}
        )
    """
    merged: Dict[str, Any] = {}
    for cf in config_files:
        deep_update(merged, load_yaml(cf))
    if overrides:
        deep_update(merged, dict(overrides))
    return merged


def load_from_configs_dir(
    names: Sequence[str],
    *,
    paths: ProjectPaths = PATHS,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience loader that assumes YAMLs live in <root>/configs.

    names: ["data", "baseline"] -> loads configs/data.yaml then configs/baseline.yaml
    """
    files = [paths.configs / f"{n}.yaml" for n in names]
    return resolve_config(files, overrides=overrides)


# -----------------------------
# Dump config artifacts
# -----------------------------
@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    artifacts_dir: Path
    reports_dir: Path
    config_csv: Path
    config_json: Path
    note_txt: Path


def dump_run_config(
    config: Mapping[str, Any],
    *,
    run_id: Optional[str] = None,
    paths: ProjectPaths = PATHS,
    prefix: str = "run",
    write_json: bool = True,
    note: Optional[str] = None,
) -> RunArtifacts:
    """
    Create run dirs and dump config as:
    - artifacts/runs/<run_id>/config.csv  (flattened; CSV-first)
    - artifacts/runs/<run_id>/config.json (optional)
    - artifacts/runs/<run_id>/note.txt    (optional)

    Also creates:
    - reports/runs/<run_id>/  (for human-facing run notes)

    Returns RunArtifacts with useful paths.
    """
    rid = run_id or make_run_id(config, prefix=prefix)

    art_dir = ensure_dir(paths.art_runs / rid)
    rep_dir = ensure_dir(paths.rep_runs / rid)

    # CSV dump (flattened)
    rows = config_to_flat_csv_rows(config)
    cfg_csv = art_dir / "config.csv"
    save_csv(pd.DataFrame(rows), cfg_csv, index=False, atomic=True)

    # JSON dump (handy for programmatic reload)
    cfg_json = art_dir / "config.json"
    if write_json:
        save_json(dict(config), cfg_json, atomic=True)

    # note
    note_path = art_dir / "note.txt"
    save_text(note or "", note_path, atomic=True)
    save_text(note or "", rep_dir / "note.txt", atomic=True)

    return RunArtifacts(
        run_id=rid,
        artifacts_dir=art_dir,
        reports_dir=rep_dir,
        config_csv=cfg_csv,
        config_json=cfg_json,
        note_txt=note_path,
    )
