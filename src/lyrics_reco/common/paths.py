"""
lyrics_reco.common.paths

Central project paths used across the repository.

Highlights
----------
- Resolves repo root robustly.
- Creates directories lazily via helpers.
- Keeps all tabular outputs CSV-first.
- Adds a central vector store under ``artifacts/vectors`` so demo / quickstart
  code does not need to guess a run directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# -----------------------------------------------------------------------------
# Root resolution
# -----------------------------------------------------------------------------


def _walk_up_find_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(20):
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists() or (cur / "configs").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # fallback: src/lyrics_reco/common/paths.py -> repo root is 3 parents up
    return start.resolve().parents[3]



def get_project_root() -> Path:
    env = os.getenv("LYRICS_RECO_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _walk_up_find_root(Path(__file__).parent)


# -----------------------------------------------------------------------------
# Directory creation helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def ensure_parent_dir(file_path: Path) -> Path:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path



def with_suffix_csv(path: Path) -> Path:
    return path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")


# -----------------------------------------------------------------------------
# Structured path bundle
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectPaths:
    # root
    root: Path

    # top-level directories
    assets: Path
    configs: Path
    data: Path
    artifacts: Path
    reports: Path
    src: Path

    # assets/*
    assets_lid: Path
    assets_lexicons: Path
    assets_preprocess: Path

    # data/*
    data_raw: Path
    data_interim: Path
    data_processed: Path

    # artifacts/*
    art_vectorizers: Path
    art_embeddings: Path
    art_indexes: Path
    art_runs: Path
    art_vectors: Path
    art_demo: Path

    # reports/*
    rep_runs: Path
    rep_tables: Path

    def make_run_dir(self, run_id: str) -> Path:
        return ensure_dir(self.art_runs / run_id)

    def make_report_run_dir(self, run_id: str) -> Path:
        return ensure_dir(self.rep_runs / run_id)

    def metrics_csv(self, run_id: str, name: str = "metrics") -> Path:
        return with_suffix_csv(self.art_runs / run_id / name)

    def table_csv(self, name: str, subdir: Optional[str] = None) -> Path:
        base = self.rep_tables if subdir is None else (self.rep_tables / subdir)
        return with_suffix_csv(base / name)

    def baseline_vectors_csv(self) -> Path:
        return self.art_vectors / "baseline_vectors.csv"

    def proposed_vectors_csv(self) -> Path:
        return self.art_vectors / "proposed_vectors.csv"

    def demo_dir(self) -> Path:
        return ensure_dir(self.art_demo)



def build_paths(root: Optional[Path] = None, create: bool = False) -> ProjectPaths:
    r = (root or get_project_root()).resolve()

    # top-level
    assets = r / "assets"
    configs = r / "configs"
    data = r / "data"
    artifacts = r / "artifacts"
    reports = r / "reports"
    src = r / "src"

    # assets/*
    assets_lid = assets / "lid"
    assets_lexicons = assets / "lexicons"
    assets_preprocess = assets / "preprocess"

    # data/*
    data_raw = data / "raw"
    data_interim = data / "interim"
    data_processed = data / "processed"

    # artifacts/*
    art_vectorizers = artifacts / "vectorizers"
    art_embeddings = artifacts / "embeddings"
    art_indexes = artifacts / "indexes"
    art_runs = artifacts / "runs"
    art_vectors = artifacts / "vectors"
    art_demo = artifacts / "demo"

    # reports/*
    rep_runs = reports / "runs"
    rep_tables = reports / "tables"

    p = ProjectPaths(
        root=r,
        assets=assets,
        configs=configs,
        data=data,
        artifacts=artifacts,
        reports=reports,
        src=src,
        assets_lid=assets_lid,
        assets_lexicons=assets_lexicons,
        assets_preprocess=assets_preprocess,
        data_raw=data_raw,
        data_interim=data_interim,
        data_processed=data_processed,
        art_vectorizers=art_vectorizers,
        art_embeddings=art_embeddings,
        art_indexes=art_indexes,
        art_runs=art_runs,
        art_vectors=art_vectors,
        art_demo=art_demo,
        rep_runs=rep_runs,
        rep_tables=rep_tables,
    )

    if create:
        for d in [
            p.assets,
            p.configs,
            p.data,
            p.artifacts,
            p.reports,
            p.assets_lid,
            p.assets_lexicons,
            p.assets_preprocess,
            p.data_raw,
            p.data_interim,
            p.data_processed,
            p.art_vectorizers,
            p.art_embeddings,
            p.art_indexes,
            p.art_runs,
            p.art_vectors,
            p.art_demo,
            p.rep_runs,
            p.rep_tables,
        ]:
            ensure_dir(d)

    return p


PATHS: ProjectPaths = build_paths(create=False)
