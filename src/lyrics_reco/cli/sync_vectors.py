from __future__ import annotations

import argparse
from pathlib import Path

from lyrics_reco.common.paths import PATHS
from lyrics_reco.common.vector_store import copy_vector_csv, latest_run_vector_path



def _resolve_optional(path_str: str) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (PATHS.root / p).resolve()



def _sync_one(kind: str, explicit_src: str) -> Path:
    src = _resolve_optional(explicit_src)
    if src is None:
        src = latest_run_vector_path(kind, paths=PATHS)
    if src is None:
        raise FileNotFoundError(
            f"Could not find latest {kind} vector CSV under artifacts/runs/. "
            f"If you never saved vectors during the run, use: python -m lyrics_reco.cli.export_vectors"
        )
    return copy_vector_csv(src, kind, paths=PATHS)



def main() -> None:
    ap = argparse.ArgumentParser(description="Copy latest run vector CSVs into artifacts/vectors/")
    ap.add_argument("--baseline-src", default="", help="Optional explicit baseline vector CSV")
    ap.add_argument("--proposed-src", default="", help="Optional explicit proposed vector CSV")
    args = ap.parse_args()

    baseline_out = _sync_one("baseline", args.baseline_src)
    proposed_out = _sync_one("proposed", args.proposed_src)

    print(f"Baseline -> {baseline_out}")
    print(f"Proposed -> {proposed_out}")


if __name__ == "__main__":
    main()
