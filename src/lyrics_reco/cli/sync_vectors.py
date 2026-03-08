from __future__ import annotations

import argparse
from pathlib import Path

from lyrics_reco.common.paths import PATHS
from lyrics_reco.common.vector_store import copy_vector_csv, latest_run_vector_path



def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PATHS.root / p).resolve()



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Copy existing run-local vector CSVs into artifacts/vectors/ for demo usage"
    )
    ap.add_argument("--baseline-src", default="", help="Optional baseline source CSV")
    ap.add_argument("--proposed-src", default="", help="Optional proposed source CSV")
    ap.add_argument(
        "--model",
        choices=["baseline", "proposed", "both"],
        default="both",
        help="Which vectors to sync",
    )
    return ap.parse_args()



def _sync_one(kind: str, explicit_src: str) -> Path:
    if explicit_src:
        src = _resolve(explicit_src)
    else:
        src = latest_run_vector_path(kind, paths=PATHS)
        if src is None:
            raise FileNotFoundError(f"Could not find latest {kind} run vector CSV under artifacts/runs/")
    return copy_vector_csv(src, kind, paths=PATHS)



def main() -> None:
    args = parse_args()

    if args.model in ("baseline", "both"):
        out = _sync_one("baseline", args.baseline_src)
        print(f"Synced baseline vectors -> {out}")

    if args.model in ("proposed", "both"):
        out = _sync_one("proposed", args.proposed_src)
        print(f"Synced proposed vectors -> {out}")


if __name__ == "__main__":
    main()
