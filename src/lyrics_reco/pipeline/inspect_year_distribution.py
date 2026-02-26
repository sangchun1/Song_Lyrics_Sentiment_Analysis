"""
lyrics_reco.pipeline.inspect_year_distribution

Select Top-N rows by views from Genius Song Lyrics.csv (chunked),
compare year distribution between:
(A) 전체 데이터
(B) 조회수 기준 Top-N

Saves CSVs under reports/tables/preprocess and an optional plot PNG.

Example:
    python -m lyrics_reco.pipeline.inspect_year_distribution \
      --input "data/raw/Genius Song Lyrics.csv" \
      --top-n 500000 --chunksize 200000 --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..common.io import save_csv
from ..common.paths import PATHS, ensure_dir


def _coerce_year(s: pd.Series) -> pd.Series:
    y = pd.to_numeric(s, errors="coerce")
    y = y.where((y >= 1900) & (y <= 2100))
    return y


def _coerce_views(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").fillna(0)
    v = v.where(v >= 0, 0)
    return v.astype(np.int64)


def _accumulate_counts(acc: Dict[int, int], years: pd.Series) -> None:
    vc = years.dropna().astype(int).value_counts()
    for k, v in vc.items():
        acc[int(k)] = acc.get(int(k), 0) + int(v)


def compute_year_counts_top_views(
    input_csv: Path,
    *,
    year_col: str = "year",
    views_col: str = "views",
    top_n: int = 500_000,
    chunksize: int = 200_000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_counts: Dict[int, int] = {}
    top_keep: Optional[pd.DataFrame] = None

    usecols = [year_col, views_col]
    for chunk in pd.read_csv(input_csv, usecols=usecols, chunksize=chunksize):
        chunk = chunk.copy()
        chunk[year_col] = _coerce_year(chunk[year_col])
        chunk[views_col] = _coerce_views(chunk[views_col])

        _accumulate_counts(overall_counts, chunk[year_col])

        cand = chunk.dropna(subset=[year_col]).copy()
        if cand.empty:
            continue

        cand = cand[[year_col, views_col]]

        if top_keep is None:
            top_keep = cand.nlargest(top_n, views_col)
        else:
            top_keep = pd.concat([top_keep, cand], ignore_index=True)
            top_keep = top_keep.nlargest(top_n, views_col)

    if top_keep is None:
        top_keep = pd.DataFrame(columns=[year_col, views_col])

    overall_df = (
        pd.DataFrame({"year": list(overall_counts.keys()), "count": list(overall_counts.values())})
        .sort_values("year")
        .reset_index(drop=True)
    )

    top_counts = top_keep[year_col].astype(int).value_counts().sort_index()
    top_df = top_counts.rename_axis("year").reset_index(name="count")

    years = sorted(set(overall_df["year"].tolist()) | set(top_df["year"].tolist()))
    overall_map = dict(zip(overall_df["year"], overall_df["count"]))
    top_map = dict(zip(top_df["year"], top_df["count"]))

    overall_total = sum(overall_map.values()) if overall_map else 0
    top_total = sum(top_map.values()) if top_map else 0

    rows = []
    for y in years:
        oc = int(overall_map.get(y, 0))
        tc = int(top_map.get(y, 0))
        op = (oc / overall_total) if overall_total else 0.0
        tp = (tc / top_total) if top_total else 0.0
        rows.append({"year": int(y), "overall_count": oc, "overall_pct": op, "top_count": tc, "top_pct": tp, "pct_diff": tp - op})

    compare_df = pd.DataFrame(rows)
    return overall_df, top_df, compare_df


def add_decade_summary(compare_df: pd.DataFrame) -> pd.DataFrame:
    df = compare_df.copy()
    df["decade"] = (df["year"] // 10) * 10
    decade = (
        df.groupby("decade", as_index=False)
          .agg(overall_count=("overall_count", "sum"),
               overall_pct=("overall_pct", "sum"),
               top_count=("top_count", "sum"),
               top_pct=("top_pct", "sum"))
    )
    decade["pct_diff"] = decade["top_pct"] - decade["overall_pct"]
    return decade


def maybe_plot(compare_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    df = compare_df.sort_values("year")
    plt.figure(figsize=(12, 4))
    plt.plot(df["year"], df["overall_pct"], label="overall_pct")
    plt.plot(df["year"], df["top_pct"], label="top_pct")
    plt.xlabel("year")
    plt.ylabel("proportion")
    plt.title("Year distribution: overall vs Top-N by views")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--year-col", default="year")
    ap.add_argument("--views-col", default="views")
    ap.add_argument("--top-n", type=int, default=500_000)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--out-dir", default="reports/tables/preprocess")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.is_absolute():
        inp = (PATHS.root / inp).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PATHS.root / out_dir).resolve()
    ensure_dir(out_dir)

    overall_df, top_df, compare_df = compute_year_counts_top_views(
        inp,
        year_col=args.year_col,
        views_col=args.views_col,
        top_n=args.top_n,
        chunksize=args.chunksize,
    )
    decade_df = add_decade_summary(compare_df)

    save_csv(overall_df, out_dir / "year_dist_overall.csv", index=False)
    save_csv(top_df, out_dir / f"year_dist_top{args.top_n}.csv", index=False)
    save_csv(compare_df, out_dir / f"year_dist_compare_top{args.top_n}.csv", index=False)
    save_csv(decade_df, out_dir / f"decade_dist_compare_top{args.top_n}.csv", index=False)

    if args.plot:
        maybe_plot(compare_df, out_dir / f"year_dist_compare_top{args.top_n}.png")


if __name__ == "__main__":
    main()
