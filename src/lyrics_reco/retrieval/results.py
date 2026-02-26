"""
lyrics_reco.retrieval.results

Build a tidy recommendation table for evaluation and inspection.

Schema (default):
- query_index, query_song_id
- rec_index, rec_song_id
- rank, score
+ optional query_* / rec_* metadata columns if present in meta_df

Why this file?
- Keeps baseline/proposed output consistent.
- Evaluation can consume a single schema.

If you truly want minimal code, you can skip using this and build tables in pipelines.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def build_recommendations_table(
    meta_df: pd.DataFrame,
    query_indices: Sequence[int],
    rec_indices_list: Sequence[np.ndarray],
    rec_scores_list: Sequence[np.ndarray],
    *,
    song_id_col: str = "song_id",
    meta_cols: Tuple[str, ...] = ("title", "artist", "year", "genre"),
) -> pd.DataFrame:
    """
    Convert per-query topK results into a single tidy DataFrame.
    """
    if song_id_col not in meta_df.columns:
        raise ValueError(f"meta_df must contain '{song_id_col}'")

    rows: List[dict] = []

    for qi, rec_idx, rec_sc in zip(query_indices, rec_indices_list, rec_scores_list):
        qi = int(qi)
        q_sid = str(meta_df.iloc[qi][song_id_col])

        for rank, (ri, sc) in enumerate(zip(rec_idx.tolist(), rec_sc.tolist()), start=1):
            ri = int(ri)
            r_sid = str(meta_df.iloc[ri][song_id_col])
            row = {
                "query_index": qi,
                "query_song_id": q_sid,
                "rec_index": ri,
                "rec_song_id": r_sid,
                "rank": int(rank),
                "score": float(sc),
            }

            for c in meta_cols:
                if c in meta_df.columns:
                    row[f"query_{c}"] = meta_df.iloc[qi][c]
                    row[f"rec_{c}"] = meta_df.iloc[ri][c]

            rows.append(row)

    return pd.DataFrame(rows)
