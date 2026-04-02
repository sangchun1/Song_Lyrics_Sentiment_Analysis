from __future__ import annotations

"""
Demo CLI for comparing baseline vs proposed music recommendations.

What it does
------------
- Takes a query song (song_id or title [+ artist])
- Loads saved baseline vectors and saved proposed vectors
- Recommends top-k songs from each model
- Shows:
    * model-space similarity score
    * emotion-space similarity score
- Saves both tables as CSV under artifacts/demo/

Recommended usage
-----------------
1) As a module (recommended)
   python -m lyrics_reco.cli.demo --title "Hello" --artist "Adele" --k 10

2) Direct file execution from repo root
   python src/lyrics_reco/cli/demo.py --title "Hello" --artist "Adele" --k 10

Supported vector formats
------------------------
- Baseline: .npz or .csv
- Proposed: .csv, .npz, .npy

Recommended artifacts/vectors layout
------------------------------------
- artifacts/vectors/baseline_vectors.npz
- artifacts/vectors/baseline_song_ids.npy (recommended with baseline npz)
- artifacts/vectors/proposed_vectors.npz  (or .npy / .csv)
- artifacts/vectors/proposed_song_ids.npy (needed when proposed npz/npy row order
  is not guaranteed to match the metadata file)
- artifacts/vectors/catalog.csv           (song_id/title/artist/year/genre mapping)
- artifacts/vectors/emotion_profiles.csv  (optional)
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

# Allow direct execution: python src/lyrics_reco/cli/demo.py ...
if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[3]
    SRC_DIR = REPO_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
else:
    REPO_ROOT = Path(__file__).resolve().parents[3]

from lyrics_reco.retrieval.cosine import topk_cosine
from lyrics_reco.retrieval.dedup import filter_query_equivalent_candidates
from lyrics_reco.retrieval.mmr import mmr_rerank


ArrayLike = np.ndarray | sparse.spmatrix


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------


def _resolve_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()



def _latest_existing(patterns: Sequence[str]) -> Optional[Path]:
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(sorted(REPO_ROOT.glob(pat), key=lambda p: p.stat().st_mtime))
    return candidates[-1].resolve() if candidates else None



def _first_existing(paths: Sequence[str]) -> Optional[Path]:
    for p in paths:
        rp = _resolve_path(p)
        if rp.exists():
            return rp
    return None



def _default_baseline_path() -> Optional[Path]:
    direct = _first_existing(
        [
            "artifacts/vectors/baseline_vectors.npz",
            "artifacts/vectors/baseline_tfidf_weighted.npz",
            "artifacts/vectors/baseline_tfidf.npz",
            "artifacts/vectors/baseline_vectors.csv",
            "artifacts/vectorizers/baseline_tfidf_weighted.npz",
            "artifacts/vectorizers/baseline_vectors.npz",
        ]
    )
    if direct is not None:
        return direct
    return _latest_existing([
        "artifacts/runs/*/baseline_tfidf_weighted.npz",
        "artifacts/runs/*/baseline_tfidf.npz",
        "artifacts/runs/*/baseline_vectors.npz",
        "artifacts/runs/*/baseline_lexicon_features.csv",
        "artifacts/runs/*/baseline_vectors.csv",
    ])



def _default_baseline_song_ids_path() -> Optional[Path]:
    direct = _first_existing(
        [
            "artifacts/vectors/baseline_song_ids.npy",
            "artifacts/vectors/baseline_tfidf_song_ids.npy",
        ]
    )
    if direct is not None:
        return direct
    return _latest_existing([
        "artifacts/runs/*/baseline_song_ids.npy",
        "artifacts/runs/*/baseline_tfidf_song_ids.npy",
    ])



def _default_proposed_path() -> Optional[Path]:
    direct = _first_existing(
        [
            "artifacts/vectors/proposed_vectors.csv",
            "artifacts/vectors/proposed_vectors.npz",
            "artifacts/vectors/proposed_vectors.npy",
        ]
    )
    if direct is not None:
        return direct
    return _latest_existing([
        "artifacts/runs/*/emotion_context_vectors.csv",
        "artifacts/runs/*/proposed_vectors.csv",
    ])



def _default_catalog_path() -> Optional[Path]:
    return _first_existing([
        "artifacts/vectors/catalog.csv",
        "data/processed/genius_processed.csv",
    ])



def _default_emotion_profiles_path() -> Optional[Path]:
    return _first_existing(["artifacts/vectors/emotion_profiles.csv"])



def _default_proposed_song_ids_path() -> Optional[Path]:
    return _first_existing(["artifacts/vectors/proposed_song_ids.npy"])


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^0-9a-zA-Z가-힣._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "query"



def _cosine_pair(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    u = np.asarray(u, dtype=np.float32).ravel()
    v = np.asarray(v, dtype=np.float32).ravel()
    den = max(float(np.linalg.norm(u)) * float(np.linalg.norm(v)), eps)
    return float(np.dot(u, v) / den)



def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data



def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")



def _print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 100}")
    print(title)
    print("=" * 100)
    if df.empty:
        print("(no recommendations)")
        return
    view = df.copy()
    for c in ["model_score", "emotion_similarity", "emotion_similarity_pct"]:
        if c in view.columns:
            view[c] = view[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
    cols = [
        c
        for c in [
            "rank",
            "rec_song_id",
            "title",
            "artist",
            "year",
            "genre",
            "model_score",
            "emotion_similarity",
            "emotion_similarity_pct",
        ]
        if c in view.columns
    ]
    print(view[cols].to_string(index=False))


# -----------------------------------------------------------------------------
# Query resolution
# -----------------------------------------------------------------------------


def _normalize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.casefold()



def _ensure_meta_cols(meta_df: pd.DataFrame) -> pd.DataFrame:
    if "song_id" not in meta_df.columns:
        raise ValueError("Metadata file must contain 'song_id'")
    out = meta_df.copy()
    for c in ["title", "artist", "year", "genre"]:
        if c not in out.columns:
            out[c] = pd.NA
    return out



def resolve_query_index(
    meta_df: pd.DataFrame,
    *,
    song_id: Optional[str],
    title: Optional[str],
    artist: Optional[str],
) -> int:
    if "song_id" not in meta_df.columns:
        raise ValueError("meta_df must contain 'song_id'")
    if song_id:
        mask = _normalize_text(meta_df["song_id"]) == str(song_id).strip().casefold()
        matches = meta_df.index[mask].tolist()
        if not matches:
            raise ValueError(f"song_id not found: {song_id}")
        return int(matches[0])

    if not title:
        raise ValueError("Provide either --song-id or --title")
    if "title" not in meta_df.columns:
        raise ValueError("meta_df must contain 'title' to use --title")

    title_norm = str(title).strip().casefold()
    mask = _normalize_text(meta_df["title"]) == title_norm

    if artist and "artist" in meta_df.columns:
        artist_norm = str(artist).strip().casefold()
        mask = mask & (_normalize_text(meta_df["artist"]) == artist_norm)

    matches = meta_df.index[mask].tolist()
    if len(matches) == 1:
        return int(matches[0])

    if len(matches) == 0:
        contains = _normalize_text(meta_df["title"]).str.contains(re.escape(title_norm), na=False)
        if artist and "artist" in meta_df.columns:
            artist_contains = _normalize_text(meta_df["artist"]).str.contains(
                re.escape(str(artist).strip().casefold()),
                na=False,
            )
            contains = contains & artist_contains
        matches = meta_df.index[contains].tolist()

    if len(matches) == 1:
        return int(matches[0])

    if len(matches) == 0:
        raise ValueError(f"No song found for title={title!r}, artist={artist!r}")

    preview_cols = [c for c in ["song_id", "title", "artist", "year", "genre"] if c in meta_df.columns]
    preview = meta_df.loc[matches[:15], preview_cols]
    raise ValueError(
        "Multiple songs matched. Narrow it down with --artist or --song-id.\n"
        + preview.to_string(index=False)
    )


# -----------------------------------------------------------------------------
# Vector loading
# -----------------------------------------------------------------------------


@dataclass
class LoadedVectors:
    matrix: ArrayLike
    vector_cols: list[str]
    source_path: Path
    auxiliary_df: Optional[pd.DataFrame] = None


EXCLUDE_NUMERIC_COLS = {
    "song_id",
    "year",
    "rank",
    "track_number",
    "chart_position",
    "views",
}

META_TEXT_COLS = {
    "title",
    "artist",
    "genre",
    "album",
    "lyrics",
    "lyrics_clean",
    "lyrics_dedup",
}



def _infer_baseline_csv_vector_cols(df: pd.DataFrame) -> list[str]:
    preferred_prefixes = ("x_", "vec_", "vector_", "feat_")
    pref = [c for c in df.columns if c.startswith(preferred_prefixes)]
    if pref:
        return sorted(pref)

    ratio_cols = sorted([c for c in df.columns if c.startswith("ratio_")])
    if ratio_cols:
        return ratio_cols

    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in EXCLUDE_NUMERIC_COLS and c not in META_TEXT_COLS
    ]
    return sorted(numeric_cols)



def _infer_proposed_csv_vector_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("z_")]
    if not cols:
        raise ValueError("No proposed vector columns found with prefix 'z_'")

    def _key(c: str) -> int:
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return 10**9

    return sorted(cols, key=_key)



def _align_vector_df_to_meta(meta_df: pd.DataFrame, vec_df: pd.DataFrame, vector_cols: list[str]) -> pd.DataFrame:
    if "song_id" not in vec_df.columns:
        if len(vec_df) != len(meta_df):
            raise ValueError(
                "Vector CSV has no song_id column and row count differs from metadata rows. "
                "Cannot align safely."
            )
        aligned = vec_df.copy()
        aligned.insert(0, "song_id", meta_df["song_id"].astype(str).tolist())
        return aligned

    left = meta_df[["song_id"]].copy()
    right = vec_df[["song_id", *vector_cols]].copy()
    left["song_id"] = left["song_id"].astype(str)
    right["song_id"] = right["song_id"].astype(str)
    aligned = left.merge(right, on="song_id", how="left")

    missing = aligned[vector_cols].isna().all(axis=1).sum()
    if missing > 0:
        raise ValueError(
            f"{missing} songs from metadata were missing in vector file. "
            "Make sure vectors were built from the same processed CSV/catalog."
        )
    return aligned



def _align_matrix_to_meta(
    meta_df: pd.DataFrame,
    X: ArrayLike,
    *,
    array_song_ids: Optional[np.ndarray] = None,
) -> ArrayLike:
    if getattr(X, "ndim", None) != 2:
        raise ValueError(f"Expected 2D vector matrix, got shape={getattr(X, 'shape', None)}")

    if array_song_ids is None:
        if X.shape[0] != len(meta_df):
            raise ValueError(
                f"Vector rows ({X.shape[0]}) do not match metadata rows ({len(meta_df)}), "
                "and no song_id mapping was provided."
            )
        return X

    song_ids = np.asarray(array_song_ids).astype(str)
    if len(song_ids) != X.shape[0]:
        raise ValueError(
            f"song_id array length ({len(song_ids)}) does not match vector rows ({X.shape[0]})."
        )

    map_index = {sid: i for i, sid in enumerate(song_ids.tolist())}
    order: list[int] = []
    missing: list[str] = []
    for sid in meta_df["song_id"].astype(str).tolist():
        idx = map_index.get(sid)
        if idx is None:
            missing.append(sid)
        else:
            order.append(idx)
    if missing:
        raise ValueError(
            f"{len(missing)} songs from metadata were missing in the song_id mapping for the vector array."
        )
    order_arr = np.asarray(order, dtype=np.int64)
    return X[order_arr]



def _load_sparse_or_dense_npz(path: Path) -> ArrayLike:
    try:
        return sparse.load_npz(path)
    except Exception:
        pass

    with np.load(path, allow_pickle=True) as obj:
        keys = list(obj.files)
        if {"data", "indices", "indptr", "shape"}.issubset(keys):
            return sparse.csr_matrix(
                (obj["data"], obj["indices"], obj["indptr"]),
                shape=tuple(obj["shape"]),
            )

        if "X" in keys and getattr(obj["X"], "ndim", 0) == 2:
            return np.asarray(obj["X"], dtype=np.float32)

        array_candidates: list[np.ndarray] = []
        for k in keys:
            arr = obj[k]
            if getattr(arr, "ndim", 0) == 2:
                array_candidates.append(arr)
        if not array_candidates:
            raise ValueError(f"Could not find a 2D array inside npz file: {path}")
        return np.asarray(array_candidates[0], dtype=np.float32)



def load_baseline_vectors(
    meta_df: pd.DataFrame,
    path: Path,
    *,
    baseline_song_ids_path: Optional[Path] = None,
) -> LoadedVectors:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        vector_cols = _infer_baseline_csv_vector_cols(df)
        if not vector_cols:
            raise ValueError(f"No baseline vector columns inferred from CSV: {path}")
        aligned = _align_vector_df_to_meta(meta_df, df, vector_cols)
        X = aligned[vector_cols].to_numpy(dtype=np.float32)
        return LoadedVectors(matrix=X, vector_cols=vector_cols, source_path=path, auxiliary_df=aligned)

    if path.suffix.lower() == ".npz":
        song_ids = None
        if baseline_song_ids_path is not None and baseline_song_ids_path.exists():
            song_ids = np.load(baseline_song_ids_path, allow_pickle=True)
        X = _load_sparse_or_dense_npz(path)
        X = _align_matrix_to_meta(meta_df, X, array_song_ids=song_ids)
        return LoadedVectors(matrix=X, vector_cols=[], source_path=path, auxiliary_df=None)

    raise ValueError(f"Unsupported baseline vector format: {path.suffix}")



def load_proposed_vectors(
    meta_df: pd.DataFrame,
    path: Path,
    *,
    proposed_song_ids_path: Optional[Path] = None,
) -> LoadedVectors:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        vector_cols = _infer_proposed_csv_vector_cols(df)
        aligned = _align_vector_df_to_meta(meta_df, df, vector_cols)
        X = aligned[vector_cols].to_numpy(dtype=np.float32)
        return LoadedVectors(matrix=X, vector_cols=vector_cols, source_path=path, auxiliary_df=aligned)

    array_song_ids: Optional[np.ndarray] = None
    if proposed_song_ids_path is not None and proposed_song_ids_path.exists():
        array_song_ids = np.load(proposed_song_ids_path, allow_pickle=True)

    if suffix == ".npz":
        X = _load_sparse_or_dense_npz(path)
        if sparse.issparse(X):
            X = X.toarray()
        X = _align_matrix_to_meta(meta_df, np.asarray(X, dtype=np.float32), array_song_ids=array_song_ids)
        return LoadedVectors(matrix=X, vector_cols=[], source_path=path, auxiliary_df=None)

    if suffix == ".npy":
        X = np.load(path, allow_pickle=True)
        X = _align_matrix_to_meta(meta_df, np.asarray(X, dtype=np.float32), array_song_ids=array_song_ids)
        return LoadedVectors(matrix=X, vector_cols=[], source_path=path, auxiliary_df=None)

    raise ValueError(f"Unsupported proposed vector format: {path.suffix}")


# -----------------------------------------------------------------------------
# Emotion-space construction
# -----------------------------------------------------------------------------


def _emotion_cols_from_df(df: pd.DataFrame, *, mode: str) -> list[str]:
    ratio_cols = sorted([c for c in df.columns if c.startswith("ratio_")])
    vad_cols = [c for c in ["valence", "arousal", "dominance"] if c in df.columns]

    if mode == "ratio":
        return ratio_cols
    if mode == "ratio_vad":
        return ratio_cols + vad_cols
    if mode != "auto":
        raise ValueError(f"Unknown emotion-space mode: {mode}")

    if ratio_cols and vad_cols:
        return ratio_cols + vad_cols
    return ratio_cols





def _emotion_matrix_from_proposed_tail(
    X: np.ndarray,
    emotion_cfg: dict[str, Any],
    *,
    mode: str,
) -> tuple[np.ndarray, list[str]]:
    emo_cfg = emotion_cfg.get("emotion", {}) or {}
    emotions = [str(e).lower() for e in emo_cfg.get("emotions", [])]
    if not emotions:
        raise ValueError("emotion_context.yaml must contain emotion.emotions")

    vector_layout = str(((emotion_cfg.get("aggregation", {}) or {}).get("vector_layout", "embedding_ratio_vad"))).strip()
    emo_dim = len(emotions)

    if vector_layout == "embedding_ratio_vad":
        tail = emo_dim + 3
        if X.shape[1] < tail:
            raise ValueError(f"Proposed vector dim={X.shape[1]} is smaller than expected tail dim={tail}")
        emb_dim = X.shape[1] - tail
        ratio = X[:, emb_dim : emb_dim + emo_dim]
        vad = X[:, emb_dim + emo_dim : emb_dim + emo_dim + 3]
    elif vector_layout == "embedding_ratio_intensity_vad":
        tail = emo_dim + emo_dim + 3
        if X.shape[1] < tail:
            raise ValueError(f"Proposed vector dim={X.shape[1]} is smaller than expected tail dim={tail}")
        emb_dim = X.shape[1] - tail
        ratio = X[:, emb_dim : emb_dim + emo_dim]
        pos = emb_dim + emo_dim + emo_dim
        vad = X[:, pos : pos + 3]
    else:
        raise ValueError(f"Unknown vector layout: {vector_layout}")

    cols = [f"ratio_{e}" for e in emotions]
    ratio = ratio.astype(np.float32, copy=False)
    vad = vad.astype(np.float32, copy=False)

    if mode == "ratio":
        return ratio, cols
    if mode == "ratio_vad":
        return np.concatenate([ratio, vad], axis=1).astype(np.float32, copy=False), cols + ["valence", "arousal", "dominance"]
    if mode == "auto":
        return np.concatenate([ratio, vad], axis=1).astype(np.float32, copy=False), cols + ["valence", "arousal", "dominance"]
    raise ValueError(f"Unknown emotion-space mode: {mode}")



def build_emotion_matrix(
    meta_df: pd.DataFrame,
    *,
    baseline_aux_df: Optional[pd.DataFrame],
    proposed_matrix: np.ndarray,
    emotion_cfg_path: Path,
    mode: str,
    emotion_profiles_df: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, list[str], str]:
    if emotion_profiles_df is not None:
        emotion_profile_cols = _emotion_cols_from_df(emotion_profiles_df, mode=mode)
        if emotion_profile_cols:
            return (
                emotion_profiles_df[emotion_profile_cols].fillna(0.0).to_numpy(dtype=np.float32),
                emotion_profile_cols,
                "emotion_profiles_csv",
            )

    direct_cols = _emotion_cols_from_df(meta_df, mode=mode)
    if direct_cols:
        return meta_df[direct_cols].fillna(0.0).to_numpy(dtype=np.float32), direct_cols, "data_csv"

    if baseline_aux_df is not None:
        aux_cols = _emotion_cols_from_df(baseline_aux_df, mode=mode)
        if aux_cols:
            return (
                baseline_aux_df[aux_cols].fillna(0.0).to_numpy(dtype=np.float32),
                aux_cols,
                "baseline_csv",
            )

    emotion_cfg = _load_yaml(emotion_cfg_path)
    E, cols = _emotion_matrix_from_proposed_tail(proposed_matrix, emotion_cfg, mode=mode)
    return E, cols, "proposed_tail"


# -----------------------------------------------------------------------------
# Recommendation logic
# -----------------------------------------------------------------------------


@dataclass
class ModelOutput:
    name: str
    table: pd.DataFrame
    source_path: Path
    used_mmr: bool
    top_m: int
    k: int



def _recommend_indices(
    meta_df: pd.DataFrame,
    dedup_df: pd.DataFrame,
    X: ArrayLike,
    query_index: int,
    *,
    k: int,
    top_m: int,
    use_mmr: bool,
    mmr_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_count = max(int(k), int(top_m))

    # dedup 이후에도 후보가 충분히 남도록 넉넉하게 먼저 뽑습니다.
    n_rows = int(X.shape[0])
    retrieval_count = min(
        n_rows - 1 if n_rows > 1 else 1,
        max(candidate_count * 5, candidate_count + 50),
    )

    cand_idx, cand_sc = topk_cosine(
        X,
        int(query_index),
        top_k=retrieval_count,
        exclude_self=True,
        normalize=True,
    )
    if cand_idx.size == 0:
        return cand_idx, cand_sc
    
    cand_idx, cand_sc = filter_query_equivalent_candidates(
        meta_df=dedup_df,
        query_index=int(query_index),
        cand_indices=cand_idx,
        cand_scores=cand_sc,
    )

    if cand_idx.size == 0:
        return cand_idx, cand_sc

    if use_mmr:
        sel_idx, sel_sc = mmr_rerank(
            X,
            int(query_index),
            cand_idx,
            cand_sc,
            top_k=int(k),
            lambda_=float(mmr_lambda),
        )
        return sel_idx, sel_sc

    return cand_idx[:k], cand_sc[:k]



def _build_output_table(
    model_name: str,
    meta_df: pd.DataFrame,
    emotion_matrix: np.ndarray,
    query_index: int,
    rec_indices: np.ndarray,
    rec_scores: np.ndarray,
) -> pd.DataFrame:
    query_row = meta_df.iloc[int(query_index)]
    rows: list[dict[str, Any]] = []

    q_emotion = emotion_matrix[int(query_index)]

    for rank, (ri, score) in enumerate(zip(rec_indices.tolist(), rec_scores.tolist()), start=1):
        rec_row = meta_df.iloc[int(ri)]
        emo_sim = _cosine_pair(q_emotion, emotion_matrix[int(ri)])
        row = {
            "model": model_name,
            "query_song_id": query_row.get("song_id"),
            "query_title": query_row.get("title"),
            "query_artist": query_row.get("artist"),
            "rec_song_id": rec_row.get("song_id"),
            "title": rec_row.get("title"),
            "artist": rec_row.get("artist"),
            "year": rec_row.get("year"),
            "genre": rec_row.get("genre"),
            "rank": int(rank),
            "model_score": float(score),
            "emotion_similarity": float(emo_sim),
            "emotion_similarity_pct": float(emo_sim * 100.0),
        }
        if "year" in meta_df.columns:
            try:
                row["year_gap"] = abs(int(query_row["year"]) - int(rec_row["year"]))
            except Exception:
                row["year_gap"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)



def run_one_model(
    *,
    model_name: str,
    X: ArrayLike,
    query_index: int,
    meta_df: pd.DataFrame,
    dedup_df: pd.DataFrame,
    emotion_matrix: np.ndarray,
    source_path: Path,
    k: int,
    top_m: int,
    use_mmr: bool,
    mmr_lambda: float,
) -> ModelOutput:
    idx, sc = _recommend_indices(
        meta_df,
        dedup_df,
        X,
        query_index,
        k=k,
        top_m=top_m,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
    )
    table = _build_output_table(model_name, meta_df, emotion_matrix, query_index, idx, sc)
    return ModelOutput(
        name=model_name,
        table=table,
        source_path=source_path,
        used_mmr=use_mmr,
        top_m=top_m,
        k=k,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare baseline vs proposed recommendations")
    ap.add_argument(
        "--data",
        default="data/processed/genius_processed.csv",
        help="Processed CSV path (used as fallback metadata/emotion source)",
    )
    ap.add_argument(
        "--catalog",
        default="",
        help="Metadata CSV with at least song_id and ideally title/artist/year/genre. "
        "Defaults to artifacts/vectors/catalog.csv if present, otherwise --data.",
    )
    ap.add_argument(
        "--emotion-profiles",
        default="",
        help="Optional emotion_profiles.csv path. If provided, emotion similarity is computed from it first.",
    )
    ap.add_argument("--baseline-vectors", default="", help="Baseline vectors (.npz preferred for TF-IDF baseline, or .csv)")
    ap.add_argument(
        "--baseline-song-ids",
        default="",
        help="Optional .npy file containing song_id order for baseline npz vectors",
    )
    ap.add_argument("--proposed-vectors", default="", help="Proposed vectors (.csv, .npz, .npy)")
    ap.add_argument(
        "--proposed-song-ids",
        default="",
        help="Optional .npy file containing song_id order for proposed npz/npy vectors",
    )
    ap.add_argument(
        "--emotion-config",
        default="configs/emotion_context.yaml",
        help="emotion_context.yaml path (used to parse proposed tail if needed)",
    )

    q = ap.add_argument_group("query")
    q.add_argument("--song-id", default="", help="Exact song_id")
    q.add_argument("--title", default="", help="Song title")
    q.add_argument("--artist", default="", help="Artist name")

    r = ap.add_argument_group("retrieval")
    r.add_argument("--k", type=int, default=10, help="Number of recommendations to return")
    r.add_argument("--top-m", type=int, default=200, help="Candidate count before reranking")
    r.add_argument(
        "--emotion-space",
        choices=["auto", "ratio", "ratio_vad"],
        default="ratio_vad",
        help="Emotion similarity space",
    )
    r.add_argument(
        "--baseline-use-mmr",
        action="store_true",
        default=False,
        help="Apply MMR to baseline recommendations too",
    )
    r.add_argument("--baseline-lambda", type=float, default=0.7, help="Baseline MMR lambda")
    r.add_argument(
        "--proposed-disable-mmr",
        action="store_true",
        default=False,
        help="Disable MMR for proposed model",
    )
    r.add_argument("--proposed-lambda", type=float, default=0.7, help="Proposed MMR lambda")

    o = ap.add_argument_group("output")
    o.add_argument("--save-dir", default="artifacts/demo", help="Directory for CSV/JSON outputs")
    o.add_argument("--output-prefix", default="", help="Optional output filename prefix")
    return ap.parse_args()



def main() -> None:
    args = parse_args()

    data_path = _resolve_path(args.data)
    catalog_path = _resolve_path(args.catalog) if args.catalog else _default_catalog_path()
    emotion_profiles_path = (
        _resolve_path(args.emotion_profiles) if args.emotion_profiles else _default_emotion_profiles_path()
    )
    emotion_cfg_path = _resolve_path(args.emotion_config)
    baseline_song_ids_path = (
        _resolve_path(args.baseline_song_ids) if args.baseline_song_ids else _default_baseline_song_ids_path()
    )
    proposed_song_ids_path = (
        _resolve_path(args.proposed_song_ids) if args.proposed_song_ids else _default_proposed_song_ids_path()
    )
    save_dir = _resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = _resolve_path(args.baseline_vectors) if args.baseline_vectors else _default_baseline_path()
    proposed_path = _resolve_path(args.proposed_vectors) if args.proposed_vectors else _default_proposed_path()

    if baseline_path is None:
        raise FileNotFoundError(
            "Could not find baseline vectors automatically. Use --baseline-vectors explicitly."
        )
    if proposed_path is None:
        raise FileNotFoundError(
            "Could not find proposed vectors automatically. Use --proposed-vectors explicitly."
        )
    if catalog_path is None:
        raise FileNotFoundError(
            "Could not find metadata automatically. Use --catalog explicitly, or keep data/processed/genius_processed.csv available."
        )

    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)
    if not emotion_cfg_path.exists():
        raise FileNotFoundError(emotion_cfg_path)
    if args.data and not data_path.exists():
        raise FileNotFoundError(data_path)

    meta_df = _ensure_meta_cols(pd.read_csv(catalog_path))
    baseline = load_baseline_vectors(
        meta_df,
        baseline_path,
        baseline_song_ids_path=baseline_song_ids_path,
    )
    proposed = load_proposed_vectors(
        meta_df,
        proposed_path,
        proposed_song_ids_path=proposed_song_ids_path,
    )

    emotion_profiles_df: Optional[pd.DataFrame] = None
    if emotion_profiles_path is not None and emotion_profiles_path.exists():
        emotion_profiles_df = _ensure_meta_cols(pd.read_csv(emotion_profiles_path))
        emotion_profiles_df = meta_df[["song_id"]].merge(emotion_profiles_df, on="song_id", how="left")

    data_df: Optional[pd.DataFrame] = None
    if data_path.exists():
        data_df = _ensure_meta_cols(pd.read_csv(data_path))
        if len(data_df) == len(meta_df) and data_df["song_id"].astype(str).equals(meta_df["song_id"].astype(str)):
            pass
        else:
            data_df = meta_df[["song_id"]].merge(data_df, on="song_id", how="left", suffixes=("", "_data"))
            for c in ["title", "artist", "year", "genre"]:
                if c not in data_df.columns and f"{c}_data" in data_df.columns:
                    data_df[c] = data_df[f"{c}_data"]

    query_index = resolve_query_index(
        meta_df,
        song_id=args.song_id or None,
        title=args.title or None,
        artist=args.artist or None,
    )

    emotion_source_df = data_df if data_df is not None else meta_df
    dedup_df = data_df if data_df is not None else meta_df
    proposed_dense = proposed.matrix.toarray() if sparse.issparse(proposed.matrix) else np.asarray(proposed.matrix, dtype=np.float32)
    emotion_matrix, emotion_cols, emotion_source = build_emotion_matrix(
        emotion_source_df,
        baseline_aux_df=baseline.auxiliary_df,
        proposed_matrix=proposed_dense,
        emotion_cfg_path=emotion_cfg_path,
        mode=args.emotion_space,
        emotion_profiles_df=emotion_profiles_df,
    )

    baseline_out = run_one_model(
        model_name="baseline",
        X=baseline.matrix,
        query_index=query_index,
        meta_df=meta_df,
        dedup_df=dedup_df,
        emotion_matrix=emotion_matrix,
        source_path=baseline.source_path,
        k=args.k,
        top_m=args.top_m,
        use_mmr=bool(args.baseline_use_mmr),
        mmr_lambda=float(args.baseline_lambda),
    )
    proposed_out = run_one_model(
        model_name="proposed",
        X=proposed_dense,
        query_index=query_index,
        meta_df=meta_df,
        dedup_df=dedup_df,
        emotion_matrix=emotion_matrix,
        source_path=proposed.source_path,
        k=args.k,
        top_m=args.top_m,
        use_mmr=not bool(args.proposed_disable_mmr),
        mmr_lambda=float(args.proposed_lambda),
    )

    query_row = meta_df.iloc[int(query_index)]
    query_label = f"{query_row.get('title', '')} - {query_row.get('artist', '')}".strip(" -")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.output_prefix.strip() or _slugify(query_label or str(query_row.get("song_id", "query")))
    prefix = f"{prefix}_{timestamp}"

    baseline_csv = save_dir / f"{prefix}_baseline_top{args.k}.csv"
    proposed_csv = save_dir / f"{prefix}_proposed_top{args.k}.csv"
    summary_json = save_dir / f"{prefix}_summary.json"

    baseline_out.table.to_csv(baseline_csv, index=False, encoding="utf-8-sig")
    proposed_out.table.to_csv(proposed_csv, index=False, encoding="utf-8-sig")

    summary = {
        "query": {
            "query_index": int(query_index),
            "song_id": str(query_row.get("song_id")),
            "title": str(query_row.get("title", "")),
            "artist": str(query_row.get("artist", "")),
            "year": _safe_float(query_row.get("year")),
            "genre": str(query_row.get("genre", "")),
        },
        "paths": {
            "catalog": str(catalog_path),
            "data": str(data_path),
            "emotion_profiles": str(emotion_profiles_path) if emotion_profiles_path else "",
            "baseline_vectors": str(baseline.source_path),
            "baseline_song_ids": str(baseline_song_ids_path) if baseline_song_ids_path else "",
            "proposed_vectors": str(proposed.source_path),
            "proposed_song_ids": str(proposed_song_ids_path) if proposed_song_ids_path else "",
            "emotion_config": str(emotion_cfg_path),
            "baseline_csv": str(baseline_csv),
            "proposed_csv": str(proposed_csv),
        },
        "emotion_similarity": {
            "source": emotion_source,
            "columns": emotion_cols,
            "mode": args.emotion_space,
        },
        "retrieval": {
            "k": int(args.k),
            "top_m": int(args.top_m),
            "baseline_use_mmr": bool(args.baseline_use_mmr),
            "baseline_lambda": float(args.baseline_lambda),
            "proposed_use_mmr": not bool(args.proposed_disable_mmr),
            "proposed_lambda": float(args.proposed_lambda),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Query]")
    preview_cols = [c for c in ["song_id", "title", "artist", "year", "genre"] if c in meta_df.columns]
    print(query_row[preview_cols].to_string())
    print(f"\nCatalog           : {catalog_path}")
    print(f"Baseline vectors  : {baseline.source_path}")
    if baseline_song_ids_path is not None and baseline_song_ids_path.exists():
        print(f"Baseline song ids : {baseline_song_ids_path}")
    print(f"Proposed vectors  : {proposed.source_path}")
    if proposed_song_ids_path is not None and proposed_song_ids_path.exists():
        print(f"Proposed song ids : {proposed_song_ids_path}")
    print(f"Emotion source    : {emotion_source} -> {emotion_cols}")
    print(f"Saved baseline    : {baseline_csv}")
    print(f"Saved proposed    : {proposed_csv}")
    print(f"Saved summary     : {summary_json}")

    _print_table("[BASELINE] Recommendations", baseline_out.table)
    _print_table("[PROPOSED] Recommendations", proposed_out.table)


if __name__ == "__main__":
    main()
