"""
lyrics_reco.baseline.tfidf

TF-IDF baselines (CSV-first artifact mindset).

Baselines:
1) TF-IDF on full lyrics (lyrics_dedup recommended)
2) Emotion-only TF-IDF: tokens restricted to NRC lexicon words
3) Emotion-weighted TF-IDF: multiply TF-IDF columns by lexicon-based term weights
   (e.g., NRC intensity max score per term)

Saving:
- Do NOT save full sparse matrix as CSV.
- Save vocab+idf (and optionally term weights) as CSV for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ..common.io import save_csv
from ..common.paths import PATHS, ProjectPaths, ensure_parent_dir
from .tokenize import prepare_text_series, simple_tokenize

PathLike = Union[str, Path]

@dataclass(frozen=True)
class TfidfArtifacts:
    vectorizer: TfidfVectorizer
    X: sparse.csr_matrix
    vocab: Dict[str, int]
    idf: np.ndarray

def build_tfidf(
    df: pd.DataFrame,
    *,
    text_col: str,
    max_features: int = 200_000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3,
    max_df: float = 0.95,
    stopwords: Optional[set] = None,
    filler_words: Optional[set] = None,
    slang_map: Optional[dict] = None,
) -> TfidfArtifacts:
    texts = prepare_text_series(df[text_col], stopwords=stopwords, filler_words=filler_words, slang_map=slang_map)

    vec = TfidfVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        dtype=np.float32,
    )
    X = vec.fit_transform(texts.tolist())
    vocab = dict(vec.vocabulary_)
    idf = vec.idf_.astype(np.float32, copy=True)
    return TfidfArtifacts(vectorizer=vec, X=X.tocsr(), vocab=vocab, idf=idf)

def build_emotion_tfidf(
    df: pd.DataFrame,
    *,
    text_col: str,
    emotion_vocab: set,
    max_features: int = 150_000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfArtifacts:
    def emo_only(text: str) -> str:
        toks = simple_tokenize(text, lowercase=True)
        toks = [t for t in toks if t in emotion_vocab]
        return " ".join(toks)

    texts = df[text_col].astype(str).map(emo_only)

    vec = TfidfVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        dtype=np.float32,
    )
    X = vec.fit_transform(texts.tolist())
    vocab = dict(vec.vocabulary_)
    idf = vec.idf_.astype(np.float32, copy=True)
    return TfidfArtifacts(vectorizer=vec, X=X.tocsr(), vocab=vocab, idf=idf)

def compute_term_weights_from_intensity(
    vocab: Dict[str, int],
    intensity_lookup: Dict[str, Dict[str, float]],
    *,
    default: float = 1.0,
    mode: str = "1+max",  # 1+max | max | 1+mean | mean
) -> np.ndarray:
    """
    Build a term-weight vector aligned to TF-IDF vocabulary indices.
    intensity_lookup: word -> {emotion: score}
    """
    weights = np.full(len(vocab), float(default), dtype=np.float32)

    for w, idx in vocab.items():
        d = intensity_lookup.get(w)
        if not d:
            continue
        scores = np.array(list(d.values()), dtype=np.float32)
        if scores.size == 0:
            continue
        if mode.endswith("max"):
            s = float(scores.max())
        else:
            s = float(scores.mean())
        if mode.startswith("1+"):
            s = 1.0 + s
        weights[idx] = np.float32(s)

    return weights

def apply_term_weights(X: sparse.csr_matrix, term_weights: np.ndarray) -> sparse.csr_matrix:
    """X_weighted = X @ diag(term_weights)"""
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    D = sparse.diags(term_weights.astype(np.float32), offsets=0, format="csr")
    return (X @ D).tocsr()

def save_vocab_idf_csv(
    vocab: Dict[str, int],
    idf: np.ndarray,
    out_path: PathLike,
    *,
    paths: ProjectPaths = PATHS,
) -> Path:
    """Save vocabulary + idf to CSV: term,index,idf"""
    p = Path(out_path)
    if not p.is_absolute():
        p = (paths.root / p).resolve()
    ensure_parent_dir(p)

    rows = [{"term": t, "index": int(i), "idf": float(idf[i])} for t, i in vocab.items()]
    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    return save_csv(df, p, index=False)

def load_vocab_idf_csv(
    path: PathLike,
    *,
    paths: ProjectPaths = PATHS,
):
    """Load vocabulary + idf from CSV saved by save_vocab_idf_csv."""
    p = Path(path)
    if not p.is_absolute():
        p = (paths.root / p).resolve()
    df = pd.read_csv(p).sort_values("index")
    vocab = {r["term"]: int(r["index"]) for _, r in df.iterrows()}
    idf = df["idf"].astype(np.float32).to_numpy()
    return vocab, idf