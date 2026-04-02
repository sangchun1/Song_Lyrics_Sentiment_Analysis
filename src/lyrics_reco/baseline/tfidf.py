"""lyrics_reco.baseline.tfidf

TF-IDF baselines with research-plan helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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


def compute_term_weights_from_intensity(vocab: Dict[str, int], intensity_lookup: Dict[str, Dict[str, float]], *, default: float = 1.0, mode: str = "1+max") -> np.ndarray:
    weights = np.full(len(vocab), float(default), dtype=np.float32)
    for w, idx in vocab.items():
        d = intensity_lookup.get(w)
        if not d:
            continue
        scores = np.array(list(d.values()), dtype=np.float32)
        if scores.size == 0:
            continue
        s = float(scores.max()) if mode.endswith("max") else float(scores.mean())
        if mode.startswith("1+"):
            s = 1.0 + s
        weights[idx] = np.float32(s)
    return weights


def apply_term_weights(X: sparse.csr_matrix, term_weights: np.ndarray) -> sparse.csr_matrix:
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    D = sparse.diags(term_weights.astype(np.float32), offsets=0, format="csr")
    return (X @ D).tocsr()


def build_vocab_emotion_matrix(vocab: Dict[str, int], bundle, emotions: Sequence[str]) -> np.ndarray:
    n_terms = len(vocab)
    emo_cols = [str(e).lower() for e in emotions]
    nrc_df = bundle.nrc.df.reindex(columns=emo_cols, fill_value=0.0)
    A = np.zeros((n_terms, len(emo_cols)), dtype=np.float32)
    for term, j in vocab.items():
        if term in nrc_df.index:
            vec = nrc_df.loc[term].to_numpy(dtype=np.float32)
            s = float(vec.sum())
            if s > 0:
                vec = vec / s
            A[j] = vec
    return A


def build_song_emotion_distribution(df: pd.DataFrame, *, text_col: str, bundle, emotions: Sequence[str]) -> np.ndarray:
    emo_cols = [str(e).lower() for e in emotions]
    nrc_df = bundle.nrc.df.reindex(columns=emo_cols, fill_value=0.0).astype(np.float32)
    vocab_words = nrc_df.index.astype(str).tolist()
    cv = CountVectorizer(vocabulary=vocab_words, lowercase=True, token_pattern=r"(?u)\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b")
    Xw = cv.fit_transform(df[text_col].astype(str).tolist())
    W = sparse.csr_matrix(nrc_df.values.astype(np.float32))
    counts = (Xw @ W).astype(np.float32)
    counts = np.asarray(counts.todense(), dtype=np.float32)
    den = np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
    return (counts / den).astype(np.float32)


def apply_song_conditioned_emotion_weights(
    X: sparse.csr_matrix,
    P_song: np.ndarray,
    A_vocab: np.ndarray,
    *,
    floor_weight: float = 0.0,
) -> sparse.csr_matrix:
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    X = X.copy().astype(np.float32)
    for r in range(X.shape[0]):
        start, end = X.indptr[r], X.indptr[r + 1]
        cols = X.indices[start:end]
        if len(cols) == 0:
            continue
        weights = A_vocab[cols] @ P_song[r]
        if floor_weight > 0:
            weights = np.maximum(weights, floor_weight)
        if np.all(weights == 0):
            continue
        X.data[start:end] *= weights.astype(np.float32)
    return X


def save_vocab_idf_csv(vocab: Dict[str, int], idf: np.ndarray, out_path: PathLike, *, paths: ProjectPaths = PATHS) -> Path:
    p = Path(out_path)
    if not p.is_absolute():
        p = (paths.root / p).resolve()
    ensure_parent_dir(p)
    rows = [{"term": t, "index": int(i), "idf": float(idf[i])} for t, i in vocab.items()]
    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    return save_csv(df, p, index=False)


def load_vocab_idf_csv(path: PathLike, *, paths: ProjectPaths = PATHS):
    p = Path(path)
    if not p.is_absolute():
        p = (paths.root / p).resolve()
    df = pd.read_csv(p).sort_values("index")
    vocab = {r["term"]: int(r["index"]) for _, r in df.iterrows()}
    idf = df["idf"].astype(np.float32).to_numpy()
    return vocab, idf
