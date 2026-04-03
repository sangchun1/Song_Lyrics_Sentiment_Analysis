# src/lyrics_reco/retrieval/dedup.py

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

_VERSION_PATTERNS = [
    r"\bremix\b",
    r"\bmix\b",
    r"\bcover\b",
    r"\blive\b",
    r"\bacoustic\b",
    r"\bversion\b",
    r"\bedit\b",
    r"\bremaster(?:ed)?\b",
    r"\bkaraoke\b",
    r"\binstrumental\b",
    r"\bdemo\b",
    r"\bradio\s+edit\b",
    r"\bsped\s*up\b",
    r"\bslowed(?:\s*down)?\b",
    r"\bflip\b",
    r"\bbootleg\b",
    r"\brework\b",
    r"\bmashup\b",
    r"\btribute\b",
    r"\breprise\b",
    r"\binterlude\s+version\b",
]
_VERSION_REGEX = re.compile("|".join(_VERSION_PATTERNS), flags=re.IGNORECASE)
_BRACKET_REGEX = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}")
_FEAT_TRAIL_REGEX = re.compile(r"\b(?:feat|ft)\.?(?:\s+|$).*?$", flags=re.IGNORECASE)

@dataclass(frozen=True)
class DedupConfig:
    title_col: str = "title"
    artist_col: str = "artist"
    lyrics_col: str = "lyrics_dedup"

    # title-based suspicion
    title_subset_overlap_thr: float = 1.0
    min_title_chars: int = 3
    cross_artist_same_title_version_min_tokens: int = 2

    # lyric similarity thresholds
    exact_lyric_thr: float = 0.90
    same_title_lyric_thr: float = 0.75
    version_lyric_thr: float = 0.60
    same_artist_same_title_lyric_thr: float = 0.55

def _safe_text(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if pd.isna(x):
        return ""
    return str(x)

def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _strip_brackets(text: str) -> str:
    return _BRACKET_REGEX.sub(" ", text)

def _contains_version_keyword(text: str) -> bool:
    return bool(_VERSION_REGEX.search(_safe_text(text)))

def _same_artist(a: object, b: object) -> bool:
    a_text = _safe_text(a).strip().casefold()
    b_text = _safe_text(b).strip().casefold()
    return bool(a_text) and a_text == b_text

def canonical_title(title: object) -> str:
    text = unicodedata.normalize("NFKC", _safe_text(title)).casefold()
    text = _strip_brackets(text)
    text = _FEAT_TRAIL_REGEX.sub(" ", text)
    text = _VERSION_REGEX.sub(" ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[/_|:+-]+", " ", text)
    text = re.sub(r"[^0-9a-z가-힣\s]", " ", text)
    text = _normalize_ws(text)
    return text

def _title_tokens(title: object) -> set[str]:
    canon = canonical_title(title)
    return {tok for tok in canon.split() if tok}

def _title_subset_overlap(a: object, b: object) -> float:
    ta = _title_tokens(a)
    tb = _title_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return float(inter) / float(min(len(ta), len(tb)))

def _normalize_lyrics(text: object) -> str:
    out = unicodedata.normalize("NFKC", _safe_text(text)).casefold()
    out = _strip_brackets(out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()

def _lyric_similarities(query_lyrics: str, cand_lyrics: list[str]) -> np.ndarray:
    if not query_lyrics.strip():
        return np.full(len(cand_lyrics), np.nan, dtype=float)

    cleaned = [c for c in cand_lyrics if c.strip()]
    if not cleaned:
        return np.full(len(cand_lyrics), np.nan, dtype=float)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=False,
        sublinear_tf=True,
        min_df=1,
    )
    corpus = [query_lyrics] + cand_lyrics
    X = vectorizer.fit_transform(corpus)
    sims = linear_kernel(X[0:1], X[1:]).ravel()
    return np.asarray(sims, dtype=float)

def _is_query_equivalent(
    *,
    query_title: str,
    cand_title: str,
    query_artist: str,
    cand_artist: str,
    lyric_sim: float,
    cfg: DedupConfig,
) -> bool:
    q_canon = canonical_title(query_title)
    c_canon = canonical_title(cand_title)
    same_title = bool(q_canon) and (q_canon == c_canon)
    title_subset_overlap = _title_subset_overlap(query_title, cand_title)
    has_version_keyword = _contains_version_keyword(cand_title)
    same_artist = _same_artist(query_artist, cand_artist)
    q_title_token_count = len(_title_tokens(query_title))

    # 1) title와 무관하게 가사가 거의 같은 경우
    if np.isfinite(lyric_sim) and lyric_sim >= cfg.exact_lyric_thr:
        return True

    # 2) canonical title이 같은데 가사도 충분히 비슷한 경우
    if same_title and np.isfinite(lyric_sim) and lyric_sim >= cfg.same_title_lyric_thr:
        return True

    # 3) cover/remix/live/acoustic 같은 버전 키워드가 있고,
    # 제목이 query title을 거의 포함하면서 가사도 비슷한 경우
    if (
        has_version_keyword
        and title_subset_overlap >= cfg.title_subset_overlap_thr
        and np.isfinite(lyric_sim)
        and lyric_sim >= cfg.version_lyric_thr
    ):
        return True

    # 4) 같은 아티스트 + 같은 canonical title이면 같은 곡 변형일 가능성이 높음
    if (
        same_artist
        and same_title
        and np.isfinite(lyric_sim)
        and lyric_sim >= cfg.same_artist_same_title_lyric_thr
    ):
        return True

    # 5) lyrics가 비어 있어도, same artist + same title + version keyword면 제외
    if same_artist and same_title and has_version_keyword:
        return True

    # 6) cross-artist라도 multi-token 제목이 같고 version keyword가 있으면
    # cover / tribute / remix 변형일 가능성이 높으므로 더 공격적으로 제외
    # 단, "Stay", "Hello" 같은 흔한 1-token 제목의 과도한 차단은 피한다.
    if (
        (not same_artist)
        and same_title
        and has_version_keyword
        and q_title_token_count >= cfg.cross_artist_same_title_version_min_tokens
    ):
        return True

    return False

def filter_query_equivalent_candidates(
    meta_df: pd.DataFrame,
    *,
    query_index: int,
    cand_indices: np.ndarray,
    cand_scores: np.ndarray,
    cfg: DedupConfig = DedupConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove query-equivalent songs from a candidate pool.

    Intended usage:

    topk_cosine -> filter_query_equivalent_candidates -> mmr_rerank

    Query-equivalent means:

    - the same song
    - cover/remix/live/acoustic/remaster variants
    - near-duplicate lyric versions
    """
    if len(cand_indices) != len(cand_scores):
        raise ValueError("cand_indices and cand_scores must have same length")
    if len(cand_indices) == 0:
        return cand_indices, cand_scores
    if cfg.title_col not in meta_df.columns:
        return cand_indices, cand_scores

    q_row = meta_df.iloc[int(query_index)]
    q_title = _safe_text(q_row.get(cfg.title_col, ""))
    q_artist = _safe_text(q_row.get(cfg.artist_col, ""))
    q_lyrics = _normalize_lyrics(q_row.get(cfg.lyrics_col, ""))

    cand_rows = meta_df.iloc[cand_indices]
    cand_titles = cand_rows[cfg.title_col].fillna("").astype(str).tolist()
    cand_artists = (
        cand_rows[cfg.artist_col].fillna("").astype(str).tolist()
        if cfg.artist_col in cand_rows.columns
        else [""] * len(cand_indices)
    )
    cand_lyrics = (
        cand_rows[cfg.lyrics_col].fillna("").astype(str).map(_normalize_lyrics).tolist()
        if cfg.lyrics_col in cand_rows.columns
        else [""] * len(cand_indices)
    )

    q_canon = canonical_title(q_title)
    q_title_token_count = len(_title_tokens(q_title))
    suspect_mask = []
    for title, artist in zip(cand_titles, cand_artists):
        c_canon = canonical_title(title)
        subset_overlap = _title_subset_overlap(q_title, title)
        same_artist = _same_artist(q_artist, artist)
        suspect = False

        if q_canon and c_canon and q_canon == c_canon:
            suspect = True
        elif (
            len(q_canon) >= cfg.min_title_chars
            and len(c_canon) >= cfg.min_title_chars
            and _contains_version_keyword(title)
            and subset_overlap >= cfg.title_subset_overlap_thr
        ):
            suspect = True
        elif (
            q_canon
            and c_canon
            and q_canon == c_canon
            and _contains_version_keyword(title)
            and (same_artist or q_title_token_count >= cfg.cross_artist_same_title_version_min_tokens)
        ):
            suspect = True

        suspect_mask.append(suspect)

    suspect_mask = np.asarray(suspect_mask, dtype=bool)
    if not suspect_mask.any():
        return cand_indices, cand_scores

    suspect_positions = np.flatnonzero(suspect_mask)
    suspect_lyrics = [cand_lyrics[pos] for pos in suspect_positions]
    suspect_sims = _lyric_similarities(q_lyrics, suspect_lyrics)

    keep = np.ones(len(cand_indices), dtype=bool)
    for local_idx, pos in enumerate(suspect_positions):
        lyric_sim = float(suspect_sims[local_idx])
        if _is_query_equivalent(
            query_title=q_title,
            cand_title=cand_titles[pos],
            query_artist=q_artist,
            cand_artist=cand_artists[pos],
            lyric_sim=lyric_sim,
            cfg=cfg,
        ):
            keep[pos] = False

    return cand_indices[keep], cand_scores[keep]
