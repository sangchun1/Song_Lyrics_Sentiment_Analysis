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
    lyrics_col: str = "lyrics_clean"
    lyric_compare_cols: Tuple[str, ...] = ("lyrics_clean", "lyrics_dedup", "lyrics")

    # title-based suspicion / fallback
    title_subset_overlap_thr: float = 1.0
    min_title_chars: int = 3
    cross_artist_same_title_version_min_tokens: int = 2

    # lyric similarity thresholds
    exact_lyric_thr: float = 0.85
    same_title_lyric_thr: float = 0.55
    version_lyric_thr: float = 0.35
    same_artist_same_title_lyric_thr: float = 0.45

    # line-overlap thresholds (lyrics-first dedup)
    exact_line_overlap_thr: float = 0.85
    same_title_line_overlap_thr: float = 0.45
    version_line_overlap_thr: float = 0.30
    same_artist_same_title_line_overlap_thr: float = 0.25
    min_line_tokens: int = 3
    min_line_chars: int = 8


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


def _normalize_lyrics_lines(text: object) -> str:
    out = unicodedata.normalize("NFKC", _safe_text(text)).casefold()
    out = _strip_brackets(out)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[^\S\n]+", " ", out)
    lines = [_normalize_ws(line) for line in out.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _lyrics_for_similarity(text: object) -> str:
    return _normalize_ws(_normalize_lyrics_lines(text).replace("\n", " "))


def _line_set(text: object, *, min_tokens: int, min_chars: int) -> set[str]:
    lines = _normalize_lyrics_lines(text).split("\n")
    out = set()
    for line in lines:
        if not line:
            continue
        if len(line) < min_chars:
            continue
        if len(line.split()) < min_tokens:
            continue
        out.add(line)
    return out


def _line_overlap_ratio(
    query_lyrics: object,
    cand_lyrics: object,
    *,
    min_tokens: int,
    min_chars: int,
) -> float:
    q_lines = _line_set(query_lyrics, min_tokens=min_tokens, min_chars=min_chars)
    c_lines = _line_set(cand_lyrics, min_tokens=min_tokens, min_chars=min_chars)
    if not q_lines or not c_lines:
        return np.nan
    return float(len(q_lines & c_lines)) / float(min(len(q_lines), len(c_lines)))


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


def _resolve_lyric_cols(meta_df: pd.DataFrame, cfg: DedupConfig) -> list[str]:
    cols: list[str] = []
    for col in (cfg.lyrics_col, *cfg.lyric_compare_cols):
        if col in meta_df.columns and col not in cols:
            cols.append(col)
    return cols


def _nanmax_rows(arrays: list[np.ndarray], *, n: int) -> np.ndarray:
    if not arrays:
        return np.full(n, np.nan, dtype=float)
    stack = np.vstack(arrays)
    out = np.nanmax(stack, axis=0)
    all_nan = np.all(np.isnan(stack), axis=0)
    out[all_nan] = np.nan
    return out.astype(float)


def _is_query_equivalent(
    *,
    query_title: str,
    cand_title: str,
    query_artist: str,
    cand_artist: str,
    lyric_sim: float,
    line_overlap: float,
    cfg: DedupConfig,
) -> bool:
    q_canon = canonical_title(query_title)
    c_canon = canonical_title(cand_title)
    same_title = bool(q_canon) and (q_canon == c_canon)
    title_subset_overlap = _title_subset_overlap(query_title, cand_title)
    has_version_keyword = _contains_version_keyword(cand_title)
    same_artist = _same_artist(query_artist, cand_artist)
    q_title_token_count = len(_title_tokens(query_title))

    lyric_ok = np.isfinite(lyric_sim)
    overlap_ok = np.isfinite(line_overlap)

    # 1) title와 무관하게 가사가 거의 같은 경우
    if lyric_ok and lyric_sim >= cfg.exact_lyric_thr:
        return True
    if overlap_ok and line_overlap >= cfg.exact_line_overlap_thr:
        return True

    # 2) canonical title이 같고 가사나 핵심 라인이 충분히 비슷한 경우
    if same_title:
        if lyric_ok and lyric_sim >= cfg.same_title_lyric_thr:
            return True
        if overlap_ok and line_overlap >= cfg.same_title_line_overlap_thr:
            return True

    # 3) version keyword가 있는 변형은 lyrics 기준으로 더 공격적으로 제거
    if has_version_keyword and title_subset_overlap >= cfg.title_subset_overlap_thr:
        if lyric_ok and lyric_sim >= cfg.version_lyric_thr:
            return True
        if overlap_ok and line_overlap >= cfg.version_line_overlap_thr:
            return True

    # 4) 같은 아티스트 + 같은 canonical title이면 같은 곡 변형일 가능성이 높음
    if same_artist and same_title:
        if lyric_ok and lyric_sim >= cfg.same_artist_same_title_lyric_thr:
            return True
        if overlap_ok and line_overlap >= cfg.same_artist_same_title_line_overlap_thr:
            return True

    # 5) lyrics가 비어 있을 때만 쓰는 fallback title rules
    if (not lyric_ok) and (not overlap_ok):
        if same_artist and same_title and has_version_keyword:
            return True
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

    cand_rows = meta_df.iloc[cand_indices]
    cand_titles = cand_rows[cfg.title_col].fillna("").astype(str).tolist()
    cand_artists = (
        cand_rows[cfg.artist_col].fillna("").astype(str).tolist()
        if cfg.artist_col in cand_rows.columns
        else [""] * len(cand_indices)
    )

    lyric_cols = _resolve_lyric_cols(meta_df, cfg)
    lyric_sims_all: list[np.ndarray] = []
    line_overlaps_all: list[np.ndarray] = []
    n_cands = len(cand_indices)

    for col in lyric_cols:
        q_lyrics_lines = _normalize_lyrics_lines(q_row.get(col, ""))
        q_lyrics_for_sim = _lyrics_for_similarity(q_lyrics_lines)

        cand_col_lines = (
            cand_rows[col].fillna("").astype(str).map(_normalize_lyrics_lines).tolist()
        )
        cand_col_for_sim = [_lyrics_for_similarity(text) for text in cand_col_lines]

        lyric_sims_all.append(_lyric_similarities(q_lyrics_for_sim, cand_col_for_sim))
        line_overlaps_all.append(
            np.asarray(
                [
                    _line_overlap_ratio(
                        q_lyrics_lines,
                        cand_text,
                        min_tokens=cfg.min_line_tokens,
                        min_chars=cfg.min_line_chars,
                    )
                    for cand_text in cand_col_lines
                ],
                dtype=float,
            )
        )

    max_lyric_sims = _nanmax_rows(lyric_sims_all, n=n_cands)
    max_line_overlaps = _nanmax_rows(line_overlaps_all, n=n_cands)

    keep = np.ones(len(cand_indices), dtype=bool)
    for pos in range(len(cand_indices)):
        if _is_query_equivalent(
            query_title=q_title,
            cand_title=cand_titles[pos],
            query_artist=q_artist,
            cand_artist=cand_artists[pos],
            lyric_sim=float(max_lyric_sims[pos]),
            line_overlap=float(max_line_overlaps[pos]),
            cfg=cfg,
        ):
            keep[pos] = False

    return cand_indices[keep], cand_scores[keep]
