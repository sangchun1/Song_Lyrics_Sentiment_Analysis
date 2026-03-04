"""
lyrics_reco.vectordb.config
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexConfig:
    backend: str = "chroma"
    persist_dir: str = "artifacts/indexes/chroma"
    collection_name: str = "lyrics_reco"
    metric: str = "cosine"          # cosine | l2 | ip (backend-dependent)
    rebuild: bool = False           # true면 컬렉션 삭제 후 재생성

    # ingestion
    batch_size: int = 5000
    normalize_before_upsert: bool = True  # cosine이면 True 추천(정규화 저장)

    # columns
    id_col: str = "song_id"