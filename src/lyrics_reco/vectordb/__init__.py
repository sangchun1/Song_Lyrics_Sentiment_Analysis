"""
lyrics_reco.vectordb

Vector database backends for retrieval (primarily Chroma).

Public API:
  - IndexConfig
  - open_vectordb_from_cfg
  - build_index_from_frames
  - query_by_vector / query_by_id
"""

from .config import IndexConfig
from .factory import open_vectordb_from_cfg
from .indexer import build_index_from_frames
from .query import query_by_vector, query_by_id

__all__ = [
    "IndexConfig",
    "open_vectordb_from_cfg",
    "build_index_from_frames",
    "query_by_vector",
    "query_by_id",
]