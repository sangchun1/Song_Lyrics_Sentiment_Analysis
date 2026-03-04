"""lyrics_reco.emotion_context.embedder

Sentence embedding wrapper (Sentence-Transformers).

Notes:
- Uses sentence_transformers.SentenceTransformer
- Handles device auto-selection
- Optionally L2-normalizes embeddings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"           # auto | cpu | cuda | cuda:0 ...
    batch_size: int = 64
    normalize_embeddings: bool = True
    max_length: int = 256

class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "sentence-transformers is required for emotion_context. "
                "Install: pip install sentence-transformers"
            ) from e

        dev = cfg.device
        if dev == "auto":
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"

        self.device = dev
        self.model = SentenceTransformer(cfg.model_name, device=self.device)

        # Some models support max_seq_length
        try:
            self.model.max_seq_length = int(cfg.max_length)
        except Exception:
            pass

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if texts is None:
            return np.zeros((0, 0), dtype=np.float32)
        texts = list(texts)
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        emb = self.model.encode(
            texts,
            batch_size=int(self.cfg.batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=bool(self.cfg.normalize_embeddings),
        ).astype(np.float32)
        return emb
