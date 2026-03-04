"""lyrics_reco.emotion_context.utils"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, List, Optional, Dict

def cfg_get(cfg: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur

def batched(seq: Sequence[Any], batch_size: int) -> List[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    out = []
    for i in range(0, len(seq), batch_size):
        out.append(seq[i:i+batch_size])
    return out
