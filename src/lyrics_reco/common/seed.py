"""
lyrics_reco.common.seed

Minimal, dependency-light seeding utilities for reproducible experiments.

Why:
- Evaluation query sampling, shuffles, and some ML components can be stochastic.
- A single `set_seed(seed)` call at the start of a pipeline reduces "why did results change?" issues.

Usage:
    from lyrics_reco.common.seed import set_seed
    set_seed(cfg["eval"]["seed"], deterministic=False)

Notes:
- `deterministic=True` can reduce speed and may still not guarantee perfect determinism on all ops.
"""

from __future__ import annotations

import os
import random
from typing import Optional


def set_seed(seed: Optional[int], *, deterministic: bool = False) -> None:
    """
    Set random seeds across common libraries.

    Parameters
    ----------
    seed : int | None
        If None, does nothing.
    deterministic : bool
        If True, sets PyTorch deterministic flags (if torch is installed).

    This function is safe even if numpy/torch are not installed.
    """
    if seed is None:
        return

    # 1) Python stdlib
    random.seed(seed)

    # 2) Hash-based determinism for Python (affects iteration order of some hashed containers)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3) NumPy (optional)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    # 4) PyTorch (optional)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Determinism flags (may impact performance)
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

            try:
                torch.backends.cudnn.deterministic = True  # type: ignore
                torch.backends.cudnn.benchmark = False  # type: ignore
            except Exception:
                pass
    except Exception:
        pass


# Alias (common naming)
seed_everything = set_seed
