"""Compatibility helpers for wrapping existing pipeline CLIs.

The current pipeline modules already expose `main()` functions that parse
`sys.argv` directly. These helpers let the new `lyrics_reco.cli.*` wrappers
forward an explicit argv list without modifying the pipeline implementations.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Sequence


@contextmanager
def patched_argv(argv: Sequence[str] | None = None, *, prog: str = "python") -> Iterator[None]:
    old_argv = sys.argv[:]
    sys.argv = [prog, *(list(argv) if argv is not None else [])]
    try:
        yield
    finally:
        sys.argv = old_argv


def run_with_argv(main_fn: Callable[[], None], argv: Sequence[str] | None = None, *, prog: str) -> None:
    with patched_argv(argv, prog=prog):
        main_fn()
