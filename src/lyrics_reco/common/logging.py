"""
lyrics_reco.common.logging

Lightweight logging utilities for reproducible experiment runs.

Goals:
- Provide a consistent "run logger" that writes to:
    artifacts/runs/<run_id>/run.log
  (optionally also to reports/runs/<run_id>/run.log)
- Avoid duplicate handlers when called multiple times.
- Keep dependencies minimal (stdlib logging). If `rich` is installed, we can
  optionally use RichHandler for nicer console output.

Typical usage:
    from lyrics_reco.common.logging import setup_run_logger
    logger = setup_run_logger(run_id)
    logger.info("Hello")

Design notes:
- Logs are text; metrics/tables should be saved as CSV elsewhere.
- This module name is `logging.py` (requested), so avoid `import logging as logging`
  patterns in other modules; use `import logging` normally is fine.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

from .paths import PATHS, ProjectPaths, ensure_dir


PathLike = Union[str, os.PathLike, Path]


# -----------------------------
# Formatting
# -----------------------------
DEFAULT_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _level(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


# -----------------------------
# Paths
# -----------------------------
@dataclass(frozen=True)
class RunLogPaths:
    run_id: str
    artifacts_dir: Path
    reports_dir: Path
    artifacts_log: Path
    reports_log: Path


def get_run_log_paths(run_id: str, *, paths: ProjectPaths = PATHS) -> RunLogPaths:
    art_dir = ensure_dir(paths.art_runs / run_id)
    rep_dir = ensure_dir(paths.rep_runs / run_id)

    return RunLogPaths(
        run_id=run_id,
        artifacts_dir=art_dir,
        reports_dir=rep_dir,
        artifacts_log=art_dir / "run.log",
        reports_log=rep_dir / "run.log",
    )


# -----------------------------
# Handler helpers (dedup)
# -----------------------------
def _has_handler(logger: logging.Logger, handler_id: str) -> bool:
    return any(getattr(h, "_lyrics_reco_handler_id", None) == handler_id for h in logger.handlers)


def _tag_handler(handler: logging.Handler, handler_id: str) -> logging.Handler:
    setattr(handler, "_lyrics_reco_handler_id", handler_id)
    return handler


def _build_file_handler(
    path: Path,
    *,
    level: Union[int, str] = logging.INFO,
    fmt: str = DEFAULT_FMT,
    datefmt: str = DEFAULT_DATEFMT,
) -> logging.Handler:
    ensure_dir(path.parent)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(_level(level))
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    return fh


def _build_console_handler(
    *,
    level: Union[int, str] = logging.INFO,
    fmt: str = DEFAULT_FMT,
    datefmt: str = DEFAULT_DATEFMT,
    use_rich: bool = True,
) -> logging.Handler:
    if use_rich:
        try:
            from rich.logging import RichHandler  # type: ignore

            ch = RichHandler(
                level=_level(level),
                rich_tracebacks=True,
                show_time=False,  # time already in formatter if you want it
                show_level=True,
                show_path=False,
            )
            ch.setFormatter(logging.Formatter(fmt="%(message)s"))
            return ch
        except Exception:
            pass

    ch = logging.StreamHandler()
    ch.setLevel(_level(level))
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    return ch


# -----------------------------
# Public API
# -----------------------------
def setup_run_logger(
    run_id: str,
    *,
    name: str = "lyrics_reco",
    level: Union[int, str] = "INFO",
    console: bool = True,
    file: bool = True,
    also_to_reports: bool = False,
    console_level: Optional[Union[int, str]] = None,
    file_level: Optional[Union[int, str]] = None,
    fmt: str = DEFAULT_FMT,
    datefmt: str = DEFAULT_DATEFMT,
    use_rich_console: bool = True,
    reset_handlers: bool = False,
    paths: ProjectPaths = PATHS,
) -> logging.Logger:
    """
    Create/configure a logger for a specific run.

    Parameters
    ----------
    run_id : str
        Unique run id.
    name : str
        Logger name.
    level : int | str
        Base logger level.
    console : bool
        Add console handler.
    file : bool
        Add file handler writing to artifacts/runs/<run_id>/run.log
    also_to_reports : bool
        If True, also write a second file to reports/runs/<run_id>/run.log
    console_level, file_level : optional
        Handler-specific override levels.
    reset_handlers : bool
        If True, remove existing handlers before adding new ones.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_level(level))
    logger.propagate = False  # avoid double logging via root logger

    if reset_handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    run_paths = get_run_log_paths(run_id, paths=paths)

    if file:
        handler_id = f"file:{run_paths.artifacts_log}"
        if not _has_handler(logger, handler_id):
            fh = _build_file_handler(
                run_paths.artifacts_log,
                level=file_level or level,
                fmt=fmt,
                datefmt=datefmt,
            )
            logger.addHandler(_tag_handler(fh, handler_id))

        if also_to_reports:
            handler_id2 = f"file:{run_paths.reports_log}"
            if not _has_handler(logger, handler_id2):
                fh2 = _build_file_handler(
                    run_paths.reports_log,
                    level=file_level or level,
                    fmt=fmt,
                    datefmt=datefmt,
                )
                logger.addHandler(_tag_handler(fh2, handler_id2))

    if console:
        handler_id = "console"
        if not _has_handler(logger, handler_id):
            ch = _build_console_handler(
                level=console_level or level,
                fmt=fmt,
                datefmt=datefmt,
                use_rich=use_rich_console,
            )
            logger.addHandler(_tag_handler(ch, handler_id))

    # Small header
    logger.info("=== Run logger initialized ===")
    logger.info("run_id=%s", run_id)
    logger.info("log_file=%s", str(run_paths.artifacts_log))

    return logger


def get_logger(name: str = "lyrics_reco") -> logging.Logger:
    """Return the logger without altering handlers (useful in submodules)."""
    return logging.getLogger(name)


@contextmanager
def log_section(logger: logging.Logger, title: str) -> Iterator[None]:
    """
    Context manager to log a section with elapsed time.

    Example:
        with log_section(logger, "build embeddings"):
            ...
    """
    logger.info("---- %s (start) ----", title)
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        logger.info("---- %s (end) | %.2fs ----", title, dt)
