"""Thin loguru wrapper so every module logs consistently.

Stage failures are logged with ``log.exception`` (full traceback), never just the
exception message: a bare message is what made the reference implementation's
errors point nowhere near their cause.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_configured = False
_file_sinks: set[Path] = set()
_FORMAT = ("<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
           "<cyan>{extra[stage]}</cyan> | {message}")


def _with_stage(record) -> bool:
    record["extra"].setdefault("stage", "-")
    return True


def setup_logging(level: str = "INFO", *, force: bool = False) -> None:
    global _configured
    if _configured and not force:
        return
    logger.remove()
    logger.add(sys.stderr, level=level, format=_FORMAT, filter=_with_stage,
               backtrace=False, diagnose=False)
    _configured = True


def add_file_sink(path: Path, level: str = "DEBUG") -> None:
    """Also log to ``path`` (append), so a detached or disconnected run leaves a record."""
    path = path.resolve()
    if path in _file_sinks:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(path), level=level, format=_FORMAT, filter=_with_stage,
               backtrace=False, diagnose=False, colorize=False, encoding="utf-8")
    _file_sinks.add(path)


def stage_logger(stage: str):
    setup_logging()
    return logger.bind(stage=stage)
