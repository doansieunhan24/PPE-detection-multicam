"""Project-wide logging setup (no print() in src/)."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int | str = logging.INFO) -> None:
    # Windows consoles default to cp1252, which cannot encode →, ≥, Vietnamese text, ...
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    logging.basicConfig(level=level, format=_FMT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
