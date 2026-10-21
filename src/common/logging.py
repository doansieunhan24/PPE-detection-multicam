"""Project-wide logging setup (no print() in src/)."""

from __future__ import annotations

import logging

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int | str = logging.INFO) -> None:
    logging.basicConfig(level=level, format=_FMT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
