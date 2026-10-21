"""Read/write PerceptionFrame JSONL (CLAUDE.md D4: metadata-first)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from src.common.schemas import PerceptionFrame


def read_frames(path: str | Path) -> Iterator[PerceptionFrame]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield PerceptionFrame.model_validate_json(line)


def write_frames(path: str | Path, frames: Iterable[PerceptionFrame]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(fr.model_dump_json() + "\n")
            n += 1
    return n
