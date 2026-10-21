"""Non-blocking PerceptionFrame export (CLAUDE.md §8: the probe only copies metadata and
enqueues; serialization and file I/O happen on a writer thread)."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from src.common.logging import get_logger
from src.common.schemas import PerceptionFrame

log = get_logger(__name__)
_STOP = object()


class JsonlExporter:
    """One JSONL file per camera: <out_dir>/<cam_id>.jsonl."""

    def __init__(self, out_dir: str | Path, maxsize: int = 1000) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.written = 0
        self._files: dict[str, object] = {}
        self._t = threading.Thread(target=self._run, name="jsonl-writer", daemon=True)
        self._t.start()

    def put(self, frame: PerceptionFrame) -> None:
        """Called from the GStreamer probe: never blocks; drops when the writer falls behind."""
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is _STOP:
                break
            f = self._files.get(item.cam_id)
            if f is None:
                f = self._files[item.cam_id] = open(
                    self.out_dir / f"{item.cam_id}.jsonl", "a", encoding="utf-8"
                )
            f.write(item.model_dump_json() + "\n")
            self.written += 1
        for f in self._files.values():
            f.close()

    def close(self, timeout: float = 10.0) -> None:
        self.q.put(_STOP)
        self._t.join(timeout)
        if self.dropped:
            log.warning("exporter dropped %d frames (writer too slow)", self.dropped)
