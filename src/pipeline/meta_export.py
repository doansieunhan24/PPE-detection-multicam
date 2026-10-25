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


def letterbox_keypoints(
    data,
    n_values: int,
    net_w: int,
    net_h: int,
    mux_w: int,
    mux_h: int,
    sx: float = 1.0,
    sy: float = 1.0,
) -> list[tuple[float, float, float]]:
    """DeepStream-Yolo-Pose keypoints → original-camera pixels.

    The parser stores (x, y, conf) per joint in obj_meta.mask_params, in the *network input*
    (letterboxed, symmetric padding) space of size mask_params.width × height. Same mapping as
    the repo's deepstream.py parse_pose_from_meta (checked 2026-10-25, master): undo the
    letterbox to streammux pixels, then scale (sx, sy) to the camera's own resolution.
    """
    n = n_values // 3
    gain = min(net_w / mux_w, net_h / mux_h)
    pad_x = (net_w - mux_w * gain) * 0.5
    pad_y = (net_h - mux_h * gain) * 0.5
    return [
        (
            float((data[3 * k] - pad_x) / gain) * sx,
            float((data[3 * k + 1] - pad_y) / gain) * sy,
            float(data[3 * k + 2]),
        )
        for k in range(n)
    ]


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
