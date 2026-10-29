"""Snapshot with overlay for live alerts (CLAUDE.md §7.8 "kèm snapshot frame có OSD").

Smart-record clips are the cameras' raw streams (no OSD). Once an alert's clip has been written,
the frame at the alert time is read from it and the perception overlay of that instant (person
boxes, the alerted person highlighted, alert text) is drawn → <evidence_dir>/<alert_id>_<cam>.jpg,
and the alert's evidence in the SQLite store is updated.

Time of the alert inside the clip: a recording covers [trigger − pre_s, trigger + post_s]
(smart_record.py, measured on DS 7.1); an alert that arrives while its camera is still
recording shares that clip (EvidenceBook), so its offset is pre_s + (t_raise − t_trigger).
Clips start on a keyframe, so the offset is accurate to about one GOP (1 s in the demo).
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from src.common.logging import get_logger
from src.common.schemas import AlertEvent, PerceptionFrame

log = get_logger(__name__)


@dataclass
class _Job:
    alert: AlertEvent
    cam: str
    clip: Path
    offset_ms: int
    frame: PerceptionFrame | None
    gid_of: dict[int, int]  # track_id → global_id for this camera
    deadline: float


def clip_offset_ms(
    t_raise_ms: int, t_trigger_ms: int | None, pre_s: float, post_s: float
) -> tuple[int, int]:
    """(offset of t_raise inside the clip, trigger time of the clip it lands in)."""
    if t_trigger_ms is None or t_raise_ms - t_trigger_ms >= post_s * 1000:
        return int(pre_s * 1000), t_raise_ms  # starts a new recording
    return int(pre_s * 1000) + t_raise_ms - t_trigger_ms, t_trigger_ms


def draw_overlay(img, frame: PerceptionFrame | None, gid_of: dict[int, int], a: AlertEvent):
    import cv2

    if frame is not None:
        sx = img.shape[1] / frame.img_w
        sy = img.shape[0] / frame.img_h
        for p in frame.persons:
            hit = a.global_id is not None and gid_of.get(p.track_id) == a.global_id
            col = (0, 0, 255) if hit else (200, 200, 200)
            x1, y1, x2, y2 = (int(v * s) for v, s in zip(p.bbox, (sx, sy, sx, sy), strict=True))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if hit else 1)
            gid = gid_of.get(p.track_id)
            cv2.putText(img, f"g{gid}" if gid is not None else f"t{p.track_id}",
                        (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)  # fmt: skip
            for d in p.ppe_dets:
                bx = [int(v * s) for v, s in zip(d.bbox, (sx, sy, sx, sy), strict=True)]
                cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 200, 0), 1)
    text = f"L{a.level} {a.type.value} gid={a.global_id} zone={a.zone}"
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 14 * len(text)), 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img


def render_snapshot(job: _Job, out_jpg: Path) -> bool:
    import cv2

    cap = cv2.VideoCapture(str(job.clip))
    if not cap.isOpened():
        return False
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    idx = int(job.offset_ms / 1000 * fps)
    if n:
        idx = min(idx, int(n) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return False
    cv2.imwrite(str(out_jpg), draw_overlay(img, job.frame, job.gid_of, job.alert))
    return True


class SnapshotMaker:
    """Feed every processed group (add_group) and every recorded alert (request); a worker
    thread waits for each clip to be complete, then writes the snapshot."""

    def __init__(
        self,
        evidence_dir: Path,
        pre_s: float,
        post_s: float,
        store=None,
        buffer_s: float = 5.0,
        timeout_s: float | None = None,
        settle_s: float = 1.0,
    ) -> None:
        self.dir = Path(evidence_dir)
        self.pre_s, self.post_s = pre_s, post_s
        self.store = store
        self.timeout_s = timeout_s if timeout_s is not None else pre_s + post_s + 30
        self.settle_s = settle_s
        self.buffer_ms = int(buffer_s * 1000)
        self.frames: dict[str, deque] = {}
        self.trigger_ms: dict[str, int] = {}
        self.done: list[dict] = []
        self.q: queue.Queue[_Job | None] = queue.Queue()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def add_group(self, group: dict[str, PerceptionFrame], gid_of: dict) -> None:
        for cam, fr in group.items():
            dq = self.frames.setdefault(cam, deque())
            gids = {tid: g for (c, tid), g in gid_of.items() if c == cam}
            dq.append((fr, gids))
            while dq and fr.ts_ms - dq[0][0].ts_ms > self.buffer_ms:
                dq.popleft()

    def request(self, a: AlertEvent, clip_name: str) -> None:
        cam = a.cams[0]
        off, trig = clip_offset_ms(a.t_raise_ms, self.trigger_ms.get(cam), self.pre_s, self.post_s)
        self.trigger_ms[cam] = trig
        frame, gids = None, {}
        if self.frames.get(cam):
            frame, gids = min(self.frames[cam], key=lambda e: abs(e[0].ts_ms - a.t_raise_ms))
        self.q.put(_Job(a, cam, self.dir / clip_name, off, frame, gids,
                        time.monotonic() + self.timeout_s))  # fmt: skip

    def _ready(self, clip: Path) -> bool:
        """Clip exists and was not modified for settle_s (recording finished)."""
        try:
            return time.time() - clip.stat().st_mtime >= self.settle_s and clip.stat().st_size > 0
        except OSError:
            return False

    def _loop(self) -> None:
        pending: list[_Job] = []
        stop = False
        while not (stop and not pending):
            try:
                job = self.q.get(timeout=0.5)
                if job is None:
                    stop = True
                else:
                    pending.append(job)
            except queue.Empty:
                pass
            for job in list(pending):
                if self._ready(job.clip):
                    jpg = job.clip.with_suffix(".jpg")
                    ok = render_snapshot(job, jpg)
                    pending.remove(job)
                    self._finish(job, jpg if ok else None)
                elif time.monotonic() > job.deadline or stop and not job.clip.exists():
                    pending.remove(job)
                    self._finish(job, None)

    def _finish(self, job: _Job, jpg: Path | None) -> None:
        if jpg is None:
            log.warning("no snapshot for %s (clip %s)", job.alert.alert_id, job.clip.name)
        else:
            job.alert.evidence.snapshot = jpg.name
            if self.store is not None:
                self.store.set_evidence(job.alert.alert_id, job.alert.evidence)
        self.done.append({"alert_id": job.alert.alert_id, "snapshot": jpg.name if jpg else None})

    def close(self, wait_s: float | None = None) -> None:
        """Stop after the pending clips are done (or have timed out)."""
        self.q.put(None)
        self.th.join(wait_s)
