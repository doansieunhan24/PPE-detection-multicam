"""Pure helpers for the offline backend: bbox geometry, timestamps, array → schema.

Kept free of Ultralytics imports so they can be unit-tested without a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.common.schemas import NUM_COCO_KPTS, BBox, PersonDet, PPEDet


def is_live_uri(uri: str) -> bool:
    return uri.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))


def file_ts_ms(frame_idx: int, fps: float, start_ms: int, offset_ms: int) -> int:
    """Timestamp for a frame of a video file: session start + frame time + cam sync offset."""
    return start_ms + round(frame_idx * 1000.0 / fps) + offset_ms


class FrameClock:
    """Media time (ms from file start) of successive frames of a video file.

    mode="pts": container timestamp (cv2 CAP_PROP_POS_MSEC after read). Phone recordings are
    often variable frame rate, so frame_idx/fps drifts by seconds over a 10-min session.
    A missing or non-increasing pts (some backends report 0) falls back to last + 1000/fps.
    mode="index": frame_idx * 1000 / fps (constant frame rate assumption).
    """

    def __init__(self, fps: float, mode: str = "pts") -> None:
        if mode not in ("pts", "index"):
            raise ValueError(f"unknown timestamp mode {mode!r}")
        self.fps, self.mode = fps, mode
        self.prev_ms: float | None = None
        self.n = 0
        self.n_fallback = 0

    def next(self, pos_msec: float | None) -> int:
        step = 1000.0 / self.fps
        if self.mode == "index":
            t = self.n * step
        elif pos_msec is not None and (self.prev_ms is None or pos_msec > self.prev_ms):
            t = float(pos_msec)
        else:
            t = 0.0 if self.prev_ms is None else self.prev_ms + step
            self.n_fallback += self.n > 0
        self.prev_ms = t
        self.n += 1
        return round(t)


def pad_clip_bbox(bbox: Sequence[float], pad: float, img_w: int, img_h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    dw, dh = (x2 - x1) * pad, (y2 - y1) * pad
    return (
        max(0.0, x1 - dw),
        max(0.0, y1 - dh),
        min(float(img_w), x2 + dw),
        min(float(img_h), y2 + dh),
    )


def crop_box_to_image(box: Sequence[float], crop_origin: Sequence[float]) -> BBox:
    """Map a bbox from crop coordinates back to original-image coordinates."""
    ox, oy = crop_origin[0], crop_origin[1]
    return (box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy)


def is_truncated_bottom(bbox: Sequence[float], img_h: int, margin_px: float) -> bool:
    return bbox[3] >= img_h - margin_px


def persons_from_arrays(
    xyxy: np.ndarray,
    conf: np.ndarray,
    track_ids: np.ndarray | None,
    kpts: np.ndarray | None,
    img_h: int,
    truncated_margin_px: float,
) -> list[PersonDet]:
    """Build PersonDet list from pose-tracker outputs.

    xyxy (N,4), conf (N,), track_ids (N,) or None, kpts (N,17,3) or None.
    Detections without a track ID (tracker not yet confirmed) are dropped.
    """
    if track_ids is None:
        return []
    persons: list[PersonDet] = []
    for i in range(len(xyxy)):
        bbox = tuple(float(v) for v in xyxy[i])
        k: list[tuple[float, float, float]] = []
        if kpts is not None and kpts.shape[1] == NUM_COCO_KPTS:
            k = [(float(x), float(y), float(c)) for x, y, c in kpts[i]]
        persons.append(
            PersonDet(
                track_id=int(track_ids[i]),
                bbox=bbox,
                conf=float(conf[i]),
                kpts=k,
                truncated_bottom=is_truncated_bottom(bbox, img_h, truncated_margin_px),
            )
        )
    return persons


def ppe_from_arrays(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls_idx: np.ndarray,
    names: Mapping[int, str],
    class_map: Mapping[str, str],
    crop_origin: Sequence[float],
) -> list[PPEDet]:
    """Build PPEDet list from detector outputs on a crop, in original-image coords."""
    dets: list[PPEDet] = []
    for i in range(len(xyxy)):
        raw = names[int(cls_idx[i])]
        dets.append(
            PPEDet(
                cls=class_map.get(raw, raw),
                bbox=crop_box_to_image([float(v) for v in xyxy[i]], crop_origin),
                conf=float(conf[i]),
            )
        )
    return dets
