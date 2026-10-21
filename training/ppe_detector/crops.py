"""Pure helpers to turn full-image YOLO labels into person-crop labels (no I/O)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 px


@dataclass(frozen=True)
class Label:
    cls: int
    box: Box


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Box:
    return ((cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h)


def xyxy_to_yolo(box: Box, img_w: float, img_h: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h)


def parse_yolo_labels(lines: Sequence[str], img_w: int, img_h: int) -> list[Label]:
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
        out.append(Label(c, yolo_to_xyxy(cx, cy, w, h, img_w, img_h)))
    return out


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersect(a: Box, b: Box) -> Box:
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def pad_clip(box: Box, pad: float, img_w: int, img_h: int) -> Box:
    """Same padding rule as src.perception_offline.convert.pad_clip_bbox, rounded to int px."""
    x1, y1, x2, y2 = box
    dw, dh = (x2 - x1) * pad, (y2 - y1) * pad
    return (
        float(round(max(0.0, x1 - dw))),
        float(round(max(0.0, y1 - dh))),
        float(round(min(float(img_w), x2 + dw))),
        float(round(min(float(img_h), y2 + dh))),
    )


def crop_labels(
    crop: Box, labels: Sequence[Label], cls_map: dict[int, int], min_keep_ratio: float
) -> list[Label]:
    """Labels of target classes inside `crop`, clipped and shifted to crop coordinates.

    cls_map: source class id -> target class id (others dropped).
    A box is kept if (area inside crop) / (box area) >= min_keep_ratio.
    """
    out = []
    for lb in labels:
        if lb.cls not in cls_map:
            continue
        a = area(lb.box)
        if a <= 0:
            continue
        inter = intersect(lb.box, crop)
        if area(inter) / a < min_keep_ratio:
            continue
        ox, oy = crop[0], crop[1]
        out.append(
            Label(cls_map[lb.cls], (inter[0] - ox, inter[1] - oy, inter[2] - ox, inter[3] - oy))
        )
    return out
