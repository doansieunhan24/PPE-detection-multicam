"""Per-(camera, person, part) visibility v (CLAUDE.md §7.4).

v = s_kp * s_size * (1 - r_occ) * s_blur * s_fallback      clipped to [0, 1]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.common.schemas import BBox
from src.parts.part_rois import PartRoi


@dataclass(frozen=True)
class VisibilityParams:
    h_ref_px: float = 32.0
    use_blur: bool = False
    blur_beta: float = 100.0
    fallback_penalty: float = 0.5
    occ_grid: int = 16  # r_occ estimated on an occ_grid x occ_grid sample grid

    @classmethod
    def from_config(cls, cfg: dict) -> VisibilityParams:
        v = cfg["visibility"]
        return cls(
            h_ref_px=v["h_ref_px"],
            use_blur=v.get("use_blur", False),
            blur_beta=v.get("blur_beta", 100.0),
            fallback_penalty=cfg["part_assign"]["fallback_penalty"],
        )


@dataclass(frozen=True)
class VisibilityTerms:
    s_kp: float
    s_size: float
    r_occ: float
    s_blur: float
    s_fallback: float

    @property
    def v(self) -> float:
        val = self.s_kp * self.s_size * (1 - self.r_occ) * self.s_blur * self.s_fallback
        return float(min(1.0, max(0.0, val)))


def occlusion_ratio(roi: BBox, occluders: Sequence[BBox], img_w: int, img_h: int, n: int) -> float:
    """Fraction of ROI covered by occluders or outside the image (grid sampling)."""
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return 1.0
    xs = x1 + (np.arange(n) + 0.5) * (x2 - x1) / n
    ys = y1 + (np.arange(n) + 0.5) * (y2 - y1) / n
    gx, gy = np.meshgrid(xs, ys)
    hidden = (gx < 0) | (gx >= img_w) | (gy < 0) | (gy >= img_h)
    for ox1, oy1, ox2, oy2 in occluders:
        hidden |= (gx >= ox1) & (gx < ox2) & (gy >= oy1) & (gy < oy2)
    return float(hidden.mean())


def closer_persons(person: BBox, others: Sequence[BBox]) -> list[BBox]:
    """Persons nearer the camera: foot point lower in the image (larger y2)."""
    return [b for b in others if b[3] > person[3]]


def blur_score(roi_img: np.ndarray | None, beta: float) -> float:
    if roi_img is None or roi_img.size == 0:
        return 1.0
    import cv2

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if roi_img.ndim == 3 else roi_img
    return float(min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / beta))


def visibility(
    roi: PartRoi,
    person_bbox: BBox,
    other_bboxes: Sequence[BBox],
    img_w: int,
    img_h: int,
    prm: VisibilityParams = VisibilityParams(),
    roi_img: np.ndarray | None = None,
) -> VisibilityTerms:
    h_roi = roi.box[3] - roi.box[1]
    occ = closer_persons(person_bbox, other_bboxes)
    return VisibilityTerms(
        s_kp=roi.s_kp,
        s_size=min(1.0, max(0.0, h_roi) / prm.h_ref_px),
        r_occ=occlusion_ratio(roi.box, occ, img_w, img_h, prm.occ_grid),
        s_blur=blur_score(roi_img, prm.blur_beta) if prm.use_blur else 1.0,
        s_fallback=prm.fallback_penalty if roi.fallback else 1.0,
    )
