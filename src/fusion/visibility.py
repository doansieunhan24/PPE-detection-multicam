"""Per-(camera, person, part) visibility v (CLAUDE.md §7.4).

v = s_kp * s_size * (1 - r_occ) * s_blur * s_fallback      clipped to [0, 1]
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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
    # ablation v_learned: per part {"w": [...], "b": float} over LEARNED_FEATURES
    learned: Mapping[str, Mapping] | None = None

    @classmethod
    def from_config(cls, cfg: dict, learned: bool | None = None) -> VisibilityParams:
        v = cfg["visibility"]
        use_learned = v.get("learned", False) if learned is None else learned
        return cls(
            h_ref_px=v["h_ref_px"],
            use_blur=v.get("use_blur", False),
            blur_beta=v.get("blur_beta", 100.0),
            fallback_penalty=cfg["part_assign"]["fallback_penalty"],
            learned=load_learned(v["learned_path"]) if use_learned else None,
        )


LEARNED_FEATURES = ("s_kp", "s_size", "vis_occ", "fallback", "log_h_roi")


def load_learned(path: str | Path) -> dict[str, dict]:
    """Learned visibility per part, written by training/visibility/calibrate.py."""
    import yaml

    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if tuple(d["features"]) != LEARNED_FEATURES:
        raise ValueError(f"{path}: features {d['features']} != {LEARNED_FEATURES}")
    return d["parts"]


@dataclass(frozen=True)
class VisibilityTerms:
    s_kp: float
    s_size: float
    r_occ: float
    s_blur: float
    s_fallback: float
    v_learned: float | None = None

    @property
    def v(self) -> float:
        if self.v_learned is not None:
            return self.v_learned
        val = self.s_kp * self.s_size * (1 - self.r_occ) * self.s_blur * self.s_fallback
        return float(min(1.0, max(0.0, val)))


def occlusion_ratio(roi: BBox, occluders: Sequence[BBox], img_w: int, img_h: int, n: int) -> float:
    """Fraction of ROI covered by occluders or outside the image (grid sampling)."""
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return 1.0
    xs = x1 + (np.arange(n) + 0.5) * (x2 - x1) / n
    ys = y1 + (np.arange(n) + 0.5) * (y2 - y1) / n
    x_out = (xs < 0) | (xs >= img_w)
    y_out = (ys < 0) | (ys >= img_h)
    # occluders that overlap the ROI at all (most calls have none: fast path)
    occ = [o for o in occluders if o[0] < x2 and o[2] > x1 and o[1] < y2 and o[3] > y1]
    if not occ and not x_out.any() and not y_out.any():
        return 0.0
    # separable grid: cell (row y, col x) hidden by an occluder iff y in its rows and x in its cols
    hidden = y_out[:, None] | x_out[None, :]
    for ox1, oy1, ox2, oy2 in occ:
        hidden = hidden | (
            ((ys >= oy1) & (ys < oy2))[:, None] & ((xs >= ox1) & (xs < ox2))[None, :]
        )
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
    s_size = min(1.0, max(0.0, h_roi) / prm.h_ref_px)
    r_occ = occlusion_ratio(roi.box, occ, img_w, img_h, prm.occ_grid)
    v_learned = None
    if prm.learned is not None:
        lp = prm.learned[roi.part.value]
        x = (roi.s_kp, s_size, 1.0 - r_occ, float(roi.fallback), math.log(max(h_roi, 1.0)))
        z = sum(wi * xi for wi, xi in zip(lp["w"], x, strict=True)) + lp["b"]
        v_learned = 1.0 / (1.0 + math.exp(-z))
    return VisibilityTerms(
        s_kp=roi.s_kp,
        s_size=s_size,
        r_occ=r_occ,
        s_blur=blur_score(roi_img, prm.blur_beta) if prm.use_blur else 1.0,
        s_fallback=prm.fallback_penalty if roi.fallback else 1.0,
        v_learned=v_learned,
    )
