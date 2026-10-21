"""Skeleton/bbox features for fall and immobility rules (CLAUDE.md §7.7)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.common.schemas import BBox, Keypoint
from src.parts.part_rois import L_HIP, L_SHO, R_HIP, R_SHO


def aspect_ratio(b: BBox) -> float:
    """bbox width / height (a lying person is wider than tall)."""
    h = b[3] - b[1]
    return (b[2] - b[0]) / h if h > 0 else 0.0


def _mid(kpts: Sequence[Keypoint], a: int, b: int, cmin: float) -> tuple[float, float] | None:
    if kpts[a][2] < cmin or kpts[b][2] < cmin:
        return None
    return ((kpts[a][0] + kpts[b][0]) / 2, (kpts[a][1] + kpts[b][1]) / 2)


def torso_angle_deg(kpts: Sequence[Keypoint], cmin: float = 0.3) -> float | None:
    """Angle between shoulder-mid → hip-mid and the image vertical (0 = upright, 90 = lying)."""
    if not kpts:
        return None
    s, h = _mid(kpts, L_SHO, R_SHO, cmin), _mid(kpts, L_HIP, R_HIP, cmin)
    if s is None or h is None:
        return None
    dx, dy = h[0] - s[0], h[1] - s[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def hip_y(kpts: Sequence[Keypoint], cmin: float = 0.3) -> float | None:
    if not kpts:
        return None
    h = _mid(kpts, L_HIP, R_HIP, cmin)
    return h[1] if h else None


def mean_kpt_displacement(
    a: Sequence[Keypoint], b: Sequence[Keypoint], cmin: float = 0.3
) -> float | None:
    """Mean displacement (px) of keypoints confidently seen in both skeletons."""
    if not a or not b:
        return None
    d = [
        math.hypot(p[0] - q[0], p[1] - q[1])
        for p, q in zip(a, b, strict=True)
        if p[2] >= cmin and q[2] >= cmin
    ]
    return sum(d) / len(d) if d else None


def body_visibility(kpts: Sequence[Keypoint], bbox: BBox, h_ref_px: float) -> float:
    """Whole-body v for multi-view fall fusion: mean kpt conf × size factor."""
    if not kpts:
        return 0.0
    s_kp = sum(k[2] for k in kpts) / len(kpts)
    return s_kp * min(1.0, (bbox[3] - bbox[1]) / h_ref_px)
