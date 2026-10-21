"""Image(px) → floor(m) homography, calibration I/O, polygon tests (CLAUDE.md §7.1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.common.config import load_yaml

Point = tuple[float, float]


@dataclass
class CameraCalibration:
    cam_id: str
    H: np.ndarray | None  # 3x3 image(px) -> floor(m)
    reprojection_error_m: float | None = None
    fov_floor_polygon_m: list[Point] = field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        return self.H is not None

    def to_floor(self, pts_px: np.ndarray) -> np.ndarray:
        if self.H is None:
            raise ValueError(f"{self.cam_id}: not calibrated")
        return apply_homography(self.H, pts_px)

    def sees(self, pt_m: Point) -> bool:
        """Inside this camera's floor FOV (no polygon set → assume yes)."""
        return not self.fov_floor_polygon_m or point_in_polygon(pt_m, self.fov_floor_polygon_m)


def load_calibration(path: str | Path) -> CameraCalibration:
    d = load_yaml(path)
    H = np.asarray(d["H"], dtype=np.float64) if d.get("H") is not None else None
    return CameraCalibration(
        cam_id=d["cam_id"],
        H=H,
        reprojection_error_m=d.get("reprojection_error_m"),
        fov_floor_polygon_m=[tuple(p) for p in d.get("fov_floor_polygon_m") or []],
    )


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts (N,2) → (N,2) via projective transform."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def fit_homography(img_pts: np.ndarray, floor_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """RANSAC homography from >= 4 correspondences; returns (H, RMS reprojection error in m)."""
    import cv2

    img_pts = np.asarray(img_pts, dtype=np.float64)
    floor_pts = np.asarray(floor_pts, dtype=np.float64)
    if len(img_pts) < 4:
        raise ValueError("need >= 4 point pairs")
    H, _ = cv2.findHomography(img_pts, floor_pts, cv2.RANSAC, 0.1)
    if H is None:
        raise ValueError("homography estimation failed")
    err = np.linalg.norm(apply_homography(H, img_pts) - floor_pts, axis=1)
    return H, float(np.sqrt(np.mean(err**2)))


def point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool:
    """Ray casting; points on the boundary may go either way."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside
