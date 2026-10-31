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
    H: np.ndarray | None  # 3x3 undistorted image(px) -> floor(m)
    reprojection_error_m: float | None = None
    fov_floor_polygon_m: list[Point] = field(default_factory=list)
    # lens: pixels are undistorted with (K, dist) before H; None = pinhole (H on raw pixels)
    K: np.ndarray | None = None
    dist: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        return self.H is not None

    def to_floor(self, pts_px: np.ndarray) -> np.ndarray:
        if self.H is None:
            raise ValueError(f"{self.cam_id}: not calibrated")
        from src.geometry.floor_grid import undistort_px

        return apply_homography(self.H, undistort_px(pts_px, self.K, self.dist))

    def sees(self, pt_m: Point, margin_m: float = 0.0) -> bool:
        """Inside this camera's floor FOV, or within margin_m of its border (people cut by the
        image edge still have a foot point just outside it). No polygon set → assume yes."""
        poly = self.fov_floor_polygon_m
        if not poly or point_in_polygon(pt_m, poly):
            return True
        return margin_m > 0 and polygon_distance(pt_m, poly) <= margin_m


def load_calibration(path: str | Path) -> CameraCalibration:
    d = load_yaml(path)

    def arr(key: str) -> np.ndarray | None:
        return np.asarray(d[key], dtype=np.float64) if d.get(key) is not None else None

    return CameraCalibration(
        cam_id=d["cam_id"],
        H=arr("H"),
        reprojection_error_m=d.get("reprojection_error_m"),
        fov_floor_polygon_m=[tuple(p) for p in d.get("fov_floor_polygon_m") or []],
        K=arr("K"),
        dist=arr("dist"),
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


def loo_errors(img_pts: np.ndarray, floor_pts: np.ndarray) -> np.ndarray:
    """Leave-one-out floor error (m) per point: fit on the others, predict the held-out one.

    In-sample RMS is optimistic with few points (a homography has 8 DOF); this is the number
    to report. Needs >= 5 points; NaN where the reduced fit fails.
    """
    img_pts = np.asarray(img_pts, dtype=np.float64)
    floor_pts = np.asarray(floor_pts, dtype=np.float64)
    if len(img_pts) < 5:
        raise ValueError("need >= 5 point pairs for leave-one-out")
    out = np.full(len(img_pts), np.nan)
    for i in range(len(img_pts)):
        keep = np.arange(len(img_pts)) != i
        try:
            H, _ = fit_homography(img_pts[keep], floor_pts[keep])
        except ValueError:
            continue
        out[i] = float(np.linalg.norm(apply_homography(H, img_pts[i : i + 1])[0] - floor_pts[i]))
    return out


def floor_fov_polygon(
    H: np.ndarray,
    img_w: int,
    img_h: int,
    ref_img_pts: np.ndarray,
    max_dist_m: float,
    top_ratio: float = 0.0,
    n_samples: int = 64,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
) -> list[list[float]]:
    """Floor footprint of the image (convex hull of mapped border samples).

    Border points above the horizon map behind the camera (homogeneous w of the other sign
    than the floor markers ref_img_pts) and are dropped, as are points farther than
    max_dist_m from the floor point of the bottom-centre pixel. Matters for low cameras,
    whose horizon is near the middle of the image. With (K, dist), H is on undistorted pixels:
    the border and ref_img_pts (raw pixels) are undistorted first.
    """
    import cv2

    from src.geometry.floor_grid import undistort_px

    top = img_h * top_ratio
    t = np.linspace(0.0, 1.0, n_samples)
    border = np.vstack(
        [
            np.c_[t * (img_w - 1), np.full_like(t, img_h - 1)],  # bottom
            np.c_[np.full_like(t, img_w - 1), img_h - 1 - t * (img_h - 1 - top)],  # right
            np.c_[(1 - t) * (img_w - 1), np.full_like(t, top)],  # top
            np.c_[np.zeros_like(t), top + t * (img_h - 1 - top)],  # left
        ]
    )
    border = undistort_px(border, K, dist)
    ref_img_pts = undistort_px(np.asarray(ref_img_pts, dtype=np.float64).reshape(-1, 2), K, dist)

    def w_of(pts: np.ndarray) -> np.ndarray:
        return (np.hstack([pts, np.ones((len(pts), 1))]) @ H.T)[:, 2]

    sign = np.sign(np.mean(w_of(np.asarray(ref_img_pts, dtype=np.float64).reshape(-1, 2))))
    ok = w_of(border) * sign > 1e-9
    fl = apply_homography(H, border[ok])
    centre = apply_homography(H, undistort_px(np.array([[img_w / 2, img_h - 1]]), K, dist))[0]
    fl = fl[np.linalg.norm(fl - centre, axis=1) <= max_dist_m]
    if len(fl) < 3:
        return []
    hull = cv2.convexHull(fl.astype(np.float32)).reshape(-1, 2)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in hull]


def polygon_distance(pt: Point, poly: Sequence[Point]) -> float:
    """Distance from pt to the polygon boundary."""
    p = np.asarray(pt, dtype=np.float64)
    a = np.asarray(poly, dtype=np.float64)
    b = np.roll(a, -1, axis=0)
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab**2).sum(1), 1e-12), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))


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
