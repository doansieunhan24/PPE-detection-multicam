"""Checkerboard floor calibration on a synthetic wide-angle view (src/geometry/floor_grid.py)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.geometry.floor_grid import (  # noqa: E402
    distort_px,
    fit_radial_distortion,
    grow_grid,
    nominal_K,
    undistort_px,
)
from src.geometry.homography import CameraCalibration, apply_homography, fit_homography  # noqa: E402

W, H_IMG = 960, 540
K_TRUE = nominal_K((W, H_IMG), 0.6)
DIST_TRUE = np.array([-0.15, 0.03, 0, 0, 0])
# floor (tiles) -> ideal pixels: camera looking down at the floor at an angle
H_FLOOR_TO_IDEAL = np.array([[60.0, -25.0, 480.0], [8.0, 18.0, 300.0], [0.0, -0.035, 1.0]])


def render() -> np.ndarray:
    """Raw (distorted) image of a black/white tile floor: raw px -> ideal px -> floor."""
    xs, ys = np.meshgrid(np.arange(W, dtype=float), np.arange(H_IMG, dtype=float))
    ideal = undistort_px(np.c_[xs.ravel(), ys.ravel()], K_TRUE, DIST_TRUE)
    hom = np.c_[ideal, np.ones(len(ideal))] @ np.linalg.inv(H_FLOOR_TO_IDEAL).T
    fl = hom[:, :2] / hom[:, 2:3]
    ok = hom[:, 2] > 0
    tile = (np.floor(fl[:, 0]) + np.floor(fl[:, 1])) % 2
    img = np.where(ok, np.where(tile == 0, 210, 40), 128).astype(np.uint8).reshape(H_IMG, W)
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def floor_to_raw(fl: np.ndarray) -> np.ndarray:
    return distort_px(apply_homography(H_FLOOR_TO_IDEAL, fl), K_TRUE, DIST_TRUE)


def test_distort_undistort_roundtrip():
    pts = np.array([[10.0, 20.0], [480.0, 270.0], [900.0, 500.0]])
    back = undistort_px(distort_px(pts, K_TRUE, DIST_TRUE), K_TRUE, DIST_TRUE)
    assert np.allclose(back, pts, atol=1e-3)
    assert np.allclose(undistort_px(pts, None, None), pts)


def test_grow_grid_finds_tile_corners():
    img = render()
    seeds = floor_to_raw(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float))
    found = grow_grid(img, seeds + np.array([2.0, -2.0]))  # clicks a few px off
    assert len(found) >= 20
    ij = np.array(list(found), float)
    err = np.linalg.norm(np.array(list(found.values())) - floor_to_raw(ij), axis=1)
    assert np.median(err) < 0.5 and err.max() < 1.5


def test_grow_grid_rejects_wrong_seed_order():
    img = render()
    seeds = floor_to_raw(np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float))  # 2×2 tiles
    with pytest.raises(ValueError):
        grow_grid(img, seeds)


def test_radial_fit_makes_floor_mapping_accurate():
    fl = np.array([(i, j) for i in range(-3, 4) for j in range(-2, 4)], float)
    raw = floor_to_raw(fl)
    keep = (raw[:, 0] > 0) & (raw[:, 0] < W) & (raw[:, 1] > 0) & (raw[:, 1] < H_IMG)
    fl, raw = fl[keep], raw[keep]
    K = nominal_K((W, H_IMG), 0.8)  # wrong f on purpose: only the curvature is observable
    dist, info = fit_radial_distortion([(fl, raw)], K)
    assert info["rms_px_after"][0] < 0.05 < 0.5 < info["rms_px_before"][0]
    Hm, _ = fit_homography(undistort_px(raw, K, dist), fl)
    cal = CameraCalibration("c", Hm, K=K, dist=dist)
    probe = np.array([[0.5, 0.5], [-2.5, 2.5], [2.5, -1.5]])
    assert np.allclose(cal.to_floor(floor_to_raw(probe)), probe, atol=0.01)
