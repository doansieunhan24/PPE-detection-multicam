"""Checkerboard floor tiles as calibration targets (CLAUDE.md §7.1, P3).

The C2 rooms have a black/white tiled floor: every tile corner is an X-corner with known floor
coordinates (tile index × tile size). From 4 clicked corners of one tile the grid is grown to
its neighbours: each candidate is predicted with a homography fitted on the nearest accepted
corners (handles perspective; lens distortion is small locally), refined with cornerSubPix and
kept only if it is a saddle of the expected colour parity. The corners then give the lens
distortion (fit_radial_distortion) and the image→floor
homography on undistorted pixels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

Index = tuple[int, int]
SEED_ORDER: tuple[Index, ...] = ((0, 0), (1, 0), (1, 1), (0, 1))  # clicked tile corners


def _patch_mean(gray: np.ndarray, p: np.ndarray, r: int) -> float | None:
    x, y = int(round(p[0])), int(round(p[1]))
    h, w = gray.shape
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return None
    return float(np.median(gray[y - r : y + r + 1, x - r : x + r + 1]))


def saddle_contrast(
    gray: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    fracs: Sequence[float] = (0.3, 0.15),
    max_pair_diff: float = 0.5,
) -> float | None:
    """mean(tiles (i,j), (i-1,j-1)) − mean(tiles (i,j-1), (i-1,j)) around corner p.

    u, v = image vectors of one tile step along i and j. Sign alternates with (i + j) parity;
    magnitude is the black/white contrast. Samples at frac × (±u ±v), retried closer to p
    near the image border. None if outside the image, or if diagonally opposite tiles differ
    by more than max_pair_diff × contrast (checkerboard: they have the same colour; a wall,
    door or cabinet next to the floor does not).
    """
    r = max(1, int(0.08 * min(np.linalg.norm(u), np.linalg.norm(v))))
    for frac in fracs:
        s = [
            _patch_mean(gray, p + frac * (a * u + b * v), r)
            for a, b in ((1, 1), (-1, -1), (1, -1), (-1, 1))
        ]
        if all(x is not None for x in s):
            break
    else:
        return None
    c = (s[0] + s[1] - s[2] - s[3]) / 2.0
    if max(abs(s[0] - s[1]), abs(s[2] - s[3])) > max_pair_diff * abs(c):
        return None
    return c


def _local_h(found: Mapping[Index, np.ndarray], ij: Index, k: int = 8) -> np.ndarray | None:
    """Homography index→px from the k accepted corners nearest to ij in index space."""
    import cv2

    keys = sorted(found, key=lambda q: (q[0] - ij[0]) ** 2 + (q[1] - ij[1]) ** 2)[:k]
    if len(keys) < 4:
        return None
    src = np.array(keys, dtype=np.float64)
    if np.linalg.matrix_rank(src - src.mean(0)) < 2:
        return None
    H, _ = cv2.findHomography(src, np.array([found[q] for q in keys], dtype=np.float64), 0)
    return H


def _proj(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def refine_corner(gray: np.ndarray, p: np.ndarray, half_win: int) -> np.ndarray:
    import cv2

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    pt = np.asarray(p, dtype=np.float32).reshape(1, 1, 2).copy()
    cv2.cornerSubPix(gray, pt, (half_win, half_win), (-1, -1), crit)
    return pt.reshape(2).astype(np.float64)


def grow_grid(
    gray: np.ndarray,
    seeds_px: Sequence[Sequence[float]],
    min_contrast: float = 25.0,
    max_shift: float = 0.25,
    min_step_px: float = 12.0,
    margin_px: int = 4,
    max_corners: int = 2000,
) -> dict[Index, np.ndarray]:
    """Grow the tile-corner grid from 4 corners of one tile, clicked in SEED_ORDER.

    gray: uint8 image. Seeds are refined too. A candidate is kept if cornerSubPix moves it less
    than max_shift × tile step from the prediction and its saddle contrast has the expected
    sign and |contrast| >= min_contrast (grey levels). Candidates whose tile step is below
    min_step_px (far away) are not tried.
    """
    if len(seeds_px) != 4:
        raise ValueError("need the 4 corners of one tile")
    seeds = np.asarray(seeds_px, dtype=np.float64)
    step0 = float(np.mean(np.linalg.norm(np.roll(seeds, -1, 0) - seeds, axis=1)))
    found: dict[Index, np.ndarray] = {}
    for ij, p in zip(SEED_ORDER, seeds, strict=True):
        found[ij] = refine_corner(gray, p, max(3, min(15, int(0.2 * step0))))

    def step_vectors(H: np.ndarray, ij: Index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i, j = ij
        q = _proj(H, np.array([[i, j], [i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]], float))
        return q[0], (q[1] - q[2]) / 2.0, (q[3] - q[4]) / 2.0

    H0 = _local_h(found, (0, 0))
    if H0 is None:
        raise ValueError("degenerate seed tile")
    sign0 = 0.0
    for k, ij in enumerate(SEED_ORDER):
        _, u, v = step_vectors(H0, ij)
        c = saddle_contrast(gray, found[ij], u, v)
        if c is None or abs(c) < min_contrast:
            raise ValueError(f"seed {k + 1} is not a clear tile corner (contrast {c})")
        if k == 0:
            sign0 = float(np.sign(c))
        elif np.sign(c) != sign0 * (-1) ** ((ij[0] + ij[1]) % 2):
            # e.g. corners of a 2×2 block clicked: all four have the same colour pattern
            raise ValueError("seeds are not the 4 corners of one tile, in order around it")

    h, w = gray.shape
    tried: set[Index] = set(found)
    frontier = list(found)
    while frontier and len(found) < max_corners:
        cands: set[Index] = set()
        for i, j in frontier:
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (i + d[0], j + d[1])
                if q not in tried:
                    cands.add(q)
        frontier = []
        for ij in sorted(cands):
            tried.add(ij)
            H = _local_h(found, ij)
            if H is None:
                continue
            pred, u, v = step_vectors(H, ij)
            step = min(np.linalg.norm(u), np.linalg.norm(v))
            if not np.all(np.isfinite(pred)) or step < min_step_px:
                continue
            if not (margin_px <= pred[0] < w - margin_px and margin_px <= pred[1] < h - margin_px):
                continue
            p = refine_corner(gray, pred, max(3, min(15, int(0.2 * step))))
            if np.linalg.norm(p - pred) > max_shift * step:
                continue
            c = saddle_contrast(gray, p, u, v)
            want = sign0 * (-1) ** ((ij[0] + ij[1]) % 2)
            if c is None or c * want < min_contrast:
                continue
            found[ij] = p
            frontier.append(ij)
    return found


def nominal_K(image_size: tuple[int, int], f_ratio: float = 0.6) -> np.ndarray:
    """Pinhole K with principal point at the image centre and f = f_ratio × width.

    From one floor plane only the image-space curvature is observable, not f and k1
    separately (k1 scales with f²): the undistorted pixels, and so the floor mapping, are
    the same for any f within a wide range (checked on cam2: f 0.5w/0.6w/0.8w give the same
    residual). f is therefore fixed, not estimated.
    """
    w, h = image_size
    return np.array([[f_ratio * w, 0, (w - 1) / 2], [0, f_ratio * w, (h - 1) / 2], [0, 0, 1.0]])


def plane_residual_px(
    floor_xy: np.ndarray, img_px: np.ndarray, K: np.ndarray | None, dist: np.ndarray | None
) -> np.ndarray:
    """Per-point error (undistorted px) of the best homography floor → undistorted image."""
    import cv2

    u = undistort_px(img_px, K, dist)
    H, _ = cv2.findHomography(np.asarray(floor_xy, np.float64), u, 0)
    return np.linalg.norm(_proj(H, np.asarray(floor_xy, np.float64)) - u, axis=1)


def fit_radial_distortion(
    views: Sequence[tuple[np.ndarray, np.ndarray]], K: np.ndarray, n_coeffs: int = 2
) -> tuple[np.ndarray, dict[str, object]]:
    """Radial distortion (k1[, k2]) that makes planar grid points fit a homography.

    views: (floor/tile coordinates (N,2), distorted image points (N,2)) per camera sharing the
    lens. Returns OpenCV dist (k1, k2, 0, 0, 0) and per-view RMS px before/after.
    """
    from scipy.optimize import least_squares

    def dist_of(k: np.ndarray) -> np.ndarray:
        d = np.zeros(5)
        d[:n_coeffs] = k
        return d

    def resid(k: np.ndarray) -> np.ndarray:
        return np.concatenate([plane_residual_px(f, i, K, dist_of(k)) for f, i in views])

    sol = least_squares(resid, np.zeros(n_coeffs))
    dist = dist_of(sol.x)
    rms = lambda r: float(np.sqrt(np.mean(r**2)))  # noqa: E731
    info = {
        "rms_px_before": [rms(plane_residual_px(f, i, None, None)) for f, i in views],
        "rms_px_after": [rms(plane_residual_px(f, i, K, dist)) for f, i in views],
        "n_points": [len(f) for f, _ in views],
    }
    return dist, info


def distort_px(pts_px: np.ndarray, K: np.ndarray | None, dist: np.ndarray | None) -> np.ndarray:
    """Inverse of undistort_px: ideal pixels → pixels as the lens images them."""
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    if K is None or dist is None:
        return pts
    import cv2

    norm = (np.hstack([pts, np.ones((len(pts), 1))]) @ np.linalg.inv(K).T)[:, :2]
    obj = np.c_[norm, np.ones(len(norm))]
    out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, np.asarray(dist, np.float64))
    return out.reshape(-1, 2)


def undistort_px(pts_px: np.ndarray, K: np.ndarray | None, dist: np.ndarray | None) -> np.ndarray:
    """Distorted image pixels → ideal (pinhole) pixels with the same K. Identity if K is None."""
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    if K is None or dist is None:
        return pts
    import cv2

    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-6)
    out = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, R=None, P=K, criteria=crit)
    return out.reshape(-1, 2)
