import numpy as np
import pytest

from src.geometry.homography import apply_homography, floor_fov_polygon, loo_errors
from tools.calibrate_homography import read_points

IMG_W, IMG_H = 1920, 1080


def low_camera_h(height_m=1.5, tilt_deg=10.0, f=1000.0):
    """Image → floor homography of a camera at (0,0,height) looking along +y, tilted down."""
    K = np.array([[f, 0, IMG_W / 2], [0, f, IMG_H / 2], [0, 0, 1]])
    a = np.deg2rad(tilt_deg)
    # world: x right, y forward, z up; camera: x right, y down, z forward
    R = np.array([[1, 0, 0], [0, -np.sin(a), -np.cos(a)], [0, np.cos(a), -np.sin(a)]])
    t = -R @ np.array([0.0, 0.0, height_m])
    G = K @ np.c_[R[:, 0], R[:, 1], t]  # floor (x, y, 1) → image
    return np.linalg.inv(G), G


def test_floor_fov_polygon_drops_points_above_horizon():
    H, G = low_camera_h()
    markers = apply_homography(G, np.array([[-1, 2], [1, 2], [-1, 4], [1, 4.0]]))
    poly = np.array(floor_fov_polygon(H, IMG_W, IMG_H, markers, max_dist_m=8.0))
    assert len(poly) >= 3
    assert np.all(poly[:, 1] > 0)  # everything in front of the camera
    centre = apply_homography(H, np.array([[IMG_W / 2, IMG_H - 1]]))[0]
    assert np.all(np.linalg.norm(poly - centre, axis=1) <= 8.0 + 1e-3)
    # a row above the horizon maps behind the camera: must not be in the polygon
    assert apply_homography(H, np.array([[IMG_W / 2, 300]]))[0][1] < 0
    # old behaviour (corners at fov_top 0.35, row 378 just below the horizon ~row 364) ran away
    far = apply_homography(H, np.array([[0, 0.35 * IMG_H]]))[0]
    assert far[1] > 50


def test_loo_errors_exact_and_noisy():
    H, G = low_camera_h()
    floor = np.array([[x, y] for x in (-1.2, 0, 1.2) for y in (2.0, 3.5, 5.0)])
    img = apply_homography(G, floor)
    assert np.nanmax(loo_errors(img, floor)) < 1e-6
    rng = np.random.default_rng(0)
    noisy = img + rng.normal(0, 2.0, img.shape)  # 2 px click noise
    e = loo_errors(noisy, floor)
    assert np.all(np.isfinite(e)) and e.max() > 0.005
    with pytest.raises(ValueError):
        loo_errors(img[:4], floor[:4])


def test_read_points_tiles(tmp_path):
    p = tmp_path / "pts.csv"
    p.write_text(
        "id,img_x,img_y,floor_x,floor_y,tile_i,tile_j\n"
        "1,10,20,,,2,3\n"
        "2,30,40,1.5,0.25,,\n"
        "3,50,60,,,,\n",
        encoding="utf-8",
    )
    img, floor = read_points(p, tile_m=0.6)
    assert img.tolist() == [[10, 20], [30, 40]]
    np.testing.assert_allclose(floor, [[1.2, 1.8], [1.5, 0.25]])
    with pytest.raises(SystemExit):
        read_points(p)
