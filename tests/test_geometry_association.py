import numpy as np
import pytest

from src.association.global_id import GlobalIdManager
from src.association.ground_plane import FloorObs, cluster, floor_centre, match_pair, sync_groups
from src.common.schemas import PerceptionFrame, PersonDet
from src.geometry.foot_point import foot_point
from src.geometry.homography import (
    CameraCalibration,
    apply_homography,
    fit_homography,
    load_calibration,
    point_in_polygon,
    polygon_distance,
)


def test_fit_and_apply_homography():
    H_true = np.array([[0.01, 0.0, -1.0], [0.0, 0.02, -2.0], [0.0, 0.0001, 1.0]])
    img = np.array([[100, 100], [900, 120], [880, 700], [120, 680], [500, 400], [300, 600]], float)
    floor = apply_homography(H_true, img)
    H, err = fit_homography(img, floor)
    assert err < 1e-6
    assert apply_homography(H, [[640, 360]]) == pytest.approx(
        apply_homography(H_true, [[640, 360]])
    )


def test_point_in_polygon():
    sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
    assert point_in_polygon((2, 2), sq)
    assert not point_in_polygon((5, 2), sq)


def test_polygon_distance_and_fov_margin():
    sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
    assert polygon_distance((5, 2), sq) == pytest.approx(1.0)
    assert polygon_distance((2, 1), sq) == pytest.approx(1.0)
    cal = CameraCalibration("c", np.eye(3), fov_floor_polygon_m=sq)
    assert not cal.sees((4.4, 2))
    assert cal.sees((4.4, 2), margin_m=0.5)
    assert not cal.sees((5.0, 2), margin_m=0.5)


def test_uncalibrated_placeholder_loads():
    cal = load_calibration("configs/calibration/cam1.yaml")
    assert not cal.calibrated and cal.sees((100.0, 100.0))


def _person(bbox, kpts=None, truncated=False):
    return PersonDet(track_id=1, bbox=bbox, conf=0.9, kpts=kpts or [], truncated_bottom=truncated)


def test_foot_point_ankles_vs_bbox():
    k = [(0.0, 0.0, 0.0)] * 17
    k[15], k[16] = (90.0, 490.0, 0.9), (110.0, 492.0, 0.8)
    fp = foot_point(_person((50, 100, 150, 500), k))
    assert fp.from_ankles and (fp.x_px, fp.y_px) == (100.0, 491.0)
    fp = foot_point(_person((50, 100, 150, 500), truncated=True))
    assert not fp.from_ankles and not fp.reliable and fp.y_px == 500
    fp = foot_point(_person((50, 100, 150, 500), k), source="bbox_bottom")
    assert not fp.from_ankles and (fp.x_px, fp.y_px) == (100.0, 500.0)
    with pytest.raises(ValueError):
        foot_point(_person((50, 100, 150, 500), k), source="knees")


def _frame(cam, ts):
    return PerceptionFrame(cam_id=cam, frame_idx=0, ts_ms=ts, img_w=10, img_h=10)


def test_sync_groups_tolerance():
    streams = {
        "cam1": [_frame("cam1", t) for t in (0, 40, 80, 120)],
        "cam2": [_frame("cam2", t) for t in (10, 200)],
    }
    groups = list(sync_groups(streams, tol_ms=50))
    assert len(groups) == 4
    assert "cam2" in groups[0] and groups[0]["cam2"].ts_ms == 10
    assert "cam2" not in groups[3]  # 120 vs nearest 200/10 → outside tol


def test_match_pair_gating():
    a = [FloorObs(("c1", 1), (0.0, 0.0)), FloorObs(("c1", 2), (5.0, 5.0))]
    b = [FloorObs(("c2", 7), (0.3, 0.1)), FloorObs(("c2", 8), (9.0, 9.0))]
    assert match_pair(a, b, d_max_m=0.8) == [(("c1", 1), ("c2", 7))]


def test_cluster_three_cams_and_fov():
    obs = {
        "c1": [FloorObs(("c1", 1), (1.0, 1.0))],
        "c2": [FloorObs(("c2", 5), (1.2, 1.1))],
        "c3": [FloorObs(("c3", 9), (1.1, 0.9))],
    }
    cl = cluster(obs, d_max_m=0.8)
    assert len(cl) == 1 and len(cl[0]) == 3
    # c3 cannot see (1,1): it must not be merged
    sees = {"c1": lambda p: True, "c2": lambda p: True, "c3": lambda p: p[0] > 5}
    sizes = sorted(len(c) for c in cluster(obs, 0.8, sees))
    assert sizes == [1, 2]


def test_cluster_never_two_obs_of_same_cam():
    obs = {
        "c1": [FloorObs(("c1", 1), (0.0, 0.0)), FloorObs(("c1", 2), (1.0, 0.0))],
        "c2": [FloorObs(("c2", 1), (0.5, 0.0))],
        "c3": [FloorObs(("c3", 1), (0.95, 0.0))],
    }
    for c in cluster(obs, d_max_m=0.8):
        cams = [o.key[0] for o in c]
        assert len(cams) == len(set(cams))


def test_global_ids_stable_and_reused():
    m = GlobalIdManager(window=5, reuse_dist_m=0.8, reuse_ttl_ms=1000)
    a, b = FloorObs(("c1", 1), (0.0, 0.0)), FloorObs(("c2", 4), (0.1, 0.0))
    g = m.update([[a, b]], 0)
    assert g[("c1", 1)] == g[("c2", 4)]
    gid = g[("c1", 1)]
    # one frame of wrong split must not change the id
    g = m.update([[a], [b]], 40)
    assert g[("c1", 1)] == gid
    # track lost then new track id at the same spot inherits the id
    g = m.update([[FloorObs(("c1", 2), (0.2, 0.1))]], 500)
    assert g[("c1", 2)] == gid
    # far away → new id
    g = m.update([[FloorObs(("c1", 3), (9.0, 9.0))]], 600)
    assert g[("c1", 3)] != gid


def test_calibration_to_floor_requires_H():
    with pytest.raises(ValueError):
        CameraCalibration("x", None).to_floor(np.zeros((1, 2)))


def test_match_pair_truncated_gate():
    a = [FloorObs(("c1", 1), (0.0, 0.0), reliable=False)]
    b = [FloorObs(("c2", 1), (1.0, 0.0))]
    assert match_pair(a, b, 0.8) == []  # default: same gate for every pair
    assert match_pair(a, b, 0.8, d_max_unreliable_m=1.2) == [(("c1", 1), ("c2", 1))]
    # two reliable obs keep the tight gate
    a_ok = [FloorObs(("c1", 1), (0.0, 0.0))]
    assert match_pair(a_ok, b, 0.8, d_max_unreliable_m=1.2) == []


def test_match_pair_normalised_cost_prefers_tight_match():
    # the cut obs c1/2 is closest to c2/1 in normalised cost (0.6/2.0), but the joint
    # assignment keeps the tight reliable pair c1/1–c2/1 and gives c1/2 the other partner
    a = [FloorObs(("c1", 1), (0.0, 0.0)), FloorObs(("c1", 2), (5.0, 5.0))]
    b = [FloorObs(("c2", 7), (0.3, 0.1)), FloorObs(("c2", 8), (9.0, 9.0))]
    assert match_pair(a, b, d_max_m=0.8) == [(("c1", 1), ("c2", 7))]


def test_cluster_three_cams_and_fov():
    obs = {
        "c1": [FloorObs(("c1", 1), (1.0, 1.0))],
        "c2": [FloorObs(("c2", 5), (1.2, 1.1))],
        "c3": [FloorObs(("c3", 9), (1.1, 0.9))],
    }
    cl = cluster(obs, d_max_m=0.8)
    assert len(cl) == 1 and len(cl[0]) == 3
    # c3 cannot see (1,1): it must not be merged
    sees = {"c1": lambda p: True, "c2": lambda p: True, "c3": lambda p: p[0] > 5}
    sizes = sorted(len(c) for c in cluster(obs, 0.8, sees))
    assert sizes == [1, 2]


def test_cluster_never_two_obs_of_same_cam():
    obs = {
        "c1": [FloorObs(("c1", 1), (0.0, 0.0)), FloorObs(("c1", 2), (1.0, 0.0))],
        "c2": [FloorObs(("c2", 1), (0.5, 0.0))],
        "c3": [FloorObs(("c3", 1), (0.95, 0.0))],
    }
    for c in cluster(obs, d_max_m=0.8):
        cams = [o.key[0] for o in c]
        assert len(cams) == len(set(cams))


def test_global_ids_stable_and_reused():
    m = GlobalIdManager(window=5, reuse_dist_m=0.8, reuse_ttl_ms=1000)
    a, b = FloorObs(("c1", 1), (0.0, 0.0)), FloorObs(("c2", 4), (0.1, 0.0))
    g = m.update([[a, b]], 0)
    assert g[("c1", 1)] == g[("c2", 4)]
    gid = g[("c1", 1)]
    # one frame of wrong split must not change the id
    g = m.update([[a], [b]], 40)
    assert g[("c1", 1)] == gid
    # track lost then new track id at the same spot inherits the id
    g = m.update([[FloorObs(("c1", 2), (0.2, 0.1))]], 500)
    assert g[("c1", 2)] == gid
    # far away → new id
    g = m.update([[FloorObs(("c1", 3), (9.0, 9.0))]], 600)
    assert g[("c1", 3)] != gid


def test_calibration_to_floor_requires_H():
    with pytest.raises(ValueError):
        CameraCalibration("x", None).to_floor(np.zeros((1, 2)))


def test_match_pair_truncated_gate():
    a = [FloorObs(("c1", 1), (0.0, 0.0), reliable=False)]
    b = [FloorObs(("c2", 1), (1.0, 0.0))]
    assert match_pair(a, b, 0.8) == []  # default: same gate for every pair
    assert match_pair(a, b, 0.8, d_max_unreliable_m=1.2) == [(("c1", 1), ("c2", 1))]
    # two reliable obs keep the tight gate
    a_ok = [FloorObs(("c1", 1), (0.0, 0.0))]
    assert match_pair(a_ok, b, 0.8, d_max_unreliable_m=1.2) == []


def test_match_pair_normalised_cost_prefers_tight_match():
    # c2/1 is 0.5 m from a reliable c1 obs and 0.6 m from a cut one: with distance / gate the
    # cut obs (gate 2.0 → 0.3) would win on raw cost… it must take the farther free partner.
    a = [FloorObs(("c1", 1), (0.0, 0.0)), FloorObs(("c1", 2), (1.1, 0.0), reliable=False)]
    b = [FloorObs(("c2", 1), (0.5, 0.0)), FloorObs(("c2", 2), (2.0, 0.0))]
    got = set(match_pair(a, b, 0.8, d_max_unreliable_m=2.0))
    assert got == {(("c1", 1), ("c2", 1)), (("c1", 2), ("c2", 2))}


def test_floor_centre_prefers_reliable():
    obs = [FloorObs(("c1", 1), (0.0, 0.0)), FloorObs(("c2", 1), (2.0, 0.0), reliable=False)]
    assert floor_centre(obs).tolist() == [0.0, 0.0]
    cut = [FloorObs(("c1", 1), (0.0, 0.0), False), FloorObs(("c2", 1), (2.0, 0.0), False)]
    assert floor_centre(cut).tolist() == [1.0, 0.0]
