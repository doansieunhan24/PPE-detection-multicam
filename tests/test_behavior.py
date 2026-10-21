import pytest

from src.alerts.episodes import EpisodeAlerter
from src.behavior.engine import BehaviorEngine
from src.behavior.pose_features import aspect_ratio, mean_kpt_displacement, torso_angle_deg
from src.common.config import load_config
from src.common.schemas import AlertType, PerceptionFrame, PersonDet


def _upright(x=500.0, top=200.0, h=400.0):
    k = [(x, top + 0.5 * h, 0.9)] * 17
    k = list(k)
    k[5], k[6] = (x - 30, top + 0.2 * h, 0.9), (x + 30, top + 0.2 * h, 0.9)
    k[11], k[12] = (x - 20, top + 0.55 * h, 0.9), (x + 20, top + 0.55 * h, 0.9)
    return (x - 60, top, x + 60, top + h), k


def _lying(x=500.0, floor=600.0, length=400.0):
    k = [(x, floor - 30, 0.9)] * 17
    k = list(k)
    k[5], k[6] = (x - 0.3 * length, floor - 40, 0.9), (x - 0.3 * length, floor - 20, 0.9)
    k[11], k[12] = (x + 0.05 * length, floor - 40, 0.9), (x + 0.05 * length, floor - 20, 0.9)
    return (x - length / 2, floor - 80, x + length / 2, floor), k


def _frame(ts, persons):
    return PerceptionFrame(
        cam_id="cam1", frame_idx=ts // 40, ts_ms=ts, img_w=1920, img_h=1080, persons=persons
    )


def _p(tid, bbox_k):
    b, k = bbox_k
    return PersonDet(track_id=tid, bbox=b, conf=0.9, kpts=k)


def test_pose_features():
    b, k = _upright()
    assert aspect_ratio(b) < 0.5 and torso_angle_deg(k) == pytest.approx(0, abs=5)
    b, k = _lying()
    assert aspect_ratio(b) > 4 and torso_angle_deg(k) > 80
    assert mean_kpt_displacement(k, k) == 0.0


def test_episode_alerter_hold_and_cooldown():
    a = EpisodeAlerter(hold_ms=1000, cooldown_ms=5000)
    assert a.update("x", True, 0) is None
    assert a.update("x", True, 1000) == 0
    assert a.update("x", True, 1500) is None  # once per episode
    a.update("x", False, 1600)
    assert a.update("x", True, 2000) is None and a.update("x", True, 3000) is None  # cooldown


def _engine(zones=None):
    return BehaviorEngine(load_config("behavior"), {"zones": zones or []})


def test_fall_detected_then_immobile():
    eng = _engine()
    alerts = []
    for ts in range(0, 2000, 40):  # standing
        alerts += eng.step({"cam1": _frame(ts, [_p(1, _upright())])}, ts, {("cam1", 1): 1}, {})
    for ts in range(2000, 2400, 40):  # falling: hips drop over 0.4 s
        a = (ts - 2000) / 400
        b, k = _upright(top=200 + a * 300)
        alerts += eng.step({"cam1": _frame(ts, [_p(1, (b, k))])}, ts, {("cam1", 1): 1}, {})
    for ts in range(2400, 16000, 40):  # lying still
        alerts += eng.step({"cam1": _frame(ts, [_p(1, _lying())])}, ts, {("cam1", 1): 1}, {})
    types = [a.type for a in alerts]
    assert types.count(AlertType.FALL) == 1
    assert types.count(AlertType.IMMOBILE) == 1
    assert all(a.level == 3 for a in alerts)


def test_lying_without_drop_is_not_a_fall():
    eng = _engine()
    alerts = []
    for ts in range(0, 5000, 40):
        alerts += eng.step({"cam1": _frame(ts, [_p(1, _lying())])}, ts, {("cam1", 1): 1}, {})
    assert AlertType.FALL not in [a.type for a in alerts]


def test_intrusion_and_crowd():
    zones = [
        {"id": "Z2", "type": "forbidden", "polygon_m": [[0, 0], [2, 0], [2, 2], [0, 2]]},
        {"id": "R1", "type": "risk", "polygon_m": [[5, 5], [9, 5], [9, 9], [5, 9]], "crowd_max": 2},
    ]
    eng = _engine(zones)
    persons = [_p(i, _upright(x=100.0 * i)) for i in range(1, 5)]
    gid_of = {("cam1", i): i for i in range(1, 5)}
    floor = {1: (1.0, 1.0), 2: (6.0, 6.0), 3: (7.0, 7.0), 4: (8.0, 8.0)}
    alerts = []
    for ts in range(0, 7000, 40):
        alerts += eng.step({"cam1": _frame(ts, persons)}, ts, gid_of, floor)
    by_type = {a.type: a for a in alerts}
    assert by_type[AlertType.INTRUSION].global_id == 1 and by_type[AlertType.INTRUSION].zone == "Z2"
    assert by_type[AlertType.CROWD].zone == "R1"
    assert sum(a.type is AlertType.INTRUSION for a in alerts) == 1
