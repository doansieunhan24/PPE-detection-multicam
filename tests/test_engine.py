"""End-to-end engine test on a synthetic 2-camera scene (scenario 4 of §9.1)."""

import numpy as np

from src.common.config import load_config
from src.common.schemas import AlertType, PartName, PerceptionFrame, PersonDet, PPEDet
from src.common.schemas import ComplianceState as CS
from src.fusion.engine import PPEEngine
from src.geometry.homography import CameraCalibration

# both cams: 100 px = 1 m on the floor
H = np.diag([0.01, 0.01, 1.0])
CALIBS = {c: CameraCalibration(c, H) for c in ("cam1", "cam2")}
BBOX = (100.0, 100.0, 200.0, 500.0)


def _kpts(face_conf: float):
    k = [(150.0, 300.0, 0.9)] * 17
    for i, (x, y) in enumerate([(150, 130), (145, 125), (155, 125), (140, 128), (160, 128)]):
        k[i] = (float(x), float(y), face_conf)
    k[15], k[16] = (140.0, 495.0, 0.9), (160.0, 495.0, 0.9)
    k[9], k[10] = (110.0, 300.0, 0.9), (190.0, 300.0, 0.9)
    return k


def _frame(cam, ts, face_conf, helmet):
    dets = [PPEDet(cls="helmet", bbox=(135, 95, 165, 125), conf=0.9)] if helmet else []
    dets.append(PPEDet(cls="vest", bbox=(130, 180, 170, 300), conf=0.8))
    p = PersonDet(track_id=3, bbox=BBOX, conf=0.9, kpts=_kpts(face_conf), ppe_dets=dets)
    return PerceptionFrame(cam_id=cam, frame_idx=ts // 40, ts_ms=ts, img_w=1920, img_h=1080,
                           persons=[p])  # fmt: skip


def _run(policy):
    cfg = load_config("fusion")
    zones = {"association": {"d_max_m": 0.8}, "default_required_ppe": ["helmet"], "zones": []}
    eng = PPEEngine(cfg, zones, CALIBS, policy)
    alerts, last = [], None
    for ts in range(0, 8000, 40):
        group = {
            "cam1": _frame("cam1", ts, face_conf=0.05, helmet=False),  # back turned
            "cam2": _frame("cam2", ts, face_conf=0.9, helmet=True),  # sees the helmet
        }
        states, a, gids = eng.step(group, ts)
        alerts += a
        last = states
    return last, alerts, gids


def test_one_global_id_across_cams():
    states, _, gids = _run("P")
    assert len(states) == 1 and gids[("cam1", 3)] == gids[("cam2", 3)]


def test_proposed_no_false_alarm_when_one_cam_sees_helmet():
    states, alerts, _ = _run("P")
    head = states[0].parts[PartName.HEAD]
    assert head.state is CS.COMPLIANT
    assert head.per_cam["cam1"].v < head.per_cam["cam2"].v
    assert not [a for a in alerts if a.type is AlertType.PPE_HELMET]


def test_single_cam_baseline_false_alarm():
    _, alerts, _ = _run("B1")
    assert [a.type for a in alerts] == [AlertType.PPE_HELMET]
