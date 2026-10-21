import pytest

from src.common.config import load_config
from src.common.schemas import PartName, PersonDet, PPEDet
from src.fusion.visibility import (
    VisibilityParams,
    closer_persons,
    occlusion_ratio,
    visibility,
)
from src.parts.part_rois import PartRoi, RoiParams, part_roi
from src.parts.ppe_assign import AssignParams, containment, observe_part

PERSON = (100.0, 100.0, 200.0, 500.0)  # 100 x 400 px, upright


def standing_kpts(conf: float = 0.9):
    k = [(0.0, 0.0, 0.0)] * 17
    pts = {
        0: (150, 130),
        1: (145, 125),
        2: (155, 125),
        3: (140, 128),
        4: (160, 128),
        5: (130, 180),
        6: (170, 180),
        9: (125, 300),
        10: (175, 300),
        11: (135, 300),
        12: (165, 300),
        15: (140, 490),
        16: (160, 490),
    }
    for i, (x, y) in pts.items():
        k[i] = (float(x), float(y), conf)
    return k


def test_head_roi_from_kpts_extends_upwards():
    roi = part_roi(PartName.HEAD, PERSON, standing_kpts())
    assert not roi.fallback
    assert roi.box[1] < 125 and roi.box[3] > 130  # covers face and above
    assert roi.s_kp == pytest.approx(0.9)


def test_fallback_when_kpts_missing():
    roi = part_roi(PartName.HEAD, PERSON, [])
    assert roi.fallback and roi.s_kp == 0.0
    assert roi.box == pytest.approx((100, 100, 200, 180))  # top 20%


def test_torso_and_feet_rois():
    k = standing_kpts()
    torso = part_roi(PartName.TORSO, PERSON, k)
    feet = part_roi(PartName.FEET, PERSON, k)
    assert torso.box[1] < 180 and torso.box[3] > 300
    assert feet.box[1] < 490 < feet.box[3]


def test_containment():
    assert containment((0, 0, 10, 10), (0, 0, 20, 20)) == 1.0
    assert containment((0, 0, 10, 10), (5, 0, 20, 20)) == 0.5
    assert containment((0, 0, 0, 0), (0, 0, 1, 1)) == 0.0


def _person(dets, checked=True):
    return PersonDet(
        track_id=1, bbox=PERSON, conf=0.9, kpts=standing_kpts(), ppe_dets=dets, ppe_checked=checked
    )


def test_helmet_on_head_matched():
    p = _person([PPEDet(cls="helmet", bbox=(138, 100, 162, 125), conf=0.8)])
    roi = part_roi(PartName.HEAD, PERSON, p.kpts)
    obs = observe_part(p, roi, "helmet")
    assert obs.p == pytest.approx(0.8) and not obs.held_helmet


def test_held_helmet_not_counted():
    p = _person([PPEDet(cls="helmet", bbox=(115, 290, 135, 310), conf=0.9)])  # at left wrist
    roi = part_roi(PartName.HEAD, PERSON, p.kpts)
    obs = observe_part(p, roi, "helmet", AssignParams(p_neg=0.05))
    assert obs.p == 0.05 and obs.held_helmet


def test_ppe_not_checked_gives_no_evidence():
    p = _person([], checked=False)
    assert observe_part(p, part_roi(PartName.HEAD, PERSON, p.kpts), "helmet").p is None


def test_occlusion_ratio():
    roi = (0.0, 0.0, 10.0, 10.0)
    assert occlusion_ratio(roi, [], 100, 100, 16) == 0.0
    assert occlusion_ratio(roi, [(0, 0, 5, 10)], 100, 100, 16) == pytest.approx(0.5)
    assert occlusion_ratio((-10, 0, 10, 10), [], 100, 100, 16) == pytest.approx(0.5)  # off-image


def test_only_closer_persons_occlude():
    others = [(0, 0, 10, 600), (0, 0, 10, 300)]
    assert closer_persons(PERSON, others) == [(0, 0, 10, 600)]


def test_visibility_terms():
    roi = PartRoi(PartName.HEAD, (100, 100, 200, 116), fallback=True, s_kp=0.8)
    t = visibility(roi, PERSON, [], 1920, 1080, VisibilityParams(h_ref_px=32, fallback_penalty=0.5))
    assert t.s_size == pytest.approx(0.5)
    assert t.v == pytest.approx(0.8 * 0.5 * 0.5)


def test_occluded_head_low_v():
    roi = part_roi(PartName.HEAD, PERSON, standing_kpts())
    front = (90.0, 90.0, 210.0, 700.0)  # someone in front covering everything
    assert visibility(roi, PERSON, [front], 1920, 1080).v == 0.0


def test_params_from_config():
    cfg = load_config("fusion")
    assert RoiParams.from_config(cfg).fb_head == (0.0, 0.2)
    assert AssignParams.from_config(cfg).p_neg == cfg["part_assign"]["p_neg"]
    assert VisibilityParams.from_config(cfg).h_ref_px == cfg["visibility"]["h_ref_px"]


def test_calibrated_p_neutral_prior():
    from src.parts.ppe_assign import calibrated_p

    cal = {"a": 1.0, "c": 3.0, "d": -1.0}
    assert calibrated_p(None, cal) == pytest.approx(1 / (1 + 2.718281828**1))
    assert calibrated_p(0.5, cal) == pytest.approx(1 / (1 + 2.718281828**-3))


def test_calibrated_assignment_uses_part_params():
    cal = {"head": {"a": 1.0, "c": 3.0, "d": -1.2}}
    p = _person([])
    obs = observe_part(
        p, part_roi(PartName.HEAD, PERSON, p.kpts), "helmet", AssignParams(calibration=cal)
    )
    assert obs.p == pytest.approx(1 / (1 + 2.718281828**1.2))
