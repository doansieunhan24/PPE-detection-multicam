from src.alerts.ppe_rules import PPEAlertParams, PPEAlertTracker
from src.common.config import load_config
from src.common.schemas import AlertType
from src.common.schemas import ComplianceState as CS
from src.fusion.policies import (
    POLICY_IDS,
    CamObs,
    ConfWeightedPolicy,
    MajorityPolicy,
    OrPolicy,
    SingleCamPolicy,
    make_policy,
)

# helmet hidden from cam1 (back turned → p_neg, low v), seen by cam2 and cam3
OCCLUDED = [
    CamObs("cam1", p=0.05, v=0.1, det_conf=0.9),
    CamObs("cam2", p=0.05, v=0.1, det_conf=0.9),
    CamObs("cam3", p=0.90, v=0.9, det_conf=0.6),
]


def test_baselines_on_occlusion_case():
    assert SingleCamPolicy("cam1").update(0, OCCLUDED) is CS.VIOLATION  # false alarm
    assert OrPolicy().update(0, OCCLUDED) is CS.COMPLIANT
    assert MajorityPolicy().update(0, OCCLUDED) is CS.VIOLATION  # false alarm
    assert ConfWeightedPolicy().update(0, OCCLUDED) is CS.VIOLATION  # false alarm


def test_proposed_on_occlusion_case():
    cfg = load_config("fusion")
    p = make_policy("P", cfg)
    states = [p.update(t, OCCLUDED) for t in range(0, 2000, 40)]
    assert states[-1] is CS.COMPLIANT


def test_proposed_unknown_when_nobody_sees_head():
    cfg = load_config("fusion")
    p = make_policy("P", cfg)
    obs = [CamObs("cam1", p=0.05, v=0.05, det_conf=0.9)]
    assert all(p.update(t, obs) is CS.UNKNOWN for t in range(0, 3000, 40))


def test_single_cam_ignores_other_cams():
    assert SingleCamPolicy("cam1").update(0, [CamObs("cam2", 0.9, 1, 1)]) is CS.UNKNOWN


def test_factory_builds_all():
    cfg = load_config("fusion")
    for pid in POLICY_IDS:
        make_policy(pid, cfg)
    make_policy("P", cfg, use_visibility=False, temporal=False, allow_unknown=False)


def _tracker(**kw):
    return PPEAlertTracker(PPEAlertParams(**{"T_alert_ms": 3000, "T_unk_ms": 10000, **kw}))


def test_violation_alert_after_persistence_once_per_episode():
    tr = _tracker()
    alerts = [tr.update(5, "helmet", CS.VIOLATION, t, required=True) for t in range(0, 6000, 100)]
    raised = [a for a in alerts if a]
    assert len(raised) == 1
    assert raised[0].type is AlertType.PPE_HELMET and raised[0].level == 2
    assert raised[0].t_start_ms == 0 and raised[0].t_raise_ms == 3000


def test_short_violation_no_alert():
    tr = _tracker()
    for t in range(0, 2000, 100):
        assert tr.update(5, "helmet", CS.VIOLATION, t, True) is None
    assert tr.update(5, "helmet", CS.COMPLIANT, 2000, True) is None


def test_not_required_zone_no_alert():
    tr = _tracker()
    assert all(tr.update(5, "vest", CS.VIOLATION, t, False) is None for t in range(0, 9000, 100))


def test_unknown_gives_l1_unverifiable_not_l2():
    tr = _tracker()
    out = [tr.update(5, "helmet", CS.UNKNOWN, t, True) for t in range(0, 12000, 500)]
    raised = [a for a in out if a]
    assert len(raised) == 1 and raised[0].type is AlertType.UNVERIFIABLE and raised[0].level == 1


def test_cooldown_blocks_second_episode():
    tr = _tracker(T_cd_ms=60000)
    seq = [CS.VIOLATION] * 40 + [CS.COMPLIANT] * 5 + [CS.VIOLATION] * 40
    out = [tr.update(5, "helmet", s, i * 100, True) for i, s in enumerate(seq)]
    assert sum(a is not None for a in out) == 1


def test_b0_no_persistence():
    tr = PPEAlertTracker(PPEAlertParams.from_config(load_config("fusion"), persistence=False))
    assert tr.update(5, "helmet", CS.VIOLATION, 0, True) is not None


def test_baselines_use_detection_flag_not_calibrated_p():
    # calibrated "no box" p can exceed pos_thr (torso: 0.33); B0 must still say VIOLATION
    o = [CamObs("cam1", p=0.33, v=1.0, det_conf=0.9, detected=False)]
    assert SingleCamPolicy("cam1", pos_thr=0.25).update(0, o) is CS.VIOLATION
    assert OrPolicy(0.25).update(0, o) is CS.VIOLATION
