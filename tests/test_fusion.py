import math

import pytest

from src.common.config import load_config
from src.common.schemas import ComplianceState as CS
from src.fusion.evidence import EvidenceAccumulator, EvidenceParams, logit
from src.fusion.state_machine import ComplianceStateMachine, StateParams


def test_logit_clipped():
    assert logit(0.5) == 0.0
    assert logit(1.0) == pytest.approx(math.log(99))
    assert logit(0.0) == pytest.approx(-math.log(99))


def test_visible_cam_dominates_occluded_cam():
    # cam1 sees no helmet but head barely visible; cam3 sees helmet clearly (example §6.2)
    acc = EvidenceAccumulator()
    m, omega = acc.update(0, [(0.10, 0.15), (0.95, 0.90)])
    assert m > 0
    assert omega == pytest.approx(1.05 * acc.params.dt_nominal_s)


def test_no_visibility_ablation_lets_occluded_cam_count_fully():
    acc = EvidenceAccumulator(EvidenceParams(use_visibility=False))
    m, omega = acc.update(0, [(0.01, 0.05), (0.01, 0.05), (0.95, 0.90)])
    assert m < 0 and omega == pytest.approx(3.0 * acc.params.dt_nominal_s)


def test_temporal_decay():
    acc = EvidenceAccumulator(EvidenceParams(tau_s=2.0))
    acc.update(0, [(0.9, 1.0)])
    _, omega = acc.update(2000, [])
    assert omega == pytest.approx(acc.params.dt_nominal_s * math.exp(-1))
    assert acc.mean == pytest.approx(logit(0.9))  # mean unchanged by pure decay


def test_no_temporal_ablation_is_memoryless():
    acc = EvidenceAccumulator(EvidenceParams(temporal=False))
    acc.update(0, [(0.9, 1.0)])
    assert acc.update(40, []) == (0.0, 0.0)


def test_empty_start_is_finite():
    m, omega = EvidenceAccumulator().update(0, [])
    assert (m, omega) == (0.0, 0.0)


def test_state_machine_transitions_and_hysteresis():
    sm = ComplianceStateMachine(StateParams(theta_obs=1.0, theta_pos=1.0, theta_neg=-1.0))
    assert sm.state is CS.UNKNOWN
    assert sm.update(m=3.0, omega=0.5) is CS.UNKNOWN  # not enough observation
    assert sm.update(m=-2.0, omega=2.0) is CS.VIOLATION
    assert sm.update(m=0.0, omega=2.0) is CS.VIOLATION  # hysteresis band keeps state
    assert sm.update(m=1.5, omega=2.0) is CS.COMPLIANT
    assert sm.update(m=-0.5, omega=2.0) is CS.COMPLIANT
    assert sm.update(m=-0.5, omega=0.2) is CS.UNKNOWN


def test_state_machine_no_unknown_ablation():
    sm = ComplianceStateMachine(StateParams(allow_unknown=False))
    assert sm.update(m=-0.2, omega=0.0) is CS.VIOLATION


def test_bad_thresholds_rejected():
    with pytest.raises(ValueError):
        StateParams(theta_pos=-1.0, theta_neg=1.0)


def test_params_from_config():
    cfg = load_config("fusion")
    assert EvidenceParams.from_config(cfg).tau_s == cfg["evidence"]["tau_s"]
    assert StateParams.from_config(cfg).initial_state is CS.UNKNOWN


def _steady_omega(fps: float, v: float, params: EvidenceParams, seconds: float = 20) -> float:
    acc = EvidenceAccumulator(params)
    step = 1000 / fps
    for i in range(int(seconds * fps)):
        _, omega = acc.update(round(i * step), [(0.5, v)])
    return omega


def test_omega_is_frame_rate_invariant():
    prm = EvidenceParams(tau_s=2.0)
    o10, o25 = _steady_omega(10, 0.5, prm), _steady_omega(25, 0.5, prm)
    assert o10 == pytest.approx(o25, rel=0.1)
    assert o25 == pytest.approx(0.5 * 2.0, rel=0.05)  # ~ W * tau


def test_without_dt_weighting_low_visibility_saturates():
    # documents the flaw of the per-frame formula: v=0.05 at 25 fps passes theta_obs=1
    assert _steady_omega(25, 0.05, EvidenceParams(dt_weighting=False)) > 1.0
    assert _steady_omega(25, 0.05, EvidenceParams()) < 1.0


def test_no_temporal_omega_matches_steady_state_scale():
    acc = EvidenceAccumulator(EvidenceParams(temporal=False, tau_s=2.0))
    _, omega = acc.update(0, [(0.9, 0.5)])
    assert omega == pytest.approx(1.0)
