import pytest

from eval.metrics import (
    GTInterval,
    event_metrics,
    load_gt_csv,
    majority_gid_to_gt,
    match_events,
    unknown_fraction,
)
from src.common.schemas import AlertEvent, AlertType
from src.common.schemas import ComplianceState as CS


def _alert(gid, t, typ=AlertType.PPE_HELMET):
    return AlertEvent(alert_id=f"a{gid}{t}", type=typ, level=2, global_id=gid,
                      t_start_ms=t - 3000, t_raise_ms=t)  # fmt: skip


GT = [
    GTInterval(1, "PPE_HELMET", 10_000, 20_000),
    GTInterval(2, "PPE_HELMET", 30_000, 40_000),
]
MAP = {5: 1, 6: 2, 7: 3}


def test_perfect_match():
    m = event_metrics([_alert(5, 13_000), _alert(6, 33_000)], GT, MAP, duration_s=3600)
    assert (m.tp, m.fp, m.fn) == (2, 0, 0)
    assert m.f1 == 1.0 and m.fa_per_hour == 0.0
    assert m.mean_ttd_s == pytest.approx(3.0)


def test_false_alarms_and_misses():
    alerts = [
        _alert(5, 13_000),
        _alert(5, 15_000),  # duplicate on same GT → FA (GT matched at most once)
        _alert(7, 13_000),  # person without GT violation → FA
        _alert(6, 50_000),  # outside [t_start-δ, t_end+δ] → FA, GT 2 missed
        _alert(5, 13_000, AlertType.PPE_VEST),  # wrong type → FA
    ]
    m = event_metrics(alerts, GT, MAP, duration_s=1800, delta_ms=2000)
    assert (m.tp, m.fp, m.fn) == (1, 4, 1)
    assert m.fa_per_hour == pytest.approx(8.0)


def test_delta_tolerance():
    matches, fps, _ = match_events([_alert(6, 41_500)], GT, MAP, delta_ms=2000)
    assert len(matches) == 1 and not fps


def test_type_filter():
    m = event_metrics([_alert(5, 13_000, AlertType.PPE_VEST)], GT, MAP, 3600, types=["PPE_VEST"])
    assert m.n_gt == 0 and m.fp == 1


def test_unknown_fraction():
    assert unknown_fraction([CS.UNKNOWN, CS.COMPLIANT, CS.UNKNOWN, CS.VIOLATION]) == 0.5
    assert unknown_fraction([]) == 0.0


def test_majority_mapping():
    assert majority_gid_to_gt([(5, 1), (5, 1), (5, 2), (6, 2)]) == {5: 1, 6: 2}


def test_load_gt_csv(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("gt_id,type,t_start_ms,t_end_ms\n1,PPE_HELMET,0,1000\n", encoding="utf-8")
    assert load_gt_csv(p) == [GTInterval(1, "PPE_HELMET", 0, 1000)]
