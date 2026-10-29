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


def test_crowd_is_matched_without_person():
    gt = [GTInterval(None, "CROWD", 10_000, 20_000)]
    a = AlertEvent(alert_id="c", type=AlertType.CROWD, level=2, zone="Z1",
                   t_start_ms=9_000, t_raise_ms=12_000)  # fmt: skip
    m = event_metrics([a, a.model_copy(update={"t_raise_ms": 50_000})], gt, {}, 3600)
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)


def test_load_gt_csv_empty_gt_id(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("gt_id,type,t_start_ms,t_end_ms\n,CROWD,0,1000\n", encoding="utf-8")
    assert load_gt_csv(p) == [GTInterval(None, "CROWD", 0, 1000)]


def test_association_metrics():
    from eval.metrics import association_metrics

    track_gt = {("c1", 1): 1, ("c2", 7): 1, ("c1", 2): 2, ("c2", 8): 2}
    obs = [
        (0, "c1", 1, 10), (0, "c2", 7, 10), (0, "c1", 2, 11), (0, "c2", 8, 11),  # all right
        (40, "c1", 1, 10), (40, "c2", 7, 12), (40, "c1", 2, 11), (40, "c2", 8, 11),  # 1 split
        (80, "c1", 1, 10), (80, "c2", 8, 10), (80, "c3", 9, 10),  # wrong merge; c3 unlabelled
    ]  # fmt: skip
    m = association_metrics(obs, track_gt)
    assert (m["pair_tp"], m["pair_fp"], m["pair_fn"]) == (3, 1, 1)
    assert m["count_mae"] == pytest.approx(2 / 3)  # t=40: 3 gids vs 2 gts; t=80: 1 vs 2
    assert m["frag"] == pytest.approx(2.0)  # gt 1 → {10, 12}, gt 2 → {11, 10}
