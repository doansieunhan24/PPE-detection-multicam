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


def test_gt_cams_column_and_filter(tmp_path):
    from eval.metrics import gt_for_cams

    p = tmp_path / "gt.csv"
    p.write_text(
        "gt_id,type,t_start_ms,t_end_ms,note,cams\n1,INTRUSION,0,1000,,cam2 cam3\n"
        "1,PPE_HELMET,0,1000,,\n",
        encoding="utf-8",
    )
    gt = load_gt_csv(p)
    assert gt[0].cams == frozenset({"cam2", "cam3"}) and gt[1].cams == frozenset()
    # cam1 cannot see the intrusion: only the all-camera interval is scored for it
    assert gt_for_cams(gt, ["cam1"]) == [gt[1]]
    assert gt_for_cams(gt, ["cam3"]) == gt


def _track_rows(*rows):
    cols = ("cam_id", "track_id", "first_ms", "last_ms", "n_frames", "gt_id")
    return [dict(zip(cols, map(str, r), strict=True)) for r in rows]


def test_track_gt_split_track_and_unlabelled_rows():
    from eval.metrics import TrackGT

    tg = TrackGT.from_rows(_track_rows(
        ("cam3", 11, 41_000, 49_000, 200, 1),  # ID swap: A, then B
        ("cam3", 11, 49_000, 63_000, 300, 2),
        ("cam3", 12, 0, 5_000, 100, 3),
        ("cam3", 13, 0, 5_000, 100, ""),  # unlabelled, not split: dropped
    ))  # fmt: skip
    assert ("cam3", 13) not in tg and len(tg) == 2
    assert tg.is_split(("cam3", 11)) and not tg.is_split(("cam3", 12))
    assert tg.at("cam3", 11, 45_000) == 1 and tg.at("cam3", 11, 55_000) == 2
    assert tg.at("cam3", 11, 70_000) == 2  # outside every stretch: the nearest
    assert tg.at("cam3", 12, 99_999) == 3  # one-row track: same person at any time
    assert tg.get(("cam3", 11)) == 2  # longest stretch


def test_gid_gt_majority_unchanged_without_split():
    from eval.metrics import GidGT, TrackGT

    tg = TrackGT({("c1", 1): [(0, 100, 1)], ("c1", 2): [(0, 100, 2)]})
    rows = [(0, "c1", 1, 5), (40, "c1", 1, 5), (80, "c1", 2, 5), (0, "c1", 2, 6)]
    m = GidGT(rows, tg)
    assert m.majority == majority_gid_to_gt([(5, 1), (5, 1), (5, 2), (6, 2)])
    assert m.at(5, 80) == 1  # no split track: majority at any time, as before


def test_split_track_alert_goes_to_person_in_box_at_raise():
    from eval.metrics import GidGT, TrackGT

    tg = TrackGT({("c3", 11): [(41_000, 49_000, 1), (49_000, 63_000, 2)]})
    rows = [(t, "c3", 11, 9) for t in range(41_000, 63_001, 1_000)]
    m = GidGT(rows, tg)
    gt = [GTInterval(1, "PPE_HELMET", 41_000, 49_000)]  # only A lacks a helmet
    hit = event_metrics([_alert(9, 45_000)], gt, m, 3600)
    assert (hit.tp, hit.fp) == (1, 0)
    late = event_metrics([_alert(9, 56_000)], gt, m, 3600)  # B is in the box: false alarm
    assert (late.tp, late.fp, late.fn) == (0, 1, 1)
    # the old whole-track majority (B) would have missed A's violation and its alert
    old = event_metrics([_alert(9, 45_000)], gt, {9: 2}, 3600)
    assert (old.tp, old.fp) == (0, 1)


def test_duplicate_alerts_counted_apart_but_still_false_alarms():
    gt = [GTInterval(2, "PPE_HELMET", 20_000, 100_000)]  # B without a helmet the whole take
    alerts = [_alert(6, 25_000), _alert(6, 70_000), _alert(5, 50_000)]  # 2nd = repeat; gid 5 = A
    m = event_metrics(alerts, gt, MAP, 3600)
    assert (m.tp, m.fp, m.fn) == (1, 2, 0)
    assert m.fp_duplicate == 1  # only the repeat for B; A's alert is a plain false alarm


def test_gid_joining_two_tracks_of_one_camera_uses_person_at_raise():
    from eval.metrics import GidGT, TrackGT

    tg = TrackGT({("c1", 1): [(0, 10_000, 1)], ("c1", 2): [(11_000, 20_000, 2)]})
    rows = [(t, "c1", 1, 7) for t in range(0, 10_001, 1000)]
    rows += [(t, "c1", 2, 7) for t in range(11_000, 20_001, 1000)]  # stitched under gid 7
    m = GidGT(rows, tg, stitched=True)
    assert m.at(7, 5_000) == 1 and m.at(7, 15_000) == 2
    assert GidGT(rows, tg).at(7, 5_000) == GidGT(rows, tg).majority[7]  # off: majority as before
