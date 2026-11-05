import csv

import numpy as np
import pytest

from tools.label_assist import (
    TrackFeat,
    color_hist,
    hellinger_sim,
    resolve_spec_rows,
    suggest_ids,
    torso_roi,
)
from tools.label_intervals import convert_rows


def _hist(bins: dict[int, float]) -> np.ndarray:
    h = np.zeros(256)
    for k, v in bins.items():
        h[k] = v
    return h / h.sum()


ORANGE, LIME, BLUE = _hist({10: 1.0}), _hist({60: 1.0}), _hist({200: 1.0})


def test_hellinger_and_color_hist():
    assert hellinger_sim(ORANGE, ORANGE) == pytest.approx(1.0)
    assert hellinger_sim(ORANGE, BLUE) == 0.0
    img = np.zeros((20, 10, 3), np.uint8)
    img[:] = (255, 0, 0)  # pure blue (BGR)
    h = color_hist(img)
    assert h is not None and h.sum() == pytest.approx(1.0) and np.count_nonzero(h) == 1
    assert color_hist(np.zeros((2, 2, 3), np.uint8)) is None


def test_torso_roi_keypoints_and_fallback():
    kpts = [[0, 0, 0.0]] * 17
    kpts = [list(k) for k in kpts]
    for i, (x, y) in {5: (40, 50), 6: (60, 50), 11: (42, 100), 12: (58, 100)}.items():
        kpts[i] = [x, y, 0.9]
    roi, from_kp = torso_roi(kpts, (30, 20, 70, 180))
    assert from_kp and roi == (40, 50, 60, 100)
    roi, from_kp = torso_roi([[0, 0, 0.0]] * 17, (0, 0, 100, 200))
    assert not from_kp and roi == (15, 50, 85, 120)


def _setup():
    # take t1, cam1: seeds orange=P1, blue=P3; cam2: only seeds from cam1 apply
    feats = {
        ("t1", "cam1", 1): TrackFeat(ORANGE, None, 5),
        ("t1", "cam1", 2): TrackFeat(BLUE, None, 5),
        ("t1", "cam1", 3): TrackFeat(0.9 * ORANGE + 0.1 * BLUE, None, 5),
        ("t1", "cam1", 4): TrackFeat(LIME, None, 5),  # like nobody: nomatch
        ("t1", "cam1", 5): TrackFeat(ORANGE, None, 5),  # too short
        ("t1", "cam1", 6): TrackFeat(None, None, 0),  # no feature
        ("t1", "cam2", 7): TrackFeat(BLUE, None, 5),
    }
    spans = {k: (0, 10_000, 100) for k in feats}
    spans[("t1", "cam1", 5)] = (0, 40, 2)
    seeds = {("t1", "cam1", 1): 1, ("t1", "cam1", 2): 3}
    return feats, spans, seeds


def test_suggest_ids_statuses():
    feats, spans, seeds = _setup()
    spans[("t1", "cam1", 3)] = (20_000, 30_000, 100)  # not overlapping the P1 seed
    s = suggest_ids(feats, spans, seeds)
    assert ("t1", "cam1", 1) not in s  # seeds are not re-suggested
    assert (s[("t1", "cam1", 3)].gt, s[("t1", "cam1", 3)].status) == (1, "auto")
    assert s[("t1", "cam1", 4)].status == "nomatch"
    assert s[("t1", "cam1", 5)].status == "short"
    assert s[("t1", "cam1", 6)].status == "nofeat"
    assert (s[("t1", "cam2", 7)].gt, s[("t1", "cam2", 7)].status) == (3, "auto")


def test_suggest_ids_overlap_constraint_forces_review():
    feats, spans, seeds = _setup()
    spans[("t1", "cam1", 2)] = (20_000, 30_000, 100)  # P3 seed does not coexist with track 3
    # track 3 looks like P1 but coexists with the P1 seed in the same camera → not P1
    s = suggest_ids(feats, spans, seeds)[("t1", "cam1", 3)]
    assert s.status == "review" and s.gt == 3
    # both people coexist with it: nobody is free, keep the colour guess for review
    spans[("t1", "cam1", 2)] = (0, 10_000, 100)
    s = suggest_ids(feats, spans, seeds)[("t1", "cam1", 3)]
    assert s.status == "review" and s.gt == 1


def test_suggest_ids_prefers_same_take_seed():
    # P1 changed clothes in take t2: its seed there (blue) beats the orange seed of t1
    feats = {
        ("t1", "cam1", 1): TrackFeat(ORANGE, None, 5),
        ("t1", "cam1", 2): TrackFeat(LIME, None, 5),
        ("t2", "cam1", 1): TrackFeat(BLUE, None, 5),
        ("t2", "cam1", 9): TrackFeat(BLUE, None, 5),
    }
    spans = {k: (0, 1000, 100) for k in feats}
    spans[("t2", "cam1", 9)] = (5000, 6000, 100)
    seeds = {("t1", "cam1", 1): 1, ("t1", "cam1", 2): 2, ("t2", "cam1", 1): 1}
    s = suggest_ids(feats, spans, seeds)[("t2", "cam1", 9)]
    assert (s.gt, s.status) == (1, "auto")


def test_resolve_spec_rows_then_convert_refuses_todo():
    spec = [
        {"gt": "P2", "type": "PPE_HELMET", "start": "first_seen", "end": "last_seen"},
        {"gt": "?", "type": "INTRUSION", "start": "TODO", "end": "TODO", "cams": "cam2 cam3"},
        {"gt": "P3", "type": "PPE_VEST", "start": "first_seen", "end": "last_seen"},
    ]
    rows = resolve_spec_rows(spec, {2: (11_000, 95_500)}, ref_t0_ms=1000)
    assert rows[0] | {"note": ""} == {
        "gt_id": "P2", "type": "PPE_HELMET", "start": "10.0", "end": "94.5", "note": "",
        "cams": "",
    }  # fmt: skip
    assert rows[1]["cams"] == "cam2;cam3" and rows[1]["start"] == "TODO"
    assert rows[2]["start"] == "TODO"  # P3 has no labelled track
    assert convert_rows(rows[:1], t0_ms=1000)[0][2:4] == [11_000, 95_500]
    with pytest.raises(SystemExit, match="fill gt_id"):
        convert_rows(rows[1:2], t0_ms=0)


def test_ids_cli_writes_only_confident_rows(tmp_path, monkeypatch):
    """End to end on cached features: auto rows get gt_id, review rows keep it empty."""
    from tools import label_assist as la

    jd = tmp_path / "jsonl" / "s01_t01"
    jd.mkdir(parents=True)
    (jd / "cam1.jsonl").write_text("", encoding="utf-8")
    ann = tmp_path / "ann" / "s01_t01"
    (ann / "feats").mkdir(parents=True)
    rows = [
        ["cam1", 1, 0, 9000, 200, "1"],
        ["cam1", 2, 0, 9000, 200, "3"],
        ["cam1", 3, 20000, 30000, 200, ""],
        ["cam1", 4, 0, 9000, 200, ""],
    ]
    with open(ann / "tracks.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(la.TRACK_COLS)
        w.writerows(rows)
    mixed = 0.5 * ORANGE + 0.5 * BLUE
    feats = {
        1: TrackFeat(ORANGE, None, 5),
        2: TrackFeat(BLUE, None, 5),
        3: TrackFeat(ORANGE, None, 5),
        4: TrackFeat(mixed, None, 5),
    }
    la.save_feats(ann / "feats" / "cam1.npz", feats, {})
    monkeypatch.setattr(la, "_review_sheet", lambda *a, **k: None)
    la.main(["ids", "--jsonl", str(jd), "--ann-root", str(tmp_path / "ann"), "--write"])
    with open(ann / "tracks.csv", encoding="utf-8", newline="") as f:
        out = {r["track_id"]: r for r in csv.DictReader(f)}
    assert (out["1"]["gt_id"], out["1"]["status"]) == ("1", "")
    assert (out["3"]["gt_id"], out["3"]["status"], out["3"]["suggest"]) == ("1", "auto", "P1")
    assert out["4"]["gt_id"] == "" and out["4"]["status"] == "review"
    assert (ann / "tracks.csv.bak").exists()
