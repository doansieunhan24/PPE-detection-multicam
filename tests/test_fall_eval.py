from eval.eval_fall import in_split, score, single_actor
from src.common.schemas import PerceptionFrame, PersonDet
from tools.urfall_prepare import gt_from_labels, runs


def test_runs_and_gt_from_labels():
    rows = [(i, -1) for i in range(1, 31)] + [(i, 0) for i in range(31, 41)]
    rows += [(i, 1) for i in range(41, 91)] + [(i, -1) for i in range(91, 101)]
    assert runs(rows, {1}) == [(41, 90)]
    g = gt_from_labels(rows, fps=10)
    assert g["fall"] == [3000, 8900] and g["lying_ms"] == [[4000, 8900]] and g["n_frames"] == 100
    # lying without preceding "falling" frames is still an episode here; urfall_prepare
    # clears it for ADL sequences (hard negatives)
    adl = [(i, -1) for i in range(1, 11)] + [(i, 1) for i in range(11, 21)]
    assert gt_from_labels(adl, fps=10)["fall"] == [1000, 1900]
    assert gt_from_labels([(i, -1) for i in range(1, 5)], fps=10)["fall"] is None


def test_score_event_matching():
    gt = {
        "fall-01": {"kind": "fall", "n_frames": 300, "fps": 30, "fall": [3000, 6000]},
        "fall-02": {"kind": "fall", "n_frames": 300, "fps": 30, "fall": [3000, 6000]},
        "adl-01": {"kind": "adl", "n_frames": 300, "fps": 30, "fall": None},
    }
    per_seq = {
        "fall-01": {"alerts": [3500, 9000]},  # TP + a late extra alert (FP)
        "fall-02": {"alerts": []},  # FN
        "adl-01": {"alerts": [1000]},  # FP
    }
    r = score(per_seq, gt, delta_ms=2000)
    assert (r["tp"], r["fp"], r["fn"], r["falls"]) == (1, 2, 1, 2)
    assert r["recall"] == 0.5 and abs(r["precision"] - 1 / 3) < 1e-9
    assert r["latency_ms_median"] == 500
    assert r["fa_per_hour"] == 2 / (30 / 3600)
    assert r["missed"] == ["fall-02"] and r["fp_sequences"] == ["adl-01", "fall-01"]


def test_single_actor_keeps_most_confident():
    ps = [
        PersonDet(track_id=1, bbox=(0, 0, 10, 20), conf=0.4),
        PersonDet(track_id=2, bbox=(0, 0, 10, 20), conf=0.9),
    ]
    fr = PerceptionFrame(cam_id="cam0", frame_idx=0, ts_ms=0, img_w=64, img_h=64, persons=ps)
    assert [p.track_id for p in single_actor(fr).persons] == [2]


def test_in_split():
    assert in_split("fall-03", "val") and not in_split("fall-03", "test")
    assert in_split("adl-12", "test") and in_split("adl-12", "all")
