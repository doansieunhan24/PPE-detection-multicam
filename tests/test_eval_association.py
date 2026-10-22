import numpy as np
import pytest

from eval.eval_association import evaluate, idf1, iou_matrix, match_to_gt
from src.common.schemas import PerceptionFrame, PersonDet
from src.geometry.homography import CameraCalibration


def _p(tid, bbox):
    return PersonDet(track_id=tid, bbox=bbox, conf=0.9)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert iou_matrix(a, b)[0] == pytest.approx([1.0, 1 / 3, 0.0])


def test_match_to_gt_threshold():
    fr = PerceptionFrame(
        cam_id="a",
        frame_idx=0,
        ts_ms=0,
        img_w=100,
        img_h=100,
        persons=[_p(7, (0, 0, 10, 10)), _p(8, (50, 50, 60, 60))],
    )
    assert match_to_gt(fr, {1: [0, 0, 10, 11], 2: [55, 50, 65, 60]}, 0.5) == {7: 1}


def test_idf1_perfect_and_swapped():
    gt = {(0, "a", 1): 10, (0, "b", 1): 10, (0, "a", 2): 20}
    assert idf1({k: g for k, g in gt.items()}, gt, 3)["idf1"] == pytest.approx(1.0)
    # one obs wrongly gets person 10's id → 2 of 3 correct
    pred = {(0, "a", 1): 1, (0, "b", 1): 1, (0, "a", 2): 1}
    assert idf1(pred, gt, 3)["idf1"] == pytest.approx(2 / 3)
    # a missed GT box lowers recall only
    assert idf1({k: g for k, g in gt.items()}, gt, 4)["idr"] == pytest.approx(0.75)


def test_evaluate_two_cams_identity_homography():
    # image px == floor m (H = I); one person seen by both cams, one only by cam a
    cal = {c: CameraCalibration(c, np.eye(3)) for c in ("a", "b")}
    fa = PerceptionFrame(
        cam_id="a",
        frame_idx=0,
        ts_ms=0,
        img_w=100,
        img_h=100,
        persons=[_p(1, (0, 0, 2, 5)), _p(2, (20, 0, 22, 5))],
    )
    fb = PerceptionFrame(
        cam_id="b", frame_idx=0, ts_ms=0, img_w=100, img_h=100, persons=[_p(9, (0.1, 0, 2.1, 5))]
    )
    gt = {
        0: [
            {"gt_id": 1, "xy_m": [1, 5], "boxes": {"a": [0, 0, 2, 5], "b": [0, 0, 2, 5]}},
            {"gt_id": 2, "xy_m": [21, 5], "boxes": {"a": [20, 0, 22, 5]}},
        ]
    }
    r = evaluate({"a": [fa], "b": [fb]}, cal, gt, 0.8, 15, 0.5, 50, None, 0.0, "bbox_bottom")
    assert r["pairwise"]["tp"] == 1 and r["pairwise"]["fp"] == 0 and r["pairwise"]["fn"] == 0
    assert r["count"]["associated"]["mae"] == 0
    assert r["count"]["no_assoc_sum"]["bias"] == 1
    assert r["id"]["idf1"] == pytest.approx(1.0)
