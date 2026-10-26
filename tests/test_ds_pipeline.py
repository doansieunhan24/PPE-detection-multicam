"""DeepStream-side helpers that run without DeepStream: FPS meter, backend comparison."""

import json

import pytest

from eval.compare_backends import compare, kpt_errors, match_persons
from src.common.schemas import PerceptionFrame, PersonDet, PPEDet
from src.pipeline.ds_app import FpsCounter


def test_fps_counter_excludes_warmup():
    m = FpsCounter(["a", "b"], interval_s=1e9, warmup=2)
    for k in range(12):  # a: 10 fps, b: 5 fps
        m.tick("a", now=k * 0.1)
        if k % 2 == 0:
            m.tick("b", now=k * 0.1)
    s = m.summary()
    assert s["a"] == pytest.approx(10.0)
    assert s["b"] == pytest.approx(5.0)
    assert m.total == {"a": 12, "b": 6}


def test_fps_counter_too_few_frames():
    m = FpsCounter(["a"], warmup=5)
    m.tick("a", now=0.0)
    assert m.summary() == {"a": 0.0}


def _p(tid, box, kpts=None, ppe=()):
    return PersonDet(
        track_id=tid,
        bbox=box,
        conf=0.9,
        kpts=kpts or [],
        ppe_dets=[PPEDet(cls=c, bbox=box, conf=0.8) for c in ppe],
    )


def test_match_persons_iou_gate():
    ref = [_p(1, (0, 0, 10, 20)), _p(2, (100, 100, 110, 120))]
    test = [_p(7, (100, 101, 110, 121)), _p(8, (50, 50, 60, 70))]
    pairs = match_persons(ref, test, 0.5)
    assert [(r, t) for r, t, _ in pairs] == [(1, 0)]


def test_kpt_errors_relative_to_height():
    hidden = [(0, 0, 0.0)] * 15  # schema needs 17 COCO joints
    a = _p(1, (0, 0, 10, 100), kpts=[(0, 0, 0.9), (5, 5, 0.1), *hidden])
    b = _p(1, (0, 0, 10, 100), kpts=[(3, 4, 0.9), (50, 50, 0.9), *hidden])
    assert kpt_errors(a, b, 0.5) == [pytest.approx(0.05)]  # 2nd joint not visible in a


def test_compare_counts(tmp_path):
    def write(d, persons_by_frame):
        d.mkdir()
        with open(d / "cam1.jsonl", "w", encoding="utf-8") as f:
            for i, persons in enumerate(persons_by_frame):
                fr = PerceptionFrame(
                    cam_id="cam1", frame_idx=i, ts_ms=i * 40, img_w=200, img_h=200, persons=persons
                )
                f.write(fr.model_dump_json() + "\n")

    box = (10, 10, 50, 110)
    write(tmp_path / "ref", [[_p(1, box, ppe=("helmet", "vest"))], [_p(1, box, ppe=("vest",))]])
    write(tmp_path / "test", [[_p(4, box, ppe=("helmet",))], []])
    res = compare(tmp_path / "ref", tmp_path / "test")
    o = res["overall"]
    assert o["frames"]["frames_both"] == 2 and o["frames"]["ts_mismatch"] == 0
    assert o["persons"]["matched"] == 1 and o["persons"]["recall"] == 0.5
    ppe = o["ppe"]["per_class"]
    assert ppe["helmet"]["both"] == 1 and ppe["vest"]["ref_only"] == 1
    assert ppe["vest"]["agreement"] == 0.0
    json.dumps(res)  # report is serialisable
