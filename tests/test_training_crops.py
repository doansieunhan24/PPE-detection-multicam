import pytest

from training.ppe_detector.crops import (
    Label,
    crop_labels,
    pad_clip,
    parse_yolo_labels,
    xyxy_to_yolo,
    yolo_to_xyxy,
)


def test_yolo_roundtrip():
    box = yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, 1000, 500)
    assert box == pytest.approx((400, 150, 600, 350))
    assert xyxy_to_yolo(box, 1000, 500) == pytest.approx((0.5, 0.5, 0.2, 0.4))


def test_parse_skips_bad_lines():
    labels = parse_yolo_labels(["10 0.5 0.5 0.1 0.1", "", "bad"], 100, 100)
    assert len(labels) == 1 and labels[0].cls == 10


def test_pad_clip():
    assert pad_clip((10, 10, 110, 210), 0.1, 1000, 1000) == (0, 0, 120, 230)


def test_crop_labels_keep_clip_shift():
    crop = (100.0, 100.0, 300.0, 500.0)
    labels = [
        Label(10, (150, 110, 200, 160)),  # helmet fully inside
        Label(16, (90, 200, 210, 300)),  # vest ~92% inside → kept, clipped
        Label(14, (280, 480, 380, 520)),  # shoes mostly outside → dropped
        Label(3, (150, 150, 160, 160)),  # face, not a target class → dropped
    ]
    out = crop_labels(crop, labels, {10: 0, 16: 1, 14: 2}, min_keep_ratio=0.6)
    assert [lb.cls for lb in out] == [0, 1]
    assert out[0].box == (50, 10, 100, 60)
    assert out[1].box == (0, 100, 110, 200)
