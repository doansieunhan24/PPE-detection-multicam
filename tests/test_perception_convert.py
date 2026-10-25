import numpy as np
import pytest

from src.perception_offline.convert import (
    FrameClock,
    crop_box_to_image,
    file_ts_ms,
    is_live_uri,
    is_truncated_bottom,
    pad_clip_bbox,
    persons_from_arrays,
    ppe_from_arrays,
)


def test_is_live_uri():
    assert is_live_uri("rtsp://localhost:8554/cam1")
    assert not is_live_uri("data/raw/cam1.mp4")
    assert not is_live_uri("D:/videos/cam1.mp4")


def test_file_ts_ms():
    assert file_ts_ms(0, 25, 1000, 0) == 1000
    assert file_ts_ms(25, 25, 1000, -40) == 1960
    assert file_ts_ms(1, 30, 0, 0) == 33


def test_frame_clock_uses_pts_for_variable_frame_rate():
    clock = FrameClock(fps=30, mode="pts")
    # 30 fps then a 15 fps stretch: pts is the truth, frame_idx/fps would drift
    assert [clock.next(t) for t in (0.0, 33.3, 66.7, 133.3, 200.0)] == [0, 33, 67, 133, 200]
    assert clock.n_fallback == 0


def test_frame_clock_falls_back_when_pts_missing_or_not_increasing():
    clock = FrameClock(fps=25, mode="pts")
    assert [clock.next(t) for t in (0.0, 40.0, 0.0, None, 200.0)] == [0, 40, 80, 120, 200]
    assert clock.n_fallback == 2


def test_frame_clock_index_mode_ignores_pts():
    clock = FrameClock(fps=25, mode="index")
    assert [clock.next(t) for t in (0.0, 999.0, 5.0)] == [0, 40, 80]
    with pytest.raises(ValueError):
        FrameClock(fps=25, mode="wallclock")


def test_pad_clip_bbox_clips_to_image():
    assert pad_clip_bbox([10, 10, 110, 210], 0.1, 1920, 1080) == (0.0, 0.0, 120.0, 230.0)
    assert pad_clip_bbox([1800, 900, 1900, 1080], 0.5, 1920, 1080)[2:] == (1920.0, 1080.0)


def test_crop_box_to_image():
    assert crop_box_to_image([5, 6, 15, 26], (100, 200)) == (105, 206, 115, 226)


def test_truncated_bottom():
    assert is_truncated_bottom([0, 0, 10, 1079], 1080, 2)
    assert not is_truncated_bottom([0, 0, 10, 1000], 1080, 2)


def test_persons_from_arrays():
    xyxy = np.array([[10, 20, 110, 1080], [300, 300, 400, 600]], dtype=np.float32)
    conf = np.array([0.9, 0.8], dtype=np.float32)
    ids = np.array([3, 7])
    kpts = np.random.default_rng(0).random((2, 17, 3)).astype(np.float32)
    persons = persons_from_arrays(xyxy, conf, ids, kpts, img_h=1080, truncated_margin_px=2)
    assert [p.track_id for p in persons] == [3, 7]
    assert persons[0].truncated_bottom and not persons[1].truncated_bottom
    assert len(persons[0].kpts) == 17
    assert persons[1].conf == pytest.approx(0.8)


def test_persons_without_track_ids_dropped():
    xyxy = np.zeros((1, 4), dtype=np.float32)
    assert persons_from_arrays(xyxy, np.ones(1), None, None, 1080, 2) == []


def test_ppe_from_arrays_maps_class_and_coords():
    dets = ppe_from_arrays(
        xyxy=np.array([[1, 2, 11, 12]], dtype=np.float32),
        conf=np.array([0.7], dtype=np.float32),
        cls_idx=np.array([2]),
        names={0: "person", 2: "safety-vest"},
        class_map={"safety-vest": "vest"},
        crop_origin=(100, 50),
    )
    assert dets[0].cls == "vest"
    assert dets[0].bbox == (101, 52, 111, 62)
