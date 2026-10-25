from src.common.jsonl import read_frames
from src.common.schemas import PerceptionFrame
from src.pipeline.meta_export import JsonlExporter


def test_exporter_writes_per_camera(tmp_path):
    ex = JsonlExporter(tmp_path)
    for cam in ("cam1", "cam2"):
        for i in range(5):
            ex.put(PerceptionFrame(cam_id=cam, frame_idx=i, ts_ms=i * 40, img_w=10, img_h=10))
    ex.close()
    assert ex.written == 10 and ex.dropped == 0
    assert [f.frame_idx for f in read_frames(tmp_path / "cam2.jsonl")] == list(range(5))


def test_exporter_drops_instead_of_blocking(tmp_path):
    ex = JsonlExporter(tmp_path, maxsize=1)
    for i in range(2000):
        ex.put(PerceptionFrame(cam_id="c", frame_idx=i, ts_ms=i, img_w=1, img_h=1))
    ex.close()
    assert ex.written + ex.dropped == 2000


def test_letterbox_keypoints_undoes_symmetric_padding():
    from src.pipeline.meta_export import letterbox_keypoints

    # mux 1920x1080 letterboxed into a 960x960 network: gain 0.5, pad_y (960 - 540) / 2 = 210
    data = [480.0, 480.0, 0.9, 0.0, 210.0, 0.5]
    kp = letterbox_keypoints(data, 6, 960, 960, 1920, 1080)
    assert kp == [(960.0, 540.0, 0.9), (0.0, 0.0, 0.5)]
    # camera recorded at 1280x720 while streammux runs at 1920x1080
    kp = letterbox_keypoints(data, 6, 960, 960, 1920, 1080, sx=1280 / 1920, sy=720 / 1080)
    assert kp[0][:2] == (640.0, 360.0)
