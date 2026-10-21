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
