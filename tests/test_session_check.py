import json

import cv2
import numpy as np
import pytest

from tools.session_check import main as session_check_main
from tools.session_check import pts_stats, session_cameras_yaml
from tools.sync_offsets import find_light_on, offsets_and_drift


def test_find_light_on_picks_largest_rise():
    t = [0, 40, 80, 120, 160, 200]
    m = [120, 121, 20, 21, 119, 120]  # lights off at 80, back on at 160
    t_on, strength = find_light_on(t, m)
    assert t_on == 160
    assert strength > 50
    with pytest.raises(ValueError):
        find_light_on([0], [1])


def test_offsets_and_drift():
    ev = {"cam1": (7000.0, 52000.0), "cam2": (5800.0, 50840.0), "cam3": (6600.0, None)}
    r = offsets_and_drift(ev, "cam1")
    assert r["cam1"] == {"time_offset_ms": 0, "drift_ms": 0.0}
    assert r["cam2"] == {"time_offset_ms": 1200, "drift_ms": 40.0}
    assert r["cam3"] == {"time_offset_ms": 400, "drift_ms": None}
    with pytest.raises(ValueError):
        offsets_and_drift(ev, "cam9")


def test_pts_stats_cfr_vs_vfr():
    cfr = pts_stats(np.arange(250) * 40.0)
    assert not cfr["vfr"] and cfr["fps_median"] == 25.0 and cfr["index_time_error_ms"] == 0.0
    t = np.concatenate([np.arange(100) * 40.0, 4000 + np.arange(1, 50) * 80.0])
    vfr = pts_stats(t)
    assert vfr["vfr"] and vfr["gaps"] == 49
    assert vfr["index_time_error_ms"] == 2000.0  # 7920 ms real vs 148 * 40 ms


def test_session_cameras_yaml_keeps_calibration():
    base = {"sync_tol_ms": 50, "cameras": [{"cam_id": "cam1", "calibration": "c1.yaml"}]}
    cams = {
        "cam1": {"fps_median": 29.97, "resolution": [1920, 1080], "time_offset_ms": 0},
        "cam2": {"fps_median": 30.0, "resolution": [1920, 1080], "time_offset_ms": -350},
    }
    from pathlib import Path

    out = session_cameras_yaml(base, cams, {"cam1": Path("s/cam1.mp4"), "cam2": Path("s/cam2.mp4")})
    assert out["sync_tol_ms"] == 50
    c1, c2 = out["cameras"]
    assert c1["calibration"] == "c1.yaml" and c1["fps"] == 30 and c1["uri"] == "s/cam1.mp4"
    assert c2["calibration"] == "configs/calibration/cam2.yaml" and c2["time_offset_ms"] == -350


def _write_video(path, lag_s, dur_s=12.0, fps=25):
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, (96, 128, 3), dtype=np.uint8)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (128, 96))
    for n in range(int(dur_s * fps)):
        t = n / fps + lag_s
        w.write(base // 6 if (2 <= t < 3) or (9 <= t < 10) else base)
    w.release()


def test_session_check_end_to_end(tmp_path, capsys):
    _write_video(tmp_path / "cam1.mp4", 0.0)
    _write_video(tmp_path / "cam2.mp4", 0.6)
    out_yaml = tmp_path / "cameras.yaml"
    session_check_main(
        [
            "--session",
            str(tmp_path),
            "--search-s",
            "5",
            "--end-s",
            "4",
            "--calib-s",
            "0.5",
            "--out-cameras",
            str(out_yaml),
        ]
    )
    rep = json.loads((tmp_path / "session_check.json").read_text(encoding="utf-8"))
    assert rep["cams"]["cam1"]["sync_start_ms"] == 3000
    assert rep["cams"]["cam2"]["time_offset_ms"] == 600
    assert rep["cams"]["cam2"]["drift_ms"] == 0.0
    assert not rep["cams"]["cam1"]["vfr"]
    assert (tmp_path / "calib" / "cam2.jpg").exists()
    assert "time_offset_ms: 600" in out_yaml.read_text(encoding="utf-8")
