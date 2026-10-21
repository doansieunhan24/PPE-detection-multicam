"""Smoke test of the whole eval chain on a synthetic labelled 2-camera session."""

import json
import shutil
from pathlib import Path

import yaml

from eval.run_baselines import main
from src.common.jsonl import write_frames
from tests.test_engine import _frame

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    cfg = tmp_path / "configs"
    shutil.copytree(CONFIGS, cfg)
    for cam in ("cam1", "cam2"):
        (cfg / "calibration" / f"{cam}.yaml").write_text(
            yaml.safe_dump({"cam_id": cam, "H": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]]}),
            encoding="utf-8",
        )
    cams = yaml.safe_load((cfg / "cameras.yaml").read_text(encoding="utf-8"))
    for c in cams["cameras"]:
        c["calibration"] = str(cfg / "calibration" / f"{c['cam_id']}.yaml")
    (cfg / "cameras.yaml").write_text(yaml.safe_dump(cams), encoding="utf-8")
    zones = {"association": {"d_max_m": 0.8}, "default_required_ppe": ["helmet"], "zones": []}
    (cfg / "zones.yaml").write_text(yaml.safe_dump(zones), encoding="utf-8")

    # 0–10 s helmet worn (cam1 sees the back), 10–20 s helmet removed (real violation)
    sess = tmp_path / "jsonl" / "s01"
    sess.mkdir(parents=True)
    ts = range(0, 20000, 40)
    write_frames(sess / "cam1.jsonl", [_frame("cam1", t, 0.05, False) for t in ts])
    write_frames(sess / "cam2.jsonl", [_frame("cam2", t, 0.9, t < 10000) for t in ts])

    gt = tmp_path / "gt" / "s01"
    gt.mkdir(parents=True)
    (gt / "tracks.csv").write_text(
        "cam_id,track_id,first_ms,last_ms,n_frames,gt_id\ncam1,3,0,0,0,1\ncam2,3,0,0,0,1\n",
        encoding="utf-8",
    )
    (gt / "intervals.csv").write_text(
        "gt_id,type,t_start_ms,t_end_ms,note\n1,PPE_HELMET,10000,20000,\n", encoding="utf-8"
    )
    return cfg, sess, gt.parent


def test_run_baselines_end_to_end(tmp_path):
    cfg, sess, gt = _setup(tmp_path)
    out = tmp_path / "report"
    main(["--sessions", str(sess), "--gt", str(gt), "--out", str(out), "--config", str(cfg),
          "--ablations"])  # fmt: skip
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))["results"]
    assert res["P"]["tp"] == 1 and res["P"]["fp"] == 0
    assert res["B1"]["fp"] >= 1  # single back-facing cam alarms during the compliant phase
    assert res["B2"]["tp"] == 1
    assert set(res) >= {"B0", "B1", "B2", "B3", "B4", "P", "P-no_v"}
    assert (out / "results.md").read_text(encoding="utf-8").startswith("<!--")
