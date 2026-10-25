import pytest
from pydantic import ValidationError

from src.common.config import load_config
from src.common.jsonl import read_frames, write_frames
from src.common.schemas import (
    AlertEvent,
    AlertType,
    PartName,
    PartState,
    PerceptionFrame,
    PersonPartState,
)

SAMPLE_LINE = {
    "cam_id": "cam1",
    "frame_idx": 1234,
    "ts_ms": 1729500000123,
    "img_w": 1920,
    "img_h": 1080,
    "persons": [
        {
            "track_id": 17,
            "bbox": [100, 200, 300, 700],
            "conf": 0.91,
            "kpts": [[150.0, 250.0, 0.9]] * 17,
            "ppe_dets": [{"cls": "helmet", "bbox": [140, 190, 210, 240], "conf": 0.82}],
        }
    ],
}


def test_perception_frame_parses():
    fr = PerceptionFrame.model_validate(SAMPLE_LINE)
    assert fr.persons[0].ppe_dets[0].cls == "helmet"
    assert fr.persons[0].truncated_bottom is False


def test_wrong_keypoint_count_rejected():
    bad = {**SAMPLE_LINE, "persons": [{**SAMPLE_LINE["persons"][0], "kpts": [[0, 0, 0]] * 5}]}
    with pytest.raises(ValidationError):
        PerceptionFrame.model_validate(bad)


def test_jsonl_roundtrip(tmp_path):
    fr = PerceptionFrame.model_validate(SAMPLE_LINE)
    path = tmp_path / "cam1.jsonl"
    assert write_frames(path, [fr, fr]) == 2
    assert list(read_frames(path)) == [fr, fr]


def test_part_state_defaults_unknown():
    s = PersonPartState(global_id=5, ts_ms=0, parts={PartName.HEAD: PartState(item="helmet")})
    assert s.parts[PartName.HEAD].state.value == "UNKNOWN"


def test_alert_level_range():
    with pytest.raises(ValidationError):
        AlertEvent(alert_id="a", type=AlertType.FALL, level=4, t_start_ms=0, t_raise_ms=0)


@pytest.mark.parametrize("name", ["fusion", "behavior", "alerts", "cameras", "zones"])
def test_configs_load(name):
    assert load_config(name)


def test_load_config_session_override(tmp_path):
    from src.common.config import load_config

    base, sess = tmp_path / "configs", tmp_path / "s01"
    base.mkdir()
    sess.mkdir()
    (base / "cameras.yaml").write_text("sync_tol_ms: 50\n", encoding="utf-8")
    (base / "fusion.yaml").write_text("tau_s: 2\n", encoding="utf-8")
    (sess / "cameras.yaml").write_text("sync_tol_ms: 40\n", encoding="utf-8")
    assert load_config("cameras", base)["sync_tol_ms"] == 50
    assert load_config("cameras", base, override_dir=sess)["sync_tol_ms"] == 40
    assert load_config("fusion", base, override_dir=sess)["tau_s"] == 2


def test_apply_overrides_dotted_keys():
    from src.common.config import apply_overrides

    cfg = {"evidence": {"theta_obs": 1.0, "ds": {"theta_obs": 0.5}}, "alert": {"T_alert_s": 3}}
    out = apply_overrides(cfg, {"evidence.theta_obs": 0.25, "evidence.ds.theta_obs": 0.1})
    assert out["evidence"]["theta_obs"] == 0.25 and out["evidence"]["ds"]["theta_obs"] == 0.1
    assert cfg["evidence"]["theta_obs"] == 1.0  # input untouched
    assert apply_overrides(cfg, None) == cfg
    with pytest.raises(KeyError):
        apply_overrides(cfg, {"evidence.theta_obz": 1})
