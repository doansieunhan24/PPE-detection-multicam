import numpy as np

from src.behavior.engine import BehaviorEngine
from src.behavior.seq_features import FEAT_DIM, build_model, flip, frame_features, window
from src.common.config import apply_overrides, load_config
from src.common.schemas import AlertType, PerceptionFrame, PersonDet


def _person(x=100.0, top=50.0, h=200.0, tid=1):
    k = [(x + 10 * (i % 2), top + h * i / 17, 0.9) for i in range(17)]
    return PersonDet(track_id=tid, bbox=(x - 40, top, x + 40, top + h), conf=0.9, kpts=k)


def test_frame_features_normalized_and_missing():
    p = _person()
    f = frame_features(p, None)
    assert f.shape == (FEAT_DIM,) and f[-1] == 1.0 and f[-4] == 80 / 200
    assert abs(f[1] - (50 - 150) / 200) < 1e-6  # nose y relative to bbox centre / height
    moved = frame_features(_person(top=70.0), p)
    assert abs(moved[-3] - 20 / 200) < 1e-6  # vertical centre velocity / previous height
    assert not frame_features(None, p).any()


def test_window_pads_front_and_flip_is_involution():
    seq = [frame_features(_person(top=50.0 + i), None) for i in range(3)]
    w = window(seq, 5)
    assert w.shape == (FEAT_DIM, 5) and not w[:, :2].any() and w[-1, 2:].all()
    np.testing.assert_array_equal(flip(flip(w)), w)
    assert np.allclose(flip(w)[0], -w[0])  # nose x mirrored


def test_engine_seq_method_runs(tmp_path):
    import torch

    torch.manual_seed(0)
    path = tmp_path / "m.pt"
    torch.save({"state_dict": build_model().state_dict(), "seq_len": 8}, path)
    cfg = apply_overrides(
        load_config("behavior"), {"fall.method": "seq", "fall.seq_model": str(path)}
    )
    eng = BehaviorEngine(cfg, {"zones": []})
    alerts = []
    for ts in range(0, 2000, 40):
        fr = PerceptionFrame(
            cam_id="c", frame_idx=ts // 40, ts_ms=ts, img_w=640, img_h=480, persons=[_person()]
        )
        alerts += eng.step({"c": fr}, ts, {("c", 1): 1}, {})
    assert all(a.type in (AlertType.FALL, AlertType.IMMOBILE) for a in alerts)
    assert len(eng.tracks[("c", 1)].seq) == 8
