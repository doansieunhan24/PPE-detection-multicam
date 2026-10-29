"""Live path without DeepStream: streaming sync, smart-record bookkeeping, OSD overlay,
ZMQ transport, live runner vs offline runner."""

import json
import time

import pytest

from src.analytics_live import replay_frames, run_live, zmq_frames
from src.analytics_runner import Analytics, run
from src.association.ground_plane import sync_groups
from src.common.config import DEFAULT_CONFIG_DIR
from src.common.schemas import PerceptionFrame, PersonDet
from src.common.time_sync import LiveSyncer
from src.dashboard.app import latest_states
from src.pipeline.ds_app import AlertOverlay
from src.pipeline.meta_export import ZmqPublisher
from src.pipeline.smart_record import EvidenceBook, clip_name, publish_clip


def _fr(cam, ts, persons=()):
    return PerceptionFrame(
        cam_id=cam, frame_idx=ts // 40, ts_ms=ts, img_w=1920, img_h=1080, persons=list(persons)
    )


def _key(groups):
    return [{c: f.ts_ms for c, f in g.items()} for g in groups]


def test_live_syncer_matches_offline_sync_groups():
    streams = {
        "cam1": [_fr("cam1", t) for t in range(0, 2000, 40)],
        "cam2": [_fr("cam2", t + 13) for t in range(0, 2000, 40)],
        "cam3": [_fr("cam3", t + 30) for t in range(0, 2000, 80)],  # half rate
    }
    ref = list(sync_groups(streams, 25))
    s = LiveSyncer(["cam1", "cam2", "cam3"], 25, max_wait_ms=10**9, stale_ms=10**9)
    # arrival interleaved and out of step between cameras (cam3 lags 300 ms)
    events = sorted(
        ((f.ts_ms + (300 if c == "cam3" else 0), f) for c, fr in streams.items() for f in fr),
        key=lambda e: e[0],
    )
    got = []
    for arrive, f in events:
        s.push(f, arrive)
        got += s.pop_ready(arrive)
    got += s.pop_ready(10**12, flush=True)
    assert _key(got) == _key(ref)


def test_live_syncer_waits_then_times_out():
    s = LiveSyncer(["a", "b"], 20, max_wait_ms=100, stale_ms=10_000)
    s.push(_fr("b", 0), 0)
    s.push(_fr("a", 40), 0)  # b has nothing ≥ 60 yet → wait
    assert s.pop_ready(50) == []
    out = s.pop_ready(100)  # waited max_wait_ms → emitted without b
    assert _key(out) == [{"a": 40}]


def test_live_syncer_skips_stale_reference():
    s = LiveSyncer(["a", "b"], 20, max_wait_ms=1000, stale_ms=500)
    s.push(_fr("a", 0), 0)
    s.push(_fr("b", 0), 0)
    s.push(_fr("b", 40), 0)  # b is past 0 + tol: frame a@0 is final
    assert _key(s.pop_ready(0)) == [{"a": 0, "b": 0}]
    for t in (80, 120):  # camera a dies, b goes on: b becomes the reference
        s.push(_fr("b", t), 1000 + t)
    assert s.reference(1100) == "b"
    assert _key(s.pop_ready(1300)) == [{"b": 40}, {"b": 80}, {"b": 120}]


def test_live_syncer_drops_backwards_frames():
    s = LiveSyncer(["a"], 20)
    s.push(_fr("a", 100), 0)
    s.push(_fr("a", 60), 0)
    assert s.out_of_order == 1


def test_evidence_book_shares_running_recording():
    b = EvidenceBook(["cam1", "cam2"])
    assert b.request("cam1", "A1") is True  # start recording
    assert b.request("cam1", "A2") is False  # already recording: same clip
    assert b.request("camX", "A3") is False  # unknown camera
    assert b.done("cam1") == ["A1", "A2"]
    assert b.request("cam1", "A4") is True
    b.failed("cam1")
    assert b.request("cam1", "A5") is True


def test_publish_clip_names(tmp_path):
    rec = tmp_path / "_sr"
    rec.mkdir()
    f = rec / "cam1_0_00000_20261027-021602_41.mp4"
    f.write_bytes(b"mp4")
    out = publish_clip(f, tmp_path, [clip_name("A1", "cam1"), clip_name("A2", "cam1")])
    assert [p.name for p in out] == ["A1_cam1.mp4", "A2_cam1.mp4"]
    assert all(p.read_bytes() == b"mp4" for p in out) and not f.exists()


def test_alert_overlay_expires():
    o = AlertOverlay(hold_s=5)
    o.add({"type": "PPE_HELMET", "level": 2, "global_id": 3, "cams": ["cam1"]}, now=0)
    assert o.active("cam1", now=1) == [(2, "L2 PPE_HELMET #3")]
    assert o.active("cam2", now=1) == []
    assert o.active("cam1", now=6) == []


def test_zmq_roundtrip():
    pytest.importorskip("zmq")
    pub = ZmqPublisher("tcp://127.0.0.1:55871")
    sub = zmq_frames("tcp://127.0.0.1:55871", poll_ms=50)
    got = None
    deadline = time.monotonic() + 5
    while got is None and time.monotonic() < deadline:  # PUB/SUB slow joiner: resend
        pub.put(_fr("cam1", 123))
        got = next(sub)
    pub.close()
    sub.close()
    assert got is not None and got.cam_id == "cam1" and got.ts_ms == 123


def _session(tmp_path):
    """2 uncalibrated cams, one person without helmet for 8 s → PPE alerts."""
    d = tmp_path / "jsonl"
    d.mkdir()
    (d / "cameras.yaml").write_text(
        "sync_tol_ms: 50\ncameras:\n"
        "  - {cam_id: cam1, uri: x, fps: 25, resolution: [1920, 1080]}\n"
        "  - {cam_id: cam2, uri: y, fps: 25, resolution: [1920, 1080]}\n",
        encoding="utf-8",
    )
    k = [(150.0, 150.0 + 20 * i, 0.9) for i in range(17)]
    p = PersonDet(track_id=1, bbox=(100, 100, 200, 500), conf=0.9, kpts=k, ppe_checked=True)
    for cam, off in (("cam1", 0), ("cam2", 7)):
        with open(d / f"{cam}.jsonl", "w", encoding="utf-8") as f:
            for t in range(0, 8000, 40):
                f.write(_fr(cam, t + off, [p]).model_dump_json() + "\n")
    return d


def test_live_runner_matches_offline(tmp_path):
    d = _session(tmp_path)
    off = run(d, tmp_path / "off", DEFAULT_CONFIG_DIR, "P", {})
    stats = run_live(replay_frames(d, speed=1e6, restamp=False), Analytics(DEFAULT_CONFIG_DIR, d),
                     tmp_path / "live")  # fmt: skip
    assert stats["groups"] == off["groups"]
    rows = lambda p: [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]  # noqa: E731
    a_off, a_live = (
        rows(tmp_path / "off" / "alerts.jsonl"),
        rows(tmp_path / "live" / "alerts.jsonl"),
    )
    assert off["alerts"] > 0
    key = lambda a: (a["type"], a["global_id"], a["t_raise_ms"])  # noqa: E731
    assert [key(a) for a in a_live] == [key(a) for a in a_off]
    snap = latest_states(tmp_path / "live")  # dashboard reads states_latest.json from the dir
    assert sorted(s["global_id"] for s in snap) == [1, 2]  # uncalibrated: one gid per cam
