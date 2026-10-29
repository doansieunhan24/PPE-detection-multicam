"""Live analytics (CLAUDE.md §4, P7): PerceptionFrames from DeepStream over ZMQ → the same
Analytics as the offline runner → alerts (SQLite, Telegram, smart-record trigger back to DS).

    # with ds_app --zmq-pub tcp://*:5555 --sr-bind tcp://*:5556 (docker/README.md)
    python -m src.analytics_live --zmq tcp://localhost:5555 --sr tcp://localhost:5556 \\
        --session data/rehearsal/wt3/jsonl --out data/results/live
    # without DeepStream: replay a JSONL session in real time
    python -m src.analytics_live --replay data/rehearsal/wt3/jsonl --out data/results/live

Frames are grouped across cameras by LiveSyncer (streaming version of sync_groups). Writes
<out>/alerts.jsonl (+ latency_ms = emit wall time − t_raise_ms; for RTSP sources ts_ms is the
probe's wall clock, so this is the end-to-end alert latency of §8 minus the T_alert hold),
<out>/states_latest.json (dashboard), optional <out>/states.jsonl, <out>/live_stats.json.
With smart record, each recorded alert also gets a snapshot with overlay taken from its clip
(src/alerts/snapshot.py) → <evidence>/<alert_id>_<cam>.jpg, SQLite evidence updated.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.alerts.notifier_telegram import TelegramNotifier
from src.alerts.snapshot import SnapshotMaker
from src.alerts.store import AlertStore
from src.analytics_runner import Analytics, git_commit
from src.common.config import DEFAULT_CONFIG_DIR
from src.common.jsonl import read_frames
from src.common.logging import get_logger, setup_logging
from src.common.schemas import AlertEvent, PerceptionFrame
from src.common.time_sync import LiveSyncer
from src.pipeline.smart_record import clip_name

log = get_logger(__name__)


def now_ms() -> int:
    return int(time.time() * 1000)


def zmq_frames(endpoint: str, poll_ms: int = 100) -> Iterator[PerceptionFrame | None]:
    """Frames from ds_app's PUB socket; None every poll_ms without data (lets the loop flush)."""
    import zmq

    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"frame")
    sock.connect(endpoint)
    log.info("subscribed to %s", endpoint)
    try:
        while True:
            if sock.poll(poll_ms, zmq.POLLIN):
                _topic, payload = sock.recv_multipart()
                yield PerceptionFrame.model_validate_json(payload)
            else:
                yield None
    finally:
        sock.close(0)


def replay_frames(
    jsonl_dir: Path, speed: float = 1.0, restamp: bool = True
) -> Iterator[PerceptionFrame | None]:
    """A recorded session paced in real time (all cameras merged by ts). restamp: shift ts_ms
    to the wall clock, as RTSP sources would have, so latency numbers mean the same."""
    frames = sorted(
        (f for p in sorted(jsonl_dir.glob("*.jsonl")) for f in read_frames(p)),
        key=lambda f: f.ts_ms,
    )
    if not frames:
        raise SystemExit(f"no *.jsonl in {jsonl_dir}")
    t0_data, t0_wall = frames[0].ts_ms, now_ms()
    for f in frames:
        due = t0_wall + (f.ts_ms - t0_data) / speed
        while (wait := due - now_ms()) > 0:
            time.sleep(min(wait, 100) / 1000)
            if wait > 100:
                yield None
        if restamp:
            f = f.model_copy(update={"ts_ms": int(t0_wall + (f.ts_ms - t0_data) / speed)})
        yield f


class _Worker:
    """Background thread for slow side effects (Telegram HTTP) so the loop never blocks."""

    def __init__(self, fn) -> None:
        self.q: queue.Queue = queue.Queue(maxsize=100)
        self.fn = fn
        threading.Thread(target=self._run, daemon=True, name="notify").start()

    def put(self, item) -> None:
        try:
            self.q.put_nowait(item)
        except queue.Full:
            log.warning("notification queue full, dropped %s", getattr(item, "alert_id", "?"))

    def _run(self) -> None:
        while True:
            self.fn(self.q.get())


class TriggerClient:
    """PUSH alerts to ds_app --sr-bind (smart record + OSD overlay)."""

    def __init__(self, endpoint: str) -> None:
        import zmq

        self.sock = zmq.Context.instance().socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 100)
        self.sock.setsockopt(zmq.LINGER, 500)
        self.sock.connect(endpoint)
        self._zmq = zmq

    def send(self, a: AlertEvent, record: bool) -> None:
        msg = a.model_dump(mode="json") | {"record": record}
        try:
            self.sock.send(json.dumps(msg).encode(), self._zmq.NOBLOCK)
        except self._zmq.Again:
            log.warning("trigger channel full (ds_app not connected?), alert %s", a.alert_id)

    def close(self) -> None:
        self.sock.close()


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _pct(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=float)
    return {
        "n": len(xs),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def run_live(
    source: Iterator[PerceptionFrame | None],
    an: Analytics,
    out_dir: Path,
    store: AlertStore | None = None,
    notifier=None,
    trigger: TriggerClient | None = None,
    save_states: bool = False,
    max_groups: int | None = None,
    snapshots: SnapshotMaker | None = None,
    duration_s: float | None = None,
) -> dict:
    live_cfg = an.alerts_cfg.get("live", {})
    sr_cfg = an.alerts_cfg.get("smart_record", {})
    cams = sorted(an.cams_cfg.get("cameras", []), key=lambda c: -c.get("fps", 25))
    syncer = LiveSyncer(
        [c["cam_id"] for c in cams],
        an.cams_cfg.get("sync_tol_ms", 50),
        live_cfg.get("sync_max_wait_ms", 500),
        live_cfg.get("stale_ms", 2000),
    )
    snap_every = live_cfg.get("snapshot_every_ms", 500)
    record_min = sr_cfg.get("min_level", 2)
    record = trigger is not None and sr_cfg.get("enabled", False)
    out_dir.mkdir(parents=True, exist_ok=True)
    notify = _Worker(notifier.send) if notifier is not None else None

    frame_lat, step_ms, alert_lat = [], [], []
    n_frames = n_groups = n_alerts = 0
    latest: dict[int, dict] = {}
    t_snap = 0
    t_start = now_ms()
    fa = open(out_dir / "alerts.jsonl", "a", encoding="utf-8")
    fs = open(out_dir / "states.jsonl", "a", encoding="utf-8") if save_states else None

    def process(group: dict) -> None:
        nonlocal n_groups, n_alerts
        ts = min(f.ts_ms for f in group.values())
        t_step = time.perf_counter()
        states, alerts, gid_of = an.step(group, ts)
        step_ms.append((time.perf_counter() - t_step) * 1000)
        if snapshots is not None:
            snapshots.add_group(group, gid_of)
        n_groups += 1
        for st in states:
            latest[st.global_id] = st.model_dump(mode="json")
            if fs is not None:
                fs.write(st.model_dump_json() + "\n")
        for a in alerts:
            rec = record and a.level >= record_min and bool(a.cams)
            if rec:
                a.evidence.clip = clip_name(a.alert_id, a.cams[0])
                if snapshots is not None:
                    snapshots.request(a, a.evidence.clip)
            lat = now_ms() - a.t_raise_ms
            alert_lat.append(lat)
            fa.write(json.dumps(a.model_dump(mode="json") | {"latency_ms": lat}) + "\n")
            fa.flush()
            if store is not None:
                store.add(a)
            if trigger is not None:
                trigger.send(a, rec)
            if notify is not None:
                notify.put(a)
            n_alerts += 1
            log.info(
                "ALERT L%d %s gid=%s cams=%s latency %d ms",
                a.level,
                a.type.value,
                a.global_id,
                a.cams,
                lat,
            )

    def snapshot() -> None:
        newest = max(d["ts_ms"] for d in latest.values())
        for g in [g for g, d in latest.items() if newest - d["ts_ms"] > 2000]:
            del latest[g]
        _write_atomic(out_dir / "states_latest.json", json.dumps(list(latest.values())))

    try:
        for item in source:
            t = now_ms()
            if item is not None:
                frame_lat.append(t - item.ts_ms)
                syncer.push(item, t)
                n_frames += 1
            for group in syncer.pop_ready(t):
                process(group)
            if t - t_snap >= snap_every and latest:
                snapshot()
                t_snap = t
            if max_groups is not None and n_groups >= max_groups:
                break
            if duration_s is not None and t - t_start >= duration_s * 1000:
                log.info("duration %.0f s reached", duration_s)
                break
        else:  # source exhausted (replay): what is still buffered
            for group in syncer.pop_ready(now_ms(), flush=True):
                process(group)
            if latest:
                snapshot()
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        fa.close()
        if fs is not None:
            fs.close()
    stats = {
        "commit": git_commit(),
        "frames": n_frames,
        "groups": n_groups,
        "alerts": n_alerts,
        "out_of_order_frames": syncer.out_of_order,
        "frame_latency_ms": _pct(frame_lat),
        "step_ms": _pct(step_ms),
        "alert_latency_ms": _pct(alert_lat),
    }
    if snapshots is not None:
        snapshots.close(snapshots.timeout_s)
        stats["snapshots"] = {
            "requested": len(snapshots.done),
            "written": sum(d["snapshot"] is not None for d in snapshots.done),
        }
    (out_dir / "live_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("live: %s", json.dumps(stats))
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zmq", help="ds_app --zmq-pub endpoint, e.g. tcp://localhost:5555")
    src.add_argument("--replay", help="JSONL session dir replayed in real time (no DeepStream)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed")
    ap.add_argument("--sr", help="ds_app --sr-bind endpoint, e.g. tcp://localhost:5556")
    ap.add_argument(
        "--session", help="dir with session cameras.yaml / zones.yaml (default: --replay)"
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-db", action="store_true", help="do not write the SQLite alert store")
    ap.add_argument("--save-states", action="store_true", help="also append states.jsonl")
    ap.add_argument("--no-snapshot", action="store_true", help="no snapshot from the clips")
    ap.add_argument("--duration-s", type=float, help="stop after this long (then finish snapshots)")
    args = ap.parse_args(argv)
    setup_logging()

    session = Path(args.session) if args.session else (Path(args.replay) if args.replay else None)
    an = Analytics(Path(args.config), session)
    source = zmq_frames(args.zmq) if args.zmq else replay_frames(Path(args.replay), args.speed)
    store = None if args.no_db else AlertStore(an.alerts_cfg["store"]["sqlite_path"])
    trigger = TriggerClient(args.sr) if args.sr else None
    sr_cfg = an.alerts_cfg.get("smart_record", {})
    snapshots = (
        SnapshotMaker(Path(sr_cfg["out_dir"]), sr_cfg["pre_s"], sr_cfg["post_s"], store)
        if trigger is not None and sr_cfg.get("enabled") and not args.no_snapshot
        else None
    )
    try:
        run_live(
            source,
            an,
            Path(args.out),
            store=store,
            notifier=TelegramNotifier.from_config(an.alerts_cfg),
            trigger=trigger,
            save_states=args.save_states,
            snapshots=snapshots,
            duration_s=args.duration_s,
        )
    finally:
        if trigger is not None:
            trigger.close()


if __name__ == "__main__":
    main()
