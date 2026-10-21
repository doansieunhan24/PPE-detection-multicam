"""Offline evidence clips for alerts (stand-in for DeepStream smart-record, CLAUDE.md §7.8).

For every alert: cut [t_raise − pre_s, t_raise + post_s] from each camera that saw it, draw
the perception overlay, write <out>/<alert_id>_<cam>.mp4 + a snapshot at t_raise, and update
the alert's evidence in the SQLite store.

    python tools/make_evidence.py --results data/results/s01/P --jsonl data/jsonl/s01 \
        --video cam1=data/raw/s01/cam1.mp4 --video cam2=... --video cam3=...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alerts.store import AlertStore  # noqa: E402
from src.common.config import DEFAULT_CONFIG_DIR, load_config  # noqa: E402
from src.common.jsonl import read_frames  # noqa: E402
from src.common.logging import get_logger, setup_logging  # noqa: E402
from src.common.schemas import AlertEvent  # noqa: E402
from tools.replay import draw_person  # noqa: E402

log = get_logger(__name__)


def cut_clip(
    video: str, frames: list, t0: int, t1: int, out_mp4: Path, snap_ts: int, out_jpg: Path
) -> bool:
    sel = [f for f in frames if t0 <= f.ts_ms <= t1]
    if not sel:
        return False
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, sel[0].frame_idx)
    writer = None
    snap_done = False
    for fr in sel:
        ok, img = cap.read()
        if not ok:
            break
        for p in fr.persons:
            draw_person(img, p, 0.5)
        if writer is None:
            writer = cv2.VideoWriter(
                str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0])
            )
        writer.write(img)
        if not snap_done and fr.ts_ms >= snap_ts:
            cv2.imwrite(str(out_jpg), img)
            snap_done = True
    cap.release()
    if writer is not None:
        writer.release()
    return writer is not None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--video", action="append", required=True, help="cam_id=path")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    args = ap.parse_args(argv)
    setup_logging()

    cfg = load_config("alerts", Path(args.config))
    sr = cfg["smart_record"]
    out = Path(sr["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    videos = dict(v.split("=", 1) for v in args.video)
    frames = {c: list(read_frames(Path(args.jsonl) / f"{c}.jsonl")) for c in videos}
    store = AlertStore(cfg["store"]["sqlite_path"])
    n = 0
    with open(Path(args.results) / "alerts.jsonl", encoding="utf-8") as f:
        alerts = [AlertEvent.model_validate_json(x) for x in f if x.strip()]
    for a in alerts:
        t0, t1 = a.t_raise_ms - int(sr["pre_s"] * 1000), a.t_raise_ms + int(sr["post_s"] * 1000)
        for cam in a.cams or list(videos):
            if cam not in videos:
                continue
            mp4, jpg = out / f"{a.alert_id}_{cam}.mp4", out / f"{a.alert_id}_{cam}.jpg"
            if cut_clip(videos[cam], frames[cam], t0, t1, mp4, a.t_raise_ms, jpg):
                if a.evidence.clip is None:  # first camera becomes the primary evidence
                    a.evidence.clip, a.evidence.snapshot = mp4.name, jpg.name
                n += 1
        store.add(a)
    log.info("%d clips for %d alerts -> %s", n, len(alerts), out)


if __name__ == "__main__":
    main()
