"""First thing to run after a C2 recording session (docs/c2_shooting_script.md, CLAUDE.md §9.1).

For every cam*.mp4/.mov in the session folder:
- frame rate, frame count, duration, resolution; variable-frame-rate check from pts
  (phones often record VFR → the offline backend uses pts, configs/perception.yaml)
- sync events (lights back on) in the first --search-s and last --end-s seconds
  → time_offset_ms relative to --ref and clock drift over the session
- an empty-floor frame at --calib-s → <session>/calib/<cam>.jpg for calibrate_homography.py

Writes <session>/session_check.json and a cameras YAML for the session (uri = the video files,
measured fps/resolution/time_offset_ms, calibration paths kept from --cameras).

    python tools/session_check.py --session data/raw/s01 --ref cam1 \
        --out-cameras configs/cameras_s01.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import DEFAULT_CONFIG_DIR, load_yaml  # noqa: E402
from tools.sync_offsets import find_light_on, offsets_and_drift, scan_brightness  # noqa: E402

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def pts_stats(pts_ms: Sequence[float]) -> dict[str, float | int | bool]:
    """Frame-interval statistics. vfr = intervals spread > 10% of median or gaps present."""
    t = np.asarray(pts_ms, dtype=float)
    if len(t) < 3:
        raise ValueError("need >= 3 frames")
    d = np.diff(t)
    med = float(np.median(d))
    p5, p95 = (float(x) for x in np.percentile(d, [5, 95]))
    gaps = int(np.sum(d > 1.5 * med))
    duration_s = float((t[-1] - t[0]) / 1000.0)
    return {
        "n_frames": int(len(t)),
        "duration_s": round(duration_s, 3),
        "fps_median": round(1000.0 / med, 3),
        "fps_effective": round((len(t) - 1) / duration_s, 3) if duration_s > 0 else 0.0,
        "dt_p5_ms": round(p5, 2),
        "dt_p95_ms": round(p95, 2),
        "dt_max_ms": round(float(d.max()), 2),
        "gaps": gaps,
        "vfr": bool((p95 - p5) > 0.1 * med or gaps > 0),
        # drift a constant frame rate assumption would accumulate by the last frame
        "index_time_error_ms": round(float(t[-1] - t[0] - (len(t) - 1) * med), 1),
    }


def find_videos(session: Path) -> dict[str, Path]:
    vids = {
        p.stem: p
        for p in sorted(session.iterdir())
        if p.suffix.lower() in VIDEO_EXTS and p.stem.lower().startswith("cam")
    }
    if not vids:
        raise SystemExit(f"no cam*.mp4 / .mov in {session}")
    return vids


def grab_frame(path: Path, t_ms: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def session_cameras_yaml(
    base: dict[str, Any], cams: dict[str, dict[str, Any]], videos: dict[str, Path]
) -> dict[str, Any]:
    """cameras.yaml for the session: entries of base (calibration etc.) updated per video."""
    by_id = {c["cam_id"]: c for c in base.get("cameras", [])}
    out = {k: v for k, v in base.items() if k != "cameras"}
    out["cameras"] = []
    for cam_id, info in cams.items():
        entry = dict(by_id.get(cam_id, {"calibration": f"configs/calibration/{cam_id}.yaml"}))
        entry.update(
            cam_id=cam_id,
            uri=videos[cam_id].as_posix(),
            fps=round(info["fps_median"]),
            resolution=list(info["resolution"]),
            time_offset_ms=int(info.get("time_offset_ms") or 0),
        )
        out["cameras"].append(entry)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--session", required=True, help="folder with cam1.mp4, cam2.mp4, ...")
    ap.add_argument("--ref", default="cam1")
    ap.add_argument("--search-s", type=float, default=30.0, help="start sync search window")
    ap.add_argument("--end-s", type=float, default=30.0, help="end sync search (0 = off)")
    ap.add_argument("--calib-s", type=float, default=2.0, help="time of the empty-floor frame")
    ap.add_argument("--min-strength", type=float, default=8.0, help="warn below this")
    ap.add_argument("--cameras", default=str(DEFAULT_CONFIG_DIR / "cameras.yaml"))
    ap.add_argument("--out-cameras", default=None)
    args = ap.parse_args(argv)

    session = Path(args.session)
    videos = find_videos(session)
    (session / "calib").mkdir(exist_ok=True)
    report: dict[str, Any] = {"session": session.as_posix(), "ref": args.ref, "cams": {}}
    warnings: list[str] = []
    events: dict[str, tuple[float, float | None]] = {}
    for cam_id, path in videos.items():
        cap = cv2.VideoCapture(str(path))
        info: dict[str, Any] = {
            "file": path.name,
            "fps_nominal": round(cap.get(cv2.CAP_PROP_FPS), 3),
            "resolution": [
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ],
        }
        cap.release()
        wins = [(0.0, args.search_s * 1000)]
        if args.end_s > 0:
            wins.append((-args.end_s * 1000, float("inf")))
        pts, scans = scan_brightness(str(path), wins)
        info.update(pts_stats(pts))
        t_s, k_s = find_light_on(*scans[0])
        info["sync_start_ms"], info["sync_start_strength"] = round(t_s), round(k_s, 1)
        t_e = None
        if len(scans) > 1 and len(scans[1][0]) > 1:
            t_e, k_e = find_light_on(*scans[1])
            info["sync_end_ms"], info["sync_end_strength"] = round(t_e), round(k_e, 1)
            if k_e < args.min_strength:
                warnings.append(f"{cam_id}: weak end sync event (strength {k_e:.1f}), ignored")
                t_e = None
        if k_s < args.min_strength:
            warnings.append(f"{cam_id}: weak start sync event (strength {k_s:.1f})")
        events[cam_id] = (t_s, t_e)
        if info["vfr"]:
            warnings.append(
                f"{cam_id}: variable frame rate (dt p5-p95 {info['dt_p5_ms']}-"
                f"{info['dt_p95_ms']} ms, {info['gaps']} gaps); keep file_timestamps: pts"
            )
        img = grab_frame(path, args.calib_s * 1000)
        if img is not None:
            out_img = session / "calib" / f"{cam_id}.jpg"
            cv2.imwrite(str(out_img), img)
            info["calib_frame"] = out_img.as_posix()
        report["cams"][cam_id] = info

    if args.ref in events:
        for cam_id, r in offsets_and_drift(events, args.ref).items():
            report["cams"][cam_id].update(r)
            if r["drift_ms"] is not None and abs(r["drift_ms"]) > 1000.0 / 25:
                warnings.append(f"{cam_id}: clock drift {r['drift_ms']:+.0f} ms over session")
    else:
        warnings.append(f"ref {args.ref} not found among {sorted(videos)}")
    fps = {round(c["fps_median"]) for c in report["cams"].values()}
    if len(fps) > 1:
        warnings.append(f"cameras recorded at different frame rates: {sorted(fps)}")
    durs = [c["duration_s"] for c in report["cams"].values()]
    report["warnings"] = warnings

    (session / "session_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for cam_id, c in report["cams"].items():
        print(
            f"{cam_id}: {c['resolution'][0]}x{c['resolution'][1]} {c['fps_median']} fps "
            f"({'VFR' if c['vfr'] else 'CFR'}), {c['duration_s']:.1f} s, "
            f"sync {c['sync_start_ms']} ms -> offset {c.get('time_offset_ms')} ms, "
            f"drift {c.get('drift_ms')}"
        )
    print(f"duration spread {max(durs) - min(durs):.1f} s")
    for w in warnings:
        print("WARNING", w)
    if args.out_cameras:
        base = load_yaml(args.cameras)
        out = session_cameras_yaml(base, report["cams"], videos)
        Path(args.out_cameras).write_text(
            yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        print(f"cameras config -> {args.out_cameras}")
    print(f"report -> {session / 'session_check.json'}")


if __name__ == "__main__":
    main()
