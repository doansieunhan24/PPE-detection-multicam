"""Estimate per-camera time offsets from a flash / light switch (CLAUDE.md §9.1).

Sync event: the frame with the largest rise in mean brightness (lights back on, flash) within a
search window. Times are container pts, so variable-frame-rate phone videos are handled.
Prints time_offset_ms per camera relative to --ref (paste into configs/cameras.yaml).
With --end-s the event is also searched at the end of each video → clock drift per camera.
tools/session_check.py runs this as part of a full session check.

    python tools/sync_offsets.py --videos cam1=data/raw/s01/cam1.mp4 cam2=... cam3=... --ref cam1
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

import cv2
import numpy as np


def find_light_on(times_ms: Sequence[float], means: Sequence[float]) -> tuple[float, float]:
    """Time of the largest brightness rise and its strength (jump / median |diff|)."""
    if len(means) < 2:
        raise ValueError("need >= 2 frames")
    d = np.diff(np.asarray(means, dtype=float))
    i = int(np.argmax(d))
    noise = float(np.median(np.abs(d))) + 0.05
    return float(times_ms[i + 1]), float(d[i] / noise)


def scan_brightness(
    path: str, windows_ms: Sequence[tuple[float, float]] = ()
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """One pass over a video: pts of every frame, (pts, mean gray) inside each window.

    Window bounds < 0 count from the end of the video (resolved after the pass, so frames
    are buffered as a cheap 1/8-scale brightness only).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    pts: list[float] = []
    means: list[float] = []
    prev = -1.0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not t > prev:
            t = prev + 1000.0 / fps if prev >= 0 else 0.0
        prev = t
        pts.append(t)
        _, f = cap.retrieve()
        small = cv2.resize(f, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
        means.append(float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()))
    cap.release()
    t_all, m_all = np.array(pts), np.array(means)
    dur = t_all[-1] if len(t_all) else 0.0
    out = []
    for a, b in windows_ms:
        a = a + dur if a < 0 else a
        b = b + dur if b < 0 else b
        sel = (t_all >= a) & (t_all <= b)
        out.append((t_all[sel], m_all[sel]))
    return t_all, out


def offsets_and_drift(
    events: Mapping[str, tuple[float, float | None]], ref: str
) -> dict[str, dict[str, float | None]]:
    """events: cam -> (t_start_event_ms, t_end_event_ms or None), in each cam's media time.

    time_offset_ms = t_ref − t_cam (added to the cam's ts so the start events coincide).
    drift_ms = how much longer the cam's clock measured start→end than the ref's (None if no
    end event on either side). Positive drift: cam runs fast.
    """
    if ref not in events:
        raise ValueError(f"ref {ref!r} has no sync event")
    rs, re_ = events[ref]
    out: dict[str, dict[str, float | None]] = {}
    for cam, (s, e) in events.items():
        drift = None if e is None or re_ is None else round((e - s) - (re_ - rs), 1)
        out[cam] = {"time_offset_ms": round(rs - s), "drift_ms": drift}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--videos", nargs="+", required=True, help="cam_id=path")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--search-s", type=float, default=20.0)
    ap.add_argument("--end-s", type=float, default=0.0, help="also search the last N s")
    args = ap.parse_args(argv)
    events: dict[str, tuple[float, float | None]] = {}
    for item in args.videos:
        cam, _, path = item.partition("=")
        wins = [(0.0, args.search_s * 1000)]
        if args.end_s > 0:
            wins.append((-args.end_s * 1000, float("inf")))
        _, scans = scan_brightness(path, wins)
        t_s, k_s = find_light_on(*scans[0])
        t_e, k_e = find_light_on(*scans[1]) if len(scans) > 1 else (None, 0.0)
        events[cam] = (t_s, t_e)
        end = "" if t_e is None else f", end event at {t_e:.0f} ms (strength {k_e:.1f})"
        print(f"{cam}: start event at {t_s:.0f} ms (strength {k_s:.1f}){end}")
    print("\ntime_offset_ms (add to ts so all cams align to", args.ref, "):")
    for cam, r in offsets_and_drift(events, args.ref).items():
        drift = "" if r["drift_ms"] is None else f"   drift {r['drift_ms']:+.0f} ms"
        print(f"  {cam}: {r['time_offset_ms']}{drift}")


if __name__ == "__main__":
    main()
