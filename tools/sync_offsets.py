"""Estimate per-camera time offsets from a flash / clap at session start (CLAUDE.md §9.1).

Flash: the frame with the largest jump in mean brightness within the first --search-s seconds.
Prints time_offset_ms per camera relative to --ref (paste into configs/cameras.yaml).

    python tools/sync_offsets.py --videos cam1=data/raw/s01/cam1.mp4 cam2=... cam3=... --ref cam1
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def flash_time_ms(path: str, search_s: float) -> tuple[float, float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    means = []
    for _ in range(int(search_s * fps)):
        ok, f = cap.read()
        if not ok:
            break
        means.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
    cap.release()
    if len(means) < 2:
        raise SystemExit(f"{path}: too few frames")
    jumps = np.diff(np.array(means))
    i = int(np.argmax(jumps)) + 1
    return i * 1000.0 / fps, float(jumps[i - 1])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--videos", nargs="+", required=True, help="cam_id=path")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--search-s", type=float, default=20.0)
    args = ap.parse_args(argv)
    t = {}
    for item in args.videos:
        cam, _, path = item.partition("=")
        t[cam], jump = flash_time_ms(path, args.search_s)
        print(f"{cam}: flash at {t[cam]:.0f} ms (brightness jump {jump:.1f})")
    print("\ntime_offset_ms (add to ts so all cams align to", args.ref, "):")
    for cam, v in t.items():
        print(f"  {cam}: {round(t[args.ref] - v)}")


if __name__ == "__main__":
    main()
