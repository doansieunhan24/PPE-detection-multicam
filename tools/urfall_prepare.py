"""UR Fall Detection Dataset → PerceptionFrame JSONL + fall GT (CLAUDE.md §9, P6 fall module).

Dataset: Kwolek & Kępski, Computer Methods and Programs in Biomedicine 117(3), 2014.
https://fenix.ur.edu.pl/~mkepski/ds/uf.html — CC BY-NC-SA 4.0 (non-commercial academic use).
30 falls (cam0 parallel to the floor, cam1 ceiling) + 40 ADL (cam0 only). The mp4 files are
depth|RGB side by side (640×240 @ 30 fps); the right half (320×240) is RGB.
Labels (urfall-cam0-{falls,adls}.csv, one row per cam0 frame): -1 not lying, 0 falling
(transition), 1 lying on the ground.

    python -m tools.urfall_prepare --root data/external/urfall            # download + pose
    python -m tools.urfall_prepare --root data/external/urfall --cams cam0 cam1

Writes <root>/jsonl/<seq>/<cam>.jsonl and <root>/gt.json:
  {seq: {"kind": "fall"|"adl", "n_frames": n, "fps": 30, "fall": [t_start_ms, t_end_ms] | null,
         "lying_ms": [[a, b], ...]}}
fall = first "falling" frame → last "lying" frame of the first fall episode (cam0 time);
null for ADL (an ADL lying down on purpose keeps only lying_ms).
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

import cv2

from src.common.config import DEFAULT_CONFIG_DIR, load_yaml
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PerceptionFrame

log = get_logger(__name__)

BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data"
N_FALL, N_ADL = 30, 40
FPS = 30.0


def seqs() -> list[tuple[str, str]]:
    return [(f"fall-{i:02d}", "fall") for i in range(1, N_FALL + 1)] + [
        (f"adl-{i:02d}", "adl") for i in range(1, N_ADL + 1)
    ]


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("download %s", url)
    urllib.request.urlretrieve(url, path)


def runs(frames_labels: list[tuple[int, int]], wanted: set[int]) -> list[tuple[int, int]]:
    """Maximal runs [first, last] of consecutive frame numbers whose label is in wanted."""
    out: list[tuple[int, int]] = []
    for fr, lab in frames_labels:
        if lab in wanted:
            if out and out[-1][1] == fr - 1:
                out[-1] = (out[-1][0], fr)
            else:
                out.append((fr, fr))
    return out


def gt_from_labels(rows: list[tuple[int, int]], fps: float = FPS) -> dict:
    """rows: (frame_number starting at 1, label). Times are (frame − 1) / fps."""
    ms = lambda fr: round((fr - 1) * 1000 / fps)  # noqa: E731
    episodes = runs(rows, {0, 1})  # falling followed by lying
    fall = None
    for a, b in episodes:
        labs = {lab for fr, lab in rows if a <= fr <= b}
        if 1 in labs:
            fall = [ms(a), ms(b)]
            break
    return {
        "n_frames": len(rows),
        "fps": fps,
        "fall": fall,
        "lying_ms": [[ms(a), ms(b)] for a, b in runs(rows, {1})],
    }


def read_labels(path: Path) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.reader(f):
            if len(r) >= 3 and r[0].strip():
                out.setdefault(r[0].strip(), []).append((int(r[1]), int(r[2])))
    return out


def run_pose(video: Path, out: Path, cam: str, perception_cfg: dict) -> int:
    from src.perception_offline.ultralytics_runner import UltralyticsPerception

    model = UltralyticsPerception(perception_cfg)
    cap = cv2.VideoCapture(str(video))
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = frame[:, frame.shape[1] // 2 :]  # depth | RGB
            pf = PerceptionFrame(
                cam_id=cam,
                frame_idx=n,
                ts_ms=round(n * 1000 / FPS),
                img_w=rgb.shape[1],
                img_h=rgb.shape[0],
                persons=model.process(rgb),
            )
            f.write(pf.model_dump_json() + "\n")
            n += 1
    cap.release()
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("data/external/urfall"))
    ap.add_argument("--cams", nargs="+", default=["cam0"], help="cam1 exists for falls only")
    ap.add_argument("--perception", default=str(DEFAULT_CONFIG_DIR / "perception.yaml"))
    ap.add_argument("--imgsz", type=int, default=640, help="pose imgsz (RGB is 320×240)")
    ap.add_argument("--no-pose", action="store_true", help="download + GT only")
    args = ap.parse_args(argv)
    setup_logging()

    root = args.root
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        download(f"{BASE}/{name}", root / name)
    labels = read_labels(root / "urfall-cam0-falls.csv")
    labels.update(read_labels(root / "urfall-cam0-adls.csv"))
    gt = {}
    for seq, kind in seqs():
        g = gt_from_labels(labels[seq])
        if kind == "adl":  # lying down on purpose is a hard negative, not a fall
            g["fall"] = None
        gt[seq] = {"kind": kind, **g}
    (root / "gt.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    n_fall = sum(1 for g in gt.values() if g["fall"])
    log.info("gt: %d sequences, %d with a fall episode", len(gt), n_fall)

    cfg = load_yaml(args.perception)
    cfg["pose"]["imgsz"] = args.imgsz
    cfg["ppe"]["enabled"] = False
    for seq, kind in seqs():
        for cam in args.cams:
            if cam != "cam0" and kind == "adl":
                continue
            video = root / "video" / f"{seq}-{cam}.mp4"
            download(f"{BASE}/{seq}-{cam}.mp4", video)
            out = root / "jsonl" / seq / f"{cam}.jsonl"
            if args.no_pose or (out.exists() and out.stat().st_size > 0):
                continue
            n = run_pose(video, out, cam, cfg)
            log.info("%s %s: %d frames", seq, cam, n)


if __name__ == "__main__":
    main()
