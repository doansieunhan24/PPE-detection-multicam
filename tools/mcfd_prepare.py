"""Multiple Cameras Fall Dataset (Montréal) → PerceptionFrame JSONL + fall GT (§7.7 multi-view).

Auvinet, Rougier, Meunier, St-Arnaud, Rousseau, "Multiple cameras fall dataset", Technical
report 1350, DIRO - Université de Montréal, 2010 (+ erratum Feng & Meunier, 2025-11-28).
https://www.iro.umontreal.ca/~labimage/Dataset/ — citation requested; licence not stated
(TODO(verify)). 24 scenarios × 8 wall-mounted wide-angle cameras (2.8 m, room 9 × 4.7 m),
720×480. One actor. Scenarios 1–22 contain falls, 23–24 only confounding events.

Facts used here (checked 2026-10-25):
- Real frame rate 30 fps: the AVI headers say 120 but the actor walks 5.0 m in 137 frames
  (1.1 m/s at 30 fps, 4.4 m/s at 120) — measured with the report's calibration of cam 2.
- Sync: delay_c = Nframe_c − min_c Nframe_c (cameras stopped together, report §4.2); frame f
  of camera c ↔ common frame f − delay_c. Delays are recomputed from the videos and checked
  against the report table (Fig. 3, erratum for scenario 5 cam 5).
- GT: per-frame position codes in the reference camera's frames (report p. 17–22, erratum:
  reference cam 11 → 8, 10 → 7, corrected ranges for scenarios 2, 7, 8, 14, 18).
  Fall episode = code 2 (falling) immediately followed by code 3 (lying on the ground).

    python -m tools.mcfd_prepare --root data/external/mcfd      # unzip, check sync, pose

Writes <root>/jsonl/<scenario>/cam<k>.jsonl (ts in common time, only the labelled span) and
<root>/gt.json: {seq: {"kind", "fps", "span_ms": [a, b], "falls": [[a, b], ...], "fall"}}.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2

from src.common.config import DEFAULT_CONFIG_DIR, load_yaml
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PerceptionFrame

log = get_logger(__name__)
FPS = 30.0
N_CAMS = 8

# report Fig. 3 (scenario → delay of cam 1..8), erratum: scenario 5 cam 5 = 18
DELAYS = {
    1: [3, 3, 8, 4, 23, 6, 6, 0],
    2: [25, 40, 0, 16, 18, 33, 33, 6],
    3: [12, 16, 8, 16, 35, 20, 20, 0],
    4: [72, 79, 78, 0, 68, 82, 83, 56],
    5: [17, 24, 5, 11, 18, 26, 28, 0],
    6: [0, 100, 106, 90, 89, 103, 104, 89],
    7: [28, 14, 16, 0, 1, 17, 18, 20],
    8: [92, 79, 0, 81, 64, 81, 82, 56],
    9: [18, 9, 1, 19, 13, 11, 12, 0],
    10: [14, 15, 19, 33, 12, 17, 19, 0],
    11: [23, 4, 20, 14, 0, 6, 7, 12],
    12: [21, 6, 13, 8, 0, 3, 7, 0],
    13: [16, 33, 0, 7, 27, 27, 36, 13],
    14: [49, 36, 38, 0, 29, 29, 7, 14],
    15: [15, 19, 19, 15, 34, 40, 23, 0],
    16: [23, 29, 0, 2, 12, 9, 3, 3],
    17: [21, 26, 15, 0, 10, 0, 29, 18],
    18: [99, 105, 86, 0, 84, 108, 109, 77],
    19: [19, 27, 16, 19, 5, 29, 0, 20],
    20: [25, 9, 3, 10, 10, 4, 5, 0],
    21: [20, 30, 22, 3, 8, 33, 32, 0],
    22: [0, 46, 51, 41, 53, 46, 47, 34],
    23: [31, 52, 52, 45, 54, 60, 50, 0],
    24: [3, 36, 7, 0, 37, 10, 33, 1],
}

# (scenario, cam) left out: chute24 frame counts give delays [32, 65, 36, 29, 66, 39, 63, 0],
# i.e. the report's values + 29 for cams 1–7 but not for cam 8 (truncated video?). No fall in 24.
SKIP = {(24, 8)}

# scenario → (reference camera, labelled span [first, last], fall episodes [(code-2 start,
# code-3 end)]) in reference-camera frames; erratum applied (ref 11 → 8, 10 → 7; ranges 2, 7,
# 8, 14, 18). Scenario 23 has code-2 segments followed by "moving up", not lying → no fall.
GT = {
    1: (8, (874, 1285), [(1080, 1285)]),
    2: (4, (308, 600), [(395, 600)]),
    3: (8, (380, 784), [(591, 784)]),
    4: (6, (230, 780), [(288, 380), (601, 780)]),
    5: (8, (288, 450), [(311, 450)]),
    6: (1, (325, 750), [(583, 750)]),
    7: (6, (330, 680), [(491, 680)]),
    8: (4, (144, 520), [(351, 520)]),
    9: (8, (310, 760), [(628, 760)]),
    10: (8, (315, 680), [(512, 680)]),
    11: (7, (378, 600), [(464, 600)]),
    12: (8, (355, 770), [(605, 770)]),
    13: (4, (301, 960), [(823, 960)]),
    14: (6, (372, 1115), [(1020, 1115)]),
    15: (8, (363, 870), [(755, 870)]),
    16: (4, (380, 1000), [(891, 1000)]),
    17: (6, (251, 860), [(730, 860)]),
    18: (6, (301, 740), [(685, 740)]),
    19: (7, (255, 770), [(499, 770)]),
    20: (8, (301, 800), [(545, 800)]),
    21: (8, (408, 1040), [(864, 1040)]),
    22: (1, (317, 930), [(767, 930)]),
    23: (8, (393, 5200), []),
    24: (6, (350, 3500), []),
}


def common_ms(frame: int, delay: int) -> int:
    """1-based frame of a camera with this delay → common time (ms)."""
    return round((frame - delay - 1) * 1000 / FPS)


def build_gt() -> dict:
    out = {}
    for s, (ref, (a, b), falls) in GT.items():
        d = DELAYS[s][ref - 1]
        ev = [[common_ms(x, d), common_ms(y, d)] for x, y in falls]
        out[f"chute{s:02d}"] = {
            "kind": "fall" if ev else "adl",
            "fps": FPS,
            "ref_cam": ref,
            "span_ms": [common_ms(a, d), common_ms(b, d)],
            "falls": ev,
            "fall": ev[0] if ev else None,
        }
    return out


def frame_counts(video_dir: Path) -> list[int]:
    out = []
    for c in range(1, N_CAMS + 1):
        cap = cv2.VideoCapture(str(video_dir / f"cam{c}.avi"))
        out.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    return out


def run_pose(video: Path, out: Path, cam: str, delay: int, span_ms, cfg: dict) -> int:
    from src.perception_offline.ultralytics_runner import UltralyticsPerception

    model = UltralyticsPerception(cfg)
    cap = cv2.VideoCapture(str(video))
    out.parent.mkdir(parents=True, exist_ok=True)
    n = f = 0
    with open(out, "w", encoding="utf-8") as fh:
        while True:
            ok = cap.grab()
            if not ok:
                break
            f += 1
            ts = common_ms(f, delay)
            if ts < span_ms[0]:
                continue
            if ts > span_ms[1]:
                break
            _, img = cap.retrieve()
            pf = PerceptionFrame(
                cam_id=cam,
                frame_idx=f - 1,
                ts_ms=ts,
                img_w=img.shape[1],
                img_h=img.shape[0],
                persons=model.process(img),
            )
            fh.write(pf.model_dump_json() + "\n")
            n += 1
    cap.release()
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("data/external/mcfd"))
    ap.add_argument("--perception", default=str(DEFAULT_CONFIG_DIR / "perception.yaml"))
    ap.add_argument("--imgsz", type=int, default=640, help="pose imgsz (videos are 720×480)")
    ap.add_argument("--scenarios", type=int, nargs="*", default=list(GT))
    ap.add_argument("--no-pose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()

    root = args.root
    gt = build_gt()
    (root / "gt.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    log.info("gt: %d scenarios, %d falls", len(gt), sum(len(g["falls"]) for g in gt.values()))

    cfg = load_yaml(args.perception)
    cfg["pose"]["imgsz"] = args.imgsz
    cfg["ppe"]["enabled"] = False
    for s in args.scenarios:
        seq = f"chute{s:02d}"
        vdir = root / "video" / seq
        if not vdir.exists():
            with zipfile.ZipFile(root / "zip" / f"{seq}.zip") as z:
                z.extractall(root / "video")
        counts = frame_counts(vdir)
        measured = [c - min(counts) for c in counts]
        if measured != DELAYS[s]:
            log.warning("%s: delays from frame counts %s != report %s", seq, measured, DELAYS[s])
        gt[seq]["frame_counts"] = counts
        if args.no_pose:
            continue
        for c in range(1, N_CAMS + 1):
            if (s, c) in SKIP:
                continue
            out = root / "jsonl" / seq / f"cam{c}.jsonl"
            if out.exists() and out.stat().st_size > 0:
                continue
            n = run_pose(
                vdir / f"cam{c}.avi", out, f"cam{c}", DELAYS[s][c - 1], gt[seq]["span_ms"], cfg
            )
            log.info("%s cam%d: %d frames", seq, c, n)
    (root / "gt.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
