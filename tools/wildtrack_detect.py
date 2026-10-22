"""Run the offline perception backend (pose + tracker, no PPE) on WILDTRACK image subsets.

    python -m tools.wildtrack_detect --root data/external/wildtrack \
        --out data/external/wildtrack/prepared/det_yolo11m

Frames are the annotated ones only (every 5th of 10 fps → 2 fps), so the tracker runs at 2 fps;
association metrics that depend on tracking are pessimistic compared to a 25 fps camera.
ts_ms matches tools/wildtrack_prepare.py (frame_idx × 100).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.common.config import DEFAULT_CONFIG_DIR, load_yaml
from src.common.jsonl import write_frames
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PerceptionFrame
from src.perception_offline.ultralytics_runner import UltralyticsPerception
from tools.wildtrack_prepare import CAMS, FPS, cam_id

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("data/external/wildtrack"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perception", default=str(DEFAULT_CONFIG_DIR / "perception.yaml"))
    ap.add_argument("--pose-model", help="override perception.yaml pose.model")
    ap.add_argument("--imgsz", type=int, help="override perception.yaml pose.imgsz")
    args = ap.parse_args(argv)
    setup_logging()

    cfg = load_yaml(args.perception)
    cfg["ppe"]["enabled"] = False  # association only
    if args.pose_model:
        cfg["pose"]["model"] = args.pose_model
    if args.imgsz:
        cfg["pose"]["imgsz"] = args.imgsz
    args.out.mkdir(parents=True, exist_ok=True)

    for v in range(len(CAMS)):
        c = cam_id(v)
        model = UltralyticsPerception(cfg)  # fresh tracker per camera (D1)
        frames = []
        for img_path in sorted((args.root / "Image_subsets" / c).glob("*.png")):
            img = cv2.imread(str(img_path))
            frame_idx = int(img_path.stem)
            frames.append(
                PerceptionFrame(
                    cam_id=c,
                    frame_idx=frame_idx,
                    ts_ms=frame_idx * 1000 // FPS,
                    img_w=img.shape[1],
                    img_h=img.shape[0],
                    persons=model.process(img),
                )
            )
        write_frames(args.out / f"{c}.jsonl", frames)
        log.info("%s: %d frames, %d persons", c, len(frames), sum(len(f.persons) for f in frames))


if __name__ == "__main__":
    main()
