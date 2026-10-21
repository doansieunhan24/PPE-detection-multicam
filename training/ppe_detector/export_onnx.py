"""Export trained PPE detector to ONNX (for DeepStream SGIE) and copy into models/.

    python -m training.ppe_detector.export_onnx --weights runs/ppe/sh17_crop_y11s/weights/best.pt

NOTE: DeepStream-Yolo has its own export script that adds the output layout its custom parser
expects — TODO(verify) against marcoslucianops/DeepStream-Yolo docs before using in SGIE.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.common.logging import get_logger, setup_logging

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--dynamic", action="store_true", help="dynamic batch (SGIE batches crops)")
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args(argv)
    setup_logging()

    from ultralytics import YOLO

    weights = Path(args.weights)
    onnx = Path(
        YOLO(str(weights)).export(
            format="onnx", imgsz=args.imgsz, dynamic=args.dynamic, simplify=True
        )
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = weights.parents[1].name
    for src, dst in ((weights, out_dir / f"{run_name}.pt"), (onnx, out_dir / f"{run_name}.onnx")):
        shutil.copy2(src, dst)
        log.info("copied %s → %s", src, dst)


if __name__ == "__main__":
    main()
