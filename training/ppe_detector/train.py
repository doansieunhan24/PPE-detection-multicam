"""Train the PPE detector (Ultralytics) on a prepared YOLO dataset.

    python -m training.ppe_detector.train                       # SH17 person crops
    python -m training.ppe_detector.train --data data/processed/sh17_full_frame/data.yaml \
        --name sh17_full_y11s --imgsz 640 --batch 16             # full-frame baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.config import DEFAULT_CONFIG_DIR, load_yaml
from src.common.logging import get_logger, setup_logging

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "training_ppe.yaml"))
    ap.add_argument("--data", help="data.yaml (default: <crops.out>/data.yaml)")
    ap.add_argument("--model")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--name")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()

    from ultralytics import YOLO

    cfg = load_yaml(args.config)
    t = cfg["train"]
    data = args.data or str(Path(cfg["crops"]["out"]) / "data.yaml")
    name = args.name or t["name"]
    project = Path(t["project"]).resolve()

    if args.resume:
        model = YOLO(str(project / name / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(args.model or t["model"])
        model.train(
            data=data,
            imgsz=args.imgsz or t["imgsz"],
            epochs=args.epochs or t["epochs"],
            batch=args.batch or t["batch"],
            patience=t["patience"],
            workers=t["workers"],
            project=str(project),
            name=name,
            exist_ok=True,
            seed=cfg["seed"],
            deterministic=True,
        )

    # Final evaluation on the held-out test split, per class
    best = project / name / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(
        data=data,
        split="test",
        imgsz=args.imgsz or t["imgsz"],
        project=str(project),
        name=f"{name}_test",
        exist_ok=True,
    )
    names = metrics.names
    report = {
        "weights": str(best),
        "data": data,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "per_class": {
            names[int(c)]: {
                "mAP50": float(metrics.box.ap50[i]),
                "mAP50-95": float(metrics.box.ap[i]),
            }
            for i, c in enumerate(metrics.box.ap_class_index)
        },
    }
    out = project / name / "test_metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("test metrics → %s\n%s", out, json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
