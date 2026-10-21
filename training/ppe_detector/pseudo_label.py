"""Fill classes a dataset does not annotate with confident predictions of a trained model.

CHV labels person/vest/helmet only; its crops still show heads, shoes and feet. Left unlabelled
they would be learned as background. Only train/val are filled — the test split keeps its
original labels and is scored on the annotated classes only.

    python -m training.ppe_detector.pseudo_label --dataset chv \
        --weights runs/ppe/sh17_crop_y11s/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from src.common.config import DEFAULT_CONFIG_DIR, load_yaml
from src.common.logging import get_logger, setup_logging

log = get_logger(__name__)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _yolo_lines_to_xyxy(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = [list(map(float, ln.split()[:5])) for ln in lines if ln.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    arr = np.array(rows)
    c, cx, cy, w, h = arr.T
    return c.astype(int), np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "training_ppe.yaml"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)
    setup_logging()

    from ultralytics import YOLO

    cfg = load_yaml(args.config)
    ds = cfg["datasets"][args.dataset]
    classes: list[str] = cfg["classes"]
    fill = [classes.index(c) for c in ds["missing_classes"]]
    pl = cfg["pseudo_label"]
    root = Path(ds["out_crops"])
    model = YOLO(args.weights)
    report: dict = {"weights": args.weights, "conf": pl["conf"], "classes": ds["missing_classes"]}

    for split in args.splits:
        lbl_dir = root / "labels" / split
        gt_dir = root / "labels_gt" / split  # untouched copy of the original labels
        if not gt_dir.exists():
            shutil.copytree(lbl_dir, gt_dir)
        imgs = sorted((root / "images" / split).glob("*.jpg"))
        added: Counter = Counter()
        for i in range(0, len(imgs), args.batch):
            chunk = imgs[i : i + args.batch]
            results = model.predict(
                [str(p) for p in chunk],
                imgsz=args.imgsz,
                conf=pl["conf"],
                classes=fill,
                verbose=False,
            )
            for path, r in zip(chunk, results, strict=True):
                gt_lines = (gt_dir / f"{path.stem}.txt").read_text(encoding="utf-8").splitlines()
                gt_cls, gt_box = _yolo_lines_to_xyxy(gt_lines)
                b = r.boxes
                pc = b.cls.int().cpu().numpy()
                pbox = b.xyxyn.cpu().numpy()
                new = []
                for c, box in zip(pc, pbox, strict=True):
                    same = gt_box[gt_cls == c]
                    if len(same) and iou(box[None], same).max() >= pl["iou_dedup"]:
                        continue
                    x1, y1, x2, y2 = box
                    new.append(
                        f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
                    )
                    added[classes[c]] += 1
                lines = [ln for ln in gt_lines if ln.strip()] + new
                (lbl_dir / f"{path.stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
        report[split] = {"images": len(imgs), "added": dict(added)}
        log.info("%s: %s", split, report[split])
    (root / "pseudo_label.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for cache in root.glob("labels/*.cache"):
        cache.unlink()


if __name__ == "__main__":
    main()
