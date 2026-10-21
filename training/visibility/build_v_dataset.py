"""Per-(person, part) evidence table on SH17 still images, for v / p_neg calibration (§7.3–7.4).

For each SH17 image of a split: pose + PPE-on-crops exactly as the offline runner, match
predicted persons to GT persons (IoU), then for head/torso/feet record the visibility terms,
the detector evidence p, and the ground truth "this person wears the item" (GT item box
inside the GT person box, in the part's vertical band).

    python -m training.visibility.build_v_dataset --split val \
        --ppe-weights runs/ppe/sh17_crop_y11s/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.config import DEFAULT_CONFIG_DIR, load_config, load_yaml
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PartName
from src.fusion.visibility import VisibilityParams, visibility
from src.parts.part_rois import RoiParams, part_roi
from src.parts.ppe_assign import AssignParams, containment, observe_part
from training.ppe_detector.crops import parse_yolo_labels
from training.ppe_detector.pseudo_label import iou

log = get_logger(__name__)

# part -> (SH17 GT class name, detector class, vertical band of the GT person box (top, bottom))
PARTS = {
    PartName.HEAD: ("helmet", "helmet", (0.0, 0.35)),
    PartName.TORSO: ("safety-vest", "vest", (0.10, 0.80)),
    PartName.FEET: ("shoes", "shoes", (0.70, 1.05)),
}
FIELDS = [
    "image",
    "person",
    "part",
    "gt",
    "p",
    "has_det",
    "pred_pos",
    "correct",
    "s_kp",
    "s_size",
    "r_occ",
    "fallback",
    "h_roi_px",
    "v",
    "person_conf",
    "person_h_px",
]


def gt_wears(person: tuple, items: list[tuple], band: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = person
    h = y2 - y1
    for b in items:
        cy = (b[1] + b[3]) / 2
        if containment(b, person) >= 0.8 and y1 + band[0] * h <= cy <= y1 + band[1] * h:
            return True
    return False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="val")
    ap.add_argument("--ppe-weights", required=True)
    ap.add_argument("--training-config", default=str(DEFAULT_CONFIG_DIR / "training_ppe.yaml"))
    ap.add_argument("--out", default="data/processed/visibility")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ppe-mode", choices=["crop", "full"], default="crop")
    ap.add_argument("--ppe-imgsz", type=int, help="override ppe.imgsz (e.g. 640 for full mode)")
    ap.add_argument("--name", help="output table name (default: <split>)")
    args = ap.parse_args(argv)
    setup_logging()

    from src.perception_offline.ultralytics_runner import UltralyticsPerception

    tcfg = load_yaml(args.training_config)
    sh = tcfg["datasets"]["sh17"]
    names = {int(k): v for k, v in sh["names"].items()}
    by_name = {v: k for k, v in names.items()}
    pcfg = load_config("perception")
    pcfg["ppe"] = {**pcfg["ppe"], "enabled": True, "model": args.ppe_weights, "mode": args.ppe_mode}
    if args.ppe_imgsz:
        pcfg["ppe"]["imgsz"] = args.ppe_imgsz
    fusion = load_config("fusion")
    roi_prm, asg_prm = RoiParams.from_config(fusion), AssignParams.from_config(fusion)
    vis_prm = VisibilityParams.from_config(fusion)
    pos_thr = fusion["baselines"]["pos_thr"]
    perception = UltralyticsPerception(pcfg)

    img_dir = Path(sh["out_full"]) / "images" / args.split
    gt_dir = Path(sh["root"]) / sh["labels"]
    images = (
        sorted(img_dir.glob("*.jpg"))[: args.limit] if args.limit else sorted(img_dir.glob("*.jpg"))
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    table = out / f"{args.name or args.split}.csv"
    with open(table, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for k, path in enumerate(images):
            img = cv2.imread(str(path))
            ih, iw = img.shape[:2]
            gt = parse_yolo_labels(
                (gt_dir / f"{path.stem}.txt").read_text(encoding="utf-8").splitlines(), iw, ih
            )
            gt_persons = [g.box for g in gt if g.cls == by_name["person"]]
            persons = perception.process(img, track=False)
            stats["gt_persons"] += len(gt_persons)
            stats["pred_persons"] += len(persons)
            if not persons or not gt_persons:
                continue
            m = iou(np.array([p.bbox for p in persons]), np.array(gt_persons))
            rows, cols = linear_sum_assignment(-m)
            for r, c in zip(rows, cols, strict=True):
                if m[r, c] < 0.5:
                    continue
                stats["matched"] += 1
                p, gp = persons[r], gt_persons[c]
                others = [q.bbox for i, q in enumerate(persons) if i != r]
                for part, (gt_name, det_cls, band) in PARTS.items():
                    items = [g.box for g in gt if g.cls == by_name[gt_name]]
                    y = gt_wears(gp, items, band)
                    roi = part_roi(part, p.bbox, p.kpts, roi_prm)
                    ob = observe_part(p, roi, det_cls, asg_prm)
                    if ob.p is None:
                        stats["ppe_not_checked"] += 1
                        continue
                    t = visibility(roi, p.bbox, others, iw, ih, vis_prm)
                    pred = ob.p >= pos_thr
                    w.writerow(
                        {
                            "image": path.stem,
                            "person": r,
                            "part": part.value,
                            "gt": int(y),
                            "p": round(ob.p, 4),
                            "has_det": int(ob.matched is not None),
                            "pred_pos": int(pred),
                            "correct": int(pred == y),
                            "s_kp": round(t.s_kp, 4),
                            "s_size": round(t.s_size, 4),
                            "r_occ": round(t.r_occ, 4),
                            "fallback": int(roi.fallback),
                            "h_roi_px": round(roi.box[3] - roi.box[1], 1),
                            "v": round(t.v, 4),
                            "person_conf": round(p.conf, 4),
                            "person_h_px": round(p.bbox[3] - p.bbox[1], 1),
                        }
                    )
            if (k + 1) % 200 == 0:
                log.info("%d/%d images %s", k + 1, len(images), dict(stats))
    log.info("done %s → %s", dict(stats), table)


if __name__ == "__main__":
    main()
