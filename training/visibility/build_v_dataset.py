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
import dataclasses
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.config import DEFAULT_CONFIG_DIR, load_config, load_yaml
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PartName, PerceptionFrame
from src.fusion.visibility import VisibilityParams, visibility
from src.parts.part_rois import RoiParams, part_roi
from src.parts.ppe_assign import AssignParams, containment, observe_part
from training.ppe_detector.crops import parse_yolo_labels
from training.ppe_detector.pseudo_label import iou

log = get_logger(__name__)

# part -> (our item class, vertical band of the GT person box (top, bottom)); the detector
# class equals the item class; GT source classes come from datasets.<name>.map
PARTS = {
    PartName.HEAD: ("helmet", (0.0, 0.35)),
    PartName.TORSO: ("vest", (0.10, 0.80)),
    PartName.FEET: ("shoes", (0.70, 1.05)),
}
FIELDS = [
    "image",
    "person",
    "part",
    "gt",
    "p",
    "has_det",
    "any_det",
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
        # 0.5: helmets often stick out above the annotated person box
        if containment(b, person) >= 0.5 and y1 + band[0] * h <= cy <= y1 + band[1] * h:
            return True
    return False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="sh17")
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
    ds = tcfg["datasets"][args.dataset]
    names = {int(k): v for k, v in ds["names"].items()}
    by_name = {v: k for k, v in names.items()}
    # our item -> source GT class ids (e.g. CHV: helmet <- 4 helmet colours); items the
    # dataset does not annotate are skipped (CHV has no shoes)
    item_src = {
        item: {by_name[src] for src, dst in ds["map"].items() if dst == item}
        for item, _ in PARTS.values()
        if item not in ds.get("missing_classes", [])
    }
    pcfg = load_config("perception")
    pcfg["ppe"] = {**pcfg["ppe"], "enabled": True, "model": args.ppe_weights, "mode": args.ppe_mode}
    if args.ppe_imgsz:
        pcfg["ppe"]["imgsz"] = args.ppe_imgsz
    fusion = load_config("fusion")
    roi_prm = RoiParams.from_config(fusion)
    # the table must hold the RAW detector confidence: it is what calibrate.py fits on
    asg_prm = dataclasses.replace(AssignParams.from_config(fusion), calibration=None)
    vis_prm = VisibilityParams.from_config(fusion)
    pos_thr = fusion["baselines"]["pos_thr"]
    perception = UltralyticsPerception(pcfg)

    root = Path(ds["root"])
    gt_dir = root / ds["labels"]
    full_dir = Path(ds["out_full"]) / "images" / args.split
    if full_dir.exists():  # prepared (downscaled) full frames
        images = sorted(full_dir.glob("*.jpg"))
    else:  # original images listed in the dataset's split file
        split_file = ds["splits"][args.split]
        stems = [
            Path(ln.strip()).stem
            for ln in (root / split_file).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        img_dir = root / ds["images"]
        images = [q for st in stems for q in img_dir.glob(f"{st}.*")]
    images = images[: args.limit] if args.limit else images
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    table = out / f"{args.name or args.dataset + '_' + args.split}.csv"
    # perception output is kept so ROI / assignment parameters can be swept offline (D4)
    frames_path = table.with_suffix(".frames.jsonl")
    with (
        open(table, "w", encoding="utf-8", newline="") as f,
        open(frames_path, "w", encoding="utf-8") as ff,
    ):
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
            ff.write(
                PerceptionFrame(
                    cam_id=f"{args.dataset}/{path.stem}",
                    frame_idx=k,
                    ts_ms=0,
                    img_w=iw,
                    img_h=ih,
                    persons=persons,
                ).model_dump_json()
                + "\n"
            )
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
                for part, (item, band) in PARTS.items():
                    if item not in item_src:
                        continue
                    det_cls = item
                    items = [g.box for g in gt if g.cls in item_src[item]]
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
                            "image": f"{args.dataset}/{path.stem}",
                            "person": r,
                            "part": part.value,
                            "gt": int(y),
                            "p": round(ob.p, 4),
                            "has_det": int(ob.matched is not None),
                            "any_det": int(any(d.cls == det_cls for d in p.ppe_dets)),
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
