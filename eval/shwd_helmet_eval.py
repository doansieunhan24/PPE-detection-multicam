"""Helmet presence per person on SHWD test (third domain: crowded Chinese construction sites).

SHWD boxes heads only (`hat` = with helmet, `person` = without); its `hat` box covers the whole
head, unlike SH17/CHV `helmet` boxes, so SHWD is used for *evaluation of presence*, not for
training. Each detected person is paired with the GT head box lying in the top 40 % of the
person box (largest overlap); worn = that head is a `hat`.

    python -m eval.shwd_helmet_eval --split test
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.analytics_runner import git_commit
from src.common.config import DEFAULT_CONFIG_DIR, load_config, load_yaml
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PartName
from src.fusion.visibility import VisibilityParams, visibility
from src.parts.part_rois import RoiParams, part_roi
from src.parts.ppe_assign import AssignParams, containment, exclusive_owners, observe_part
from training.ppe_detector.prepare_shwd_person_crops import parse_voc
from training.visibility.calibrate import BINS, auc

log = get_logger(__name__)


def pair_head(person: tuple, heads: list[tuple[str, tuple]]) -> str | None:
    x1, y1, x2, y2 = person
    top = (x1, y1 - 0.1 * (y2 - y1), x2, y1 + 0.4 * (y2 - y1))
    best, best_c = None, 0.5
    for name, b in heads:
        c = containment(b, top)
        if c >= best_c:
            best, best_c = name, c
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--exclusive", choices=["config", "on", "off"], default="config")
    ap.add_argument("--out", default="eval/reports/shwd")
    args = ap.parse_args(argv)
    setup_logging()

    from src.perception_offline.ultralytics_runner import UltralyticsPerception

    ds = load_yaml(DEFAULT_CONFIG_DIR / "training_ppe.yaml")["datasets"]["shwd"]
    root = Path(ds["root"])
    fusion = load_config("fusion")
    roi_prm, asg_prm = RoiParams.from_config(fusion), AssignParams.from_config(fusion)
    if args.exclusive != "config":
        asg_prm = dataclasses.replace(asg_prm, exclusive=args.exclusive == "on")
    vis_prm = VisibilityParams.from_config(fusion)
    perception = UltralyticsPerception(load_config("perception"))
    ids = (root / ds["splits"][args.split]).read_text(encoding="utf-8").split()
    ids = ids[: args.limit] if args.limit else ids

    rows, st = [], Counter()
    for i in ids:
        objs = parse_voc(root / "Annotations" / f"{i}.xml")
        if not any(n == "hat" for n, _ in objs):
            continue
        img = cv2.imread(str(root / "JPEGImages" / f"{i}.jpg"))
        ih, iw = img.shape[:2]
        st["images"] += 1
        persons = perception.process(img, track=False)
        rois = [part_roi(PartName.HEAD, q.bbox, q.kpts, roi_prm) for q in persons]
        owners = (
            exclusive_owners(persons, rois, "helmet", asg_prm.containment_min)
            if asg_prm.exclusive
            else [None] * len(persons)
        )
        for r, p in enumerate(persons):
            head = pair_head(p.bbox, objs)
            if head is None:
                st["person_without_gt_head"] += 1
                continue
            roi = rois[r]
            ob = observe_part(p, roi, "helmet", asg_prm, owners[r])
            if ob.p is None:
                st["ppe_not_checked"] += 1
                continue
            others = [q.bbox for k, q in enumerate(persons) if k != r]
            v = visibility(roi, p.bbox, others, iw, ih, vis_prm).v
            rows.append((head == "hat", ob.matched is not None, v, p.conf))
            st["persons"] += 1
    a = np.array(rows, dtype=float)
    worn, det, v, conf = a[:, 0] == 1, a[:, 1], a[:, 2], a[:, 3]
    res = {
        "commit": git_commit(),
        "split": args.split,
        "stats": dict(st),
        "n_worn": int(worn.sum()),
        "n_not_worn": int((~worn).sum()),
        "recall": float(det[worn].mean()),
        "fpr": float(det[~worn].mean()),
        "auc_detected_given_worn": {
            "v": auc(v[worn], det[worn]),
            "person_conf": auc(conf[worn], det[worn]),
        },
        "bins_v": [],
    }
    for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
        m = (v >= lo) & (v < hi)
        res["bins_v"].append(
            {
                "bin": f"[{lo:.1f},{min(hi, 1):.1f})",
                "n_worn": int((m & worn).sum()),
                "recall": float(det[m & worn].mean()) if (m & worn).any() else None,
                "n_not_worn": int((m & ~worn).sum()),
                "fpr": float(det[m & ~worn].mean()) if (m & ~worn).any() else None,
            }
        )

    def f(x):
        return "–" if x is None else f"{x:.3f}"

    md = [
        f"<!-- generated by eval/shwd_helmet_eval.py @ {res['commit']} -->",
        f"SHWD {args.split} (images with ≥1 helmet): {res['n_worn']} persons with helmet, "
        f"{res['n_not_worn']} without; recall {f(res['recall'])}, FPR {f(res['fpr'])}; "
        f"AUC(v, detected|worn) {f(res['auc_detected_given_worn']['v'])} vs person conf "
        f"{f(res['auc_detected_given_worn']['person_conf'])}",
        "",
        "| v bin | worn n | recall | not-worn n | FPR |",
        "|---|---|---|---|---|",
    ]
    md += [
        f"| {b['bin']} | {b['n_worn']} | {f(b['recall'])} | {b['n_not_worn']} | {f(b['fpr'])} |"
        for b in res["bins_v"]
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_excl" if asg_prm.exclusive else args.split
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / f"{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(md))


if __name__ == "__main__":
    main()
