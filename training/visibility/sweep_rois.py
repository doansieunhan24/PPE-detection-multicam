"""Tune part-ROI / assignment parameters offline on val (§7.3, CLAUDE.md D8).

Re-uses the perception output stored next to the evidence tables (*.frames.jsonl), so no model
runs. Objective per part: Youden J = recall | worn − FPR | not worn of "a PPE box is assigned
to this part", pooled over the given val sets.

    python -m training.visibility.sweep_rois \
        --frames sh17=data/processed/visibility/sh17_val.frames.jsonl \
                 chv=data/processed/visibility/chv_val.frames.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.config import DEFAULT_CONFIG_DIR, load_config, load_yaml
from src.common.jsonl import read_frames
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PartName
from src.parts.part_rois import RoiParams, part_roi
from src.parts.ppe_assign import containment, exclusive_owners
from training.ppe_detector.crops import parse_yolo_labels
from training.ppe_detector.pseudo_label import iou
from training.visibility.build_v_dataset import PARTS, gt_wears

log = get_logger(__name__)

GRID = {
    PartName.HEAD: {
        "head_up_ratio": [0.6, 0.8, 1.0, 1.3, 1.6],
        "head_expand": [1.5, 1.8, 2.2, 2.6, 3.0],
    },
    PartName.TORSO: {"torso_expand": [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]},
    PartName.FEET: {
        "feet_half_w_ratio": [0.08, 0.12, 0.16, 0.2],
        "feet_down_ratio": [0.05, 0.1, 0.15],
    },
}
CONTAINMENT = [0.1, 0.2, 0.3, 0.4, 0.5]


def load_matched(frames_path: Path, ds: dict) -> list[tuple]:
    """(PersonDet, gt_person_box, {item: [gt boxes]}) for persons matched to GT (IoU >= 0.5)."""
    names = {int(k): v for k, v in ds["names"].items()}
    by_name = {v: k for k, v in names.items()}
    item_src = {
        item: {by_name[s] for s, d in ds["map"].items() if d == item}
        for item, _ in PARTS.values()
        if item not in ds.get("missing_classes", [])
    }
    gt_dir = Path(ds["root"]) / ds["labels"]
    out = []
    for fr in read_frames(frames_path):
        stem = fr.cam_id.split("/", 1)[1]
        lines = (gt_dir / f"{stem}.txt").read_text(encoding="utf-8").splitlines()
        gt = parse_yolo_labels(lines, fr.img_w, fr.img_h)
        gpers = [g.box for g in gt if g.cls == by_name[ds["person_class"]]]
        if not fr.persons or not gpers:
            continue
        m = iou(np.array([p.bbox for p in fr.persons]), np.array(gpers))
        items = {it: [g.box for g in gt if g.cls in src] for it, src in item_src.items()}
        for r, c in zip(*linear_sum_assignment(-m), strict=True):
            if m[r, c] >= 0.5 and fr.persons[r].ppe_checked:
                out.append((fr.persons[r], gpers[c], items, fr.persons, r))
    return out


def score(
    samples: list[tuple], part: PartName, prm: RoiParams, cmin: float, exclusive: bool = False
) -> dict:
    item, band = PARTS[part]
    c: Counter = Counter()
    owners_cache: dict[int, list[set[int]]] = {}
    for p, gp, items, frame_persons, idx in samples:
        if item not in items:
            continue
        y = gt_wears(gp, items[item], band)
        roi = part_roi(part, p.bbox, p.kpts, prm)
        ok = range(len(p.ppe_dets))
        if exclusive:
            key = id(frame_persons)
            if key not in owners_cache:
                rois = [part_roi(part, q.bbox, q.kpts, prm) for q in frame_persons]
                owners_cache[key] = exclusive_owners(frame_persons, rois, item, cmin)
            ok = owners_cache[key][idx]
        det = any(
            d.cls == item and containment(d.bbox, roi.box) >= cmin
            for j, d in enumerate(p.ppe_dets)
            if j in ok
        )
        c[(y, det)] += 1
    worn, notw = c[(True, True)] + c[(True, False)], c[(False, True)] + c[(False, False)]
    rec = c[(True, True)] / worn if worn else 0.0
    fpr = c[(False, True)] / notw if notw else 0.0
    return {"recall": rec, "fpr": fpr, "J": rec - fpr, "n_worn": worn, "n_not_worn": notw}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--frames", nargs="+", required=True, help="dataset=frames.jsonl")
    ap.add_argument("--training-config", default=str(DEFAULT_CONFIG_DIR / "training_ppe.yaml"))
    ap.add_argument("--out", default="eval/reports/roi_sweep")
    args = ap.parse_args(argv)
    setup_logging()

    tcfg = load_yaml(args.training_config)
    fusion = load_config("fusion")
    base = RoiParams.from_config(fusion)
    base_cmin = fusion["part_assign"]["containment_min"]
    samples = []
    for spec in args.frames:
        name, path = spec.split("=", 1)
        samples += load_matched(Path(path), tcfg["datasets"][name])
    log.info("%d matched persons", len(samples))

    report: dict = {}
    lines = ["| part | params | containment | recall | FPR | J |", "|---|---|---|---|---|---|"]
    for part, grid in GRID.items():
        keys = list(grid)
        rows = []
        for vals in itertools.product(*grid.values()):
            prm = replace(base, **dict(zip(keys, vals, strict=True)))
            for cmin in CONTAINMENT:
                rows.append(
                    (dict(zip(keys, vals, strict=True)), cmin, score(samples, part, prm, cmin))
                )
        current = score(samples, part, base, base_cmin)
        best = max(rows, key=lambda r: r[2]["J"])
        report[part.value] = {
            "current": {
                "params": {k: getattr(base, k) for k in keys},
                "containment": base_cmin,
                **current,
            },
            "best": {"params": best[0], "containment": best[1], **best[2]},
            "all": [{"params": p, "containment": c, **s} for p, c, s in rows],
        }
        for tag, (p, cm, s) in (
            ("current", (report[part.value]["current"]["params"], base_cmin, current)),
            ("best", best),
        ):
            lines.append(
                f"| {part.value} ({tag}) | {p} | {cm} | {s['recall']:.3f} | "
                f"{s['fpr']:.3f} | {s['J']:.3f} |"
            )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))


if __name__ == "__main__":
    main()
