"""Evidence table on labelled C2 sessions, for v_learned / p calibration (§7.3–7.4).

Same columns as build_v_dataset.py (SH17/CHV still images), so training/visibility/calibrate.py
fits on it unchanged. Unlike single images, C2 lets visibility vary while the GT stays fixed:
the same person is seen from the back, occluded, far... by different cameras at once.

GT per (person, part, instant): the item is worn unless the instant lies in a PPE_<ITEM>
violation interval of that person (intervals.csv). Samples within --margin-s of an interval
edge are skipped (label timing is approximate); tracks without gt_id are skipped.

    python -m training.visibility.build_v_dataset_c2 --sessions data/jsonl/s01 \
        --gt data/annotations --out data/processed/visibility/c2_val.csv --stride 5
    python -m training.visibility.calibrate --val data/processed/visibility/c2_val.csv \
        --test c2_test=data/processed/visibility/c2_test.csv \
        --out eval/reports/c2/visibility --write-config configs/visibility_learned_c2.yaml

Fit on val sessions only. The p in the table is the RAW detector confidence (calibrate.py fits
the calibration on it).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from collections import Counter
from pathlib import Path

from eval.metrics import load_gt_csv
from eval.run_baselines import load_track_gt
from src.common.config import DEFAULT_CONFIG_DIR, load_config
from src.common.jsonl import read_frames
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PartName
from src.fusion.visibility import VisibilityParams, visibility
from src.parts.part_rois import RoiParams, part_roi
from src.parts.ppe_assign import AssignParams, observe_part
from training.visibility.build_v_dataset import FIELDS

log = get_logger(__name__)


def violation_spans(
    intervals_csv: Path, margin_ms: int
) -> dict[tuple[int, str], list[tuple[int, int]]]:
    """(gt_id, alert type) → violation intervals."""
    out: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for g in load_gt_csv(intervals_csv):
        if g.gt_id is not None and g.type.startswith("PPE_"):
            out.setdefault((g.gt_id, g.type), []).append((g.t_start_ms, g.t_end_ms))
    return out


def label_at(spans: list[tuple[int, int]], ts: int, margin_ms: int) -> int | None:
    """1 = worn, 0 = violation, None = too close to an interval edge."""
    for a, b in spans:
        if abs(ts - a) < margin_ms or abs(ts - b) < margin_ms:
            return None
        if a <= ts <= b:
            return 0
    return 1


def build(
    sessions: list[Path],
    gt_root: Path,
    out: Path,
    config_dir: Path,
    stride: int = 5,
    margin_s: float = 1.0,
) -> Counter:
    fusion = load_config("fusion", config_dir)
    roi_prm = RoiParams.from_config(fusion)
    asg_prm = dataclasses.replace(AssignParams.from_config(fusion), calibration=None)
    vis_prm = VisibilityParams.from_config(fusion)
    pos_thr = fusion["baselines"]["pos_thr"]
    parts = {
        PartName(k): (v["det_cls"], f"PPE_{v['item'].upper()}") for k, v in fusion["parts"].items()
    }
    margin_ms = int(margin_s * 1000)
    stats: Counter = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for sess in sessions:
            gt_dir = gt_root / sess.name
            track_gt = load_track_gt(gt_dir / "tracks.csv")
            viol = violation_spans(gt_dir / "intervals.csv", margin_ms)
            for path in sorted(sess.glob("*.jsonl")):
                for k, fr in enumerate(read_frames(path)):
                    if k % stride:
                        continue
                    for i, p in enumerate(fr.persons):
                        gt_id = track_gt.at(fr.cam_id, p.track_id, fr.ts_ms)
                        if gt_id is None:
                            stats["unlabelled"] += 1
                            continue
                        others = [q.bbox for j, q in enumerate(fr.persons) if j != i]
                        for part, (det_cls, typ) in parts.items():
                            y = label_at(viol.get((gt_id, typ), []), fr.ts_ms, margin_ms)
                            if y is None:
                                stats["edge_skipped"] += 1
                                continue
                            roi = part_roi(part, p.bbox, p.kpts, roi_prm)
                            ob = observe_part(p, roi, det_cls, asg_prm)
                            if ob.p is None:
                                stats["ppe_not_checked"] += 1
                                continue
                            t = visibility(roi, p.bbox, others, fr.img_w, fr.img_h, vis_prm)
                            pred = ob.p >= pos_thr
                            stats[f"{part.value}_worn" if y else f"{part.value}_viol"] += 1
                            w.writerow({
                                "image": f"{sess.name}/{fr.cam_id}/{fr.frame_idx}",
                                "person": gt_id,
                                "part": part.value,
                                "gt": y,
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
                            })  # fmt: skip
    log.info("%s → %s", dict(stats), out)
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--gt", required=True, help="root with <session>/{tracks,intervals}.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--stride", type=int, default=5, help="use every n-th frame of each camera")
    ap.add_argument("--margin-s", type=float, default=1.0)
    args = ap.parse_args(argv)
    setup_logging()
    build([Path(s) for s in args.sessions], Path(args.gt), Path(args.out), Path(args.config),
          args.stride, args.margin_s)  # fmt: skip


if __name__ == "__main__":
    main()
