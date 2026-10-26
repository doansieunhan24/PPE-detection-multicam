"""Compare two perception backends on the same videos (CLAUDE.md §11 P1 "JSONL từ 2 backend so
khớp được"; D5: DeepStream and the offline Ultralytics backend share one schema).

    python -m eval.compare_backends --ref data/rehearsal/wt3/jsonl --test data/ds/wt3/jsonl \
        --out eval/reports/backends/wt3_offline_vs_ds.json

Frames are paired by (cam_id, frame_idx); persons by Hungarian IoU ≥ --iou. Reported per camera
and overall:
  frames      frame counts, frames whose ts_ms differ
  persons     counts, matched, recall (= matched / ref), precision (= matched / test), mean IoU
  kpts        joints visible (conf ≥ --kpt-conf) in both: median / p90 distance as a fraction of
              the ref person's bbox height
  ppe         per class, over matched persons: present (any det of that class) in both / ref only
              / test only, and the agreement rate
  tracks      distinct track ids and mean track length (frames)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval.eval_association import iou_matrix
from src.analytics_runner import git_commit
from src.common.jsonl import read_frames
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PerceptionFrame, PersonDet

log = get_logger(__name__)


def match_persons(
    ref: list[PersonDet], test: list[PersonDet], iou_min: float
) -> list[tuple[int, int, float]]:
    """(ref index, test index, IoU) pairs with IoU ≥ iou_min."""
    if not ref or not test:
        return []
    iou = iou_matrix(
        np.array([p.bbox for p in ref], dtype=float), np.array([p.bbox for p in test], dtype=float)
    )
    rows, cols = linear_sum_assignment(-iou)
    return [
        (r, c, float(iou[r, c])) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= iou_min
    ]


def kpt_errors(a: PersonDet, b: PersonDet, conf_min: float) -> list[float]:
    """Distances of joints visible in both, normalised by a's bbox height."""
    h = max(a.bbox[3] - a.bbox[1], 1e-6)
    return [
        float(np.hypot(ka[0] - kb[0], ka[1] - kb[1]) / h)
        for ka, kb in zip(a.kpts, b.kpts, strict=False)
        if ka[2] >= conf_min and kb[2] >= conf_min
    ]


class CamStats:
    def __init__(self) -> None:
        self.c: Counter = Counter()
        self.ious: list[float] = []
        self.kerr: list[float] = []
        self.ppe: dict[str, Counter] = defaultdict(Counter)
        self.tracks: dict[str, Counter] = {"ref": Counter(), "test": Counter()}

    def add_frame(
        self, ref: PerceptionFrame | None, test: PerceptionFrame | None, iou_min: float, kc: float
    ) -> None:
        self.c["frames_ref"] += ref is not None
        self.c["frames_test"] += test is not None
        if ref is None or test is None:
            return
        self.c["frames_both"] += 1
        self.c["ts_mismatch"] += ref.ts_ms != test.ts_ms
        self.c["persons_ref"] += len(ref.persons)
        self.c["persons_test"] += len(test.persons)
        for side, fr in (("ref", ref), ("test", test)):
            self.tracks[side].update(p.track_id for p in fr.persons)
        for r, t, iou in match_persons(ref.persons, test.persons, iou_min):
            a, b = ref.persons[r], test.persons[t]
            self.c["matched"] += 1
            self.ious.append(iou)
            self.kerr.extend(kpt_errors(a, b, kc))
            if not (a.ppe_checked and b.ppe_checked):
                continue
            ca, cb = {d.cls for d in a.ppe_dets}, {d.cls for d in b.ppe_dets}
            for cls in ca | cb:
                key = (
                    "both"
                    if cls in ca and cls in cb
                    else ("ref_only" if cls in ca else "test_only")
                )
                self.ppe[cls][key] += 1
            self.c["ppe_compared"] += 1

    def report(self) -> dict:
        c = self.c
        k = np.array(self.kerr) if self.kerr else np.zeros(1)
        ppe = {}
        for cls, n in sorted(self.ppe.items()):
            neither = c["ppe_compared"] - n["both"] - n["ref_only"] - n["test_only"]
            ppe[cls] = {
                "both": n["both"],
                "ref_only": n["ref_only"],
                "test_only": n["test_only"],
                "neither": neither,
                "agreement": (n["both"] + neither) / max(c["ppe_compared"], 1),
            }
        return {
            "frames": {
                k_: c[k_] for k_ in ("frames_ref", "frames_test", "frames_both", "ts_mismatch")
            },
            "persons": {
                "ref": c["persons_ref"],
                "test": c["persons_test"],
                "matched": c["matched"],
                "recall": c["matched"] / max(c["persons_ref"], 1),
                "precision": c["matched"] / max(c["persons_test"], 1),
                "mean_iou": float(np.mean(self.ious)) if self.ious else 0.0,
            },
            "kpts": {
                "n_joints": len(self.kerr),
                "median_err_rel_h": float(np.median(k)),
                "p90_err_rel_h": float(np.percentile(k, 90)),
            },
            "ppe": {"persons_compared": c["ppe_compared"], "per_class": ppe},
            "tracks": {
                side: {
                    "n": len(t),
                    "mean_len": float(np.mean(list(t.values()))) if t else 0.0,
                }
                for side, t in self.tracks.items()
            },
        }


def load_dir(d: Path) -> dict[str, dict[int, PerceptionFrame]]:
    out: dict[str, dict[int, PerceptionFrame]] = {}
    for f in sorted(d.glob("*.jsonl")):
        for fr in read_frames(f):
            out.setdefault(fr.cam_id, {})[fr.frame_idx] = fr
    return out


def compare(ref_dir: Path, test_dir: Path, iou_min: float = 0.5, kpt_conf: float = 0.5) -> dict:
    ref, test = load_dir(ref_dir), load_dir(test_dir)
    total = CamStats()
    per_cam = {}
    for cam in sorted(set(ref) | set(test)):
        st = CamStats()
        r, t = ref.get(cam, {}), test.get(cam, {})
        for idx in sorted(set(r) | set(t)):
            for s in (st, total):
                s.add_frame(r.get(idx), t.get(idx), iou_min, kpt_conf)
        per_cam[cam] = st.report()
    return {"overall": total.report(), "per_cam": per_cam}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ref", required=True, help="reference JSONL dir (offline backend)")
    ap.add_argument("--test", required=True, help="JSONL dir to compare (DeepStream)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--kpt-conf", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    setup_logging()
    res = compare(Path(args.ref), Path(args.test), args.iou, args.kpt_conf)
    res["meta"] = {
        "ref": args.ref,
        "test": args.test,
        "iou": args.iou,
        "kpt_conf": args.kpt_conf,
        "commit": git_commit(),
    }
    o = res["overall"]
    log.info("frames %s", o["frames"])
    log.info(
        "persons %s",
        {k: round(v, 3) if isinstance(v, float) else v for k, v in o["persons"].items()},
    )
    log.info(
        "kpts %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in o["kpts"].items()}
    )
    for cls, v in o["ppe"]["per_class"].items():
        log.info(
            "ppe %-7s %s",
            cls,
            {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()},
        )
    log.info("tracks %s", o["tracks"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        log.info("-> %s", args.out)


if __name__ == "__main__":
    main()
