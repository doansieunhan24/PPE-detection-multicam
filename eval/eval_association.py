"""Evaluate cross-camera association (CLAUDE.md §7.2, §10.1 "Association") from JSONL.

    python -m eval.eval_association --jsonl data/external/wildtrack/prepared/gt_oracle \
        --calib-dir data/external/wildtrack/prepared/calib \
        --gt data/external/wildtrack/prepared/gt.jsonl --d-max 0.3 0.5 0.8 1.0 \
        --out eval/reports/association/wildtrack_oracle.json

GT (gt.jsonl, one line per instant):
    {"ts_ms": ..., "persons": [{"gt_id": 3, "xy_m": [x, y] | null, "boxes": {"C1": [x1,y1,x2,y2]}}]}

Detections are mapped to GT per (instant, camera) by Hungarian IoU ≥ --iou. Metrics:
  floor_err_m      |floor(foot point) − GT floor position| for matched detections
  pairwise P/R/F1  over every cross-camera pair of detections at the same instant:
                   GT-positive = both matched to the same gt_id, predicted = same global_id
  count            people per instant: distinct global_ids vs GT persons with ≥ 1 box,
                   against two no-association baselines (sum over cams, max over cams)
  IDF1             observation-level, optimal 1:1 global_id ↔ gt_id; "assoc" variant uses only
                   matched detections (isolates association from detector misses / FPs)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.analytics_runner import git_commit
from src.association.associator import Associator
from src.association.ground_plane import sync_groups
from src.common.config import DEFAULT_CONFIG_DIR, load_config
from src.common.jsonl import read_frames
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PerceptionFrame
from src.geometry.homography import CameraCalibration, load_calibration, point_in_polygon

log = get_logger(__name__)

ObsKey = tuple[int, str, int]  # (ts_ms, cam_id, track_id)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_to_gt(
    fr: PerceptionFrame, gt_boxes: dict[int, list[float]], iou_min: float
) -> dict[int, int]:
    """track_id → gt_id for detections of one camera frame."""
    if not fr.persons or not gt_boxes:
        return {}
    gids = list(gt_boxes)
    iou = iou_matrix(
        np.array([p.bbox for p in fr.persons], dtype=float),
        np.array([gt_boxes[g] for g in gids], dtype=float),
    )
    rows, cols = linear_sum_assignment(-iou)
    return {
        fr.persons[r].track_id: gids[c]
        for r, c in zip(rows, cols, strict=True)
        if iou[r, c] >= iou_min
    }


def idf1(pred: dict[ObsKey, int], gt: dict[ObsKey, int], n_gt: int) -> dict[str, float]:
    """pred: obs → global_id (all detections); gt: obs → gt_id (matched detections only)."""
    co = Counter((pred[k], g) for k, g in gt.items())
    p_ids = sorted({p for p, _ in co})
    g_ids = sorted({g for _, g in co})
    if not co:
        return {"idf1": 0.0, "idp": 0.0, "idr": 0.0}
    m = np.zeros((len(p_ids), len(g_ids)))
    pi = {p: i for i, p in enumerate(p_ids)}
    gi = {g: i for i, g in enumerate(g_ids)}
    for (p, g), n in co.items():
        m[pi[p], gi[g]] = n
    r, c = linear_sum_assignment(-m)
    idtp = float(m[r, c].sum())
    n_pred = len(pred)
    return {
        "idf1": 2 * idtp / max(n_pred + n_gt, 1),
        "idp": idtp / max(n_pred, 1),
        "idr": idtp / max(n_gt, 1),
    }


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / max(p + r, 1e-9),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def in_roi(xy: tuple[float, float], roi: list[tuple[float, float]] | None) -> bool:
    return roi is None or (not np.isnan(xy[0]) and point_in_polygon(xy, roi))


def evaluate(
    streams: dict[str, list[PerceptionFrame]],
    calibs: dict[str, CameraCalibration],
    gt: dict[int, list[dict]],
    d_max_m: float,
    vote_window: int,
    iou_min: float,
    sync_tol_ms: int,
    roi: list[tuple[float, float]] | None,
    fov_margin_m: float = 0.0,
    foot_point: str = "ankles",
) -> dict:
    assoc = Associator(calibs, d_max_m, vote_window, fov_margin_m, foot_point)
    floor_err: list[float] = []
    pred: dict[ObsKey, int] = {}
    gt_of: dict[ObsKey, int] = {}
    n_gt_obs = 0
    tp = fp = fn = 0
    cnt_err, cnt_sum_err, cnt_max_err, cnt_gt = [], [], [], []

    for group in sync_groups(streams, sync_tol_ms):
        ts = min(f.ts_ms for f in group.values())
        persons_gt = gt.get(ts)
        if persons_gt is None:
            continue
        gt_xy = {p["gt_id"]: p.get("xy_m") for p in persons_gt}
        # ROI: drop detections whose floor point is outside the evaluated area (a deployed system
        # does the same with its workspace polygon)
        kept: dict[str, PerceptionFrame] = {}
        for cam, fr in group.items():
            ps = [p for p in fr.persons if in_roi(assoc.floor_xy(cam, p), roi)]
            kept[cam] = fr.model_copy(update={"persons": ps})
        gid_of = assoc.step(kept, ts)

        obs: list[tuple[ObsKey, int, int | None]] = []  # (key, gid, gt_id)
        for cam, fr in kept.items():
            boxes = {p["gt_id"]: p["boxes"][cam] for p in persons_gt if cam in p["boxes"]}
            n_gt_obs += len(boxes)
            m = match_to_gt(fr, boxes, iou_min)
            for p in fr.persons:
                key = (ts, cam, p.track_id)
                g = m.get(p.track_id)
                gid = gid_of[(cam, p.track_id)]
                pred[key] = gid
                obs.append((key, gid, g))
                if g is not None:
                    gt_of[key] = g
                    if gt_xy.get(g) is not None:
                        xy = assoc.floor_xy(cam, p)
                        if not np.isnan(xy[0]):
                            floor_err.append(
                                float(np.hypot(xy[0] - gt_xy[g][0], xy[1] - gt_xy[g][1]))
                            )

        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                (ki, gi, ti), (kj, gj, tj) = obs[i], obs[j]
                if ki[1] == kj[1]:
                    continue
                same_gt = ti is not None and ti == tj
                same_pred = gi == gj
                tp += same_gt and same_pred
                fp += same_pred and not same_gt
                fn += same_gt and not same_pred

        # people visible in at least one of the evaluated cameras
        n_true = sum(1 for p in persons_gt if set(p["boxes"]) & set(kept))
        per_cam = [len(fr.persons) for fr in kept.values()]
        cnt_gt.append(n_true)
        cnt_err.append(len(set(gid_of.values())) - n_true)
        cnt_sum_err.append(sum(per_cam) - n_true)
        cnt_max_err.append(max(per_cam, default=0) - n_true)

    fe = np.array(floor_err)
    n_matched = len(gt_of)

    def _cnt(e: list[int]) -> dict[str, float]:
        a = np.array(e, dtype=float)
        return {"mae": float(np.abs(a).mean()), "bias": float(a.mean())}

    return {
        "d_max_m": d_max_m,
        "fov_margin_m": fov_margin_m,
        "foot_point": foot_point,
        "instants": len(cnt_gt),
        "detections": len(pred),
        "matched_detections": n_matched,
        "gt_boxes": n_gt_obs,
        "det_recall": n_matched / max(n_gt_obs, 1),
        "det_precision": n_matched / max(len(pred), 1),
        "floor_err_m": {
            "median": float(np.median(fe)) if len(fe) else None,
            "p90": float(np.percentile(fe, 90)) if len(fe) else None,
            "frac_lt_dmax": float((fe < d_max_m).mean()) if len(fe) else None,
        },
        "pairwise": prf(tp, fp, fn),
        "count": {
            "gt_mean": float(np.mean(cnt_gt)),
            "associated": _cnt(cnt_err),
            "no_assoc_sum": _cnt(cnt_sum_err),
            "no_assoc_max": _cnt(cnt_max_err),
        },
        "id": idf1(pred, gt_of, n_gt_obs),
        "id_assoc_only": idf1({k: pred[k] for k in gt_of}, gt_of, n_matched),
    }


def load_gt(path: Path) -> dict[int, list[dict]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[int(d["ts_ms"])] = d["persons"]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jsonl", type=Path, required=True, help="dir with <cam_id>.jsonl")
    ap.add_argument("--calib-dir", type=Path, required=True, help="dir with <cam_id>.yaml")
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    ap.add_argument("--d-max", type=float, nargs="*", help="sweep; default zones.yaml d_max_m")
    ap.add_argument(
        "--fov-margin", type=float, nargs="*", help="sweep; default zones.yaml fov_margin_m"
    )
    ap.add_argument(
        "--foot-point",
        nargs="*",
        choices=["ankles", "bbox_bottom"],
        help="sweep; default zones.yaml foot_point",
    )
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="evaluate only detections whose floor point is in this rectangle (m)",
    )
    ap.add_argument("--cams", nargs="*", help="subset of cameras (default: all)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    setup_logging()

    assoc_cfg = load_config("zones", args.config_dir).get("association", {})
    sync_tol = load_config("cameras", args.config_dir).get("sync_tol_ms", 50)
    streams = {
        p.stem: sorted(read_frames(p), key=lambda f: f.ts_ms)
        for p in sorted(args.jsonl.glob("*.jsonl"))
        if not args.cams or p.stem in args.cams
    }
    calibs = {c: load_calibration(args.calib_dir / f"{c}.yaml") for c in streams}
    gt = load_gt(args.gt)
    roi = None
    if args.roi:
        x0, y0, x1, y1 = args.roi
        roi = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    results = []
    sweep = [
        (d, fm, fpt)
        for d in args.d_max or [assoc_cfg.get("d_max_m", 0.8)]
        for fm in args.fov_margin or [assoc_cfg.get("fov_margin_m", 0.0)]
        for fpt in args.foot_point or [assoc_cfg.get("foot_point", "ankles")]
    ]
    for d, fm, fpt in sweep:
        r = evaluate(
            streams,
            calibs,
            gt,
            d,
            assoc_cfg.get("vote_window_frames", 15),
            args.iou,
            sync_tol,
            roi,
            fm,
            fpt,
        )
        results.append(r)
        log.info(
            "d_max %.2f fov_margin %.2f foot %s: pair P/R/F1 %.3f/%.3f/%.3f  IDF1 %.3f "
            "(assoc-only %.3f)  count MAE %.2f bias %+.2f (sum %.2f, max %.2f)  "
            "floor err med %.3f m",
            d,
            fm,
            fpt,
            r["pairwise"]["precision"],
            r["pairwise"]["recall"],
            r["pairwise"]["f1"],
            r["id"]["idf1"],
            r["id_assoc_only"]["idf1"],
            r["count"]["associated"]["mae"],
            r["count"]["associated"]["bias"],
            r["count"]["no_assoc_sum"]["mae"],
            r["count"]["no_assoc_max"]["mae"],
            r["floor_err_m"]["median"] or float("nan"),
        )
    report = {
        "jsonl": str(args.jsonl),
        "calib_dir": str(args.calib_dir),
        "gt": str(args.gt),
        "cams": sorted(streams),
        "iou": args.iou,
        "roi": args.roi,
        "vote_window_frames": assoc_cfg.get("vote_window_frames", 15),
        "commit": git_commit(),
        "results": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("→ %s", args.out)


if __name__ == "__main__":
    main()
