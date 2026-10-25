"""Intrusion and crowd event metrics on WILDTRACK (CLAUDE.md §7.7, §10.1 Behavior; P6).

WILDTRACK gives every person's floor position at every annotated frame (2 fps), so zone
events can be derived from ground truth for any floor polygon. Zones are picked
deterministically from GT occupancy (no tuning on predictions): forbidden = 2 × 2 m cells
with a moderate number of distinct visitors, risk = 4 × 4 m cells, crowd K = 75th percentile of
the GT count there. The full system runs on the detector JSONL (7 cams): association → floor
position per global id → BehaviorEngine (same code as live). Cooldown is 0 here (it only
de-duplicates notifications).

GT episodes use the alerter's own definition: inside continuously for ≥ T (T_intr, crowd T_s).
Matching is per zone and person-agnostic: an alert is a TP if t_raise ∈ [start − δ, end + δ]
of a still unmatched GT episode of that zone; other alerts are false alarms.

    python -m eval.eval_zones --jsonl data/external/wildtrack/prepared/det_yolo11m \
        --calib-dir data/external/wildtrack/prepared/calib \
        --gt data/external/wildtrack/prepared/gt.jsonl --out eval/reports/zones/wildtrack
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.eval_association import in_roi, load_gt
from src.analytics_runner import git_commit
from src.association.associator import Associator
from src.association.ground_plane import sync_groups
from src.behavior.engine import BehaviorEngine
from src.common.config import DEFAULT_CONFIG_DIR, apply_overrides, load_config
from src.common.jsonl import read_frames
from src.common.logging import get_logger, setup_logging
from src.common.schemas import AlertType
from src.geometry.homography import load_calibration, point_in_polygon

log = get_logger(__name__)


def rect(x0: float, y0: float, s: float) -> list[list[float]]:
    return [[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]]


def pick_zones(
    gt: dict[int, list[dict]],
    roi: tuple[float, float, float, float],
    n_forbidden: int,
    n_risk: int,
    crowd_pct: float = 75.0,
) -> tuple[list[dict], dict[str, int]]:
    """Forbidden cells with the median-ranked visitor counts (not the busiest: an intrusion is
    rare), risk cells with the most people; K = crowd_pct percentile of the GT count there."""
    x0, y0, x1, y1 = roi

    def cells(s: float):
        return [
            (x, y) for x in np.arange(x0, x1 - s + 1e-9, s) for y in np.arange(y0, y1 - s + 1e-9, s)
        ]

    def inside(xy, c, s):
        return c[0] <= xy[0] < c[0] + s and c[1] <= xy[1] < c[1] + s

    visitors = {c: set() for c in cells(2.0)}
    counts = {c: [] for c in cells(4.0)}
    for persons in gt.values():
        pts = [(p["gt_id"], p["xy_m"]) for p in persons if p.get("xy_m")]
        for c in visitors:
            visitors[c].update(g for g, xy in pts if inside(xy, c, 2.0))
        for c in counts:
            counts[c].append(sum(inside(xy, c, 4.0) for _, xy in pts))
    busy = sorted((c for c in visitors if len(visitors[c]) >= 3), key=lambda c: len(visitors[c]))
    mid = len(busy) // 2
    forb = busy[max(0, mid - n_forbidden // 2) : max(0, mid - n_forbidden // 2) + n_forbidden]
    risk = sorted(counts, key=lambda c: -float(np.mean(counts[c])))[:n_risk]
    zones, ks = [], {}
    for i, c in enumerate(forb):
        zones.append({"id": f"F{i + 1}", "type": "forbidden", "polygon_m": rect(*c, 2.0)})
    for i, c in enumerate(risk):
        k = int(np.percentile(counts[c], crowd_pct))
        zid = f"R{i + 1}"
        zones.append({"id": zid, "type": "risk", "polygon_m": rect(*c, 4.0), "crowd_max": k})
        ks[zid] = k
    return zones, ks


def episodes(active: list[tuple[int, bool]], hold_ms: int) -> list[tuple[int, int]]:
    """[start, end] runs of active instants lasting ≥ hold_ms (start → last active instant)."""
    out, start, last = [], None, None
    for ts, a in active:
        if a:
            start = ts if start is None else start
            last = ts
        elif start is not None:
            if last - start >= hold_ms:
                out.append((start, last))
            start = None
    if start is not None and last - start >= hold_ms:
        out.append((start, last))
    return out


def gt_events(gt, zones, hold_intr_ms: int, hold_crowd_ms: int) -> dict[str, list]:
    ts_all = sorted(gt)
    ev: dict[str, list] = {}
    for z in zones:
        poly = [tuple(p) for p in z["polygon_m"]]
        if z["type"] == "forbidden":
            ids = {p["gt_id"] for ps in gt.values() for p in ps}
            out = []
            for g in ids:
                act = []
                for ts in ts_all:
                    xy = next((p.get("xy_m") for p in gt[ts] if p["gt_id"] == g), None)
                    act.append((ts, xy is not None and point_in_polygon(tuple(xy), poly)))
                out += episodes(act, hold_intr_ms)
            ev[z["id"]] = sorted(out)
        else:
            act = [
                (
                    ts,
                    sum(
                        1
                        for p in gt[ts]
                        if p.get("xy_m") and point_in_polygon(tuple(p["xy_m"]), poly)
                    )
                    > z["crowd_max"],
                )
                for ts in ts_all
            ]
            ev[z["id"]] = episodes(act, hold_crowd_ms)
    return ev


def match(alerts: list[int], events: list[tuple[int, int]], delta_ms: int) -> tuple[int, int, int]:
    free = sorted(alerts)
    tp = 0
    for a, b in events:
        hit = next((t for t in free if a - delta_ms <= t <= b + delta_ms), None)
        if hit is not None:
            free.remove(hit)
            tp += 1
    return tp, len(free), len(events) - tp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--roi", type=float, nargs=4, default=[-3, -9, 9, 27])
    ap.add_argument("--n-forbidden", type=int, default=6)
    ap.add_argument("--n-risk", type=int, default=4)
    ap.add_argument("--crowd-pct", type=float, default=75.0, help="K percentile of GT count")
    ap.add_argument("--delta-ms", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    setup_logging()

    gt = load_gt(args.gt)
    zones, _ = pick_zones(gt, tuple(args.roi), args.n_forbidden, args.n_risk, args.crowd_pct)
    zones_cfg = {**load_config("zones", Path(args.config)), "zones": zones}
    bcfg = apply_overrides(load_config("behavior", Path(args.config)), {"cooldown_s": 0.0})
    streams = {
        p.stem: sorted(read_frames(p), key=lambda f: f.ts_ms)
        for p in sorted(args.jsonl.glob("*.jsonl"))
    }
    calibs = {c: load_calibration(args.calib_dir / f"{c}.yaml") for c in streams}
    assoc = Associator.from_config(zones_cfg, calibs)
    engine = BehaviorEngine(bcfg, zones_cfg)
    x0, y0, x1, y1 = args.roi
    roi = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    alerts: dict[str, list[int]] = defaultdict(list)
    count_err: dict[str, list[int]] = defaultdict(list)
    sync_tol = load_config("cameras", Path(args.config)).get("sync_tol_ms", 50)
    for group in sync_groups(streams, sync_tol):
        ts = min(f.ts_ms for f in group.values())
        if ts not in gt:
            continue
        kept = {
            c: f.model_copy(
                update={"persons": [p for p in f.persons if in_roi(assoc.floor_xy(c, p), roi)]}
            )
            for c, f in group.items()
        }
        gid_of = assoc.step(kept, ts)
        members: dict[int, list] = defaultdict(list)
        for c, f in kept.items():
            for p in f.persons:
                xy = assoc.floor_xy(c, p)
                if not np.isnan(xy[0]):
                    members[gid_of[(c, p.track_id)]].append((xy, p.truncated_bottom))
        floor_of = {}
        for g, obs in members.items():
            xs = [xy for xy, cut in obs if not cut] or [xy for xy, _ in obs]
            floor_of[g] = tuple(np.mean(xs, axis=0).tolist())
        for a in engine.step(kept, ts, gid_of, floor_of):
            if a.type in (AlertType.INTRUSION, AlertType.CROWD):
                alerts[a.zone].append(a.t_raise_ms)
        for z in zones:
            if z["type"] != "risk":
                continue
            poly = [tuple(p) for p in z["polygon_m"]]
            n_pred = sum(1 for xy in floor_of.values() if point_in_polygon(xy, poly))
            n_gt = sum(
                1 for p in gt[ts] if p.get("xy_m") and point_in_polygon(tuple(p["xy_m"]), poly)
            )
            count_err[z["id"]].append(n_pred - n_gt)

    hold_intr = int(bcfg["intrusion"]["T_intr_s"] * 1000)
    hold_crowd = int(bcfg["crowd"]["T_s"] * 1000)
    events = gt_events(gt, zones, hold_intr, hold_crowd)
    duration_h = (max(gt) - min(gt)) / 3.6e6
    rows, totals = [], {"forbidden": [0, 0, 0], "risk": [0, 0, 0]}
    for z in zones:
        tp, fp, fn = match(alerts[z["id"]], events[z["id"]], args.delta_ms)
        for i, v in enumerate((tp, fp, fn)):
            totals[z["type"]][i] += v
        ce = np.array(count_err.get(z["id"], [0]))
        rows.append(
            {
                "zone": z["id"],
                "type": z["type"],
                "polygon_m": z["polygon_m"],
                "crowd_max": z.get("crowd_max"),
                "gt_events": len(events[z["id"]]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "count_mae": float(np.abs(ce).mean()) if z["type"] == "risk" else None,
                "count_bias": float(ce.mean()) if z["type"] == "risk" else None,
            }
        )

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, 2 * p * r / (p + r) if p + r else 0.0

    lines = [
        f"<!-- generated by eval/eval_zones.py @ {git_commit()} -->",
        f"WILDTRACK {args.jsonl.name}, {len(streams)} cams, {duration_h * 3600:.0f} s @ 2 fps, "
        f"δ {args.delta_ms} ms, T_intr {hold_intr} ms, crowd T {hold_crowd} ms, cooldown 0",
        "",
        "| Type | GT events | TP | FP | FN | Precision | Recall | F1 | FA/hour |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t, name in (("forbidden", "INTRUSION"), ("risk", "CROWD")):
        tp, fp, fn = totals[t]
        p, r, f1 = prf(tp, fp, fn)
        lines.append(
            f"| {name} | {tp + fn} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | {f1:.3f} | "
            f"{fp / duration_h:.1f} |"
        )
    lines += [
        "",
        "| Zone | Type | K | GT events | TP | FP | FN | count MAE (bias) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cnt = "" if r["count_mae"] is None else f"{r['count_mae']:.2f} ({r['count_bias']:+.2f})"
        k = "" if r["crowd_max"] is None else r["crowd_max"]
        lines.append(
            f"| {r['zone']} | {r['type']} | {k} | "
            f"{r['gt_events']} | {r['tp']} | {r['fp']} | {r['fn']} | {cnt} |"
        )
    print("\n".join(lines))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "results.json").write_text(
            json.dumps({"commit": git_commit(), "zones": rows, "totals": totals}, indent=2),
            encoding="utf-8",
        )
        (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
