"""C2 rehearsal with REAL ground truth from WILDTRACK (dry run of eval/run_c2.py before C2).

Takes the 3-camera rehearsal session (data/rehearsal/wt3: WILDTRACK C1/C4/C6 through the
offline perception) and writes C2-format labels from the WILDTRACK annotations:

  tracks.csv     gt_id of each (cam, track): majority of per-frame IoU ≥ --iou matches to the
                 GT boxes of that camera (kept when ≥ --min-share of the track's frames agree)
  intervals.csv  WILDTRACK people wear no helmet / vest → PPE_HELMET and PPE_VEST violation
                 while visible in ≥ 1 of the 3 cameras (runs ≥ --min-s, gaps ≤ --gap-s merged);
                 INTRUSION / CROWD episodes from GT floor positions and auto-picked zones
                 (eval/eval_zones.pick_zones), written to the session's zones.yaml

The session is cut in two halves (<name>_a = val, <name>_b = test) so the val → test chain of
eval/run_c2.py can run, plus a split YAML. Every person violates the whole time, so the PPE
part only checks that the chain runs (no COMPLIANT phase → nothing about C1's false alarms);
association and zone events are meaningful.

    python tools/wildtrack_rehearsal.py --src data/rehearsal/wt3/jsonl \
        --gt data/external/wildtrack/prepared/gt.jsonl --out data/rehearsal/wt3_gt
    python -m eval.run_c2 --split data/rehearsal/wt3_gt/split.yaml \
        --out data/rehearsal/wt3_gt/report --jobs 4
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.eval_association import iou_matrix, load_gt  # noqa: E402
from eval.eval_zones import episodes, pick_zones  # noqa: E402
from src.common.config import DEFAULT_CONFIG_DIR, load_config  # noqa: E402
from src.common.jsonl import read_frames, write_frames  # noqa: E402
from src.geometry.homography import point_in_polygon  # noqa: E402
from tools.label_intervals import INTERVAL_COLS, TRACK_COLS, track_spans  # noqa: E402


def match_tracks(
    frames: dict[str, list], gt: dict[int, list[dict]], cam_map: dict[str, str], iou_min: float
) -> dict[tuple[str, int], Counter]:
    votes: dict[tuple[str, int], Counter] = defaultdict(Counter)
    for cam, frs in frames.items():
        for fr in frs:
            persons = gt.get(fr.ts_ms, [])
            g_boxes = [(p["gt_id"], p["boxes"][cam_map[cam]]) for p in persons
                       if cam_map[cam] in p.get("boxes", {})]  # fmt: skip
            if not g_boxes or not fr.persons:
                continue
            m = iou_matrix(np.array([p.bbox for p in fr.persons]),
                           np.array([b for _, b in g_boxes]))  # fmt: skip
            for r, c in zip(*linear_sum_assignment(-m), strict=True):
                if m[r, c] >= iou_min:
                    votes[(cam, fr.persons[r].track_id)][g_boxes[c][0]] += 1
    return votes


def merge_runs(ts_list: list[int], gap_ms: int, min_ms: int) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for t in sorted(ts_list):
        if out and t - out[-1][1] <= gap_ms:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out if b - a >= min_ms]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, help="rehearsal JSONL dir (+ cameras.yaml)")
    ap.add_argument("--gt", required=True, help="WILDTRACK gt.jsonl (tools/wildtrack_prepare.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam-map", nargs="+", default=["cam1=C1", "cam2=C4", "cam3=C6"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-share", type=float, default=0.5)
    ap.add_argument("--gap-s", type=float, default=2.0)
    ap.add_argument("--min-s", type=float, default=5.0)
    args = ap.parse_args(argv)
    cam_map = dict(x.split("=") for x in args.cam_map)
    src, out = Path(args.src), Path(args.out)
    gt = load_gt(Path(args.gt))
    frames = {p.stem: list(read_frames(p)) for p in sorted(src.glob("*.jsonl"))}
    t_all = sorted({f.ts_ms for frs in frames.values() for f in frs})
    t_mid = t_all[len(t_all) // 2]
    # GT restricted to the rehearsal cameras' views and time span
    gt_v = {
        ts: [p for p in ps if any(c in p.get("boxes", {}) for c in cam_map.values())]
        for ts, ps in gt.items()
        if t_all[0] <= ts <= t_all[-1]
    }
    pts = [p["xy_m"] for ps in gt_v.values() for p in ps if p.get("xy_m")]
    x0, y0 = np.floor(np.min(pts, 0))
    x1, y1 = np.ceil(np.max(pts, 0))
    zones, _ = pick_zones(gt_v, (x0, y0, x1, y1), n_forbidden=1, n_risk=1)
    zones = json.loads(json.dumps(zones, default=float))  # numpy scalars → plain YAML
    zcfg = load_config("zones", DEFAULT_CONFIG_DIR)
    zcfg["zones"] = zones
    zcfg["default_required_ppe"] = ["helmet", "vest"]
    bcfg = load_config("behavior", DEFAULT_CONFIG_DIR)
    hold_intr = int(bcfg["intrusion"]["T_intr_s"] * 1000)
    hold_crowd = int(bcfg["crowd"]["T_s"] * 1000)

    votes = match_tracks(frames, gt_v, cam_map, args.iou)
    spans = track_spans(src)
    gap, min_ms = int(args.gap_s * 1000), int(args.min_s * 1000)
    stats: Counter = Counter()
    split = {"jsonl_root": str(out / "jsonl"), "gt_root": str(out / "ann"), "val": [], "test": []}
    for half, (lo, hi) in (("a", (t_all[0], t_mid)), ("b", (t_mid, t_all[-1] + 1))):
        name = f"{src.parent.name}_{half}"
        jd, ad = out / "jsonl" / name, out / "ann" / name
        jd.mkdir(parents=True, exist_ok=True)
        ad.mkdir(parents=True, exist_ok=True)
        for cam, frs in frames.items():
            write_frames(jd / f"{cam}.jsonl", [f for f in frs if lo <= f.ts_ms < hi])
        shutil.copy(src / "cameras.yaml", jd / "cameras.yaml")
        (jd / "zones.yaml").write_text(yaml.safe_dump(zcfg, sort_keys=False), encoding="utf-8")
        with open(ad / "tracks.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(TRACK_COLS)
            for (cam, tid), (a, b, n) in sorted(track_spans(jd).items()):
                v = votes.get((cam, tid))
                g = ""
                if v:
                    g_id, k = v.most_common(1)[0]
                    if k >= args.min_share * spans[(cam, tid)][2]:
                        g = g_id
                stats[f"{half}_tracks_labelled" if g != "" else f"{half}_tracks_unlabelled"] += 1
                w.writerow([cam, tid, a, b, n, g])
        gts = {ts: ps for ts, ps in gt_v.items() if lo <= ts < hi}
        rows = []
        seen: dict[int, list[int]] = defaultdict(list)
        for ts, ps in gts.items():
            for p in ps:
                seen[p["gt_id"]].append(ts)
        for g, tss in sorted(seen.items()):
            for a, b in merge_runs(tss, gap, min_ms):
                rows += [[g, "PPE_HELMET", a, b, "rehearsal"], [g, "PPE_VEST", a, b, "rehearsal"]]
        ts_sorted = sorted(gts)
        for z in zones:
            poly = [tuple(q) for q in z["polygon_m"]]
            if z["type"] == "forbidden":
                for g in seen:
                    act = []
                    for ts in ts_sorted:
                        xy = next((p.get("xy_m") for p in gts[ts] if p["gt_id"] == g), None)
                        act.append((ts, xy is not None and point_in_polygon(tuple(xy), poly)))
                    rows += [[g, "INTRUSION", a, b, z["id"]] for a, b in episodes(act, hold_intr)]
            elif z["type"] == "risk":
                act = [(ts, sum(1 for p in gts[ts] if p.get("xy_m")
                               and point_in_polygon(tuple(p["xy_m"]), poly)) > z["crowd_max"])
                       for ts in ts_sorted]  # fmt: skip
                rows += [["", "CROWD", a, b, z["id"]] for a, b in episodes(act, hold_crowd)]
        with open(ad / "tracks.csv", encoding="utf-8") as f:
            labelled = {int(r["gt_id"]) for r in csv.DictReader(f) if r["gt_id"]}
        # a GT person no detector track ever matched cannot pass label validation; keep them
        # out (detector misses still count through partially matched people)
        rows = [r for r in rows if r[0] == "" or r[0] in labelled]
        for r in rows:
            stats[f"{half}_{r[1]}"] += 1
        with open(ad / "intervals.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(INTERVAL_COLS)
            w.writerows(sorted(rows, key=lambda r: (r[2], r[1])))
        split["val" if half == "a" else "test"].append(name)
    (out / "split.yaml").write_text(yaml.safe_dump(split, sort_keys=False), encoding="utf-8")
    (out / "rehearsal.json").write_text(
        json.dumps({"zones": zones, "stats": dict(stats), "t_mid_ms": t_mid}, indent=2), "utf-8"
    )
    print(json.dumps(dict(stats), indent=1))
    print(f"-> {out / 'split.yaml'}")


if __name__ == "__main__":
    main()
