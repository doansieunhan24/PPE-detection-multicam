"""Fall detection event metrics on UR Fall (CLAUDE.md §7.7, §10.1 Behavior; P6 "Done khi").

Runs src/behavior BehaviorEngine on PerceptionFrame JSONL from tools/urfall_prepare.py.
UR Fall has one actor per sequence and no floor calibration, so each camera's most confident
person is that actor (global id 0); with --cams cam0 cam1 the per-camera fall scores are
fused by body visibility exactly as in the live system (falls only: ADL has no cam1).

Event matching (§10.2): a FALL alert on a fall sequence is a TP if t_raise lies in
[t_start − δ, t_end + δ] of the GT episode (first "falling" → last "lying" frame); at most one
TP per sequence; every other FALL alert is a false alarm. ADL sequences (lying down on purpose,
sitting, crouching, picking objects up) only contribute false alarms.

    python -m eval.eval_fall --root data/external/urfall --out eval/reports/fall/urfall_cam0
    python -m eval.eval_fall ... --cams cam0 cam1 --kinds fall
    python -m eval.eval_fall ... --param fall.score_threshold 0.4 0.5 0.6 0.7 0.8 --split val

--split val = odd sequence numbers (tune here), test = even (report here), all = everything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.analytics_runner import git_commit
from src.association.ground_plane import sync_groups
from src.behavior.engine import BehaviorEngine
from src.common.config import DEFAULT_CONFIG_DIR, apply_overrides, load_config
from src.common.jsonl import read_frames
from src.common.logging import get_logger, setup_logging
from src.common.schemas import AlertType, PerceptionFrame

log = get_logger(__name__)


def single_actor(fr: PerceptionFrame) -> PerceptionFrame:
    """Keep the most confident person (the dataset's only actor)."""
    if len(fr.persons) <= 1:
        return fr
    return fr.model_copy(update={"persons": [max(fr.persons, key=lambda p: p.conf)]})


def _num(v: str) -> float | str:
    try:
        return float(v)
    except ValueError:
        return v


def in_split(seq: str, split: str) -> bool:
    """val = odd sequence number, test = even (fall-03 → val, adl-12 → test)."""
    if split == "all":
        return True
    odd = int(seq.rsplit("-", 1)[1]) % 2 == 1
    return odd == (split == "val")


def run_sequence(
    streams: dict[str, list[PerceptionFrame]], behavior_cfg: dict, sync_tol_ms: int = 50
) -> list[int]:
    """t_raise_ms of every FALL alert."""
    engine = BehaviorEngine(behavior_cfg, {"zones": []})
    raised = []
    for group in sync_groups(streams, sync_tol_ms):
        group = {c: single_actor(f) for c, f in group.items()}
        ts = min(f.ts_ms for f in group.values())
        gid_of = {(c, p.track_id): 0 for c, f in group.items() for p in f.persons}
        for a in engine.step(group, ts, gid_of, {}):
            if a.type == AlertType.FALL:
                raised.append(a.t_raise_ms)
    return raised


def score(per_seq: dict[str, dict], gt: dict[str, dict], delta_ms: int) -> dict:
    tp = fp = n_pos = 0
    latency: list[int] = []
    dur_ms = 0
    for seq, r in per_seq.items():
        g = gt[seq]
        dur_ms += round(g["n_frames"] * 1000 / g["fps"])
        alerts = sorted(r["alerts"])
        hit = None
        if g["fall"]:
            n_pos += 1
            a, b = g["fall"]
            hit = next((t for t in alerts if a - delta_ms <= t <= b + delta_ms), None)
        if hit is not None:
            tp += 1
            latency.append(hit - g["fall"][0])
            fp += len(alerts) - 1
        else:
            fp += len(alerts)
        r["tp"] = hit is not None
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_pos if n_pos else 0.0
    hours = dur_ms / 3.6e6
    return {
        "sequences": len(per_seq),
        "falls": n_pos,
        "tp": tp,
        "fp": fp,
        "fn": n_pos - tp,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fa_per_hour": fp / hours if hours else float("nan"),
        "duration_s": dur_ms / 1000,
        "latency_ms_median": float(np.median(latency)) if latency else None,
        "fp_sequences": sorted(s for s, r in per_seq.items() if len(r["alerts"]) > r["tp"]),
        "missed": sorted(s for s, r in per_seq.items() if gt[s]["fall"] and not r["tp"]),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("data/external/urfall"))
    ap.add_argument("--cams", nargs="+", default=["cam0"])
    ap.add_argument("--kinds", nargs="+", default=["fall", "adl"], choices=["fall", "adl"])
    ap.add_argument("--split", choices=["all", "val", "test"], default="all")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--delta-ms", type=int, default=2000)
    ap.add_argument(
        "--param", nargs="+", metavar=("KEY", "VALUE"), help="behavior.yaml key and values"
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    setup_logging()

    gt = json.loads((args.root / "gt.json").read_text(encoding="utf-8"))
    base = load_config("behavior", Path(args.config))
    data: dict[str, dict[str, list[PerceptionFrame]]] = {}
    for seq, g in gt.items():
        if g["kind"] not in args.kinds or not in_split(seq, args.split):
            continue
        paths = [args.root / "jsonl" / seq / f"{c}.jsonl" for c in args.cams]
        if not all(p.exists() for p in paths):
            continue  # cam1 exists for falls only
        data[seq] = {p.stem: sorted(read_frames(p), key=lambda f: f.ts_ms) for p in paths}
    if not data:
        raise SystemExit(f"no JSONL under {args.root / 'jsonl'} for {args.cams}")

    key, values = None, [None]
    if args.param:
        key, values = args.param[0], [_num(v) for v in args.param[1:]]
    results = []
    for v in values:
        cfg = apply_overrides(base, {key: v}) if key else base
        per_seq = {s: {"alerts": run_sequence(st, cfg)} for s, st in data.items()}
        r = {"param": key, "value": v, **score(per_seq, gt, args.delta_ms)}
        results.append(r)
        log.info(
            "%s=%s: P/R/F1 %.3f/%.3f/%.3f  TP %d FP %d FN %d  FA/h %.1f  latency %s ms",
            key,
            v,
            r["precision"],
            r["recall"],
            r["f1"],
            r["tp"],
            r["fp"],
            r["fn"],
            r["fa_per_hour"],
            r["latency_ms_median"],
        )
    meta = {
        "commit": git_commit(),
        "cams": args.cams,
        "kinds": args.kinds,
        "split": args.split,
        "delta_ms": args.delta_ms,
        "fall_cfg": base["fall"],
    }
    lines = [
        f"<!-- generated by eval/eval_fall.py @ {meta['commit']} -->",
        f"UR Fall, cams {' + '.join(args.cams)}, sequences {'+'.join(args.kinds)} "
        f"({args.split}), "
        f"δ {args.delta_ms} ms",
        "",
        "| Param | Value | Seq | Falls | TP | FP | FN | Precision | Recall | F1 | FA/hour | "
        "latency median (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        v = r["value"]
        value = "" if v is None else f"{v:g}" if isinstance(v, float) else str(v)
        lines.append(
            f"| {r['param'] or '(config)'} | {value} | "
            f"{r['sequences']} | {r['falls']} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['fa_per_hour']:.1f} | "
            f"{r['latency_ms_median']} |"
        )
    print("\n".join(lines))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "results.json").write_text(
            json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8"
        )
        (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
