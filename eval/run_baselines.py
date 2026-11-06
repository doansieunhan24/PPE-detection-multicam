"""Main table: B0–B4 vs P on one or more labelled sessions (CLAUDE.md §10.3-1, §10.3-2).

    python -m eval.run_baselines --sessions data/jsonl/s01 data/jsonl/s02 \
        --gt data/annotations --out eval/reports/main
    python -m eval.run_baselines ... --ablations   # also P variants (no v / no UNKNOWN / ...)
    python -m eval.run_baselines ... --single-cam-all   # B0/B1 on every camera, not only primary
    python -m eval.run_baselines ... --tuned eval/reports/c2/tune/tuned.json   # val-tuned params
    python -m eval.run_baselines ... --cams cam1 cam2   # camera subset (§10.3-3)
    python -m eval.run_baselines ... --per-cam yes   # score every camera as its own system

Per camera (default when zones.yaml association.mode = none, i.e. no cross-camera matching):
each (session, camera) runs alone and is scored against the full GT, then units are pooled;
FA/hour is per camera-hour. Multi-camera rules (B2, B3) and B0/B1@cam are dropped, B0/B1 run
on the unit's camera. INTRUSION is scored only on calibrated cameras.

Each session dir holds <cam_id>.jsonl; GT lives in <gt>/<session_name>/{tracks,intervals}.csv.
Writes per-run analytics outputs and <out>/results.{json,md}: the PPE table (all PPE types
pooled), per-type rows (PPE_HELMET / PPE_VEST / PPE_BOOTS and the behavior types, which do not
depend on the PPE policy) and cross-camera association of the P run.

--tuned JSON: {"*": {dotted fusion key: value}, "P": {...}, "B1": {...}}; a variant uses "*",
then its policy's entry, then its own name's entry (eval/tune_c2.py writes it).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from eval.metrics import (
    GidGT,
    TrackGT,
    association_metrics,
    event_metrics,
    gt_for_cams,
    load_gt_csv,
    unknown_fraction,
)
from src.analytics_runner import git_commit, run
from src.common.config import DEFAULT_CONFIG_DIR, load_config
from src.common.logging import get_logger, setup_logging
from src.common.schemas import AlertEvent, PersonPartState

log = get_logger(__name__)

PPE_TYPES = ["PPE_HELMET", "PPE_VEST", "PPE_BOOTS"]
BEHAVIOR_TYPES = ["INTRUSION", "FALL", "IMMOBILE", "CROWD"]
ABLATIONS = {
    "P-no_v": {"use_visibility": False},
    "P-no_unknown": {"allow_unknown": False},
    "P-no_temporal": {"temporal": False},
    "P-dempster_shafer": {"method": "dempster_shafer"},
    "P-v_learned": {"v_learned": True},
}
SINGLE_CAM = ("B0", "B1")
MULTI_CAM = ("B2", "B3")  # rules over several cameras; same as B1 on a single camera


def per_camera_default(config_dir: Path) -> bool:
    """Score per camera when the system does not match people across cameras."""
    assoc = load_config("zones", config_dir).get("association", {})
    return assoc.get("mode", "ground_plane") == "none"


def eval_units(
    sessions: list[Path], per_cam: bool, cams: list[str] | None = None
) -> list[tuple[Path, list[str] | None, str]]:
    """(session, cameras to run, tag): one unit per session, or per (session, camera)."""
    if not per_cam:
        return [(s, cams, s.name) for s in sessions]
    return [
        (s, [c], f"{s.name}/{c}")
        for s in sessions
        for c in sorted(p.stem for p in s.glob("*.jsonl"))
        if cams is None or c in cams
    ]


def unit_overrides(unit_cams: list[str] | None, per_cam: bool) -> dict:
    """Single-camera baselines read baselines.primary_cam: point it at the unit's camera."""
    return {"baselines.primary_cam": unit_cams[0]} if per_cam and unit_cams else {}


def _read_jsonl(path: Path, model):
    with open(path, encoding="utf-8") as f:
        return [model.model_validate_json(ln) for ln in f if ln.strip()]


def load_track_gt(tracks_csv: Path) -> TrackGT:
    return TrackGT.from_csv(tracks_csv)


def _gid_rows(run_dir: Path):
    with open(run_dir / "gid_map.jsonl", encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            yield d["ts_ms"], d["cam_id"], d["track_id"], d["global_id"]


def gid_to_gt(run_dir: Path, tracks_csv: Path) -> GidGT:
    run_json = run_dir / "run.json"
    ov: dict = {}
    if run_json.exists():
        ov = json.loads(run_json.read_text(encoding="utf-8")).get("fusion_overrides", {})
    stitched = ov.get("tracking.stitch_gap_ms") is not None
    return GidGT(_gid_rows(run_dir), load_track_gt(tracks_csv), stitched=stitched)


def evaluate(run_dir: Path, gt_dir: Path, delta_ms: int, association: bool = False) -> dict:
    """PPE metrics (types pooled) + per-type counts (+ association when asked)."""
    alerts = _read_jsonl(run_dir / "alerts.jsonl", AlertEvent)
    states = _read_jsonl(run_dir / "states.jsonl", PersonPartState)
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    gt = gt_for_cams(load_gt_csv(gt_dir / "intervals.csv"), summary.get("cams", []))
    mapping = gid_to_gt(run_dir, gt_dir / "tracks.csv")
    dur = summary["duration_s"]
    m = event_metrics(alerts, gt, mapping, dur, delta_ms, types=PPE_TYPES)
    per_type = {
        t: {
            **event_metrics(alerts, gt, mapping, dur, delta_ms, types=[t]).as_dict(),
            "duration_s": dur,
        }
        for t in PPE_TYPES + BEHAVIOR_TYPES
    }
    if not set(summary.get("calibrated_cams", [])) & set(summary.get("cams", [])):
        # no floor map → intrusion cannot fire; leave these cameras out of the INTRUSION row
        per_type["INTRUSION"] = {**_prf(0, 0, 0, dur), "n_gt": 0, "mean_ttd_s": None}
    unk = unknown_fraction(ps.state for s in states for ps in s.parts.values())
    out = {**m.as_dict(), "unknown_frac": unk, "duration_s": dur, "per_type": per_type}
    if association:
        out["association"] = association_metrics(
            _gid_rows(run_dir), load_track_gt(gt_dir / "tracks.csv")
        )
    return out


def _prf(tp: int, fp: int, fn: int, dur: float) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fa_per_hour": fp / (dur / 3600) if dur else float("nan"),
        "duration_s": dur,
    }


def _pool_ttd(rows: list[dict]) -> float | None:
    """Time-to-detect pooled over matched alerts (weighted by each session's TP)."""
    w = [(r["mean_ttd_s"], r["tp"]) for r in rows if r.get("mean_ttd_s") is not None]
    n = sum(tp for _, tp in w)
    return sum(t * tp for t, tp in w) / n if n else None


def _pool(rows: list[dict]) -> dict:
    """Pool sessions: sum counts, recompute rates."""
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    dur = sum(r["duration_s"] for r in rows)
    out = _prf(tp, fp, fn, dur)
    out["n_gt"] = tp + fn
    out["fp_duplicate"] = sum(r.get("fp_duplicate", 0) for r in rows)
    out["mean_ttd_s"] = _pool_ttd(rows)
    out["unknown_frac"] = sum(r["unknown_frac"] * r["duration_s"] for r in rows) / dur if dur else 0
    if all("per_type" in r for r in rows) and rows:
        out["per_type"] = {}
        for t in rows[0]["per_type"]:
            pr = [r["per_type"][t] for r in rows]
            d = _prf(*(sum(p[k] for p in pr) for k in ("tp", "fp", "fn")), dur)
            out["per_type"][t] = {**d, "n_gt": d["tp"] + d["fn"], "mean_ttd_s": _pool_ttd(pr),
                                  "fp_duplicate": sum(p.get("fp_duplicate", 0) for p in pr)}
    if all("association" in r for r in rows) and rows:
        a = [r["association"] for r in rows]
        tp_a, fp_a, fn_a = (sum(x[k] for x in a) for k in ("pair_tp", "pair_fp", "pair_fn"))
        n = sum(x["instants"] for x in a)
        p = _prf(tp_a, fp_a, fn_a, dur)
        out["association"] = {
            "pair_tp": tp_a,
            "pair_fp": fp_a,
            "pair_fn": fn_a,
            "pair_precision": p["precision"],
            "pair_recall": p["recall"],
            "pair_f1": p["f1"],
            "count_mae": sum(x["count_mae"] * x["instants"] for x in a) / n if n else 0.0,
            "frag": sum(x["frag"] for x in a) / len(a),
            "instants": n,
        }
    return out


def session_cams(sessions: list[Path]) -> list[str]:
    """Cameras present in every session (sorted)."""
    sets = [{p.stem for p in s.glob("*.jsonl")} for s in sessions]
    return sorted(set.intersection(*sets)) if sets else []


def build_variants(
    ablations: bool, config_dir: Path, single_cam_all: bool, cams: list[str], per_cam: bool = False
) -> dict[str, tuple[str, dict, dict]]:
    """name → (policy, ablation, extra fusion overrides)."""
    variants: dict[str, tuple[str, dict, dict]] = {
        p: (p, {}, {})
        for p in ("B0", "B1", "B2", "B3", "B4", "P")
        if not (per_cam and p in MULTI_CAM)
    }
    if single_cam_all and not per_cam:
        primary = load_config("fusion", config_dir)["baselines"]["primary_cam"]
        for b in SINGLE_CAM:
            for cam in cams:
                if cam != primary:
                    variants[f"{b}@{cam}"] = (b, {}, {"baselines.primary_cam": cam})
    if ablations:
        variants.update({k: ("P", v, {}) for k, v in ABLATIONS.items()})
        learned = load_config("fusion", config_dir)["visibility"]["learned_path"]
        if not Path(learned).exists():
            log.warning(
                "skip P-v_learned: %s missing (run training/visibility/calibrate.py)", learned
            )
            variants.pop("P-v_learned")
    return variants


def overrides_for(name: str, policy: str, tuned: dict) -> dict:
    return {**tuned.get("*", {}), **tuned.get(policy, {}), **tuned.get(name, {})}


def _f(x: float | None, fmt: str = ".3f") -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def render_md(meta: dict, results: dict[str, dict]) -> list[str]:
    lines = [
        f"<!-- generated by eval/run_baselines.py @ {meta['commit']} -->",
        f"Sessions: {', '.join(Path(s).name for s in meta['sessions'])}"
        + (f" · cams: {', '.join(meta['cams'])}" if meta.get("cams") else ""),
        "",
    ]
    if meta.get("per_cam"):
        lines += [
            "Per camera: each camera scored as its own system, pooled; FA/hour per camera-hour.",
            "",
        ]
    lines += [
        "| Method | Precision | Recall | F1 | FA/hour | %UNKNOWN | TTD (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, r in results.items():
        lines.append(
            f"| {k} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['fa_per_hour']:.2f} | {100 * r['unknown_frac']:.1f} | "
            f"{_f(r.get('mean_ttd_s'), '.2f')} |"
        )
    if any("per_type" in r for r in results.values()):
        lines += ["", "PPE per item (F1 / FA/hour, n_gt):", ""]
        lines += ["| Method | " + " | ".join(PPE_TYPES) + " |", "|---" * (len(PPE_TYPES) + 1) + "|"]
        for k, r in results.items():
            cells = [
                f"{r['per_type'][t]['f1']:.3f} / {r['per_type'][t]['fa_per_hour']:.2f} "
                f"({r['per_type'][t]['n_gt']})"
                for t in PPE_TYPES
            ]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
    by_cam = [k for k, r in results.items() if "by_cam" in r]
    if by_cam:
        cams = list(results[by_cam[0]]["by_cam"])
        lines += ["", "Per camera (F1 / FA/hour / %UNKNOWN):", ""]
        lines += ["| Method | " + " | ".join(cams) + " |", "|---" * (len(cams) + 1) + "|"]
        for k in by_cam:
            bc = results[k]["by_cam"]
            cells = [
                f"{bc[c]['f1']:.3f} / {bc[c]['fa_per_hour']:.2f} / "
                f"{100 * bc[c]['unknown_frac']:.1f}"
                for c in cams
            ]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
    p = results.get("P")
    if p and "per_type" in p:
        lines += [
            "",
            "Behavior (P run; independent of the PPE policy):",
            "",
            "| Type | n_gt | Precision | Recall | F1 | FA/hour | TTD (s) |",
            "|---|---|---|---|---|---|---|",
        ]
        for t in BEHAVIOR_TYPES:
            r = p["per_type"][t]
            lines.append(
                f"| {t} | {r['n_gt']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['f1']:.3f} | {r['fa_per_hour']:.2f} | {_f(r['mean_ttd_s'], '.2f')} |"
            )
    if p and "association" in p:
        a = p["association"]
        lines += [
            "",
            "Association (P run, labelled tracks):",
            "",
            "| Pair P | Pair R | Pair F1 | Count MAE | global_id / person | Instants |",
            "|---|---|---|---|---|---|",
            f"| {a['pair_precision']:.3f} | {a['pair_recall']:.3f} | {a['pair_f1']:.3f} | "
            f"{a['count_mae']:.3f} | {a['frag']:.2f} | {a['instants']} |",
        ]
    return lines


def run_all(
    sessions: list[Path],
    gt: Path,
    out: Path,
    config_dir: Path,
    delta_ms: int = 2000,
    ablations: bool = False,
    single_cam_all: bool = False,
    tuned: dict | None = None,
    cams: list[str] | None = None,
    obs_cache: Path | None = None,
    per_cam: bool | None = None,
) -> dict[str, dict]:
    tuned = tuned or {}
    per_cam = per_camera_default(config_dir) if per_cam is None else per_cam
    avail = session_cams(sessions)
    variants = build_variants(ablations, config_dir, single_cam_all, cams or avail, per_cam)
    units = eval_units(sessions, per_cam, cams)
    if cams and not per_cam:
        primary = load_config("fusion", config_dir)["baselines"]["primary_cam"]
        if primary not in cams:  # single-cam baselines must run on a camera of the subset
            for name, (pol, abl, ov) in list(variants.items()):
                if pol in SINGLE_CAM and "baselines.primary_cam" not in ov:
                    variants[name] = (pol, abl, {**ov, "baselines.primary_cam": cams[0]})
    results: dict[str, dict] = {}
    for name, (policy, abl, extra) in variants.items():
        ov = {**overrides_for(name, policy, tuned), **extra}
        rows, by_cam = [], {}
        for s, ucams, tag in units:
            run_dir = out / "runs" / tag / name
            run(s, run_dir, config_dir, policy, abl,
                fusion_overrides={**ov, **unit_overrides(ucams, per_cam)}, cams=ucams,
                obs_cache=obs_cache)  # fmt: skip
            r = evaluate(run_dir, gt / s.name, delta_ms, association=name == "P" and not per_cam)
            rows.append(r)
            if per_cam:
                by_cam.setdefault(ucams[0], []).append(r)
        results[name] = _pool(rows)
        results[name]["overrides"] = ov
        if per_cam:
            results[name]["by_cam"] = {c: _pool(rs) for c, rs in sorted(by_cam.items())}

    meta = {
        "commit": git_commit(),
        "sessions": [str(s) for s in sessions],
        "cams": cams or avail,
        "per_cam": per_cam,
        "delta_ms": delta_ms,
        "tuned": tuned,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8"
    )
    lines = render_md(meta, results)
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--delta-ms", type=int, default=2000)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--single-cam-all", action="store_true", help="B0/B1 on every camera")
    ap.add_argument("--tuned", help="JSON of val-tuned fusion overrides (eval/tune_c2.py)")
    ap.add_argument("--cams", nargs="+", help="camera subset (default: all)")
    ap.add_argument("--obs-cache", help="dir caching policy-independent work (faster reruns)")
    ap.add_argument(
        "--per-cam",
        choices=["auto", "yes", "no"],
        default="auto",
        help="score each camera alone (auto: when association.mode is none)",
    )
    args = ap.parse_args(argv)
    setup_logging()
    tuned = json.loads(Path(args.tuned).read_text(encoding="utf-8")) if args.tuned else {}
    run_all(
        [Path(s) for s in args.sessions],
        Path(args.gt),
        Path(args.out),
        Path(args.config),
        args.delta_ms,
        args.ablations,
        args.single_cam_all,
        tuned.get("overrides", tuned),
        args.cams,
        Path(args.obs_cache) if args.obs_cache else None,
        None if args.per_cam == "auto" else args.per_cam == "yes",
    )


if __name__ == "__main__":
    main()
