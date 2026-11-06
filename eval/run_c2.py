"""All C2 experiments in one command, val → test (CLAUDE.md §10.3; split in configs/c2_split.yaml).

    python -m eval.run_c2 --out eval/reports/c2 --jobs 4
    python -m eval.run_c2 --out eval/reports/c2 --steps check          # only the checks
    python -m eval.run_c2 --out eval/reports/c2 --steps main numcams   # reuse earlier steps

Steps (each writes <out>/<step>/ and can be rerun alone; later steps read earlier outputs):
  check    sessions exist, val ∩ test = ∅, labels validate, cameras calibrated
  calib    evidence tables (training/visibility/build_v_dataset_c2.py) on val/test, fit p
           calibration and v_learned on val (parts without enough violations keep the SH17 fit)
           → base.json: C2 p calibration for every method, C2 v_learned for P-v_learned
  tune     per-method coordinate descent on val (eval/tune_c2.py) → tune/tuned.json
  main     B0–B4 (+ B0/B1 on every camera) vs P + ablations on test with tuned values;
           also with untuned config values (main_default) and on val (main_val, sanity)
  sens     sensitivity curves around the tuned point on test (eval/run_sensitivity.py)
  numcams  1 / 2 / 3 cameras on test (eval/run_num_cams.py)
  report   <out>/REPORT.md gathering every table, with commit and split

--no-refit skips calib (base = config files fitted on SH17/CHV).

--cv (sessions listed under `cv:` in the split file, e.g. the takes of one recording session):
leave-one-session-out cross-validation. For each fold, every method is tuned on the other
sessions with the same procedure and scored on the held-out one; held-out counts are pooled
over the folds (<out>/cv/pooled). Runs are shared between folds (a run depends only on its
session, camera, method and values); calib is not refitted per fold (SH17/CHV fit).

    python -m eval.run_c2 --cv --out eval/reports/c2_s01 --jobs 4

Per camera (zones.yaml association.mode = none, the default since 2026-10-31): tune, main and
sens score every (session, camera) as its own system and pool them (FA/hour per camera-hour);
B2/B3 are dropped, numcams is skipped and the main table gets a per-camera breakdown instead.
Cameras without homography are allowed (they only miss the INTRUSION rows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eval import run_baselines, run_num_cams, run_sensitivity, tune_c2
from src.analytics_runner import git_commit
from src.common.config import DEFAULT_CONFIG_DIR, load_config
from src.common.logging import get_logger, setup_logging

log = get_logger(__name__)
SCORING = {True: "per camera (each camera alone, pooled)", False: "per person across cameras"}
STEPS = ["check", "calib", "tune", "main", "sens", "numcams", "report"]
# --space per_part (exploratory, 2026-11-06): P tunes theta_neg separately for head and torso
_PP = {k: v for k, v in tune_c2.SPACE["P"].items() if k != "evidence.theta_neg"}
SPACES = {
    "default": tune_c2.SPACE,
    "per_part": {"P": {**_PP, "evidence.theta_neg_part.head": [-2.0, -1.0, -0.5, -0.25],
                       "evidence.theta_neg_part.torso": [-2.0, -1.0, -0.5, -0.25, -0.1]}},
}  # fmt: skip


def load_split(path: Path) -> dict:
    sp = yaml.safe_load(path.read_text(encoding="utf-8"))
    root, gt = Path(sp["jsonl_root"]), Path(sp["gt_root"])
    return {
        "val": [root / s for s in sp["val"]],
        "test": [root / s for s in sp["test"]],
        "gt": gt,
        "raw": sp,
    }


def check(split: dict, config_dir: Path) -> list[str]:
    """Problems that would make the numbers wrong; empty list = OK."""
    from tools.label_intervals import main as label_main

    errs: list[str] = []
    names_v = {s.name for s in split["val"]}
    names_t = {s.name for s in split["test"]}
    if not names_v or not names_t:
        errs.append("split: val and test must both be non-empty")
    if names_v & names_t:
        errs.append(f"split: sessions in both val and test: {sorted(names_v & names_t)}")
    per_cam = run_baselines.per_camera_default(config_dir)
    cams_ref = None
    for s in split["val"] + split["test"]:
        jl = sorted(p.stem for p in s.glob("*.jsonl"))
        if len(jl) < (1 if per_cam else 2):
            errs.append(f"{s}: {len(jl)} camera JSONL (need ≥ {1 if per_cam else 2})")
            continue
        if cams_ref is not None and jl != cams_ref:
            log.warning("%s: cameras %s differ from %s", s, jl, cams_ref)
        cams_ref = cams_ref or jl
        g = split["gt"] / s.name
        for f in ("tracks.csv", "intervals.csv"):
            if not (g / f).exists():
                errs.append(f"{g / f} missing (tools/label_intervals.py)")
        if all((g / f).exists() for f in ("tracks.csv", "intervals.csv")):
            try:
                label_main(["validate", "--jsonl", str(s), "--ann", str(g)])
            except SystemExit as e:
                if e.code:
                    errs.append(f"{g}: label validation failed (see output above)")
        from src.analytics_runner import load_calibs

        cams_cfg = load_config("cameras", config_dir, override_dir=s)
        calibs = load_calibs(cams_cfg, config_dir.parent)
        missing = [c for c in jl if c not in calibs or not calibs[c].calibrated]
        if missing and per_cam and len(missing) < len(jl):
            # homography only serves the forbidden zone; these cams are left out of INTRUSION
            log.warning("%s: no homography for %s (not scored for INTRUSION)", s, missing)
        elif missing:
            errs.append(f"{s}: no homography for {missing} (tools/calibrate_homography.py)")
    return errs


def step_calib(split: dict, out: Path, config_dir: Path) -> dict:
    from training.visibility import calibrate
    from training.visibility.build_v_dataset_c2 import build

    d = out / "calib"
    d.mkdir(parents=True, exist_ok=True)
    build(split["val"], split["gt"], d / "c2_val.csv", config_dir)
    build(split["test"], split["gt"], d / "c2_test.csv", config_dir)
    fusion = load_config("fusion", config_dir)
    pcal, vl = d / "p_calibration_c2.yaml", d / "visibility_learned_c2.yaml"
    calibrate.main([
        "--val", str(d / "c2_val.csv"),
        "--test", f"c2_test={d / 'c2_test.csv'}",
        "--out", str(d),
        "--min-class", "50",
        "--write-p-calibration", str(pcal),
        "--write-config", str(vl),
        "--fallback-p-calibration", fusion["part_assign"]["p_calibration"],
        "--fallback-config", fusion["visibility"]["learned_path"],
    ])  # fmt: skip
    base = {
        "*": {"part_assign.p_calibration": str(pcal)},
        "P-v_learned": {"visibility.learned_path": str(vl)},
    }
    (out / "base.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    return base


def _parse_set(kv: str) -> tuple[str, object]:
    k, v = kv.split("=", 1)
    return k, yaml.safe_load(v)


def pool_results(per_fold: list[dict[str, dict]]) -> dict[str, dict]:
    """Pool held-out results of the folds per method (counts summed, rates recomputed)."""
    out: dict[str, dict] = {}
    for name in per_fold[0]:
        rows = [f[name] for f in per_fold if name in f]
        out[name] = run_baselines._pool(rows)
        if all("by_cam" in r for r in rows):
            cams = sorted({c for r in rows for c in r["by_cam"]})
            out[name]["by_cam"] = {
                c: run_baselines._pool([r["by_cam"][c] for r in rows if c in r["by_cam"]])
                for c in cams
            }
    return out


def step_cv(sessions: list[Path], gt: Path, out: Path, cfg: Path, args) -> Path:
    """Leave-one-session-out: tune on the others, score the held-out one, pool (see --cv)."""
    d = out / "cv"
    shared, cache = d / "tune_runs", out / "obs_cache"
    folds: list[dict[str, dict]] = []
    tuned_rows = []
    for held in sessions:
        val = [s for s in sessions if s != held]
        fd = d / f"fold_{held.name}"
        tp = fd / "tune" / "tuned.json"
        base: dict = {}
        if args.cv_refit:  # p calibration + v_learned fitted on this fold's val sessions only
            bp = fd / "base.json"
            if bp.exists() and not args.retune:
                base = json.loads(bp.read_text(encoding="utf-8"))
            else:
                base = step_calib({"val": val, "test": [held], "gt": gt}, fd, cfg)
        if args.set:  # fixed fusion overrides for every method (e.g. tracking.stitch_gap_ms=1500)
            fixed = dict(_parse_set(kv) for kv in args.set)
            base = {**base, "*": {**base.get("*", {}), **fixed}}
        if tp.exists() and not args.retune:
            res = json.loads(tp.read_text(encoding="utf-8"))
        else:
            space = SPACES[args.space]
            res = tune_c2.tune([str(s) for s in val], str(gt), fd / "tune", str(cfg),
                               list(space), args.objective, args.min_recall,
                               jobs=args.jobs, delta_ms=args.delta_ms, obs_cache=cache,
                               base=base, runs_root=fd / "tune" if base else shared,
                               space=space)  # fmt: skip
            res["meta"] = {"commit": git_commit(), "sessions": [str(s) for s in val],
                           "held_out": str(held), "objective": args.objective}  # fmt: skip
            tp.parent.mkdir(parents=True, exist_ok=True)
            tp.write_text(json.dumps(res, indent=2), encoding="utf-8")
            _write_tuned_md(res, fd / "tune" / "tuned.md", [str(s) for s in val], args.objective)
        tuned_rows.append((held.name, res))
        folds.append(run_baselines.run_all([held], gt, fd / "main", cfg, tuned=res["overrides"],
                                           delta_ms=args.delta_ms, ablations=True,
                                           obs_cache=cache))  # fmt: skip
    pooled = pool_results(folds)
    meta = {
        "commit": git_commit(),
        "sessions": [str(s) for s in sessions],
        "cams": run_baselines.session_cams(sessions),
        "per_cam": run_baselines.per_camera_default(cfg),
        "delta_ms": args.delta_ms,
        "tuned": "per fold (leave-one-session-out)",
    }
    pd = d / "pooled"
    pd.mkdir(parents=True, exist_ok=True)
    (pd / "results.json").write_text(json.dumps({"meta": meta, "results": pooled}, indent=2),
                                     encoding="utf-8")  # fmt: skip
    lines = run_baselines.render_md(meta, pooled)
    lines[0] = f"<!-- generated by eval/run_c2.py --cv @ {git_commit()} -->"
    lines.insert(1, "Leave-one-session-out: every number is on held-out sessions, pooled.")
    (pd / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    keys = sorted({k for _, r in tuned_rows for p in r["searched"] for k in r["searched"][p]})
    tl = ["| Held out | Method | " + " | ".join(k.split(".")[-1] for k in keys) + " |",
          "|---|---|" + "---|" * len(keys)]  # fmt: skip
    for name, r in tuned_rows:
        for p, ov in r["searched"].items():
            tl.append(f"| {name} | {p} | " + " | ".join(
                f"{ov[k]:g}" if k in ov else "" for k in keys) + " |")  # fmt: skip
    (d / "tuned_per_fold.md").write_text("\n".join(tl) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))
    return pd / "results.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else f"_(missing: {path})_\n"


def report(out: Path, split: dict, per_cam: bool = False) -> Path:
    tuned = out / "tune" / "tuned.json"
    base = out / "base.json"
    base_txt = base.read_text(encoding="utf-8").strip() if base.exists() else "none (SH17/CHV fit)"
    lines = [
        f"<!-- generated by eval/run_c2.py @ {git_commit()} -->",
        "# C2 results",
        "",
        f"- val (tuning, calibration): {', '.join(s.name for s in split['val'])}",
        f"- test (reported): {', '.join(s.name for s in split['test'])}",
        f"- base overrides: `{' '.join(base_txt.split())}`",
        f"- scoring: {SCORING[per_cam]}",
        "",
        "## 1. Main table — test, tuned on val (§10.3-1, -2)",
        "",
        _read(out / "main" / "results.md"),
        "## 2. Same, untuned config values — test",
        "",
        _read(out / "main_default" / "results.md"),
        "## 3. Tuned values (val)",
        "",
        _read(out / "tune" / "tuned.md"),
    ]
    if per_cam:
        lines += [
            "## 4. Number of cameras",
            "",
            "Not run: each camera is scored alone (association.mode = none); "
            "see the per-camera table in section 1.",
            "",
        ]
    else:
        lines += [
            "## 4. Number of cameras — test (§10.3-3)",
            "",
            _read(out / "numcams" / "results.md"),
            "![num cams](numcams/num_cams.png)",
            "",
        ]
    lines += [
        "## 5. Sensitivity around the tuned point — test (§10.3-4)",
        "",
        _read(out / "sens" / "results.md"),
        "## 6. Visibility / p calibration on C2 (fit val, report test)",
        "",
        _read(out / "calib" / "report.md"),
        "",
        "## 7. Sanity: tuned values on val",
        "",
        _read(out / "main_val" / "results.md"),
    ]
    if tuned.exists():
        lines += ["", f"Tuned overrides: `{json.loads(_read(tuned))['overrides']}`"]
    path = out / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default=str(DEFAULT_CONFIG_DIR / "c2_split.yaml"))
    ap.add_argument("--out", default="eval/reports/c2")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--objective", choices=["f1", "fa@recall"], default="f1")
    ap.add_argument("--min-recall", type=float, default=0.9)
    ap.add_argument("--no-refit", action="store_true", help="skip calib (keep SH17/CHV fit)")
    ap.add_argument("--delta-ms", type=int, default=2000)
    ap.add_argument("--cv", action="store_true", help="leave-one-session-out over split `cv:`")
    ap.add_argument("--retune", action="store_true", help="--cv: redo folds already tuned")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE",
                    help="--cv: fusion.yaml overrides fixed for every method and fold")
    ap.add_argument("--space", choices=list(SPACES), default="default",
                    help="--cv: tuning space (per_part = P with theta_neg per body part)")
    ap.add_argument("--cv-refit", action="store_true",
                    help="--cv: refit p calibration / v_learned on each fold's val sessions")
    args = ap.parse_args(argv)
    setup_logging()
    out, cfg = Path(args.out), Path(args.config)
    out.mkdir(parents=True, exist_ok=True)
    if args.cv:
        sp = yaml.safe_load(Path(args.split).read_text(encoding="utf-8"))
        sessions = [Path(sp["jsonl_root"]) / s for s in sp["cv"]]
        gt = Path(sp["gt_root"])
        errs = check({"val": sessions[:-1], "test": sessions[-1:], "gt": gt}, cfg)
        for e in errs:
            log.error("check: %s", e)
        if errs:
            raise SystemExit(f"{len(errs)} problem(s); fix them before running the experiments")
        log.info("-> %s", step_cv(sessions, gt, out, cfg, args))
        return
    split = load_split(Path(args.split))
    steps = [s for s in STEPS if s in args.steps and not (s == "calib" and args.no_refit)]
    val, test, gt = split["val"], split["test"], split["gt"]
    sv, st = [str(s) for s in val], [str(s) for s in test]
    cache = out / "obs_cache"  # policy-independent work, shared by every step (see run())

    if "check" in steps:
        errs = check(split, cfg)
        for e in errs:
            log.error("check: %s", e)
        if errs:
            raise SystemExit(f"{len(errs)} problem(s); fix them before running the experiments")
        log.info("check: OK (%d val, %d test sessions)", len(val), len(test))
    base: dict = {}
    if "calib" in steps:
        base = step_calib(split, out, cfg)
    elif (out / "base.json").exists() and not args.no_refit:
        base = json.loads((out / "base.json").read_text(encoding="utf-8"))
    tuned_path = out / "tune" / "tuned.json"
    if "tune" in steps:
        res = tune_c2.tune(sv, str(gt), out / "tune", str(cfg), list(tune_c2.SPACE),
                           args.objective, args.min_recall, jobs=args.jobs,
                           delta_ms=args.delta_ms, base=base, obs_cache=cache)  # fmt: skip
        res["meta"] = {"commit": git_commit(), "sessions": sv, "objective": args.objective}
        tuned_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        _write_tuned_md(res, out / "tune" / "tuned.md", sv, args.objective)
    tuned = (
        json.loads(tuned_path.read_text(encoding="utf-8"))["overrides"]
        if tuned_path.exists()
        else base
    )
    if "main" in steps:
        common = dict(delta_ms=args.delta_ms, ablations=True, single_cam_all=True, obs_cache=cache)
        run_baselines.run_all(test, gt, out / "main", cfg, tuned=tuned, **common)
        run_baselines.run_all(test, gt, out / "main_default", cfg, tuned=base, **common)
        run_baselines.run_all(val, gt, out / "main_val", cfg, tuned=tuned, delta_ms=args.delta_ms,
                              obs_cache=cache)  # fmt: skip
    if "sens" in steps:
        argv_s = ["--sessions", *st, "--gt", str(gt), "--out", str(out / "sens"), "--config",
                  str(cfg), "--jobs", str(args.jobs), "--obs-cache", str(cache)]  # fmt: skip
        if tuned_path.exists():
            argv_s += ["--tuned", str(tuned_path)]
        run_sensitivity.main(argv_s)
    if "numcams" in steps and run_baselines.per_camera_default(cfg):
        log.info("numcams: skipped (per-camera evaluation; see the per-camera table in main)")
    elif "numcams" in steps:
        argv_n = ["--sessions", *st, "--gt", str(gt), "--out", str(out / "numcams"),
                  "--config", str(cfg), "--obs-cache", str(cache)]  # fmt: skip
        if tuned_path.exists():
            argv_n += ["--tuned", str(tuned_path)]
        run_num_cams.main(argv_n)
    if "report" in steps:
        log.info("-> %s", report(out, split, run_baselines.per_camera_default(cfg)))


def _write_tuned_md(res: dict, path: Path, sessions: list[str], objective: str) -> None:
    lines = [
        f"<!-- generated by eval/run_c2.py (tune) @ {git_commit()} -->",
        f"Val sessions: {', '.join(Path(s).name for s in sessions)} · objective {objective}",
        "",
        "| Method | Searched values | Val P | Val R | Val F1 | Val FA/hour |",
        "|---|---|---|---|---|---|",
    ]
    for p, ov in res["searched"].items():
        r = res["val_metrics"][p]
        vals = ", ".join(f"{k}={v:g}" for k, v in ov.items())
        lines.append(
            f"| {p} | {vals} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['fa_per_hour']:.2f} |"
        )
    if res.get("at_edge"):
        edge = ", ".join(f"{p} {k}={v:g}" for p, k, v in res["at_edge"])
        lines += ["", f"**At the grid edge (widen eval/tune_c2.SPACE and rerun):** {edge}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
