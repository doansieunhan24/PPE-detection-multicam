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
STEPS = ["check", "calib", "tune", "main", "sens", "numcams", "report"]


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
    cams_ref = None
    for s in split["val"] + split["test"]:
        jl = sorted(p.stem for p in s.glob("*.jsonl"))
        if len(jl) < 2:
            errs.append(f"{s}: {len(jl)} camera JSONL (need ≥ 2)")
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
        if missing:
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


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else f"_(missing: {path})_\n"


def report(out: Path, split: dict) -> Path:
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
        "## 4. Number of cameras — test (§10.3-3)",
        "",
        _read(out / "numcams" / "results.md"),
        "![num cams](numcams/num_cams.png)",
        "",
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
    args = ap.parse_args(argv)
    setup_logging()
    split = load_split(Path(args.split))
    out, cfg = Path(args.out), Path(args.config)
    out.mkdir(parents=True, exist_ok=True)
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
    if "numcams" in steps:
        argv_n = ["--sessions", *st, "--gt", str(gt), "--out", str(out / "numcams"),
                  "--config", str(cfg), "--obs-cache", str(cache)]  # fmt: skip
        if tuned_path.exists():
            argv_n += ["--tuned", str(tuned_path)]
        run_num_cams.main(argv_n)
    if "report" in steps:
        log.info("-> %s", report(out, split))


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
