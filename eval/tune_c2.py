"""Tune thresholds on val sessions, per method, with the same search for every method (§14.6).

Coordinate descent over SPACE[policy] (dotted fusion.yaml keys): start from the config values,
sweep one key at a time with the others fixed at their current best, keep the best value,
repeat for --rounds. Objective on the pooled PPE events of the val sessions:

  f1          maximise F1; ties → lower FA/hour, then the value closest to the default
  fa@recall   minimise FA/hour subject to recall ≥ --min-recall (else maximise recall)

    python -m eval.tune_c2 --sessions data/jsonl/s01 --gt data/annotations \
        --out eval/reports/c2/tune --jobs 4
    python -m eval.run_baselines --sessions data/jsonl/s02 data/jsonl/s03 --gt data/annotations \
        --out eval/reports/c2/main --tuned eval/reports/c2/tune/tuned.json --ablations

Writes <out>/tuned.json ({"overrides": {policy: {key: value}}, "trace": [...]}) and tuned.md;
"overrides" = --base merged with the searched values (what run_baselines --tuned applies).
P ablation variants reuse P's values. Never tune on the sessions you report.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from eval.run_baselines import _pool, evaluate
from src.analytics_runner import git_commit, run
from src.common.config import DEFAULT_CONFIG_DIR, apply_overrides, load_config
from src.common.logging import get_logger, setup_logging

log = get_logger(__name__)

T_ALERT = [1.0, 2.0, 3.0, 5.0, 8.0]
POS_THR = [0.1, 0.25, 0.4, 0.6]
SPACE: dict[str, dict[str, list[float]]] = {
    "P": {
        "evidence.theta_obs": [0.25, 0.5, 1.0, 2.0, 4.0],
        "evidence.theta_neg": [-2.0, -1.0, -0.5, -0.25],
        "evidence.theta_pos": [0.25, 0.5, 1.0, 2.0],
        "evidence.tau_s": [0.5, 1.0, 2.0, 4.0],
        "alert.T_alert_s": T_ALERT,
    },
    "B0": {"baselines.pos_thr": POS_THR},  # per-frame: no persistence to tune
    "B1": {"baselines.pos_thr": POS_THR, "alert.T_alert_s": T_ALERT},
    "B2": {"baselines.pos_thr": POS_THR, "alert.T_alert_s": T_ALERT},
    "B3": {"baselines.pos_thr": POS_THR, "alert.T_alert_s": T_ALERT},
    "B4": {"alert.T_alert_s": T_ALERT},  # B4 reads calibrated p, not pos_thr
}


def _get(cfg: dict, key: str):
    node = cfg
    for k in key.split("."):
        node = node[k]
    return node


def score(r: dict, objective: str, min_recall: float) -> tuple:
    """Larger is better."""
    if objective == "f1":
        return (r["f1"], -r["fa_per_hour"])
    if r["recall"] >= min_recall:
        return (1, -r["fa_per_hour"], r["recall"])
    return (0, r["recall"], -r["fa_per_hour"])


def at_edge(searched: dict[str, dict], space: dict[str, dict[str, list[float]]] | None = None):
    """(policy, key, value) tuned to the smallest / largest grid value: widen that grid."""
    space = space or SPACE
    return [
        (p, k, v)
        for p, ov in searched.items()
        for k, v in ov.items()
        if k in space.get(p, {})
        and len(space[p][k]) > 1
        and v in (min(space[p][k]), max(space[p][k]))
    ]


def _eval(job: tuple) -> tuple[str, dict, dict]:
    policy, ov, fixed, sessions, gt, out, config, delta_ms, obs_cache = job
    tag = ",".join(f"{k}={v:g}" for k, v in sorted(ov.items()))
    rows = []
    for s in map(Path, sessions):
        d = Path(out) / "runs" / s.name / policy / tag
        if not (d / "run.json").exists():  # resume an interrupted search
            run(s, d, Path(config), policy, {}, fusion_overrides={**fixed, **ov},
                obs_cache=obs_cache)  # fmt: skip
        rows.append(evaluate(d, Path(gt) / s.name, delta_ms))
    r = _pool(rows)
    r.pop("per_type", None)
    return policy, ov, r


def tune(
    sessions: list[str],
    gt: str,
    out: Path,
    config: str,
    policies: list[str],
    objective: str = "f1",
    min_recall: float = 0.9,
    rounds: int = 2,
    jobs: int = 1,
    delta_ms: int = 2000,
    space: dict[str, dict[str, list[float]]] | None = None,
    base: dict[str, dict] | None = None,
    obs_cache: Path | None = None,
) -> dict:
    """base: fixed overrides per policy ("*" = all), e.g. a C2-fitted p calibration for P."""
    space = space or SPACE
    base = base or {}
    fusion_cfg = load_config("fusion", Path(config))
    best: dict[str, dict] = {}
    trace: list[dict] = []
    for pol in policies:
        keys = space[pol]
        for k in keys:
            apply_overrides(fusion_cfg, {k: 0})  # fail fast on a typo
        best[pol] = {k: float(_get(fusion_cfg, k)) for k in keys}
    cache: dict[tuple, dict] = {}

    def evaluate_many(cands: list[tuple[str, dict]]) -> None:
        todo = [(p, ov) for p, ov in cands if (p, tuple(sorted(ov.items()))) not in cache]
        jl = [
            (p, ov, {**base.get("*", {}), **base.get(p, {})}, sessions, gt, str(out), config,
             delta_ms, obs_cache)
            for p, ov in todo
        ]  # fmt: skip
        if jobs > 1 and len(jl) > 1:
            done = [_eval(jl[0])]  # builds the observation cache once, before the pool
            with ProcessPoolExecutor(jobs) as ex:
                done += list(ex.map(_eval, jl[1:]))
        else:
            done = [_eval(j) for j in jl]
        for p, ov, r in done:
            cache[(p, tuple(sorted(ov.items())))] = r

    for rnd in range(rounds):
        changed = False
        for key in dict.fromkeys(k for p in policies for k in space[p]):
            cands = []
            for pol in policies:
                if key not in space[pol]:
                    continue
                vals = sorted(set(space[pol][key]) | {best[pol][key]})
                cands += [(pol, {**best[pol], key: v}) for v in vals]
            if not cands:
                continue
            evaluate_many(cands)
            for pol in policies:
                if key not in space[pol]:
                    continue
                default = best[pol][key]
                vals = sorted(set(space[pol][key]) | {default})
                scored = [
                    (score(cache[(pol, tuple(sorted({**best[pol], key: v}.items())))],
                           objective, min_recall), -abs(v - default), v)
                    for v in vals
                ]  # fmt: skip
                v_best = max(scored)[2]
                r = cache[(pol, tuple(sorted({**best[pol], key: v_best}.items())))]
                trace.append({"round": rnd, "policy": pol, "key": key, "value": v_best,
                              "from": default, **{m: r[m] for m in
                              ("precision", "recall", "f1", "fa_per_hour")}})  # fmt: skip
                if v_best != default:
                    best[pol][key] = v_best
                    changed = True
        log.info("round %d: %s", rnd, best)
        if not changed:
            break
    final = {p: cache[(p, tuple(sorted(best[p].items())))] for p in policies}
    merged = {"*": base.get("*", {})} if base.get("*") else {}
    for k, v in base.items():
        if k != "*" and k not in policies:
            merged[k] = v  # e.g. "P-v_learned" entries pass through
    for p in policies:
        merged[p] = {**base.get(p, {}), **best[p]}
    edge = at_edge(best, space)
    for p, k, v in edge:
        log.warning("%s: %s tuned to the grid edge (%g); consider widening SPACE", p, k, v)
    return {"overrides": merged, "searched": best, "val_metrics": final, "trace": trace,
            "at_edge": edge}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sessions", nargs="+", required=True, help="VAL sessions only")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--policies", nargs="+", default=list(SPACE))
    ap.add_argument("--objective", choices=["f1", "fa@recall"], default="f1")
    ap.add_argument("--min-recall", type=float, default=0.9)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--delta-ms", type=int, default=2000)
    ap.add_argument("--base", help="JSON {policy|'*': {key: value}} kept fixed while tuning")
    ap.add_argument("--obs-cache", help="dir caching policy-independent work (default <out>/obs)")
    args = ap.parse_args(argv)
    setup_logging()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(args.base).read_text(encoding="utf-8")) if args.base else None
    res = tune(args.sessions, args.gt, out, args.config, args.policies, args.objective,
               args.min_recall, args.rounds, args.jobs, args.delta_ms, base=base,
               obs_cache=Path(args.obs_cache) if args.obs_cache else out / "obs")  # fmt: skip
    res["meta"] = {"commit": git_commit(), "sessions": args.sessions,
                   "objective": args.objective, "min_recall": args.min_recall}  # fmt: skip
    (out / "tuned.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    lines = [
        f"<!-- generated by eval/tune_c2.py @ {res['meta']['commit']} -->",
        f"Val sessions: {', '.join(Path(s).name for s in args.sessions)} · "
        f"objective {args.objective}",
        "",
        "| Method | Tuned values | Val P | Val R | Val F1 | Val FA/hour |",
        "|---|---|---|---|---|---|",
    ]
    for p, ov in res["overrides"].items():
        if p not in res["val_metrics"]:
            continue
        r = res["val_metrics"][p]
        vals = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in ov.items())  # fmt: skip
        lines.append(
            f"| {p} | {vals} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['fa_per_hour']:.2f} |"
        )
    (out / "tuned.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))


if __name__ == "__main__":
    main()
