"""Parameter sensitivity: FA/hour vs recall curves (CLAUDE.md §10.3-4).

One-at-a-time sweeps of fusion.yaml keys (others at their config value) for several policies
on labelled sessions. Same detector, tracker and alert stage for every policy (§14.6).

    python -m eval.run_sensitivity --sessions data/jsonl/s01 --gt data/annotations \
        --out eval/reports/sensitivity --jobs 4
    python -m eval.run_sensitivity ... --param evidence.theta_obs 0.25 0.5 1 2 \
        --param alert.T_alert_s 1 3 5 --policies P B2

Writes <out>/results.json, results.md and one figure per parameter (<key>.png).
Tune on val sessions only; report the chosen point on test sessions.
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

DEFAULT_SWEEPS: dict[str, list[float]] = {
    "evidence.theta_obs": [0.25, 0.5, 1.0, 2.0, 4.0],
    "evidence.theta_neg": [-2.0, -1.0, -0.5, -0.25],
    "evidence.tau_s": [0.5, 1.0, 2.0, 4.0],
    "alert.T_alert_s": [1.0, 2.0, 3.0, 5.0, 8.0],
}
# keys a policy reads; other sweeps would just repeat its default row
P_ONLY = ("evidence.theta_obs", "evidence.theta_neg", "evidence.theta_pos")
# fixed categorical order (dataviz reference palette, light): P first, baselines after
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def _one(job: tuple) -> tuple[str, float, str, dict]:
    key, value, policy, sessions, gt, out, config, delta_ms = job
    rows = []
    for s in map(Path, sessions):
        run_dir = Path(out) / "runs" / s.name / f"{key}={value}" / policy
        run(s, run_dir, Path(config), policy, {}, fusion_overrides={key: value})
        rows.append(evaluate(run_dir, Path(gt) / s.name, delta_ms))
    return key, value, policy, _pool(rows)


def plot(key: str, rows: list[dict], default: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e4e3dd"
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=150)
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    for i, pol in enumerate(policies):
        pts = sorted((r for r in rows if r["policy"] == pol), key=lambda r: r["value"])
        x = [r["recall"] for r in pts]
        y = [r["fa_per_hour"] for r in pts]
        color = COLORS[i % len(COLORS)]
        ax.plot(x, y, "-o", color=color, lw=2, ms=6, label=pol, mec="white", mew=1.5)
        for r in pts:  # label the swept values once, on the proposed method only
            if i == 0:
                bold = "bold" if r["value"] == default else "normal"
                ax.annotate(
                    f"{r['value']:g}",
                    (r["recall"], r["fa_per_hour"]),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7,
                    color=muted,
                    fontweight=bold,
                )
    ax.set_xlabel("Event recall", color=ink, fontsize=9)
    ax.set_ylabel("False alarms / hour", color=ink, fontsize=9)
    ax.set_title(f"Sensitivity: {key} (default {default:g})", color=ink, fontsize=10, loc="left")
    ax.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--delta-ms", type=int, default=2000)
    ap.add_argument("--policies", nargs="+", default=["P", "B1", "B2", "B4"])
    ap.add_argument(
        "--param",
        nargs="+",
        action="append",
        metavar=("KEY", "VALUE"),
        help="fusion.yaml dotted key and values (repeatable); default: DEFAULT_SWEEPS",
    )
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    setup_logging()

    sweeps = {p[0]: [float(v) for v in p[1:]] for p in args.param} if args.param else DEFAULT_SWEEPS
    fusion_cfg = load_config("fusion", Path(args.config))
    defaults = {}
    for key in sweeps:
        apply_overrides(fusion_cfg, {key: 0})  # fail fast on a typo
        node = fusion_cfg
        for k in key.split("."):
            node = node[k]
        defaults[key] = node

    jobs = [
        (key, v, pol, args.sessions, args.gt, args.out, args.config, args.delta_ms)
        for key, values in sweeps.items()
        for pol in args.policies
        if pol == "P" or key not in P_ONLY
        for v in values
    ]
    log.info("%d runs x %d sessions", len(jobs), len(args.sessions))
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            done = list(ex.map(_one, jobs))
    else:
        done = [_one(j) for j in jobs]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"param": k, "value": v, "policy": p, **r} for k, v, p, r in done]
    meta = {
        "commit": git_commit(),
        "sessions": args.sessions,
        "delta_ms": args.delta_ms,
        "defaults": defaults,
    }
    (out / "results.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8"
    )
    lines = [f"<!-- generated by eval/run_sensitivity.py @ {meta['commit']} -->"]
    for key in sweeps:
        lines += [
            "",
            f"### {key} (default {defaults[key]:g})",
            "",
            "| Policy | Value | Precision | Recall | F1 | FA/hour | %UNKNOWN |",
            "|---|---|---|---|---|---|---|",
        ]
        sel = [r for r in rows if r["param"] == key]
        for r in sel:
            lines.append(
                f"| {r['policy']} | {r['value']:g} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['f1']:.3f} | {r['fa_per_hour']:.2f} | {100 * r['unknown_frac']:.1f} |"
            )
        plot(key, sel, defaults[key], out / f"{key}.png")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("-> %s", out / "results.md")


if __name__ == "__main__":
    main()
