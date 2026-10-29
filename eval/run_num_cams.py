"""Number of cameras: 1 / 2 / 3 cams → FA/hour and recall (CLAUDE.md §10.3-3).

Every k-subset of the session cameras is run as if only those cameras existed (association,
fusion and behavior see only their JSONL). Rows per subset, plus the mean over the subsets of
each size k (so k = 1 is the average single camera, not the best one).

    python -m eval.run_num_cams --sessions data/jsonl/s02 data/jsonl/s03 --gt data/annotations \
        --out eval/reports/c2/num_cams --tuned eval/reports/c2/tune/tuned.json

Writes <out>/results.json, results.md and num_cams.png.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from eval.run_baselines import _pool, evaluate, overrides_for, session_cams
from eval.run_sensitivity import COLORS
from src.analytics_runner import git_commit, run
from src.common.config import DEFAULT_CONFIG_DIR
from src.common.logging import get_logger, setup_logging

log = get_logger(__name__)
METRICS = ("precision", "recall", "f1", "fa_per_hour", "unknown_frac")


def run_subsets(
    sessions: list[Path],
    gt: Path,
    out: Path,
    config_dir: Path,
    policies: list[str],
    tuned: dict,
    delta_ms: int = 2000,
    obs_cache: Path | None = None,
) -> dict:
    cams = session_cams(sessions)
    rows = []
    for k in range(1, len(cams) + 1):
        for subset in combinations(cams, k):
            tag = "+".join(subset)
            for pol in policies:
                ov = overrides_for(pol, pol, tuned)
                if pol in ("B0", "B1"):  # single-cam baselines run on the subset's first camera
                    ov = {**ov, "baselines.primary_cam": subset[0]}
                per = []
                for s in sessions:
                    d = out / "runs" / s.name / tag / pol
                    run(s, d, config_dir, pol, {}, fusion_overrides=ov, cams=list(subset),
                        obs_cache=obs_cache)  # fmt: skip
                    per.append(evaluate(d, gt / s.name, delta_ms))
                r = _pool(per)
                rows.append({"k": k, "cams": list(subset), "policy": pol,
                             **{m: r[m] for m in METRICS}, "tp": r["tp"], "fp": r["fp"],
                             "fn": r["fn"]})  # fmt: skip
    by_k = []
    for k in sorted({r["k"] for r in rows}):
        for pol in policies:
            sel = [r for r in rows if r["k"] == k and r["policy"] == pol]
            by_k.append(
                {
                    "k": k,
                    "policy": pol,
                    "n_subsets": len(sel),
                    **{m: sum(r[m] for r in sel) / len(sel) for m in METRICS},
                }  # fmt: skip
            )
    return {"cams": cams, "rows": rows, "by_k": by_k}


def plot(by_k: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e4e3dd"
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=150)
    pols = list(dict.fromkeys(r["policy"] for r in by_k))
    for ax, key, label in ((axes[0], "fa_per_hour", "False alarms / hour"),
                           (axes[1], "recall", "Event recall")):  # fmt: skip
        for i, pol in enumerate(pols):
            pts = [r for r in by_k if r["policy"] == pol]
            ax.plot([r["k"] for r in pts], [r[key] for r in pts], "-o", lw=2, ms=6,
                    color=COLORS[i % len(COLORS)], label=pol, mec="white", mew=1.5)  # fmt: skip
        ax.set_xlabel("Number of cameras", color=ink, fontsize=9)
        ax.set_ylabel(label, color=ink, fontsize=9)
        ax.set_xticks(sorted({r["k"] for r in by_k}))
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.set_ylim(bottom=0)
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=ink)
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
    ap.add_argument("--tuned", help="JSON of val-tuned fusion overrides (eval/tune_c2.py)")
    ap.add_argument("--obs-cache", help="dir caching policy-independent work (faster reruns)")
    args = ap.parse_args(argv)
    setup_logging()
    tuned = json.loads(Path(args.tuned).read_text(encoding="utf-8")) if args.tuned else {}
    tuned = tuned.get("overrides", tuned)
    out = Path(args.out)
    res = run_subsets([Path(s) for s in args.sessions], Path(args.gt), out, Path(args.config),
                      args.policies, tuned, args.delta_ms,
                      Path(args.obs_cache) if args.obs_cache else None)  # fmt: skip
    meta = {"commit": git_commit(), "sessions": args.sessions, "tuned": tuned}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"meta": meta, **res}, indent=2), "utf-8")
    lines = [
        f"<!-- generated by eval/run_num_cams.py @ {meta['commit']} -->",
        "Mean over camera subsets of size k:",
        "",
        "| k | Method | Subsets | Precision | Recall | F1 | FA/hour | %UNKNOWN |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in res["by_k"]:
        lines.append(
            f"| {r['k']} | {r['policy']} | {r['n_subsets']} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {r['fa_per_hour']:.2f} | "
            f"{100 * r['unknown_frac']:.1f} |"
        )
    lines += ["", "Per subset:", "", "| Cams | Method | TP | FP | FN | Recall | FA/hour |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in res["rows"]:
        lines.append(
            f"| {'+'.join(r['cams'])} | {r['policy']} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['recall']:.3f} | {r['fa_per_hour']:.2f} |"
        )
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(res["by_k"], out / "num_cams.png")
    log.info("\n%s", "\n".join(lines))


if __name__ == "__main__":
    main()
