"""Figures for the visibility analysis (thesis): recall | worn and log LR- by v bin.

python -m eval.plot_visibility --results eval/reports/visibility/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PART_LABEL = {"head": "head / helmet", "torso": "torso / vest", "feet": "feet / shoes"}
COLORS = {"head": "#2a6fdb", "torso": "#e08a1e", "feet": "#3a9d5d"}
MIN_N = 10  # hide bins with too few worn samples


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="eval/reports/visibility/results.json")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = Path(args.results).parent
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for part, r in res["parts"].items():
        bins = r["bins_test_v"]
        x = [i for i, b in enumerate(bins) if b["n_worn"] >= MIN_N and b["recall"] is not None]
        axes[0].plot(
            x,
            [bins[i]["recall"] for i in x],
            "o-",
            color=COLORS[part],
            label=f"{PART_LABEL[part]} (n={r['n_worn_test']})",
        )
        xl = [i for i in x if bins[i]["log_lr_neg"] is not None]
        axes[1].plot(
            xl,
            [bins[i]["log_lr_neg"] for i in xl],
            "o-",
            color=COLORS[part],
            label=PART_LABEL[part],
        )
    labels = [b["bin"] for b in next(iter(res["parts"].values()))["bins_test_v"]]
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel("visibility v")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("detection recall | PPE worn")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_ylabel("log LR of 'no PPE box'")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("negative evidence strength", fontsize=9)
    axes[0].set_title(f"SH17 test, bins with ≥{MIN_N} worn samples", fontsize=9)
    fig.tight_layout()
    path = out / "visibility_reliability.png"
    fig.savefig(path, dpi=150)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
