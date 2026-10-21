"""Calibrate p_neg and v_learned from the evidence tables; reliability analysis (§7.3, §7.4).

The question C1 depends on: when a person DOES wear the item, does low visibility predict that
the detector misses it (the source of false alarms)? Fit on val, report on test:

- detection rate among worn items (recall) by v bin, and AUC of v for "detected | worn"
- prior-free negative evidence: LR- = P(no box | worn) / P(no box | not worn) per part and per
  v bin; p_neg = LR- / (1 + LR-) is the "no box" probability under a neutral prior. (The raw
  P(worn | no box) is the dataset prevalence in disguise: SH17 has 5 % helmets, a site ~90 %.)
- v_learned = sigmoid(w · [s_kp, s_size, 1 − r_occ, fallback, log h_roi] + b) fitted to
  P(detected | worn) on worn items only

An earlier version scored "evidence correct" over all rows; with 95 % negatives that label is
dominated by prevalence (invisible part → no box → "correct") and learned the opposite of v.

    python -m training.visibility.calibrate --tables data/processed/visibility \
        --out eval/reports/visibility --write-config configs/visibility_learned.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

from src.analytics_runner import git_commit
from src.common.logging import get_logger, setup_logging

log = get_logger(__name__)
FEATURES = ["s_kp", "s_size", "vis_occ", "fallback", "log_h_roi"]
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]


def load(path: Path) -> dict[str, np.ndarray]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out = {k: v.astype(float) for k, v in cols.items() if k not in ("image", "part")}
    out["part"] = cols["part"]
    out["vis_occ"] = 1.0 - out["r_occ"]
    out["log_h_roi"] = np.log(np.clip(out["h_roi_px"], 1, None))
    return out


def subset(d: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in d.items()}


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC by the rank-sum formula (ties get average rank)."""
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for s in np.unique(score):  # average ties
        m = score == s
        ranks[m] = ranks[m].mean()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> tuple[np.ndarray, float]:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd

    def loss(theta):
        w, b = theta[:-1], theta[-1]
        z = Z @ w + b
        return np.mean(np.logaddexp(0, z) - y * z) + l2 * np.sum(w**2)

    res = minimize(loss, np.zeros(Z.shape[1] + 1), method="L-BFGS-B")
    w_std, b_std = res.x[:-1], res.x[-1]
    w = w_std / sd  # back to raw feature scale
    b = b_std - np.sum(w_std * mu / sd)
    return w, float(b)


def fit_p_calibration(d: dict[str, np.ndarray], p_lo: float = 0.01) -> dict[str, float]:
    """Map detector output to P(worn) under a neutral prior.

    logit P = a·logit(conf) + c   if a box was matched
            = d                   if no box
    fitted by class-balanced logistic regression, so the logit is a log-likelihood ratio
    (prior odds 1) and does not inherit the dataset prevalence.
    """
    y = d["gt"]
    det = d["has_det"] == 1
    pc = np.clip(d["p"], p_lo, 1 - p_lo)
    lc = np.log(pc / (1 - pc))
    X = np.stack([np.where(det, lc, 0.0), det.astype(float), (~det).astype(float)], 1)
    pi = y.mean()
    sw = np.where(y == 1, 0.5 / pi, 0.5 / (1 - pi))

    def loss(th):
        z = X @ th
        return np.sum(sw * (np.logaddexp(0, z) - y * z)) / sw.sum() + 1e-4 * np.sum(th**2)

    th = minimize(loss, np.array([1.0, 0.0, 0.0]), method="L-BFGS-B").x
    return {"a": round(float(th[0]), 5), "c": round(float(th[1]), 5), "d": round(float(th[2]), 5)}


def predict(d: dict[str, np.ndarray], w: np.ndarray, b: float) -> np.ndarray:
    X = np.stack([d[f] for f in FEATURES], 1)
    return 1 / (1 + np.exp(-(X @ w + b)))


def by_bins(d: dict[str, np.ndarray], key: str) -> list[dict]:
    """Per v bin: recall among worn, false-positive rate among not worn, log LR-."""
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
        m = (d[key] >= lo) & (d[key] < hi)
        worn, notw = m & (d["gt"] == 1), m & (d["gt"] == 0)
        rec = float(d["has_det"][worn].mean()) if worn.any() else None
        fpr = float(d["has_det"][notw].mean()) if notw.any() else None
        lr = None
        if rec is not None and fpr is not None and rec < 1 and fpr < 1:
            lr = float(np.log((1 - rec) / (1 - fpr)))
        out.append(
            {
                "bin": f"[{lo:.1f},{min(hi, 1):.1f})",
                "n_worn": int(worn.sum()),
                "n_not_worn": int(notw.sum()),
                "recall": rec,
                "fpr": fpr,
                "log_lr_neg": lr,
            }
        )
    return out


def lr_neg(d: dict[str, np.ndarray]) -> float | None:
    worn, notw = d["gt"] == 1, d["gt"] == 0
    if not worn.any() or not notw.any():
        return None
    miss = 1 - d["has_det"][worn].mean()
    tn = 1 - d["has_det"][notw].mean()
    return float(miss / tn) if tn > 0 else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tables", default="data/processed/visibility")
    ap.add_argument("--out", default="eval/reports/visibility")
    ap.add_argument("--write-config", help="write learned params (v_learned, p_neg) YAML here")
    ap.add_argument("--write-p-calibration", help="write detector-output calibration YAML here")
    args = ap.parse_args(argv)
    setup_logging()

    val, test = load(Path(args.tables) / "val.csv"), load(Path(args.tables) / "test.csv")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"commit": git_commit(), "parts": {}}
    learned: dict = {"features": FEATURES, "parts": {}}
    p_calib: dict = {"model": "logit P = a*logit(conf) + c if box else d", "parts": {}}
    md = [f"<!-- generated by training/visibility/calibrate.py @ {res['commit']} -->", ""]

    def f(x, nd=3):
        return "–" if x is None else f"{x:.{nd}f}"

    for part in ("head", "torso", "feet"):
        v_, t_ = subset(val, val["part"] == part), subset(test, test["part"] == part)
        wv, wt = v_["gt"] == 1, t_["gt"] == 1
        w, b = fit_logreg(np.stack([v_[k][wv] for k in FEATURES], 1), v_["has_det"][wv])
        t_["v_learned"] = predict(t_, w, b)
        lr = lr_neg(v_)
        p_neg = lr / (1 + lr) if lr is not None else None
        r = {
            "n_test": int(len(t_["gt"])),
            "n_worn_test": int(wt.sum()),
            "recall_test": float(t_["has_det"][wt].mean()) if wt.any() else None,
            "fpr_test": float(t_["has_det"][~wt].mean()),
            "lr_neg_val": lr,
            "p_neg_neutral_val": p_neg,
            "p_worn_given_no_box_val": float(v_["gt"][v_["has_det"] == 0].mean()),
            "auc_detected_given_worn_test": {
                "v_manual": auc(t_["v"][wt], t_["has_det"][wt]),
                "v_learned": auc(t_["v_learned"][wt], t_["has_det"][wt]),
                "person_conf": auc(t_["person_conf"][wt], t_["has_det"][wt]),
            },
            "bins_test_v": by_bins(t_, "v"),
            "learned": {"w": w.round(5).tolist(), "b": round(b, 5)},
        }
        res["parts"][part] = r
        learned["parts"][part] = r["learned"] | {"p_neg": round(p_neg, 4) if p_neg else None}
        r["p_calibration"] = pc = fit_p_calibration(v_)
        p_calib["parts"][part] = pc
        a = r["auc_detected_given_worn_test"]
        md += [
            f"### {part} — test: {r['n_worn_test']} worn / {r['n_test']}; "
            f"recall {f(r['recall_test'])}, FPR {f(r['fpr_test'])}",
            "",
            f"- AUC of score for *detected | worn*: v manual **{f(a['v_manual'])}**, "
            f"v_learned **{f(a['v_learned'])}**, person conf {f(a['person_conf'])}",
            f"- LR- (val) = {f(lr)} → p_neg (neutral prior) = **{f(p_neg)}**; "
            f"raw P(worn | no box) = {f(r['p_worn_given_no_box_val'])} (prevalence-biased)",
            f"- p calibration (val, neutral prior): logit P = {pc['a']:.3f}·logit(conf) "
            f"{pc['c']:+.3f} if box, {pc['d']:.3f} if none",
            "",
            "| v bin | worn n | recall | not-worn n | FPR | log LR- |",
            "|---|---|---|---|---|---|",
        ]
        md += [
            f"| {x['bin']} | {x['n_worn']} | {f(x['recall'])} | {x['n_not_worn']} | "
            f"{f(x['fpr'])} | {f(x['log_lr_neg'], 2)} |"
            for x in r["bins_test_v"]
        ]
        md.append("")

    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    log.info("\n%s", "\n".join(md))
    if args.write_p_calibration:
        Path(args.write_p_calibration).write_text(
            "# Generated by training/visibility/calibrate.py — fit on SH17 val (neutral prior)\n"
            + yaml.safe_dump(p_calib, sort_keys=False),
            encoding="utf-8",
        )
    if args.write_config:
        Path(args.write_config).write_text(
            "# Generated by training/visibility/calibrate.py — fit on SH17 val\n"
            + yaml.safe_dump(learned, sort_keys=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
