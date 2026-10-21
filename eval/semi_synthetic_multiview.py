"""Semi-synthetic multi-view benchmark from real single-view images (pre-C2 sanity check).

Each test image becomes several synchronized "cameras" by degrading the real photo:
    orig  – the image as is
    occ   – head of every person covered by an occluder (helmet not observable, person is)
    far   – downscaled ×4 and back (person far from the camera)
    blur  – strong Gaussian blur
Identity across views is known (GT person boxes; all views share pixel coordinates), so the
benchmark isolates the *decision rule* (B0–B4 vs P) from association. The scene is static: the
same observations are fed for `--seconds` at 10 Hz, then PPE alert rules apply as in the system.

Per person × item (helmet, vest) with GT worn label:
    worn     → an alert is a FALSE ALARM
    not worn → an alert is a correct DETECTION
    UNKNOWN  → abstention (reported as coverage)

THIS IS NOT A SUBSTITUTE FOR C2: occlusion is synthetic and GT is "visible PPE" in the original
photo. Use it as a pre-check of the fusion logic on real detector outputs.

    python -m eval.semi_synthetic_multiview --dataset chv --split test \
        --setups mixed=orig,occ,far degraded=occ,far,blur
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.alerts.ppe_rules import PPEAlertParams, PPEAlertTracker
from src.analytics_runner import git_commit
from src.common.config import DEFAULT_CONFIG_DIR, load_config, load_yaml
from src.common.logging import get_logger, setup_logging
from src.common.schemas import ComplianceState, PartName
from src.fusion.policies import CamObs, make_policy
from src.fusion.visibility import VisibilityParams, visibility
from src.parts.part_rois import RoiParams, part_roi
from src.parts.ppe_assign import AssignParams, observe_part
from training.ppe_detector.crops import parse_yolo_labels
from training.ppe_detector.pseudo_label import iou
from training.visibility.build_v_dataset import PARTS, gt_wears

log = get_logger(__name__)
ITEMS = {PartName.HEAD: "helmet", PartName.TORSO: "vest"}
VARIANTS = {  # name: (policy id, ablation); B0/B1 are expanded per camera of each setup
    "B2": ("B2", {}),
    "B3": ("B3", {}),
    "B4": ("B4", {}),
    "P": ("P", {}),
    "P-no_v": ("P", {"use_visibility": False}),
    "P-no_unknown": ("P", {"allow_unknown": False}),
}
SWEEP = {
    "theta_obs": [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    "theta_neg": [-0.1, -0.25, -0.5, -1.0, -1.5],
}


def degrade(img: np.ndarray, kind: str, persons: list[tuple], rng: np.random.Generator):
    if kind == "orig":
        return img
    if kind == "far":
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=6)
    if kind == "occ":
        out = img.copy()
        for x1, y1, x2, y2 in persons:
            ph = y2 - y1
            # occluder over the head only (top ~22 % of the person, slightly above and wider):
            # the person stays detectable, the helmet is not observable (back turned / blocked)
            pad = 0.1 * (x2 - x1)
            a, b = int(max(0, x1 - pad)), int(min(img.shape[1], x2 + pad))
            c, d = int(max(0, y1 - 0.1 * ph)), int(y1 + 0.22 * ph)
            color = rng.integers(40, 200, size=3).tolist()
            cv2.rectangle(out, (a, c), (b, d), color, thickness=-1)
        return out
    raise ValueError(kind)


Key = tuple  # (source, image stem, gt person index, part value)


def collect(source: str, kinds: list[str], limit: int | None, seed: int, cache: Path):
    """Run perception on every degraded view once; cache {key: {kind: obs}} and {key: worn}."""
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        if set(kinds) <= set(d["kinds"]):
            obs = {
                tuple(k.split("|")): {kd: CamObs(**o) for kd, o in v.items()}
                for k, v in d["obs"].items()
            }
            return obs, {tuple(k.split("|")): w for k, w in d["labels"].items()}
    from src.perception_offline.ultralytics_runner import UltralyticsPerception

    dname, split = source.split(":")
    tcfg = load_yaml(DEFAULT_CONFIG_DIR / "training_ppe.yaml")
    ds = tcfg["datasets"][dname]
    by_name = {v: int(k) for k, v in ds["names"].items()}
    item_src = {
        it: {by_name[s] for s, d in ds["map"].items() if d == it}
        for it in ITEMS.values()
        if it not in ds.get("missing_classes", [])
    }
    fusion = load_config("fusion")
    roi_prm, asg_prm = RoiParams.from_config(fusion), AssignParams.from_config(fusion)
    vis_prm = VisibilityParams.from_config(fusion)
    perception = UltralyticsPerception(load_config("perception"))
    rng = np.random.default_rng(seed)
    root = Path(ds["root"])
    full_dir = Path(ds["out_full"]) / "images" / split
    if full_dir.exists():
        images = sorted(full_dir.glob("*.jpg"))
    else:
        lines = (root / ds["splits"][split]).read_text(encoding="utf-8").splitlines()
        stems = [Path(x.strip()).stem for x in lines if x.strip()]
        images = [q for st in stems for q in (root / ds["images"]).glob(f"{st}.*")]
    images = images[:limit] if limit else images

    observations: dict[Key, dict[str, CamObs]] = {}
    labels: dict[Key, bool] = {}
    for n, path in enumerate(images):
        img = cv2.imread(str(path))
        ih, iw = img.shape[:2]
        lines = (root / ds["labels"] / f"{path.stem}.txt").read_text(encoding="utf-8").splitlines()
        gt = parse_yolo_labels(lines, iw, ih)
        gpers = [g.box for g in gt if g.cls == by_name[ds["person_class"]]]
        if not gpers:
            continue
        for gi, gp in enumerate(gpers):
            for part, item in ITEMS.items():
                if item in item_src:
                    band = dict(PARTS.values())[item]
                    items = [g.box for g in gt if g.cls in item_src[item]]
                    labels[(source, path.stem, str(gi), part.value)] = gt_wears(gp, items, band)
        for kind in kinds:
            view = degrade(img, kind, gpers, rng)
            persons = perception.process(view, track=False)
            if not persons:
                continue
            m = iou(np.array([p.bbox for p in persons]), np.array(gpers))
            for r, c in zip(*linear_sum_assignment(-m), strict=True):
                if m[r, c] < 0.5:
                    continue
                p = persons[r]
                others = [q.bbox for i, q in enumerate(persons) if i != r]
                for part, item in ITEMS.items():
                    key = (source, path.stem, str(c), part.value)
                    if key not in labels:
                        continue
                    roi = part_roi(part, p.bbox, p.kpts, roi_prm)
                    ob = observe_part(p, roi, item, asg_prm)
                    if ob.p is None:
                        continue
                    v = visibility(roi, p.bbox, others, iw, ih, vis_prm).v
                    observations.setdefault(key, {})[kind] = CamObs(
                        kind, ob.p, v, p.conf, detected=ob.matched is not None
                    )
        if (n + 1) % 100 == 0:
            log.info("%s: %d/%d images", source, n + 1, len(images))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "kinds": kinds,
                "obs": {
                    "|".join(k): {kd: dataclasses.asdict(o) for kd, o in v.items()}
                    for k, v in observations.items()
                },
                "labels": {"|".join(k): w for k, w in labels.items()},
            }
        ),
        encoding="utf-8",
    )
    return observations, labels


def simulate(observations, labels, cams, pid, abl, cam, fusion, part, steps) -> dict:
    """Static scene: feed the same observations for `steps` × 100 ms, apply alert rules."""
    item = ITEMS[part]
    alert_prm = PPEAlertParams.from_config(fusion)
    if pid == "B0":
        alert_prm = dataclasses.replace(alert_prm, T_alert_ms=0)
    c: Counter = Counter()
    for key, worn in labels.items():
        if key[3] != part.value:
            continue
        obs_all = observations.get(key, {})
        obs = [obs_all[k] for k in cams if k in obs_all]
        pol = make_policy(pid, fusion, primary_cam=cam, **abl)
        tracker = PPEAlertTracker(alert_prm)
        alerted, state = False, ComplianceState.UNKNOWN
        for t in range(steps):
            state = pol.update(t * 100, obs)
            a = tracker.update(0, item, state, t * 100, required=True)
            alerted = alerted or (a is not None and state is ComplianceState.VIOLATION)
        c[(worn, alerted, state is ComplianceState.UNKNOWN)] += 1
    nw = sum(v for k, v in c.items() if k[0])
    nn = sum(v for k, v in c.items() if not k[0])
    return {
        "worn_n": nw,
        "not_worn_n": nn,
        "fa_rate": sum(v for k, v in c.items() if k[0] and k[1]) / nw if nw else None,
        "det_rate": sum(v for k, v in c.items() if not k[0] and k[1]) / nn if nn else None,
        "unknown_worn": sum(v for k, v in c.items() if k[0] and k[2]) / nw if nw else None,
    }


def load_sources(sources, kinds, limit, seed, out):
    obs, labels = {}, {}
    for s in sources:
        o, lab = collect(s, kinds, limit, seed, out / f"obs_{s.replace(':', '_')}.json")
        obs.update(o)
        labels.update(lab)
    return obs, labels


def sweep(obs, labels, setups, fusion, steps) -> tuple[dict, list[dict]]:
    """Grid over P's thresholds on val; objective = mean over setups × items of DET − FA."""
    rows = []
    for th_obs, th_neg in itertools.product(SWEEP["theta_obs"], SWEEP["theta_neg"]):
        cfg = copy.deepcopy(fusion)
        cfg["evidence"].update(theta_obs=th_obs, theta_neg=th_neg)
        js = []
        for cams in setups.values():
            for part in ITEMS:
                r = simulate(obs, labels, cams, "P", {}, None, cfg, part, steps)
                if r["det_rate"] is not None and r["fa_rate"] is not None:
                    js.append(r["det_rate"] - r["fa_rate"])
        rows.append({"theta_obs": th_obs, "theta_neg": th_neg, "J": float(np.mean(js))})
    best = max(rows, key=lambda r: r["J"])
    return best, rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--test", nargs="+", default=["chv:test", "sh17:test"])
    ap.add_argument(
        "--val",
        nargs="*",
        default=["chv:val", "sh17:val"],
        help="sources to tune P's theta_obs/theta_neg on (empty: use config)",
    )
    ap.add_argument("--setups", nargs="+", default=["mixed=orig,occ,far", "degraded=occ,far,blur"])
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="eval/reports/semi_synthetic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    setup_logging()

    out = Path(args.out)
    setups = {k: v.split(",") for k, v in (s.split("=", 1) for s in args.setups)}
    kinds = sorted({k for v in setups.values() for k in v})
    steps = int(args.seconds * 10)
    fusion = load_config("fusion")
    results: dict = {
        "commit": git_commit(),
        "test": args.test,
        "val": args.val,
        "seconds": args.seconds,
        "setups": {},
    }

    tuned = copy.deepcopy(fusion)
    if args.val:
        vobs, vlab = load_sources(args.val, kinds, args.limit, args.seed, out)
        best, rows = sweep(vobs, vlab, setups, fusion, steps)
        tuned["evidence"].update(theta_obs=best["theta_obs"], theta_neg=best["theta_neg"])
        results["sweep"] = {"best": best, "grid": rows}
        log.info("val sweep best: %s", best)

    obs, labels = load_sources(args.test, kinds, args.limit, args.seed, out)

    def f(x):
        return "–" if x is None else f"{x:.3f}"

    md = [
        f"<!-- generated by eval/semi_synthetic_multiview.py @ {results['commit']} -->",
        f"Semi-synthetic multi-view on **{', '.join(args.test)}**, static {args.seconds:g} s. "
        "FA = alert on a person wearing the item; DET = alert on a person not wearing it.",
    ]
    if args.val:
        md.append(
            f"P thresholds tuned on {', '.join(args.val)}: theta_obs={best['theta_obs']}, "
            f"theta_neg={best['theta_neg']} (val J={best['J']:.3f}); "
            f"config: theta_obs={fusion['evidence']['theta_obs']}, "
            f"theta_neg={fusion['evidence']['theta_neg']}"
        )
    md.append("")
    for setup, cams in setups.items():
        results["setups"][setup] = {}
        md += [
            f"### setup `{setup}` = {', '.join(cams)}",
            "",
            "| Method | item | worn n | FA rate | not-worn n | DET rate | UNKNOWN (worn) |",
            "|---|---|---|---|---|---|---|",
        ]
        variants = [(f"{b}@{cam}", b, {}, cam, fusion) for b in ("B0", "B1") for cam in cams]
        variants += [(n, pid, abl, None, fusion) for n, (pid, abl) in VARIANTS.items()]
        if args.val:
            variants.append(("P (tuned)", "P", {}, None, tuned))
        for name, pid, abl, cam, cfg in variants:
            for part, item in ITEMS.items():
                r = simulate(obs, labels, cams, pid, abl, cam, cfg, part, steps)
                results["setups"][setup].setdefault(name, {})[item] = r
                md.append(
                    f"| {name} | {item} | {r['worn_n']} | {f(r['fa_rate'])} | "
                    f"{r['not_worn_n']} | {f(r['det_rate'])} | {f(r['unknown_worn'])} |"
                )
        md.append("")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(md))


if __name__ == "__main__":
    main()
