"""Build YOLO datasets (SH17, CHV, ...) for the PPE detector (CLAUDE.md D7, §9, §10.3-5).

--mode crops: one sample per annotated person, cropped with the same padding / min size as
              the inference SGIE (configs/perception.yaml → ppe), PPE labels clipped to crop
--mode full:  full-frame images with the same target classes (baseline for experiment 5)

Splits follow each dataset's own lists (configs/training_ppe.yaml → datasets.<name>.splits),
always by source image; a missing val list is carved (seeded val_ratio) from train.

    python -m training.ppe_detector.prepare_sh17_person_crops --dataset sh17 --mode crops
    python -m training.ppe_detector.prepare_sh17_person_crops --dataset chv --mode crops
    python -m training.ppe_detector.prepare_sh17_person_crops --dataset sh17 --mode full
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from src.common.config import DEFAULT_CONFIG_DIR, load_yaml
from src.common.logging import get_logger, setup_logging
from training.ppe_detector.crops import (
    Label,
    crop_labels,
    pad_clip,
    parse_yolo_labels,
    xyxy_to_yolo,
)

log = get_logger(__name__)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _read_list(path: Path) -> list[str]:
    return [
        Path(ln.strip()).stem for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]


def split_ids(
    root: Path, lists: dict[str, str], all_ids: list[str], val_ratio: float, seed: int
) -> dict[str, list[str]]:
    have = set(all_ids)
    out = {k: sorted(i for i in _read_list(root / f) if i in have) for k, f in lists.items()}
    if "train" not in out or "test" not in out:
        raise SystemExit(f"split lists need train and test, got {sorted(lists)}")
    if "val" not in out:
        train = list(out["train"])
        random.Random(seed).shuffle(train)
        n_val = int(round(len(train) * val_ratio))
        out["train"], out["val"] = sorted(train[n_val:]), sorted(train[:n_val])
    return {k: out[k] for k in ("train", "val", "test")}


def _write_labels(path: Path, labels: list[Label], w: float, h: float) -> None:
    lines = []
    for lb in labels:
        cx, cy, bw, bh = xyxy_to_yolo(lb.box, w, h)
        if bw > 0 and bh > 0:
            lines.append(f"{lb.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _process(job: tuple) -> Counter:
    (img_path, lbl_path, out_img, out_lbl, mode, person_cls, cls_map, crop_cfg, quality) = job
    stats: Counter = Counter()
    img = cv2.imread(str(img_path))
    if img is None:
        stats["unreadable"] += 1
        return stats
    h, w = img.shape[:2]
    lines = lbl_path.read_text(encoding="utf-8").splitlines() if lbl_path.exists() else []
    labels = parse_yolo_labels(lines, w, h)
    stem = img_path.stem
    if mode == "full":
        kept = crop_labels((0.0, 0.0, float(w), float(h)), labels, cls_map, 1.0)
        scale = crop_cfg["full_max_side_px"] / max(h, w)
        full = (
            cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1
            else img
        )
        cv2.imwrite(str(out_img / f"{stem}.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, quality])
        _write_labels(out_lbl / f"{stem}.txt", kept, w, h)
        stats["images"] += 1
        stats.update(f"cls{lb.cls}" for lb in kept)
        return stats
    for k, lb in enumerate(x for x in labels if x.cls == person_cls):
        crop = pad_clip(lb.box, crop_cfg["crop_pad"], w, h)
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        if cw < crop_cfg["min_crop_w_px"] or ch < crop_cfg["min_crop_h_px"]:
            stats["persons_too_small"] += 1
            continue
        kept = crop_labels(crop, labels, cls_map, crop_cfg["min_keep_ratio"])
        x1, y1, x2, y2 = (int(v) for v in crop)
        name = f"{stem}_p{k:02d}"
        patch = img[y1:y2, x1:x2]
        scale = crop_cfg["max_side_px"] / max(patch.shape[:2])
        if scale < 1:  # labels are normalised, so resizing the crop does not change them
            patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_img / f"{name}.jpg"), patch, [cv2.IMWRITE_JPEG_QUALITY, quality])
        _write_labels(out_lbl / f"{name}.txt", kept, cw, ch)
        stats["crops"] += 1
        stats.update(f"cls{lb.cls}" for lb in kept)
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "training_ppe.yaml"))
    ap.add_argument("--perception", default=str(DEFAULT_CONFIG_DIR / "perception.yaml"))
    ap.add_argument("--dataset", default="sh17")
    ap.add_argument("--mode", choices=["crops", "full"], default="crops")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, help="debug: only first N images per split")
    args = ap.parse_args(argv)
    setup_logging()

    cfg = load_yaml(args.config)
    pcfg = load_yaml(args.perception)["ppe"]
    ds = cfg["datasets"][args.dataset]
    root = Path(ds["root"])
    names = {int(k): v for k, v in ds["names"].items()}
    by_name = {v: k for k, v in names.items()}
    our_names: list[str] = cfg["classes"]
    cls_map = {by_name[src]: our_names.index(dst) for src, dst in ds["map"].items()}
    crop_cfg = {
        "crop_pad": pcfg["crop_pad"],
        "min_crop_w_px": pcfg["min_crop_w_px"],
        "min_crop_h_px": pcfg["min_crop_h_px"],
        "min_keep_ratio": cfg["crops"]["min_keep_ratio"],
        "max_side_px": cfg["crops"].get("max_side_px", 640),
        "full_max_side_px": cfg["full_frame"].get("max_side_px", 1280),
    }

    img_dir, lbl_dir = root / ds["images"], root / ds["labels"]
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    splits = split_ids(root, ds["splits"], sorted(imgs), cfg["crops"]["val_ratio"], cfg["seed"])
    out = Path(ds["out_crops"] if args.mode == "crops" else ds["out_full"])
    if out.exists():
        shutil.rmtree(out)

    report = {
        "dataset": args.dataset,
        "mode": args.mode,
        "source": str(root),
        "crop_cfg": crop_cfg,
        "missing_classes": ds.get("missing_classes", []),
        "splits": {},
    }
    for split, ids in splits.items():
        ids = ids[: args.limit] if args.limit else ids
        oi, ol = out / "images" / split, out / "labels" / split
        oi.mkdir(parents=True)
        ol.mkdir(parents=True)
        jobs = [
            (
                imgs[i],
                lbl_dir / f"{i}.txt",
                oi,
                ol,
                args.mode,
                by_name[ds["person_class"]],
                cls_map,
                crop_cfg,
                cfg["crops"]["jpeg_quality"],
            )
            for i in ids
        ]
        total: Counter = Counter()
        with ProcessPoolExecutor(args.workers) as ex:
            for st in ex.map(_process, jobs, chunksize=16):
                total.update(st)
        per_cls = {our_names[int(k[3:])]: v for k, v in total.items() if k.startswith("cls")}
        other = {k: v for k, v in total.items() if not k.startswith("cls")}
        report["splits"][split] = {"source_images": len(ids), **other, "instances": per_cls}
        log.info("%s: %s", split, report["splits"][split])

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(our_names)),
    }
    import yaml

    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("dataset → %s", out / "data.yaml")


if __name__ == "__main__":
    main()
