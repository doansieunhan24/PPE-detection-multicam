"""Build YOLO datasets from SH17 for the PPE detector (CLAUDE.md D7, §9, §10.3-5).

--mode crops: one sample per annotated person, cropped with the same padding / min size as
              the inference SGIE (configs/perception.yaml → ppe), PPE labels clipped to crop
--mode full:  full-frame images with the same target classes (baseline for experiment 5)

Splits are made by source image: SH17 val list → test; a seeded val_ratio of SH17 train → val.

    python -m training.ppe_detector.prepare_sh17_person_crops --mode crops
    python -m training.ppe_detector.prepare_sh17_person_crops --mode full
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


def _find_dir(root: Path, name: str) -> Path:
    hits = [p for p in root.rglob(name) if p.is_dir()]
    if not hits:
        raise SystemExit(f"no '{name}/' directory under {root}")
    return min(hits, key=lambda p: len(p.parts))


def _read_list(path: Path) -> list[str]:
    return [
        Path(ln.strip()).stem for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]


def split_ids(root: Path, all_ids: list[str], val_ratio: float, seed: int) -> dict[str, list[str]]:
    train_f = next(iter(root.rglob("train_files.txt")), None)
    test_f = next(iter(root.rglob("val_files.txt")), None) or next(
        iter(root.rglob("test_files.txt")), None
    )
    if train_f is None or test_f is None:
        raise SystemExit("SH17 split lists (train_files.txt / val_files.txt) not found")
    have = set(all_ids)
    train = [i for i in _read_list(train_f) if i in have]
    test = [i for i in _read_list(test_f) if i in have]
    rng = random.Random(seed)
    train = sorted(train)
    rng.shuffle(train)
    n_val = int(round(len(train) * val_ratio))
    return {"train": sorted(train[n_val:]), "val": sorted(train[:n_val]), "test": sorted(test)}


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
        shutil.copy2(img_path, out_img / img_path.name)
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
        cv2.imwrite(
            str(out_img / f"{name}.jpg"), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
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
    ap.add_argument("--mode", choices=["crops", "full"], default="crops")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, help="debug: only first N images per split")
    args = ap.parse_args(argv)
    setup_logging()

    cfg = load_yaml(args.config)
    pcfg = load_yaml(args.perception)["ppe"]
    sh = cfg["sh17"]
    root = Path(sh["root"])
    names = {int(k): v for k, v in sh["names"].items()}
    by_name = {v: k for k, v in names.items()}
    target = list(cfg["target_classes"].items())  # [(sh17_name, our_name)]
    cls_map = {by_name[src]: i for i, (src, _) in enumerate(target)}
    our_names = [dst for _, dst in target]
    crop_cfg = {
        "crop_pad": pcfg["crop_pad"],
        "min_crop_w_px": pcfg["min_crop_w_px"],
        "min_crop_h_px": pcfg["min_crop_h_px"],
        "min_keep_ratio": cfg["crops"]["min_keep_ratio"],
    }

    img_dir, lbl_dir = _find_dir(root, "images"), _find_dir(root, "labels")
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    splits = split_ids(root, sorted(imgs), cfg["crops"]["val_ratio"], cfg["seed"])
    out = Path(cfg["crops"]["out"] if args.mode == "crops" else cfg["full_frame"]["out"])
    if out.exists():
        shutil.rmtree(out)

    report = {"mode": args.mode, "source": str(root), "crop_cfg": crop_cfg, "splits": {}}
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
                by_name[sh["person_class"]],
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
