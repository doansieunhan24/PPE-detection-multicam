"""SHWD (Safety Helmet Wearing Dataset, VOC2028, MIT) → person-crop YOLO set (D7).

SHWD boxes heads only: `hat` = head wearing a helmet, `person` = head without one. There are no
person boxes, so persons come from the pose model (same as inference); each person crop gets
    helmet ← hat boxes,   head ← hat + person boxes
Only images with at least one `hat` are used (the rest are mostly SCUT-HEAD classroom scenes).
vest / shoes / foot are not annotated → fill them with training/ppe_detector/pseudo_label.py.

    python -m training.ppe_detector.prepare_shwd_person_crops
    python -m training.ppe_detector.pseudo_label --dataset shwd \
        --weights models/sh17chv_crop_y11s.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2

from src.common.config import DEFAULT_CONFIG_DIR, load_config, load_yaml
from src.common.logging import get_logger, setup_logging
from training.ppe_detector.crops import Label, crop_labels, pad_clip, xyxy_to_yolo

log = get_logger(__name__)


def parse_voc(path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    root = ET.parse(path).getroot()
    out = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        box = tuple(float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        out.append((obj.find("name").text.strip(), box))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "training_ppe.yaml"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    setup_logging()

    from src.perception_offline.ultralytics_runner import UltralyticsPerception

    cfg = load_yaml(args.config)
    ds = cfg["datasets"]["shwd"]
    classes: list[str] = cfg["classes"]
    helmet, head = classes.index("helmet"), classes.index("head")
    pcfg = load_config("perception")
    pp = pcfg["ppe"]
    pcfg["ppe"] = {**pp, "enabled": False}
    perception = UltralyticsPerception(pcfg)
    root = Path(ds["root"])
    out = Path(ds["out_crops"])
    if out.exists():
        shutil.rmtree(out)
    report: dict = {"source": str(root), "splits": {}}
    for split, list_file in ds["splits"].items():
        ids = (root / list_file).read_text(encoding="utf-8").split()
        ids = ids[: args.limit] if args.limit else ids
        oi, ol = out / "images" / split, out / "labels" / split
        oi.mkdir(parents=True)
        ol.mkdir(parents=True)
        st: Counter = Counter()
        for k, i in enumerate(ids):
            objs = parse_voc(root / "Annotations" / f"{i}.xml")
            if not any(n == "hat" for n, _ in objs):
                st["skipped_no_hat"] += 1
                continue
            img = cv2.imread(str(root / "JPEGImages" / f"{i}.jpg"))
            if img is None:
                st["unreadable"] += 1
                continue
            h, w = img.shape[:2]
            labels = [Label(0, b) for n, b in objs if n == "hat"]
            labels += [Label(1, b) for n, b in objs if n in ("hat", "person")]
            st["images"] += 1
            for pi, p in enumerate(perception.process(img, track=False)):
                crop = pad_clip(p.bbox, pp["crop_pad"], w, h)
                cw, ch = crop[2] - crop[0], crop[3] - crop[1]
                if cw < pp["min_crop_w_px"] or ch < pp["min_crop_h_px"]:
                    st["persons_too_small"] += 1
                    continue
                kept = crop_labels(
                    crop, labels, {0: helmet, 1: head}, cfg["crops"]["min_keep_ratio"]
                )
                if not any(lb.cls == head for lb in kept):
                    st["persons_without_head_label"] += 1  # head not annotated: skip, labels unsafe
                    continue
                x1, y1, x2, y2 = (int(v) for v in crop)
                patch = img[y1:y2, x1:x2]
                s = cfg["crops"]["max_side_px"] / max(patch.shape[:2])
                if s < 1:
                    patch = cv2.resize(patch, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                name = f"{i}_p{pi:02d}"
                cv2.imwrite(
                    str(oi / f"{name}.jpg"),
                    patch,
                    [cv2.IMWRITE_JPEG_QUALITY, cfg["crops"]["jpeg_quality"]],
                )
                lines = [
                    f"{lb.cls} " + " ".join(f"{v:.6f}" for v in xyxy_to_yolo(lb.box, cw, ch))
                    for lb in kept
                ]
                (ol / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                st["crops"] += 1
                st.update(classes[lb.cls] for lb in kept)
            if (k + 1) % 500 == 0:
                log.info("%s %d/%d %s", split, k + 1, len(ids), dict(st))
        report["splits"][split] = dict(st)
        log.info("%s: %s", split, dict(st))
    import yaml

    names = dict(enumerate(classes))
    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
