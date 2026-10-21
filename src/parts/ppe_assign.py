"""Assign PPE detections to body parts → p (CLAUDE.md §7.3)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.common.schemas import BBox, Keypoint, PersonDet, PPEDet
from src.parts.part_rois import WRISTS, PartRoi


@dataclass(frozen=True)
class AssignParams:
    containment_min: float = 0.5
    p_neg: float = 0.05
    held_helmet_wrist_dist: float = 0.15  # x person bbox height
    wrist_conf_min: float = 0.3
    # per part {"a", "c", "d"}: logit p = a*logit(conf) + c if box else d (neutral prior);
    # None → p = raw detector conf, p_neg when no box
    calibration: Mapping[str, Mapping[str, float]] | None = None
    exclusive: bool = False  # one PPE box → one person (exclusive_owners)

    @classmethod
    def from_config(cls, cfg: dict) -> AssignParams:
        a = cfg["part_assign"]
        path = a.get("p_calibration")
        return cls(
            containment_min=a["containment_min"],
            p_neg=a["p_neg"],
            held_helmet_wrist_dist=a["held_helmet_wrist_dist"],
            calibration=load_p_calibration(path) if path else None,
            exclusive=a.get("exclusive", False),
        )


def load_p_calibration(path: str | Path) -> dict[str, dict[str, float]]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["parts"]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def calibrated_p(conf: float | None, cal: Mapping[str, float], lo: float = 0.01) -> float:
    if conf is None:
        return _sigmoid(cal["d"])
    c = min(max(conf, lo), 1 - lo)
    return _sigmoid(cal["a"] * math.log(c / (1 - c)) + cal["c"])


@dataclass(frozen=True)
class PartObservation:
    p: float | None  # None: PPE detector not run → no evidence from this camera
    matched: PPEDet | None = None
    held_helmet: bool = False


def containment(inner: BBox, outer: BBox) -> float:
    """Intersection area / area of `inner`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / a if a > 0 else 0.0


def _near_wrist(box: BBox, kpts: Sequence[Keypoint], max_dist: float, conf_min: float) -> bool:
    if not kpts:
        return False
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(
        kpts[w][2] >= conf_min and math.hypot(kpts[w][0] - cx, kpts[w][1] - cy) < max_dist
        for w in WRISTS
    )


def _iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def exclusive_owners(
    persons: Sequence[PersonDet],
    rois: Sequence[PartRoi],
    item_cls: str,
    containment_min: float,
    same_object_iou: float = 0.5,
) -> list[set[int]]:
    """One physical PPE box belongs to one person (crowds: a neighbour's helmet must not count).

    Crop detections of different persons that overlap (IoU >= same_object_iou) are the same
    object; it is given to the person whose part ROI contains it most (ties: nearest ROI
    centre). Returns, per person, the indices into person.ppe_dets they may use.
    """
    cands = []  # (person idx, det idx, det, containment, distance to roi centre)
    for i, (p, roi) in enumerate(zip(persons, rois, strict=True)):
        rcx, rcy = (roi.box[0] + roi.box[2]) / 2, (roi.box[1] + roi.box[3]) / 2
        rs = max(roi.box[2] - roi.box[0], roi.box[3] - roi.box[1], 1.0)
        for j, d in enumerate(p.ppe_dets):
            if d.cls != item_cls:
                continue
            c = containment(d.bbox, roi.box)
            if c >= containment_min:
                dx = (d.bbox[0] + d.bbox[2]) / 2 - rcx
                dy = (d.bbox[1] + d.bbox[3]) / 2 - rcy
                cands.append((i, j, d, c, math.hypot(dx, dy) / rs))
    allowed: list[set[int]] = [set() for _ in persons]
    taken = [False] * len(cands)
    for k in sorted(range(len(cands)), key=lambda k: (-cands[k][3], cands[k][4])):
        if taken[k]:
            continue
        i, j, d, _, _ = cands[k]
        allowed[i].add(j)
        for m, other in enumerate(cands):  # same object seen from other persons' crops
            if not taken[m] and other[0] != i and _iou(other[2].bbox, d.bbox) >= same_object_iou:
                taken[m] = True
        taken[k] = True
    return allowed


def observe_part(
    person: PersonDet,
    roi: PartRoi,
    item_cls: str,
    prm: AssignParams = AssignParams(),
    allowed: set[int] | None = None,
) -> PartObservation:
    """p for one part in one camera: best matching PPE box, else p_neg.
    allowed: restrict to these indices of person.ppe_dets (see exclusive_owners)."""
    if not person.ppe_checked:
        return PartObservation(p=None)
    ph = person.bbox[3] - person.bbox[1]
    best: PPEDet | None = None
    held = False
    for j, d in enumerate(person.ppe_dets):
        if d.cls != item_cls:
            continue
        if allowed is not None and j not in allowed:
            continue
        inside = containment(d.bbox, roi.box) >= prm.containment_min
        if (
            item_cls == "helmet"
            and not inside
            and _near_wrist(
                d.bbox, person.kpts, prm.held_helmet_wrist_dist * ph, prm.wrist_conf_min
            )
        ):
            held = True
            continue
        if inside and (best is None or d.conf > best.conf):
            best = d
    if prm.calibration is not None:
        p = calibrated_p(best.conf if best else None, prm.calibration[roi.part.value])
    else:
        p = best.conf if best else prm.p_neg
    return PartObservation(p=p, matched=best, held_helmet=held)
