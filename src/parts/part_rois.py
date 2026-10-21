"""Body-part ROIs from COCO keypoints, with bbox-ratio fallback (CLAUDE.md §7.3)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.common.schemas import BBox, Keypoint, PartName

# COCO-17 indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_WRI, R_WRI = 5, 6, 9, 10
L_HIP, R_HIP, L_ANK, R_ANK = 11, 12, 15, 16

PART_KPTS: dict[PartName, tuple[int, ...]] = {
    PartName.HEAD: (NOSE, L_EYE, R_EYE, L_EAR, R_EAR),
    PartName.TORSO: (L_SHO, R_SHO, L_HIP, R_HIP),
    PartName.FEET: (L_ANK, R_ANK),
}
WRISTS = (L_WRI, R_WRI)


@dataclass(frozen=True)
class RoiParams:
    kpt_conf_min: float = 0.3  # keypoint counts as present above this
    min_kpts: dict[PartName, int] | None = None  # min present kpts to use the kpt ROI
    head_expand: float = 1.5  # scale head kpt bbox around its centre
    head_up_ratio: float = 0.6  # extend top upward by this x (expanded head ROI height)
    torso_expand: float = 0.10  # pad torso kpt bbox by this ratio per side
    feet_half_w_ratio: float = 0.08  # half-width around ankles, x person bbox height
    feet_up_ratio: float = 0.03  # above highest ankle, x person bbox height
    feet_down_ratio: float = 0.05  # below lowest ankle, x person bbox height
    # fallback bands as (top, bottom) fraction of person bbox height
    fb_head: tuple[float, float] = (0.0, 0.20)
    fb_torso: tuple[float, float] = (0.25, 0.60)
    fb_feet: tuple[float, float] = (0.88, 1.0)

    def min_kpts_for(self, part: PartName) -> int:
        default = {PartName.HEAD: 2, PartName.TORSO: 3, PartName.FEET: 1}
        return (self.min_kpts or default)[part]

    @classmethod
    def from_config(cls, cfg: dict) -> RoiParams:
        r = dict(cfg.get("rois", {}))
        for k in ("fb_head", "fb_torso", "fb_feet"):
            if k in r:
                r[k] = tuple(r[k])
        if "min_kpts" in r:
            r["min_kpts"] = {PartName(k): v for k, v in r["min_kpts"].items()}
        return cls(**r)


@dataclass(frozen=True)
class PartRoi:
    part: PartName
    box: BBox
    fallback: bool
    s_kp: float  # mean confidence of the part keypoints (0 if no keypoints)


def _kpt_conf_mean(kpts: Sequence[Keypoint], idx: Sequence[int]) -> float:
    if not kpts:
        return 0.0
    return sum(kpts[i][2] for i in idx) / len(idx)


def _band(person: BBox, top: float, bottom: float) -> BBox:
    x1, y1, x2, y2 = person
    h = y2 - y1
    return (x1, y1 + top * h, x2, y1 + bottom * h)


def part_roi(
    part: PartName, person: BBox, kpts: Sequence[Keypoint], prm: RoiParams = RoiParams()
) -> PartRoi:
    idx = PART_KPTS[part]
    s_kp = _kpt_conf_mean(kpts, idx)
    pts = [kpts[i] for i in idx if kpts and kpts[i][2] >= prm.kpt_conf_min]
    ph = person[3] - person[1]

    if len(pts) >= prm.min_kpts_for(part):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if part is PartName.HEAD:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # keypoints of the face span ~ eye-to-ear width; guarantee a minimum size
            half = max(x2 - x1, y2 - y1, 0.05 * ph) * prm.head_expand / 2
            x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
            y1 -= prm.head_up_ratio * (y2 - y1)
        elif part is PartName.TORSO:
            dw, dh = (x2 - x1) * prm.torso_expand, (y2 - y1) * prm.torso_expand
            x1, y1, x2, y2 = x1 - dw, y1 - dh, x2 + dw, y2 + dh
        else:  # FEET
            hw = prm.feet_half_w_ratio * ph
            x1, x2 = x1 - hw, x2 + hw
            y1, y2 = y1 - prm.feet_up_ratio * ph, y2 + prm.feet_down_ratio * ph
        return PartRoi(part, (x1, y1, x2, y2), fallback=False, s_kp=s_kp)

    band = {PartName.HEAD: prm.fb_head, PartName.TORSO: prm.fb_torso, PartName.FEET: prm.fb_feet}
    return PartRoi(part, _band(person, *band[part]), fallback=True, s_kp=s_kp)
