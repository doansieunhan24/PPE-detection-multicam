"""Foot point in the image (CLAUDE.md §7.1)."""

from __future__ import annotations

from dataclasses import dataclass

from src.common.schemas import PersonDet
from src.parts.part_rois import L_ANK, R_ANK


@dataclass(frozen=True)
class FootPoint:
    x_px: float
    y_px: float
    from_ankles: bool
    reliable: bool  # False when the bbox is cut by the image bottom


def foot_point(p: PersonDet, ankle_conf_min: float = 0.5) -> FootPoint:
    if p.kpts and p.kpts[L_ANK][2] > ankle_conf_min and p.kpts[R_ANK][2] > ankle_conf_min:
        la, ra = p.kpts[L_ANK], p.kpts[R_ANK]
        x, y, ank = (la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2, True
    else:
        x, y, ank = (p.bbox[0] + p.bbox[2]) / 2, p.bbox[3], False
    return FootPoint(x, y, from_ankles=ank, reliable=not p.truncated_bottom)
