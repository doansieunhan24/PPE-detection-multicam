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


def foot_point(p: PersonDet, ankle_conf_min: float = 0.5, source: str = "ankles") -> FootPoint:
    """source="ankles": ankle midpoint when both are confident, else bbox bottom-centre (§7.1).
    source="bbox_bottom": always bbox bottom-centre. Ankles sit ~8–10 cm above the floor, and the
    floor homography pushes a raised point away along the viewing ray (≈ h·dist/cam_height);
    on WILDTRACK this tripled the median floor error (0.68 m vs 0.23 m, docs/experiments.md)."""
    if source not in ("ankles", "bbox_bottom"):
        raise ValueError(f"unknown foot point source: {source}")
    if (
        source == "ankles"
        and p.kpts
        and p.kpts[L_ANK][2] > ankle_conf_min
        and p.kpts[R_ANK][2] > ankle_conf_min
    ):
        la, ra = p.kpts[L_ANK], p.kpts[R_ANK]
        x, y, ank = (la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2, True
    else:
        x, y, ank = (p.bbox[0] + p.bbox[2]) / 2, p.bbox[3], False
    return FootPoint(x, y, from_ankles=ank, reliable=not p.truncated_bottom)
