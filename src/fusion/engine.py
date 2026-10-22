"""PPE analytics for one synced multi-camera instant (CLAUDE.md §4 analytics block).

frames → foot point → floor (if calibrated) → association → global_id
       → part ROI → p, v per camera → policy (P or B0–B4) → state → PPE alert rules
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from src.alerts.ppe_rules import PPEAlertParams, PPEAlertTracker
from src.association.associator import Associator
from src.association.ground_plane import PersonKey
from src.common.schemas import (
    AlertEvent,
    CamEvidence,
    PartName,
    PartState,
    PerceptionFrame,
    PersonDet,
    PersonPartState,
)
from src.fusion.policies import CamObs, Policy, make_policy
from src.fusion.visibility import VisibilityParams, visibility
from src.geometry.homography import CameraCalibration, point_in_polygon
from src.parts.part_rois import RoiParams, part_roi
from src.parts.ppe_assign import AssignParams, exclusive_owners, observe_part


@dataclass
class Zone:
    id: str
    type: str
    polygon_m: list[tuple[float, float]]
    required_ppe: list[str]


@dataclass
class PPEEngine:
    fusion_cfg: dict
    zones_cfg: dict
    calibs: Mapping[str, CameraCalibration]
    policy_id: str = "P"
    ablation: dict = field(default_factory=dict)
    alert_levels: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        f = self.fusion_cfg
        self.roi_prm = RoiParams.from_config(f)
        self.assign_prm = AssignParams.from_config(f)
        self.vis_prm = VisibilityParams.from_config(f, learned=self.ablation.get("v_learned"))
        # v_learned only changes visibility; the policy must not receive it
        self.ablation = {k: v for k, v in self.ablation.items() if k != "v_learned"}
        self.parts = {PartName(k): v for k, v in f["parts"].items()}
        self.assoc = Associator.from_config(self.zones_cfg, self.calibs)
        self.zones = [
            Zone(z["id"], z["type"], [tuple(p) for p in z["polygon_m"]], z.get("required_ppe", []))
            for z in self.zones_cfg.get("zones", [])
        ]
        self.default_required = self.zones_cfg.get("default_required_ppe", [])
        self.policies: dict[tuple[int, PartName], Policy] = {}
        persistence = self.policy_id != "B0"
        self.alerts = PPEAlertTracker(
            PPEAlertParams.from_config(f, persistence=persistence), levels=self.alert_levels
        )

    # --- helpers ------------------------------------------------------------------------

    def _zone_of(self, xy: tuple[float, float]) -> Zone | None:
        if math.isnan(xy[0]):
            return None
        return next((z for z in self.zones if point_in_polygon(xy, z.polygon_m)), None)

    def _policy(self, gid: int, part: PartName) -> Policy:
        key = (gid, part)
        if key not in self.policies:
            self.policies[key] = make_policy(self.policy_id, self.fusion_cfg, **self.ablation)
        return self.policies[key]

    # --- main ---------------------------------------------------------------------------

    def _floor_xy(self, cam: str, p: PersonDet) -> tuple[float, float]:
        return self.assoc.floor_xy(cam, p)

    def associate(self, group: Mapping[str, PerceptionFrame], ts_ms: int) -> dict[PersonKey, int]:
        return self.assoc.step(group, ts_ms)

    def step(
        self, group: Mapping[str, PerceptionFrame], ts_ms: int
    ) -> tuple[list[PersonPartState], list[AlertEvent], dict[PersonKey, int]]:
        gid_of = self.associate(group, ts_ms)
        members: dict[int, list[tuple[str, PersonDet, PerceptionFrame]]] = {}
        for cam, fr in group.items():
            for p in fr.persons:
                members.setdefault(gid_of[(cam, p.track_id)], []).append((cam, p, fr))

        # exclusive PPE ownership per camera frame and part (crowds), computed once per step
        allowed: dict[tuple[str, int, PartName], set[int]] = {}
        if self.assign_prm.exclusive:
            for cam, fr in group.items():
                for part, spec in self.parts.items():
                    rois = [part_roi(part, q.bbox, q.kpts, self.roi_prm) for q in fr.persons]
                    owners = exclusive_owners(
                        fr.persons, rois, spec["det_cls"], self.assign_prm.containment_min
                    )
                    for q, a in zip(fr.persons, owners, strict=True):
                        allowed[(cam, q.track_id, part)] = a

        states, alerts = [], []
        for gid, mem in members.items():
            xys = [xy for cam, p, _ in mem if not math.isnan((xy := self._floor_xy(cam, p))[0])]
            floor = tuple(np.mean(xys, axis=0).tolist()) if xys else None
            zone = self._zone_of(floor) if floor else None
            required = zone.required_ppe if zone else self.default_required
            parts: dict[PartName, PartState] = {}
            for part, spec in self.parts.items():
                obs, per_cam = [], {}
                for cam, p, fr in mem:
                    roi = part_roi(part, p.bbox, p.kpts, self.roi_prm)
                    ob = observe_part(
                        p,
                        roi,
                        spec["det_cls"],
                        self.assign_prm,
                        allowed.get((cam, p.track_id, part)),
                    )
                    if ob.p is None:
                        continue
                    others = [q.bbox for q in fr.persons if q.track_id != p.track_id]
                    v = visibility(roi, p.bbox, others, fr.img_w, fr.img_h, self.vis_prm).v
                    obs.append(CamObs(cam, ob.p, v, p.conf, detected=ob.matched is not None))
                    per_cam[cam] = CamEvidence(p=ob.p, v=v)
                pol = self._policy(gid, part)
                st = pol.update(ts_ms, obs)
                ev = getattr(pol, "evidence", None)
                parts[part] = PartState(
                    item=spec["item"],
                    state=st,
                    score=ev.mean if ev else 0.0,
                    obs=ev.omega if ev else 0.0,
                    per_cam=per_cam,
                )
                a = self.alerts.update(
                    gid,
                    spec["item"],
                    st,
                    ts_ms,
                    required=spec["item"] in required,
                    zone=zone.id if zone else None,
                    cams=sorted({c for c, _, _ in mem}),
                )
                if a:
                    alerts.append(a)
            states.append(
                PersonPartState(
                    global_id=gid,
                    ts_ms=ts_ms,
                    floor_xy=floor,
                    parts=parts,
                    zone=zone.id if zone else None,
                )
            )
        return states, alerts, gid_of
