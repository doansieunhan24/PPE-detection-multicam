"""Per-instant association: foot point → floor → cross-camera clusters → stable global IDs.

Shared by the PPE engine and eval/eval_association.py so both run the same code path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from src.association.global_id import GlobalIdManager
from src.association.ground_plane import FloorObs, PersonKey, cluster
from src.common.schemas import PerceptionFrame, PersonDet
from src.geometry.foot_point import foot_point
from src.geometry.homography import CameraCalibration

NAN_XY = (math.nan, math.nan)


@dataclass
class Associator:
    calibs: Mapping[str, CameraCalibration]
    d_max_m: float = 0.8
    vote_window: int = 15
    fov_margin_m: float = 0.0
    foot_point_source: str = "ankles"
    d_max_truncated_m: float | None = None
    # ground_plane: cross-camera clusters (§7.2) | none: every (cam, track) is its own person
    mode: str = "ground_plane"
    gids: GlobalIdManager = field(init=False)
    _own_gid: dict[PersonKey, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("ground_plane", "none"):
            raise ValueError(f"association.mode must be ground_plane or none, got {self.mode!r}")
        self.gids = GlobalIdManager(window=self.vote_window, reuse_dist_m=self.d_max_m)

    @classmethod
    def from_config(cls, zones_cfg: dict, calibs: Mapping[str, CameraCalibration]) -> Associator:
        a = zones_cfg.get("association", {})
        return cls(
            calibs,
            a.get("d_max_m", 0.8),
            a.get("vote_window_frames", 15),
            a.get("fov_margin_m", 0.0),
            a.get("foot_point", "ankles"),
            a.get("d_max_truncated_m"),
            a.get("mode", "ground_plane"),
        )

    def floor_xy(self, cam: str, p: PersonDet) -> tuple[float, float]:
        cal = self.calibs.get(cam)
        if cal is None or not cal.calibrated:
            return NAN_XY
        fp = foot_point(p, source=self.foot_point_source)
        x, y = cal.to_floor(np.array([[fp.x_px, fp.y_px]]))[0]
        return float(x), float(y)

    def step(self, group: Mapping[str, PerceptionFrame], ts_ms: int) -> dict[PersonKey, int]:
        if self.mode == "none":
            return self._step_per_camera(group)
        located: dict[str, list[FloorObs]] = {}
        singles: list[list[FloorObs]] = []
        for cam, fr in group.items():
            for p in fr.persons:
                xy = self.floor_xy(cam, p)
                o = FloorObs((cam, p.track_id), xy, reliable=not p.truncated_bottom)
                if math.isnan(xy[0]):
                    singles.append([o])
                else:
                    located.setdefault(cam, []).append(o)
        sees = {c: partial(cal.sees, margin_m=self.fov_margin_m) for c, cal in self.calibs.items()}
        clusters = cluster(located, self.d_max_m, sees, self.d_max_truncated_m) if located else []
        return self.gids.update(clusters + singles, ts_ms)

    def _step_per_camera(self, group: Mapping[str, PerceptionFrame]) -> dict[PersonKey, int]:
        result: dict[PersonKey, int] = {}
        for cam, fr in group.items():
            for p in fr.persons:
                key = (cam, p.track_id)
                if key not in self._own_gid:
                    self._own_gid[key] = len(self._own_gid) + 1
                result[key] = self._own_gid[key]
        return result
