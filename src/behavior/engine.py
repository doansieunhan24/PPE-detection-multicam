"""Rule-based behavior analytics (CLAUDE.md §7.7): fall, immobility, intrusion, crowd.

Runs after association (needs global_id and floor position). Multi-view fall: per-camera
fall scores are fused by whole-body visibility, the same idea as C1.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from src.alerts.episodes import EpisodeAlerter
from src.association.ground_plane import PersonKey
from src.behavior.pose_features import (
    aspect_ratio,
    body_visibility,
    hip_y,
    mean_kpt_displacement,
    torso_angle_deg,
)
from src.common.schemas import AlertEvent, AlertType, PerceptionFrame, PersonDet
from src.geometry.homography import point_in_polygon


def _ramp(x: float, lo: float, hi: float) -> float:
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


@dataclass
class ZoneDef:
    id: str
    type: str
    polygon_m: list[tuple[float, float]]
    crowd_max: int | None = None

    def contains(self, xy: tuple[float, float] | None) -> bool:
        return xy is not None and not math.isnan(xy[0]) and point_in_polygon(xy, self.polygon_m)


@dataclass
class _TrackHist:
    hips: deque = field(default_factory=lambda: deque(maxlen=200))  # (ts, hip_y, bbox_h)
    last_drop_ms: int | None = None
    prev_kpts: list | None = None
    seq: deque | None = None  # per-frame features (fall.method: seq)
    prev_det: PersonDet | None = None


class BehaviorEngine:
    def __init__(self, cfg: dict, zones_cfg: dict, levels: Mapping[str, int] | None = None):
        self.cfg = cfg
        self.levels = dict(levels or {})
        self.zones = [
            ZoneDef(z["id"], z["type"], [tuple(p) for p in z["polygon_m"]], z.get("crowd_max"))
            for z in zones_cfg.get("zones", [])
        ]
        # without cross-camera matching a person seen by k cameras has k global_ids
        assoc_mode = zones_cfg.get("association", {}).get("mode", "ground_plane")
        self.cross_cam_ids = assoc_mode != "none"
        cd = int(cfg.get("cooldown_s", 60) * 1000)
        self.fall_alerter = EpisodeAlerter(int(cfg["fall"]["hold_s"] * 1000), cd)
        self.immobile_alerter = EpisodeAlerter(int(cfg["immobility"]["T_immobile_s"] * 1000), cd)
        self.intr_alerter = EpisodeAlerter(int(cfg["intrusion"]["T_intr_s"] * 1000), cd)
        self.crowd_alerter = EpisodeAlerter(int(cfg["crowd"]["T_s"] * 1000), cd)
        # per (cam_id, global_id), not (cam_id, track_id): trackers often switch IDs during the
        # fast motion of a fall, which would erase the hip-drop history (UR Fall, s05)
        self.tracks: dict[tuple[str, int], _TrackHist] = {}
        self.last_fall_ms: dict[int, int] = {}
        self.seq_model = None
        if cfg["fall"].get("method", "rule") == "seq":
            self._load_seq_model(cfg["fall"]["seq_model"])

    def _load_seq_model(self, path: str) -> None:
        import torch

        from src.behavior.seq_features import build_model

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.seq_len = int(ckpt["seq_len"])
        self.seq_dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.seq_model = build_model().to(self.seq_dev).eval()
        self.seq_model.load_state_dict(ckpt["state_dict"])

    def _seq_scores(self, keys: list[tuple[str, int]], persons: list[PersonDet]) -> list[float]:
        """Learned per-camera fall probability for each (cam, gid) this instant (batched)."""
        import numpy as np
        import torch

        from src.behavior.seq_features import frame_features, window

        wins = []
        for key, p in zip(keys, persons, strict=True):
            h = self.tracks.setdefault(key, _TrackHist())
            if h.seq is None:
                h.seq = deque(maxlen=self.seq_len)
            h.seq.append(frame_features(p, h.prev_det))
            h.prev_det = p
            wins.append(window(h.seq, self.seq_len))
        if not wins:
            return []
        with torch.no_grad():
            x = torch.from_numpy(np.stack(wins)).to(self.seq_dev)
            return torch.sigmoid(self.seq_model(x)).cpu().tolist()

    # --- per-camera fall score ------------------------------------------------------------

    def pose_score(self, p: PersonDet) -> float:
        f = self.cfg["fall"]
        s_ar = _ramp(aspect_ratio(p.bbox), *f["aspect_ratio"])
        ang = torso_angle_deg(p.kpts, f["kpt_conf_min"])
        return s_ar if ang is None else max(s_ar, _ramp(ang, *f["angle_deg"]))

    def cam_fall_score(self, key: tuple[str, int], p: PersonDet, ts_ms: int) -> float:
        f = self.cfg["fall"]
        h = self.tracks.setdefault(key, _TrackHist())
        hy = hip_y(p.kpts, f["kpt_conf_min"])
        if hy is None:
            hy = (p.bbox[1] + p.bbox[3]) / 2
        h.hips.append((ts_ms, hy, p.bbox[3] - p.bbox[1]))
        win = [x for x in h.hips if ts_ms - x[0] <= f["drop_window_s"] * 1000]
        if win:
            ref_h = max(x[2] for x in win)
            if hy - min(x[1] for x in win) >= f["drop_ratio"] * ref_h:
                h.last_drop_ms = ts_ms
        recent_drop = (
            h.last_drop_ms is not None and ts_ms - h.last_drop_ms <= f["drop_memory_s"] * 1000
        )
        return self.pose_score(p) * (1.0 if recent_drop else f["no_drop_factor"])

    # --- main -----------------------------------------------------------------------------

    def _alert(
        self,
        t: AlertType,
        default_level: int,
        ts_ms: int,
        start: int,
        gid: int | None,
        zone: str | None,
        cams: Sequence[str],
    ) -> AlertEvent:
        return AlertEvent(
            alert_id=uuid.uuid4().hex[:12],
            type=t,
            level=self.levels.get(t.value, default_level),
            global_id=gid,
            zone=zone,
            t_start_ms=start,
            t_raise_ms=ts_ms,
            cams=sorted(cams),
        )

    def step(
        self,
        group: Mapping[str, PerceptionFrame],
        ts_ms: int,
        gid_of: Mapping[PersonKey, int],
        floor_of: Mapping[int, tuple[float, float] | None],
    ) -> list[AlertEvent]:
        f, im = self.cfg["fall"], self.cfg["immobility"]
        alerts: list[AlertEvent] = []
        per_gid: dict[int, list[tuple[str, PersonDet, float, float]]] = {}
        obs = [(cam, p, gid_of[(cam, p.track_id)]) for cam, fr in group.items() for p in fr.persons]
        if self.seq_model is not None:
            scores = self._seq_scores([(c, g) for c, _, g in obs], [p for _, p, _ in obs])
            threshold = f.get("seq_threshold", 0.5)
        else:
            scores = [self.cam_fall_score((c, g), p, ts_ms) for c, p, g in obs]
            threshold = f["score_threshold"]
        for (cam, p, gid), s in zip(obs, scores, strict=True):
            v = body_visibility(p.kpts, p.bbox, f["h_ref_px"])
            per_gid.setdefault(gid, []).append((cam, p, s, v))

        for gid, views in per_gid.items():
            cams = [c for c, *_ in views]
            if f.get("fusion", "vmean") == "max":  # ablation: most confident view decides
                fused = max(s for _, _, s, _ in views)
            else:  # §7.7: visibility-weighted mean
                w = sum(v for *_, v in views)
                fused = sum(s * v for _, _, s, v in views) / w if w > 0 else 0.0
            start = self.fall_alerter.update(gid, fused >= threshold, ts_ms)
            if start is not None:
                self.last_fall_ms[gid] = ts_ms
                alerts.append(self._alert(AlertType.FALL, 3, ts_ms, start, gid, None, cams))

            # immobility: the best-seen camera decides whether the body moved this frame;
            # the alerter's hold time (T_immobile) measures how long it stayed still
            cam, p, _, _ = max(views, key=lambda o: o[3])
            h = self.tracks.setdefault((cam, gid), _TrackHist())
            disp = mean_kpt_displacement(h.prev_kpts or [], p.kpts)
            h.prev_kpts = list(p.kpts)
            bh = max(1.0, p.bbox[3] - p.bbox[1])
            still = disp is not None and disp / bh <= im["disp_eps_ratio"]
            xy = floor_of.get(gid)
            in_risk = any(z.type == "risk" and z.contains(xy) for z in self.zones)
            lying = self.pose_score(p) >= im["lying_score_min"]
            after_fall = ts_ms - self.last_fall_ms.get(gid, -(10**12)) <= im["recent_fall_s"] * 1000
            cond = still and ((lying and in_risk) or after_fall)
            start = self.immobile_alerter.update(gid, cond, ts_ms)
            if start is not None:
                alerts.append(self._alert(AlertType.IMMOBILE, 3, ts_ms, start, gid, None, cams))

            for z in self.zones:
                if z.type != "forbidden":
                    continue
                start = self.intr_alerter.update((gid, z.id), z.contains(xy), ts_ms)
                if start is not None:
                    a = self._alert(AlertType.INTRUSION, 2, ts_ms, start, gid, z.id, cams)
                    alerts.append(a)

        c = self.cfg["crowd"]
        for z in self.zones:
            if z.type not in c["zone_types"]:
                continue
            inside = [g for g in per_gid if z.contains(floor_of.get(g))]
            if self.cross_cam_ids:
                n = len(inside)
            else:  # count in each camera, take the camera that sees the most (no double count)
                per_cam = Counter(c for g in inside for c in {v[0] for v in per_gid[g]})
                n = max(per_cam.values(), default=0)
            k = z.crowd_max if z.crowd_max is not None else c["K"]
            start = self.crowd_alerter.update(z.id, n > k, ts_ms)
            if start is not None:
                alerts.append(self._alert(AlertType.CROWD, 2, ts_ms, start, None, z.id, []))

        present = set(per_gid)
        self.fall_alerter.forget_missing(present)
        self.immobile_alerter.forget_missing(present)
        self.intr_alerter.forget_missing({(g, z.id) for g in present for z in self.zones})
        return alerts
