"""Per-(person, part) compliance policies: proposed P and baselines B0–B4 (CLAUDE.md §7.6).

All policies consume the same per-camera observations and return a per-instant state.
Persistence (T_alert) is applied downstream by the alert stage, identically for all
policies except B0, which alerts per frame (T_alert = 0) by definition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from src.common.schemas import ComplianceState as CS
from src.fusion.dempster_shafer import DSAccumulator
from src.fusion.evidence import EvidenceAccumulator, EvidenceParams, logit
from src.fusion.state_machine import ComplianceStateMachine, StateParams


@dataclass(frozen=True)
class CamObs:
    cam_id: str
    p: float  # PPE probability for the part (calibrated, or raw conf / p_neg)
    v: float  # visibility
    det_conf: float  # person detector confidence
    detected: bool | None = None  # a PPE box was matched; None → infer from p >= pos_thr

    def seen(self, pos_thr: float) -> bool:
        return self.detected if self.detected is not None else self.p >= pos_thr


class Policy(Protocol):
    def update(self, ts_ms: int, obs: Sequence[CamObs]) -> CS: ...


@dataclass
class SingleCamPolicy:
    """B0/B1: one camera only; no PPE detected ⇒ violation."""

    cam_id: str
    pos_thr: float = 0.25

    def update(self, ts_ms: int, obs: Sequence[CamObs]) -> CS:
        mine = [o for o in obs if o.cam_id == self.cam_id]
        if not mine:
            return CS.UNKNOWN  # person not in this camera: nothing to say
        return CS.COMPLIANT if mine[0].seen(self.pos_thr) else CS.VIOLATION


@dataclass
class OrPolicy:
    """B2: compliant if any camera sees the PPE."""

    pos_thr: float = 0.25

    def update(self, ts_ms: int, obs: Sequence[CamObs]) -> CS:
        if not obs:
            return CS.UNKNOWN
        return CS.COMPLIANT if any(o.seen(self.pos_thr) for o in obs) else CS.VIOLATION


@dataclass
class MajorityPolicy:
    """B3: per-frame majority vote over cameras (tie → compliant)."""

    pos_thr: float = 0.25

    def update(self, ts_ms: int, obs: Sequence[CamObs]) -> CS:
        if not obs:
            return CS.UNKNOWN
        pos = sum(o.seen(self.pos_thr) for o in obs)
        return CS.COMPLIANT if 2 * pos >= len(obs) else CS.VIOLATION


@dataclass
class ConfWeightedPolicy:
    """B4: per-frame log-odds weighted by person-detector confidence (no visibility)."""

    p_lo: float = 0.01
    p_hi: float = 0.99

    def update(self, ts_ms: int, obs: Sequence[CamObs]) -> CS:
        w = sum(o.det_conf for o in obs)
        if w <= 0:
            return CS.UNKNOWN
        m = sum(o.det_conf * logit(o.p, self.p_lo, self.p_hi) for o in obs) / w
        return CS.COMPLIANT if m >= 0 else CS.VIOLATION


@dataclass
class ProposedPolicy:
    """P: visibility-weighted log-odds + temporal accumulation + UNKNOWN (§7.5).
    `evidence` may also be a DSAccumulator (Dempster–Shafer ablation)."""

    evidence: EvidenceAccumulator | DSAccumulator
    sm: ComplianceStateMachine

    @classmethod
    def create(cls, ev: EvidenceParams, st: StateParams) -> ProposedPolicy:
        return cls(EvidenceAccumulator(ev), ComplianceStateMachine(st))

    def update(self, ts_ms: int, obs: Sequence[CamObs]) -> CS:
        m, omega = self.evidence.update(ts_ms, [(o.p, o.v) for o in obs])
        return self.sm.update(m, omega)


POLICY_IDS = ("B0", "B1", "B2", "B3", "B4", "P")


def make_policy(
    policy_id: str, fusion_cfg: dict, primary_cam: str | None = None, **ablation
) -> Policy:
    """Factory. `ablation` overrides EvidenceParams/StateParams fields for P variants,
    e.g. use_visibility=False, temporal=False, allow_unknown=False,
    method="dempster_shafer"."""
    b = fusion_cfg.get("baselines", {})
    thr = b.get("pos_thr", 0.25)
    if policy_id in ("B0", "B1"):
        return SingleCamPolicy(primary_cam or b.get("primary_cam", "cam1"), thr)
    if policy_id == "B2":
        return OrPolicy(thr)
    if policy_id == "B3":
        return MajorityPolicy(thr)
    if policy_id == "B4":
        lo, hi = fusion_cfg["evidence"].get("p_clip", [0.01, 0.99])
        return ConfWeightedPolicy(lo, hi)
    if policy_id == "P":
        ev = EvidenceParams.from_config(fusion_cfg)
        st = StateParams.from_config(fusion_cfg)
        ev_kw = {k: v for k, v in ablation.items() if k in ("use_visibility", "temporal")}
        st_kw = {k: v for k, v in ablation.items() if k in ("allow_unknown",)}
        ev = EvidenceParams(**{**ev.__dict__, **ev_kw})
        method = ablation.get("method", fusion_cfg["evidence"].get("method", "logodds"))
        if method == "dempster_shafer":
            ds = fusion_cfg["evidence"]["ds"]
            st = StateParams(
                **{
                    **st.__dict__,
                    "theta_obs": ds["theta_obs"],
                    "theta_pos": ds["theta_pos"],
                    "theta_neg": ds["theta_neg"],
                    **st_kw,
                }
            )
            acc = DSAccumulator(tau_s=ev.tau_s, temporal=ev.temporal)
            return ProposedPolicy(acc, ComplianceStateMachine(st))
        st = StateParams(**{**st.__dict__, **st_kw})
        return ProposedPolicy.create(ev, st)
    raise ValueError(f"unknown policy {policy_id!r}")
