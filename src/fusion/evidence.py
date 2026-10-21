"""Visibility-weighted log-odds evidence with exponential temporal decay (CLAUDE.md §7.5).

e_c = v_c * logit(clip(p_c))      E_t = sum_c e_c      W_t = sum_c v_c
lam = exp(-dt / tau)
S_t = lam * S_{t-1} + k_t * E_t   Omega_t = lam * Omega_{t-1} + k_t * W_t
m_t = S_t / max(Omega_t, eps)

k_t = dt (seconds) when dt_weighting is on: a leaky integrator dS/dt = -S/tau + E(t), so
Omega converges to ~ W * tau independent of the frame rate and theta_obs is expressed in
visibility-seconds. With k_t = 1 (formula as first written in §7.5) Omega converges to
~ W * fps * tau, i.e. even v = 0.05 at 25 fps exceeds theta_obs = 1 and UNKNOWN never fires.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field


def logit(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    p = min(max(p, lo), hi)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class EvidenceParams:
    tau_s: float = 2.0
    p_lo: float = 0.01
    p_hi: float = 0.99
    eps: float = 1e-6
    use_visibility: bool = True  # ablation "no v": every observation weighted 1
    temporal: bool = True  # ablation "no temporal": memoryless, Omega = W * tau
    dt_weighting: bool = True  # frame-rate invariant accumulation (see module doc)
    dt_nominal_s: float = 0.04  # weight of the first observation of a track
    dt_max_s: float = 0.2  # cap after gaps so one frame cannot outweigh a gap

    @classmethod
    def from_config(cls, cfg: dict) -> EvidenceParams:
        e = cfg["evidence"]
        lo, hi = e.get("p_clip", [0.01, 0.99])
        return cls(
            tau_s=e["tau_s"],
            p_lo=lo,
            p_hi=hi,
            eps=e.get("eps", 1e-6),
            dt_weighting=e.get("dt_weighting", True),
            dt_nominal_s=e.get("dt_nominal_s", 0.04),
            dt_max_s=e.get("dt_max_s", 0.2),
        )


@dataclass
class EvidenceAccumulator:
    """Running evidence for one (person, part)."""

    params: EvidenceParams = field(default_factory=EvidenceParams)
    S: float = 0.0
    omega: float = 0.0
    last_ts_ms: int | None = None

    def update(self, ts_ms: int, obs: Iterable[tuple[float, float]]) -> tuple[float, float]:
        """Add per-camera observations (p, v) at ts_ms. Returns (m_t, Omega_t)."""
        pr = self.params
        dt_s = (
            pr.dt_nominal_s if self.last_ts_ms is None else max(0, ts_ms - self.last_ts_ms) / 1000.0
        )
        if not pr.temporal:
            lam, k = 0.0, (pr.tau_s if pr.dt_weighting else 1.0)
        else:
            lam = 0.0 if self.last_ts_ms is None else math.exp(-dt_s / pr.tau_s)
            k = min(dt_s, pr.dt_max_s) if pr.dt_weighting else 1.0
        E = W = 0.0
        for p, v in obs:
            w = v if pr.use_visibility else 1.0
            E += w * logit(p, pr.p_lo, pr.p_hi)
            W += w
        self.S = lam * self.S + k * E
        self.omega = lam * self.omega + k * W
        self.last_ts_ms = ts_ms
        return self.mean, self.omega

    @property
    def mean(self) -> float:
        return self.S / max(self.omega, self.params.eps)
