"""Visibility-weighted log-odds evidence with exponential temporal decay (CLAUDE.md §7.5).

e_c = v_c * logit(clip(p_c))      E_t = sum_c e_c      W_t = sum_c v_c
lam = exp(-dt / tau)
S_t = lam * S_{t-1} + E_t         Omega_t = lam * Omega_{t-1} + W_t
m_t = S_t / max(Omega_t, eps)
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
    temporal: bool = True  # ablation "no temporal": lam = 0 (memoryless)

    @classmethod
    def from_config(cls, cfg: dict) -> EvidenceParams:
        e = cfg["evidence"]
        lo, hi = e.get("p_clip", [0.01, 0.99])
        return cls(tau_s=e["tau_s"], p_lo=lo, p_hi=hi, eps=e.get("eps", 1e-6))


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
        if self.last_ts_ms is None or not pr.temporal:
            lam = 0.0
        else:
            dt_s = max(0, ts_ms - self.last_ts_ms) / 1000.0
            lam = math.exp(-dt_s / pr.tau_s)
        E = W = 0.0
        for p, v in obs:
            w = v if pr.use_visibility else 1.0
            E += w * logit(p, pr.p_lo, pr.p_hi)
            W += w
        self.S = lam * self.S + E
        self.omega = lam * self.omega + W
        self.last_ts_ms = ts_ms
        return self.mean, self.omega

    @property
    def mean(self) -> float:
        return self.S / max(self.omega, self.params.eps)
