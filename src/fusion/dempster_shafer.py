"""Dempster–Shafer variant of the evidence accumulator (ablation, CLAUDE.md §7.5).

Frame {PPE, noPPE}. Per camera: m(PPE) = v·p, m(noPPE) = v·(1−p), m(Θ) = 1 − v (ignorance).
Temporal: the previous fused mass is discounted by λ = exp(−dt/τ) (moved to Θ), then
combined with the new per-camera masses by Dempster's rule.

Returned (score, obs) mimic EvidenceAccumulator so the same state machine applies:
score = m(PPE) − m(noPPE) ∈ [−1, 1], obs = 1 − m(Θ) ∈ [0, 1].
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

Mass = tuple[float, float, float]  # (PPE, noPPE, Θ)
VACUOUS: Mass = (0.0, 0.0, 1.0)


def combine(a: Mass, b: Mass) -> Mass:
    """Dempster's rule on the binary frame. Total conflict → vacuous."""
    k = a[0] * b[1] + a[1] * b[0]
    if k >= 1.0 - 1e-12:
        return VACUOUS
    yes = a[0] * b[0] + a[0] * b[2] + a[2] * b[0]
    no = a[1] * b[1] + a[1] * b[2] + a[2] * b[1]
    th = a[2] * b[2]
    n = 1.0 - k
    return (yes / n, no / n, th / n)


def discount(m: Mass, alpha: float) -> Mass:
    """Shafer discounting: keep a fraction alpha, move the rest to Θ."""
    return (alpha * m[0], alpha * m[1], 1.0 - alpha * (m[0] + m[1]))


def cam_mass(p: float, v: float, p_lo: float = 0.01, p_hi: float = 0.99) -> Mass:
    p = min(max(p, p_lo), p_hi)
    v = min(max(v, 0.0), 1.0)
    return (v * p, v * (1.0 - p), 1.0 - v)


@dataclass
class DSAccumulator:
    tau_s: float = 2.0
    temporal: bool = True
    mass: Mass = VACUOUS
    last_ts_ms: int | None = None

    def update(self, ts_ms: int, obs: Iterable[tuple[float, float]]) -> tuple[float, float]:
        if self.last_ts_ms is None or not self.temporal:
            prior = VACUOUS
        else:
            dt_s = max(0, ts_ms - self.last_ts_ms) / 1000.0
            prior = discount(self.mass, math.exp(-dt_s / self.tau_s))
        m = prior
        for p, v in obs:
            m = combine(m, cam_mass(p, v))
        self.mass, self.last_ts_ms = m, ts_ms
        return self.mean, self.omega

    @property
    def mean(self) -> float:
        return self.mass[0] - self.mass[1]

    @property
    def omega(self) -> float:
        return 1.0 - self.mass[2]
