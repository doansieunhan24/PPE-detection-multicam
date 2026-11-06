"""3-state compliance decision with hysteresis (CLAUDE.md §7.5).

Omega < theta_obs  -> UNKNOWN
m >= theta_pos     -> COMPLIANT
m <= theta_neg     -> VIOLATION
otherwise          -> previous state
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.schemas import ComplianceState


@dataclass(frozen=True)
class StateParams:
    theta_obs: float = 1.0
    theta_pos: float = 1.0
    theta_neg: float = -1.0
    allow_unknown: bool = True  # ablation "no UNKNOWN": never return UNKNOWN after first decision
    initial_state: ComplianceState = ComplianceState.UNKNOWN

    def __post_init__(self) -> None:
        if self.theta_neg > self.theta_pos:
            raise ValueError("theta_neg must be <= theta_pos")

    @classmethod
    def from_config(cls, cfg: dict, part: str | None = None) -> StateParams:
        """part: evidence.theta_neg_part[part] when set (not null) replaces theta_neg; parts differ
        in how negative one no-box frame is (log-LR about -0.7 torso vs -1.2 head on SH17/CHV)."""
        e = cfg["evidence"]
        per_part = (e.get("theta_neg_part") or {}).get(part) if part is not None else None
        return cls(
            theta_obs=e["theta_obs"],
            theta_pos=e["theta_pos"],
            theta_neg=e["theta_neg"] if per_part is None else per_part,
            initial_state=ComplianceState(e.get("initial_state", "UNKNOWN")),
        )


@dataclass
class ComplianceStateMachine:
    params: StateParams
    state: ComplianceState | None = None

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = self.params.initial_state

    def update(self, m: float, omega: float) -> ComplianceState:
        pr = self.params
        if pr.allow_unknown and omega < pr.theta_obs:
            self.state = ComplianceState.UNKNOWN
        elif m >= pr.theta_pos:
            self.state = ComplianceState.COMPLIANT
        elif m <= pr.theta_neg:
            self.state = ComplianceState.VIOLATION
        elif not pr.allow_unknown and self.state is ComplianceState.UNKNOWN:
            # without UNKNOWN, an undecided start falls back to the sign of the evidence
            self.state = ComplianceState.COMPLIANT if m >= 0 else ComplianceState.VIOLATION
        return self.state
