"""PPE alert rules on top of per-instant compliance states (CLAUDE.md §7.5, §7.8).

- PPE_<ITEM> (L2): VIOLATION continuously for >= T_alert while the zone requires the item
- UNVERIFIABLE (L1): UNKNOWN continuously for >= T_unk while the zone requires the item
- one alert per episode; cooldown T_cd per (global_id, type)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.common.schemas import AlertEvent, AlertType
from src.common.schemas import ComplianceState as CS

ITEM_ALERT = {
    "helmet": AlertType.PPE_HELMET,
    "vest": AlertType.PPE_VEST,
    "boots": AlertType.PPE_BOOTS,
}


@dataclass(frozen=True)
class PPEAlertParams:
    T_alert_ms: int = 3000
    T_unk_ms: int = 30000
    T_cd_ms: int = 60000

    @classmethod
    def from_config(cls, cfg: dict, persistence: bool = True) -> PPEAlertParams:
        a = cfg["alert"]
        return cls(
            T_alert_ms=int(a["T_alert_s"] * 1000) if persistence else 0,
            T_unk_ms=int(a["T_unk_s"] * 1000),
            T_cd_ms=int(a["T_cooldown_s"] * 1000),
        )


@dataclass
class _Episode:
    state: CS | None = None
    since_ms: int = 0
    raised: bool = False


@dataclass
class PPEAlertTracker:
    params: PPEAlertParams
    levels: dict[str, int] = field(default_factory=dict)  # AlertType name -> level
    _ep: dict[tuple[int, str], _Episode] = field(default_factory=dict)
    _last_alert: dict[tuple[int, AlertType], int] = field(default_factory=dict)

    def _level(self, t: AlertType, default: int) -> int:
        return self.levels.get(t.value, default)

    def update(
        self,
        gid: int,
        item: str,
        state: CS,
        ts_ms: int,
        required: bool,
        zone: str | None = None,
        cams: list[str] | None = None,
    ) -> AlertEvent | None:
        key = (gid, item)
        ep = self._ep.setdefault(key, _Episode())
        if state is not ep.state:
            ep.state, ep.since_ms, ep.raised = state, ts_ms, False
        if not required or ep.raised:
            return None
        if state is CS.VIOLATION and ts_ms - ep.since_ms >= self.params.T_alert_ms:
            atype, level = ITEM_ALERT[item], 2
        elif state is CS.UNKNOWN and ts_ms - ep.since_ms >= self.params.T_unk_ms:
            atype, level = AlertType.UNVERIFIABLE, 1
        else:
            return None
        ep.raised = True
        last = self._last_alert.get((gid, atype))
        if last is not None and ts_ms - last < self.params.T_cd_ms:
            return None
        self._last_alert[(gid, atype)] = ts_ms
        return AlertEvent(
            alert_id=uuid.uuid4().hex[:12],
            type=atype,
            level=self._level(atype, level),
            global_id=gid,
            zone=zone,
            t_start_ms=ep.since_ms,
            t_raise_ms=ts_ms,
            cams=cams or [],
        )
