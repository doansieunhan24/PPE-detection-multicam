"""Generic "condition held for >= T → one alert per episode, cooldown per key" helper."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field


@dataclass
class EpisodeAlerter:
    hold_ms: int
    cooldown_ms: int = 60000
    _since: dict[Hashable, int] = field(default_factory=dict)
    _raised: set[Hashable] = field(default_factory=set)
    _last: dict[Hashable, int] = field(default_factory=dict)

    def update(self, key: Hashable, active: bool, ts_ms: int) -> int | None:
        """Returns the episode start time when an alert should be raised now, else None."""
        if not active:
            self._since.pop(key, None)
            self._raised.discard(key)
            return None
        start = self._since.setdefault(key, ts_ms)
        if key in self._raised or ts_ms - start < self.hold_ms:
            return None
        self._raised.add(key)
        last = self._last.get(key)
        if last is not None and ts_ms - last < self.cooldown_ms:
            return None
        self._last[key] = ts_ms
        return start

    def forget_missing(self, present: set[Hashable]) -> None:
        """End episodes of keys not observed this instant (person left / track lost)."""
        for k in list(self._since):
            if k not in present:
                self._since.pop(k)
                self._raised.discard(k)
