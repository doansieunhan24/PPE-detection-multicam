"""Temporally stable global IDs from per-instant clusters (CLAUDE.md §7.2 step 4–5).

- (cam, track) → global_id by majority vote over the last `window` assignments
- a cluster with no known member inherits a recently lost global_id if it reappears
  within `reuse_dist_m` and `reuse_ttl_ms`; otherwise a new id is allocated
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from src.association.ground_plane import FloorObs, PersonKey


@dataclass
class _Lost:
    xy_m: tuple[float, float]
    ts_ms: int


@dataclass
class GlobalIdManager:
    window: int = 15
    reuse_dist_m: float = 0.8
    reuse_ttl_ms: int = 2000
    _next: int = 1
    _hist: dict[PersonKey, deque[int]] = field(default_factory=dict)
    _last_pos: dict[int, _Lost] = field(default_factory=dict)

    def _vote(self, key: PersonKey) -> int | None:
        h = self._hist.get(key)
        return Counter(h).most_common(1)[0][0] if h else None

    def _new_id(self) -> int:
        gid = self._next
        self._next += 1
        return gid

    def _reuse_lost(self, xy: np.ndarray, ts_ms: int, taken: dict[int, set[str]]) -> int | None:
        best, best_d = None, self.reuse_dist_m
        for gid, lost in self._last_pos.items():
            if gid in taken or ts_ms - lost.ts_ms > self.reuse_ttl_ms:
                continue
            d = float(np.linalg.norm(np.array(lost.xy_m) - xy))
            if d < best_d:
                best, best_d = gid, d
        return best

    def update(self, clusters: Sequence[Sequence[FloorObs]], ts_ms: int) -> dict[PersonKey, int]:
        """Assign a global_id to every observation of this instant."""
        # gid -> cameras already using it this instant. Two clusters may share a gid only if
        # they come from disjoint cameras (one person split by association noise); two people
        # in the same camera never share one.
        taken: dict[int, set[str]] = {}
        result: dict[PersonKey, int] = {}
        # clusters with more members decide first: multi-view evidence is more reliable
        for cl in sorted(clusters, key=len, reverse=True):
            cams = {o.key[0] for o in cl}
            votes = Counter(v for o in cl if (v := self._vote(o.key)) is not None)
            gid = next((g for g, _ in votes.most_common() if not cams & taken.get(g, set())), None)
            centre = np.mean([o.xy_m for o in cl], axis=0)
            if gid is None:
                gid = self._reuse_lost(centre, ts_ms, taken) or self._new_id()
            taken.setdefault(gid, set()).update(cams)
            self._last_pos[gid] = _Lost((float(centre[0]), float(centre[1])), ts_ms)
            for o in cl:
                self._hist.setdefault(o.key, deque(maxlen=self.window)).append(gid)
                result[o.key] = gid
        # forget long-lost ids
        self._last_pos = {
            g: lost for g, lost in self._last_pos.items() if ts_ms - lost.ts_ms <= self.reuse_ttl_ms
        }
        return result
