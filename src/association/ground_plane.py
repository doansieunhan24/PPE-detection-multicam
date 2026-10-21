"""Cross-camera association on the floor plane (CLAUDE.md §7.2, D3).

1. sync frames of N cameras within sync_tol_ms
2. per camera pair: Hungarian on floor distance, gated by d_max and shared FOV
3. union-find over matched pairs → clusters (one per physical person)
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.schemas import PerceptionFrame

PersonKey = tuple[str, int]  # (cam_id, track_id)


@dataclass(frozen=True)
class FloorObs:
    key: PersonKey
    xy_m: tuple[float, float]
    reliable: bool = True


def sync_groups(
    streams: Mapping[str, Sequence[PerceptionFrame]], tol_ms: int
) -> Iterator[dict[str, PerceptionFrame]]:
    """For each frame of the densest camera, the nearest frame of every other camera
    within tol_ms. Streams must be sorted by ts_ms."""
    if not streams:
        return
    ref = max(streams, key=lambda c: len(streams[c]))
    ts = {c: np.array([f.ts_ms for f in fr], dtype=np.int64) for c, fr in streams.items()}
    for f in streams[ref]:
        group = {ref: f}
        for c, arr in ts.items():
            if c == ref or len(arr) == 0:
                continue
            i = int(np.searchsorted(arr, f.ts_ms))
            cand = [j for j in (i - 1, i) if 0 <= j < len(arr)]
            j = min(cand, key=lambda j: abs(int(arr[j]) - f.ts_ms))
            if abs(int(arr[j]) - f.ts_ms) <= tol_ms:
                group[c] = streams[c][j]
        yield group


def match_pair(
    a: Sequence[FloorObs], b: Sequence[FloorObs], d_max_m: float
) -> list[tuple[PersonKey, PersonKey]]:
    if not a or not b:
        return []
    pa = np.array([o.xy_m for o in a])
    pb = np.array([o.xy_m for o in b])
    cost = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return [(a[r].key, b[c].key) for r, c in zip(rows, cols, strict=True) if cost[r, c] < d_max_m]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[PersonKey, PersonKey] = {}

    def find(self, x: PersonKey) -> PersonKey:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: PersonKey, b: PersonKey) -> None:
        self.parent[self.find(a)] = self.find(b)


def cluster(
    obs_by_cam: Mapping[str, Sequence[FloorObs]],
    d_max_m: float,
    sees: Mapping[str, object] | None = None,
) -> list[list[FloorObs]]:
    """Cluster observations of one synced instant. `sees[cam](xy) -> bool` restricts a pair
    match to points inside both cameras' floor FOV. At most one obs per camera per cluster."""
    uf = _UnionFind()
    by_key: dict[PersonKey, FloorObs] = {}
    for obs in obs_by_cam.values():
        for o in obs:
            uf.find(o.key)
            by_key[o.key] = o
    for ca, cb in combinations(sorted(obs_by_cam), 2):
        a = [o for o in obs_by_cam[ca] if sees is None or sees[cb](o.xy_m)]
        b = [o for o in obs_by_cam[cb] if sees is None or sees[ca](o.xy_m)]
        for ka, kb in match_pair(a, b, d_max_m):
            uf.union(ka, kb)
    groups: dict[PersonKey, list[FloorObs]] = {}
    for k, o in by_key.items():
        groups.setdefault(uf.find(k), []).append(o)
    out = []
    for g in groups.values():
        out.extend(_split_same_cam(g))
    return out


def _split_same_cam(group: list[FloorObs]) -> list[list[FloorObs]]:
    """Union-find chains can merge two people of the same camera (a~b, b~c). Keep, per
    camera, the obs closest to the cluster centre; split the rest off as singletons."""
    cams = [o.key[0] for o in group]
    if len(cams) == len(set(cams)):
        return [group]
    centre = np.mean([o.xy_m for o in group], axis=0)
    best: dict[str, FloorObs] = {}
    rest: list[FloorObs] = []
    for o in sorted(group, key=lambda o: float(np.linalg.norm(np.array(o.xy_m) - centre))):
        if o.key[0] in best:
            rest.append(o)
        else:
            best[o.key[0]] = o
    return [list(best.values())] + [[o] for o in rest]
