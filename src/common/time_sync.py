"""Streaming counterpart of association.ground_plane.sync_groups (live analytics, §7.2 step 1).

Frames arrive per camera in ts order but cameras interleave arbitrarily and may lag. A frame
of the reference camera is emitted together with the nearest frame (|Δts| ≤ tol_ms) of every
other camera once the decision cannot change any more: each other camera has delivered a
frame with ts ≥ ref.ts + tol_ms, or has gone stale, or the reference frame has waited
max_wait_ms of wall time. With complete streams and no timeouts the groups are the same as
sync_groups() with the same reference camera.

The reference is the first camera of `priority` that is not stale (no frame for stale_ms),
so a dead reference camera does not stop the analytics. A camera that has not sent anything
yet counts as alive for stale_ms after the first frame of any camera (start-up).
"""

from __future__ import annotations

from collections import deque

from src.common.schemas import PerceptionFrame


class LiveSyncer:
    def __init__(
        self,
        priority: list[str],
        tol_ms: int,
        max_wait_ms: int = 500,
        stale_ms: int = 2000,
    ) -> None:
        self.priority = list(priority)
        self.tol_ms = tol_ms
        self.max_wait_ms = max_wait_ms
        self.stale_ms = stale_ms
        self.buf: dict[str, deque[tuple[PerceptionFrame, int]]] = {c: deque() for c in priority}
        self.last_arrival: dict[str, int] = {}
        self.last_ts: dict[str, int] = {}
        self._last_ref_ts: int | None = None
        self._t0: int | None = None
        self.out_of_order = 0

    def push(self, frame: PerceptionFrame, now_ms: int) -> None:
        cam = frame.cam_id
        if cam not in self.buf:
            self.priority.append(cam)
            self.buf[cam] = deque()
        if cam in self.last_ts and frame.ts_ms < self.last_ts[cam]:
            self.out_of_order += 1  # late duplicate / clock step back: drop
            return
        if self._t0 is None:
            self._t0 = now_ms
        self.buf[cam].append((frame, now_ms))
        self.last_ts[cam] = frame.ts_ms
        self.last_arrival[cam] = now_ms

    def _alive(self, cam: str, now_ms: int) -> bool:
        t = self.last_arrival.get(cam, self._t0)
        return t is not None and now_ms - t <= self.stale_ms

    def reference(self, now_ms: int) -> str | None:
        return next((c for c in self.priority if self._alive(c, now_ms)), None)

    def _nearest(self, cam: str, ts: int) -> PerceptionFrame | None:
        best = None
        for f, _ in self.buf[cam]:
            d = abs(f.ts_ms - ts)
            if d <= self.tol_ms and (best is None or d < abs(best.ts_ms - ts)):
                best = f
        return best

    def pop_ready(self, now_ms: int, flush: bool = False) -> list[dict[str, PerceptionFrame]]:
        """Groups that are final at now_ms. flush=True (end of a replay): everything still
        buffered for the reference camera as it was at the last arrival."""
        out = []
        if flush and self.last_arrival:
            now_ms = max(self.last_arrival.values())
        while True:
            ref = self.reference(now_ms)
            if ref is None or not self.buf[ref]:
                break
            f, arrived = self.buf[ref][0]
            if not flush and now_ms - arrived < self.max_wait_ms:
                waiting = [
                    c
                    for c in self.buf
                    if c != ref
                    and self._alive(c, now_ms)
                    and self.last_ts.get(c, -(2**62)) < f.ts_ms + self.tol_ms
                ]
                if waiting:
                    break
            self.buf[ref].popleft()
            group = {ref: f}
            for c in self.buf:
                if c == ref:
                    continue
                # frames that can no longer be nearest to this or any later reference frame
                while self.buf[c] and self.buf[c][0][0].ts_ms < f.ts_ms - self.tol_ms:
                    self.buf[c].popleft()
                m = self._nearest(c, f.ts_ms)
                if m is not None:
                    group[c] = m
            if self._last_ref_ts is not None and f.ts_ms <= self._last_ref_ts:
                continue  # reference switched to a camera that is behind: keep time monotonic
            self._last_ref_ts = f.ts_ms
            out.append(group)
        return out
