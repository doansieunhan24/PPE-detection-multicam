"""Event-level metrics (CLAUDE.md §10.2).

An alert matches a GT interval if: same person (global_id → gt_id mapping), same type, and
t_raise ∈ [t_start − δ, t_end + δ]. Each GT interval matches at most one alert; unmatched
alerts are false alarms. FA/hour = false alarms / video duration (h).
Zone-level types (CROWD) have no person: GT gt_id is empty and matching uses type + time only.
A track whose identity switches mid-way (ID swap) is labelled with one tracks.csv row per
stretch; an alert then belongs to the person in the box at t_raise (TrackGT / GidGT).
"""

from __future__ import annotations

import bisect
import csv
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src.common.schemas import AlertEvent, ComplianceState

PERSONLESS_TYPES = frozenset({"CROWD"})

TrackKey = tuple[str, int]  # (cam_id, track_id)


class TrackGT:
    """GT person of each (cam_id, track_id) from tracks.csv. A track labelled on one row has one
    person. A track on several rows (ID swap) is split in time: each row holds over its own
    [first_ms, last_ms]; an empty gt_id there = no actor in that stretch."""

    def __init__(self, segs: Mapping[TrackKey, Sequence[tuple[int, int, int | None]]]):
        self.segs = {k: sorted(v) for k, v in segs.items() if v}

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, str]]) -> TrackGT:
        by_key: dict[TrackKey, list[tuple[int, int, int | None]]] = {}
        for r in rows:
            g = int(r["gt_id"]) if r["gt_id"].strip() else None
            key = (r["cam_id"], int(r["track_id"]))
            by_key.setdefault(key, []).append((int(r["first_ms"]), int(r["last_ms"]), g))
        return cls({k: v for k, v in by_key.items() if len(v) > 1 or v[0][2] is not None})

    @classmethod
    def from_csv(cls, path: str | Path) -> TrackGT:
        with open(path, encoding="utf-8", newline="") as f:
            return cls.from_rows(csv.DictReader(f))

    def is_split(self, key: TrackKey) -> bool:
        return len(self.segs.get(key, ())) > 1

    def at(self, cam: str, tid: int, ts_ms: int) -> int | None:
        s = self.segs.get((cam, tid))
        if not s:
            return None
        if len(s) == 1:
            return s[0][2]
        for a, b, g in s:
            if a <= ts_ms <= b:
                return g
        return min(s, key=lambda x: min(abs(ts_ms - x[0]), abs(ts_ms - x[1])))[2]

    def get(self, key: TrackKey, default: int | None = None) -> int | None:
        """Whole-track person (longest labelled stretch), for time-free uses."""
        s = [x for x in self.segs.get(key, ()) if x[2] is not None]
        return max(s, key=lambda x: x[1] - x[0])[2] if s else default

    def __contains__(self, key: object) -> bool:
        return key in self.segs

    def __len__(self) -> int:
        return len(self.segs)


class GidGT:
    """global_id → gt_id from per-frame gid_map rows (ts_ms, cam_id, track_id, global_id).
    Majority over the frames, as before, unless a track of that global_id is split (ID swap) or
    the global_id joins several tracks of one camera and `stitched` (a run with in-camera track
    stitching): then the person of the frame nearest in time."""

    def __init__(
        self,
        rows: Iterable[tuple[int, str, int, int]],
        track_gt: TrackGT,
        stitched: bool = False,
    ):
        votes: dict[int, Counter] = {}
        frames: dict[int, list[tuple[int, int | None]]] = {}
        split: set[int] = set()
        tracks: dict[int, set[TrackKey]] = {}
        for ts, cam, tid, gid in rows:
            if (cam, tid) not in track_gt:
                continue
            g = track_gt.at(cam, tid, ts)
            frames.setdefault(gid, []).append((ts, g))
            if g is not None:
                votes.setdefault(gid, Counter())[g] += 1
            if track_gt.is_split((cam, tid)):
                split.add(gid)
            seen = tracks.setdefault(gid, set())
            if stitched and (cam, tid) not in seen and any(c == cam for c, _ in seen):
                split.add(gid)  # a second track of the same camera under this global id
            seen.add((cam, tid))
        self.majority = {gid: c.most_common(1)[0][0] for gid, c in votes.items()}
        self.timeline = {gid: sorted(frames[gid]) for gid in split}

    def at(self, gid: int, ts_ms: int) -> int | None:
        tl = self.timeline.get(gid)
        if tl is None:
            return self.majority.get(gid)
        i = bisect.bisect_left(tl, (ts_ms, -1))
        near = [tl[j] for j in (i - 1, i) if 0 <= j < len(tl)]
        return min(near, key=lambda x: abs(x[0] - ts_ms))[1]

    def get(self, gid: int, default: int | None = None) -> int | None:
        return self.majority.get(gid, default)


def _person_of(gid_to_gt: Mapping[int, int] | GidGT, gid: int, ts_ms: int) -> int | None:
    if isinstance(gid_to_gt, GidGT):
        return gid_to_gt.at(gid, ts_ms)
    return gid_to_gt.get(gid)


@dataclass(frozen=True)
class GTInterval:
    gt_id: int | None  # None for PERSONLESS_TYPES
    type: str  # AlertType value, e.g. PPE_HELMET
    t_start_ms: int
    t_end_ms: int
    cams: frozenset[str] = frozenset()  # cameras that can see the event; empty = all


def parse_cams(text: str | None) -> frozenset[str]:
    """'cam2;cam3' | 'cam2 cam3' | '' → cameras of a GT interval (empty = every camera)."""
    return frozenset((text or "").replace(";", " ").replace("|", " ").split())


def gt_for_cams(gt: Iterable[GTInterval], cams: Iterable[str]) -> list[GTInterval]:
    """GT intervals a run on `cams` can be scored on (§10.2, per-camera scoring): an event
    labelled for other cameras only is not a miss of this one."""
    run = set(cams)
    return [g for g in gt if not g.cams or g.cams & run]


@dataclass(frozen=True)
class EventMetrics:
    n_gt: int
    n_alerts: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    fa_per_hour: float
    mean_ttd_s: float | None  # time-to-detect over matched alerts
    fp_duplicate: int = 0  # false alarms repeating a real, already matched violation (info only)

    def as_dict(self) -> dict:
        return asdict(self)


def load_gt_csv(path: str | Path) -> list[GTInterval]:
    """CSV columns: gt_id,type,t_start_ms,t_end_ms[,note][,cams] (tools/label_intervals.py)."""
    with open(path, encoding="utf-8", newline="") as f:
        return [
            GTInterval(
                int(r["gt_id"]) if r["gt_id"].strip() else None,
                r["type"],
                int(r["t_start_ms"]),
                int(r["t_end_ms"]),
                parse_cams(r.get("cams")),
            )
            for r in csv.DictReader(f)
        ]


def match_events(
    alerts: Sequence[AlertEvent],
    gt: Sequence[GTInterval],
    gid_to_gt: Mapping[int, int] | GidGT,
    delta_ms: int = 2000,
) -> tuple[list[tuple[AlertEvent, GTInterval]], list[AlertEvent], list[GTInterval]]:
    """Greedy in time order: each alert takes the earliest-starting unmatched GT it fits."""
    used: set[int] = set()
    matches, fps = [], []
    for a in sorted(alerts, key=lambda a: a.t_raise_ms):
        gid = a.global_id
        gt_id = _person_of(gid_to_gt, gid, a.t_raise_ms) if gid is not None else None
        cand = [
            (i, g)
            for i, g in enumerate(gt)
            if i not in used
            and (g.gt_id == gt_id or g.type in PERSONLESS_TYPES)
            and g.type == a.type.value
            and g.t_start_ms - delta_ms <= a.t_raise_ms <= g.t_end_ms + delta_ms
        ]
        if cand:
            i, g = min(cand, key=lambda c: c[1].t_start_ms)
            used.add(i)
            matches.append((a, g))
        else:
            fps.append(a)
    missed = [g for i, g in enumerate(gt) if i not in used]
    return matches, fps, missed


def duplicate_alerts(
    fps: Sequence[AlertEvent],
    gt: Sequence[GTInterval],
    gid_to_gt: Mapping[int, int] | GidGT,
    delta_ms: int = 2000,
) -> list[AlertEvent]:
    """False alarms that fall in a GT interval of the same person and type: a repeat alert for
    a real violation (e.g. after a track break). Still false alarms under §10.2 (one alert per
    GT interval); counted apart only to say how many of them repeat a true event."""
    out = []
    for a in fps:
        gid = a.global_id
        gt_id = _person_of(gid_to_gt, gid, a.t_raise_ms) if gid is not None else None
        if any(
            (g.gt_id == gt_id or g.type in PERSONLESS_TYPES)
            and g.type == a.type.value
            and g.t_start_ms - delta_ms <= a.t_raise_ms <= g.t_end_ms + delta_ms
            for g in gt
        ):
            out.append(a)
    return out


def event_metrics(
    alerts: Sequence[AlertEvent],
    gt: Sequence[GTInterval],
    gid_to_gt: Mapping[int, int] | GidGT,
    duration_s: float,
    delta_ms: int = 2000,
    types: Iterable[str] | None = None,
) -> EventMetrics:
    if types is not None:
        ts = set(types)
        alerts = [a for a in alerts if a.type.value in ts]
        gt = [g for g in gt if g.type in ts]
    matches, fps, missed = match_events(alerts, gt, gid_to_gt, delta_ms)
    tp, fp, fn = len(matches), len(fps), len(missed)
    dup = len(duplicate_alerts(fps, gt, gid_to_gt, delta_ms))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ttd = [max(0, a.t_raise_ms - g.t_start_ms) / 1000 for a, g in matches]
    return EventMetrics(
        n_gt=len(gt),
        n_alerts=len(alerts),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=prec,
        recall=rec,
        f1=f1,
        fa_per_hour=fp / (duration_s / 3600) if duration_s > 0 else float("nan"),
        mean_ttd_s=sum(ttd) / len(ttd) if ttd else None,
        fp_duplicate=dup,
    )


def unknown_fraction(states: Iterable[ComplianceState]) -> float:
    """Coverage metric: share of (person, part, instant) samples in UNKNOWN."""
    n = u = 0
    for s in states:
        n += 1
        u += s is ComplianceState.UNKNOWN
    return u / n if n else 0.0


def majority_gid_to_gt(pairs: Iterable[tuple[int, int]]) -> dict[int, int]:
    """global_id → gt_id by majority over per-frame (global_id, gt_id) co-occurrences."""
    votes: dict[int, Counter] = {}
    for gid, gt_id in pairs:
        votes.setdefault(gid, Counter())[gt_id] += 1
    return {gid: c.most_common(1)[0][0] for gid, c in votes.items()}


def association_metrics(
    obs: Iterable[tuple[int, str, int, int]], track_gt: Mapping[TrackKey, int] | TrackGT
) -> dict[str, float]:
    """Cross-camera association on labelled tracks (§10.1), from gid_map rows
    (ts_ms, cam_id, track_id, global_id) and the GT person of each (cam_id, track_id).

    pairwise: every pair of labelled observations from different cams at the same instant;
              GT-positive = same gt_id, predicted-positive = same global_id
    count:    distinct global_ids vs distinct gt_ids per instant (labelled obs only)
    frag:     mean number of global_ids per gt_id over the session (1 = never split)
    """
    from collections import defaultdict
    from itertools import combinations

    by_ts: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    gids_of_gt: dict[int, set[int]] = defaultdict(set)
    for ts, cam, tid, gid in obs:
        if isinstance(track_gt, TrackGT):
            gt = track_gt.at(cam, tid, ts)
        else:
            gt = track_gt.get((cam, tid))
        if gt is not None:
            by_ts[ts].append((cam, gt, gid))
            gids_of_gt[gt].add(gid)
    tp = fp = fn = 0
    count_err = []
    for rows in by_ts.values():
        for (ca, ga, pa), (cb, gb, pb) in combinations(rows, 2):
            if ca == cb:
                continue
            same_gt, same_pred = ga == gb, pa == pb
            tp += same_gt and same_pred
            fp += same_pred and not same_gt
            fn += same_gt and not same_pred
        count_err.append(abs(len({r[2] for r in rows}) - len({r[1] for r in rows})))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "pair_tp": tp,
        "pair_fp": fp,
        "pair_fn": fn,
        "pair_precision": prec,
        "pair_recall": rec,
        "pair_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "count_mae": sum(count_err) / len(count_err) if count_err else 0.0,
        "frag": sum(map(len, gids_of_gt.values())) / len(gids_of_gt) if gids_of_gt else 0.0,
        "instants": len(by_ts),
    }
