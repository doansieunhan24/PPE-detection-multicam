"""Event-level metrics (CLAUDE.md §10.2).

An alert matches a GT interval if: same person (global_id → gt_id mapping), same type, and
t_raise ∈ [t_start − δ, t_end + δ]. Each GT interval matches at most one alert; unmatched
alerts are false alarms. FA/hour = false alarms / video duration (h).
Zone-level types (CROWD) have no person: GT gt_id is empty and matching uses type + time only.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src.common.schemas import AlertEvent, ComplianceState

PERSONLESS_TYPES = frozenset({"CROWD"})


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
    gid_to_gt: Mapping[int, int],
    delta_ms: int = 2000,
) -> tuple[list[tuple[AlertEvent, GTInterval]], list[AlertEvent], list[GTInterval]]:
    """Greedy in time order: each alert takes the earliest-starting unmatched GT it fits."""
    used: set[int] = set()
    matches, fps = [], []
    for a in sorted(alerts, key=lambda a: a.t_raise_ms):
        gt_id = gid_to_gt.get(a.global_id) if a.global_id is not None else None
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


def event_metrics(
    alerts: Sequence[AlertEvent],
    gt: Sequence[GTInterval],
    gid_to_gt: Mapping[int, int],
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
    from collections import Counter

    votes: dict[int, Counter] = {}
    for gid, gt_id in pairs:
        votes.setdefault(gid, Counter())[gt_id] += 1
    return {gid: c.most_common(1)[0][0] for gid, c in votes.items()}


def association_metrics(
    obs: Iterable[tuple[int, str, int, int]], track_gt: Mapping[tuple[str, int], int]
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
