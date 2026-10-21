"""Event-level metrics (CLAUDE.md §10.2).

An alert matches a GT interval if: same person (global_id → gt_id mapping), same type, and
t_raise ∈ [t_start − δ, t_end + δ]. Each GT interval matches at most one alert; unmatched
alerts are false alarms. FA/hour = false alarms / video duration (h).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src.common.schemas import AlertEvent, ComplianceState


@dataclass(frozen=True)
class GTInterval:
    gt_id: int
    type: str  # AlertType value, e.g. PPE_HELMET
    t_start_ms: int
    t_end_ms: int


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
    """CSV columns: gt_id,type,t_start_ms,t_end_ms (written by tools/label_intervals.py)."""
    with open(path, encoding="utf-8", newline="") as f:
        return [
            GTInterval(int(r["gt_id"]), r["type"], int(r["t_start_ms"]), int(r["t_end_ms"]))
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
            and g.gt_id == gt_id
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
