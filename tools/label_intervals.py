"""Interval labels for the self-collected dataset C2 (CLAUDE.md §9.1).

Workflow per session:
1) template: list every (cam_id, track_id) with first/last time seen → tracks.csv
       python tools/label_intervals.py template --jsonl data/jsonl/s01 --out data/annotations/s01
   Watch the replay videos (tools/replay.py) and fill gt_id for each track (same physical
   person = same gt_id across cameras; leave empty for false tracks).
   Fill intervals.csv: gt_id,type,t_start_ms,t_end_ms with type in AlertType
   (PPE_HELMET, PPE_VEST, PPE_BOOTS, INTRUSION, FALL, IMMOBILE, CROWD).
2) validate: check both files against the JSONL
       python tools/label_intervals.py validate --jsonl data/jsonl/s01 --ann data/annotations/s01

Shortcut for intervals.csv: write intervals_src.csv by hand with clock times instead of ts_ms,
    gt_id,type,start,end,note      e.g.  P1,PPE_HELMET,1:02.5,1:31,V1
then convert (P1 → gt_id 1):
    # times read off the reference camera's video player
    python tools/label_intervals.py convert --jsonl data/jsonl/s01 --src .../intervals_src.csv \\
        --out data/annotations/s01/intervals.csv --clock video --ref cam1
    # wall-clock times from the paper sheet; --anchor = sheet time of the first light switch,
    # matched to sync_start_ms of --ref in session_check.json (tools/session_check.py)
    python tools/label_intervals.py convert ... --clock wall --anchor 14:03:20 \\
        --session-check data/raw/s01/session_check.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.jsonl import read_frames  # noqa: E402
from src.common.schemas import AlertType  # noqa: E402

TRACK_COLS = ["cam_id", "track_id", "first_ms", "last_ms", "n_frames", "gt_id"]
INTERVAL_COLS = ["gt_id", "type", "t_start_ms", "t_end_ms", "note"]


def track_spans(jsonl_dir: Path) -> dict[tuple[str, int], list[int]]:
    spans: dict[tuple[str, int], list[int]] = {}
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        for fr in read_frames(path):
            for p in fr.persons:
                s = spans.setdefault((fr.cam_id, p.track_id), [fr.ts_ms, fr.ts_ms, 0])
                s[0], s[1], s[2] = min(s[0], fr.ts_ms), max(s[1], fr.ts_ms), s[2] + 1
    return spans


def parse_clock_ms(text: str) -> int:
    """'83.5' | '1:23.5' | '0:01:23.5' → 83500 ms."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(p.strip() for p in parts):
        raise ValueError(f"bad time {text!r}")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p)
    return round(sec * 1000)


def parse_gt_id(text: str) -> int:
    t = text.strip()
    return int(t[1:] if t[:1] in "Pp" else t)


def ref_start_ts_ms(jsonl_dir: Path, ref: str) -> int:
    """ts_ms of the reference camera's first frame (its media time 0)."""
    path = jsonl_dir / f"{ref}.jsonl"
    for fr in read_frames(path):
        return fr.ts_ms
    raise SystemExit(f"{path} is empty")


def convert_rows(
    rows: list[dict[str, str]], t0_ms: int, anchor_ms: int = 0
) -> list[list[str | int]]:
    """Source rows (clock times) → intervals.csv rows (ts_ms = t0 + clock − anchor)."""
    out: list[list[str | int]] = []
    for i, r in enumerate(rows, 2):
        a = t0_ms + parse_clock_ms(r["start"]) - anchor_ms
        b = t0_ms + parse_clock_ms(r["end"]) - anchor_ms
        if b <= a:
            raise SystemExit(f"row {i}: end {r['end']} not after start {r['start']}")
        out.append([parse_gt_id(r["gt_id"]), r["type"].strip(), a, b, r.get("note") or ""])
    return out


def cmd_convert(args: argparse.Namespace) -> None:
    t0 = ref_start_ts_ms(Path(args.jsonl), args.ref)
    anchor = 0
    if args.clock == "wall":
        if not (args.anchor and args.session_check):
            raise SystemExit("--clock wall needs --anchor and --session-check")
        rep = json.loads(Path(args.session_check).read_text(encoding="utf-8"))
        # wall clock w ↔ ref media time sync_start + (w − anchor)
        t0 += int(rep["cams"][args.ref]["sync_start_ms"])
        anchor = parse_clock_ms(args.anchor)
    with open(args.src, encoding="utf-8", newline="") as f:
        rows = convert_rows(list(csv.DictReader(f)), t0, anchor)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(INTERVAL_COLS)
        w.writerows(rows)
    print(f"{len(rows)} intervals -> {out}; run validate next")


def cmd_template(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spans = track_spans(Path(args.jsonl))
    with open(out / "tracks.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRACK_COLS)
        for (cam, tid), (a, b, n) in sorted(spans.items()):
            w.writerow([cam, tid, a, b, n, ""])
    iv = out / "intervals.csv"
    if not iv.exists():
        iv.write_text(",".join(INTERVAL_COLS) + "\n", encoding="utf-8")
    print(f"{len(spans)} tracks -> {out / 'tracks.csv'}; fill gt_id, then intervals.csv")


def cmd_validate(args: argparse.Namespace) -> None:
    ann = Path(args.ann)
    spans = track_spans(Path(args.jsonl))
    errors: list[str] = []
    gt_ids: set[int] = set()
    with open(ann / "tracks.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        key = (r["cam_id"], int(r["track_id"]))
        if key not in spans:
            errors.append(f"tracks.csv: {key} not in JSONL")
        if r["gt_id"].strip():
            gt_ids.add(int(r["gt_id"]))
    unlabeled = sum(1 for r in rows if not r["gt_id"].strip())
    t_min = min(s[0] for s in spans.values()) if spans else 0
    t_max = max(s[1] for s in spans.values()) if spans else 0
    valid_types = {t.value for t in AlertType} - {"UNVERIFIABLE"}
    with open(ann / "intervals.csv", encoding="utf-8", newline="") as f:
        intervals = list(csv.DictReader(f))
    for i, r in enumerate(intervals, 2):
        if int(r["gt_id"]) not in gt_ids:
            errors.append(f"intervals.csv:{i}: gt_id {r['gt_id']} has no track")
        if r["type"] not in valid_types:
            errors.append(f"intervals.csv:{i}: bad type {r['type']}")
        a, b = int(r["t_start_ms"]), int(r["t_end_ms"])
        if not (t_min <= a < b <= t_max):
            errors.append(f"intervals.csv:{i}: [{a},{b}] outside session [{t_min},{t_max}]")
    print(
        f"tracks: {len(rows)} ({unlabeled} without gt_id), persons: {len(gt_ids)}, "
        f"intervals: {len(intervals)}"
    )
    for e in errors:
        print("ERROR", e)
    if errors:
        raise SystemExit(1)
    print("OK")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--jsonl", required=True)
    t.add_argument("--out", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--jsonl", required=True)
    v.add_argument("--ann", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--jsonl", required=True)
    c.add_argument("--src", required=True, help="gt_id,type,start,end,note with clock times")
    c.add_argument("--out", required=True)
    c.add_argument("--clock", choices=["video", "wall"], default="video")
    c.add_argument("--ref", default="cam1")
    c.add_argument("--anchor", help="wall time of the first sync event (hh:mm:ss)")
    c.add_argument("--session-check", help="session_check.json (for --clock wall)")
    args = ap.parse_args(argv)
    {"template": cmd_template, "validate": cmd_validate, "convert": cmd_convert}[args.cmd](args)


if __name__ == "__main__":
    main()
