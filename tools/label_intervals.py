"""Interval labels for the self-collected dataset C2 (CLAUDE.md §9.1).

Workflow per session:
1) template: list every (cam_id, track_id) with first/last time seen → tracks.csv
       python tools/label_intervals.py template --jsonl data/jsonl/s01 --out data/annotations/s01
   Watch the replay videos (tools/replay.py) and fill gt_id for each track (same physical
   person = same gt_id across cameras; leave empty for false tracks). Faster: contact sheets
   with 3 crops per track (largest box, first, last), one page per 20 tracks and camera:
       python tools/label_intervals.py thumbs --jsonl data/jsonl/s01 --out data/annotations/s01
   (videos from the session cameras.yaml `uri`, or --video cam1=path ...)
   Fill intervals.csv: gt_id,type,t_start_ms,t_end_ms with type in AlertType
   (PPE_HELMET, PPE_VEST, PPE_BOOTS, INTRUSION, FALL, IMMOBILE, CROWD).
   CROWD is per zone, not per person: leave its gt_id empty (or "-" in intervals_src.csv).
2) validate: check both files against the JSONL
       python tools/label_intervals.py validate --jsonl data/jsonl/s01 --ann data/annotations/s01

Shortcut for intervals.csv: write intervals_src.csv by hand with clock times instead of ts_ms,
    gt_id,type,start,end,note,cams      e.g.  P1,PPE_HELMET,1:02.5,1:31,V1,cam2;cam3
`cams` (optional) = cameras that can see the event; empty = all. Per-camera scoring (§10.2)
only counts an interval for these cameras, so an event out of a camera's view is not its miss.
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

from eval.metrics import PERSONLESS_TYPES, parse_cams  # noqa: E402
from src.common.jsonl import read_frames  # noqa: E402
from src.common.schemas import AlertType  # noqa: E402

TRACK_COLS = ["cam_id", "track_id", "first_ms", "last_ms", "n_frames", "gt_id"]
INTERVAL_COLS = ["gt_id", "type", "t_start_ms", "t_end_ms", "note", "cams"]


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


def parse_gt_id(text: str) -> int | str:
    """'P3' | '3' → 3; '' | '-' → '' (zone-level CROWD rows)."""
    t = text.strip()
    if t in ("", "-"):
        return ""
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
        try:
            a = t0_ms + parse_clock_ms(r["start"]) - anchor_ms
            b = t0_ms + parse_clock_ms(r["end"]) - anchor_ms
            gt_id = parse_gt_id(r["gt_id"])
        except ValueError:
            raise SystemExit(
                f"row {i}: fill gt_id/start/end (got {r['gt_id']!r}, {r['start']!r}, "
                f"{r['end']!r}; TODO and '?' come from label_assist.py prefill)"
            ) from None
        if b <= a:
            raise SystemExit(f"row {i}: end {r['end']} not after start {r['start']}")
        typ = r["type"].strip()
        if (gt_id == "") != (typ in PERSONLESS_TYPES):
            raise SystemExit(f"row {i}: gt_id must be empty exactly for {sorted(PERSONLESS_TYPES)}")
        cams = " ".join(sorted(parse_cams(r.get("cams"))))
        out.append([gt_id, typ, a, b, r.get("note") or "", cams])
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


def pick_thumb_frames(jsonl_path: Path) -> dict[int, list[tuple[int, list[float], int]]]:
    """track_id → [(frame_idx, bbox, ts_ms)] for its largest box, first and last frame."""
    seen: dict[int, dict[str, tuple]] = {}
    for fr in read_frames(jsonl_path):
        for p in fr.persons:
            e = (fr.frame_idx, list(p.bbox), fr.ts_ms)
            area = (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])
            d = seen.setdefault(p.track_id, {"first": e, "big": (area, e)})
            d["last"] = e
            if area > d["big"][0]:
                d["big"] = (area, e)
    return {tid: _dedupe([d["big"][1], d["first"], d["last"]]) for tid, d in seen.items()}


def _dedupe(entries: list[tuple]) -> list[tuple]:
    out: list[tuple] = []
    for e in entries:
        if all(e[0] != o[0] for o in out):
            out.append(e)
    return out


def cmd_thumbs(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    import yaml

    jdir, out = Path(args.jsonl), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = dict(v.split("=", 1) for v in args.video or [])
    cams_yaml = jdir / "cameras.yaml"
    if cams_yaml.exists():
        for c in yaml.safe_load(cams_yaml.read_text(encoding="utf-8")).get("cameras", []):
            videos.setdefault(c["cam_id"], c.get("uri", ""))
    t0 = min((fr.ts_ms for p in jdir.glob("*.jsonl") for fr in read_frames(p)), default=0)
    h = args.height
    for path in sorted(jdir.glob("*.jsonl")):
        cam = path.stem
        src = videos.get(cam)
        if not src or not Path(src).exists():
            print(f"{cam}: no video (cameras.yaml uri or --video {cam}=...), skipped")
            continue
        picks = pick_thumb_frames(path)
        need: dict[int, list[tuple[int, list[float]]]] = {}
        for tid, es in picks.items():
            for fi, box, _ in es:
                need.setdefault(fi, []).append((tid, box))
        crops: dict[tuple[int, int], np.ndarray] = {}
        cap = cv2.VideoCapture(src)
        fi, last = 0, max(need) if need else -1
        while fi <= last:
            ok, img = cap.read()
            if not ok:
                break
            for tid, (x1, y1, x2, y2) in need.get(fi, []):
                c = img[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)]
                if c.size:
                    crops[(tid, fi)] = cv2.resize(c, (max(1, int(c.shape[1] * h / c.shape[0])), h))
            fi += 1
        cap.release()
        rows = []
        for tid in sorted(picks, key=lambda t: min(e[2] for e in picks[t])):
            es = picks[tid]
            tiles = [crops[(tid, e[0])] for e in es if (tid, e[0]) in crops]
            a, b = min(e[2] for e in es) - t0, max(e[2] for e in es) - t0
            cap_img = np.full((h, 260, 3), 255, np.uint8)
            cv2.putText(cap_img, f"{cam} track {tid}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 2)  # fmt: skip
            cv2.putText(cap_img, f"{a / 1000:.1f}-{b / 1000:.1f} s", (8, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)  # fmt: skip
            rows.append(np.hstack([cap_img, *tiles]) if tiles else cap_img)
        for page in range(0, len(rows), args.per_page):
            chunk = rows[page : page + args.per_page]
            w = max(r.shape[1] for r in chunk)
            sheet = np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0)),
                                      constant_values=255) for r in chunk])  # fmt: skip
            cv2.imwrite(str(out / f"thumbs_{cam}_{page // args.per_page + 1:02d}.jpg"), sheet)
        print(f"{cam}: {len(rows)} tracks -> {out}/thumbs_{cam}_*.jpg")


def cmd_validate(args: argparse.Namespace) -> None:
    ann = Path(args.ann)
    spans = track_spans(Path(args.jsonl))
    errors: list[str] = []
    gt_ids: set[int] = set()
    with open(ann / "tracks.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    stretches: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for r in rows:
        key = (r["cam_id"], int(r["track_id"]))
        if key not in spans:
            errors.append(f"tracks.csv: {key} not in JSONL")
        stretches.setdefault(key, []).append((int(r["first_ms"]), int(r["last_ms"])))
        if r["gt_id"].strip():
            gt_ids.add(int(r["gt_id"]))
    for key, st in stretches.items():  # a track on several rows = ID swap, split in time
        if len(st) < 2:
            continue
        st.sort()
        gaps = zip(st, st[1:], strict=False)
        if any(a >= b for a, b in st) or any(b0 > a1 for (_, b0), (a1, _) in gaps):
            errors.append(f"tracks.csv: {key} split rows must be ordered, non-overlapping {st}")
        lo, hi = (spans[key][0] - 1000, spans[key][1] + 1000) if key in spans else (0, 0)
        if key in spans and not (lo <= st[0][0] and st[-1][1] <= hi):
            errors.append(f"tracks.csv: {key} split rows {st} outside the track {spans[key][:2]}")
    unlabeled = sum(1 for r in rows if not r["gt_id"].strip())
    t_min = min(s[0] for s in spans.values()) if spans else 0
    t_max = max(s[1] for s in spans.values()) if spans else 0
    session_cams = {p.stem for p in Path(args.jsonl).glob("*.jsonl")}
    valid_types = {t.value for t in AlertType} - {"UNVERIFIABLE"}
    with open(ann / "intervals.csv", encoding="utf-8", newline="") as f:
        intervals = list(csv.DictReader(f))
    for i, r in enumerate(intervals, 2):
        if r["type"] in PERSONLESS_TYPES:
            if r["gt_id"].strip():
                errors.append(f"intervals.csv:{i}: {r['type']} is per zone, leave gt_id empty")
        elif not r["gt_id"].strip() or int(r["gt_id"]) not in gt_ids:
            errors.append(f"intervals.csv:{i}: gt_id {r['gt_id']!r} has no track")
        if r["type"] not in valid_types:
            errors.append(f"intervals.csv:{i}: bad type {r['type']}")
        if unknown := parse_cams(r.get("cams")) - session_cams:
            errors.append(f"intervals.csv:{i}: cams {sorted(unknown)} not in JSONL")
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
    th = sub.add_parser("thumbs")
    th.add_argument("--jsonl", required=True)
    th.add_argument("--out", required=True)
    th.add_argument("--video", action="append", help="cam_id=path (default: cameras.yaml uri)")
    th.add_argument("--height", type=int, default=160)
    th.add_argument("--per-page", type=int, default=20)
    args = ap.parse_args(argv)
    cmds = {"template": cmd_template, "validate": cmd_validate, "convert": cmd_convert,
            "thumbs": cmd_thumbs}  # fmt: skip
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
