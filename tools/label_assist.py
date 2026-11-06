"""Semi-automatic help for C2 interval labels (docs/c2_labeling_guide.md). Suggestions only:
a person reviews every one of them before it becomes ground truth.

What is automated and why it is not circular (CLAUDE.md §10, §14.2):
- `ids`: track → person (gt_id) from the colour of the torso/legs, propagated from a few seed
  tracks the labeller marks by hand. It only links detector tracks to actors; it says nothing
  about PPE / zones, and the same mapping is used by every method (B0, B1, B4, P).
- `prefill`: intervals_src.csv rows from the take script (what the actors were told to wear),
  not from the detector. Times that need watching the video are written as TODO.
- `timeline`: contact sheets (one frame per second, zones drawn from the calibration, no
  detector output on them) to read start/end times without scrubbing the video.
INTRUSION / CROWD / IMMOBILE times are never derived from the system's floor positions.

    python tools/label_assist.py ids --jsonl data/jsonl/s01_t* --ann-root data/annotations
    python tools/label_assist.py prefill --spec data/annotations/s01_script.yaml \\
        --jsonl data/jsonl/s01_t* --ann-root data/annotations --ref cam2
    python tools/label_assist.py timeline --jsonl data/jsonl/s01_t08 \\
        --out data/annotations/s01_t08/timeline --cams cam2 cam3

`ids` seeds: in tracks.csv of any take of the session, fill gt_id of one long, clear track per
person and camera (status empty); only these rows are seeds. Rows written by `ids` carry status
auto / review / short / nofeat / nomatch and are re-suggested on every run. To correct one, fix
gt_id and set status to `fixed`: kept as is, not a seed (corrections are mostly close-ups or
partial bodies, poor colour models); `fixed` with an empty gt_id = checked, no actor. A track
whose identity switches mid-way (ID swap): the person of each stretch on its own row (same
cam_id, track_id; first_ms / last_ms = the stretch) with status `swap`. `accept` freezes a
checked take (auto -> ok). Only `auto` rows get a gt_id; `review` rows keep it empty with the
guess in `suggest`; `nomatch` = like no actor (reflection, cabinet), listed last on the sheet. A
take where someone changed clothes needs its own seed for that person (same take + camera seeds
are preferred). Check review_ids_<cam>.jpg (tracks grouped by person) before scoring.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_yaml  # noqa: E402
from src.common.jsonl import read_frames  # noqa: E402
from tools.label_intervals import TRACK_COLS  # noqa: E402

TORSO_KPTS = (5, 6, 11, 12)  # COCO shoulders, hips
LEG_KPTS = (11, 12, 15, 16)  # hips, ankles
AUTO_STATUSES = {"auto", "review", "short", "nofeat", "nomatch"}
MANUAL_SKIP = "swap"  # labelled by hand, kept out of seeds and suggestions
ACCEPTED = "ok"  # auto row checked by a person (`accept`): kept, not re-suggested, not a seed
EXTRA_COLS = ["suggest", "sim", "margin", "status"]
H_BINS, S_BINS, V_BINS = 16, 4, 4


# ---------------------------------------------------------------- appearance features


def kp_roi(
    kpts: Sequence[Sequence[float]], idx: Iterable[int], min_conf: float
) -> tuple[float, float, float, float] | None:
    pts = [kpts[i][:2] for i in idx if len(kpts) > i and kpts[i][2] >= min_conf]
    if len(pts) < 3:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def torso_roi(
    kpts: Sequence[Sequence[float]], bbox: Sequence[float], min_conf: float = 0.5
) -> tuple[tuple[float, float, float, float], bool]:
    """Torso box (where the vest is) from shoulders/hips; fallback 25–60 % of the person box.
    Returns (roi, from_keypoints)."""
    r = kp_roi(kpts, TORSO_KPTS, min_conf)
    if r is not None and r[2] - r[0] >= 4 and r[3] - r[1] >= 8:
        return r, True
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    return (x1 + 0.15 * (x2 - x1), y1 + 0.25 * h, x2 - 0.15 * (x2 - x1), y1 + 0.6 * h), False


def leg_roi(
    kpts: Sequence[Sequence[float]], min_conf: float = 0.5
) -> tuple[float, float, float, float] | None:
    r = kp_roi(kpts, LEG_KPTS, min_conf)
    if r is None or r[3] - r[1] < 10:
        return None
    pad = max(6.0, 0.25 * (r[2] - r[0]))
    return r[0] - pad, r[1], r[2] + pad, r[3]


def color_hist(img_bgr: np.ndarray) -> np.ndarray | None:
    """L1-normalised HSV histogram (16 hue × 4 sat × 4 value bins)."""
    import cv2

    if img_bgr.size == 0 or min(img_bgr.shape[:2]) < 4:
        return None
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256])
    h = h.ravel().astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else None


def hellinger_sim(p: np.ndarray, q: np.ndarray) -> float:
    """Bhattacharyya coefficient of two L1-normalised histograms: 1 = identical, 0 = disjoint."""
    return float(np.sqrt(p * q).sum())


@dataclass
class TrackFeat:
    torso: np.ndarray | None
    legs: np.ndarray | None
    n_used: int


def track_sim(a: TrackFeat, b: TrackFeat, w_torso: float = 0.7) -> float | None:
    if a.torso is None or b.torso is None:
        return None
    s = hellinger_sim(a.torso, b.torso)
    if a.legs is not None and b.legs is not None:
        s = w_torso * s + (1 - w_torso) * hellinger_sim(a.legs, b.legs)
    return s


def pick_samples(
    jsonl_path: Path, k: int, min_h_px: float
) -> dict[int, list[tuple[int, list[float], list[list[float]]]]]:
    """track_id → up to k evenly spaced (frame_idx, bbox, kpts), preferring frames where the
    torso keypoints are visible and the person is not tiny."""
    cand: dict[int, list[tuple[int, int, list[float], list[list[float]]]]] = {}
    for fr in read_frames(jsonl_path):
        for p in fr.persons:
            h = p.bbox[3] - p.bbox[1]
            good = int(h >= min_h_px and kp_roi(p.kpts, TORSO_KPTS, 0.5) is not None)
            cand.setdefault(p.track_id, []).append(
                (good, fr.frame_idx, list(p.bbox), [list(x) for x in p.kpts])
            )
    out = {}
    for tid, rows in cand.items():
        good = [r for r in rows if r[0]] or [r for r in rows if r[2][3] - r[2][1] >= min_h_px]
        if not good:
            out[tid] = []
            continue
        sel = np.linspace(0, len(good) - 1, min(k, len(good))).round().astype(int)
        out[tid] = [good[i][1:] for i in sorted(set(sel.tolist()))]
    return out


def _crop(img: np.ndarray, box: Sequence[float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    return img[y1:y2, x1:x2]


def extract_cam_feats(
    video: str, jsonl_path: Path, k: int = 12, min_h_px: float = 80, thumb_h: int = 96
) -> tuple[dict[int, TrackFeat], dict[int, list[np.ndarray]]]:
    """Decode the video once; per track: mean torso/leg histograms and up to 3 person crops."""
    import cv2

    samples = pick_samples(jsonl_path, k, min_h_px)
    need: dict[int, list[tuple[int, list[float], list[list[float]]]]] = {}
    for tid, es in samples.items():
        for fi, box, kp in es:
            need.setdefault(fi, []).append((tid, box, kp))
    torso: dict[int, list[np.ndarray]] = {}
    legs: dict[int, list[np.ndarray]] = {}
    thumbs: dict[int, list[np.ndarray]] = {}
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fi, last = 0, max(need) if need else -1
    while fi <= last:
        if fi not in need:
            if not cap.grab():
                break
            fi += 1
            continue
        ok, img = cap.read()
        if not ok:
            break
        for tid, box, kp in need[fi]:
            roi, _ = torso_roi(kp, box)
            if (hist := color_hist(_crop(img, roi))) is not None:
                torso.setdefault(tid, []).append(hist)
            lr = leg_roi(kp)
            if lr is not None and (hist := color_hist(_crop(img, lr))) is not None:
                legs.setdefault(tid, []).append(hist)
            c = _crop(img, box)
            if c.size and len(thumbs.setdefault(tid, [])) < 12:
                thumbs[tid].append(
                    cv2.resize(c, (max(1, int(c.shape[1] * thumb_h / c.shape[0])), thumb_h))
                )
        fi += 1
    cap.release()
    feats = {}
    for tid in samples:
        t, lg = torso.get(tid), legs.get(tid)
        feats[tid] = TrackFeat(
            np.mean(t, axis=0) if t else None, np.mean(lg, axis=0) if lg else None, len(t or [])
        )
    # keep 3 spread-out crops for the review sheet
    for tid, ts in thumbs.items():
        idx = np.linspace(0, len(ts) - 1, min(3, len(ts))).round().astype(int)
        thumbs[tid] = [ts[i] for i in sorted(set(idx.tolist()))]
    return feats, thumbs


def save_feats(path: Path, feats: dict[int, TrackFeat], thumbs: dict[int, list[np.ndarray]]):
    d: dict[str, np.ndarray] = {}
    for tid, f in feats.items():
        if f.torso is not None:
            d[f"t{tid}"] = f.torso
        if f.legs is not None:
            d[f"l{tid}"] = f.legs
        d[f"n{tid}"] = np.array(f.n_used)
        for j, im in enumerate(thumbs.get(tid, [])):
            d[f"c{tid}_{j}"] = im
    np.savez_compressed(path, **d)


def load_feats(path: Path) -> tuple[dict[int, TrackFeat], dict[int, list[np.ndarray]]]:
    z = np.load(path)
    feats: dict[int, TrackFeat] = {}
    thumbs: dict[int, list[tuple[int, np.ndarray]]] = {}
    for key in z.files:
        if key[0] == "n":
            tid = int(key[1:])
            feats[tid] = TrackFeat(
                z[f"t{tid}"] if f"t{tid}" in z.files else None,
                z[f"l{tid}"] if f"l{tid}" in z.files else None,
                int(z[key]),
            )
        elif key[0] == "c":
            tid, j = (int(x) for x in key[1:].split("_"))
            thumbs.setdefault(tid, []).append((j, z[key]))
    return feats, {t: [im for _, im in sorted(v, key=lambda e: e[0])] for t, v in thumbs.items()}


# ---------------------------------------------------------------- assignment (pure)

TrackKey = tuple[str, str, int]  # (take, cam, track_id)


@dataclass
class Suggestion:
    gt: int | None  # best person after the overlap constraint
    sim: float | None
    margin: float | None
    status: str  # auto | review | short | nofeat | nomatch


def suggest_ids(
    feats: dict[TrackKey, TrackFeat],
    spans: dict[TrackKey, tuple[int, int, int]],
    seeds: dict[TrackKey, int],
    min_frames: int = 5,
    min_sim: float = 0.55,
    min_margin: float = 0.05,
    overlap_ms: int = 1000,
    top_k: int = 3,
    nomatch_sim: float = 0.4,
) -> dict[TrackKey, Suggestion]:
    """Nearest-person by colour, with seeds of the same take/camera preferred (lighting differs
    per camera, clothes may change per take), then greedy by confidence under the constraint
    that two tracks of one camera that coexist for > overlap_ms are different people."""
    people = sorted(set(seeds.values()))
    out: dict[TrackKey, Suggestion] = {}
    scored: list[tuple[float, TrackKey, list[tuple[float, int]]]] = []
    for key, f in feats.items():
        if key in seeds:
            continue
        if spans[key][2] < min_frames:
            out[key] = Suggestion(None, None, None, "short")
            continue
        per: list[tuple[float, int]] = []
        for g in people:
            mine = [s for s in seeds if seeds[s] == g and s in feats]
            # same take + camera (clothes changed in this take) > same camera > any camera
            ref = [s for s in mine if s[:2] == key[:2]] or [s for s in mine if s[1] == key[1]]
            ref = ref or mine
            sims = sorted(
                (v for s in ref if (v := track_sim(f, feats[s])) is not None), reverse=True
            )
            if sims:
                per.append((float(np.mean(sims[:top_k])), g))
        if not per:
            out[key] = Suggestion(None, None, None, "nofeat")
            continue
        per.sort(reverse=True)
        margin = per[0][0] - (per[1][0] if len(per) > 1 else 0.0)
        if per[0][0] < nomatch_sim:  # like no actor: reflection, cabinet, poster ...
            out[key] = Suggestion(per[0][1], per[0][0], margin, "nomatch")
            continue
        scored.append((margin, key, per))

    assigned: dict[TrackKey, int] = dict(seeds)

    def conflicts(key: TrackKey, g: int) -> bool:
        a0, a1, _ = spans[key]
        for other, og in assigned.items():
            if og != g or other[:2] != key[:2] or other == key:
                continue
            b0, b1, _ = spans[other]
            if min(a1, b1) - max(a0, b0) > overlap_ms:
                return True
        return False

    for margin, key, per in sorted(scored, key=lambda e: -e[0]):
        free = [(s, g) for s, g in per if not conflicts(key, g)]
        if not free:
            out[key] = Suggestion(per[0][1], per[0][0], margin, "review")
            continue
        s, g = free[0]
        blocked = free[0] != per[0]
        rest = [x for x, _ in free[1:]]
        m = s - (rest[0] if rest else 0.0)
        ok = not blocked and s >= min_sim and m >= min_margin
        out[key] = Suggestion(g, s, m, "auto" if ok else "review")
        if ok:
            assigned[key] = g
    return out


# ---------------------------------------------------------------- ids command


def _read_tracks(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _session_videos(jdir: Path) -> dict[str, str]:
    cams = jdir / "cameras.yaml"
    if not cams.exists():
        return {}
    return {c["cam_id"]: c.get("uri", "") for c in load_yaml(cams).get("cameras", [])}


def cmd_ids(args: argparse.Namespace) -> None:
    feats: dict[TrackKey, TrackFeat] = {}
    thumbs: dict[TrackKey, list[np.ndarray]] = {}
    spans: dict[TrackKey, tuple[int, int, int]] = {}
    seeds: dict[TrackKey, int] = {}
    tables: dict[str, list[dict[str, str]]] = {}
    skip: dict[TrackKey, tuple[int | None, str]] = {}  # hand-labelled, not seeds
    for jd in map(Path, args.jsonl):
        take, ann = jd.name, Path(args.ann_root) / jd.name
        if not (ann / "tracks.csv").exists():
            raise SystemExit(f"{ann / 'tracks.csv'} missing: run label_intervals.py template")
        rows = tables[take] = _read_tracks(ann / "tracks.csv")
        for r in rows:
            key = (take, r["cam_id"], int(r["track_id"]))
            a, b, n = int(r["first_ms"]), int(r["last_ms"]), int(r["n_frames"])
            if key in spans:  # split rows of an ID-swap track: the whole track
                a0, b0, n0 = spans[key]
                a, b, n = min(a, a0), max(b, b0), n + n0
            spans[key] = (a, b, n)
            status = r.get("status", "").strip()
            gt = int(r["gt_id"]) if r["gt_id"].strip() else None
            if status in (MANUAL_SKIP, ACCEPTED, "fixed"):
                # corrections are mostly hard tracks (close-ups, partial bodies): poor colour
                # models, so they keep their label but are not seeds
                skip[key] = (gt, status)
            elif gt is not None and status not in AUTO_STATUSES:
                seeds[key] = gt
        videos = _session_videos(jd)
        (ann / "feats").mkdir(parents=True, exist_ok=True)
        for jp in sorted(jd.glob("*.jsonl")):
            cam, cache = jp.stem, ann / "feats" / f"{jp.stem}.npz"
            if cache.exists() and not args.refresh:
                f, th = load_feats(cache)
            else:
                if not Path(videos.get(cam, "")).exists():
                    print(f"{take}/{cam}: no video, skipped")
                    continue
                print(f"{take}/{cam}: decoding {videos[cam]} ...", flush=True)
                f, th = extract_cam_feats(videos[cam], jp, k=args.samples)
                save_feats(cache, f, th)
            for tid, v in f.items():
                if (take, cam, tid) in skip:
                    continue
                feats[(take, cam, tid)] = v
                thumbs[(take, cam, tid)] = th.get(tid, [])
    missing = sorted({k[1] for k in feats} - {k[1] for k in seeds})
    if not seeds:
        raise SystemExit("no seed tracks: fill gt_id of one clear track per person and camera")
    if missing:
        print(f"WARNING: no seed on {missing}: using seeds of other cameras there")
    sug = suggest_ids(
        feats,
        spans,
        seeds,
        args.min_frames,
        args.min_sim,
        args.min_margin,
        nomatch_sim=args.nomatch_sim,
    )
    counts: dict[str, int] = {}
    for take, rows in tables.items():
        ann = Path(args.ann_root) / take
        if args.write:
            bak = ann / "tracks.csv.bak"
            if not bak.exists():
                shutil.copy(ann / "tracks.csv", bak)
            with open(ann / "tracks.csv", "w", encoding="utf-8", newline="") as fo:
                w = csv.DictWriter(fo, TRACK_COLS + EXTRA_COLS, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    key = (take, r["cam_id"], int(r["track_id"]))
                    if key in sug:
                        s = sug[key]
                        r["gt_id"] = str(s.gt) if s.status == "auto" else ""
                        r["suggest"] = "" if s.gt is None else f"P{s.gt}"
                        r["sim"] = "" if s.sim is None else f"{s.sim:.3f}"
                        r["margin"] = "" if s.margin is None else f"{s.margin:.3f}"
                        r["status"] = s.status
                    w.writerow(r)
        for cam in sorted({k[1] for k in spans if k[0] == take}):
            _review_sheet(
                ann / f"review_ids_{cam}.jpg", take, cam, spans, seeds, sug, thumbs, skip
            )
    for s in sug.values():
        counts[s.status] = counts.get(s.status, 0) + 1
    print(f"seeds {len(seeds)}, suggestions {dict(sorted(counts.items()))}")
    if not args.write:
        print("dry run: tracks.csv unchanged (add --write); review sheets written")


def _review_sheet(
    path: Path,
    take: str,
    cam: str,
    spans: dict[TrackKey, tuple[int, int, int]],
    seeds: dict[TrackKey, int],
    sug: dict[TrackKey, Suggestion],
    thumbs: dict[TrackKey, list[np.ndarray]],
    swaps: dict[TrackKey, tuple[int | None, str]] | None = None,
    h: int = 96,
) -> None:
    """Tracks grouped by (suggested) person: a wrong one stands out by its colours."""
    import cv2

    keys = sorted(k for k in spans if k[:2] == (take, cam))
    if not keys:
        return
    t0 = min(spans[k][0] for k in spans if k[0] == take)

    swaps = swaps or {}

    def group(k: TrackKey) -> str:
        if k in seeds:
            return f"P{seeds[k]}"
        if k in swaps:
            return "~" if swaps[k][0] is None else f"P{swaps[k][0]}"
        s = sug.get(k)
        if s is None or s.status in ("short", "nofeat", "nomatch") or s.gt is None:
            return "~"
        return f"P{s.gt}" if s.status == "auto" else f"?P{s.gt}"

    rows = []
    for k in sorted(keys, key=lambda k: (group(k).lstrip("?"), group(k), spans[k][0])):
        s = sug.get(k)
        tag = "seed" if k in seeds else swaps[k][1] if k in swaps else (s.status if s else "")
        if tag == "short":
            continue
        info = np.full((h, 330, 3), 255, np.uint8)
        color = (0, 0, 200) if tag == "review" else (0, 0, 0)
        cv2.putText(info, f"{group(k)}  track {k[2]}", (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    color, 2)  # fmt: skip
        a, b, n = spans[k]
        detail = f"{(a - t0) / 1000:.0f}-{(b - t0) / 1000:.0f}s {tag}"
        if s and s.sim is not None:
            detail += f" {s.sim:.2f}/{s.margin:.2f}"
        cv2.putText(info, detail, (6, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1)
        rows.append(np.hstack([info, *thumbs.get(k, [])]))
    if not rows:
        return
    w = max(r.shape[1] for r in rows)
    sheet = np.vstack(
        [np.pad(r, ((0, 3), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    )
    cv2.imwrite(str(path), sheet)


# ---------------------------------------------------------------- prefill (from the script)


def resolve_spec_rows(
    rows: list[dict], person_spans: dict[int, tuple[int, int]], ref_t0_ms: int
) -> list[dict[str, str]]:
    """Script rows → intervals_src.csv rows. start/end: a clock time, `first_seen` /
    `last_seen` (span of the person's labelled tracks, all cameras, on the ref camera clock;
    only for violations that last the whole take), or TODO (watch the video)."""
    out = []
    for r in rows:
        gt = str(r.get("gt", "?")).strip()
        gid = int(gt[1:]) if gt[:1] in "Pp" and gt[1:].isdigit() else None
        times = []
        for key in ("start", "end"):
            v = str(r.get(key, "TODO")).strip()
            if v in ("first_seen", "last_seen"):
                if gid is None or gid not in person_spans:
                    v = "TODO"
                else:
                    ms = person_spans[gid][0 if v == "first_seen" else 1] - ref_t0_ms
                    # first_seen rounds up, last_seen down: stays inside the session
                    t = max(ms, 0) / 100
                    t = math.ceil(t) if v == "first_seen" else math.floor(t)
                    v = f"{t / 10:.1f}"
            times.append(v)
        out.append(
            {
                "gt_id": gt,
                "type": r["type"],
                "start": times[0],
                "end": times[1],
                "note": r.get("note", ""),
                "cams": str(r.get("cams", "")).replace(" ", ";"),
            }
        )
    return out


def cmd_prefill(args: argparse.Namespace) -> None:
    spec = load_yaml(args.spec).get("takes", {})
    for jd in map(Path, args.jsonl):
        take, ann = jd.name, Path(args.ann_root) / jd.name
        key = take.rsplit("_", 1)[-1]  # s01_t02 → t02
        if key not in spec:
            print(f"{take}: not in spec, skipped")
            continue
        dst = ann / "intervals_src.csv"
        if dst.exists() and not args.force:
            print(f"{take}: {dst} exists (use --force), skipped")
            continue
        person_spans: dict[int, tuple[int, int]] = {}
        if (ann / "tracks.csv").exists():
            for r in _read_tracks(ann / "tracks.csv"):
                if r["gt_id"].strip():
                    g, a, b = int(r["gt_id"]), int(r["first_ms"]), int(r["last_ms"])
                    lo, hi = person_spans.get(g, (a, b))
                    person_spans[g] = (min(lo, a), max(hi, b))
        t0 = next(iter(read_frames(jd / f"{args.ref}.jsonl"))).ts_ms
        rows = resolve_spec_rows(spec[key] or [], person_spans, t0)
        ann.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, ["gt_id", "type", "start", "end", "note", "cams"])
            w.writeheader()
            w.writerows(rows)
        todo = sum(1 for r in rows if "TODO" in (r["start"], r["end"]) or r["gt_id"] == "?")
        print(f"{take}: {len(rows)} rows -> {dst} ({todo} with TODO / '?' to fill)")


# ---------------------------------------------------------------- timeline contact sheets


def _zone_overlay(cam_cfg: dict, zones: list[dict]) -> list[tuple[np.ndarray, tuple]]:
    """Zone outlines in raw image px for a calibrated camera ([] otherwise)."""
    from src.geometry.homography import load_calibration
    from tools.calibrate_homography import floor_to_raw_px

    calib = cam_cfg.get("calibration")
    if not calib or not Path(calib).exists():
        return []
    cal = load_calibration(calib)
    if not cal.calibrated:
        return []
    colors = {"forbidden": (0, 0, 255), "risk": (0, 165, 255)}
    out = []
    t = np.linspace(0, 1, 50)
    for z in zones:
        if z.get("type") not in colors:
            continue
        poly = np.asarray(z["polygon_m"], dtype=np.float64)
        edge = np.vstack(
            [a + t[:, None] * (b - a) for a, b in zip(poly, np.roll(poly, -1, 0), strict=True)]
        )
        px, ok = floor_to_raw_px(cal, edge)
        if ok.all():
            out.append((np.round(px).astype(np.int32), colors[z["type"]], z["id"]))
    return out


def cmd_timeline(args: argparse.Namespace) -> None:
    import cv2

    jd, out = Path(args.jsonl), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = {c["cam_id"]: c for c in load_yaml(jd / "cameras.yaml").get("cameras", [])}
    zfile = jd / "zones.yaml" if (jd / "zones.yaml").exists() else Path("configs/zones.yaml")
    zones = load_yaml(zfile).get("zones", [])
    for cam in args.cams:
        cfg = cams[cam]
        overlay = _zone_overlay(cfg, zones)
        crop = None
        if args.crop_zone:
            hit = [pts for pts, _, zid in overlay if zid == args.crop_zone]
            if not hit:
                print(f"{cam}: zone {args.crop_zone} not drawable (no calibration?), skipped")
                continue
            x0, y0 = hit[0].min(axis=0)
            x1, y1 = hit[0].max(axis=0)
            w, h = x1 - x0, y1 - y0
            # room for a standing person above the zone, whose feet decide in / out
            crop = (int(x0 - 0.4 * w), int(y0 - 2.5 * h), int(x1 + 0.4 * w), int(y1 + 0.4 * h))
        cap = cv2.VideoCapture(cfg["uri"])
        fps = cap.get(cv2.CAP_PROP_FPS) or cfg.get("fps", 25)
        every = max(1, round(fps * args.step))
        tiles, fi = [], 0
        while True:
            if fi % every:
                if not cap.grab():
                    break
                fi += 1
                continue
            ok, img = cap.read()
            if not ok:
                break
            if fi / fps < args.t0 or (args.t1 is not None and fi / fps > args.t1):
                fi += 1
                continue
            for pts, color, zid in overlay:
                cv2.polylines(img, [pts], True, color, 4, cv2.LINE_AA)
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.putText(img, zid, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
            if crop is not None:
                H, W = img.shape[:2]
                img = img[max(crop[1], 0) : min(crop[3], H), max(crop[0], 0) : min(crop[2], W)]
            small = cv2.resize(img, (args.width, int(img.shape[0] * args.width / img.shape[1])))
            label = f"{cam} {fi / fps:.0f}s"
            for color, th in (((0, 0, 0), 5), ((255, 255, 255), 2)):
                cv2.putText(small, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, th)
            tiles.append(small)
            fi += 1
        cap.release()
        per = args.cols * args.rows
        for page in range(0, len(tiles), per):
            chunk = tiles[page : page + per]
            chunk += [np.full_like(tiles[0], 255)] * ((-len(chunk)) % args.cols)
            grid = np.vstack(
                [np.hstack(chunk[i : i + args.cols]) for i in range(0, len(chunk), args.cols)]
            )
            cv2.imwrite(str(out / f"{cam}_{page // per + 1:02d}.jpg"), grid)
        print(f"{cam}: {len(tiles)} frames every {args.step:g} s -> {out}/{cam}_*.jpg")


def cmd_accept(args: argparse.Namespace) -> None:
    """Freeze checked takes: their `auto` rows become `ok` and are no longer re-suggested."""
    for d in map(Path, args.ann):
        path = d / "tracks.csv"
        rows = _read_tracks(path)
        n = 0
        for r in rows:
            if r.get("status", "").strip() == "auto":
                r["status"], n = ACCEPTED, n + 1
        left = sum(1 for r in rows if r.get("status", "").strip() == "review")
        with open(path, "w", encoding="utf-8", newline="") as fo:
            w = csv.DictWriter(fo, TRACK_COLS + EXTRA_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        warn = f"; WARNING {left} review rows left" if left else ""
        print(f"{d.name}: {n} auto -> {ACCEPTED}{warn}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ids", help="suggest gt_id per track from seed tracks")
    i.add_argument("--jsonl", nargs="+", required=True, help="take dirs of one session")
    i.add_argument("--ann-root", required=True)
    i.add_argument("--write", action="store_true", help="write suggestions into tracks.csv")
    i.add_argument("--refresh", action="store_true", help="recompute cached features")
    i.add_argument("--samples", type=int, default=12, help="frames per track")
    i.add_argument("--min-frames", type=int, default=5, help="shorter tracks: left empty")
    i.add_argument("--min-sim", type=float, default=0.55)
    i.add_argument("--min-margin", type=float, default=0.05)
    i.add_argument("--nomatch-sim", type=float, default=0.4, help="below: like no actor")
    p = sub.add_parser("prefill", help="intervals_src.csv from the take script")
    p.add_argument("--spec", required=True)
    p.add_argument("--jsonl", nargs="+", required=True)
    p.add_argument("--ann-root", required=True)
    p.add_argument("--ref", default="cam2", help="camera whose video clock the times use")
    p.add_argument("--force", action="store_true")
    t = sub.add_parser("timeline", help="one frame per step with zones, for reading times")
    t.add_argument("--jsonl", required=True, help="take dir (cameras.yaml, zones.yaml)")
    t.add_argument("--out", required=True)
    t.add_argument("--cams", nargs="+", default=["cam2"])
    t.add_argument("--step", type=float, default=1.0, help="seconds between frames")
    t.add_argument("--width", type=int, default=480)
    t.add_argument("--cols", type=int, default=6)
    t.add_argument("--rows", type=int, default=8)
    t.add_argument("--crop-zone", help="zone id (e.g. Z2): zoom on that zone, feet in / out")
    t.add_argument("--t0", type=float, default=0.0, help="first second to show")
    t.add_argument("--t1", type=float, default=None, help="last second to show")
    a = sub.add_parser("accept", help="freeze checked takes: auto rows -> ok")
    a.add_argument("--ann", nargs="+", required=True, help="annotation dirs of checked takes")
    args = ap.parse_args(argv)
    cmds = {"ids": cmd_ids, "prefill": cmd_prefill, "timeline": cmd_timeline, "accept": cmd_accept}
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
