"""Throughput vs number of streams (CLAUDE.md §10.3-6), offline (Ultralytics) backend.

N streams are served round-robin in one process, one model instance (own tracker) per
stream, as a live deployment of the offline backend would. Measures perception fps
(total and per stream), per-frame latency percentiles and peak GPU memory, then analytics
throughput (sync groups/s) on the produced frames. The DeepStream backend is benchmarked
separately once P0 runs (TODO).

    python -m eval.bench_fps --videos data/raw/samples/vtest.avi --streams 1 2 3 4 \
        --frames 300 --out eval/reports/bench_fps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.analytics_runner import git_commit
from src.association.ground_plane import sync_groups
from src.common.config import DEFAULT_CONFIG_DIR, load_config, load_yaml
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PerceptionFrame
from src.fusion.engine import PPEEngine

log = get_logger(__name__)


def _read_frames(path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"cannot read {path}")
    return frames


def bench_streams(
    perception_cfg: dict, clips: list[list[np.ndarray]], n_frames: int, warmup: int
) -> tuple[dict, dict[str, list[PerceptionFrame]]]:
    import torch

    from src.perception_offline.ultralytics_runner import UltralyticsPerception

    models = [UltralyticsPerception(perception_cfg) for _ in clips]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    lat_ms: list[float] = []
    out: dict[str, list[PerceptionFrame]] = {f"cam{i + 1}": [] for i in range(len(clips))}
    t_start = None
    for k in range(warmup + n_frames):
        if k == warmup:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()
        for i, (m, clip) in enumerate(zip(models, clips, strict=True)):
            img = clip[k % len(clip)]
            t0 = time.perf_counter()
            persons = m.process(img)
            if k >= warmup:
                lat_ms.append((time.perf_counter() - t0) * 1000)
                out[f"cam{i + 1}"].append(
                    PerceptionFrame(
                        cam_id=f"cam{i + 1}",
                        frame_idx=k - warmup,
                        ts_ms=(k - warmup) * 40,
                        img_w=img.shape[1],
                        img_h=img.shape[0],
                        persons=persons,
                    )
                )
    dt = time.perf_counter() - t_start
    total = n_frames * len(clips)
    lat = np.array(lat_ms)
    res = {
        "streams": len(clips),
        "fps_total": round(total / dt, 2),
        "fps_per_stream": round(n_frames / dt, 2),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "persons_per_frame": round(sum(len(f.persons) for v in out.values() for f in v) / total, 2),
        "gpu_peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if torch.cuda.is_available()
        else None,
    }
    return res, out


def bench_analytics(streams: dict[str, list[PerceptionFrame]], config_dir: Path) -> dict:
    """Analytics only (no calibration: association degenerates to per-camera singletons)."""
    engine = PPEEngine(
        load_config("fusion", config_dir),
        load_config("zones", config_dir),
        {},
        "P",
        {},
        load_config("alerts", config_dir).get("levels", {}),
    )
    groups = list(sync_groups(streams, 20))
    t0 = time.perf_counter()
    for g in groups:
        engine.step(g, min(f.ts_ms for f in g.values()))
    dt = time.perf_counter() - t0
    return {"groups": len(groups), "groups_per_s": round(len(groups) / max(dt, 1e-9), 1)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--videos", nargs="+", required=True, help="reused cyclically per stream")
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--frames", type=int, default=300, help="timed frames per stream")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--perception", default=str(DEFAULT_CONFIG_DIR / "perception.yaml"))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--target-fps", type=float, default=25.0, help="real-time threshold")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    setup_logging()

    perception_cfg = load_yaml(args.perception)
    n_read = min(args.frames + args.warmup, 500)
    clips_all = [_read_frames(v, n_read) for v in args.videos]
    rows = []
    for n in args.streams:
        clips = [clips_all[i % len(clips_all)] for i in range(n)]
        res, frames = bench_streams(perception_cfg, clips, args.frames, args.warmup)
        res.update(bench_analytics(frames, Path(args.config)))
        res["real_time"] = res["fps_per_stream"] >= args.target_fps
        rows.append(res)
        log.info("%s", res)

    import torch

    meta = {
        "commit": git_commit(),
        "videos": args.videos,
        "resolution": list(clips_all[0][0].shape[1::-1]),
        "pose": {k: perception_cfg["pose"][k] for k in ("model", "imgsz")},
        "ppe": {k: perception_cfg["ppe"].get(k) for k in ("enabled", "model", "imgsz")},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "frames": args.frames,
        "target_fps": args.target_fps,
    }
    lines = [
        f"<!-- generated by eval/bench_fps.py @ {meta['commit']} -->",
        f"GPU {meta['gpu']}, {meta['resolution'][0]}x{meta['resolution'][1]}, "
        f"pose {meta['pose']['model']}@{meta['pose']['imgsz']}, "
        f"PPE {'on' if meta['ppe']['enabled'] else 'off'}",
        "",
        "| Streams | fps total | fps / stream | latency p50 / p95 (ms) | persons/frame | "
        "GPU peak (MB) | analytics groups/s | ≥ target |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['streams']} | {r['fps_total']} | {r['fps_per_stream']} | "
            f"{r['latency_ms_p50']} / {r['latency_ms_p95']} | {r['persons_per_frame']} | "
            f"{r['gpu_peak_mem_mb']} | {r['groups_per_s']} | {'yes' if r['real_time'] else 'no'} |"
        )
    print("\n".join(lines))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "results.json").write_text(
            json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8"
        )
        (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
