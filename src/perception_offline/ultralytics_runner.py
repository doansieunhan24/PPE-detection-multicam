"""Offline perception backend (CLAUDE.md D5): Ultralytics YOLO-pose + tracker (+ PPE on crops).

Writes one JSONL file per camera (<out>/<cam_id>.jsonl), same PerceptionFrame schema as the
DeepStream backend. Cameras are processed one after another, each with its own model instance
so tracker IDs stay per-camera (D1). For live RTSP sources run one process per camera (--cams).

    python -m src.perception_offline.ultralytics_runner --cameras configs/cameras.yaml \
        --out data/jsonl/session01/ --source cam1=data/raw/session01/cam1.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.config import DEFAULT_CONFIG_DIR, load_yaml
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PerceptionFrame, PersonDet
from src.parts.ppe_assign import containment
from src.perception_offline.convert import (
    file_ts_ms,
    is_live_uri,
    pad_clip_bbox,
    persons_from_arrays,
    ppe_from_arrays,
)

log = get_logger(__name__)


class UltralyticsPerception:
    """Per-camera perception: pose + tracking, optional PPE detector on person crops."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from ultralytics import YOLO  # heavy import, only when actually running

        self.cfg = cfg
        self.device = cfg.get("device")
        self.pose_cfg = cfg["pose"]
        self.ppe_cfg = cfg["ppe"]
        self.margin_px = cfg.get("truncated_bottom_margin_px", 2)
        self.pose = YOLO(self.pose_cfg["model"])
        self.ppe = YOLO(self.ppe_cfg["model"]) if self.ppe_cfg.get("enabled") else None

    def process(self, frame: np.ndarray, track: bool = True) -> list[PersonDet]:
        """track=False: independent still images — detections get IDs 0..N-1."""
        img_h, img_w = frame.shape[:2]
        kw = dict(
            imgsz=self.pose_cfg["imgsz"],
            conf=self.pose_cfg["conf"],
            iou=self.pose_cfg["iou"],
            classes=[0],
            device=self.device,
            verbose=False,
        )
        if track:
            res = self.pose.track(frame, persist=True, tracker=self.pose_cfg["tracker"], **kw)[0]
        else:
            res = self.pose.predict(frame, **kw)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        if track:
            ids = None if boxes.id is None else boxes.id.int().cpu().numpy()
        else:
            ids = np.arange(len(boxes))
        persons = persons_from_arrays(
            xyxy=boxes.xyxy.cpu().numpy(),
            conf=boxes.conf.cpu().numpy(),
            track_ids=ids,
            kpts=None if res.keypoints is None else res.keypoints.data.cpu().numpy(),
            img_h=img_h,
            truncated_margin_px=self.margin_px,
        )
        self._run_ppe(frame, persons, img_w, img_h)
        return persons

    def _run_ppe(self, frame: np.ndarray, persons: list[PersonDet], img_w: int, img_h: int) -> None:
        if self.ppe is None:
            for p in persons:
                p.ppe_checked = False
            return
        c = self.ppe_cfg
        if c.get("mode", "crop") == "full":
            self._run_ppe_full(frame, persons, img_w, img_h)
            return
        crops, owners, origins = [], [], []
        for p in persons:
            x1, y1, x2, y2 = (
                int(round(v)) for v in pad_clip_bbox(p.bbox, c["crop_pad"], img_w, img_h)
            )
            if x2 - x1 < c["min_crop_w_px"] or y2 - y1 < c["min_crop_h_px"]:
                p.ppe_checked = False
                continue
            crops.append(frame[y1:y2, x1:x2])
            owners.append(p)
            origins.append((x1, y1))
        if not crops:
            return
        results = self.ppe.predict(
            crops, imgsz=c["imgsz"], conf=c["conf"], iou=c["iou"], device=self.device, verbose=False
        )
        for p, origin, r in zip(owners, origins, results, strict=True):
            b = r.boxes
            p.ppe_dets = ppe_from_arrays(
                xyxy=b.xyxy.cpu().numpy(),
                conf=b.conf.cpu().numpy(),
                cls_idx=b.cls.int().cpu().numpy(),
                names=r.names,
                class_map=c.get("class_map") or {},
                crop_origin=origin,
            )

    def _run_ppe_full(
        self, frame: np.ndarray, persons: list[PersonDet], img_w: int, img_h: int
    ) -> None:
        """Baseline (§10.3-5): detector on the whole frame; each box goes to the persons whose
        padded bbox contains it (containment >= 0.5)."""
        c = self.ppe_cfg
        r = self.ppe.predict(
            frame, imgsz=c["imgsz"], conf=c["conf"], iou=c["iou"], device=self.device, verbose=False
        )[0]
        b = r.boxes
        dets = ppe_from_arrays(
            xyxy=b.xyxy.cpu().numpy(),
            conf=b.conf.cpu().numpy(),
            cls_idx=b.cls.int().cpu().numpy(),
            names=r.names,
            class_map=c.get("class_map") or {},
            crop_origin=(0, 0),
        )
        for p in persons:
            region = pad_clip_bbox(p.bbox, c["crop_pad"], img_w, img_h)
            p.ppe_dets = [d for d in dets if containment(d.bbox, region) >= 0.5]


def run_camera(
    cam: dict[str, Any],
    uri: str,
    perception_cfg: dict[str, Any],
    out_path: Path,
    start_ms: int,
    max_frames: int | None,
) -> int:
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source for {cam['cam_id']}: {uri}")
    live = is_live_uri(uri)
    fps = cap.get(cv2.CAP_PROP_FPS) or cam.get("fps")
    if not live and not fps:
        raise RuntimeError(f"unknown fps for {uri}; set fps in cameras.yaml")
    offset_ms = int(cam.get("time_offset_ms", 0))
    model = UltralyticsPerception(perception_cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n, t0 = 0, time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as f:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            ts_ms = (
                int(time.time() * 1000) + offset_ms
                if live
                else file_ts_ms(n, fps, start_ms, offset_ms)
            )
            img_h, img_w = frame.shape[:2]
            pf = PerceptionFrame(
                cam_id=cam["cam_id"],
                frame_idx=n,
                ts_ms=ts_ms,
                img_w=img_w,
                img_h=img_h,
                persons=model.process(frame),
            )
            f.write(pf.model_dump_json() + "\n")
            n += 1
            if n % 500 == 0:
                log.info(
                    "%s: %d frames, %.1f fps", cam["cam_id"], n, n / (time.perf_counter() - t0)
                )
    cap.release()
    dt = time.perf_counter() - t0
    log.info(
        "%s: done %d frames in %.1fs (%.1f fps) → %s",
        cam["cam_id"],
        n,
        dt,
        n / max(dt, 1e-9),
        out_path,
    )
    return n


def _parse_sources(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        cam_id, sep, uri = it.partition("=")
        if not sep:
            raise SystemExit(f"--source must be cam_id=uri, got {it!r}")
        out[cam_id] = uri
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cameras", default=str(DEFAULT_CONFIG_DIR / "cameras.yaml"))
    ap.add_argument("--perception", default=str(DEFAULT_CONFIG_DIR / "perception.yaml"))
    ap.add_argument("--out", required=True, help="output dir; writes <cam_id>.jsonl")
    ap.add_argument(
        "--source", action="append", default=[], help="override uri: cam_id=path_or_rtsp"
    )
    ap.add_argument(
        "--cams", nargs="*", help="subset of cam_ids (default: all, or those in --source)"
    )
    ap.add_argument(
        "--start-ms", type=int, default=0, help="session start epoch ms for file sources"
    )
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    setup_logging()
    cams_cfg = load_yaml(args.cameras)
    perception_cfg = load_yaml(args.perception)
    sources = _parse_sources(args.source)

    wanted = args.cams or (list(sources) if sources else None)
    cams = [c for c in cams_cfg["cameras"] if wanted is None or c["cam_id"] in wanted]
    missing = set(wanted or []) - {c["cam_id"] for c in cams}
    if missing:
        raise SystemExit(f"cam_id not in {args.cameras}: {sorted(missing)}")

    out_dir = Path(args.out)
    for cam in cams:
        uri = sources.get(cam["cam_id"], cam["uri"])
        run_camera(
            cam,
            uri,
            perception_cfg,
            out_dir / f"{cam['cam_id']}.jsonl",
            args.start_ms,
            args.max_frames,
        )


if __name__ == "__main__":
    main()
