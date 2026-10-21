"""Replay a PerceptionFrame JSONL on its video: draw bbox, track ID, skeleton, PPE boxes.

    python tools/replay.py --jsonl data/jsonl/session01/cam1.jsonl \
        --video data/raw/session01/cam1.mp4 --out data/jsonl/session01/cam1_vis.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.jsonl import read_frames  # noqa: E402
from src.common.logging import get_logger, setup_logging  # noqa: E402
from src.common.schemas import PersonDet  # noqa: E402

log = get_logger(__name__)

# COCO-17 skeleton edges
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]  # fmt: skip
PPE_COLORS = {"helmet": (0, 255, 255), "vest": (0, 165, 255), "boots": (255, 0, 255)}


def _color(track_id: int) -> tuple[int, int, int]:
    return ((track_id * 67) % 255, (track_id * 131) % 255, (track_id * 197) % 255)


def draw_person(img, p: PersonDet, kpt_conf_min: float) -> None:
    col = _color(p.track_id)
    x1, y1, x2, y2 = (int(v) for v in p.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    label = f"id{p.track_id} {p.conf:.2f}" + ("" if p.ppe_checked else " noPPE?")
    cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    if p.kpts:
        for a, b in SKELETON:
            ka, kb = p.kpts[a], p.kpts[b]
            if ka[2] >= kpt_conf_min and kb[2] >= kpt_conf_min:
                cv2.line(img, (int(ka[0]), int(ka[1])), (int(kb[0]), int(kb[1])), col, 2)
        for x, y, c in p.kpts:
            if c >= kpt_conf_min:
                cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)
    for d in p.ppe_dets:
        dc = PPE_COLORS.get(d.cls, (200, 200, 200))
        bx1, by1, bx2, by2 = (int(v) for v in d.bbox)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), dc, 1)
        cv2.putText(
            img, f"{d.cls} {d.conf:.2f}", (bx1, by2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, dc, 1
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kpt-conf-min", type=float, default=0.5)
    args = ap.parse_args(argv)
    setup_logging()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    n = 0
    for fr in read_frames(args.jsonl):
        if fr.frame_idx != n:  # JSONL skipped frames → seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr.frame_idx)
        ok, img = cap.read()
        if not ok:
            break
        n = fr.frame_idx + 1
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, fps, (img.shape[1], img.shape[0]))
        for p in fr.persons:
            draw_person(img, p, args.kpt_conf_min)
        cv2.putText(img, f"{fr.cam_id} f{fr.frame_idx} t={fr.ts_ms}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # fmt: skip
        writer.write(img)
    cap.release()
    if writer is not None:
        writer.release()
    log.info("wrote %s (%d frames)", args.out, n)


if __name__ == "__main__":
    main()
