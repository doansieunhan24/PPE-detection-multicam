"""Convert WILDTRACK (EPFL, 7 static cams, 400 annotated frames @ 2 fps) to project formats.

Used to test association (§7.2) on real multi-view data before C2 exists.

    python -m tools.wildtrack_prepare --root data/external/wildtrack \
        --out data/external/wildtrack/prepared

Writes
  calib/C{1..7}.yaml      H image(px) → floor(m) + fov_floor_polygon_m (same format as
                          configs/calibration/*.yaml)
  gt_oracle/C{n}.jsonl    PerceptionFrame per annotated frame, persons = GT boxes
                          (track_id = GT personID: oracle detector + oracle tracker)
  gt.jsonl                per frame: GT persons with floor position (m) and per-cam boxes

Dataset facts (README.txt in the release): world unit cm, ground plane z = 0, positionID on a
480×1440 grid, X-first, 2.5 cm spacing, origin (−300, −900) cm. Calibrations: extrinsic
rvec/tvec + intrinsic_zero (images are undistorted). Frames are 10 fps; annotated every 5th.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.common.jsonl import write_frames
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PerceptionFrame, PersonDet
from src.geometry.homography import apply_homography

log = get_logger(__name__)

CAMS = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]  # view 0..6
IMG_W, IMG_H = 1920, 1080
FPS = 10
GRID_W, GRID_H, CELL_CM = 480, 1440, 2.5
ORIGIN_CM = (-300.0, -900.0)
CM_TO_M = 0.01


def cam_id(view: int) -> str:
    return f"C{view + 1}"


def pos_to_floor_m(pos_id: int) -> tuple[float, float]:
    x = ORIGIN_CM[0] + CELL_CM * (pos_id % GRID_W)
    y = ORIGIN_CM[1] + CELL_CM * (pos_id // GRID_W)
    return x * CM_TO_M, y * CM_TO_M


def grid_bounds_m() -> tuple[float, float, float, float]:
    x0, y0 = ORIGIN_CM
    return (
        x0 * CM_TO_M,
        y0 * CM_TO_M,
        (x0 + CELL_CM * (GRID_W - 1)) * CM_TO_M,
        (y0 + CELL_CM * (GRID_H - 1)) * CM_TO_M,
    )


def _floats(node: ET.Element) -> list[float]:
    return [float(v) for v in node.text.split()]


def load_camera(root: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K (3×3), R (3×3), t (3, cm)."""
    ex = ET.parse(root / "calibrations" / "extrinsic" / f"extr_{name}.xml").getroot()
    rvec = np.array(_floats(ex.find("rvec")))
    t = np.array(_floats(ex.find("tvec")))
    intr = ET.parse(root / "calibrations" / "intrinsic_zero" / f"intr_{name}.xml").getroot()
    K = np.array(_floats(intr.find("camera_matrix").find("data"))).reshape(3, 3)
    R, _ = cv2.Rodrigues(rvec)
    return K, R, t


def image_to_floor_h(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Plane z=0: x_img ~ K [r1 r2 t] [X Y 1]^T (cm). Invert, then scale cm → m."""
    h_w2i = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H = np.diag([CM_TO_M, CM_TO_M, 1.0]) @ np.linalg.inv(h_w2i)
    return H / H[2, 2]


def fov_polygon_m(K: np.ndarray, R: np.ndarray, t: np.ndarray, step_m: float) -> list:
    """Convex hull of grid floor points that project inside the image, in front of the camera.
    Purely geometric (no GT), like a polygon drawn during calibration."""
    x0, y0, x1, y1 = grid_bounds_m()
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1e-9, step_m), np.arange(y0, y1 + 1e-9, step_m))
    pts_m = np.column_stack([xs.ravel(), ys.ravel()])
    pts_cm = np.column_stack([pts_m / CM_TO_M, np.zeros(len(pts_m))])
    cam = pts_cm @ R.T + t
    front = cam[:, 2] > 0
    uvw = cam @ K.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
    if inside.sum() < 3:
        return []
    hull = cv2.convexHull(pts_m[inside].astype(np.float32)).reshape(-1, 2)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in hull]


def clip_box(v: dict) -> tuple[float, float, float, float] | None:
    if v["xmin"] < 0 and v["xmax"] < 0:  # -1 everywhere = not visible in this view
        return None
    x1, y1 = max(0, v["xmin"]), max(0, v["ymin"])
    x2, y2 = min(IMG_W - 1, v["xmax"]), min(IMG_H - 1, v["ymax"])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return float(x1), float(y1), float(x2), float(y2)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("data/external/wildtrack"))
    ap.add_argument("--out", type=Path, default=Path("data/external/wildtrack/prepared"))
    ap.add_argument("--fov-step-m", type=float, default=0.1)
    ap.add_argument("--truncated-margin-px", type=int, default=2)
    args = ap.parse_args(argv)
    setup_logging()

    (args.out / "calib").mkdir(parents=True, exist_ok=True)
    (args.out / "gt_oracle").mkdir(parents=True, exist_ok=True)

    Hs = {}
    for v, name in enumerate(CAMS):
        K, R, t = load_camera(args.root, name)
        H = image_to_floor_h(K, R, t)
        Hs[cam_id(v)] = H
        fov = fov_polygon_m(K, R, t, args.fov_step_m)
        doc = {
            "cam_id": cam_id(v),
            "source": f"WILDTRACK {name} (extrinsic + intrinsic_zero)",
            "H": H.tolist(),
            "reprojection_error_m": None,  # filled below from GT foot points
            "fov_floor_polygon_m": fov,
        }
        (args.out / "calib" / f"{cam_id(v)}.yaml").write_text(
            yaml.safe_dump(doc, sort_keys=False), encoding="utf-8"
        )

    frames: dict[str, list[PerceptionFrame]] = {cam_id(v): [] for v in range(len(CAMS))}
    foot_err: dict[str, list[float]] = {c: [] for c in frames}
    ann_files = sorted((args.root / "annotations_positions").glob("*.json"))
    with open(args.out / "gt.jsonl", "w", encoding="utf-8") as fg:
        for f in ann_files:
            frame_idx = int(f.stem)
            ts_ms = frame_idx * 1000 // FPS
            persons_gt = []
            per_cam: dict[str, list[PersonDet]] = {c: [] for c in frames}
            for a in json.loads(f.read_text(encoding="utf-8")):
                xy = pos_to_floor_m(a["positionID"])
                boxes = {}
                for vw in a["views"]:
                    b = clip_box(vw)
                    if b is None:
                        continue
                    c = cam_id(vw["viewNum"])
                    boxes[c] = list(b)
                    per_cam[c].append(
                        PersonDet(
                            track_id=a["personID"],
                            bbox=b,
                            conf=1.0,
                            ppe_checked=False,
                            truncated_bottom=b[3] >= IMG_H - 1 - args.truncated_margin_px,
                        )
                    )
                    if vw["ymax"] < IMG_H - 1:  # bottom-centre is a foot point only if visible
                        fp = np.array([[(b[0] + b[2]) / 2, b[3]]])
                        foot_err[c].append(
                            float(np.linalg.norm(apply_homography(Hs[c], fp)[0] - xy))
                        )
                persons_gt.append({"gt_id": a["personID"], "xy_m": list(xy), "boxes": boxes})
            fg.write(
                json.dumps({"ts_ms": ts_ms, "frame_idx": frame_idx, "persons": persons_gt}) + "\n"
            )
            for c, ps in per_cam.items():
                frames[c].append(
                    PerceptionFrame(
                        cam_id=c,
                        frame_idx=frame_idx,
                        ts_ms=ts_ms,
                        img_w=IMG_W,
                        img_h=IMG_H,
                        persons=ps,
                    )
                )

    for c, frs in frames.items():
        write_frames(args.out / "gt_oracle" / f"{c}.jsonl", frs)
        e = np.array(foot_err[c])
        path = args.out / "calib" / f"{c}.yaml"
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        # No calibration point pairs in the release; report how far GT box bottom-centres land
        # from the GT floor position instead (calibration + annotation error together).
        doc["reprojection_error_m"] = round(float(np.sqrt(np.mean(e**2))), 4) if len(e) else None
        doc["gt_footpoint_median_err_m"] = round(float(np.median(e)), 4) if len(e) else None
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        log.info(
            "%s: %d frames, %d boxes, GT foot→floor RMS %.3f m, median %.3f m, FOV %d vertices",
            c,
            len(frs),
            sum(len(fr.persons) for fr in frs),
            doc["reprojection_error_m"] or -1,
            doc["gt_footpoint_median_err_m"] or -1,
            len(doc["fov_floor_polygon_m"]),
        )


if __name__ == "__main__":
    main()
