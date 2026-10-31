"""Calibrate image→floor homography for one camera (CLAUDE.md §7.1, P3).

1) Click floor markers on a frame (writes/extends a CSV with img_x,img_y; you fill floor_x,floor_y
   in metres measured on site):
       python tools/calibrate_homography.py click --image cam1.jpg \
           --points configs/calibration/cam1_points.csv
2) Fit H (RANSAC, >= 4 points, 6–10 recommended) and write the YAML used by analytics:
       python tools/calibrate_homography.py fit --cam cam1 \
           --points configs/calibration/cam1_points.csv \
           --image cam1.jpg --out configs/calibration/cam1.yaml
   The camera FOV on the floor is the image border mapped through H: parts above the horizon
   are dropped, the rest is limited to --fov-max-m from the bottom-centre floor point (use
   about the room size, e.g. 8 for the C2 room). Reports in-sample RMS and leave-one-out
   error (>= 5 points); the leave-one-out number is the one to put in the thesis.

Tiled floor (docs/c2_shooting_script.md §1.3): instead of floor_x/floor_y, fill tile_i/tile_j
(tile-corner index from the chosen origin) and pass --tile-m 0.6 (tile size in metres).

Black/white checkerboard floor (C2 lobby): click the 4 corners of ONE tile, going around it
(corner 1 = floor origin, 1→2 = +x, 1→4 = +y); every visible tile corner is then found
automatically (src/geometry/floor_grid.py) and written with tile_i/tile_j:
       python tools/calibrate_homography.py grid --image cam2.jpg \
           --points configs/calibration/cam2_points.csv \
           --figure docs/figures/calibration/cam2_grid.jpg
   (--seed x,y x,y x,y x,y instead of clicking). Fit with the wide-angle lens distortion
   (k1, k2 estimated so that the corners lie on one plane), then the report figures:
       python tools/calibrate_homography.py fit --cam cam2 --points ... --tile-m <m> \
           --undistort --image cam2.jpg --out configs/calibration/cam2.yaml
       python tools/calibrate_homography.py figures --calib configs/calibration/cam2.yaml \
           --image cam2.jpg --points ... --tile-m <m> --out-dir docs/figures/calibration
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_yaml  # noqa: E402
from src.geometry.floor_grid import (  # noqa: E402
    distort_px,
    fit_radial_distortion,
    grow_grid,
    nominal_K,
    undistort_px,
)
from src.geometry.homography import (  # noqa: E402
    CameraCalibration,
    apply_homography,
    fit_homography,
    floor_fov_polygon,
    load_calibration,
    loo_errors,
    point_in_polygon,
)


def read_points(path: Path, tile_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """img_x,img_y + floor_x,floor_y (m), or tile_i,tile_j × tile_m. Rows without both skipped."""
    img, floor = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            fx, fy = r.get("floor_x") or "", r.get("floor_y") or ""
            ti, tj = r.get("tile_i") or "", r.get("tile_j") or ""
            if fx.strip() and fy.strip():
                xy = (float(fx), float(fy))
            elif ti.strip() and tj.strip():
                if tile_m is None:
                    raise SystemExit(f"{path}: tile_i/tile_j given, pass --tile-m")
                xy = (float(ti) * tile_m, float(tj) * tile_m)
            else:
                continue
            img.append((float(r["img_x"]), float(r["img_y"])))
            floor.append(xy)
    return np.array(img), np.array(floor)


def cmd_click(args: argparse.Namespace) -> None:
    import cv2

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    pts: list[tuple[int, int]] = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                img, str(len(pts)), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

    cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calib", on_mouse)
    print("Left-click floor markers in order; press q/Esc to finish.")
    while True:
        cv2.imshow("calib", img)
        if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    out = Path(args.points)
    new = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "img_x", "img_y", "floor_x", "floor_y", "tile_i", "tile_j"])
        for i, (x, y) in enumerate(pts, 1):
            w.writerow([i, x, y, "", "", "", ""])
    print(
        f"{len(pts)} points -> {out}. Fill floor_x/floor_y (metres) or tile_i/tile_j, "
        "then run 'fit'."
    )


def draw_corners(img: np.ndarray, found: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    import cv2

    for (i, j), p in found.items():
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(img, (x, y), 7, (0, 0, 255), -1)
        for color, thick in (((0, 0, 0), 4), ((0, 255, 0), 2)):
            cv2.putText(
                img, f"{i},{j}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thick
            )
    return img


def cmd_grid(args: argparse.Namespace) -> None:
    import cv2

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    if args.seed:
        seeds = [tuple(float(v) for v in s.split(",")) for s in args.seed]
    else:
        seeds = []
        view = img.copy()

        def on_mouse(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and len(seeds) < 4:
                seeds.append((float(x), float(y)))
                cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(view, str(len(seeds)), (x + 8, y - 8), 0, 1.0, (0, 0, 255), 2)

        cv2.namedWindow("grid", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("grid", on_mouse)
        print("Click the 4 corners of one tile, going around it; q/Esc when done.")
        while True:
            cv2.imshow("grid", view)
            if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
    if len(seeds) != 4:
        raise SystemExit(f"need 4 seed corners, got {len(seeds)}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = grow_grid(gray, seeds, min_contrast=args.min_contrast)
    out = Path(args.points)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "img_x", "img_y", "floor_x", "floor_y", "tile_i", "tile_j"])
        for k, ((i, j), p) in enumerate(sorted(found.items()), 1):
            w.writerow([k, round(float(p[0]), 2), round(float(p[1]), 2), "", "", i, j])
    seed_txt = " ".join(f"{x:.0f},{y:.0f}" for x, y in seeds)
    print(f"{len(found)} tile corners -> {out} (seeds: {seed_txt})")
    if args.figure:
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.figure, draw_corners(img.copy(), found))
        print(f"figure -> {args.figure}")


def cmd_fit(args: argparse.Namespace) -> None:
    img_raw, floor_pts = read_points(Path(args.points), args.tile_m)
    K = dist = None
    lens: dict = {}
    if args.undistort:
        import cv2

        h, w = cv2.imread(args.image).shape[:2] if args.image else (args.height, args.width)
        K = nominal_K((w, h), args.f_ratio)
        dist, info = fit_radial_distortion([(floor_pts, img_raw)], K)
        lens = {
            "K": [[float(v) for v in row] for row in K],
            "dist": [float(v) for v in dist],
            "plane_fit_rms_px_raw": round(info["rms_px_before"][0], 2),
            "plane_fit_rms_px_undistorted": round(info["rms_px_after"][0], 2),
        }
        print(
            f"lens: k1={dist[0]:.4f} k2={dist[1]:.4f} (f fixed {K[0, 0]:.0f} px); corners vs "
            f"plane {info['rms_px_before'][0]:.2f} px raw -> "
            f"{info['rms_px_after'][0]:.2f} px undistorted"
        )
    img_pts = undistort_px(img_raw, K, dist)
    H, rms = fit_homography(img_pts, floor_pts)
    per_pt = np.linalg.norm(apply_homography(H, img_pts) - floor_pts, axis=1)
    loo = loo_errors(img_pts, floor_pts) if len(img_pts) >= 5 else None
    fov: list[list[float]] = []
    if args.image:
        import cv2

        h, w = cv2.imread(args.image).shape[:2]
        fov = floor_fov_polygon(H, w, h, img_raw, args.fov_max_m, args.fov_top, K=K, dist=dist)
    data = {
        "cam_id": args.cam,
        "H": [[float(v) for v in row] for row in H],
        **lens,
        "reprojection_error_m": round(rms, 4),
        "loo_error_rms_m": None if loo is None else round(float(np.sqrt(np.nanmean(loo**2))), 4),
        "loo_error_max_m": None if loo is None else round(float(np.nanmax(loo)), 4),
        "n_points": int(len(img_pts)),
        "fov_floor_polygon_m": fov,
    }
    Path(args.out).write_text(
        "# Generated by tools/calibrate_homography.py\n" + yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"RMS reprojection error: {rms:.3f} m (max {per_pt.max():.3f} m) over {len(img_pts)} points"
    )
    if loo is not None:
        worst = int(np.nanargmax(loo))
        print(
            f"leave-one-out: RMS {data['loo_error_rms_m']:.3f} m, max {loo[worst]:.3f} m "
            f"(point #{worst + 1})"
        )
    else:
        print("leave-one-out: needs >= 5 points")
    if loo is not None and args.undistort:
        # same points, plain homography on raw pixels: the lens model's effect for the thesis.
        # (with --undistort, k1/k2 are fitted on all points, so its leave-one-out is slightly
        # optimistic; H itself is still left out per point)
        _, rms0 = fit_homography(img_raw, floor_pts)
        loo0 = loo_errors(img_raw, floor_pts)
        print(
            f"without lens model: RMS {rms0:.3f} m, leave-one-out RMS "
            f"{float(np.sqrt(np.nanmean(loo0**2))):.3f} m, max {float(np.nanmax(loo0)):.3f} m"
        )
    print(f"-> {args.out}")
    if max(rms, data["loo_error_rms_m"] or 0.0) > 0.2:
        print("WARNING: error > 0.2 m (P3 target). Check point order / measurements.")
    if args.image and not fov:
        print("WARNING: empty FOV polygon; check --fov-max-m and the markers")


def floor_to_raw_px(cal: CameraCalibration, floor_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Floor (m) → raw image px through H⁻¹ and the lens model; ok = in front of the camera."""
    Hinv = np.linalg.inv(cal.H)
    hom = np.hstack([floor_m, np.ones((len(floor_m), 1))]) @ Hinv.T
    ok = hom[:, 2] * np.sign(np.linalg.det(Hinv)) > 1e-9
    und = hom[:, :2] / np.where(np.abs(hom[:, 2:3]) < 1e-12, 1e-12, hom[:, 2:3])
    return distort_px(und, cal.K, cal.dist), ok


def cmd_figures(args: argparse.Namespace) -> None:
    """Report figures: raw vs undistorted frame, floor grid reprojected on the raw frame (with
    the calibration points and their errors), bird's-eye view of the floor."""
    import cv2

    cal = load_calibration(args.calib)
    if not cal.calibrated:
        raise SystemExit(f"{args.calib}: no H")
    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cam = cal.cam_id
    written = []

    if cal.K is not None:
        und = cv2.undistort(img, cal.K, cal.dist)
        cv2.imwrite(str(out / f"{cam}_undistorted.jpg"), und)
        pair = np.hstack([img, und])
        for x, label in ((20, "raw"), (w + 20, "undistorted")):
            cv2.putText(pair, label, (x, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        cv2.imwrite(str(out / f"{cam}_lens_compare.jpg"), cv2.resize(pair, (w, h // 2)))
        written += [f"{cam}_undistorted.jpg", f"{cam}_lens_compare.jpg"]

    img_pts = floor_pts = None
    if args.points:
        img_pts, floor_pts = read_points(Path(args.points), args.tile_m)
    step = args.grid_m or args.tile_m or 1.0
    if floor_pts is not None and len(floor_pts):
        lo, hi = floor_pts.min(0) - 4 * step, floor_pts.max(0) + 4 * step
    else:
        poly = np.asarray(cal.fov_floor_polygon_m)
        lo, hi = poly.min(0), poly.max(0)
    fov = cal.fov_floor_polygon_m

    def inside(p: np.ndarray) -> bool:
        return not fov or point_in_polygon((float(p[0]), float(p[1])), fov)

    grid_img = img.copy()
    t = np.linspace(0.0, 1.0, 200)
    lines = [np.c_[np.full_like(t, x), lo[1] + t * (hi[1] - lo[1])]
             for x in np.arange(np.floor(lo[0] / step) * step, hi[0] + 1e-9, step)]  # fmt: skip
    lines += [np.c_[lo[0] + t * (hi[0] - lo[0]), np.full_like(t, y)]
              for y in np.arange(np.floor(lo[1] / step) * step, hi[1] + 1e-9, step)]  # fmt: skip
    for ln in lines:
        px, ok = floor_to_raw_px(cal, ln)
        ok &= np.array([inside(p) for p in ln])
        for a, b, k in zip(px[:-1], px[1:], ok[:-1] & ok[1:], strict=True):
            if k:
                cv2.line(grid_img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                         (255, 200, 0), 2, cv2.LINE_AA)  # fmt: skip
    if img_pts is not None:
        reproj, _ = floor_to_raw_px(cal, floor_pts)
        for p, q in zip(img_pts, reproj, strict=True):
            cv2.circle(grid_img, tuple(np.round(p).astype(int)), 7, (0, 0, 255), 2)
            cv2.circle(grid_img, tuple(np.round(q).astype(int)), 3, (0, 255, 0), -1)
    cv2.putText(grid_img, "floor grid from calibration (cyan), clicked/detected corners (red), "
                "reprojected (green)", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)  # fmt: skip
    cv2.imwrite(str(out / f"{cam}_floor_grid.jpg"), grid_img)
    written.append(f"{cam}_floor_grid.jpg")

    zones = load_yaml(args.zones).get("zones", []) if args.zones else []
    if zones:
        colors = {"forbidden": (0, 0, 255), "risk": (0, 165, 255), "ppe_required": (0, 200, 0)}
        zimg, fill = img.copy(), img.copy()
        for z in reversed(zones):  # first zone wins in analytics → drawn on top
            poly = np.asarray(z["polygon_m"], dtype=np.float64)
            nxt = np.roll(poly, -1, 0)
            edge = np.vstack([a + t[:, None] * (b - a) for a, b in zip(poly, nxt, strict=True)])
            px, ok = floor_to_raw_px(cal, edge)
            pts = np.round(px[ok]).astype(np.int32)
            color = colors.get(z["type"], (255, 255, 255))
            cv2.fillPoly(fill, [pts], color)
            cv2.polylines(zimg, [pts], True, color, 3, cv2.LINE_AA)
        zimg = cv2.addWeighted(fill, 0.3, zimg, 0.7, 0)
        for k, z in enumerate(zones):  # legend (a centroid label can fall in another zone)
            y = 50 + 45 * k
            cv2.rectangle(zimg, (20, y - 25), (55, y + 5), colors.get(z["type"], (255,) * 3), -1)
            label = f"{z['id']} {z['type']} ({', '.join(z.get('required_ppe', []))})"
            for color, thick in (((0, 0, 0), 5), ((255, 255, 255), 2)):
                cv2.putText(zimg, label, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thick)
        cv2.imwrite(str(out / f"{cam}_zones.jpg"), zimg)
        written.append(f"{cam}_zones.jpg")

    # bird's-eye view over the calibrated area (floor x right, y down)
    s = args.px_per_m
    bw, bh = int((hi[0] - lo[0]) * s), int((hi[1] - lo[1]) * s)
    xs, ys = np.meshgrid(lo[0] + (np.arange(bw) + 0.5) / s, lo[1] + (np.arange(bh) + 0.5) / s)
    px, ok = floor_to_raw_px(cal, np.c_[xs.ravel(), ys.ravel()])
    px[~ok] = -1
    maps = px.reshape(bh, bw, 2).astype(np.float32)
    bird = cv2.remap(img, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderValue=(40, 40, 40))
    for x in np.arange(np.ceil(lo[0]), hi[0], 1.0):
        cv2.line(bird, (int((x - lo[0]) * s), 0), (int((x - lo[0]) * s), bh), (0, 200, 255), 1)
    for y in np.arange(np.ceil(lo[1]), hi[1], 1.0):
        cv2.line(bird, (0, int((y - lo[1]) * s)), (bw, int((y - lo[1]) * s)), (0, 200, 255), 1)
    cv2.imwrite(str(out / f"{cam}_birdseye.jpg"), bird)
    written.append(f"{cam}_birdseye.jpg (1 m grid, {s:.0f} px/m)")
    for name in written:
        print(f"-> {out / name}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("click")
    c.add_argument("--image", required=True)
    c.add_argument("--points", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--cam", required=True)
    f.add_argument("--points", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--image", help="frame of this camera, to compute the floor FOV polygon")
    f.add_argument(
        "--fov-top", type=float, default=0.0, help="FOV uses image rows below this ratio"
    )
    f.add_argument("--fov-max-m", type=float, default=50.0)
    f.add_argument("--tile-m", type=float, help="tile size (m) for tile_i/tile_j points")
    f.add_argument("--undistort", action="store_true", help="fit radial lens distortion")
    f.add_argument("--f-ratio", type=float, default=0.6, help="fixed focal length / width")
    f.add_argument("--width", type=int, default=1920, help="image size when no --image")
    f.add_argument("--height", type=int, default=1080)
    g = sub.add_parser("grid", help="checkerboard floor: all tile corners from one tile")
    g.add_argument("--image", required=True)
    g.add_argument("--points", required=True)
    g.add_argument("--seed", nargs=4, help="4 corners x,y of one tile instead of clicking")
    g.add_argument("--min-contrast", type=float, default=25.0)
    g.add_argument("--figure", help="image with the found corners")
    r = sub.add_parser("figures", help="report figures of a fitted calibration")
    r.add_argument("--calib", required=True)
    r.add_argument("--image", required=True)
    r.add_argument("--points", help="calibration points to draw (with --tile-m if tiles)")
    r.add_argument("--tile-m", type=float)
    r.add_argument("--grid-m", type=float, help="floor grid spacing drawn (default tile-m or 1)")
    r.add_argument("--zones", help="zones.yaml to draw on the frame")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--px-per-m", type=float, default=100.0, help="bird's-eye view scale")
    args = ap.parse_args(argv)
    {"click": cmd_click, "fit": cmd_fit, "grid": cmd_grid, "figures": cmd_figures}[args.cmd](args)


if __name__ == "__main__":
    main()
