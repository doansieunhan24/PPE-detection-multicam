"""Record a C2 session from the real cameras (CLAUDE.md §9.1, docs/c2_shooting_script.md).

One session = one continuous video per camera, data/raw/<session>/cam1.mp4, cam2.mp4, ...
(the layout tools/session_check.py and label_intervals.py expect). Takes T01..T11 are marked
while recording and written to takes.csv (wall clock + approximate time in each video), so
labelling does not have to search the whole file.

ffmpeg stream-copies RTSP over TCP (no re-encode, camera timestamps kept) into .mkv, which
stays readable if the laptop or ffmpeg dies; a clean stop remuxes it to .mp4.

    # aim the cameras: live grid, s = snapshot, q = quit
    python tools/record_c2.py preview --cameras configs/cameras_lan.yaml
    # record session s01; type T01 + Enter to mark a take, x = end take, q = stop
    python tools/record_c2.py record --cameras configs/cameras_lan.yaml --session data/raw/s01
    # leftover .mkv after a crash
    python tools/record_c2.py remux --session data/raw/s01

Sync between cameras still comes from the light switch at the start/end (session_check.py);
the wall-clock start times in record_meta.json are only accurate to a few hundred ms.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_cameras  # noqa: E402

log = logging.getLogger("record_c2")

STALL_S = 3.0  # no new frame for this long -> warning


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise SystemExit("ffmpeg not found: put it on PATH or pip install imageio-ffmpeg") from e


def mask_uri(uri: str) -> str:
    """rtsp://user:pass@host/... -> rtsp://***@host/... for logs."""
    if "@" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    return f"{scheme}://***@{rest.rsplit('@', 1)[1]}"


def record_cmd(ffmpeg: str, uri: str, out: Path, max_s: float | None) -> list[str]:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1"]
    cmd += ["-rtsp_transport", "tcp", "-timeout", "5000000", "-i", uri]
    cmd += ["-map", "0:v:0", "-c", "copy"]
    if max_s:
        cmd += ["-t", str(max_s)]
    return cmd + ["-y", str(out)]


def remux(ffmpeg: str, src: Path, dst: Path) -> bool:
    r = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(src), "-c", "copy",
         "-movflags", "+faststart", "-y", str(dst)],
        capture_output=True, text=True,
    )  # fmt: skip
    if r.returncode != 0 or not dst.exists():
        log.error("remux %s failed: %s", src.name, r.stderr.strip()[-500:])
        return False
    return True


def video_info(path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"frames": n, "fps": round(fps, 3), "resolution": [w, h],
            "duration_s": round(n / fps, 1) if fps else None}  # fmt: skip


class CamRecorder:
    """One ffmpeg process; a reader thread parses -progress output for frame count."""

    def __init__(self, cam: dict[str, Any], out: Path, ffmpeg: str, max_s: float | None):
        self.cam_id = cam["cam_id"]
        self.uri = cam["uri"]
        self.out = out
        self.cmd = record_cmd(ffmpeg, self.uri, out, max_s)
        self.frames = 0
        self.last_frame_wall = 0.0
        self.start_wall = 0.0
        self.first_frame_wall: float | None = None
        self.stall_warned = False
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.start_wall = time.time()
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )  # fmt: skip
        threading.Thread(target=self._read_progress, daemon=True).start()
        threading.Thread(target=self._read_errors, daemon=True).start()

    def _read_progress(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            if line.startswith("frame="):
                n = int(line.split("=", 1)[1].strip() or 0)
                if n > self.frames:
                    now = time.time()
                    if self.first_frame_wall is None:
                        self.first_frame_wall = now
                    self.frames, self.last_frame_wall = n, now
                    self.stall_warned = False

    def _read_errors(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            log.warning("%s ffmpeg: %s", self.cam_id, line.rstrip().replace(self.uri, "<uri>"))

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stalled(self) -> bool:
        ref = self.last_frame_wall or self.start_wall
        return time.time() - ref > STALL_S

    def stop(self, timeout_s: float = 10.0) -> int | None:
        if not self.proc:
            return None
        if self.alive():
            try:
                assert self.proc.stdin
                self.proc.stdin.write("q")
                self.proc.stdin.flush()
            except OSError:
                pass
            try:
                self.proc.wait(timeout_s)
            except subprocess.TimeoutExpired:
                log.warning("%s: ffmpeg did not stop on 'q', killing (.mkv stays readable)",
                            self.cam_id)  # fmt: skip
                self.proc.kill()
                self.proc.wait()
        return self.proc.returncode


def fmt_s(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def monitor(recs: list[CamRecorder], stop: threading.Event, every_s: float) -> None:
    last_print = time.time()
    while not stop.wait(0.5):
        for r in recs:
            if not r.alive():
                if not r.stall_warned:
                    log.error("!!! %s: ffmpeg exited (code %s) -- recording of this camera "
                              "stopped. q to stop the session and check the cable/IP.",
                              r.cam_id, r.proc.returncode if r.proc else None)  # fmt: skip
                    r.stall_warned = True
            elif r.stalled() and not r.stall_warned:
                log.error("!!! %s: no frame for %.0f s (network?)", r.cam_id, STALL_S)
                r.stall_warned = True
        if time.time() - last_print >= every_s:
            last_print = time.time()
            parts = []
            for r in recs:
                el = time.time() - (r.first_frame_wall or time.time())
                fps = r.frames / el if el > 1 else 0.0
                parts.append(f"{r.cam_id} {r.frames} fr {fps:.1f} fps")
            log.info("[%s] %s", fmt_s(time.time() - recs[0].start_wall), " | ".join(parts))


def cmd_record(args: argparse.Namespace) -> None:
    cfg = load_cameras(args.cameras)
    cams = [c for c in cfg["cameras"] if not args.cams or c["cam_id"] in args.cams]
    session = Path(args.session)
    existing = [p.name for p in session.glob("cam*.*")] if session.exists() else []
    if existing and not args.overwrite:
        raise SystemExit(f"{session} already has {existing}; new --session or --overwrite")
    session.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_exe()

    recs = [CamRecorder(c, session / f"{c['cam_id']}.mkv", ffmpeg, args.max_s) for c in cams]
    for r in recs:
        log.info("%s <- %s", r.cam_id, mask_uri(r.uri))
        r.start()
    stop = threading.Event()
    threading.Thread(target=monitor, args=(recs, stop, args.status_s), daemon=True).start()

    takes: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def video_t(wall: float) -> dict[str, float | None]:
        return {r.cam_id: round(wall - r.first_frame_wall, 2) if r.first_frame_wall else None
                for r in recs}  # fmt: skip

    def close_take(wall: float) -> None:
        nonlocal current
        if current:
            current["end_wall"] = wall
            takes.append(current)
            log.info("== end %s (%s)", current["take"], fmt_s(wall - current["start_wall"]))
            current = None

    print(
        "\nRecording. Commands (+ Enter):\n"
        "  T01..T11  start a take (ends the running one)   x  end take\n"
        "  n <text>  note with timestamp                    q  stop recording\n"
        "Start the session with the light switch (off -> on) seen by every camera.\n"
    )
    try:
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            now = time.time()
            if not line:
                continue
            if line.lower() == "q":
                break
            if line.lower() == "x":
                close_take(now)
            elif line.lower().startswith("n "):
                notes.append({"wall": now, "text": line[2:].strip(), "video_t": video_t(now)})
                log.info("note @ %s: %s", fmt_s(now - recs[0].start_wall), line[2:].strip())
            elif line[0] in "tT" and line[1:].isdigit():
                close_take(now)
                current = {"take": f"T{int(line[1:]):02d}", "start_wall": now}
                log.info("== start %s", current["take"])
            else:
                log.info("unknown command %r", line)
    except KeyboardInterrupt:
        log.info("Ctrl-C: stopping")
    close_take(time.time())
    stop.set()
    end_wall = time.time()
    codes = {r.cam_id: r.stop() for r in recs}

    meta: dict[str, Any] = {
        "session": session.as_posix(),
        "cameras_config": Path(args.cameras).as_posix(),
        "started": datetime.fromtimestamp(recs[0].start_wall).isoformat(timespec="milliseconds"),
        "ended": datetime.fromtimestamp(end_wall).isoformat(timespec="milliseconds"),
        "cams": {},
        "notes": notes,
    }
    for r in recs:
        mp4 = r.out.with_suffix(".mp4")
        ok = r.out.exists() and r.out.stat().st_size > 0 and remux(ffmpeg, r.out, mp4)
        if ok and not args.keep_mkv:
            r.out.unlink()
        final = mp4 if ok else r.out
        meta["cams"][r.cam_id] = {
            "uri": mask_uri(r.uri),
            "file": final.name if final.exists() else None,
            "ffmpeg_exit": codes[r.cam_id],
            "frames_reported": r.frames,
            "first_frame_wall": r.first_frame_wall,
            **(video_info(final) if final.exists() else {}),
        }
    (session / "record_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with open(session / "takes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["take", "start_wall", "end_wall", "duration_s"]
                   + [f"{r.cam_id}_start_s" for r in recs]
                   + [f"{r.cam_id}_end_s" for r in recs])  # fmt: skip
        for t in takes:
            vs, ve = video_t(t["start_wall"]), video_t(t["end_wall"])
            w.writerow(
                [t["take"],
                 datetime.fromtimestamp(t["start_wall"]).strftime("%H:%M:%S.%f")[:-3],
                 datetime.fromtimestamp(t["end_wall"]).strftime("%H:%M:%S.%f")[:-3],
                 round(t["end_wall"] - t["start_wall"], 1)]
                + [vs[r.cam_id] for r in recs] + [ve[r.cam_id] for r in recs]
            )  # fmt: skip

    for cam_id, c in meta["cams"].items():
        log.info("%s: %s, %s frames, %s fps, %s s, ffmpeg exit %s", cam_id, c["file"],
                 c.get("frames"), c.get("fps"), c.get("duration_s"), c["ffmpeg_exit"])  # fmt: skip
    log.info("%d takes -> %s", len(takes), session / "takes.csv")
    log.info("next: python tools/session_check.py --session %s --ref %s "
             "--cameras %s --out-cameras %s/cameras.yaml",
             session.as_posix(), recs[0].cam_id, args.cameras, session.as_posix())  # fmt: skip


def cmd_remux(args: argparse.Namespace) -> None:
    ffmpeg = ffmpeg_exe()
    mkvs = sorted(Path(args.session).glob("cam*.mkv"))
    if not mkvs:
        raise SystemExit(f"no cam*.mkv in {args.session}")
    for src in mkvs:
        dst = src.with_suffix(".mp4")
        if remux(ffmpeg, src, dst):
            log.info("%s -> %s %s", src.name, dst.name, video_info(dst))


def cmd_preview(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cfg = load_cameras(args.cameras)
    cams = [c for c in cfg["cameras"] if not args.cams or c["cam_id"] in args.cams]
    frames: dict[str, np.ndarray | None] = {c["cam_id"]: None for c in cams}
    stop = threading.Event()

    def reader(cam: dict[str, Any]) -> None:
        uri = cam["uri"].replace("/1/1", "/1/2") if args.substream else cam["uri"]
        while not stop.is_set():
            cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                log.warning("%s: cannot open %s, retrying", cam["cam_id"], mask_uri(uri))
                time.sleep(2)
                continue
            while not stop.is_set():
                ok, f = cap.read()
                if not ok:
                    log.warning("%s: stream lost, reconnecting", cam["cam_id"])
                    break
                frames[cam["cam_id"]] = f
            cap.release()

    for c in cams:
        threading.Thread(target=reader, args=(c,), daemon=True).start()
    out_dir = Path(args.snap_dir)
    tile_w = args.tile_w
    print("preview: s = snapshot (full resolution), q / Esc = quit")
    while True:
        tiles = []
        for cam_id, f in frames.items():
            if f is None:
                t = np.zeros((tile_w * 9 // 16, tile_w, 3), np.uint8)
            else:
                t = cv2.resize(f, (tile_w, tile_w * f.shape[0] // f.shape[1]))
                t = cv2.resize(t, (tile_w, tile_w * 9 // 16))
            cv2.putText(t, cam_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            tiles.append(t)
        cv2.imshow("record_c2 preview", np.hstack(tiles))
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for cam_id, f in frames.items():
                if f is not None:
                    p = out_dir / f"{cam_id}_{stamp}.jpg"
                    cv2.imwrite(str(p), f)
                    log.info("snapshot -> %s", p)
    stop.set()
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("record", help="record a session")
    p.add_argument("--cameras", default="configs/cameras_lan.yaml")
    p.add_argument("--session", required=True, help="output folder, e.g. data/raw/s01")
    p.add_argument("--cams", nargs="*", help="subset of cam_id (default: all)")
    p.add_argument("--max-s", type=float, default=None, help="safety limit per camera")
    p.add_argument("--status-s", type=float, default=15.0, help="status line period")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-mkv", action="store_true")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("preview", help="live view to aim the cameras")
    p.add_argument("--cameras", default="configs/cameras_lan.yaml")
    p.add_argument("--cams", nargs="*")
    p.add_argument("--substream", action="store_true", help="704x576 /1/2 (lighter)")
    p.add_argument("--tile-w", type=int, default=800)
    p.add_argument("--snap-dir", default="data/raw/preview")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("remux", help="cam*.mkv -> cam*.mp4 after a crash")
    p.add_argument("--session", required=True)
    p.set_defaults(func=cmd_remux)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
