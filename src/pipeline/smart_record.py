"""Alert-triggered evidence clips with nvurisrcbin smart record (CLAUDE.md §7.8).

analytics_live  --PUSH {alert}-->  ds_app (TriggerServer, PULL) --start-sr--> nvurisrcbin
                                   sr-done → <out_dir>/<alert_id>_<cam_id>.mp4

Checked on DS 7.1 (gst-inspect + gst-nvurisrcbin source, 2026-10-27):
- properties smart-record (2 = cloud + local events), smart-rec-cache (s), smart-rec-dir-path,
  smart-rec-file-prefix, smart-rec-mode (1 = video only), smart-rec-container (0 = mp4);
  only for type=rtsp sources. File name: <prefix>_<source-id>_<session>_<UTC time>_<pid>.mp4
- action signal start-sr(guint *session_id, guint start_s, guint duration_s, gpointer udata):
  session_id must be a valid pointer (NULL → assertion) → passed as a PyCapsule. Measured:
  start_s=4, duration_s=8 gave an 11.3 s clip, i.e. [trigger − start_s, trigger + duration_s]
  (aligned to a key frame)
- sr-done(NvDsSRRecordingInfo *info, gpointer udata): filename / dirpath read with ctypes

The recording is the camera's own encoded stream (before decode), so it has no OSD.
One recording per source at a time: alerts that arrive while it records share that clip.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.common.logging import get_logger

log = get_logger(__name__)


class _SRInfo(ctypes.Structure):
    """Head of NvDsSRRecordingInfo (gst-nvdssr.h, DS 7.1)."""

    _fields_ = [
        ("ctx", ctypes.c_void_p),
        ("sessionId", ctypes.c_uint32),
        ("filename", ctypes.c_char_p),
        ("dirpath", ctypes.c_char_p),
        ("duration", ctypes.c_uint64),
    ]


def read_sr_info(gpointer) -> tuple[str, str, int]:
    """(dirpath, filename, duration_ms) from the sr-done GPointer."""
    s = _SRInfo.from_address(hash(gpointer))  # GPointer hashes to its address
    return s.dirpath.decode(), s.filename.decode(), int(s.duration)


def _capsule(addr: int):
    new = ctypes.pythonapi.PyCapsule_New
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return new(addr, None, None)


def clip_name(alert_id: str, cam_id: str) -> str:
    """Evidence file name, relative to alerts.smart_record.out_dir (analytics side too)."""
    return f"{alert_id}_{cam_id}.mp4"


@dataclass
class _Cam:
    busy: bool = False
    alerts: list[str] = field(default_factory=list)


class EvidenceBook:
    """Which alerts each camera's current recording belongs to (no GStreamer, unit-tested).

    request() → True when a new recording must be started for that camera."""

    def __init__(self, cam_ids: list[str]) -> None:
        self.cams = {c: _Cam() for c in cam_ids}

    def request(self, cam_id: str, alert_id: str) -> bool:
        c = self.cams.get(cam_id)
        if c is None:
            return False
        c.alerts.append(alert_id)
        if c.busy:
            return False
        c.busy = True
        return True

    def failed(self, cam_id: str) -> None:
        self.cams[cam_id] = _Cam()

    def done(self, cam_id: str) -> list[str]:
        c = self.cams[cam_id]
        out, self.cams[cam_id] = c.alerts, _Cam()
        return out


def publish_clip(src: Path, out_dir: Path, names: list[str]) -> list[Path]:
    """Hard-link (or copy) the recorded file to every alert's evidence name, remove the original."""
    out = []
    for n in names:
        dst = out_dir / n
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        out.append(dst)
    if out:
        src.unlink(missing_ok=True)
    return out


class SmartRecorder:
    """Drives start-sr on the nvurisrcbin of each camera. Call trigger() on the GLib main loop."""

    def __init__(self, srcs: dict[str, object], out_dir: Path, pre_s: int, post_s: int) -> None:
        self.srcs = srcs
        self.out_dir = Path(out_dir)
        self.pre_s, self.post_s = int(pre_s), int(post_s)
        self.book = EvidenceBook(list(srcs))
        self._sid = {c: ctypes.c_uint32(0) for c in srcs}  # start-sr writes the session id here
        for cam, src in srcs.items():
            src.connect("sr-done", self._on_done, cam)

    @staticmethod
    def configure(src, cam_id: str, source_id: int, rec_dir: Path, cache_s: int) -> None:
        """Set before PLAYING (properties are changeable only in NULL/READY)."""
        rec_dir.mkdir(parents=True, exist_ok=True)
        src.set_property("type", 2)  # rtsp: needed for smart record
        src.set_property("source-id", source_id)
        src.set_property("smart-record", 2)
        src.set_property("smart-rec-mode", 1)
        src.set_property("smart-rec-container", 0)
        src.set_property("smart-rec-cache", int(cache_s))
        src.set_property("smart-rec-dir-path", str(rec_dir))
        src.set_property("smart-rec-file-prefix", cam_id)

    def trigger(self, alert_id: str, cams: list[str]) -> None:
        for cam in cams:
            if not self.book.request(cam, alert_id):
                continue
            src = self.srcs[cam]
            src.emit(
                "start-sr",
                _capsule(ctypes.addressof(self._sid[cam])),
                self.pre_s,
                self.post_s,
                None,
            )
            if not src.get_property("smart-rec-status"):
                log.warning("smart record did not start on %s (alert %s)", cam, alert_id)
                self.book.failed(cam)
            else:
                log.info("smart record %s for alert %s", cam, alert_id)

    def _on_done(self, _src, info, _udata, cam: str) -> None:
        dirpath, filename, dur_ms = read_sr_info(info)
        names = [clip_name(a, cam) for a in self.book.done(cam)]
        out = publish_clip(Path(dirpath) / filename, self.out_dir, names)
        log.info("evidence %s (%.1f s): %s", cam, dur_ms / 1000, [p.name for p in out])


class TriggerServer:
    """ZMQ PULL socket on a thread; each message (JSON AlertEvent) → on_alert(dict) via
    GLib.idle_add so GStreamer elements are only touched from the main loop."""

    def __init__(self, endpoint: str, on_alert: Callable[[dict], None]) -> None:
        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.bind(endpoint)
        self._on_alert = on_alert
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="sr-trigger", daemon=True)
        self._t.start()

    def _run(self) -> None:
        import zmq
        from gi.repository import GLib

        while not self._stop.is_set():
            if not self._sock.poll(200, zmq.POLLIN):
                continue
            try:
                msg = json.loads(self._sock.recv())
            except ValueError:
                log.warning("bad trigger message")
                continue
            GLib.idle_add(lambda m=msg: (self._on_alert(m), False)[1])

    def close(self) -> None:
        self._stop.set()
        self._t.join(1.0)
        self._sock.close(0)
