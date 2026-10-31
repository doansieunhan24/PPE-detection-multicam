"""DeepStream perception app (CLAUDE.md §4, §8). Runs in the ppe-ds:7.1 image (docker/README.md).

nvurisrcbin ×N → nvstreammux → PGIE YOLO-pose → nvtracker → SGIE PPE (crops) → probe → tee
    tee → fakesink                       (metadata: JSONL and/or ZMQ PUB)
    tee → tiler → nvdsosd → H.264 → RTSP (mediamtx) or .mp4      (--osd, optional)
The probe copies metadata into PerceptionFrame and hands it to the exporters (no heavy work).

    python3 -m src.pipeline.ds_app --cameras configs/cameras.yaml --out data/jsonl/live/
    # live demo (docker/README.md): frames to analytics_live, alerts back → smart record + OSD
    python3 -m src.pipeline.ds_app --cameras configs/cameras.yaml --out data/jsonl/live/ \\
        --zmq-pub tcp://*:5555 --sr-bind tcp://*:5556 --osd rtsp://localhost:8554/osd

- keypoints: DeepStream-Yolo-Pose stores them in mask_params (letterboxed network space)
  → letterbox_keypoints()
- metadata coordinates are in streammux resolution → scaled back to each camera's resolution
- ts_ms: RTSP = wall clock at the probe; file = --start-ms + buffer pts (same as the offline
  backend with file_timestamps: pts); both + the camera's time_offset_ms
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

from src.common.config import DEFAULT_CONFIG_DIR, load_cameras, load_yaml
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PerceptionFrame, PersonDet, PPEDet
from src.perception_offline.convert import pad_clip_bbox
from src.pipeline.meta_export import Fanout, JsonlExporter, ZmqPublisher, letterbox_keypoints

log = get_logger(__name__)
PGIE_ID, SGIE_ID = 1, 2
PPE_LABELS = ["helmet", "vest", "shoes", "head", "foot"]  # configs/deepstream/labels_ppe.txt
# OSD colours (r, g, b): PPE items green, the matching "missing" classes red
OSD_COLORS = {
    "person": (1.0, 1.0, 1.0),
    "helmet": (0.2, 0.9, 0.3),
    "vest": (0.2, 0.9, 0.3),
    "shoes": (0.2, 0.9, 0.3),
    "head": (0.95, 0.25, 0.25),
    "foot": (0.95, 0.25, 0.25),
}
LEVEL_COLORS = {1: (0.8, 0.8, 0.8), 2: (1.0, 0.6, 0.1), 3: (1.0, 0.2, 0.2)}


class AlertOverlay:
    """Recent alerts per camera, drawn on the OSD for `hold_s` (fed by the trigger channel)."""

    def __init__(self, hold_s: float) -> None:
        self.hold_s = hold_s
        self._items: dict[str, list[tuple[float, int, str]]] = {}

    def add(self, alert: dict, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        gid = alert.get("global_id")
        text = f"L{alert.get('level', '?')} {alert.get('type', '?')}" + (
            f" #{gid}" if gid is not None else ""
        )
        for cam in alert.get("cams", []):
            self._items.setdefault(cam, []).append(
                (now + self.hold_s, int(alert.get("level", 1)), text)
            )

    def active(self, cam: str, now: float | None = None) -> list[tuple[int, str]]:
        now = time.monotonic() if now is None else now
        items = [it for it in self._items.get(cam, []) if it[0] > now]
        self._items[cam] = items
        return [(lvl, text) for _, lvl, text in items[-4:]]


def _set_color(rp, rgb, width: int) -> None:
    rp.border_width = width
    rp.border_color.set(*rgb, 1.0)
    rp.has_bg_color = 0


def _draw_overlay(pyds, batch, fm, lines: list[tuple[int, str]]) -> None:
    dm = pyds.nvds_acquire_display_meta_from_pool(batch)
    dm.num_labels = len(lines)
    for i, (lvl, text) in enumerate(lines):
        tp = dm.text_params[i]
        tp.display_text = text
        tp.x_offset, tp.y_offset = 20, 20 + 60 * i
        tp.font_params.font_name = "Sans"
        tp.font_params.font_size = 28
        tp.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        tp.set_bg_clr = 1
        tp.text_bg_clr.set(*LEVEL_COLORS.get(lvl, LEVEL_COLORS[1]), 0.85)
    pyds.nvds_add_display_meta_to_frame(fm, dm)


def _keypoints(obj_meta, mux_w: int, mux_h: int, sx: float, sy: float) -> list:
    """17 COCO joints from DeepStream-Yolo-Pose mask_params (letterboxed network space)."""
    mp = obj_meta.mask_params
    if mp is None or mp.size < 17 * 3 * 4:  # float32
        return []
    kp = letterbox_keypoints(
        mp.get_mask_array(), mp.size // 4, mp.width, mp.height, mux_w, mux_h, sx, sy
    )
    return kp[:17]


class FpsCounter:
    """Frames per camera at the export probe (CLAUDE.md §8: FPS per source), logged every
    `interval_s`. summary() counts only frames after the first `warmup` of each camera."""

    def __init__(
        self, cam_ids: list[str], interval_s: float = 5.0, warmup: int = 0, on_first=None
    ) -> None:
        self.cam_ids = cam_ids
        self.on_first = on_first  # called once at the first frame (engines loaded by then)
        self.interval_s = interval_s
        self.warmup = warmup
        self.total = dict.fromkeys(cam_ids, 0)
        self._window = dict.fromkeys(cam_ids, 0)
        self._t_log: float | None = None
        self._t_first: dict[str, float] = {}  # first frame after warmup
        self._t_last: dict[str, float] = {}

    def tick(self, cam_id: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self._t_log is None:
            self._t_log = now
            if self.on_first is not None:
                self.on_first()
        self.total[cam_id] += 1
        self._window[cam_id] += 1
        if self.total[cam_id] == self.warmup + 1:
            self._t_first[cam_id] = now
        self._t_last[cam_id] = now
        if now - self._t_log >= self.interval_s:
            dt = now - self._t_log
            log.info("FPS %s", " ".join(f"{c}={self._window[c] / dt:.2f}" for c in self.cam_ids))
            self._window = dict.fromkeys(self.cam_ids, 0)
            self._t_log = now

    def summary(self) -> dict[str, float]:
        out = {}
        for c in self.cam_ids:
            n = self.total[c] - self.warmup - 1  # intervals between timed frames
            dt = self._t_last.get(c, 0.0) - self._t_first.get(c, 0.0)
            out[c] = n / dt if n > 0 and dt > 0 else 0.0
        return out


def build_crop_pad_probe(pyds, crop_pad: float, mux_w: int, mux_h: int):
    """SGIE sink: enlarge person rects by crop_pad (clipped), the padding the PPE detector was
    trained with (training/ppe_detector, offline backend pad_clip_bbox). nvinfer crops rect_params;
    the export probe reads the unpadded box back from detector/tracker_bbox_info."""
    from gi.repository import Gst

    def probe(pad, info, _u):
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
        l_frame = batch.frame_meta_list
        while l_frame is not None:
            fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = fm.obj_meta_list
            while l_obj is not None:
                o = pyds.NvDsObjectMeta.cast(l_obj.data)
                if o.unique_component_id == PGIE_ID:
                    r = o.rect_params
                    x1, y1, x2, y2 = pad_clip_bbox(
                        (r.left, r.top, r.left + r.width, r.top + r.height), crop_pad, mux_w, mux_h
                    )
                    r.left, r.top, r.width, r.height = x1, y1, x2 - x1, y2 - y1
                l_obj = l_obj.next
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK

    return probe


def _style_object(o, r) -> None:
    """OSD look of one object; persons get their unpadded box back (r: detector/tracker box)."""
    if o.unique_component_id == PGIE_ID:
        rp = o.rect_params
        rp.left, rp.top, rp.width, rp.height = r.left, r.top, r.width, r.height
        _set_color(rp, OSD_COLORS["person"], 2)
        o.text_params.display_text = (
            f"id {o.object_id}" if o.object_id != 0xFFFFFFFFFFFFFFFF else ""
        )
    elif o.unique_component_id == SGIE_ID:
        cls = PPE_LABELS[o.class_id] if o.class_id < len(PPE_LABELS) else ""
        _set_color(o.rect_params, OSD_COLORS.get(cls, (1.0, 1.0, 0.0)), 3)
        o.text_params.display_text = cls


def build_probe(
    pyds,
    cams: list[dict],
    mux_w: int,
    mux_h: int,
    exporter: JsonlExporter,
    meter: FpsCounter,
    pcfg: dict,
    start_ms: int = 0,
    person_bbox: str = "detector",
    osd: bool = False,
    overlay: AlertOverlay | None = None,
):
    """osd=True: also restore the person rect (padded for the SGIE crop) and set OSD colours /
    labels for nvdsosd downstream; overlay: alert texts per camera."""
    from gi.repository import Gst

    min_w, min_h = pcfg["ppe"]["min_crop_w_px"], pcfg["ppe"]["min_crop_h_px"]
    trunc_px = pcfg.get("truncated_bottom_margin_px", 2)

    def probe(pad, info, _u):
        buf = info.get_buffer()
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch.frame_meta_list
        while l_frame is not None:
            fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            cam = cams[fm.pad_index]
            w, h = cam["resolution"]
            sx, sy = w / mux_w, h / mux_h
            # keyed by tracker object_id: pyds returns a new wrapper on every cast / .parent
            persons: dict[int, PersonDet] = {}
            ppe: list[tuple[int, PPEDet]] = []
            l_obj = fm.obj_meta_list
            while l_obj is not None:
                o = pyds.NvDsObjectMeta.cast(l_obj.data)
                # persons: rect_params was padded for the SGIE crop → read the unpadded box from
                # the detector (same frame as the keypoints) or, if not detected, the tracker
                r = o.rect_params
                if o.unique_component_id == PGIE_ID:
                    r = o.tracker_bbox_info.org_bbox_coords
                    d = o.detector_bbox_info.org_bbox_coords
                    if person_bbox == "detector" and d.width > 0 and d.height > 0:
                        r = d
                box = (r.left * sx, r.top * sy, (r.left + r.width) * sx, (r.top + r.height) * sy)
                if osd:
                    _style_object(o, r)
                if o.unique_component_id == PGIE_ID and o.object_id != 0xFFFFFFFFFFFFFFFF:
                    persons[int(o.object_id)] = PersonDet(
                        track_id=int(o.object_id),
                        bbox=box,
                        conf=float(o.confidence),
                        kpts=_keypoints(o, mux_w, mux_h, sx, sy),
                        truncated_bottom=box[3] >= h - trunc_px,
                        ppe_checked=r.width * sx >= min_w and r.height * sy >= min_h,
                    )
                elif o.unique_component_id == SGIE_ID and o.parent is not None:
                    # SGIE detections are child objects whose parent is the person crop
                    cls = (
                        PPE_LABELS[o.class_id] if o.class_id < len(PPE_LABELS) else str(o.class_id)
                    )
                    parent = int(o.parent.object_id)
                    ppe.append((parent, PPEDet(cls=cls, bbox=box, conf=float(o.confidence))))
                l_obj = l_obj.next
            for parent, d in ppe:
                if parent in persons:
                    persons[parent].ppe_dets.append(d)
            if cam["uri"].startswith("rtsp://"):
                ts_ms = int(time.time() * 1000)
            else:
                ts_ms = start_ms + int(fm.buf_pts // 1_000_000)
            meter.tick(cam["cam_id"])
            if overlay is not None and (lines := overlay.active(cam["cam_id"])):
                _draw_overlay(pyds, batch, fm, lines)
            exporter.put(
                PerceptionFrame(
                    cam_id=cam["cam_id"],
                    frame_idx=fm.frame_num,
                    ts_ms=ts_ms + int(cam.get("time_offset_ms", 0)),
                    img_w=w,
                    img_h=h,
                    persons=list(persons.values()),
                )
            )
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK

    return probe


def _osd_branch(make, osd: str, n_cams: int, tiler_wh: tuple[int, int], bitrate: int, gop: int):
    """tee → queue (leaky: never slows inference) → tiler → nvdsosd → H.264 → RTSP or mp4.
    Returns the queue to link from the tee."""
    from gi.repository import Gst

    cols = math.ceil(math.sqrt(n_cams))
    rows = math.ceil(n_cams / cols)
    q = make("queue", "osd_queue", leaky=2, max_size_buffers=4)
    tiler = make(
        "nvmultistreamtiler",
        "tiler",
        rows=rows,
        columns=cols,
        width=tiler_wh[0],
        height=tiler_wh[1],
    )
    conv1 = make("nvvideoconvert", "osd_conv1")
    osd_el = make("nvdsosd", "osd", process_mode=1)  # 1 = GPU
    conv2 = make("nvvideoconvert", "osd_conv2")
    caps = make(
        "capsfilter",
        "osd_caps",
        caps=Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"),
    )
    enc = make(
        "nvv4l2h264enc",
        "osd_enc",
        bitrate=bitrate,
        iframeinterval=gop,
        idrinterval=gop,
    )
    # SPS/PPS before every IDR so viewers can join mid-stream (dGPU encoder has no
    # insert-sps-pps property, gst-inspect DS 7.1)
    parse = make("h264parse", "osd_parse", config_interval=-1)
    chain = [q, tiler, conv1, osd_el, conv2, caps, enc, parse]
    if osd.startswith("rtsp://"):
        out = make("rtspclientsink", "osd_sink", location=osd)
        Gst.util_set_object_arg(out, "protocols", "tcp")  # UDP fails in WSL2 (docker/README.md)
        chain.append(out)
    else:
        Path(osd).parent.mkdir(parents=True, exist_ok=True)
        chain += [make("mp4mux", "osd_mux"), make("filesink", "osd_sink", location=osd, sync=0)]
    for a, b in zip(chain, chain[1:], strict=False):
        if not a.link(b):
            raise RuntimeError(f"cannot link {a.get_name()} -> {b.get_name()}")
    return q


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cameras", default="configs/cameras.yaml")
    ap.add_argument("--out", help="JSONL output dir (<cam_id>.jsonl)")
    ap.add_argument("--zmq-pub", help="also publish frames on a ZMQ PUB endpoint (tcp://*:5555)")
    ap.add_argument(
        "--sr-bind",
        help="ZMQ PULL endpoint for alerts from analytics_live (smart record + OSD overlay), "
        "e.g. tcp://*:5556",
    )
    ap.add_argument("--osd", help="tiled OSD output: rtsp://host:8554/osd (mediamtx) or out.mp4")
    ap.add_argument("--osd-size", default="1920x1080")
    ap.add_argument("--osd-bitrate", type=int, default=4_000_000)
    ap.add_argument("--alerts-config", default=str(DEFAULT_CONFIG_DIR / "alerts.yaml"))
    ap.add_argument("--mux", default="1920x1080")
    ap.add_argument("--start-ms", type=int, default=0, help="session start epoch ms (file sources)")
    ap.add_argument("--pgie", default="configs/deepstream/pgie_yolo_pose.txt")
    ap.add_argument("--sgie", default="configs/deepstream/sgie_ppe.txt")
    ap.add_argument("--perception", default=str(DEFAULT_CONFIG_DIR / "perception.yaml"))
    ap.add_argument(
        "--person-bbox",
        choices=["detector", "tracker"],
        default="detector",
        help="exported person box: detector (falls back to tracker when not detected) or tracker",
    )
    ap.add_argument(
        "--duration-s",
        type=float,
        default=0,
        help="stop this many seconds after the first frame (file sources then loop); 0 = until EOS",
    )
    ap.add_argument("--warmup-frames", type=int, default=30, help="per camera, excluded from FPS")
    ap.add_argument("--stats-out", help="write FPS / frame counts as JSON (eval/bench_fps_ds.py)")
    # NVIDIA's config (proprietary, not in this repo), see configs/deepstream/tracker_nvdcf.yml
    ap.add_argument(
        "--tracker",
        default="/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
    )
    args = ap.parse_args(argv)
    if not args.out and not args.zmq_pub:
        ap.error("need --out and/or --zmq-pub")
    setup_logging()

    # Several RTSP sources connecting at once crash in GIO's libproxy module
    # (px_proxy_factory_get_proxies, gdb 2026-10-27, DS 7.1 image); no proxy is used here
    os.environ.setdefault("GIO_USE_PROXY_RESOLVER", "dummy")
    import gi

    gi.require_version("Gst", "1.0")
    import pyds
    from gi.repository import GLib, Gst

    Gst.init(None)
    cams = load_cameras(args.cameras)["cameras"]
    for c in cams:  # relative file paths (as in session cameras.yaml) → file:// URIs
        if "://" not in c["uri"]:
            c["uri"] = Path(c["uri"]).resolve().as_uri()
    live = all(c["uri"].startswith("rtsp://") for c in cams)
    mux_w, mux_h = (int(v) for v in args.mux.split("x"))
    alerts_cfg = load_yaml(args.alerts_config)
    sr_cfg = alerts_cfg.get("smart_record", {})
    use_sr = bool(args.sr_bind and sr_cfg.get("enabled", False))
    if use_sr and not live:
        log.warning("smart record needs RTSP sources: disabled (evidence: tools/make_evidence.py)")
        use_sr = False
    evidence_dir = Path(sr_cfg.get("out_dir", "data/evidence"))
    pipe = Gst.Pipeline()

    def make(factory, name, **props):
        e = Gst.ElementFactory.make(factory, name)
        if e is None:
            raise RuntimeError(f"cannot create {factory}")
        for k, v in props.items():
            e.set_property(k.replace("_", "-"), v)
        pipe.add(e)
        return e

    fps = max(c.get("fps", 25) for c in cams)
    mux = make(
        "nvstreammux",
        "mux",
        batch_size=len(cams),
        width=mux_w,
        height=mux_h,
        live_source=int(live),
        batched_push_timeout=int(1e6 / fps),
    )
    srcs = {}
    for i, c in enumerate(cams):
        src = make("nvurisrcbin", f"src{i}", uri=c["uri"])
        if args.duration_s > 0 and c["uri"].startswith("file://"):
            src.set_property("file-loop", True)
        if c["uri"].startswith("rtsp://"):
            # 4 = rtp-tcp: RTP over UDP fails in WSL2 (IPv6 udpsink), docker/README.md
            src.set_property("select-rtp-protocol", 4)
        if use_sr:
            from src.pipeline.smart_record import SmartRecorder

            SmartRecorder.configure(
                src, c["cam_id"], i, evidence_dir / "_sr", sr_cfg.get("cache_s", 15)
            )
        srcs[c["cam_id"]] = src
        sinkpad = mux.request_pad_simple(f"sink_{i}")
        src.connect("pad-added", lambda _s, pad, sp=sinkpad: pad.link(sp))
    pgie = make("nvinfer", "pgie", config_file_path=args.pgie, batch_size=len(cams))
    tracker = make(
        "nvtracker",
        "tracker",
        ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        ll_config_file=args.tracker,
    )
    sgie = make("nvinfer", "sgie", config_file_path=args.sgie)
    sink = make("fakesink", "sink", sync=0)
    for a, b in ((mux, pgie), (pgie, tracker), (tracker, sgie)):
        a.link(b)
    if args.osd:
        tee = make("tee", "tee")
        q_meta = make("queue", "meta_queue")
        sgie.link(tee)
        tee.link(q_meta)
        q_meta.link(sink)
        osd_wh = tuple(int(v) for v in args.osd_size.split("x"))
        gop = max(2, int(2 * fps))  # key frame every ~2 s
        tee.link(_osd_branch(make, args.osd, len(cams), osd_wh, args.osd_bitrate, gop))
    else:
        sgie.link(sink)

    pcfg = load_yaml(args.perception)
    sinks = []
    if args.out:
        sinks.append(JsonlExporter(args.out))
    if args.zmq_pub:
        sinks.append(ZmqPublisher(args.zmq_pub))
    exporter = Fanout(*sinks)
    loop = GLib.MainLoop()
    eos_sent = []

    def stop(*_a) -> bool:
        # EOS first so mp4mux (OSD file) writes its index; quit anyway if EOS does not arrive
        if not eos_sent:
            eos_sent.append(True)
            log.info("stopping: sending EOS")
            pipe.send_event(Gst.Event.new_eos())
            GLib.timeout_add(5000, lambda: (loop.quit(), False)[1])
        return False

    def start_timer() -> None:
        # --duration-s counts from the first frame, not from start-up (engine build can take
        # minutes); called from the streaming thread, timeout_add is thread-safe
        if args.duration_s > 0:
            GLib.timeout_add(int(args.duration_s * 1000), stop)

    overlay = AlertOverlay(float(alerts_cfg.get("live", {}).get("overlay_s", 5.0)))
    trigger = None
    if args.sr_bind:
        from src.pipeline.smart_record import SmartRecorder, TriggerServer

        recorder = None
        if use_sr:
            recorder = SmartRecorder(
                srcs, evidence_dir, sr_cfg.get("pre_s", 5), sr_cfg.get("post_s", 10)
            )

        def on_alert(a: dict) -> None:
            overlay.add(a)
            if recorder is not None and a.get("record", True):
                recorder.trigger(a["alert_id"], [c for c in a.get("cams", []) if c in srcs])

        trigger = TriggerServer(args.sr_bind, on_alert)

    meter = FpsCounter([c["cam_id"] for c in cams], warmup=args.warmup_frames, on_first=start_timer)
    sgie.get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER,
        build_crop_pad_probe(pyds, pcfg["ppe"]["crop_pad"], mux_w, mux_h),
        0,
    )
    sgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        build_probe(
            pyds,
            cams,
            mux_w,
            mux_h,
            exporter,
            meter,
            pcfg,
            args.start_ms,
            args.person_bbox,
            osd=bool(args.osd),
            overlay=overlay if args.sr_bind else None,
        ),
        0,
    )
    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message::eos", lambda *_: loop.quit())
    bus.connect("message::error", lambda _b, m: (log.error("%s", m.parse_error()), loop.quit()))
    for sig in (signal.SIGINT, signal.SIGTERM):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, stop)
    t0 = time.monotonic()
    pipe.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipe.set_state(Gst.State.NULL)
        if trigger is not None:
            trigger.close()
        exporter.close()
        log.info("wrote %d frames -> %s", exporter.written, args.out or args.zmq_pub)
        avg = meter.summary()
        log.info("FPS avg %s", {c: round(v, 2) for c, v in avg.items()})
        if args.stats_out:
            stats = {
                "fps_per_cam": avg,
                "frames_per_cam": meter.total,
                "warmup_frames": args.warmup_frames,
                "wall_s": time.monotonic() - t0,
                "exported": exporter.written,
                "dropped": exporter.dropped,
            }
            Path(args.stats_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
