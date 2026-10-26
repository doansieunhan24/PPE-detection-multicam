"""DeepStream perception app (CLAUDE.md §4, §8) — NOT YET RUN: the dev machine has no DeepStream.

nvurisrcbin ×N → nvstreammux → PGIE YOLO-pose → nvtracker → SGIE PPE (crops) → probe → JSONL
The probe copies metadata into PerceptionFrame and hands it to JsonlExporter (no heavy work).

    python3 -m src.pipeline.ds_app --cameras configs/cameras.yaml --out data/jsonl/live/

TODO(verify) before first run, against the installed DeepStream / pyds version:
- element/property names below (nvurisrcbin, nvstreammux width/height/batched-push-timeout)
- keypoints: verified against DeepStream-Yolo-Pose deepstream.py (mask_params, letterboxed
  network space) → letterbox_keypoints(); still run once and overlay to confirm
- metadata coordinates are in streammux resolution → scaled back to each camera's resolution
"""

from __future__ import annotations

import argparse
import time

from src.common.config import load_yaml
from src.common.logging import get_logger, setup_logging
from src.common.schemas import PerceptionFrame, PersonDet, PPEDet
from src.pipeline.meta_export import JsonlExporter, letterbox_keypoints

log = get_logger(__name__)
PGIE_ID, SGIE_ID = 1, 2
PPE_LABELS = ["helmet", "vest", "shoes", "head", "foot"]  # configs/deepstream/labels_ppe.txt


def _keypoints(obj_meta, mux_w: int, mux_h: int, sx: float, sy: float) -> list:
    """17 COCO joints from DeepStream-Yolo-Pose mask_params (letterboxed network space)."""
    mp = obj_meta.mask_params
    if mp is None or mp.size < 17 * 3 * 4:  # float32
        return []
    kp = letterbox_keypoints(
        mp.get_mask_array(), mp.size // 4, mp.width, mp.height, mux_w, mux_h, sx, sy
    )
    return kp[:17]


def build_probe(pyds, cams: list[dict], mux_w: int, mux_h: int, exporter: JsonlExporter):
    from gi.repository import Gst

    def probe(pad, info, _u):
        buf = info.get_buffer()
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch.frame_meta_list
        while l_frame is not None:
            fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            cam = cams[fm.pad_index]
            w, h = cam["resolution"]
            sx, sy = w / mux_w, h / mux_h
            persons: dict[int, PersonDet] = {}
            ppe: list[tuple[int, PPEDet]] = []
            l_obj = fm.obj_meta_list
            while l_obj is not None:
                o = pyds.NvDsObjectMeta.cast(l_obj.data)
                r = o.rect_params
                box = (r.left * sx, r.top * sy, (r.left + r.width) * sx, (r.top + r.height) * sy)
                if o.unique_component_id == PGIE_ID and o.object_id != 0xFFFFFFFFFFFFFFFF:
                    persons[id(o)] = PersonDet(
                        track_id=int(o.object_id),
                        bbox=box,
                        conf=float(o.confidence),
                        kpts=_keypoints(o, mux_w, mux_h, sx, sy),
                        truncated_bottom=box[3] >= h - 2,
                        ppe_checked=r.width * sx >= 24 and r.height * sy >= 48,
                    )
                elif o.unique_component_id == SGIE_ID and o.parent is not None:
                    # TODO(verify): SGIE detections attached as child objects with parent=person
                    cls = (
                        PPE_LABELS[o.class_id] if o.class_id < len(PPE_LABELS) else str(o.class_id)
                    )
                    ppe.append((id(o.parent), PPEDet(cls=cls, bbox=box, conf=float(o.confidence))))
                l_obj = l_obj.next
            for parent, d in ppe:
                if parent in persons:
                    persons[parent].ppe_dets.append(d)
            exporter.put(
                PerceptionFrame(
                    cam_id=cam["cam_id"],
                    frame_idx=fm.frame_num,
                    ts_ms=int(time.time() * 1000) + int(cam.get("time_offset_ms", 0)),
                    img_w=w,
                    img_h=h,
                    persons=list(persons.values()),
                )
            )
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK

    return probe


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cameras", default="configs/cameras.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mux", default="1920x1080")
    ap.add_argument("--pgie", default="configs/deepstream/pgie_yolo_pose.txt")
    ap.add_argument("--sgie", default="configs/deepstream/sgie_ppe.txt")
    # NVIDIA's config (proprietary, not in this repo), see configs/deepstream/tracker_nvdcf.yml
    ap.add_argument(
        "--tracker",
        default="/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
    )
    args = ap.parse_args(argv)
    setup_logging()

    import gi

    gi.require_version("Gst", "1.0")
    import pyds
    from gi.repository import GLib, Gst

    Gst.init(None)
    cams = load_yaml(args.cameras)["cameras"]
    mux_w, mux_h = (int(v) for v in args.mux.split("x"))
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
        live_source=1,
        batched_push_timeout=int(1e6 / fps),
    )
    for i, c in enumerate(cams):
        src = make("nvurisrcbin", f"src{i}", uri=c["uri"])
        if c["uri"].startswith("rtsp://"):
            # 4 = rtp-tcp: RTP over UDP fails in WSL2 (IPv6 udpsink), docker/README.md
            src.set_property("select-rtp-protocol", 4)
        sinkpad = mux.request_pad_simple(f"sink_{i}")
        src.connect("pad-added", lambda _s, pad, sp=sinkpad: pad.link(sp))
    pgie = make("nvinfer", "pgie", config_file_path=args.pgie)
    tracker = make(
        "nvtracker",
        "tracker",
        ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        ll_config_file=args.tracker,
    )
    sgie = make("nvinfer", "sgie", config_file_path=args.sgie)
    sink = make("fakesink", "sink", sync=0)
    for a, b in ((mux, pgie), (pgie, tracker), (tracker, sgie), (sgie, sink)):
        a.link(b)

    exporter = JsonlExporter(args.out)
    sgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, build_probe(pyds, cams, mux_w, mux_h, exporter), 0
    )
    loop = GLib.MainLoop()
    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message::eos", lambda *_: loop.quit())
    bus.connect("message::error", lambda _b, m: (log.error("%s", m.parse_error()), loop.quit()))
    pipe.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipe.set_state(Gst.State.NULL)
        exporter.close()
        log.info("wrote %d frames -> %s", exporter.written, args.out)


if __name__ == "__main__":
    main()
