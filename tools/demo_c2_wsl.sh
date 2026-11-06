#!/usr/bin/env bash
# WSL side of tools/demo_c2.ps1 (run from the repo root).
#   bash tools/demo_c2_wsl.sh sim  data/demo/c2   # RTSP cameras from <demo>/raw/cam*.mp4
#   bash tools/demo_c2_wsl.sh ds   data/demo/c2   # DeepStream on them (ZMQ :5555, SR :5556, OSD)
#   bash tools/demo_c2_wsl.sh stop
set -euo pipefail
cmd="${1:?sim|ds|stop}"
demo="${2:-data/demo/c2}"
case "$cmd" in
  sim)
    docker rm -f rtsp_sim >/dev/null 2>&1 || true
    exec bash tools/rtsp_sim.sh "$demo/raw/" ;;
  ds)
    docker rm -f ds_demo >/dev/null 2>&1 || true
    exec docker run --rm --name ds_demo --gpus all --network host -v "$PWD":/workspace \
      -w /workspace --entrypoint bash ppe-ds:7.1 -c \
      "python3 -m src.pipeline.ds_app --cameras $demo/session/cameras.yaml --out $demo/live/jsonl \
       --zmq-pub 'tcp://*:5555' --sr-bind 'tcp://*:5556' --osd rtsp://localhost:8554/osd" ;;
  stop)
    docker rm -f ds_demo rtsp_sim >/dev/null 2>&1 || true
    echo "stopped ds_demo, rtsp_sim" ;;
  *) echo "unknown: $cmd" >&2; exit 2 ;;
esac
