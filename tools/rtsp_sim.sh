#!/usr/bin/env bash
# Simulate N RTSP cameras from video files (CLAUDE.md §8): mediamtx + ffmpeg loop.
#   bash tools/rtsp_sim.sh data/raw/session01/        # cam1.mp4, cam2.mp4, ... -> rtsp://localhost:8554/camX
# Uses mediamtx + ffmpeg from PATH if both exist; otherwise runs them in Docker
# (image bluenviron/mediamtx:<ver>-ffmpeg, host network). Ctrl-C stops everything.
# Path "osd" accepts the tiled OSD stream of ds_app (--osd).
set -euo pipefail
DIR="${1:?usage: rtsp_sim.sh <dir with camX.mp4>}"
HOST="${RTSP_HOST:-localhost}"
PORT="${RTSP_PORT:-8554}"
MTX_IMAGE="${MTX_IMAGE:-bluenviron/mediamtx:1.9.3-ffmpeg}"
DIR="$(cd "$DIR" && pwd)"

if ! command -v mediamtx >/dev/null || ! command -v ffmpeg >/dev/null; then
  # mediamtx publishes each file itself on startup (runOnInit), looping forever without re-encode
  cfg="$(mktemp)"
  {
    echo "rtspAddress: :${PORT}"
    echo "paths:"
    for f in "$DIR"/cam*.mp4; do
      cam="$(basename "$f" .mp4)"
      echo "  ${cam}:"
      echo "    runOnInit: ffmpeg -hide_banner -loglevel error -re -stream_loop -1 -i /media/${cam}.mp4 -c copy -f rtsp rtsp://localhost:${PORT}/${cam}"
      echo "    runOnInitRestart: yes"
      echo "${cam}: rtsp://${HOST}:${PORT}/${cam}" >&2
    done
    # published by ds_app --osd rtsp://localhost:8554/osd; mediamtx also serves it as
    # WebRTC http://<host>:8889/osd and HLS http://<host>:8888/osd (dashboard --video-url)
    echo "  osd:"
  } >"$cfg"
  exec docker run --rm --name rtsp_sim --network host \
    -v "$DIR":/media:ro -v "$cfg":/mediamtx.yml:ro "$MTX_IMAGE"
fi

if ! (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null; then
  mediamtx >/tmp/mediamtx.log 2>&1 &
  echo "started mediamtx (pid $!)"
  sleep 1
fi

pids=()
for f in "$DIR"/cam*.mp4; do
  cam="$(basename "$f" .mp4)"
  # -re: real-time pacing; -stream_loop -1: loop forever; -c copy: no re-encode
  ffmpeg -hide_banner -loglevel error -re -stream_loop -1 -i "$f" -c copy -f rtsp \
    "rtsp://${HOST}:${PORT}/${cam}" &
  pids+=($!)
  echo "${cam}: rtsp://${HOST}:${PORT}/${cam}"
done
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM EXIT
wait
