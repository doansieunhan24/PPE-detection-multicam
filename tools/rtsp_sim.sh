#!/usr/bin/env bash
# Simulate N RTSP cameras from video files (CLAUDE.md §8): mediamtx + ffmpeg loop.
#   bash tools/rtsp_sim.sh data/raw/session01/        # cam1.mp4, cam2.mp4, ... -> rtsp://localhost:8554/camX
# Needs: mediamtx (https://github.com/bluenviron/mediamtx) running or in PATH, ffmpeg in PATH.
set -euo pipefail
DIR="${1:?usage: rtsp_sim.sh <dir with camX.mp4>}"
HOST="${RTSP_HOST:-localhost}"
PORT="${RTSP_PORT:-8554}"

if ! (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null; then
  command -v mediamtx >/dev/null || { echo "mediamtx not running and not in PATH" >&2; exit 1; }
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
