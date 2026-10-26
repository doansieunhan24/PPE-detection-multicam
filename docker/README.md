# DeepStream (P0) — đã chạy (2026-10-26)

## Trạng thái máy dev (kiểm tra 2026-10-26)
- WSL2 `Ubuntu-22.04`: Docker 29.8.0, runtime `nvidia`, RTX 4060 Laptop, driver Windows 591.66. Trong container: DeepStream 7.1.0, CUDA runtime 12.6, TensorRT 10.3, cuDNN 9.0, `pyds` import được.
- User WSL đã vào group `docker` (2026-10-26, `wsl -u root -- usermod -aG docker <user>`). Shell mở trước đó chưa nhận group: dùng `sg docker -c '...'` hoặc mở lại WSL.
- Image dự án: `docker build -f docker/Dockerfile.deepstream -t ppe-ds:7.1 .` (`.dockerignore` loại toàn bộ context; repo được mount vào `/workspace` khi chạy).

## P0: 3 RTSP → DeepStream (đã chạy)
```bash
# WSL, tại thư mục repo
bash tools/rtsp_sim.sh data/raw/p0/        # cam1..3.mp4 = samples/streams/sample_1080p_h264.mp4 của DS 7.1
docker run --rm --gpus all --network host     -v ds71_sample_models:/opt/nvidia/deepstream/deepstream-7.1/samples/models     -v "$PWD":/workspace -w /workspace --entrypoint bash ppe-ds:7.1     -c "deepstream-app -c configs/deepstream/p0_sample_3rtsp.txt"
```
- Volume `ds71_sample_models` giữ lại engine TensorRT INT8 của model mẫu (build mất ~55 s mỗi lần nếu không có volume).
- `tools/rtsp_sim.sh` tự chạy mediamtx + ffmpeg trong Docker (`bluenviron/mediamtx:1.9.3-ffmpeg`, host network) nếu máy không có sẵn. Không chạy cùng service `mediamtx` của compose (trùng cổng 8554).
- **RTSP phải dùng TCP** (`select-rtp-protocol=4`): trong WSL2, RTP/UDP lỗi `udpsink ... Invalid address family (got 10)` (IPv6) và chỉ chạy được sau 5 s tự chuyển sang TCP.
- Kết quả (log `data/p0/bench_3rtsp.log`, 150 s, sink fakesink): mỗi stream ổn định **~30.0 fps** = fps nguồn (trung bình tích luỹ 31.17 do bù đầu chuỗi). Video tiled kiểm tra bằng mắt: `data/p0/tiled_3rtsp.mp4` (sink1, encoder NVENC chạy được trong WSL).
- Đây mới là model mẫu của NVIDIA (resnet18 trafficcamnet) để kiểm tra stack; FPS với YOLO-pose + SGIE PPE đo ở bước sau. `nvidia-smi --query-gpu` không hỗ trợ trong WSL nên chưa có GPU util.

## Chọn version
- **DeepStream 7.1** (`nvcr.io/nvidia/deepstream:7.1-triton-multiarch`). Theo trang chính thức "DeepStream On WSL" của 7.1: Windows 11, GPU GeForce/Quadro ở chế độ WDDM, **Ubuntu 22.04**, chỉ cài driver Windows (không cài driver Linux trong WSL). Bản 8.0 trở lên yêu cầu Ubuntu 24.04 trên WSL, khác với distro đang có. Driver đã kiểm chứng trong docs là 546.65; máy này dùng 591.66, đã chạy thật được (P0, 2026-10-26).
- Lệnh chạy theo docs 7.1:
  ```bash
  docker run -it --rm --privileged --net=host --gpus all -e DISPLAY=$DISPLAY -e CUDA_CACHE_DISABLE=0 \
      -v /tmp/.X11-unix/:/tmp/.X11-unix -v /mnt/d/Download/Document/DATN/ppe_de:/workspace \
      nvcr.io/nvidia/deepstream:7.1-triton-multiarch
  deepstream-app --version-all
  ```

## Parser YOLO (trong container)
Đã đối chiếu với repo `marcoslucianops/DeepStream-Yolo-Pose` (master, 2026-10-25). Repo hỗ trợ DS 8.0 / 7.1 / 7.0 / …, có sẵn `config_infer_primary_yolo11_pose.txt`.
```bash
mkdir -p third_party && cd third_party
git clone https://github.com/marcoslucianops/DeepStream-Yolo-Pose
CUDA_VER=12.6 make -C DeepStream-Yolo-Pose/nvdsinfer_custom_impl_Yolo_pose   # 12.6 = DS 7.1 x86
git clone https://github.com/marcoslucianops/DeepStream-Yolo                  # SGIE PPE detector
CUDA_VER=12.6 make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
```
Export ONNX bằng script của **từng repo**, vì layout output khác với export mặc định của Ultralytics:
- Pose: chép `DeepStream-Yolo-Pose/utils/export_yolo11_pose.py` vào thư mục `ultralytics/`, rồi chạy `python3 export_yolo11_pose.py -w yolo11m-pose.pt -s 960 --dynamic`
- PPE: script export YOLO11 của DeepStream-Yolo (TODO(verify) tên script), dùng `-s 320`.

Keypoint nằm trong `obj_meta.mask_params` ở **toạ độ đầu vào mạng (letterbox)**. Probe đổi về pixel camera bằng `src/pipeline/meta_export.letterbox_keypoints` (cùng công thức với `deepstream.py` của repo).

## Chạy
1. `bash tools/rtsp_sim.sh data/raw/s01/` (mediamtx + ffmpeg loop), rồi `docker compose -f docker/docker-compose.yml up`.
2. Đối chiếu JSONL của DeepStream với JSONL offline trên cùng video (P1 "Done khi": 2 backend so khớp được).
3. Benchmark: `eval/bench_fps.py` mới đo offline backend. Offline đạt khoảng 16 fps tổng ở 1080p (RTX 4060 Laptop, `eval/reports/bench_fps_wildtrack1080`); số liệu DeepStream cần đo riêng.
