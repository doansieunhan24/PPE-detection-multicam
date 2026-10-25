# DeepStream (P0) — chưa chạy thử

## Trạng thái máy dev (kiểm tra 2026-10-25)
- WSL2 `Ubuntu-22.04`: có Docker Engine (daemon đang chạy), NVIDIA Container Toolkit (`nvidia-ctk`), runtime `nvidia` trong `/etc/docker/daemon.json`. `nvidia-smi` trong WSL thấy RTX 4060 Laptop, driver 591.66. Ổ còn 923 GB.
- **Đang vướng:** user WSL chưa thuộc group `docker` (`permission denied ... docker.sock`). Cần sudo nên phải tự chạy (một lần):
  ```bash
  # trong WSL Ubuntu-22.04
  sudo usermod -aG docker $USER
  # rồi ở PowerShell: wsl --shutdown, mở lại WSL
  docker run --rm hello-world
  docker run --rm --gpus all nvcr.io/nvidia/deepstream:7.1-triton-multiarch nvidia-smi
  ```

## Chọn version
- **DeepStream 7.1** (`nvcr.io/nvidia/deepstream:7.1-triton-multiarch`). Theo trang chính thức "DeepStream On WSL" của 7.1: Windows 11, GPU GeForce/Quadro ở chế độ WDDM, **Ubuntu 22.04**, chỉ cài driver Windows (không cài driver Linux trong WSL). Bản 8.0 trở lên yêu cầu Ubuntu 24.04 trên WSL, khác với distro đang có. Driver đã kiểm chứng trong docs là 546.65; máy này dùng 591.66 (mới hơn, TODO(verify) chạy thật).
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
