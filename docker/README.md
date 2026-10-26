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

## Model dự án trong DeepStream (đã chạy 2026-10-26)
Parser lấy từ `marcoslucianops/DeepStream-Yolo-Pose` (commit `2fa5bd8`) và `DeepStream-Yolo` (commit `2894bab`), clone vào `third_party/` (gitignored), build **trong container**:
```bash
mkdir -p third_party && cd third_party
git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo-Pose
git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo
# trong ppe-ds:7.1, tại /workspace:
CUDA_VER=12.6 make -C third_party/DeepStream-Yolo-Pose/nvdsinfer_custom_impl_Yolo_pose   # 12.6 = DS 7.1 x86
CUDA_VER=12.6 make -C third_party/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
```
Export ONNX (ở venv Windows/Linux có ultralytics) bằng script của từng repo. Layout output khác bản export của Ultralytics nên để riêng ở `models/ds/`. `tools/export_ds_onnx.py` ép exporter TorchScript (`dynamo=False`) vì torch ≥ 2.9 mặc định dynamo; cần `pip install onnxscript onnxslim`:
```bash
mkdir -p models/ds && cp models/yolo11m-pose.pt models/sh17chv_crop_y11s.pt models/ds/
python -m tools.export_ds_onnx pose -w models/ds/yolo11m-pose.pt -s 960 --dynamic --simplify      # output [b,18900,56]
python -m tools.export_ds_onnx det  -w models/ds/sh17chv_crop_y11s.pt -s 320 --dynamic --simplify  # output [b,2100,6]
```
Engine TensorRT FP16 build lần đầu: PGIE khoảng 6 phút, SGIE khoảng 4 phút. Engine SGIE do hàm của DeepStream-Yolo ghi ra thư mục làm việc với tên `model_b16_gpu0_fp16.engine`; chuyển nó thành `models/ds/sh17chv_crop_y11s.onnx_b16_gpu0_fp16.engine` để các lần sau dùng lại. PGIE build một engine batch 4, `ds_app` đặt `batch-size` bằng số cam.

```bash
python3 -m src.pipeline.ds_app --cameras data/rehearsal/wt3/jsonl/cameras.yaml --out data/ds/wt3/jsonl_detector
python3 -m eval.bench_fps_ds --videos data/rehearsal/wt3/raw/cam{1,2,3}.mp4 --streams 4 1 2 3 --duration-s 60 --out eval/reports/bench_fps_ds_wt3
```

Ghi chú khi nối vào DeepStream:
- Keypoint nằm trong `obj_meta.mask_params` ở **toạ độ đầu vào mạng (letterbox)**. `meta_export.letterbox_keypoints` đổi về pixel camera; keypoint vẫn còn sau NvDCF.
- PPE của SGIE là object con có `parent` là người. pyds tạo wrapper mới ở mỗi lần `cast` hay `.parent`, nên phải ghép bằng `object_id`, không dùng `id()`.
- **Crop SGIE phải nới `ppe.crop_pad`** như lúc train (probe ở sink pad của SGIE nới `rect_params`). Không nới thì mức khớp PPE với offline thấp hơn (shoes 0.818, head 0.901 so với 0.925 / 0.944 khi nới).
- **Bbox người xuất ra là bbox detector** (`--person-bbox detector`, fallback tracker khi frame đó không detect). Dùng bbox NvDCF thì pair F1 association trên WILDTRACK giảm 0.822 → 0.738.
- NMS của parser pose cố định IoU 0.45 (README repo, `cluster-mode=4`); offline dùng 0.7.
- `ts_ms`: RTSP lấy đồng hồ hệ thống, file lấy `--start-ms` + pts của buffer (khớp offline 100 % frame trên wt3).

## So khớp với offline
`python -m eval.compare_backends --ref <jsonl offline> --test <jsonl DS>`: ghép người theo IoU từng frame, so bbox, keypoint, PPE và track. Kết quả trên wt3 ghi ở `docs/experiments.md`.
