# DeepStream (P0) — chưa chạy thử (máy dev chưa có Docker)

1. Cài Docker Desktop (backend WSL2) + NVIDIA Container Toolkit; kiểm tra `docker run --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`.
2. Chọn tag DeepStream 7.x khớp driver → ghi vào CLAUDE.md §15. **TODO(verify)** hỗ trợ WSL2 của tag đó.
3. Build parser YOLO trong container (theo README của repo, TODO(verify) lệnh đúng cho version):
   - `git clone https://github.com/marcoslucianops/DeepStream-Yolo` → `CUDA_VER=<x.y> make -C nvdsinfer_custom_impl_Yolo`
   - `git clone https://github.com/marcoslucianops/DeepStream-Yolo-Pose` → tương tự
   - Export ONNX bằng script export của **từng repo** (output layout khác export mặc định của Ultralytics).
4. `bash tools/rtsp_sim.sh data/raw/session01/` rồi `docker compose -f docker/docker-compose.yml up`.
5. Đối chiếu JSONL DeepStream với JSONL offline (P1 "Done khi": 2 backend so khớp được).
