# ppe-multicam — Intelligent Safety Monitoring

Đồ án tốt nghiệp: giám sát an toàn đa camera (PPE + hành vi nguy hiểm) trên NVIDIA DeepStream.
Thiết kế chi tiết, contribution và quy ước: xem [CLAUDE.md](CLAUDE.md). Tiến độ: [process/STATUS.md](process/STATUS.md).

## Cài đặt (dev, không cần DeepStream)

```powershell
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # GPU
pip install -e ".[dev]"
```

```bash
# Linux / WSL
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quy trình

```bash
# 1. PPE detector trên SH17 (crop người, D7)
python -m training.ppe_detector.prepare_sh17_person_crops --mode crops
python -m training.ppe_detector.train
python -m training.ppe_detector.export_onnx --weights runs/ppe/sh17_crop_y11s/weights/best.pt

# 2. Thu dữ liệu C2 (kịch bản: docs/c2_shooting_script.md)
#    data/raw/s01/cam1.mp4, cam2.mp4, cam3.mp4 → kiểm tra fps/VFR, đồng bộ (tắt/bật đèn đầu + cuối),
#    trích frame sàn trống ra data/raw/s01/calib/, ghi cameras.yaml của session cạnh JSONL
python tools/session_check.py --session data/raw/s01 --ref cam1 --out-cameras data/jsonl/s01/cameras.yaml
python tools/calibrate_homography.py click --image data/raw/s01/calib/cam1.jpg \
    --points configs/calibration/cam1_points.csv
python tools/calibrate_homography.py fit --cam cam1 --points configs/calibration/cam1_points.csv \
    --image data/raw/s01/calib/cam1.jpg --out configs/calibration/cam1.yaml
#    zones riêng cho session (nếu phòng khác): chép configs/zones.yaml → data/jsonl/s01/zones.yaml
#    (cameras.yaml / zones.yaml trong thư mục JSONL ghi đè configs/ cho session đó)

# 3. Perception offline → JSONL (1 file / cam), timestamp theo pts (video điện thoại thường VFR)
python -m src.perception_offline.ultralytics_runner --cameras data/jsonl/s01/cameras.yaml --out data/jsonl/s01
python tools/replay.py --jsonl data/jsonl/s01/cam1.jsonl --video data/raw/s01/cam1.mp4 --out cam1_vis.mp4

# 4. Gán nhãn interval
python tools/label_intervals.py template --jsonl data/jsonl/s01 --out data/annotations/s01
#    intervals_src.csv: gt_id,type,start,end,note với giờ video cam1 (1:02.5) hoặc giờ phiếu (14:03:30);
#    CROWD theo zone, để trống gt_id (hoặc "-")
python tools/label_intervals.py convert --jsonl data/jsonl/s01 --src data/annotations/s01/intervals_src.csv \
    --out data/annotations/s01/intervals.csv --clock video --ref cam1
python tools/label_intervals.py validate --jsonl data/jsonl/s01 --ann data/annotations/s01

# 5. Toàn bộ thí nghiệm C2 bằng một lệnh (split val/test trong configs/c2_split.yaml):
#    check → calib (p calibration + v_learned trên val) → tune (từng phương pháp, trên val)
#    → main (B0–B4 + B0/B1 mỗi cam vs P + ablation, trên test; cả bản chưa tune) → sens → numcams
#    → eval/reports/c2/REPORT.md. Bảng có thêm từng item PPE, behavior (fall/immobile/intrusion/crowd)
#    và association (pairwise P/R/F1 theo tracks.csv).
python -m eval.run_c2 --steps check          # chạy trước: nhãn, homography, split
python -m eval.run_c2 --out eval/reports/c2 --jobs 4
#    Từng bước riêng (cùng code):
python -m src.analytics_runner --jsonl data/jsonl/s01 --out data/results/s01/P [--cams cam1 cam2]
python -m eval.tune_c2 --sessions data/jsonl/s01 --gt data/annotations --out eval/reports/c2/tune --jobs 4
python -m eval.run_baselines --sessions data/jsonl/s02 data/jsonl/s03 --gt data/annotations \
    --out eval/reports/c2/main --ablations --single-cam-all --tuned eval/reports/c2/tune/tuned.json
python -m eval.run_sensitivity --sessions data/jsonl/s02 --gt data/annotations --out eval/reports/c2/sens --jobs 4
python -m eval.run_num_cams --sessions data/jsonl/s02 data/jsonl/s03 --gt data/annotations --out eval/reports/c2/numcams
#    Chạy thử toàn chuỗi trước khi có C2 (GT thật của WILDTRACK, 3 cam):
python tools/wildtrack_rehearsal.py --src data/rehearsal/wt3/jsonl \
    --gt data/external/wildtrack/prepared/gt.jsonl --out data/rehearsal/wt3_gt
python -m eval.run_c2 --split data/rehearsal/wt3_gt/split.yaml --out data/rehearsal/wt3_gt/report --jobs 4

# 6. Alert + dashboard (P7): lưu SQLite (+Telegram nếu bật trong configs/alerts.yaml, token trong .env)
python -m src.analytics_runner --jsonl data/jsonl/s01 --out data/results/s01/P --db
python tools/make_evidence.py --results data/results/s01/P --jsonl data/jsonl/s01 --video cam1=... --video cam2=...
python -m src.dashboard.app --results data/results/s01/P --port 8000   # http://127.0.0.1:8000
```

## Kiểm tra

```bash
pytest -q
ruff check src eval tools tests training
```

## Cấu trúc

- `configs/` — mọi ngưỡng và tham số (không hardcode)
- `src/` — pipeline DeepStream, offline backend, analytics (geometry, association, parts, fusion, alerts)
- `training/` — PPE detector (SH17)
- `eval/` — metrics, baselines, ablation, benchmark
- `tools/` — calibrate, đồng bộ, replay, gán nhãn
- `process/` — nhật ký tiến độ theo session
- `docs/experiments.md` — nhật ký thí nghiệm (bắt buộc)
