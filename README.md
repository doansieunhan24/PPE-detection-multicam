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

# 2. Thu dữ liệu C2: calibrate + đồng bộ
python tools/calibrate_homography.py click --image cam1.jpg --points configs/calibration/cam1_points.csv
python tools/calibrate_homography.py fit --cam cam1 --points configs/calibration/cam1_points.csv \
    --image cam1.jpg --out configs/calibration/cam1.yaml
python tools/sync_offsets.py --videos cam1=... cam2=... cam3=... --ref cam1

# 3. Perception offline → JSONL (1 file / cam)
python -m src.perception_offline.ultralytics_runner --out data/jsonl/s01 \
    --source cam1=data/raw/s01/cam1.mp4 --source cam2=... --source cam3=...
python tools/replay.py --jsonl data/jsonl/s01/cam1.jsonl --video data/raw/s01/cam1.mp4 --out cam1_vis.mp4

# 4. Gán nhãn interval
python tools/label_intervals.py template --jsonl data/jsonl/s01 --out data/annotations/s01
python tools/label_intervals.py validate --jsonl data/jsonl/s01 --ann data/annotations/s01

# 5. Analytics + bảng chính (B0–B4 vs P + ablation)
python -m src.analytics_runner --jsonl data/jsonl/s01 --out data/results/s01/P
python -m eval.run_baselines --sessions data/jsonl/s01 --gt data/annotations --out eval/reports/main --ablations
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
