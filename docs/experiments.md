# Nhật ký thí nghiệm

> Mỗi mục: ngày, commit hash, config, lệnh chạy, số liệu (copy từ output script — không điền tay).

## Datasets — license & nguồn

| Dataset | Link chính thức | License | Ghi chú |
|---|---|---|---|
| SH17 | https://github.com/ahmadmughees/SH17dataset · Kaggle `mugheesahmad/sh17-dataset-for-ppe-detection` | **CC BY-NC-SA 4.0** (README + Kaggle metadata, đã kiểm 2026-10-21); ảnh gốc từ Pexels → theo Pexels license | 17 class (sh17.yaml). Archive Kaggle 14.24 GB. Chỉ dùng phi thương mại (đồ án OK) |
| SHWD | TODO(verify) | TODO(verify) | |
| CHV | TODO(verify) | TODO(verify) | |
| Le2i | TODO(verify) | TODO(verify) | |
| UR Fall | TODO(verify) | TODO(verify) | |
| CAUCAFall | TODO(verify) | TODO(verify) | |
| Multiple Cameras Fall (Montréal) | TODO(verify) | TODO(verify) | |

## Log

<!-- ### YYYY-MM-DD — <tên thí nghiệm>
- Commit:
- Config:
- Lệnh:
- Kết quả:
-->

### 2026-10-21 — M1: PPE detector trên SH17 person crops
- Commit: `9eb1b13`-era code (train started at `a075b18`+ prepare script of `1e765fc`); config `configs/training_ppe.yaml` (yolo11s.pt, imgsz 320, batch 64, seed 0, patience 20)
- Lệnh: `python -m training.ppe_detector.prepare_sh17_person_crops --mode crops` → `python -m training.ppe_detector.train`
- Dữ liệu: SH17 crops train 9,910 / val 1,125 / test 2,730 (test = SH17 `val_files.txt`, 1,620 ảnh gốc)
- Kết quả: early stop, best epoch 65 / 85 (1.30 h, RTX 4060 Laptop). **Test** (`runs/ppe/sh17_crop_y11s/test_metrics.json`):

| Class | AP50 | AP50-95 |
|---|---|---|
| helmet | 0.724 | 0.518 |
| vest | 0.546 | 0.341 |
| shoes | 0.676 | 0.378 |
| head | 0.951 | 0.728 |
| foot | 0.231 | 0.104 |
| **all** | **0.626** | **0.414** |

- Nhận xét: vest yếu (SH17 chỉ 392 vest trong train crops); foot rất yếu (ít mẫu, dễ nhầm với shoes). → bổ sung CHV (M2).
