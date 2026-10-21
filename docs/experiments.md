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

### 2026-10-21 — Visibility & evidence calibration trên SH17 (M1)
- Lệnh: `python -m training.visibility.build_v_dataset --split {val,test} --ppe-weights runs/ppe/sh17_crop_y11s/weights/best.pt` → `python -m training.visibility.calibrate --write-config configs/visibility_learned.yaml --write-p-calibration configs/p_calibration.yaml`
- Pose `yolo11m-pose`@960 + M1 trên crop, person khớp GT bằng IoU ≥ 0.5: val 978/1,128 người, test 2,450/2,734.
- Báo cáo đầy đủ: `eval/reports/visibility/report.md`, `results.json`.
- **Lần phân tích đầu (nhãn "evidence correct" trên mọi hàng) bị confound bởi prevalence** (SH17: 5 % đội mũ) → v có vẻ "ngược", v_learned học sai chiều. Đã thay bằng *detected | worn* và LR không phụ thuộc prior. Không dùng kết quả lần đầu.

| Part (test) | worn n | recall khi v<0.6 | recall khi v≥0.6 | AUC v (detected\|worn) | AUC v_learned | AUC person conf | log LR⁻ (val) |
|---|---|---|---|---|---|---|---|
| head/helmet | 123 | 0.47–0.59 | 0.91–1.00 | **0.792** | 0.781 | 0.691 | −1.18 |
| torso/vest | 63 | (n quá nhỏ) | 0.50–0.61 | 0.754 | 0.840 | 0.740 | −0.73 |
| feet/shoes | 463 | 0.28–0.56 | 0.47–0.68 | **0.690** | 0.719 | 0.598 | −0.88 |

- log LR⁻ của head theo bin v (test): −0.88, −0.64, −0.63, **−2.35**, (∞) → "không thấy mũ" chỉ là bằng chứng mạnh khi v cao.
- Hiệu chỉnh p (val, prior trung tính): head `logit p = 0.877·logit(conf) + 3.24` nếu có box, `−1.18` nếu không; torso 0.183/+4.34/−0.73; feet 0.873/+3.00/−0.88. → bằng chứng **bất đối xứng**: thấy box ≈ +3 log-LR, không thấy ≈ −1.
- Hệ quả cấu hình: `p_neg=0.05` cũ (logit −2.94) phóng đại bằng chứng âm ~2.5–4×; dùng `p_calibration` mặc định, `theta_neg` −1.0 → −0.5 (torso no-box chỉ −0.73).
