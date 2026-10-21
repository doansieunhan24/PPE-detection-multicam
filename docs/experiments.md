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

### 2026-10-22 — M2 (SH17+CHV), tuning part-ROI, hiệu chỉnh lại (thay thế số liệu calibration ở mục trước)
- **M2** = YOLO11s trên SH17 crops + CHV crops (CHV head/shoes/foot pseudo-label bằng M1, conf ≥ 0.5, chỉ train/val). Best epoch 74/94 (1.51 h). Detector table (`python -m eval.eval_detector ...`, `eval/reports/detector/results.md`):

| Model | Test | mAP50 | helmet | vest | shoes | head | foot |
|---|---|---|---|---|---|---|---|
| M1 (SH17) | SH17 crops | 0.626 | 0.724 | 0.546 | 0.676 | 0.951 | 0.231 |
| M2 (SH17+CHV) | SH17 crops | 0.640 | 0.741 | 0.577 | 0.697 | 0.949 | 0.237 |
| M1 | CHV crops | 0.842 | 0.890 | 0.794 | – | – | – |
| M2 | CHV crops | **0.935** | **0.957** | **0.913** | – | – | – |

  M2 chọn làm detector chính: `models/sh17chv_crop_y11s.{pt,onnx}`, `configs/perception.yaml` ppe.enabled=true.
- **Bug đã sửa:** bảng bằng chứng từng ghi p *đã hiệu chỉnh* (do config bật p_calibration) → calibrate fit chồng. Nay luôn ghi conf thô.
- **GT SH17/CHV:** containment item trong person box 0.8 → 0.5 (mũ thò lên trên box người; 13/159 người CHV bị coi nhầm "không đội").
- **Part-ROI tuning** (`python -m training.visibility.sweep_rois`, val SH17+CHV, mục tiêu Youden J): chọn containment 0.2, head_up 1.0, head_expand 1.8, torso_expand 0.4, feet_half_w 0.12, feet_down 0.1 (đều trong lưới, không ở biên). Kiểm tra trên **test** (không dùng khi tune):

| Test | Part | recall cũ → mới | FPR cũ → mới |
|---|---|---|---|
| SH17 | head | 0.782 → **0.903** | 0.023 → 0.025 |
| SH17 | torso | 0.609 → **0.766** | 0.010 → 0.013 |
| SH17 | feet | 0.606 → **0.834** | 0.022 → 0.035 |
| CHV | head | 0.816 → **0.951** | 0.158 → 0.184 (n not-worn ≈ 38, GT nhiễu) |
| CHV | torso | 0.768 → **0.944** | 0.048 → 0.075 |

- **Hiệu chỉnh p** (val SH17+CHV, prior trung tính, **chỉ dòng v ≥ 0.6**): head `2.474·logit(conf)+1.496` / no-box −4.27; torso 1.683/+2.107/−1.87; feet 0.971/+0.302/−1.75.
  - Lý do chỉ dùng v ≥ 0.6: GT ảnh đơn bị **điều kiện theo visibility** (người gán nhãn chỉ khoanh PPE nhìn thấy; chân bị cắt khỏi khung → "không giày"). Fit có scale theo v trên mọi dòng cho feet d = +0.56 (vô lý). → *Không thể hiệu chỉnh tác động của v bằng dataset ảnh đơn; cần C2.*
- **Visibility vs detection (test, M2, ROI mới)** — `eval/reports/visibility/report.md`, hình `visibility_reliability_{sh17,chv}.png`:
  - SH17 head: recall|worn 0.56 (v<0.2) → 0.90–0.92 (0.2–0.6) → 0.98–1.0 (v≥0.6); log LR⁻ −0.8 → −2.3/−2.5 → −3.8. Mô hình P dự đoán −4.27·v ≈ −0.4/−1.3/−2.1/−3.0 → **cùng xu hướng, gần tỷ lệ với v**.
  - AUC(v, detected|worn): head 0.83 (SH17) / 0.80 (CHV), feet 0.78 > person conf 0.66/0.53/0.67. Torso: v không dự đoán được (AUC 0.47/0.58) — sau khi sửa ROI, vest gần như luôn được thấy khi mặc.
