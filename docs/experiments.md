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

### 2026-10-22 — Thí nghiệm 5: detector trên crop người (SGIE, D7) vs full-frame
- Cùng dữ liệu train (SH17), cùng YOLO11s, cùng pose/ROI/GT người (SH17 test, 2,450 người khớp GT).
- F1 full-frame: imgsz 640, batch 16, best/early-stop trong 94 epoch; mAP50 trên SH17 full-frame test 0.553 (helmet 0.682, vest 0.446, shoes 0.583) — *không so trực tiếp với mAP crop vì khác ảnh test*.
- So sánh ở mức bài toán (`eval/reports/exp5_crop_vs_full/results.md`, ROI đã tune):

| Detector | Part | recall\|worn | recall v<0.6 | recall v≥0.6 | FPR |
|---|---|---|---|---|---|
| crop M1 (320) | head | **0.847** | **0.726** | **0.968** | 0.020 |
| full F1 (640) | head | 0.694 | 0.516 | 0.871 | 0.016 |
| crop M1 | torso | **0.672** | 0.500 | **0.712** | 0.012 |
| full F1 | torso | 0.609 | 0.833 (n nhỏ) | 0.558 | 0.010 |
| crop M1 | feet | **0.836** | **0.623** | **0.903** | 0.029 |
| full F1 | feet | 0.794 | 0.570 | 0.864 | 0.023 |

- Kết luận: crop người tăng recall helmet +15 điểm (+21 khi v<0.6) với FPR gần như không đổi → **xác nhận D7**.

### 2026-10-22 — M3: YOLO11m vs M2 YOLO11s (cùng SH17+CHV crops, imgsz 320)
- M3: batch 32, best epoch 58 / 78 (2.06 h). `eval/reports/detector_m2_m3/results.md`:

| Model | Test | mAP50 | mAP50-95 | helmet | vest | shoes | head | foot |
|---|---|---|---|---|---|---|---|---|
| M2 y11s | SH17 | 0.640 | 0.424 | 0.741 | 0.577 | 0.697 | 0.949 | 0.237 |
| M3 y11m | SH17 | 0.660 | 0.442 | 0.753 | 0.624 | 0.707 | 0.945 | 0.271 |
| M2 y11s | CHV | 0.935 | 0.540 | 0.957 | 0.913 | – | – | – |
| M3 y11m | CHV | 0.930 | 0.544 | 0.959 | 0.901 | – | – | – |

- Tốc độ (PyTorch FP32, batch 64 crop 320, RTX 4060 Laptop): M2 2.75 ms/crop, M3 4.51 ms/crop (×1.64).
- Quyết định: giữ **M2** mặc định (real-time đa cam, 8 GB); M3 tăng chủ yếu ở SH17 vest, không tăng trên CHV (miền công trường).

### 2026-10-22 — Benchmark đa góc **bán tổng hợp** (trước C2) — `eval/semi_synthetic_multiview.py`
- Mỗi ảnh SH17/CHV test → các "camera" giả: `orig`, `occ` (che vùng đầu, người vẫn phát hiện được), `far` (thu nhỏ ×4), `blur` (σ=6). Danh tính người theo GT; cảnh tĩnh 15 s ở 10 Hz; luật alert như hệ thống. Ngưỡng P được tune trên SH17+CHV **val** (J = DET − FA). Kết quả đầy đủ: `eval/reports/semi_synthetic/results.md`.
- **Không thay thế C2**: che khuất là nhân tạo, GT là "PPE nhìn thấy" trên ảnh gốc.

| Setup | Method | helmet FA | helmet DET | vest FA | vest DET |
|---|---|---|---|---|---|
| mixed (orig,occ,far) | B1@occ (single cam bị che) | **0.696** | 0.807 | 0.126 | 0.786 |
| mixed | B2 OR | 0.048 | 0.862 | 0.061 | 0.868 |
| mixed | B3 majority | 0.096 | 0.872 | 0.116 | 0.879 |
| mixed | B4 conf-weighted | 0.157 | 0.883 | 0.109 | 0.879 |
| mixed | P (config θ_obs=1, θ_neg=−0.5) | 0.046 | 0.686 | 0.089 | 0.725 |
| mixed | P (tuned val: θ_obs=0.02, θ_neg=−1.5) | 0.061 | 0.826 | 0.061 | 0.845 |
| degraded (occ,far,blur) | B2 OR | 0.124 | 0.853 | 0.082 | 0.849 |
| degraded | B3 | 0.329 | 0.865 | 0.143 | 0.863 |
| degraded | B4 | 0.451 | 0.871 | 0.147 | 0.862 |
| degraded | P (tuned) | 0.159 | 0.817 | 0.078 | 0.829 |
| degraded | P-no_v | 0.388 | 0.870 | 0.109 | 0.860 |
| degraded | P-no_unknown | 0.296 | 0.861 | 0.143 | 0.852 |

- Kết luận trung thực:
  - Single-cam bị che đầu báo nhầm ~70 % người có đội mũ → vấn đề C1 là thật.
  - Ablation: bỏ v ⇒ FA helmet 0.159 → 0.388; bỏ UNKNOWN ⇒ 0.296 → các thành phần của P có tác dụng; P ≫ B3, B4.
  - **B2 (OR) mạnh ngang/hơn P** trên benchmark này (không có view nào báo nhầm "có mũ", nên OR không bị phạt). → Claim C1 phải chứng minh **so với B2** trên C2; C2 cần kịch bản OR thất bại (box mũ người bên cạnh lọt vào ROI, vật giống mũ, cầm mũ gần đầu).
  - Ngưỡng tune nằm ở biên lưới và mục tiêu J không phạt abstention → **không** đưa vào config; tune lại trên C2.
