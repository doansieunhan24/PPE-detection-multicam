# Nhật ký thí nghiệm

> Mỗi mục: ngày, commit hash, config, lệnh chạy, số liệu (copy từ output script — không điền tay).

## Datasets — license & nguồn

| Dataset | Link chính thức | License | Ghi chú |
|---|---|---|---|
| SH17 | https://github.com/ahmadmughees/SH17dataset · Kaggle `mugheesahmad/sh17-dataset-for-ppe-detection` | **CC BY-NC-SA 4.0** (README + Kaggle metadata, đã kiểm 2026-10-21); ảnh gốc từ Pexels → theo Pexels license | 17 class (sh17.yaml). Archive Kaggle 14.24 GB. Chỉ dùng phi thương mại (đồ án OK) |
| SHWD | https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset (Google Drive id 1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX) | MIT (nhãn trên repo) | 7,581 ảnh VOC; `hat`=đầu có mũ, `person`=đầu không mũ; negatives phần lớn từ SCUT-HEAD. Chỉ dùng để eval |
| CHV | https://github.com/ZijianWang-ZW/PPE_detection (Google Drive id 1fdGn67W0B7ShpBDbbQpUF0ScPQa4DR0a) | "open for free use", yêu cầu trích dẫn repo + bài Sensors 2021 | 1,330 ảnh; person, vest, 4 màu mũ |
| Le2i | TODO(verify) | TODO(verify) | |
| UR Fall | https://fenix.ur.edu.pl/~mkepski/ds/uf.html | **CC BY-NC-SA 4.0** (trang dataset, kiểm 2026-10-25); trích Kwolek & Kępski, CMPB 117(3) 2014 | 30 fall (cam0 ngang sàn + cam1 trần) + 40 ADL (cam0). mp4 depth\|RGB 640×240 @30 fps, RGB 320×240. Nhãn từng frame −1/0/1 (`urfall-cam0-*.csv`). `tools/urfall_prepare.py` |
| CAUCAFall | TODO(verify) | TODO(verify) | |
| Multiple Cameras Fall (Montréal) | https://www.iro.umontreal.ca/~labimage/Dataset/ (+ Erratum-MCF.pdf, 2025-11-28) | Trang chỉ yêu cầu trích dẫn TR 1350 (Auvinet et al., DIRO 2010); **license không ghi, TODO(verify)** | 24 scenario × 8 cam treo tường 2.8 m, 720×480, **30 fps thật** (header AVI ghi 120; đo tốc độ đi 1.1 m/s ở 30 fps). 22 scenario có ngã (23 cú), 23–24 chỉ confounder. Delay khớp số frame ở 23/24 scenario; chute24 cam8 lệch nên bỏ. `tools/mcfd_prepare.py` |

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

### 2026-10-22 — SHWD (miền thứ ba) & gán PPE độc quyền
- SHWD (VOC2028, MIT, 7,581 ảnh): box `hat` = **cả đầu + mũ** (khác `helmet` SH17/CHV chỉ phần mũ) → **không dùng để train** (định nghĩa box lệch, đa số crop là cận mặt). Dùng để **đánh giá có đội mũ hay không theo người** (`eval/shwd_helmet_eval.py`, chỉ ảnh có ≥1 `hat`).
- M2, ROI đã tune, SHWD test: 1,492 người đội / 235 không; recall 0.940, FPR 0.264; recall theo v: 0.856 (v<0.2) → 0.972 (v≥0.8); AUC(v) 0.667 > person conf 0.607 → xu hướng visibility lặp lại ở miền thứ ba.
- FPR tăng theo mức đông người (SH17 0.025, CHV 0.18, SHWD 0.26). Soi ảnh: lẫn (a) lỗi ghép đầu GT↔người trong đám đông (nhiễu đo) và (b) mũ người bên cạnh/phía sau lọt vào head ROI (lỗi thật; head_up_ratio=1.0 làm nặng thêm) — đúng kịch bản làm luật OR (B2) thất bại.
- Thử **gán độc quyền** (`part_assign.exclusive`: một box PPE → một người, theo containment rồi khoảng cách tâm ROI):

| Val | recall off→on | FPR off→on | J off→on |
|---|---|---|---|
| SHWD | 0.942→0.908 | 0.231→**0.174** | 0.711→**0.734** |
| SH17 head | 0.912→0.877 | 0.025→0.024 | 0.887→0.853 |
| CHV head | 0.983→0.967 | 0.250→0.250 | 0.733→0.717 |

  Test cùng xu hướng (SHWD FPR 0.264→0.191, recall 0.940→0.895). → Đánh đổi; **mặc định tắt**, quyết định trên C2 theo độ đông thực tế.

### 2026-10-22 — Association đa cam trên **WILDTRACK** (dữ liệu thật, trước C2) — `eval/eval_association.py`
- **Dataset:** WILDTRACK (EPFL CVLab; Chavdarova et al., arXiv 1707.09299). 7 cam tĩnh, 1920×1080, 400 frame có nhãn (2 fps), khoảng 24 người/frame, 42,721 box GT (đếm từ `annotations_positions`). Link chính thức `documents.epfl.ch/.../Wildtrack_dataset_full.zip` bị reset kết nối từ mạng này nên tải bản mirror `huggingface.co/datasets/disl/my_dataset` (`Wildtrack_dataset_full/`; README trùng README gốc). **License: trang EPFL không ghi, TODO(verify).** Mirror khác (ShantyCam) ghi "CC-BY (research)", chưa xác minh.
- **Chuẩn bị** (`tools/wildtrack_prepare.py`): `H = diag(0.01,0.01,1)·inv(K·[r1 r2 t])` (thế giới tính bằng cm, mặt phẳng z=0). FOV polygon lấy hình học: bao lồi các điểm lưới sàn chiếu vào trong ảnh, không dùng GT. ROI eval là lưới 12×36 m (`--roi -3 -9 9 27`).
- **Kiểm tra calibration:** chiếu điểm giữa đáy box GT xuống sàn rồi so với vị trí GT, RMS từng cam 0.073–0.153 m, median 0.070–0.120 m.
- **Detector:** `yolo11m-pose@960` + ByteTrack (`tools/wildtrack_detect.py`). Tracker chạy ở **2 fps** nên IDF1 thời gian bị bi quan. Recall detection (IoU ≥ 0.5) đạt 0.436 trên 7 cam (C1: 0.317), precision 0.818. WILDTRACK có đám đông dày và người ở xa.
- Commit `9459826`. Report: `eval/reports/association/wildtrack_*.json`.

**Chẩn đoán và sửa (oracle = box GT + track ID GT):**
1. **FOV gate quá chặt:** trong 83,957 cặp cùng người, 2,443 cặp ở gần nhau nhưng bị loại vì điểm chân nằm ngay ngoài FOV polygon của cam kia (người bị cắt ở mép ảnh). Đã thêm `association.fov_margin_m`. Với d_max 0.8: margin 0 → 0.5 m nâng pairwise recall 0.940 → 0.973, count MAE 3.12 → 1.09.
2. **Điểm chân từ ankle bị lệch:** ankle cao hơn sàn khoảng 8–10 cm, nên khi chiếu bằng homography mặt sàn sẽ bị đẩy ra xa khoảng `h·dist/cam_height`. Trên 18,773 detection khớp GT (không bị cắt đáy), sai số median theo từng rule: trung điểm ankle 0.681 m (p90 1.327), ankle + 0.04h 0.321, **đáy bbox 0.230 m (p90 0.623)**, x lấy từ ankle + y lấy từ đáy bbox 0.231. Đã thêm `association.foot_point: bbox_bottom` (**thay đổi §7.1, chờ xác nhận**).

**Kết quả (cấu hình mặc định: d_max 0.8, fov_margin 0.5, foot bbox_bottom, 7 cam):**

| Input | floor err median | pair P | pair R | pair F1 | IDF1 | IDF1 assoc-only | count MAE (bias) | MAE không association: sum / max |
|---|---|---|---|---|---|---|---|---|
| Oracle (box GT) | 0.089 m | 0.991 | 0.973 | 0.982 | 0.952 | 0.958 | 1.09 (+0.98) | 81.72 / 0.44 |
| YOLO11m-pose, foot = ankles (rule cũ) | 0.662 m | 0.395 | 0.344 | 0.367 | 0.230 | 0.407 | 8.26 (+8.06)¹ | 30.35 / 12.28 |
| YOLO11m-pose, foot = bbox_bottom | 0.228 m | 0.725 | 0.854 | 0.784 | 0.348 | 0.613 | 3.35 (+2.34) | 33.07 / 11.73 |

¹ Dòng ankles lấy từ sweep `wildtrack_yolo11m_sweep.json` (d_max 0.8, fov_margin 1.0).

- **Sweep d_max (detector, bbox_bottom, margin 0.5):** d_max 0.5 cho F1 0.804 nhưng count MAE 4.34; d_max 0.8 cho F1 0.784 và count MAE 3.35. d_max ≥ 1.0 làm F1 giảm (1.5: 0.61–0.66). Giữ 0.8 và tune lại trên C2.
- **Theo số cam** (`wildtrack_*_cams_*.json`, chạy detector):

| Cam | pair F1 | count MAE (bias) | det recall |
|---|---|---|---|
| C1 | — | 14.46 (−14.46) | 0.317 |
| C1, C3 | 0.763 | 11.47 (−11.47) | 0.376 |
| C1, C3, C5 | 0.791 | 10.30 (−10.29) | 0.435 |
| 7 cam | 0.784 | 3.35 (+2.34) | 0.436 |

  Với oracle: C1 có count MAE 0.30, C1+C3 có 0.77, ba cam có 0.64, 7 cam có 1.09. Như vậy với detector thật, sai số đếm do **bỏ sót detection** chi phối, không phải do association.
- **Kết luận:**
  - Association trên mặt sàn đúng khi vị trí chân đúng: oracle đạt pairwise F1 0.98.
  - Với detector thật, nút thắt là (a) sai số điểm chân trên sàn và (b) recall detection trong đám đông.
  - IDF1 thấp (0.35) một phần vì tracker chỉ chạy 2 fps trên WILDTRACK.
  - Baseline "max theo cam" đếm tốt trên WILDTRACK chỉ vì C6/C1 nhìn gần hết quảng trường, nên **không tổng quát** cho bố trí FOV chồng một phần.

### 2026-10-25 — Chuẩn bị nhận dữ liệu C2: timestamp VFR, đồng bộ, chạy thử pipeline (commit `ed07203`, `d6c52bc`, `9868856`)
- **Video điện thoại có frame rate thay đổi (VFR).** Trước đây offline backend tính `ts = frame_idx / fps`. Đã kiểm tra trên video VFR tổng hợp (ffmpeg, 150 frame @30 fps rồi @15 fps): OpenCV `CAP_PROP_POS_MSEC` sau `read()` trả **đúng pts của frame vừa đọc**, còn cách tính theo index lệch 2.4 s tại frame 150. Đã đổi sang pts (`perception.file_timestamps: pts`, `FrameClock`), fallback +1000/fps khi pts thiếu hoặc không tăng.
- **`tools/session_check.py`** (fps / VFR / thời lượng, sự kiện tắt-bật đèn ở đầu + cuối → offset + drift, frame sàn trống, cameras.yaml của session):
  - Session giả 3 cam (độ lệch đặt trước 0 / 1.2 / 0.4 s, cam3 VFR): offset đo được 0 / 1200 / 400 ms, drift 0, VFR của cam3 được phát hiện.
  - WILDTRACK 3 cam ghép thành video 2 fps, cam2 cắt 2 frame đầu: offset đo được 1000 ms (đúng).
- **Chạy thử toàn bộ quy trình C2** trên `data/rehearsal/wt3` (WILDTRACK C1/C4/C6 → session_check → ultralytics_runner (pose + PPE, 11.7–14.4 fps) → analytics_runner (tự nhận cameras.yaml của session) → label_intervals template/convert/validate → run_baselines --ablations → make_evidence → dashboard). Mọi bước chạy được. **Số liệu của lần chạy thử không có ý nghĩa** (nhãn giả; người WILDTRACK không đội mũ) nên không ghi ở đây.

### 2026-10-25 — Association: người bị cắt ở mép dưới ảnh (§7.1) — `eval_association --d-max-truncated` (commit `9868856`)
- Trước đây cờ `truncated_bottom` được gán nhưng association không dùng tới. Đo trên WILDTRACK (ROI −3 −9 9 27, 7 cam), sai số sàn median của detection bị cắt so với không bị cắt:

| Input | n bị cắt | median bị cắt | median không bị cắt |
|---|---|---|---|
| Oracle (box GT) | 1,196 | 0.766 m | 0.087 m |
| YOLO11m-pose@960 | 208 | 0.606 m | 0.227 m |

- Thay đổi: cặp có một obs bị cắt dùng gate `d_max_truncated_m`, chi phí Hungarian = khoảng cách / gate; vị trí sàn của cụm (global_id, zone) lấy từ obs không bị cắt nếu có.
- Sweep (d_max 0.8, fov_margin 0.5, foot bbox_bottom; report `eval/reports/association/wildtrack_*_truncated_sweep.json`):

| d_max_truncated | Oracle F1 | Oracle IDF1 | Oracle count MAE | Det F1 | Det IDF1 | Det count MAE |
|---|---|---|---|---|---|---|
| = d_max (0.8) | 0.982 | 0.953 | 1.09 | 0.784 | 0.349 | 3.36 |
| **1.2** | **0.985** | 0.959 | **0.92** | **0.786** | 0.348 | 3.35 |
| 1.6 | 0.985 | 0.960 | 0.92 | 0.786 | 0.348 | 3.35 |
| 2.0 | 0.985 | 0.959 | 0.94 | 0.784 | 0.348 | 3.37 |

- Chọn 1.2 m. Hiệu ứng nhỏ vì WILDTRACK chỉ khoảng 3 % box bị cắt. C2 (cam thấp, phòng nhỏ, cảnh C4 đứng sát cam) sẽ có nhiều hơn, cần tune lại ở đó. Benchmark bán tổng hợp không dùng association nên không bị ảnh hưởng.

### 2026-10-25 — WILDTRACK: pose imgsz 1280 vs 960 (commit `9868856`)
- `tools/wildtrack_detect.py --imgsz 1280`, 7 cam, cùng cấu hình association mặc định.

| imgsz | detections | det recall | det precision | pair F1 | IDF1 | count MAE (bias) | floor err median |
|---|---|---|---|---|---|---|---|
| 960 | 22,746 | 0.436 | 0.818 | 0.786 | 0.348 | 3.35 (+2.35) | 0.227 m |
| 1280 | 27,135 | 0.488 | 0.768 | 0.741 | 0.356 | 7.04 (+6.88) | 0.239 m |

- 1280 bắt thêm người ở xa nhưng điểm chân của họ nhiễu hơn, nên cụm bị tách ra và đếm thừa gấp đôi. **Giữ 960.** C2 là phòng nhỏ, người to trong ảnh, nên không cần imgsz lớn.

### 2026-10-25 — Fall (rule-based) trên **UR Fall** — `eval/eval_fall.py` (commit `d03b492`)
- **Dữ liệu:** `tools/urfall_prepare.py`, 30 cảnh té (cam0 ngang sàn + cam1 trần) + 40 ADL (cam0), RGB 320×240 @30 fps. Pose `yolo11m-pose@640`, ByteTrack, không bật PPE. GT episode = frame "falling" đầu tiên → frame "lying" cuối cùng. ADL (kể cả nằm xuống có chủ đích) chỉ là mẫu âm. Có đúng 1 người mỗi sequence nên giữ person conf cao nhất làm actor, global id 0. δ = 2 s.
- **Lỗi tìm được:** lịch sử hông (drop) lưu theo `(cam, track_id)`. Khi người ngã, tracker hay đổi ID nên mất drop, score bị nhân 0.5 (dưới ngưỡng). Đổi sang `(cam, global_id)`. cam0, tất cả sequence, hold 0.5: recall 0.267 → 0.433, FP giữ nguyên 8. Có test hồi quy `test_fall_survives_track_id_switch`.
- **Sweep trên val** (sequence số lẻ; score_threshold, hold_s, drop_ratio, drop_memory_s): chỉ `hold_s` tác động mạnh. 0.5 → 0.2: recall 0.267 → 0.600, FP 5 → 5. Nguyên nhân là sequence kết thúc ngay sau khi nằm (đoạn "lying" chỉ 0.2–2 s). Chọn **0.2** (0.1 cho F1 val 0.667 so với 0.621, cùng FP, nhưng dễ bị kích bởi nhiễu một frame).
- **Kết quả cam0** (report `eval/reports/fall/urfall_cam0_{val,test,all}`):

| Split | hold_s | Seq | Falls | TP | FP | FN | Precision | Recall | F1 | FA/giờ | latency median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 0.5 | 35 | 15 | 4 | 5 | 11 | 0.444 | 0.267 | 0.333 | 91.0 | 1700 ms |
| val | 0.2 | 35 | 15 | 9 | 5 | 6 | 0.643 | 0.600 | 0.621 | 91.0 | 1300 ms |
| **test** | 0.5 | 35 | 15 | 9 | 3 | 6 | 0.750 | 0.600 | 0.667 | 57.8 | 1567 ms |
| **test** | **0.2** | 35 | 15 | 13 | 7 | 2 | 0.650 | **0.867** | **0.743** | 134.8 | 1167 ms |
| all | 0.2 | 70 | 30 | 22 | 12 | 8 | 0.647 | 0.733 | 0.688 | 112.3 | 1217 ms |

  - Trên test, hold 0.2 tăng recall nhưng FA/giờ cũng tăng (57.8 → 134.8): đây là đánh đổi thật, không phải cải thiện thuần. Báo động sai đến từ ADL nằm hoặc ngồi xuống nhanh (adl-10, 16, 30, 32, 34, 38, 40). FA/giờ cao vì thời lượng ADL rất ngắn (tổng khoảng 4 phút).
- **Đa góc (chỉ 30 cảnh té, không có ADL cho cam1):**

| Cam | Fusion | TP | FP | Recall | Precision |
|---|---|---|---|---|---|
| cam0 (ngang) | — | 22 | 0 | 0.733 | 1.000 |
| cam1 (trần) | — | 13 | 2 | 0.433 | 0.867 |
| cam0 + cam1 | vmean (§7.7) | 16 | 0 | 0.533 | 1.000 |
| cam0 + cam1 | max (ablation) | 22 | 2 | 0.733 | 0.917 |

  - **Fuse theo v (§7.7) kém hơn cam0 một mình.** Rule tư thế (tỉ lệ bbox, góc thân) không có nghĩa khi nhìn từ trần, mà v chỉ đo được "thấy rõ" chứ không đo được "góc nhìn này có thông tin không". `max` thì bằng cam0 nhưng thêm FP của cam1. **Hệ quả cho thiết kế:** fusion đa cam cho fall cần trọng số theo mức hữu ích của góc nhìn (hình học cam, hoặc học), không chỉ visibility. C2 dùng cam đặt thấp, ngang (giống cam0) nên đánh giá lại trên đó.
- Giới hạn: té giả trong phòng, sequence ngắn; cam0 và cam1 không đồng bộ chặt (ghép theo frame index, sai lệch tối đa 50 ms).

### 2026-10-26 — Fall đa góc trên **Multiple Cameras Fall (MCFD)** + cấu hình cuối (commit `b72d115`)
- **Chuẩn bị** (`tools/mcfd_prepare.py`): 24 scenario × 8 cam, pose `yolo11m-pose@640` trên đoạn có nhãn; thời gian chung = (frame − delay − 1)/30 s. GT là code 2 nối tiếp code 3 trong frame của cam tham chiếu (đã áp dụng erratum). Có 23 cú ngã ở 22 scenario; scenario 23–24 chỉ có confounder (ngồi, cúi, nằm sofa). **fps thật là 30**: đo tốc độ đi bằng calibration của báo cáo, ra 1.1 m/s ở 30 fps, so với 4.4 m/s nếu là 120 fps.
- **Chẩn đoán** (cấu hình cũ, tất cả (cú ngã, cam)): người được phát hiện ở 59.2 % số frame trong lúc ngã/nằm; 74.5 % cặp có tư thế nằm (pose ≥ 0.6) nhưng chỉ 42.4 % có cả drop. Cam cao nhìn xiên làm độ hạ của hông trong ảnh nhỏ đi.
- **Tune trên val** (scenario lẻ, 11 cú ngã; UR Fall val): `drop_ratio` 0.3 → 0.2. MCFD 8 cam vmean F1 0.333 → 0.476 (FA 44 → 56/giờ); UR Fall F1 0.621 → 0.625. `score_threshold` giữ 0.6 (vmean 0.3–0.5 tăng FA nhiều hơn recall; max gần như không đổi theo ngưỡng).
- **Kết quả test, cấu hình cuối** (hold 0.2, drop 0.2):

MCFD test (scenario chẵn, 12 cú ngã; trung bình ± std trên các tổ hợp cam, `eval/reports/fall/final_mcfd_test_subsets`):

| Số cam | Fusion | Recall | Precision | F1 | FA/giờ |
|---|---|---|---|---|---|
| 1 | — | 0.417 ± 0.208 | 0.579 ± 0.186 | 0.474 ± 0.201 | 42.6 ± 15.0 |
| 2 | vmean | 0.549 ± 0.154 | 0.690 ± 0.137 | 0.605 ± 0.140 | 37.1 ± 16.0 |
| 2 | max | 0.583 ± 0.113 | 0.615 ± 0.057 | 0.596 ± 0.083 | 55.1 ± 9.5 |
| 4 | vmean | 0.563 ± 0.196 | 0.661 ± 0.178 | 0.605 ± 0.189 | 39.2 ± 12.1 |
| 4 | max | 0.625 ± 0.054 | 0.580 ± 0.043 | 0.602 ± 0.048 | 68.9 ± 6.3 |
| **8** | **vmean** | **0.667** | **0.727** | **0.696** | **38.2** |
| 8 | max | 0.583 | 0.538 | 0.560 | 76.4 |

UR Fall (cấu hình cuối): cam0 test P/R/F1 0.682/1.000/0.811, FA 134.8/giờ. Chỉ tính cảnh té: cam0 recall 0.833 (FP 0), cam1 trần 0.433 (FP 2), cam0 + cam1 vmean 0.600 (FP 0), max 0.800 (FP 2).

- **Kết luận:**
  - Đa cam giúp phát hiện ngã: 8 cam vmean có F1 0.696, so với 0.474 ± 0.201 của một cam, và FA/giờ còn thấp hơn. Độ lệch giữa các cam đơn rất lớn (std recall 0.21), tức là vị trí cam quan trọng; fusion bù được điểm này.
  - **vmean (§7.7) tốt nhất khi mọi góc nhìn đều có thông tin** (MCFD, cam treo cao nhìn xiên). **Cam nhìn thẳng từ trên xuống thì làm hỏng vmean** (UR Fall cam1): rule tư thế không có nghĩa từ góc đó, nhưng v vẫn cao. Nên dùng vmean và loại cam top-down khỏi phần fuse fall (đề xuất, chưa làm, **chờ người dùng**).
  - Số mẫu nhỏ: 12 cú ngã ở test MCFD, 15 ở test UR Fall. Chênh lệch 1–2 sự kiện là trong biên nhiễu; std giữa các tổ hợp cam cũng lớn.
  - Điểm nghẽn còn lại: recall detection lúc nằm (59 %) trên ảnh góc rộng bị méo, và rule drop. Bước tiếp theo nếu cần là ST-GCN hoặc mô hình học trên chuỗi skeleton.

### 2026-10-26 — Fall học từ chuỗi skeleton (`fall.method: seq`) so với rule (commit `b358974`)
- **Model:** feature mỗi frame (`src/behavior/seq_features.py`): 17 kpt chuẩn hoá theo tâm/chiều cao bbox, tỉ lệ bbox, vận tốc dọc, dh, mask. Cửa sổ 45 frame (1.5 s). 1D-CNN giãn (dilated) 3 tầng, pool mean+max. BCE có pos_weight, lật ngang làm augmentation, 30 epoch, Adam 1e-3.
- **Dữ liệu train** (`training/fall/train_seq.py`): **chỉ split val** (sequence lẻ) của UR Fall + MCFD, mọi cam: 27,554 cửa sổ từ 47 cặp (sequence, cam), 9.6 % dương. Test (số chẵn) không dùng khi train. `seq_threshold` cố định trước là 0.5, hold 0.2, fusion giữ như rule. Không tune gì trên test.
- **Kết quả test** (3 seed: 0 / 1 / 2):

| Test | Rule | Seq seed 0 | Seq seed 1 | Seq seed 2 |
|---|---|---|---|---|
| UR Fall cam0, F1 (FA/giờ) | 0.811 (134.8) | 0.828 (38.5) | 0.769 (173.4) | 0.667 (134.8) |
| MCFD 8 cam vmean, F1 (FA/giờ) | 0.696 (38.2) | 0.762 (12.7) | 0.762 (12.7) | 0.762 (12.7) |
| MCFD 4 cam vmean, F1 | 0.605 ± 0.189 | 0.779 ± 0.045 | 0.786 ± 0.035 | 0.827 ± 0.053 |
| MCFD 1 cam, F1 | 0.474 ± 0.201 | 0.634 ± 0.147 | 0.620 ± 0.142 | 0.630 ± 0.170 |

  (MCFD: trung bình ± std trên các tổ hợp cam, report `eval/reports/fall/seq_s*_mcfd_test_subsets`.)
- **Kết luận:**
  - Trên MCFD (đa cam, cam treo cao), seq **tốt hơn rule ổn định ở mọi seed và mọi số cam**, FA/giờ giảm khoảng 3 lần. Latency 8 cam cũng giảm từ 1.9 s xuống khoảng 0.7–0.9 s.
  - Với seq, vmean luôn tốt hơn max: score đã được hiệu chỉnh hơn nên phép trung bình có ý nghĩa hơn.
  - Trên UR Fall một cam, seq **không** tốt hơn rule một cách tin cậy (F1 0.667–0.828 tuỳ seed), vì dữ liệu train ít (15 cảnh té UR Fall ở split val). Không chọn seed hay ensemble sau khi xem test.
  - Mặc định vẫn là `method: rule` (§7.7 baseline). seq là biến thể; nếu dùng thì nên train lại có thêm C2.

### 2026-10-26 — Xâm nhập vùng cấm và tụ tập trên **WILDTRACK** — `eval/eval_zones.py` (commit `920a711`)
- **Cách làm:** GT vị trí sàn của mọi người ở mọi frame có nhãn (2 fps, 200 s, khoảng 24 người/frame). Vùng chọn tất định từ GT, không nhìn dự đoán:
  - 6 vùng cấm 2 × 2 m, xếp hạng theo số người khác nhau đi qua, lấy nhóm ở giữa (không lấy ô đông nhất).
  - 8 vùng rủi ro 4 × 4 m đông nhất; K = phân vị 60 của số người GT (chạy với 4 vùng và phân vị 75 thì chỉ có 2 sự kiện).
  - GT episode = ở trong vùng liên tục ≥ T (T_intr 1 s, crowd T 5 s), cùng định nghĩa với alerter. Chạy toàn bộ hệ thống (7 cam → association → floor theo global id → BehaviorEngine). Cooldown 0. Ghép theo vùng, không theo người, δ = 2 s.
- **Kết quả** (report `eval/reports/zones/wildtrack_*_crowd8`):

| Input | Loại | GT events | TP | FP | FN | Precision | Recall | F1 | FA/giờ |
|---|---|---|---|---|---|---|---|---|---|
| Box GT (oracle) | INTRUSION | 54 | 51 | 5 | 3 | 0.911 | 0.944 | 0.927 | 90.2 |
| Box GT (oracle) | CROWD | 15 | 12 | 3 | 3 | 0.800 | 0.800 | 0.800 | 54.1 |
| YOLO11m-pose | INTRUSION | 54 | 32 | 21 | 22 | 0.604 | 0.593 | 0.598 | 378.9 |
| YOLO11m-pose | CROWD | 15 | 6 | 5 | 9 | 0.545 | 0.400 | 0.462 | 90.2 |

- Đếm người trong vùng rủi ro (detector): MAE 0.38–0.79 người/vùng; ở vài vùng bias âm tới −0.68 (đếm thiếu), so với oracle MAE ≤ 0.39.
- **Kết luận:** logic vùng + association + alerter đúng khi detection đúng (oracle F1 0.93 / 0.80). Với detector thật, lỗi chủ yếu đến từ recall detection trong đám đông dày (det recall 0.436, xem mục association). Hai vùng cấm (F3, F6) bỏ sót toàn bộ. FP của oracle là do điểm chân dao động quanh mép vùng. WILDTRACK đông hơn nhiều so với kịch bản C2 (3 người), nên đây là trường hợp khó.
- Giới hạn: 200 s, 2 fps; FA/giờ ngoại suy từ thời lượng ngắn nên dao động lớn.

### 2026-10-26 — DeepStream 7.1 với model dự án: so khớp offline + FPS (commit `f88ee71`)
- **Pipeline:** `src/pipeline/ds_app.py` trong `ppe-ds:7.1` (WSL2, RTX 4060 Laptop): nvurisrcbin ×N → streammux → YOLO11m-pose@960 FP16 (parser DeepStream-Yolo-Pose `2fa5bd8`) → NvDCF perf → PPE M2 @320 FP16 trên crop người (parser DeepStream-Yolo `2894bab`) → probe → JSONL. ONNX export bằng `tools/export_ds_onnx.py`.
- **Dữ liệu:** `data/rehearsal/wt3` (WILDTRACK C1/C4/C6 ghép thành video 2 fps, MPEG-4, 400/398/400 frame). Tham chiếu là JSONL của backend offline đã có sẵn (`yolo11m-pose@960` + ByteTrack + M2 trên crop).
- **Sửa trong lúc nối** (số trước/sau cùng đo bằng `eval.compare_backends`, chạy trước commit):
  - PPE không được gắn vào người vì ghép theo `id()` của wrapper pyds → ghép theo `object_id`.
  - Crop SGIE không nới `crop_pad` như lúc train → mức khớp PPE với offline: head 0.901 → 0.944, helmet 0.930 → 0.962, shoes 0.818 → 0.925.
  - Bbox người lấy từ tracker NvDCF làm association kém (bảng dưới) → mặc định xuất bbox detector.
- **So khớp DS vs offline** (`eval.compare_backends`, ghép người IoU ≥ 0.5 theo từng frame; report `eval/reports/backends/wt3_offline_vs_ds_*.json`):

| Bbox xuất ra | Frame khớp / ts lệch | Người ref / DS | Recall / Precision | IoU TB | Keypoint lệch median / p90 (× chiều cao) | PPE agreement head / helmet / vest / shoes | Track (số / độ dài TB) |
|---|---|---|---|---|---|---|---|
| detector (mặc định) | 1198 / 0 | 9608 / 9208 | 0.837 / 0.873 | 0.874 | 0.009 / 0.018 | 0.948 / 0.963 / 0.987 / 0.922 | 289 / 31.9 (offline 398 / 24.1) |
| tracker | 1198 / 0 | 9608 / 9208 | 0.829 / 0.865 | 0.872 | 0.009 / 0.018 | 0.948 / 0.963 / 0.987 / 0.924 | 289 / 31.9 |

- **Association trên GT WILDTRACK** (`eval.eval_association --cams C1 C4 C6`, d_max 0.8, cam đổi tên cam1/2/3 → C1/C4/C6; report `eval/reports/backends/wt3_assoc_*.json`):

| Input | det recall / precision | pair P / R / F1 | IDF1 (assoc-only) | count MAE | floor err median |
|---|---|---|---|---|---|
| Offline | 0.325 / 0.689 | 0.769 / 0.856 / 0.810 | 0.299 (0.677) | 6.81 | 0.202 m |
| DS, bbox detector | 0.310 / 0.686 | 0.779 / 0.862 / 0.818 | 0.324 (0.758) | 7.41 | 0.240 m |
| DS, bbox tracker | 0.297 / 0.657 | 0.658 / 0.824 / 0.732 | 0.309 (0.754) | 7.38 | 0.239 m |

- **FPS theo số stream** (`eval.bench_fps_ds`, video wt3 lặp, không pacing, 60 s sau frame đầu, bỏ 30 frame warmup/stream; GPU util lấy từ bảng `nvidia-smi`, bộ nhớ là toàn GPU; report `eval/reports/bench_fps_ds_wt3`):

| Stream | fps tổng | fps / stream | GPU util TB | GPU mem peak (MiB) | ≥ 25 fps |
|---|---|---|---|---|---|
| 1 | 67.7 | 67.7 | 68 % | 2504 | có |
| 2 | 81.2 | 40.6 | 85 % | 2638 | có |
| 3 | 78.3 | 26.1 | 88 % | 2757 | có |
| 4 | 79.0 | 19.8 | 87 % | 2969 | không |

  So với backend offline cùng video (`eval/reports/bench_fps_wildtrack1080`, `baaaeb4`): 15.6–16.7 fps tổng → DeepStream nhanh hơn khoảng 5 lần, **3 cam 1080p đạt real-time 25 fps** trên RTX 4060 Laptop.
- **Kết luận:**
  - Hai backend cho cùng schema và cùng kết quả ở mức hệ thống (association F1 0.818 so với 0.810). Khác biệt còn lại đến từ NMS (parser cố định IoU 0.45, offline 0.7), tracker (NvDCF giữ track dài hơn, ít track hơn ByteTrack) và FP16.
  - Sai số vị trí sàn của DS cao hơn (0.240 so với 0.202 m) dù dùng bbox detector. Chưa rõ nguyên nhân (letterbox vuông 960×960 so với 960×544 của Ultralytics, FP16); cần kiểm tra lại trên C2.
  - GPU đã bão hoà từ 2 stream (khoảng 80 fps tổng). Muốn thêm stream: YOLO11s-pose, INT8, hoặc `interval` > 0 cho PGIE (TODO, chưa đo).
  - wt3 chỉ 2 fps nên tracker chạy ở điều kiện khó; trên C2 (25–30 fps) cần đo lại khác biệt tracker.
