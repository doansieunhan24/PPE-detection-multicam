# STATUS — cập nhật lần cuối 2026-10-26 (s11)

**Phase hiện tại:** **P0, P1, P2 xong cả phần DeepStream** (YOLO11m-pose + NvDCF + SGIE PPE trong DS 7.1, JSONL khớp offline, 3 cam 1080p real-time). Code P5–P7 xong và test được. Chuỗi xử lý C2 đã chạy thử từ đầu đến cuối (trên WILDTRACK). **Chờ dữ liệu C2 (P4).**

## Đã xong
- (s10) Người dùng xác nhận 4 thay đổi spec: dt_weighting, p calibration + theta_neg −0.5, foot_point = bbox_bottom, d_max_truncated_m = 1.2
- P1 offline backend (YOLO11m-pose@960 + ByteTrack → JSONL, timestamp theo pts) + replay
- **PPE detector M2** (YOLO11s@320, SH17+CHV person crops): CHV test helmet AP50 0.957, vest 0.913
- C1 core: part ROI, gán PPE, visibility, evidence (dt-weighted), state machine, Dempster–Shafer, B0–B4, alert rules
- Association đa cam (homography, Hungarian, global ID, gate riêng cho bbox bị cắt), behavior rule-based, alerts/dashboard/Telegram (P7)
- Eval: event metrics, `run_baselines --ablations`, **`run_sensitivity`**, **`bench_fps`** (offline), `eval_detector`, `compare_evidence`, bán tổng hợp, SHWD, `eval_association`
- **Fall** (`eval/eval_fall.py`): UR Fall cam0 test P/R/F1 0.682/1.000/0.811; **MCFD test 8 cam vmean F1 0.696** so với 1 cam 0.474 ± 0.201 (FA/giờ 38 vs 43). Cam top-down làm hỏng vmean (UR Fall cam1)
- **Fall học** (`fall.method: seq`, 1D-CNN skeleton, train chỉ trên split val): MCFD 8 cam F1 0.762 / FA 12.7/giờ so với rule 0.696 / 38.2, ổn định qua 3 seed; UR Fall một cam phụ thuộc seed. Mặc định vẫn là rule
- **Intrusion / crowd** trên WILDTRACK (`eval/eval_zones.py`): oracle F1 0.927 / 0.800, YOLO11m-pose 0.598 / 0.462 (lỗi do recall detection trong đám đông)
- Association trên WILDTRACK: oracle pair F1 0.985, YOLO11m-pose F1 0.786 (`docs/experiments.md`)
- **Chuẩn bị nhận C2:** `tools/session_check.py` (VFR, offset + drift từ tắt/bật đèn, frame calibrate, cameras.yaml của session), `label_intervals convert` (phiếu ghi → intervals), calibrate với chỉ số gạch + sai số leave-one-out + FOV an toàn cho cam thấp
- **P0** (s10): image `ppe-ds:7.1` (pyds), `tools/rtsp_sim.sh` (mediamtx Docker), `configs/deepstream/p0_sample_3rtsp.txt`: 3 RTSP qua TCP, model mẫu NVIDIA + NvDCF + tiler, **~30 fps mỗi stream** (= fps nguồn), video `data/p0/tiled_3rtsp.mp4`. Lệnh trong `docker/README.md`
- **DS với model dự án** (s11, `f88ee71`): `ds_app.py` chạy được trên file; so khớp offline trên wt3: người recall/precision 0.837/0.873, keypoint lệch 0.9 % chiều cao, PPE agreement 0.92–0.99; association GT WILDTRACK pair F1 0.818 (offline 0.810). FPS: 1/2/3/4 stream → 67.7/40.6/**26.1**/19.8 fps mỗi stream (`eval/reports/bench_fps_ds_wt3`)
- 146 test pass, ruff sạch (pytest cần `--basetemp` ghi được trên máy này)

## Đang dở
- (không có)

## Blocker / chờ người dùng
- Quay C2 theo `docs/c2_shooting_script.md`; trả lời câu hỏi mở §16
- License WILDTRACK: TODO(verify)
- Fuse fall: giữ vmean (§7.7, tốt nhất trên MCFD); có thêm cờ loại cam nhìn từ trên xuống không?

## Ưu tiên tiếp theo
1. Có C2: README bước 2–6 (`session_check` → `calibrate_homography` → `ultralytics_runner` → `label_intervals` → `run_baselines --ablations` → `run_sensitivity`), tune trên s01 (val), báo cáo trên s02/s03
2. Demo live: `ds_app` với 3 RTSP + OSD/tiler, nối analytics qua JSONL/ZMQ, smart record (P7)
3. Tune θ_obs/θ_pos/θ_neg, τ, T_alert, d_max, d_max_truncated trên C2 val

## Session gần nhất
- [2026-10-26_s11.md](2026-10-26_s11.md) (trước đó: s05–s10)
