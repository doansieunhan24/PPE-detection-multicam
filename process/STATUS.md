# STATUS — cập nhật lần cuối 2026-10-27 (s12)

**Phase hiện tại:** P0, P1, P2 xong (cả DeepStream). Code P5–P7 xong; **P7 demo live end-to-end đã chạy** (3 RTSP → DS → ZMQ → analytics → SQLite/dashboard + smart record + OSD). **Chờ dữ liệu C2 (P4).**

## Đã xong
- (s10) Người dùng xác nhận 4 thay đổi spec: dt_weighting, p calibration + theta_neg −0.5, foot_point = bbox_bottom, d_max_truncated_m = 1.2
- P1 offline backend (YOLO11m-pose@960 + ByteTrack → JSONL, timestamp theo pts) + replay
- **PPE detector M2** (YOLO11s@320, SH17+CHV person crops): CHV test helmet AP50 0.957, vest 0.913
- C1 core: part ROI, gán PPE, visibility, evidence (dt-weighted), state machine, Dempster–Shafer, B0–B4, alert rules
- Association đa cam, behavior rule-based, alerts/dashboard/Telegram
- Eval: event metrics, `run_baselines --ablations`, `run_sensitivity`, `bench_fps`, `eval_detector`, `compare_evidence`, bán tổng hợp, SHWD, `eval_association`
- Fall: UR Fall cam0 F1 0.811; MCFD 8 cam vmean F1 0.696 (học 1D-CNN 0.762)
- Intrusion / crowd trên WILDTRACK: oracle F1 0.927 / 0.800, YOLO11m-pose 0.598 / 0.462
- Association WILDTRACK: oracle pair F1 0.985, YOLO11m-pose 0.786
- Chuẩn bị nhận C2: `session_check`, `label_intervals convert`, calibrate LOO
- P0 (s10): DS 7.1 WSL2, 3 RTSP TCP ~30 fps/stream (model mẫu)
- DS với model dự án (s11): khớp offline (assoc F1 0.818 vs 0.810), 1/2/3/4 stream → 67.7/40.6/26.1/19.8 fps mỗi stream
- **Demo live (s12):** `ds_app --zmq-pub --sr-bind --osd`, `src/analytics_live.py` (ZMQ hoặc `--replay`), `LiveSyncer`, smart record, dashboard có video WebRTC. Chạy 180 s trên wt3 RTSP 2 fps: 2 fps/cam, 0 frame mất, 91 alert, độ trễ alert median 554 ms, 37/37 clip bằng chứng. Sửa crash RTSP (libproxy GIO). Lệnh: `docker/README.md` mục "Demo live"
- 155 test pass, ruff sạch (pytest cần `--basetemp` ghi được trên máy này)

## Đang dở
- Thay đổi s12 **chưa commit** (chờ người dùng)

## Blocker / chờ người dùng
- Quay C2 theo `docs/c2_shooting_script.md`; trả lời câu hỏi mở §16
- License WILDTRACK: TODO(verify)
- Fuse fall: có thêm cờ loại cam nhìn từ trên xuống không?
- Telegram thật: cần `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` trong `.env`

## Ưu tiên tiếp theo
1. Có C2: README bước 2–6 (`session_check` → `calibrate_homography` → `ultralytics_runner` → `label_intervals` → `run_baselines --ablations` → `run_sensitivity`), tune trên s01 (val), báo cáo trên s02/s03
2. Đo độ trễ alert / FPS live ở 25–30 fps (C2), snapshot có OSD cho alert live
3. Tune θ_obs/θ_pos/θ_neg, τ, T_alert, d_max, d_max_truncated trên C2 val

## Session gần nhất
- [2026-10-27_s12.md](2026-10-27_s12.md) (trước đó: s05–s11)
