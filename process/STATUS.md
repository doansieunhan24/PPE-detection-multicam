# STATUS — cập nhật lần cuối 2026-10-29 (s13)

**Phase hiện tại:** P0, P1, P2 xong (cả DeepStream). Code P5–P7 xong; **P7 demo live end-to-end đã chạy** (3 RTSP → DS → ZMQ → analytics → SQLite/dashboard + smart record + OSD). **Chờ dữ liệu C2 (P4).** Chuỗi đánh giá C2 đã gom thành một lệnh (`python -m eval.run_c2`), đã chạy thử bằng GT thật của WILDTRACK.

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
- **(s13) Sẵn sàng cho C2:** `eval/run_c2.py` (check → calib p/v_learned trên val → tune từng phương pháp trên val → B0–B4 + B0/B1 mỗi cam + P + ablation trên test → sensitivity → số cam 1/2/3 → `REPORT.md`), bảng thêm từng item PPE, behavior, association theo `tracks.csv`; CROWD gán nhãn không gt_id; `label_intervals thumbs` (ảnh track để điền gt_id); cache observe (toàn chuỗi chạy thử 100 s, cùng kết quả); snapshot có overlay cho alert live
- 168 test pass, ruff sạch (pytest cần `--basetemp` ghi được trên máy này)

## Đang dở
- Không có (s13 đã commit)

## Blocker / chờ người dùng
- Quay C2 theo `docs/c2_shooting_script.md`; trả lời câu hỏi mở §16
- License WILDTRACK: TODO(verify)
- Fuse fall: có thêm cờ loại cam nhìn từ trên xuống không?
- Telegram thật: cần `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` trong `.env`

## Ưu tiên tiếp theo
1. Có C2: README bước 2–4 (`session_check` → `calibrate_homography` → `ultralytics_runner` → `label_intervals template/thumbs/convert/validate`), sửa `configs/c2_split.yaml`, rồi `python -m eval.run_c2 --steps check` và `python -m eval.run_c2 --jobs 4` → `eval/reports/c2/REPORT.md`
2. Nếu tune báo biên lưới thì mở rộng `eval/tune_c2.SPACE`; tune thêm d_max / d_max_truncated / ROI nếu association hoặc gán PPE kém trên val
3. Đo độ trễ alert / FPS live ở 25–30 fps (C2); thử snapshot live với DS thật

## Session gần nhất
- [2026-10-29_s13.md](2026-10-29_s13.md) (trước đó: s12, s05–s11)
