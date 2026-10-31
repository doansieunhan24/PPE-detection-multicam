# STATUS — cập nhật lần cuối 2026-10-31 (s14)

**Phase hiện tại:** P0, P1, P2 xong (cả DeepStream). Code P5–P7 xong; demo live end-to-end đã chạy. **Chờ dữ liệu C2 (P4).** Từ s14: **bài chỉ xử lý từng camera** (không ghép người giữa các cam, `association.mode: none`; bài chủ yếu là detection), đánh giá C2 chấm từng camera riêng, baseline B0/B1/B4 vs P; không chạy thêm WILDTRACK.

## Đã xong
- P1 offline backend (YOLO11m-pose@960 + ByteTrack → JSONL) + replay; DS 7.1 khớp offline, 3 cam 1080p 26.1 fps/stream
- PPE detector M2 (YOLO11s@320, SH17+CHV crops): CHV test helmet AP50 0.957, vest 0.913
- C1 core: part ROI, gán PPE, visibility, evidence (dt-weighted), state machine, Dempster–Shafer, B0–B4, alert rules
- Behavior: fall (UR Fall cam0 F1 0.811; MCFD 8 cam vmean 0.696, 1D-CNN 0.762), intrusion/crowd (WILDTRACK)
- Demo live (s12): DS → ZMQ → `analytics_live` → SQLite/dashboard, smart record, OSD, snapshot (s13)
- `eval/run_c2.py` một lệnh val → test (s13)
- **(s14)** `association.mode: none` (mặc định) / `ground_plane`; eval chấm từng camera (`--per-cam auto`, bỏ B2/B3 và numcams); CROWD đếm theo cam thấy nhiều người nhất; `crowd.K` = 3; CLAUDE.md §2/§7/§10/§17 sửa theo hướng từng camera; kịch bản C2 = 11 take (`docs/c2_shooting_script.md/.pdf`); camera thật 2 × Tiandy TC-C131N qua LAN (`configs/cameras_lan.yaml`, mật khẩu trong `.env`)
- 175 test pass, ruff sạch (pytest cần `--basetemp` ghi được trên máy này)

## Đang dở
- Không có (s14 đã commit: `9d08c4d`, `9438fc0`)

## Blocker / chờ người dùng
- Quay C2; trả lời câu hỏi mở §16; license WILDTRACK TODO(verify); Telegram cần token trong `.env`

## Ưu tiên tiếp theo
1. Có C2: `session_check` → `calibrate_homography` (cam thấy vùng cấm; ống kính Tiandy méo, xử lý trước) → `ultralytics_runner` → `label_intervals` → sửa `configs/c2_split.yaml` → `python -m eval.run_c2 --steps check` → `python -m eval.run_c2 --jobs 4`
2. Đo độ trễ alert / FPS live ở 25 fps với cam Tiandy

## Session gần nhất
- [2026-10-31_s14.md](2026-10-31_s14.md) (trước đó: s13, s12, s05–s11)
