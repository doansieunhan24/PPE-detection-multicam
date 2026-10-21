# STATUS — cập nhật lần cuối 2026-10-22 (s02)

**Phase hiện tại:** P2 xong (PPE detector), phần lớn code P5/P6 xong và test được. **Đang chờ dữ liệu thật C2.**

## Đã xong
- P1 offline backend (YOLO11m-pose@960 + ByteTrack → JSONL) + replay
- **PPE detector M2** (YOLO11s@320, SH17+CHV person crops): CHV test helmet AP50 0.957, vest 0.913 → `models/sh17chv_crop_y11s.{pt,onnx}`
- C1 core: part ROI (tune trên val), gán PPE, visibility, evidence (dt-weighted), state machine, Dempster–Shafer, B0–B4, alert rules
- Association đa cam (homography, Hungarian, global ID), behavior rule-based (fall đa góc, immobility, intrusion, crowd)
- Eval: event metrics, `run_baselines` (bảng chính + ablation), `eval_detector`, `compare_evidence`
- Tools thu C2: calibrate_homography, sync_offsets, label_intervals
- Phân tích visibility trên SH17/CHV: recall|worn và |log LR⁻| tăng theo v (hình trong `eval/reports/visibility/`)
- Thí nghiệm 5: crop người tăng recall helmet +15 điểm so với full-frame (xác nhận D7)
- Benchmark đa góc bán tổng hợp: single-cam bị che báo nhầm ~70 %; P ≫ B3/B4, ablation v/UNKNOWN có tác dụng; **B2 (OR) ngang P** → C2 cần kịch bản OR thất bại
- 93 test pass, ruff sạch

## Đang dở
- (không có) — thí nghiệm 5 (crop vs full-frame) và M3 đã xong, xem `docs/experiments.md`

## Blocker / chờ người dùng
- **Xác nhận** các thay đổi công thức: dt_weighting (§7.5), p calibration + theta_neg −0.5 (Decision Log)
- Thu dữ liệu C2 (3 cam), trả lời câu hỏi mở §16
- Docker + DeepStream (P0) chưa cài

## Ưu tiên tiếp theo
1. Có C2 → `calibrate_homography`, `sync_offsets`, `ultralytics_runner`, `label_intervals`, `run_baselines --ablations`
2. Tune θ_obs/θ_pos/θ_neg, τ, T_alert trên C2 val
3. P0 DeepStream

## Session gần nhất
- [2026-10-21_s02.md](2026-10-21_s02.md)
