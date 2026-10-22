# STATUS — cập nhật lần cuối 2026-10-22 (s04)

**Phase hiện tại:** P2 xong (PPE detector). Phần lớn code P5–P7 xong và test được. Association đã kiểm chứng trên WILDTRACK. **Chờ dữ liệu C2 (P4) và DeepStream (P0).**

## Đã xong
- P1 offline backend (YOLO11m-pose@960 + ByteTrack → JSONL) + replay
- **PPE detector M2** (YOLO11s@320, SH17+CHV person crops): CHV test helmet AP50 0.957, vest 0.913
- C1 core: part ROI, gán PPE, visibility, evidence (dt-weighted), state machine, Dempster–Shafer, B0–B4, alert rules
- Association đa cam (homography, Hungarian, global ID), behavior rule-based, alerts/dashboard/Telegram (P7)
- Eval: event metrics, `run_baselines`, `eval_detector`, `compare_evidence`, benchmark bán tổng hợp, SHWD
- **Association trên WILDTRACK** (`eval/eval_association.py`, `tools/wildtrack_*.py`):
  - Oracle: pairwise F1 0.982, IDF1 0.952.
  - YOLO11m-pose: F1 0.784, count MAE 3.35.
  - Foot point đáy bbox giảm sai số sàn 0.68 → 0.23 m so với ankle.
- P0 scaffolding (chưa chạy được vì chưa có Docker)
- 107 test pass, ruff sạch

## Đang dở
- (không có)

## Blocker / chờ người dùng
- **Xác nhận** dt_weighting (§7.5), p calibration + theta_neg −0.5, **foot_point = bbox_bottom (§7.1)**
- Thu dữ liệu C2 (3 cam), trả lời câu hỏi mở §16
- Docker + DeepStream (P0) chưa cài
- License WILDTRACK: TODO(verify)

## Ưu tiên tiếp theo
1. Có C2: chạy `calibrate_homography`, `sync_offsets`, `ultralytics_runner`, `label_intervals`, `run_baselines --ablations`, `eval_association`
2. Tune θ_obs/θ_pos/θ_neg, τ, T_alert, d_max trên C2 val
3. P0 DeepStream

## Session gần nhất
- [2026-10-22_s04.md](2026-10-22_s04.md)
