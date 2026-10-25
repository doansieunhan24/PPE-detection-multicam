# STATUS — cập nhật lần cuối 2026-10-25 (s06)

**Phase hiện tại:** P2 xong (PPE detector). Code P5–P7 xong và test được. **Chuỗi xử lý C2 đã chạy thử từ đầu đến cuối** (trên WILDTRACK). **Chờ dữ liệu C2 (P4) và quyền Docker (P0).**

## Đã xong
- P1 offline backend (YOLO11m-pose@960 + ByteTrack → JSONL, timestamp theo pts) + replay
- **PPE detector M2** (YOLO11s@320, SH17+CHV person crops): CHV test helmet AP50 0.957, vest 0.913
- C1 core: part ROI, gán PPE, visibility, evidence (dt-weighted), state machine, Dempster–Shafer, B0–B4, alert rules
- Association đa cam (homography, Hungarian, global ID, gate riêng cho bbox bị cắt), behavior rule-based, alerts/dashboard/Telegram (P7)
- Eval: event metrics, `run_baselines --ablations`, **`run_sensitivity`**, **`bench_fps`** (offline), `eval_detector`, `compare_evidence`, bán tổng hợp, SHWD, `eval_association`
- **Fall trên UR Fall** (`eval/eval_fall.py`): cam0 test P/R/F1 0.650/0.867/0.743, latency 1.2 s; fuse theo v với cam trần **kém hơn** cam0 một mình (recall 0.533 so với 0.733)
- Association trên WILDTRACK: oracle pair F1 0.985, YOLO11m-pose F1 0.786 (`docs/experiments.md`)
- **Chuẩn bị nhận C2:** `tools/session_check.py` (VFR, offset + drift từ tắt/bật đèn, frame calibrate, cameras.yaml của session), `label_intervals convert` (phiếu ghi → intervals), calibrate với chỉ số gạch + sai số leave-one-out + FOV an toàn cho cam thấp
- P0: scaffolding + probe đã đối chiếu với DeepStream-Yolo-Pose; chọn DS 7.1 (docs WSL2, Ubuntu 22.04); hướng dẫn trong `docker/README.md`
- 128 test pass, ruff sạch (pytest cần `--basetemp` ghi được trên máy này)

## Đang dở
- (không có)

## Blocker / chờ người dùng
- **P0:** trong WSL chạy `sudo usermod -aG docker $USER` rồi `wsl --shutdown` (Docker, nvidia runtime, GPU trong WSL đều đã sẵn)
- **Xác nhận** dt_weighting (§7.5), p calibration + theta_neg −0.5, foot_point = bbox_bottom (§7.1), d_max_truncated_m = 1.2
- Quay C2 theo `docs/c2_shooting_script.md`; trả lời câu hỏi mở §16
- License WILDTRACK: TODO(verify)
- Có đổi fusion fall §7.7 (thêm trọng số theo mức hữu ích của góc nhìn) không?

## Ưu tiên tiếp theo
1. Có C2: README bước 2–6 (`session_check` → `calibrate_homography` → `ultralytics_runner` → `label_intervals` → `run_baselines --ablations` → `run_sensitivity`), tune trên s01 (val), báo cáo trên s02/s03
2. Có quyền Docker: P0 theo `docker/README.md`, so khớp JSONL DS với offline, `bench_fps` cho DS
3. Tune θ_obs/θ_pos/θ_neg, τ, T_alert, d_max, d_max_truncated trên C2 val

## Session gần nhất
- [2026-10-25_s06.md](2026-10-25_s06.md) (trước đó: s05)
