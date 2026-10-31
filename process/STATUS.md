# STATUS — cập nhật lần cuối 2026-10-31 (s15)

**Phase hiện tại:** P4 đang làm: **đã quay C2 session 1** (T01–T10, 3 cam). P3 xong cho cam2 + cam3 (calib sàn gạch + khử méo ống kính). **Không ghép người giữa các cam** (người dùng chốt): 3 cam chấm riêng, `association.mode: none`.

## Đã xong
- P0–P2 (DS + offline), code P5–P7, demo live (s12–s13)
- (s15) `tools/record_c2.py` ghi RTSP 3 cam; `session_check --sync wallclock` (không cần flash)
- (s15) Calibration sàn gạch caro + mô hình méo: cam2 LOO 0.039 m, cam3 LOO 0.012 m, chung hệ toạ độ sàn (gạch 0.67 m); cam1 (hành lang) không calib được
- (s15) Zones thật: Z2 vùng cấm trước cửa thoát hiểm, Z3 vùng rủi ro giữa sảnh, Z1 PPE; `crowd.K` = 2
- (s15) **Dữ liệu C2 s01**: `data/raw/s01_t01` … `s01_t10` (cam1 hành lang, cam2 + cam3 sảnh, 25 fps, không mất frame), gile phản quang; detector (JSONL) xong cho t01–t03
- 180 test pass, ruff sạch

## Đang dở
- JSONL s01_t04 (dở), s01_t05 (dở), s01_t06–t10 (chưa chạy): tác vụ nền bị dừng vì thiếu RAM → chạy lại **tuần tự**

## Blocker / chờ người dùng
- s02 tuần sau (giày bảo hộ, đổi vai/đồ, T11)

## Ưu tiên tiếp theo
1. Detector cho s01_t04 … t10 (tuần tự): `python -m src.perception_offline.ultralytics_runner --cameras data/raw/s01_tNN/cameras.yaml --out data/jsonl/s01_tNN` (chạy `session_check --sync wallclock` trước cho t05–t10)
2. Cross-validation theo take trong `eval.run_c2`
3. Gán nhãn s01 (`tools/label_intervals.py`), ghi chú take trong `process/2026-10-31_s15.md`
4. Cập nhật `docs/c2_shooting_script.md` theo cách quay thực tế

## Session gần nhất
- [2026-10-31_s15.md](2026-10-31_s15.md) (trước đó: s14, s13, s12)
