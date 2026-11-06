# STATUS — cập nhật lần cuối 2026-11-06 (s17)

**Phase hiện tại:** P4 xong cho s01 (nhãn đủ 10 take); **P5/P8 lần đầu trên dữ liệu thật**: cross-validation theo take xong. Báo cáo + slide tiến độ đã có. Không ghép người giữa các cam.

## Đã xong
- P0–P2, code P5–P7, demo live (s12–s13); calib cam2/cam3, zones, quay s01 (s15); `label_assist` (s16)
- (s17) Nhãn s01: A xanh dương (P1), B cam (P2), C xanh nõn chuối (P3); 17 interval
- (s17) `eval.run_c2 --cv [--cv-refit]`; `fp_duplicate`; `eval/ppe_rate_c2.py`, `eval/per_take_c2.py`; `evidence.theta_neg_part` (tuỳ chọn, mặc định tắt)
- (s17) Kết quả CV refit: F1 B4 0.393 > B1 0.343 > P 0.255; báo sai không lặp P 1 / B4 14 / B1 20; TTD P 7.2 s; P tắt gile (0/6)
- (s17) Thử θ_neg riêng theo bộ phận (`--space per_part`): **âm tính** (held-out F1 0.222, gile vẫn 0/6) → giữ θ_neg chung
- (s17) Báo cáo Claude Docs https://claude.ai/artifact/6Vbx6PvUKNWgqPQ7cYEDXT ; slide `eval/reports/c2_s01/bao_cao_tien_do_2026-11-06.pptx` (`tools/report/build_deck.py`)

## Đang dở
- CV có nối track (`--set tracking.stitch_gap_ms=1500`) — `eval/reports/c2_s01_refit_stitch` (chạy nền)

## Blocker / chờ người dùng
- Gile xanh dương: chấp nhận là hạn chế, hay đổi gile cho s02 / bổ sung dữ liệu gile xanh cho detector (không dùng C2)?
- **Quay s02 ngày 2026-11-07** (`docs/c2_s02_plan.md`: gile cam / xanh nõn chuối, băng keo Z2/Z3, giày bảo hộ, 12 take)

## Ưu tiên tiếp theo
1. Quay + gán nhãn s02, `configs/c2_split.yaml` val s01_t* / test s02_t*
2. Đọc kết quả CV nối track, cập nhật báo cáo nếu đáng kể
3. Quay s02, chạy `eval.run_c2` val s01 → test s02

## Session gần nhất
- [2026-11-06_s17.md](2026-11-06_s17.md) (trước đó: s16, s15, s14)
