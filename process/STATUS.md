# STATUS — cập nhật lần cuối 2026-11-05 (s16)

**Phase hiện tại:** P4: **gán nhãn C2 session 1** (T01–T10 đã quay, JSONL detector đủ cả 10 take × 3 cam). P3 xong cho cam2 + cam3. **Không ghép người giữa các cam**: 3 cam chấm riêng, `association.mode: none`.

## Đã xong
- P0–P2 (DS + offline), code P5–P7, demo live (s12–s13)
- (s15) `record_c2`, calib sàn gạch + khử méo (cam2 LOO 0.039 m, cam3 0.012 m), zones thật, quay s01 T01–T10
- (s16) JSONL s01_t01…t10 đủ (khớp số frame `record_meta.json`)
- (2026-11-02, commit s16) Cột `cams` trong nhãn GT + chấm theo cam (`gt_for_cams`), `docs/c2_labeling_guide.md`, s01 không chấm giày
- (s16) `tools/label_assist.py`: `ids` (gợi ý gt_id theo màu từ track mẫu), `prefill` (khung nhãn từ `data/annotations/s01_script.yaml`), `timeline` (ảnh 1 frame/giây có Z2/Z3); đặc trưng đã cache cho cả 10 take
- 189 test pass, ruff sạch (pytest cần `--basetemp` trong scratchpad khi chạy trong sandbox)

## Đang dở
- Nhãn s01: `tracks.csv` chưa có `gt_id`, `intervals.csv` trống (template + thumbs + feats đã có)

## Blocker / chờ người dùng
- Track mẫu (1 track/người/cam; thêm mẫu cho T06, T07 vì đổi đồ) và vai trong `s01_script.yaml` (ai không đội mũ T02, ai cầm mũ T03, ai cởi gile T06, ai vào Z2 T08, ai nằm T09)
- s02 tuần sau (giày bảo hộ, đổi vai/đồ, T11)

## Ưu tiên tiếp theo
1. Gán nhãn s01 theo `docs/c2_labeling_guide.md` mục 0b: mẫu → `label_assist ids --write` → xem `review_ids_camX.jpg` → `prefill` → điền TODO (timeline) → `convert` + `validate`
2. Cross-validation theo take trong `eval.run_c2`
3. Cập nhật `docs/c2_shooting_script.md`

## Session gần nhất
- [2026-11-05_s16.md](2026-11-05_s16.md) (trước đó: s15, s14, s13)
