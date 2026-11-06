# Kế hoạch quay C2 session 2 (s02) — bài học từ s01

> s02 là **session test riêng**: tune trên s01, báo số trên s02 (`configs/c2_split.yaml`: val s01_t*, test
> s02_t*). Không train detector trên C2 (CLAUDE.md §9). Kết quả s01: `docs/experiments.md` (2026-11-06).

## Vì sao phải đổi so với s01

| Vấn đề ở s01 | Hệ quả khi chấm | Cách quay s02 |
|---|---|---|
| Detector bắt gile xanh dương của A chỉ 29–44 % frame (gile cam 79–91 %) | Báo sai gile hàng loạt; P tune xong tắt gile (0/6) | **Cả 3 người mặc gile cam / xanh nõn chuối** (đề xuất 2026-11-06, không cần quay lại s01). Hạn chế gile xanh giữ làm phát hiện từ s01 (`eval/reports/c2_s01/ppe_rate`). Ngưỡng tune trên s01 có thể vẫn "tắt gile": báo thêm CV trong s02 |
| T10: 3 người đứng cạnh Z3, chỉ 1 người trong vùng | Không có sự kiện CROWD nào | Dán băng keo viền Z2 / Z3 trên sàn; đứng hẳn trong vùng ≥ 10 s |
| T09: nằm chỉ có chân trong Z3 | Điểm sàn (đáy bbox) có thể ra ngoài vùng | Nằm **cả người** trong Z3, song song một cạnh vùng |
| T03 là take duy nhất có tháo mũ ngắn | Khi T03 làm take test, tập tune không còn kiểu này → cả 3 phương pháp bỏ sót 13/13 | Có ít nhất **2 take** tháo / đội mũ ngắn (3–8 s), khác người diễn |
| Track đứt khi một người đi ngang trước người khác | 2 ca nhảy người; nhiều alert lặp | Giữ, nhưng ghi giờ những lần đi ngang (giúp gán nhãn); không cần tránh |
| Không ai mang giày bảo hộ | Không chấm được `PPE_BOOTS` | Ít nhất 1 người mang giày bảo hộ, 1 người giày thể thao |
| Không quay té ngã | Recall FALL chỉ có trên MCFD / UR Fall | Giữ (an toàn); có thể ngồi xuống đệm từ từ để đo IMMOBILE |

## Danh sách take đề xuất (mỗi take một thư mục `data/raw/s02_tNN`)

| Take | Nội dung | Ai | Dài | Nhãn |
|---|---|---|---|---|
| T01 | Đủ PPE, đi lại bình thường cả 3 cam | 3 người | 90 s | — (mọi cảnh báo PPE là báo sai) |
| T02 | 1 người không đội mũ cả take | B | 90 s | `PPE_HELMET` |
| T03 | Tháo mũ cầm tay 3–8 s, đội lại, lặp ~5 lần | A | 90 s | `PPE_HELMET` theo từng đoạn |
| T04 | Đủ PPE, đi sau vật cản / quay lưng / ngồi sau ghế | 3 người | 120 s | — (take quan trọng nhất cho C1) |
| T05 | Đứng xa cam, góc xa nhất | 3 người | 60 s | — |
| T06 | Cởi gile giữa chừng, để trên ghế | C | 90 s | `PPE_VEST` từ lúc cởi |
| T07 | Hard negative: mũ xe máy, áo khoác thường, giày thể thao | A | 60 s | `PPE_HELMET`, `PPE_VEST`, `PPE_BOOTS` |
| T08 | Vào Z2 (đứng hẳn trong vạch băng keo) 3–4 lần, mỗi lần ≥ 3 s | B | 90 s | `INTRUSION` mỗi lần |
| T09 | Ngồi bệt / nằm yên cả người trong Z3 ≥ 15 s | A | 90 s | `IMMOBILE` |
| T10 | Cả 3 người đứng hẳn trong Z3 ≥ 15 s, 2 lần | 3 người | 90 s | `CROWD` theo zone |
| T11 | Hỗn hợp tự do, nhiều sự kiện chồng nhau (ghi phiếu giờ) | 3 người | 120 s | theo phiếu |
| T12 | Tháo mũ ngắn lần 2 (người khác T03), kèm che khuất | B | 90 s | `PPE_HELMET` theo đoạn |

Tổng ≈ 19 phút mỗi cam. Đồng bộ theo đồng hồ PC lúc frame đầu (`session_check --sync wallclock`, như s01).

## Checklist trước khi quay

- [ ] Gile cam / xanh nõn chuối cho cả 3 người (không dùng gile xanh dương)
- [ ] Băng keo viền Z2, Z3 theo mạch gạch (toạ độ trong `configs/zones.yaml`, ảnh `docs/figures/calibration/cam2_zones.jpg`)
- [ ] Kiểm tra cam2, cam3 chưa bị xoay (calib sàn gạch còn đúng): chụp 1 frame mỗi cam, so với `cam*_raw.jpg`
- [ ] Ghi vai từng take vào `data/annotations/s02_script.yaml` **ngay khi quay** (s01 phải đoán lại sau)
- [ ] Giày bảo hộ cho ít nhất 1 người
- [ ] Phiếu ghi giờ cho T11

## Sau khi quay

1. `record_c2` → JSONL (`perception_offline` hoặc DeepStream), kiểm số frame với `record_meta.json`.
2. Gán nhãn như s01 (`docs/c2_labeling_guide.md`, mục 0b: track mẫu → `label_assist ids` → `accept` → `prefill` → `timeline --crop-zone` → `convert` + `validate`).
3. `configs/c2_split.yaml`: `val` = s01_t*, `test` = s02_t* → `python -m eval.run_c2 --out eval/reports/c2 --jobs 6`.
