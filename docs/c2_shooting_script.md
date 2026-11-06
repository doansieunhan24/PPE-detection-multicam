# Kịch bản quay bộ dữ liệu C2

> Phục vụ CLAUDE.md §9.1. Bộ này **chỉ để eval**, không dùng để train detector.

| Take | Nội dung | Dài |
|---|---|---|
| T01 | Đủ PPE, đi lại bình thường | 60s |
| T02 | 1 người không đội mũ | 60s |
| T03 | Cầm mũ trên tay, đội vào rồi tháo ra | 60s |
| T04 | Đủ PPE nhưng đi vòng sau vật cản / quay lưng 1 cam | 90s |
| T05 | Đứng xa cam, góc xa nhất | 60s |
| T06 | Cởi áo phản quang giữa chừng | 60s |
| T07 | Hard negative: mũ xe máy, áo vàng thường, giày thể thao | 60s |
| T08 | Đi vào vùng cấm | 60s |
| T09 | Té lên đệm, nằm bất động 10–15s | 60s |
| T10 | Tụ tập ≥ 4 người ở vùng rủi ro | 60s |
| T11 | Hỗn hợp tự do, nhiều sự kiện chồng nhau | 120s |

## Nhãn cần ghi khi gán nhãn (`intervals.csv`)

| Take | Nhãn |
|---|---|
| T01, T04, T05 | — (đủ PPE: mọi cảnh báo là báo sai) |
| T02 | `PPE_HELMET` của người không đội mũ, cả take |
| T03 | `PPE_HELMET` lúc mũ không ở trên đầu |
| T06 | `PPE_VEST` từ lúc cởi áo |
| T07 | `PPE_HELMET` (mũ xe máy), `PPE_VEST` (áo vàng thường), `PPE_BOOTS` (giày thể thao) cho người mặc đồ đó |
| T08 | `INTRUSION` từ lúc chân vào đến lúc ra khỏi vùng cấm |
| T09 | `FALL` lúc té, `IMMOBILE` lúc nằm bất động |
| T10 | `CROWD` theo zone (để trống `gt_id`) |
| T11 | Theo phiếu ghi giờ |

## s01 (quay 2026-10-31): thực tế so với kịch bản

Vai: A = gile xanh dương (P1), B = gile cam (P2), C = gile xanh nõn chuối (P3). Nhãn ở
`data/annotations/s01_tNN/` (gán 2026-11-06, `process/2026-11-06_s17.md`).

| Take | Thực tế | Nhãn |
|---|---|---|
| T01 | A, B (không có C) | — |
| T02 | B không đội mũ cả take | `PPE_HELMET` P2 23.1–99.5 s |
| T03 | A tháo / đội mũ **7 lần** | 7 dòng `PPE_HELMET` P1 |
| T04 | A, B; ghế xanh che thân | — |
| T05 | A, B, C ở xa | — |
| T06 | A cởi gile lúc ~27 s, để trên ghế | `PPE_VEST` P1 27–93.1 s |
| T07 | A mũ xe máy đen + áo khoác xanh (không có giày thể thao riêng; s01 không chấm giày) | `PPE_HELMET`, `PPE_VEST` P1 cả take |
| T08 | A vào Z2 **5 lần** (lần 2, 5 không chắc) | 5 dòng `INTRUSION` P1 |
| T09 | Không té (an toàn): A ngồi xổm 28–36 s, nằm 51–62 s; lúc nằm chỉ chân trong Z3 | `IMMOBILE` P1 51–62 s |
| T10 | 3 người đứng cạnh Z3, chỉ A trong Z3 | — (không phải CROWD theo vùng) |
| T11 | Chưa quay | — |

**Cho s02:** dán băng keo viền Z2 / Z3 trên sàn để diễn viên đứng hẳn trong vùng (T09, T10);
gile cùng màu với dữ liệu train hoặc ghi rõ màu (detector bắt gile xanh dương kém, xem
`docs/experiments.md` 2026-11-06); giày bảo hộ; T11.
