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
