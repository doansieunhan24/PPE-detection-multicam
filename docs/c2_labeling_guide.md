# Hướng dẫn gán nhãn C2 (interval labels)

Áp dụng cho mỗi take `s01_tNN` (và s02 sau này). Nhãn nằm ở `data/annotations/s01_tNN/`.
Công cụ: `tools/label_intervals.py`. Chấm điểm: từng camera riêng (CLAUDE.md §10.2).

## 0. Đã chuẩn bị sẵn

- `tracks.csv`: mọi `(cam_id, track_id)` detector tạo ra, kèm thời gian xuất hiện. Cột `gt_id` để trống.
- `thumbs/thumbs_camX_NN.jpg`: mỗi dòng là một track (tên cam + track, khoảng thời gian tính từ đầu take,
  3 ảnh crop: lớn nhất, đầu, cuối).

## 1. Quy ước người

Dùng **một mã cố định cho cả session**: `P1` = A, `P2` = B, `P3` = C (theo vai trong kịch bản).
Cùng một người thì cùng `gt_id` ở mọi camera và mọi take.

## 2. Bước 1: điền `gt_id` trong `tracks.csv`

Mở `tracks.csv` (Excel / VS Code), xem ảnh thumbs, điền số `1`, `2`, `3` (không có chữ P):

- Một người bị tách thành nhiều track (mất track rồi có ID mới): **các track đó cùng `gt_id`**.
- Track sai (không phải người: poster, bóng, phản chiếu kính): **để trống**.
- Track bị đổi người giữa chừng (ID swap): lấy người chiếm phần lớn thời gian của track, ghi chú lại.
- Track ngắn (< 1 s) khó nhìn: có thể để trống, ảnh hưởng không đáng kể.

## 3. Bước 2: viết `intervals_src.csv`

Tạo file `data/annotations/s01_tNN/intervals_src.csv`. Thời gian đọc từ **trình phát video cam2**
(`data/raw/s01_tNN/cam2.mp4`), tính từ đầu video, dạng `62.5` hoặc `1:02.5`.
Độ lệch giữa các cam ≤ 92 ms, nhỏ hơn nhiều so với dung sai khớp δ = 2 s, nên xem cam nào cũng được.

```
gt_id,type,start,end,note,cams
P2,PPE_HELMET,0:05,1:40,khong doi mu ca take,cam1;cam2;cam3
-,CROWD,0:20,0:55,3 nguoi trong Z3,cam2;cam3
```

| Cột | Ý nghĩa |
|---|---|
| `gt_id` | `P1`/`P2`/`P3`; `-` cho CROWD (theo zone, không theo người) |
| `type` | `PPE_HELMET`, `PPE_VEST`, `PPE_BOOTS`, `INTRUSION`, `IMMOBILE`, `CROWD` (`FALL` không dùng ở s01) |
| `start`, `end` | Khoảng thời gian **sự thật** đang vi phạm / đang xảy ra sự kiện |
| `note` | Tuỳ ý |
| `cams` | Camera **có người/sự kiện trong khung hình** trong khoảng đó (≥ khoảng 2 s). Để trống = cả 3 cam |

### Quy tắc quan trọng

1. **Gán theo sự thật, không theo việc camera có nhìn rõ hay không.** Người không đội mũ mà quay lưng
   vẫn là vi phạm. Việc hệ thống trả về UNKNOWN hay bỏ sót là chuyện của hệ thống, nhãn không được nương theo.
2. **`cams` = có trong khung hình, kể cả bị che / quay lưng / ở xa.** Không bỏ camera chỉ vì nhìn khó;
   nếu bỏ, recall của phương pháp P sẽ được đánh giá cao hơn thực tế. Chỉ bỏ camera mà người
   **hoàn toàn không xuất hiện** (ví dụ cam1 ở hành lang khi mọi việc diễn ra trong sảnh).
3. Mỗi **đợt vi phạm liên tục** là một dòng. Đội mũ vào rồi tháo ra lần nữa thì thêm dòng mới.
4. Người ra khỏi mọi camera rồi quay lại mà vẫn vi phạm: vẫn một dòng (vi phạm không đổi).
5. Không gán nhãn khoảng "đủ PPE". Take nào không có vi phạm thì `intervals_src.csv` chỉ có dòng tiêu đề.

## 4. Nhãn theo từng take (s01)

| Take | Nhãn cần gán |
|---|---|
| T01 (đủ PPE) | Không có dòng nào. Mọi alert là báo sai |
| T02 (1 người không đội mũ) | `PPE_HELMET` của người đó, từ lúc xuất hiện đến hết take |
| T03 (cầm mũ, đội, tháo) | `PPE_HELMET` cho **mỗi** đoạn mũ không nằm trên đầu (đang cầm tay cũng tính là không đội) |
| T04 (che khuất, quay lưng, đủ PPE) | Không có dòng nào. Take quan trọng nhất cho C1: đo báo sai |
| T05 (đứng xa, đủ PPE) | Không có dòng nào |
| T06 (cởi gile) | `PPE_VEST` từ lúc gile rời vai đến hết take (hoặc đến khi mặc lại) |
| T07 (hard negative) | A (P1): `PPE_HELMET` + `PPE_VEST` cả take (mũ xe máy + áo giả, xem `data/raw/s01_t07/notes.csv`) |
| T08 (vùng cấm Z2) | `INTRUSION` cho mỗi lần vào Z2: từ lúc chân bước vào đến lúc ra hẳn. `cams` = `cam2;cam3` |
| T09 (ngồi/nằm xuống từ từ) | `IMMOBILE` từ lúc đã nằm/ngồi bệt và đứng yên trong Z3 đến lúc bắt đầu đứng dậy. `cams` = `cam2;cam3` |
| T10 (tụ tập) | `CROWD`, `gt_id` = `-`: từ lúc **đủ 3 người** trong Z3 đến lúc còn < 3. `cams` = `cam2;cam3` |

Z2 (vùng cấm) = 2×2 viên gạch trước cửa thoát hiểm; Z3 (vùng rủi ro) = 2×3 viên giữa sảnh
(ảnh `docs/figures/calibration/`). Hệ thống báo IMMOBILE sau ≥ 10 s đứng yên, nhưng nhãn vẫn gán
từ lúc bắt đầu nằm yên (thời gian phát hiện được đo riêng).

> **Không gán `PPE_BOOTS` ở s01**: chưa ai mang giày bảo hộ, nên Z2 của s01 không yêu cầu `boots`
> (`data/jsonl/s01_tNN/zones.yaml` ghi đè `configs/zones.yaml`). Giày chấm từ s02.

## 5. Bước 3: chuyển đổi và kiểm tra

```bash
python tools/label_intervals.py convert --jsonl data/jsonl/s01_tNN \
    --src data/annotations/s01_tNN/intervals_src.csv \
    --out data/annotations/s01_tNN/intervals.csv --clock video --ref cam2
python tools/label_intervals.py validate --jsonl data/jsonl/s01_tNN --ann data/annotations/s01_tNN
```

`validate` phải in `OK`. Lỗi thường gặp: `gt_id` trong intervals không có track nào trong `tracks.csv`
(quên điền `gt_id` cho track của người đó), thời gian nằm ngoài take, tên camera sai trong `cams`.
