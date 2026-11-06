# process/ — Nhật ký tiến độ

Mỗi session làm việc có một file. Claude **đọc lại thư mục này ở đầu mỗi session** và **ghi file mới ở cuối session** (xem CLAUDE.md §0).

## Cấu trúc

- `STATUS.md`: trạng thái hiện tại, ghi đè mỗi session. **Đọc file này trước tiên.**
- `YYYY-MM-DD_sNN.md`: log của từng session (NN là số thứ tự trong ngày, 01, 02, ...). Đã ghi thì không sửa lại.

## Template cho file session

```markdown
# Session YYYY-MM-DD sNN

**Phase:** P? — <tên phase>

## Mục tiêu session
-

## Đã làm
- <việc> — <file liên quan>

## Kết quả / số liệu
- (copy từ output script; không điền tay)

## Vấn đề / blocker
-

## Quyết định mới
- (nếu có → cũng ghi vào CLAUDE.md §15 Decision Log)

## Việc tiếp theo
- [ ]
```

## Danh sách session

| Ngày | File | Tóm tắt |
|---|---|---|
| 2026-10-21 | [2026-10-21_s01.md](2026-10-21_s01.md) | Kiểm tra dự án, dựng khung repo |
| 2026-10-21→22 | [2026-10-21_s02.md](2026-10-21_s02.md) | P1 offline, C1 core, eval, train PPE (SH17+CHV), phân tích visibility |
| 2026-10-22 | [2026-10-22_s03.md](2026-10-22_s03.md) | Tạm dừng, chờ quay dữ liệu C2 |
| 2026-10-22 | [2026-10-22_s04.md](2026-10-22_s04.md) | Association trên WILDTRACK; FOV margin; foot point = đáy bbox (chờ xác nhận) |
| 2026-10-25 | [2026-10-25_s05.md](2026-10-25_s05.md) | Chuẩn bị nhận C2 (pts/VFR, session_check, nhãn từ phiếu, config theo session), bbox bị cắt, LOO calibration, sensitivity, bench FPS, probe DS, hướng dẫn P0 |
| 2026-10-25 | [2026-10-25_s06.md](2026-10-25_s06.md) | Fall trên UR Fall: sửa lịch sử theo global id, hold 0.2 (test F1 0.743), fuse theo v thua cam0 |
| 2026-10-26 | [2026-10-26_s07.md](2026-10-26_s07.md) | Fall đa góc trên MCFD: 8 cam vmean F1 0.696 so với 1 cam 0.474; drop_ratio 0.2; cam top-down làm hỏng vmean (UR Fall) |
| 2026-10-26 | [2026-10-26_s08.md](2026-10-26_s08.md) | Fall học (1D-CNN skeleton): MCFD 8 cam F1 0.762 so với rule 0.696, FA giảm khoảng 3 lần; UR Fall phụ thuộc seed |
| 2026-10-26 | [2026-10-26_s09.md](2026-10-26_s09.md) | Intrusion/crowd trên WILDTRACK: oracle F1 0.93/0.80, detector 0.60/0.46 |
| 2026-10-26 | [2026-10-26_s10.md](2026-10-26_s10.md) | Xác nhận 4 thay đổi spec; P0 xong: DS 7.1 trong WSL2, 3 RTSP (TCP) ~30 fps/stream với model mẫu |
| 2026-10-26 | [2026-10-26_s11.md](2026-10-26_s11.md) | Model dự án trong DS: JSONL khớp offline (assoc F1 0.818 vs 0.810), 3 cam 1080p 26.1 fps/stream |
| 2026-10-27 | [2026-10-27_s12.md](2026-10-27_s12.md) | Demo live P7: RTSP → DS → ZMQ → analytics → smart record/OSD/dashboard; sửa crash RTSP (libproxy); độ trễ alert median 554 ms @2 fps |
| 2026-10-29 | [2026-10-29_s13.md](2026-10-29_s13.md) | Sẵn sàng cho C2: `eval.run_c2` một lệnh val → test (tune từng phương pháp, calib C2, số cam, behavior/association), cache observe (toàn chuỗi 100 s), snapshot live, ảnh track để gán nhãn; chạy thử bằng GT WILDTRACK |
| 2026-10-31 | [2026-10-31_s14.md](2026-10-31_s14.md) | Bỏ ghép người giữa các cam (`association.mode: none`); eval chấm từng cam; CROWD đếm theo cam; kịch bản C2 = 11 take; commit camera Tiandy LAN |
| 2026-10-31 | [2026-10-31_s15.md](2026-10-31_s15.md) | `record_c2`; calib sàn gạch + khử méo (cam2 LOO 0.039 m, cam3 0.012 m); thêm cam3 trong sảnh; zones thật; quay C2 s01 T01–T10 (3 cam, gile phản quang); chọn cross-validation theo take |
| 2026-11-05 | [2026-11-05_s16.md](2026-11-05_s16.md) | JSONL s01 đủ; commit cột `cams` (2/11); `label_assist` gán nhãn bán tự động (ids theo màu, prefill từ kịch bản, timeline) |
| 2026-11-06 | [2026-11-06_s17.md](2026-11-06_s17.md) | Nhãn s01 đủ 10 take (17 interval); CV theo take: F1 B4 0.393 > P 0.255 nhưng báo sai không lặp P 1 / B4 14, P tắt gile; báo cáo + slide |
