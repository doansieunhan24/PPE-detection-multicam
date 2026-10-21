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
