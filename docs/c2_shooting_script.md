# Kịch bản quay bộ dữ liệu C2 (bản rút gọn: phòng nhỏ, cam đặt thấp, 3 người)

> Phục vụ CLAUDE.md §9.1. Bộ này **chỉ để eval**, không dùng để train detector.
> Điều kiện thực tế: phòng hoặc hành lang khoảng **3×4 m**, **không lắp cam cao được**, có **3 người** (P1, P2, P3).
> Ngưỡng hiện tại trong `configs/`: `T_alert` 3 s, `T_intr` 1 s, `T_immobile` 10 s. Mỗi cảnh dài **20–40 s**, tức là dài hơn ngưỡng nhiều.

**Phòng nhỏ và cam thấp không làm hỏng thí nghiệm.** Cam đặt ngang ngực thì người hay che nhau, hay quay lưng về một cam. Đó chính là tình huống C1 cần chứng minh. Chỉ cần 3 cam **cùng nhìn thấy sàn** là đủ.

---

## 1. Chuẩn bị (khoảng 30 phút)

### 1.1 Đồ cần có
- **3 điện thoại** (thiếu thì 2 cũng quay được, xem mục 5). Mỗi máy kèm sạc dự phòng.
- Chỗ đặt máy: mép bàn, nóc tủ, kệ, chồng sách trên ghế, tripod mini. **Cao khoảng 1.2–1.8 m**, càng cao càng tốt.
- 3 mũ bảo hộ, 3 áo phản quang **cùng kiểu**. Mũ lưỡi trai, cái thau hoặc mũ bảo hiểm xe máy.
- 1–2 thùng carton hoặc ghế để che. 1 tấm nệm (chỉ khi quay cảnh té ngã, không bắt buộc).
- Băng dính, thước dây.

### 1.2 Đặt cam
```
        tường
  +---------------------------+
  |[cam2]               [cam3]|
  |                           |
  |                           |
  |        vùng đi lại        |
  |                           |
  |                    +------+
  |                    |  Z2  |   Z2 = góc cấm ~1 x 1 m
  |[cam1]              |      |        (dán băng dính hoặc theo viền gạch)
  +--------------------+------+
                 cửa
```
- Đặt 3 máy ở **3 góc phòng**, quay chéo vào giữa phòng. Góc thứ tư dùng làm **vùng cấm Z2**.
- Chúc máy **hơi cúi xuống** để thấy được sàn ở giữa phòng và chân người, đừng để máy nhìn ngang.
- Đặt máy **sát góc, sát tường** để máy cách giữa phòng càng xa càng tốt.
- Dùng **ống kính 1x**, quay ngang. Ống 1x chỉ thấy đủ người khi người đứng cách máy **≥ 2.3–2.5 m**, nên người đi lại chủ yếu ở **giữa phòng**.
- **Kiểm tra trước khi quay:** 1 người đứng giữa phòng, rồi đứng ở vài vị trí gần từng máy. Xem cả 3 máy có thấy **từ đầu tới chân** không. Nếu vẫn bị cắt nhiều thì chuyển sang **0.5x** (bật sửa méo ống kính trong cài đặt camera), dùng **8–10 điểm mốc** và ghi vào phiếu.
- Cài đặt máy: **1080p, 25 hoặc 30 fps**, tắt chống rung, khoá nét và phơi sáng (chạm giữ trên màn hình), tắt HDR. **Ba máy dùng cùng fps.**
- **Đặt máy cố định suốt session**, không chạm vào máy. Lỡ xê dịch máy thì ghi vào phiếu.

### 1.3 Điểm mốc trên sàn (để tính homography)
- Phòng lát gạch thì **góc gạch chính là điểm mốc**, không cần dán gì. Đo kích thước viên gạch (vd. 60 cm) và chọn một góc gạch làm gốc (0,0).
- Sàn không có gạch thì dán **6–8 dấu chữ thập** rải đều khắp sàn, rồi đo toạ độ (m) từ một góc phòng.
- **Mỗi cam phải thấy ≥ 6 mốc**, mốc trải ra cả gần lẫn xa, không dồn một chỗ.
- Chụp ảnh sàn và ghi lại vị trí gốc (0,0).

---

## 2. Quy trình mỗi session (khoảng 10 phút quay)

1. Đặt máy, kiểm tra khung hình của cả 3 máy.
2. Bấm quay cả 3 máy (không cần bấm cùng lúc). **Để sàn trống 5–10 s** (lấy frame này để calibrate).
3. **Đồng bộ:** **tắt đèn phòng 2 s rồi bật lại**, làm trong 20 s đầu. Vỗ tay to một cái ngay khi bật đèn.
4. Lần lượt P1, P2, P3 đứng giữa phòng xoay một vòng chậm (để sau này nhận ra ai là ai).
5. Quay các cảnh ở mục 3. Trước mỗi cảnh **đọc to số cảnh** ("cảnh C2"). Người cầm phiếu ghi **giờ** bắt đầu và các sự kiện.
6. Cuối session **tắt và bật đèn thêm lần nữa**, rồi dừng quay.
7. Đặt tên file `data/raw/sNN/cam1.mp4`, `cam2.mp4`, `cam3.mp4`.

> Quay **liền** cả session, không dừng giữa các cảnh. Lúc chuyển cảnh, mọi người cứ đủ PPE và đi lại bình thường. Đoạn đó cũng là dữ liệu để đo false alarm.
> Diễn hỏng thì không cần dừng: đọc "làm lại cảnh C2", ghi vào phiếu và diễn lại.

---

## 3. Các cảnh

Cột **Nhãn**: interval cần ghi vào `intervals.csv` khi gán nhãn. Ghi "—" nghĩa là ai cũng đủ PPE, nên **mọi cảnh báo phát ra đều là báo động sai**.

### Nhóm C: Đủ PPE nhưng cam khó nhìn — **quan trọng nhất**, bắt buộc quay

| # | Cảnh | Làm gì | Thời lượng | Nhãn |
|---|---|---|---|---|
| C1 | Quay lưng | Cả 3 người đứng làm việc, quay lưng về cam1. Sau 15 s quay lưng về cam2, rồi về cam3 | 45 s | — |
| C2 | Người che người | 3 người đứng thành hàng dọc trước một cam (người sau bị người trước che), nói chuyện | 30 s | — |
| C3 | Vật che | Đứng sau thùng carton hoặc ghế, một cam chỉ thấy nửa trên hoặc đầu, các cam khác thấy đủ | 30 s | — |
| C4 | Đứng sát cam | Đi lại gần một cam, người bị cắt mất chân hoặc mất đầu trong khung hình | 20 s | — |
| C5 | Cúi, ngồi xổm | Cúi nhặt đồ, ngồi xổm buộc dây giày, đầu cúi xuống | 30 s | — |

### Nhóm V: Vi phạm thật — bắt buộc quay

| # | Cảnh | Làm gì | Thời lượng | Nhãn |
|---|---|---|---|---|
| V1 | Không mũ | P1 không đội mũ, đi lại | 30 s | P1 `PPE_HELMET` |
| V2 | Không mũ, chỉ 1 cam thấy đầu | P1 không mũ, đứng sau thùng hoặc quay lưng, sao cho chỉ 1 cam thấy rõ đầu | 30 s | P1 `PPE_HELMET` |
| V3 | Cầm mũ | P2 cầm mũ ở tay, lúc ngang hông, lúc ôm trước ngực | 30 s | P2 `PPE_HELMET` |
| V4 | Tháo mũ giữa chừng | P3 đội mũ 10 s, tháo ra cầm 20 s, đội lại | 40 s | P3 `PPE_HELMET` lúc không đội |
| V5 | Cởi áo giữa chừng | P2 mặc áo 10 s, cởi ra vắt vai 20 s, mặc lại | 40 s | P2 `PPE_VEST` lúc không mặc |
| V6 | Mũ lưỡi trai | P1 đội mũ lưỡi trai thay mũ bảo hộ | 30 s | P1 `PPE_HELMET` |

### Nhóm O: Đánh lừa luật OR (B2) — nên quay

Ý tưởng: có một cam "nhìn thấy mũ" ở vùng đầu người, nhưng thực tế người đó **không đội**.

| # | Cảnh | Làm gì | Thời lượng | Nhãn |
|---|---|---|---|---|
| O1 | Đứng sau người có mũ | P1 (không mũ) đứng **sát ngay sau** P2 (có mũ). Nhìn từ một cam, mũ của P2 chồng lên đầu P1, còn cam khác thấy rõ P1 không mũ | 30 s | P1 `PPE_HELMET` |
| O2 | Cầm mũ sát đầu | P3 giơ mũ ngang tai hoặc ngay trên đầu như sắp đội, nhưng không đội | 30 s | P3 `PPE_HELMET` |
| O3 | Mũ để sau đầu | Đặt mũ trên nóc tủ hoặc kệ ngang đầu, P1 không mũ đứng ngay trước | 30 s | P1 `PPE_HELMET` |
| O4 | Vật giống mũ | P2 đội thau hoặc mũ bảo hiểm xe máy | 30 s | P2 `PPE_HELMET` |

### Nhóm Z: Vùng cấm — nên quay

| # | Cảnh | Làm gì | Thời lượng | Nhãn |
|---|---|---|---|---|
| Z1 | Vào vùng cấm | P1 bước vào góc Z2, đứng 5 s rồi ra. Làm 3 lần | 40 s | P1 `INTRUSION` mỗi lần (từ lúc chân vào đến lúc ra) |
| Z2 | Đi sát vùng cấm | Đi sát viền Z2 nhưng không bước vào | 20 s | — |

### Nhóm T: Tự do — bắt buộc quay

| # | Cảnh | Làm gì | Thời lượng | Nhãn |
|---|---|---|---|---|
| T1 | Làm việc tự do | Cả 3 đủ PPE, đi lại, bê thùng, nói chuyện tự nhiên, không diễn | 2 phút | — |
| T2 | Tự do có vi phạm | Như T1, nhưng người ghi phiếu bí mật bảo 1 người tháo mũ khoảng 20 s. **Ghi giờ chính xác** | 2 phút | theo phiếu |

### Nhóm F: Té ngã — **không bắt buộc**

Chỉ quay khi có nệm và đủ chỗ. Module té ngã có thể đánh giá trên dataset public (Le2i, UR Fall, Multiple Cameras Fall).

| # | Cảnh | Làm gì | Thời lượng | Nhãn |
|---|---|---|---|---|
| F1 | Té và nằm yên | Té xuống nệm, nằm yên 15 s rồi đứng dậy (có người đứng cạnh) | 25 s | `FALL`, `IMMOBILE` |
| F2 | Ngồi, nằm từ từ | Ngồi bệt xuống nệm chậm rãi, nằm nghỉ 10 s | 20 s | — |

> Tụ tập (CROWD) **bỏ khỏi C2**: 3 người là quá ít để có ý nghĩa. Đưa vào phần Limitations.

---

## 4. Lịch quay

| Session | Cảnh | Dùng để | Thời lượng |
|---|---|---|---|
| s01 | C1–C5, V1, V3, V5, O1, O2, Z1, T1 | **val** (tune tham số) | ~9 phút |
| s02 | C1–C5, V2, V4, V6, O3, O4, Z2, T2 | test | ~9 phút |
| s03 | C1–C5, V1, V3, O1, O2, Z1, T2 (+ F1, F2) | test | ~10 phút |

- Giữa các session **đổi vai** (P1 ↔ P2 ↔ P3), đổi vị trí đứng và hướng đi. Nếu được, đổi ánh sáng (mở hoặc đóng rèm, tắt bớt đèn).
- Không dời máy giữa các session thì dùng chung 1 lần calibrate. Dời máy thì phải quay lại sàn trống.
- Tổng khoảng 28 phút mỗi cam, đủ mục tiêu 15–30 phút. Thiếu thời gian thì quay s01 + s02 trước.

---

## 5. Nếu chỉ có 2 máy

Quay được hết các cảnh trên. Đặt 2 máy ở **2 góc đối diện** để hai máy nhìn thấy hai phía khác nhau của người. Thiếu mất thí nghiệm "1 / 2 / 3 cam", còn lại (B0–B4 so với P, ablation) vẫn chạy đủ.

---

## 6. Phiếu ghi (in ra)

**Session:** `s__` · ngày ____ · phòng ____ · ánh sáng ____ · kích thước gạch ____ · gốc (0,0) ở ____
**Trang phục:** P1 = ____ · P2 = ____ · P3 = ____ · fps / ống kính: ____

| Giờ (hh:mm:ss) | Cảnh | Ai | Sự kiện (bắt đầu/kết thúc vi phạm, vào Z2, té…) | Ghi chú |
|---|---|---|---|---|
| | sync | — | tắt/bật đèn lần 1 | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | sync | — | tắt/bật đèn lần 2 | |

---

## 7. Quy tắc gán nhãn (làm sau khi quay)

- Vi phạm PPE bắt đầu khi mũ rời khỏi đầu (áo rời khỏi vai), kết thúc khi đội hoặc mặc lại. Mũ đội lệch vẫn tính là đang đội.
- `INTRUSION`: từ lúc chân bước vào viền Z2 đến lúc ra khỏi viền.
- Nhãn theo **thời gian thật**. Lúc không cam nào thấy người đó thì vẫn ghi nhãn (đây là lý do cần trạng thái UNKNOWN).
- Lệnh xử lý: README bước 2–6 (`calibrate_homography`, `sync_offsets`, `ultralytics_runner`, `label_intervals`, `run_baselines --ablations`).

## 8. Ghi vào Limitations

- Dữ liệu quay trong phòng nhỏ, cam đặt thấp (~1.2–1.8 m), chưa phải xưởng thật với cam treo cao.
- Cam thấp thì sai số chiếu điểm chân xuống sàn tăng ở xa. Báo cáo reprojection error của từng cam.
- Chỉ 3 người, không đánh giá được tụ tập trên C2.
- Phòng nhỏ: không có cảnh người ở xa (người nhỏ trong ảnh). Ảnh hưởng của kích thước người lấy từ WILDTRACK và dataset public.
