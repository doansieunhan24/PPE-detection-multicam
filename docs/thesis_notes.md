# Ghi chú cho thuyết minh

## Ý tưởng / lập luận

## Hình & bảng cần có

## Limitations bổ sung (ngoài CLAUDE.md §17)

- Phần cứng dev: RTX 4060 Laptop 8 GB — giới hạn số stream tối đa trong benchmark.

## Bằng chứng cho C1 từ dữ liệu public (SH17, 2026-10-21) — số liệu M1, ROI cũ; xem bản cập nhật bên dưới
- Recall phát hiện mũ của người *thực sự đội mũ* phụ thuộc mạnh vào v: ~0.5 khi v<0.6, ≥0.9 khi v≥0.6 (SH17 test, M1). → luật "không thấy mũ ⇒ vi phạm" sai khoảng một nửa số lần khi đầu khó nhìn.
- v (thủ công) phân biệt "detector sẽ thấy PPE nếu có" tốt hơn confidence của detector người (AUC 0.79 vs 0.69 cho head) → biện minh cho việc dùng v thay vì conf (B4).
- Log-likelihood ratio: có box ≈ +3, không box ≈ −1 → bất đối xứng; không được coi "không thấy" mạnh ngang "thấy".
- Bẫy phương pháp: đo "độ đúng của bằng chứng" trên dataset mất cân bằng bị prevalence chi phối; phải đo recall | worn và LR. (Có thể đưa vào phần thảo luận.)
- Hình nên vẽ: recall|worn theo bin v (3 part), và log LR⁻ theo bin v.

## Cập nhật 2026-10-22 (M2, ROI đã tune, hiệu chỉnh trên v ≥ 0.6)
- Hình chính: `eval/reports/visibility/visibility_reliability_sh17.png` — recall|worn tăng đơn điệu theo v (head 0.56 → 1.0, feet 0.40 → 0.92); |log LR⁻| tăng gần tỷ lệ với v (head −0.8 → −3.8) → biện minh trực tiếp cho e = v·logit(p).
- Bằng chứng bất đối xứng ở v≈1: head no-box −4.3, box@0.5 +1.5; feet box@0.5 chỉ +0.3 (giày thường vs giày bảo hộ không phân biệt — Limitations).
- **Lập luận quan trọng cho C2:** nhãn dataset ảnh đơn là "PPE nhìn thấy được" chứ không phải "PPE đang đeo" → không thể hiệu chỉnh/đánh giá tác động của che khuất bằng SH17/CHV; cần nhãn theo người, đa góc (C2).
- Gán PPE theo part-ROI là nguồn mất recall lớn (−13…−19 điểm) nếu ROI hẹp; tune ROI trên val tăng recall head 0.78 → 0.90 (SH17 test) với FPR gần như không đổi.
- Thêm CHV + pseudo-label các class thiếu: helmet AP50 trên CHV 0.890 → 0.957, vest 0.794 → 0.913.

## Benchmark bán tổng hợp (2026-10-22) — điều cần nói trong thuyết minh
- Có thể dùng làm thí nghiệm sơ bộ (chương phương pháp/thảo luận), ghi rõ giới hạn (che khuất nhân tạo).
- B2 (OR) là đối thủ mạnh nhất về false alarm; cần thiết kế C2 để có trường hợp một camera cho bằng chứng *dương sai* (mũ người khác chồng ROI, vật giống mũ, cầm mũ sát đầu) — nơi OR thất bại còn P (trung bình có trọng số v) không.
- Mục tiêu tune ngưỡng nên là FA/giờ ở recall cố định (đường cong FA/h–recall, §10.3-4), không phải J = DET − FA.

## Association trên WILDTRACK (2026-10-22)
- Kiểm chứng ground-plane association (D3) trên dữ liệu đa cam **thật** trước khi có C2. Với box GT: pairwise F1 0.98, IDF1 0.95. Với YOLO11m-pose: F1 0.78 (`docs/experiments.md`).
- **Phát hiện đáng viết vào thuyết minh:** điểm chân lấy từ ankle keypoint (cách làm hay gặp) bị lệch vì ankle cao hơn sàn. Homography mặt sàn khuếch đại sai lệch này theo `h·dist/cam_height`. Đáy bbox giảm sai số median 0.68 → 0.23 m. Nên có một hình minh hoạ tia chiếu và một bảng so sánh các biến thể foot point.
- Thêm dung sai mép FOV (người bị cắt ở mép ảnh) giúp tăng recall ghép cặp.
- Thí nghiệm số cam: với detector thật, đếm người bị chi phối bởi recall detection. Đây là lý do nên dùng nhiều cam cho crowd counting (§7.7).
- Giới hạn: WILDTRACK là quảng trường ngoài trời, cam xa (> 10 m) và đông người, khác bố trí xưởng 6×8 m của C2. Tracker chạy 2 fps.

## Chuẩn bị C2 (2026-10-25)
- **Đồng bộ đa cam bằng điện thoại:** video điện thoại thường là VFR, nên phải dùng timestamp của container. Tính theo `frame_idx/fps` có thể lệch vài giây sau một vài phút. Có thể viết 1 đoạn ngắn ở chương hệ thống: sự kiện tắt/bật đèn ở đầu và cuối session cho offset và drift (`tools/session_check.py`).
- **Người bị cắt chân ở mép dưới ảnh:** điểm chân lấy từ đáy bbox bị đẩy ra xa camera. WILDTRACK đo được sai số median 0.6–0.77 m, so với 0.09–0.23 m khi không bị cắt. Dùng gate riêng cho nhóm này và lấy vị trí cụm từ obs tin cậy. Với cam đặt thấp trong phòng nhỏ (C2) tình huống này hay gặp.
- **Calibration:** báo cáo sai số **leave-one-out**, không báo RMS trên chính các điểm dùng để fit. Với 6 điểm và homography 8 bậc tự do, RMS in-sample lạc quan.
- imgsz lớn hơn không cải thiện association: recall detection tăng nhưng đếm tệ hơn vì người ở xa có điểm sàn nhiễu (WILDTRACK, 1280 vs 960).

## Fall trên UR Fall (2026-10-25)
- Rule-based (tỉ lệ bbox + góc thân + drop của hông), cam0, test split: P/R/F1 0.650/0.867/0.743, latency median 1.2 s. Đủ làm baseline cho §7.7. ST-GCN là bước sau nếu cần.
- Bài học kỹ thuật: lịch sử chuyển động phải đi theo **global id**, không theo track id. Tracker đổi ID đúng lúc người ngã (chuyển động nhanh, hình dạng thay đổi). Lỗi này làm mất gần nửa recall.
- **Phản ví dụ cho fusion theo visibility:** cam trần có v cao nhưng không mang thông tin về tư thế, nên vmean kéo recall từ 0.733 xuống 0.533. Có thể viết vào phần thảo luận: visibility cần đi kèm "informativeness" của góc nhìn. Với PPE (C1) thì khác: cam thấy rõ đầu thì đúng là có thông tin về mũ, nên v hợp lý; còn với tư thế thì phụ thuộc góc nhìn.

## Fall đa góc (MCFD, 2026-10-26)
- Bảng chính cho §7.7: số cam 1 / 2 / 4 / 8 × vmean / max (MCFD test). 8 cam vmean F1 0.696 so với 0.474 của một cam, FA/giờ thấp hơn. Có thể vẽ biểu đồ F1 theo số cam, kèm error bar.
- Thảo luận: visibility là điều kiện cần chứ không đủ. Một góc nhìn phải vừa *thấy được* vừa *có thông tin* với đại lượng cần đo. Với PPE (C1) hai điều này trùng nhau (thấy rõ đầu thì biết có mũ), còn với tư thế thì không (cam top-down thấy rõ nhưng không phân biệt được đứng với nằm). UR Fall cam1 là phản ví dụ cụ thể.
- Phương pháp: phải đo fps thật của dataset (header AVI của MCFD ghi sai 120 fps). Nếu dùng sai thì mọi ngưỡng tính theo giây (hold, drop window) sai theo hệ số 4.
- **Rule vs học (2026-10-26):** 1D-CNN trên skeleton, train chỉ trên split val, cho MCFD 8 cam F1 0.762 (FA 12.7/giờ) so với rule 0.696 (38.2/giờ), ổn định qua 3 seed. Trên UR Fall một cam thì kết quả phụ thuộc seed. Nên báo cáo theo seed, không chọn seed tốt nhất.

## DeepStream so với backend Python (2026-10-26, `docs/experiments.md`)
- Cùng model (YOLO11m-pose@960 + PPE M2@320), DeepStream 7.1 (TensorRT FP16, batch theo cam) đạt khoảng 80 fps tổng so với 16 fps của Ultralytics: **3 cam 1080p real-time** trên GPU laptop. Đây là lý do kỹ thuật cho việc chọn DeepStream (C3).
- Hai backend xuất cùng schema JSONL (D4/D5) và cho kết quả association gần như nhau (pair F1 0.818 so với 0.810 trên WILDTRACK), nên mọi thí nghiệm offline vẫn đại diện cho hệ thống live.
- Bài học khi nối: crop SGIE phải giống crop lúc train (nới 10%), và bbox xuất ra nên lấy từ detector chứ không phải tracker khi fps thấp. Cả hai đều có số liệu trước/sau.
