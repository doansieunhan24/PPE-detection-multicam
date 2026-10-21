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
