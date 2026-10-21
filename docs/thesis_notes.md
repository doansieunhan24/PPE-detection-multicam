# Ghi chú cho thuyết minh

## Ý tưởng / lập luận

## Hình & bảng cần có

## Limitations bổ sung (ngoài CLAUDE.md §17)

- Phần cứng dev: RTX 4060 Laptop 8 GB — giới hạn số stream tối đa trong benchmark.

## Bằng chứng cho C1 từ dữ liệu public (SH17, 2026-10-21)
- Recall phát hiện mũ của người *thực sự đội mũ* phụ thuộc mạnh vào v: ~0.5 khi v<0.6, ≥0.9 khi v≥0.6 (SH17 test, M1). → luật "không thấy mũ ⇒ vi phạm" sai khoảng một nửa số lần khi đầu khó nhìn.
- v (thủ công) phân biệt "detector sẽ thấy PPE nếu có" tốt hơn confidence của detector người (AUC 0.79 vs 0.69 cho head) → biện minh cho việc dùng v thay vì conf (B4).
- Log-likelihood ratio: có box ≈ +3, không box ≈ −1 → bất đối xứng; không được coi "không thấy" mạnh ngang "thấy".
- Bẫy phương pháp: đo "độ đúng của bằng chứng" trên dataset mất cân bằng bị prevalence chi phối; phải đo recall | worn và LR. (Có thể đưa vào phần thảo luận.)
- Hình nên vẽ: recall|worn theo bin v (3 part), và log LR⁻ theo bin v.
