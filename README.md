# ppe-multicam — Intelligent Safety Monitoring

Đồ án tốt nghiệp: giám sát an toàn đa camera (PPE + hành vi nguy hiểm) trên NVIDIA DeepStream.
Thiết kế chi tiết, contribution và quy ước: xem [CLAUDE.md](CLAUDE.md).

## Cài đặt (dev, không cần DeepStream)

```powershell
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

```bash
# Linux / WSL
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Kiểm tra

```bash
pytest -q
ruff check src eval tools tests
```

## Cấu trúc

- `configs/` — mọi ngưỡng và tham số (không hardcode)
- `src/` — pipeline DeepStream, offline backend, analytics
- `eval/` — metrics, baselines, ablation, benchmark
- `tools/` — calibrate, vẽ zone, RTSP sim, gán nhãn
- `docs/experiments.md` — nhật ký thí nghiệm (bắt buộc)
