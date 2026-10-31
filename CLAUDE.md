# CLAUDE.md — Intelligent Safety Monitoring (Multi-Camera PPE + Hazard Analytics)

> File này là "bộ não" của dự án. Đọc hết trước khi viết code. Khi có quyết định thiết kế mới → cập nhật file này (mục 15 Decision Log).

---

## 0. Quy trình mỗi session (BẮT BUỘC)

Tiến độ được lưu trong thư mục `process/` (xem `process/README.md`).

**Đầu session**, trước khi làm việc gì khác:
1. Đọc `process/STATUS.md`.
2. Đọc file session gần nhất `process/YYYY-MM-DD_sNN.md` (đọc thêm các file cũ hơn nếu cần ngữ cảnh).
3. Tóm tắt ngắn cho người dùng: đang ở phase nào, việc dở dang, việc tiếp theo.

**Cuối session**, hoặc khi người dùng báo kết thúc, hoặc sau một mốc lớn:
1. Tạo file `process/YYYY-MM-DD_sNN.md` theo template trong `process/README.md`. Mỗi session một file, không sửa file của session cũ.
2. Ghi đè `process/STATUS.md` bằng trạng thái mới.
3. Thêm 1 dòng vào bảng "Danh sách session" trong `process/README.md`.
4. Nếu có quyết định mới thì ghi vào §15 Decision Log. Nếu phase thay đổi thì cập nhật dòng "Trạng thái hiện tại" ở §11.
5. Số liệu ghi vào process phải lấy từ output thật của script/test, không điền tay.

---

## 1. Bối cảnh

- **Loại dự án:** Đồ án tốt nghiệp (ĐH Phenikaa, ngành Kỹ thuật Điều khiển & Tự động hóa, lớp AI&RB).
- **Tên đề tài:** *Hệ thống giám sát an toàn thông minh trong khu vực nguy hiểm sử dụng phân tích video thời gian thực* (Intelligent Safety Monitoring System in Hazardous Areas Using Real-Time Video Analytics).
- **Yêu cầu của GVHD:**
  1. Không được chỉ là "train YOLO" → phải có **challenge kỹ thuật thật**.
  2. Phải có **contribution riêng** (đề xuất của bản thân, có baseline, có số liệu chứng minh).
- **Nền tảng bắt buộc:** NVIDIA DeepStream, đa camera, real-time.

### Mục tiêu theo phiếu giao đề → module tương ứng

| Mục tiêu trong phiếu | Module | Metric chính |
|---|---|---|
| Pipeline video real-time đa camera trên DeepStream | `src/pipeline/` | FPS, latency end-to-end, số stream |
| Phát hiện vi phạm PPE (mũ, áo phản quang, giày) + xâm nhập vùng cấm | `src/fusion/`, `src/behavior/intrusion.py` | Event P/R/F1, **False alarms/giờ** |
| Phân tích hành vi bất thường: té ngã, bất động, tụ tập | `src/behavior/` | Event P/R/F1, false alarms/giờ |
| Cảnh báo phân cấp + lưu video bằng chứng + thông báo tức thời | `src/alerts/` | Alert latency, độ đầy đủ evidence |
| So sánh với phương pháp hiện có | `eval/` | Bảng baseline vs proposed + ablation |

---

## 2. Contribution (KHÔNG thay đổi định nghĩa nếu chưa hỏi người dùng)

### C1 — Visibility-Aware Temporal PPE Evidence Fusion, từng camera (đóng góp chính)

**Vấn đề:** Hệ thống PPE truyền thống làm theo luật *"không detect được mũ ⇒ không đội mũ"*. Nhưng thường là do **camera không nhìn thấy** đầu: bị che, quay lưng, ở quá xa, bị mờ. Đây là nguồn **false alarm** lớn nhất khi triển khai thật.

**Đề xuất** (từ 2026-10-31: **chỉ xử lý trong từng camera**, không ghép người giữa các camera; người dùng quyết định, bài tập trung vào detection):
- Mỗi camera, với mỗi người (track) và mỗi bộ phận (head / torso / feet), mỗi frame xuất 2 giá trị:
  - `p` = xác suất có PPE trên bộ phận đó
  - `v` = **độ quan sát được** (visibility) của bộ phận đó trong frame đó
- Tích lũy bằng chứng **theo thời gian** trên track, trọng số `v`: frame bị che, quay lưng, mờ, người nhỏ đóng góp ít.
- Trạng thái có **3 mức**: `COMPLIANT` / `VIOLATION` / **`UNKNOWN`**. Nếu chưa đủ bằng chứng quan sát thì trả về `UNKNOWN` và **không báo động**.
- Gán PPE vào bộ phận bằng pose: mũ phải nằm ở vùng đầu. Mũ cầm trên tay (gần cổ tay) ⇒ **không** tính là đang đội.

**Claim mục tiêu:** giảm X% false alarm/giờ so với baseline mà recall vi phạm không giảm đáng kể.

### C2 — Bộ dữ liệu video đánh giá PPE (tự thu)

Các dataset PPE public (SH17, SHWD, CHV) đều là ảnh tĩnh nên không đánh giá được tích lũy theo thời gian, UNKNOWN và false alarm/giờ. Bộ tự thu là video từ các camera thật (2 × Tiandy), mỗi camera chấm như một hệ thống độc lập, kịch bản 11 take (`docs/c2_shooting_script.md`): che khuất, quay lưng, cầm mũ, đứng xa, hard negative, vùng cấm, té ngã giả, tụ tập. Nhãn là **khoảng thời gian trạng thái theo từng người** (interval labels).

### C3 — Hệ thống hoàn chỉnh real-time trên DeepStream (đóng góp kỹ thuật)

Gồm PPE, vùng cấm, té ngã/bất động, tụ tập và alert phân cấp. Phần này **không claim novelty**, chỉ cần chạy ổn định và có benchmark.

---

## 3. Quyết định kiến trúc đã chốt

| # | Quyết định | Lý do |
|---|---|---|
| D1 | **Giữ tracking trong từng camera** (`nvtracker`, NvDCF hoặc IOU) | Cần track ID để tích lũy theo thời gian, có 1 alert/người, chuỗi skeleton cho té ngã. Bỏ tracking = quay về detect từng frame. |
| D2 | **KHÔNG dùng appearance Re-ID cross-cam** | Công nhân mặc đồng phục giống nhau làm Re-ID hỏng. Tốn thêm model + công train. |
| D3 | **Không ghép người giữa các camera** (từ 2026-10-31, `association.mode: none`): mỗi `(cam, track_id)` là một người. Homography chỉ dùng cho vùng cấm / vùng rủi ro. Code Ground-Plane Association (§7.2) giữ làm tuỳ chọn `mode: ground_plane`, không dùng trong bài | Bài tập trung vào detection; mỗi camera là một hệ thống độc lập (người dùng quyết định). |
| D4 | **Metadata-first**: DeepStream chỉ làm perception rồi xuất metadata JSONL. Association, fusion, behavior, eval đều **chạy lại offline được** từ JSONL | Chạy ablation không cần chạy lại DeepStream. Kết quả tái lập được. |
| D5 | Có **perception backend thay thế** bằng Ultralytics (Python thuần), xuất **cùng schema JSONL** | Dev/eval trên máy không có DeepStream. Eval trên dataset ảnh/video public. |
| D6 | PGIE = **YOLO-pose** (person bbox + 17 keypoints COCO, 1 lần infer) | Bỏ được 1 model detect người riêng. |
| D7 | SGIE = **PPE detector chạy trên crop người** (helmet, vest, boots/shoes, + head/foot negatives) | Vật nhỏ được phóng to qua crop người, đỡ bỏ sót hơn detect toàn khung. |
| D8 | Mọi ngưỡng nằm trong `configs/*.yaml`, **không hardcode** | Phục vụ tune và ablation. |

---

## 4. Kiến trúc tổng thể

```
 RTSP/File x N
      │
      ▼
┌──────────────────────── DeepStream (perception only) ────────────────────────┐
│ nvurisrcbin ×N → nvstreammux(batch=N) → PGIE: YOLO-pose (person + kpts)      │
│   → nvtracker (per-cam IDs) → SGIE: PPE detector on person crops             │
│   → [probe] build PerceptionFrame → queue → JSONL writer / ZMQ publisher     │
│   → (optional) nvdsosd → tiler → display / RTSP out                          │
│ nvurisrcbin smart-record ← trigger từ alert engine                           │
└──────────────────────────────────────────────────────────────────────────────┘
      │ PerceptionFrame (JSONL / ZMQ)
      ▼
┌──────────────────────── Analytics (Python, offline-replayable) ──────────────┐
│ geometry: foot point → homography → floor (x,y) mét                          │
│ association: tắt (mode none) → global_id riêng cho mỗi (cam, track)         │
│ parts: keypoints → part ROIs (head/torso/feet) → gán PPE box → p             │
│ visibility: v cho từng (cam, person, part)                                   │
│ fusion: visibility-weighted log-odds + temporal → COMPLIANT/VIOLATION/UNKNOWN│
│ behavior: fall / immobility / intrusion / crowd                              │
│ alerts: rule engine → level L1/L2/L3 → dedupe/cooldown → DB + evidence + noti│
└──────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
 Dashboard (floor map + alert list + clip)      Telegram bot
```

---

## 5. Cấu trúc repo

```
ppe-multicam/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── docker/
│   ├── Dockerfile.deepstream          # base: nvcr.io/nvidia/deepstream:<ver>-triton-multiarch hoặc -devel
│   └── docker-compose.yml             # deepstream + mediamtx (RTSP sim) + dashboard
├── configs/
│   ├── cameras.yaml                   # cam_id, uri, fps, resolution, time_offset_ms
│   ├── calibration/cam{1,2,3}.yaml    # homography H (image→floor, mét), FOV polygon trên sàn
│   ├── zones.yaml                     # polygon vùng cấm / vùng rủi ro trên floor map + PPE yêu cầu theo zone
│   ├── fusion.yaml                    # λ/τ, θ_obs, θ_pos, θ_neg, T_alert, h_ref, ...
│   ├── behavior.yaml                  # fall/immobility/crowd thresholds
│   ├── alerts.yaml                    # level mapping, cooldown, telegram, smart-record pre/post sec
│   └── deepstream/
│       ├── pgie_yolo_pose.txt
│       ├── sgie_ppe.txt
│       ├── tracker_nvdcf.yml
│       └── labels_ppe.txt
├── models/                            # (gitignored) *.onnx, *.engine, *.pt
├── src/
│   ├── common/        schemas.py (pydantic/dataclass), time_sync.py, logging.py, config.py
│   ├── pipeline/      ds_app.py, probes.py, meta_export.py, smart_record.py
│   ├── perception_offline/  ultralytics_runner.py   # backend thay thế, cùng schema
│   ├── geometry/      homography.py, foot_point.py, floor_map.py
│   ├── association/   ground_plane.py, global_id.py
│   ├── parts/         part_rois.py, ppe_assign.py
│   ├── fusion/        visibility.py, evidence.py, state_machine.py, baselines.py
│   ├── behavior/      fall.py, immobility.py, intrusion.py, crowd.py
│   ├── alerts/        engine.py, levels.py, store.py (SQLite), notifier_telegram.py
│   ├── analytics_runner.py   # đọc JSONL/ZMQ → chạy toàn bộ analytics
│   └── dashboard/     app.py (FastAPI) + static/
├── training/
│   ├── ppe_detector/  prepare_sh17_person_crops.py, train.py, export_onnx.py
│   ├── part_classifier/  (giai đoạn sau, cho feet/boots)
│   └── fall/          prepare_skeleton_seq.py, train_stgcn.py
├── tools/
│   ├── calibrate_homography.py        # click điểm ảnh ↔ toạ độ sàn (mét) → YAML
│   ├── draw_zones.py
│   ├── rtsp_sim.sh                    # mediamtx + ffmpeg loop file → rtsp://
│   ├── sync_offsets.py                # tìm offset giữa các cam (clap/flash)
│   ├── label_intervals.py             # gán nhãn interval trạng thái theo người
│   └── replay.py                      # replay JSONL + video → visualize
├── eval/
│   ├── metrics.py                     # event matching, FA/hour, latency, coverage
│   ├── run_baselines.py
│   ├── run_ablation.py
│   ├── bench_fps.py
│   └── reports/                       # (generated) bảng, biểu đồ
├── data/                              # (gitignored) raw/, processed/, annotations/, jsonl/
├── process/                           # nhật ký tiến độ theo session (xem §0)
│   ├── README.md                      # template + danh sách session
│   ├── STATUS.md                      # trạng thái hiện tại (đọc đầu mỗi session)
│   └── YYYY-MM-DD_sNN.md
├── docs/
│   ├── experiments.md                 # nhật ký thí nghiệm (bắt buộc ghi)
│   └── thesis_notes.md                # ghi chú để viết thuyết minh
└── tests/
```

---

## 6. Schema dữ liệu

### 6.1 PerceptionFrame (output của DeepStream hoặc offline backend, 1 dòng JSONL / cam / frame)

```json
{
  "cam_id": "cam1",
  "frame_idx": 1234,
  "ts_ms": 1729500000123,
  "img_w": 1920, "img_h": 1080,
  "persons": [
    {
      "track_id": 17,
      "bbox": [x1, y1, x2, y2],
      "conf": 0.91,
      "kpts": [[x, y, c], ... 17 điểm COCO],
      "ppe_dets": [
        {"cls": "helmet", "bbox": [x1,y1,x2,y2], "conf": 0.82},
        {"cls": "vest",   "bbox": [...], "conf": 0.77}
      ],
      "truncated_bottom": false
    }
  ]
}
```
- `ppe_dets` bbox ở **toạ độ ảnh gốc** (convert từ crop nếu cần).
- `ts_ms` = wall-clock đã cộng `time_offset_ms` của cam (xem `time_sync.py`).

### 6.2 PersonPartState (output fusion)

```json
{"global_id": 5, "ts_ms": ..., "floor_xy": [3.2, 7.9],
 "parts": {
   "head":  {"item": "helmet", "state": "COMPLIANT", "score": 1.8, "obs": 2.4,
             "per_cam": {"cam1": {"p": 0.10, "v": 0.15}, "cam3": {"p": 0.95, "v": 0.90}}},
   "torso": {...}, "feet": {...}},
 "zone": "Z2"}
```

### 6.3 AlertEvent

```json
{"alert_id": "...", "type": "PPE_HELMET|PPE_VEST|PPE_BOOTS|INTRUSION|FALL|IMMOBILE|CROWD|UNVERIFIABLE",
 "level": 1|2|3, "global_id": 5, "zone": "Z2", "t_start_ms": ..., "t_raise_ms": ...,
 "cams": ["cam1","cam3"], "evidence": {"clip": "path.mp4", "snapshot": "path.jpg"}}
```

---

## 7. Đặc tả thuật toán

### 7.1 Foot point & homography (`geometry/`)

- Foot point: bottom-center của bbox (`association.foot_point: bbox_bottom`, mặc định). Biến thể `ankles`: trung điểm 2 ankle nếu cả 2 có `c > 0.5` (ankle cao hơn sàn ~8–10 cm nên homography đẩy điểm ra xa, xem §15).
- Nếu bbox chạm mép dưới ảnh (`truncated_bottom`) thì foot point không tin được: cặp có obs bị cắt dùng gate `association.d_max_truncated_m` (1.2 m), cost = khoảng cách/gate; vị trí sàn của cụm lấy từ obs không bị cắt.
- `H` (3×3) map ảnh → sàn (mét), tính bằng `cv2.findHomography` từ ≥ 4 cặp điểm (khuyến nghị 6–10 điểm, RANSAC). Lưu reprojection error vào YAML. **Báo cáo sai số này trong thuyết minh.**
- Mỗi cam có `fov_floor_polygon`: vùng sàn mà cam nhìn thấy. Chỉ ghép cặp người khi điểm nằm trong vùng giao FOV.

### 7.2 Ground-plane association (`association/`) — KHÔNG dùng từ 2026-10-31

> Mặc định `association.mode: none`. Phần dưới chỉ còn là tuỳ chọn `mode: ground_plane`.

1. Gom các frame của N cam trong cửa sổ đồng bộ `|Δts| ≤ sync_tol_ms` (mặc định 50 ms).
2. Với mỗi cặp cam (a, b): ma trận khoảng cách Euclid trên sàn → Hungarian, gating `d < d_max` (mặc định 0.8 m).
3. Union-find các cặp đã match để tạo cụm, mỗi cụm là 1 người. Gán `global_id`.
4. **Ổn định theo thời gian:** map `(cam_id, track_id) → global_id` bằng majority vote trên N frame gần nhất, tránh ID nhảy. Track mới được kế thừa global_id nếu khớp vị trí.
5. Người chỉ thấy ở 1 cam vẫn có global_id (cụm 1 phần tử).

### 7.3 Part ROIs & gán PPE (`parts/`)

| Part | Item | ROI từ keypoints | Fallback (thiếu kpt) |
|---|---|---|---|
| head | helmet | bbox(nose, eyes, ears), mở rộng ×1.5, **kéo lên trên 0.6×chiều cao** (mũ nằm trên đầu) | 20% trên của person bbox |
| torso | vest | bbox(shoulders, hips) mở rộng 10% | 25–60% chiều cao person bbox |
| feet | boots | bbox(ankles) mở rộng ngang, kéo xuống 0.5× | 12% dưới của person bbox |

- `p_part` = max conf của PPE box đúng class có **containment** (diện tích giao / diện tích PPE box) ≥ 0.5 với ROI.
- `p` được hiệu chỉnh theo prior trung tính (`configs/p_calibration.yaml`, fit trên val) cho cả trường hợp có box và không có box (thay cho `p_neg = 0.05` + conf thô). Baseline B0–B3 dùng cờ "có box".
- **Rule cầm mũ:** helmet box nằm gần wrist keypoints (khoảng cách < 0.15 × chiều cao người) và không chứa trong head ROI thì không tính cho head, chỉ log `held_helmet=true`.
- Fallback ROI (thiếu kpt) thì nhân `v` thêm `fallback_penalty` (0.5).
- (Giai đoạn sau) Feet: thay detector bằng **part classifier** trên crop độ phân giải cao (crop từ frame gốc, không phải từ input đã resize).

### 7.4 Visibility `v` (`fusion/visibility.py`)

```
v = s_kp · s_size · (1 − r_occ) · s_blur · s_fallback        (clip về [0,1])

s_kp     = mean confidence các keypoint của part
           (head: nose, l/r eye, l/r ear | torso: l/r shoulder, l/r hip | feet: l/r ankle)
s_size   = min(1, h_roi_px / h_ref)        # h_ref mặc định 32 px
r_occ    = tỷ lệ diện tích ROI bị che bởi bbox người khác đứng GẦN cam hơn
           (gần hơn ≈ foot y lớn hơn) + phần ROI nằm ngoài khung ảnh
s_blur   = min(1, var(Laplacian(ROI)) / β)  # optional, có cờ bật/tắt
s_fallback = 1 hoặc fallback_penalty
```
- **Biến thể ablation `v_learned`:** logistic regression trên các feature (s_kp, h_roi, r_occ, blur) để dự đoán "detector đúng/sai" trên tập val.

### 7.5 Evidence fusion + state machine (`fusion/evidence.py`, `state_machine.py`)

Với người `g`, part `k`, tại thời điểm `t` (khi không ghép cam, mỗi người chỉ có một camera `c` nên tổng Σ_c chỉ có một số hạng):

```
e_c   = v_c · logit(clip(p_c, 0.01, 0.99))          # bằng chứng từ cam c
E_t   = Σ_c e_c                                     # tổng bằng chứng
W_t   = Σ_c v_c                                     # tổng độ quan sát
λ     = exp(−Δt / τ)                                # τ mặc định 2 s
S_t   = λ·S_{t−1} + Δt·E_t                         # dt_weighting (mặc định bật)
Ω_t   = λ·Ω_{t−1} + Δt·W_t                         # Ω ≈ W·τ (visibility-giây), không phụ thuộc FPS
m_t   = S_t / max(Ω_t, ε)                           # log-odds trung bình có trọng số

if   Ω_t < θ_obs:        state = UNKNOWN
elif m_t ≥ θ_pos:        state = COMPLIANT
elif m_t ≤ θ_neg:        state = VIOLATION
else:                    state = state_{t−1}        # hysteresis
```
- **Alert PPE:** `VIOLATION` liên tục ≥ `T_alert` (mặc định 3 s), và zone hiện tại yêu cầu item đó (theo `zones.yaml`).
- `UNKNOWN` kéo dài ≥ `T_unk` (mặc định 30 s) trong vùng bắt buộc PPE thì phát L1 `UNVERIFIABLE`, **không** phát L2.
- Cooldown theo `(global_id, type)`: `T_cd` (mặc định 60 s).
- Biến thể ablation: **Dempster–Shafer** (mass cho {PPE}, {noPPE}, {PPE,noPPE}=ignorance, với ignorance = 1 − v).

### 7.6 Baselines (`fusion/baselines.py`) — bắt buộc implement để so sánh

| ID | Mô tả |
|---|---|
| B0 | Per-frame: không detect được PPE thì vi phạm ngay (cách làm phổ biến) |
| B1 | Persistence: vi phạm liên tục ≥ T_alert mới báo |
| B4 | Tích lũy theo thời gian có trọng số **detector conf** (không có v) + persistence |
| **P** | **Proposed:** v-weighted log-odds + UNKNOWN + temporal |

B2 (OR nhiều cam) và B3 (majority vote nhiều cam) chỉ có nghĩa khi ghép người giữa các camera; không dùng từ 2026-10-31 (code giữ lại).

### 7.7 Behavior (`behavior/`)

- **Fall:**
  - Baseline rule: tỉ lệ khung hình w/h của bbox, góc thân (vector vai→hông so với phương thẳng đứng), vận tốc dọc của hông.
  - Learned: chuỗi skeleton 30–60 frame (chuẩn hoá theo bbox) → ST-GCN (hoặc 1D-CNN/LSTM nhẹ).
  - Mỗi camera tự quyết (không ghép cam). Phần fuse đa góc theo `v` (`fall.fusion`) chỉ còn chạy khi `association.mode: ground_plane`; số liệu MCFD đa cam (s07) là thí nghiệm trước khi đổi hướng.
- **Immobility:** dịch chuyển keypoint trung bình < ε trong ≥ `T_immobile` **và** tư thế nằm/ngồi bệt **và** nằm trong vùng rủi ro, hoặc ngay sau một sự kiện fall. Phân biệt với người đứng yên làm việc bình thường.
- **Intrusion:** `floor_xy` nằm trong polygon vùng cấm ≥ `T_intr` (mặc định 1 s). Dùng toạ độ sàn đã fuse, không dùng ROI ảnh, để không bị lệch phối cảnh.
- **Crowd:** số người trong zone rủi ro > K (mặc định K = 3, tức ≥ 4 người) trong ≥ T giây. Không ghép cam thì đếm trong từng camera và lấy camera thấy nhiều người nhất (không cộng các cam, tránh đếm trùng).

### 7.8 Alerts (`alerts/`)

| Level | Loại |
|---|---|
| L1 Info | UNVERIFIABLE, PPE thiếu nhẹ ở zone ít rủi ro |
| L2 Warning | PPE_* trong zone bắt buộc, INTRUSION, CROWD |
| L3 Critical | FALL, IMMOBILE |

- Evidence: kích **smart record** của DeepStream (`nvurisrcbin` có các thuộc tính `smart-record*` và signal start-sr/stop-sr; **kiểm tra lại docs đúng version DS đang dùng**), quay pre/post N giây. Kèm snapshot frame có OSD.
- Lưu trữ: SQLite (`alerts.db`). Thông báo qua Telegram bot (token để trong `.env`, **không commit**).
- Dashboard: FastAPI + 1 trang HTML gồm floor map với chấm người (màu theo state), danh sách alert, link clip.

---

## 8. DeepStream — ghi chú kỹ thuật & bẫy hay gặp

- **Version:** dùng DeepStream 7.x trên dGPU (Docker `nvcr.io/nvidia/deepstream`). Chốt 1 version, ghi vào Decision Log. Kiểm tra tương thích driver/CUDA/TensorRT.
- **Jetson:** DeepStream 7.x cần dòng **Orin** (JetPack 6.x). **Jetson Nano đời đầu chỉ chạy tới DS 6.0.1 (JetPack 4.6)**, không chạy DS 7. Nếu có edge thì benchmark riêng, đừng phụ thuộc vào nó.
- **YOLO trong DS:** dùng custom parser cộng đồng (repo `marcoslucianops/DeepStream-Yolo` cho detector, `DeepStream-Yolo-Pose` cho pose). Export ONNX theo script của repo. **Kiểm tra repo lưu keypoints ở metadata nào** (`mask_params` hay user meta) rồi viết probe đọc đúng chỗ.
- **nvstreammux:** `batch-size = số cam`, `live-source=1` với RTSP, set `width/height`, `batched-push-timeout` ≈ 1/fps. Cân nhắc new streammux (`USE_NEW_NVSTREAMMUX=yes`) nếu các cam khác fps hoặc độ phân giải.
- **SGIE trên crop:** `process-mode=2`, `operate-on-gie-id=<pgie id>`, `operate-on-class-ids=0`. Đặt `input-object-min-width/height` để bỏ qua người quá nhỏ, nhưng vẫn phải xuất người đó với `v` thấp.
- **Tracker:** NvDCF (`config_tracker_NvDCF_perf.yml`) làm mặc định, IOU dùng khi cần tiết kiệm. Track ID chỉ duy nhất **trong từng source**.
- **Probe:** đặt sau SGIE. Trong probe **chỉ copy metadata ra dict rồi đẩy vào queue**, không xử lý nặng (sẽ block pipeline). Ghi JSONL hoặc publish ZMQ ở thread riêng.
- **Toạ độ:** metadata trả theo độ phân giải của streammux. Convert về độ phân giải gốc của từng cam trước khi ghi JSONL.
- **RTSP trong WSL2 phải dùng TCP** (`select-rtp-protocol=4` ở cả deepstream-app và `nvurisrcbin`, đã kiểm bằng gst-inspect DS 7.1): RTP/UDP lỗi IPv6 `Invalid address family`.
- **RTSP simulation:** `mediamtx` + `ffmpeg -re -stream_loop -1 -i camX.mp4 -c copy -f rtsp rtsp://localhost:8554/camX` (script ở `tools/rtsp_sim.sh`).
- **Đo FPS:** dùng `GST_DEBUG` hoặc probe đếm frame/giây theo từng source. Đo latency = `ts output alert − ts frame capture`.

---

## 9. Dữ liệu

| Mục đích | Dataset | Ghi chú |
|---|---|---|
| Train PPE detector | **SH17** (17 class: person, head, helmet, safety-vest, shoes, foot, gloves, ...) | Tạo **tập crop người** (`prepare_sh17_person_crops.py`) cho khớp với phân phối input của SGIE. Lưu ý: class "shoes" **không phân biệt giày bảo hộ với giày thường**, phải ghi rõ ở phần Limitations. |
| Bổ sung helmet | SHWD, CHV | Merge class cẩn thận (map tên class). |
| **Đã có (2026-10-21)** | SH17: 8,099 ảnh / 75,994 instance (khớp README), chỉ 927 helmet + 530 safety-vest. CHV: 1,330 ảnh, person/vest/4 màu mũ (3,887 person, 1,784 vest, 3,538 helmet), không nhãn head/shoes/foot → pseudo-label bằng model SH17 (chỉ train/val) | Chi tiết số crop/instance: `data/processed/*/stats.json`, `docs/experiments.md` |
| Association (đa cam, public) | **WILDTRACK** (EPFL, 7 cam, 400 frame có nhãn @2 fps) | Chỉ để eval association; **không dùng nữa từ 2026-10-31** (không ghép cam). License TODO(verify) |
| Fall (đơn góc) | Le2i, UR Fall, CAUCAFall | Train/eval module fall. |
| Fall (đa góc) | Multiple Cameras Fall Dataset (Univ. Montréal, 8 cam) | Đánh giá fusion fall đa camera. Đã dùng (`tools/mcfd_prepare.py`): fps thật 30, GT theo TR 1350 + erratum 2025 |
| **Đánh giá C1** | **Bộ tự thu (C2)** | Chỉ dùng để **eval** (hoặc chia theo session), **không train detector trên bộ này**. |

> **Kiểm tra license và link chính thức của từng dataset trước khi dùng, ghi vào `docs/experiments.md`.** Không tự bịa số lượng ảnh hay class. Đọc từ file của dataset.

### 9.1 Kế hoạch thu bộ tự thu (C2)

- Camera thật: 2 × Tiandy TC-C131N (`configs/cameras_lan.yaml`). Các cam **không cần nhìn chung vùng sàn**; mỗi cam chấm riêng. Homography chỉ cần cho cam thấy vùng cấm / vùng rủi ro.
- Đồng bộ (vỗ tay hoặc bật/tắt đèn đầu session, `tools/sync_offsets.py`) chỉ để gán nhãn một lần dùng cho mọi cam.
- **Kịch bản:** 11 take T01–T11 trong `docs/c2_shooting_script.md` (kèm bảng nhãn). T04 (quay lưng / vật cản, đủ PPE) là take quan trọng nhất cho C1.
- **Nhãn:** interval `(person_gt_id, item, state, t_start, t_end)` qua `tools/label_intervals.py`. Kèm GT person ID cho mỗi `(cam, track)` để nối alert với người. Nếu có thể, thêm bbox keyframe trên CVAT để đo detector.
- Tổng mục tiêu: khoảng 15–30 phút mỗi cam.

---

## 10. Đánh giá

### 10.1 Metric

| Mức | Metric |
|---|---|
| Detector | mAP50, mAP50-95 theo class (riêng helmet/vest/shoes), trên SH17 test |
| **PPE compliance (C1)** | Event P/R/F1, **False Alarms / giờ**, time-to-detect, **% thời gian UNKNOWN** (coverage) |
| Behavior | Event P/R/F1 cho fall/immobility/intrusion/crowd, FA/giờ |
| Hệ thống | FPS theo từng stream, số stream tối đa @ real-time, latency end-to-end, GPU util/mem |

### 10.2 Event matching (`eval/metrics.py`)

- Alert dự đoán khớp với GT interval nếu cùng người (qua mapping global_id ↔ gt_id), cùng type, và `t_raise ∈ [t_start − δ, t_end + δ]` (δ mặc định 2 s).
- Mỗi GT interval khớp tối đa 1 alert. Alert không khớp với GT nào là **false alarm**.
- FA/giờ = số false alarm / tổng thời lượng video (giờ).
- **Chấm từng camera** (khi `association.mode: none`, mặc định): mỗi (session, cam) chạy và chấm riêng như một hệ thống đơn cam rồi gộp; FA/giờ tính theo giờ-cam; GT interval tính một lần cho mỗi cam. Cam không có homography không được chấm INTRUSION.

### 10.3 Thí nghiệm bắt buộc

1. B0, B1, B4 vs P trên bộ tự thu (chấm từng camera), bảng chính của luận văn, kèm bảng theo từng camera.
2. Ablation của P: bỏ `v`, bỏ UNKNOWN, bỏ temporal, `v` thủ công vs `v_learned`, log-odds vs Dempster–Shafer.
3. ~~Số camera 1 / 2 / 3~~: bỏ từ 2026-10-31 (không ghép cam).
4. Độ nhạy tham số: θ_obs, τ, T_alert (vẽ đường cong FA/giờ vs recall).
5. Detector: full-frame vs crop người (SGIE) vs (sau này) part classifier cho feet.
6. Benchmark FPS/latency theo số stream (1, 2, 3, 4, ...).

> **Mọi con số trong báo cáo phải sinh ra từ script trong `eval/`, có ghi commit hash và config.** Không điền số tay, không ước lượng.

---

## 11. Lộ trình (khoảng 16 tuần, điều chỉnh theo hạn thật)

| Phase | Tuần | Việc | Done khi |
|---|---|---|---|
| P0 | 1 | Docker DeepStream, chạy sample 3 RTSP (mediamtx), tiler hiển thị | 3 stream chạy, có số FPS |
| P1 | 2–3 | YOLO-pose vào PGIE, tracker, export JSONL. Offline backend Ultralytics cùng schema | JSONL từ 2 backend so khớp được |
| P2 | 3–4 | Train PPE detector (SH17 crop người), export ONNX, SGIE | mAP trên SH17 test, SGIE chạy trong DS |
| P3 | 4–5 | `calibrate_homography.py`, `draw_zones.py`, foot point, floor map | Reprojection error < ~0.2 m |
| P4 | 5–6 | **Thu bộ dữ liệu C2** + gán nhãn interval | Có GT cho ≥ 15 phút |
| P5 | 6–8 | Association + part ROIs + visibility + fusion + baselines B0–B4 | Bảng B0–B4 vs P chạy từ JSONL |
| P6 | 8–10 | Fall/immobility/intrusion/crowd | Event metrics cho từng module |
| P7 | 10–12 | Alert engine, smart record, Telegram, dashboard | Demo end-to-end live |
| P8 | 12–14 | Ablation, sensitivity, FPS bench, (optional) Jetson Orin | Toàn bộ bảng/biểu đồ trong `eval/reports/` |
| P9 | 14–16 | Viết thuyết minh, slide, video demo | — |

**Trạng thái hiện tại:** P1, P2 xong; code P5–P7 xong và test offline (association đã kiểm chứng trên WILDTRACK). Chuỗi xử lý C2 (session_check → nhãn → baselines/ablation/sensitivity) đã chạy thử từ đầu đến cuối. P6: event metrics cho fall (UR Fall + MCFD), intrusion và crowd (WILDTRACK); immobility chờ C2. P0 xong; P1/P2 phần DeepStream xong: YOLO11m-pose + NvDCF + SGIE PPE chạy trong DS 7.1, JSONL khớp backend offline, 3 cam 1080p đạt 26 fps/stream (4 cam 19.8). P7 demo live end-to-end chạy được (RTSP → DS → ZMQ → `analytics_live` → SQLite/dashboard, smart record theo alert, OSD qua mediamtx). Chờ dữ liệu C2 (P4). Đánh giá C2 đã gom thành một lệnh `python -m eval.run_c2` (val → test, tune từng phương pháp, mọi bảng §10.3 trừ detector/FPS), đã chạy thử bằng GT WILDTRACK. Từ 2026-10-31 không ghép người giữa các cam (`association.mode: none`), đánh giá C2 chấm từng camera; camera thật: 2 × Tiandy qua LAN. Chi tiết: `process/STATUS.md`.

---

## 12. Môi trường & lệnh

```bash
# Dev Python (analytics, eval, offline backend), không cần DeepStream
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # ultralytics, opencv-python, numpy, scipy, pydantic, pyyaml, fastapi, uvicorn, pyzmq, pytest, ruff

# RTSP simulation
bash tools/rtsp_sim.sh data/raw/session01/

# DeepStream (trong container)
docker compose -f docker/docker-compose.yml up deepstream
python src/pipeline/ds_app.py --cameras configs/cameras.yaml --out data/jsonl/session01/

# Offline perception (không DS)
python -m src.perception_offline.ultralytics_runner --cameras configs/cameras.yaml --out data/jsonl/session01/

# Analytics từ JSONL (offline, tái lập)
python -m src.analytics_runner --jsonl data/jsonl/session01/ --config configs/ --out data/results/session01/

# Eval
python -m eval.run_baselines --session data/results/session01 --gt data/annotations/session01
python -m eval.run_ablation  --session ... --gt ...
python -m eval.bench_fps --streams 1 2 3 4

# Test & lint
pytest -q && ruff check src eval tools
```

---

## 13. Quy ước code

- Python ≥ 3.10, có type hints. Schema dùng pydantic hoặc dataclass trong `src/common/schemas.py` làm **nguồn duy nhất**.
- Không hardcode đường dẫn hay ngưỡng. Mọi thứ lấy từ `configs/`.
- Toạ độ: ảnh là pixel ảnh gốc, sàn là mét. Đặt tên biến rõ `*_px`, `*_m`. Thời gian là `*_ms` (int).
- Hàm thuần (geometry, fusion, metrics) phải có unit test, nhất là `evidence.py`, `state_machine.py`, `metrics.py`.
- Logging bằng `logging`, không dùng print trong `src/`.
- Seed cố định cho mọi thứ ngẫu nhiên (training, split).
- `data/`, `models/`, `*.engine`, `.env` nằm trong `.gitignore`.

---

## 14. Quy tắc cho Claude

1. **Không đổi định nghĩa contribution** (mục 2) hay các quyết định D1–D8 nếu chưa hỏi người dùng.
2. **Không bịa số liệu, tên class, kích thước dataset, API DeepStream.** Không chắc thì đọc docs hoặc source, hoặc ghi `TODO(verify)`.
3. Sửa logic fusion, association hay metrics thì phải chạy lại test và eval liên quan, ghi kết quả vào `docs/experiments.md` (ngày, commit, config, số liệu).
4. Code DeepStream: probe phải nhẹ, phần xử lý nặng đưa sang analytics.
5. Feature mới phải chạy được **offline từ JSONL** trước, rồi mới nối vào live.
6. Baseline phải implement **công bằng**: cùng detector, cùng tracker, cùng tham số persistence ở chỗ tương ứng.
7. Ưu tiên chạy được end-to-end sớm (P0–P1), sau đó mới tối ưu.
8. Khi có ý tưởng hay kết quả đáng đưa vào thuyết minh thì ghi vào `docs/thesis_notes.md`.
9. **Luôn tuân theo quy trình session ở §0:** đầu session đọc `process/`, cuối session ghi vào `process/`.

---

## 15. Decision Log

| Ngày | Quyết định | Lý do |
|---|---|---|
| 2026-10-21 | Chốt contribution C1 (visibility-aware multi-view PPE fusion + UNKNOWN), C2 (dataset đa góc), C3 (hệ thống DS) | GVHD yêu cầu challenge + contribution riêng |
| 2026-10-21 | Giữ tracking per-cam, bỏ appearance Re-ID, dùng ground-plane association | Re-ID kém với đồng phục; homography tái dùng cho vùng cấm |
| 2026-10-25 | DeepStream version: **chốt 7.1** (`7.1-triton-multiarch`); P0 chạy được 2026-10-26 (3 RTSP ~30 fps/stream, `docker/README.md`) | Docs "DeepStream On WSL" 7.1 hỗ trợ Ubuntu 22.04 (distro WSL hiện có); từ 8.0 trở đi yêu cầu Ubuntu 24.04 |
| 2026-10-21 | GPU dev: RTX 4060 Laptop 8 GB, Windows 11 + WSL2 Ubuntu-22.04 | Máy hiện có. DeepStream 7.1 chạy qua Docker trong WSL2 (đã kiểm chứng 2026-10-26) |
| 2026-10-21 | Thư mục repo thực tế là `ppe_de/` (package name `ppe-multicam`) | Tên trong mục 5 chỉ là tên package |
| 2026-10-21 | **§7.5 chuẩn hoá theo Δt** (`dt_weighting`, mặc định bật): `S_t = λS + Δt·E`, `Ω_t = λΩ + Δt·W` → Ω hội tụ ≈ W·τ (visibility-giây), không phụ thuộc FPS. **Đã xác nhận 2026-10-26** | Công thức gốc cộng theo frame ⇒ Ω ≈ W·fps·τ: ở 25 fps v=0.05 đã vượt θ_obs=1 → UNKNOWN gần như không bao giờ xảy ra. Có test minh hoạ (`tests/test_fusion.py`). Có thể tắt cờ để so sánh |
| 2026-10-21 | Schema thêm `PersonDet.ppe_checked` | Phân biệt "detector không chạy" (không có bằng chứng) với "chạy mà không thấy PPE" (p_neg) |
| 2026-10-21 | Pose PGIE offline: `yolo11m-pose`, imgsz 960 | Trên vtest.avi (người ~70 px) imgsz 640 bỏ sót gần hết; 960 tăng ~3.7× số người phát hiện |
| 2026-10-21 | Hiệu chỉnh p theo prior trung tính (`configs/p_calibration.yaml`, fit SH17 val) thay `p_neg=0.05` + conf thô; `theta_neg` −1.0 → −0.5; baseline B0–B3 dùng cờ "có box" (không dùng p đã hiệu chỉnh). **Đã xác nhận 2026-10-26** | Đo trên SH17: không-box chỉ ≈ −1 log-LR, có box ≈ +3; p_neg 0.05 phóng đại bằng chứng âm. Tune lại trên C2 |
| 2026-10-21 | Bổ sung CHV cho helmet/vest; pseudo-label head/shoes/foot | SH17 quá ít helmet/vest |
| 2026-10-22 | Detector chính = M2 (SH17+CHV crops, YOLO11s@320) `models/sh17chv_crop_y11s` | CHV test helmet AP50 0.957, vest 0.913 (M1: 0.890 / 0.794) |
| 2026-10-22 | Part-ROI/containment tune trên val SH17+CHV (containment 0.2, head_up 1.0, head_expand 1.8, torso_expand 0.4, feet 0.12/0.1); hiệu chỉnh p chỉ trên v ≥ 0.6 | Tăng recall gán PPE 12–23 điểm trên test; GT ảnh đơn bị điều kiện theo visibility nên không fit được tác động của v |
| 2026-10-22 | Giữ M2 (YOLO11s) thay vì M3 (YOLO11m) | M3 +2 điểm mAP50 SH17 nhưng ngang trên CHV, chậm ×1.64 |
| 2026-10-22 | Association eval trên **WILDTRACK** (`eval/eval_association.py`, `tools/wildtrack_*.py`); tách `src/association/associator.py` dùng chung cho engine và eval | Có số liệu association trên dữ liệu đa cam thật trước C2 |
| 2026-10-22 | `association.fov_margin_m = 0.5`: FOV gate có dung sai quanh polygon | Người bị cắt ở mép ảnh có điểm chân nằm ngay ngoài FOV. WILDTRACK oracle: pairwise recall 0.940 → 0.973 |
| 2026-10-22 | **§7.1 foot point = đáy bbox** (`association.foot_point: bbox_bottom`, có thể đổi lại `ankles`). **Đã xác nhận 2026-10-26** | Ankle cao hơn sàn khoảng 8–10 cm, homography đẩy điểm ra xa. WILDTRACK: sai số sàn median 0.68 m (ankle) → 0.23 m (bbox), pairwise F1 0.37 → 0.78 |
| 2026-10-25 | Video file: timestamp theo **pts** (`perception.file_timestamps: pts`) | Điện thoại quay VFR. Video tổng hợp: tính theo index lệch 2.4 s sau 150 frame |
| 2026-10-25 | `cameras.yaml` / `zones.yaml` đặt trong thư mục JSONL của session ghi đè `configs/`; ngưỡng luôn lấy từ `configs/` | Mỗi session C2 có offset (và có thể cả calibration/zone) riêng |
| 2026-10-25 | **§7.1 bbox bị cắt mép dưới:** cặp có obs bị cắt dùng gate `association.d_max_truncated_m` (1.2 m), cost = khoảng cách/gate; vị trí sàn của cụm lấy từ obs không bị cắt. **Đã xác nhận 2026-10-26** | WILDTRACK: sai số sàn median của box bị cắt 0.61–0.77 m, so với 0.09–0.23 m khi không bị cắt; oracle F1 0.982 → 0.985, count MAE 1.09 → 0.92 |
| 2026-10-25 | Giữ pose imgsz 960 (không dùng 1280) | WILDTRACK 1280: det recall +5 điểm nhưng pair F1 0.786 → 0.741, count MAE 3.35 → 7.04 |
| 2026-10-25 | Calibration báo cáo sai số **leave-one-out** (`loo_error_rms_m`) | RMS in-sample lạc quan khi chỉ có 6–8 điểm |
| 2026-10-25 | Lịch sử behavior (drop hông, keypoint trước) theo `(cam, global_id)` thay vì `(cam, track_id)`; `fall.hold_s` 0.5 → 0.2 | UR Fall: tracker đổi ID lúc ngã làm mất drop; hold 0.2 tune trên val (số lẻ), test F1 0.667 → 0.743 nhưng FA/giờ 57.8 → 134.8 |
| 2026-10-25 | `fall.fusion: vmean` (§7.7) giữ nguyên, thêm `max` làm ablation. **Hỏi người dùng** có thêm trọng số theo mức hữu ích của góc nhìn không | UR Fall cam0 + cam1 trần: vmean recall 0.533 < cam0 0.733 |
| 2026-10-26 | `fall.drop_ratio` 0.3 → 0.2; giữ `fall.fusion: vmean` | Val MCFD + UR Fall. MCFD test: 8 cam vmean F1 0.696 so với 1 cam 0.474, max 0.560. Cam top-down (UR Fall cam1) làm vmean giảm, đề xuất loại cam đó khỏi phần fuse fall (chờ người dùng) |
| 2026-10-26 | Biến thể fall học `fall.method: seq` (1D-CNN trên chuỗi skeleton 1.5 s), mặc định vẫn `rule` | Thay ST-GCN bằng model nhẹ hơn vì dữ liệu ít. Train chỉ trên split val; MCFD test 8 cam F1 0.762 so với 0.696 (3 seed); UR Fall một cam không ổn định |
| 2026-10-26 | **DeepStream xuất bbox người của detector** (`ds_app --person-bbox detector`, fallback bbox tracker khi frame đó không detect) | Bbox NvDCF ở WILDTRACK 2 fps: pair F1 0.732 so với 0.818 (offline 0.810) |
| 2026-10-26 | **SGIE crop nới `ppe.crop_pad`** bằng probe ở sink pad SGIE (giống lúc train và backend offline) | Không nới: mức khớp PPE với offline shoes 0.818, head 0.901; có nới 0.925 / 0.944 |
| 2026-10-26 | Engine PGIE batch 4 dùng chung cho 1–4 cam (`ds_app` đặt batch-size = số cam); ONNX DS để ở `models/ds/` | Build engine pose mất khoảng 6 phút; layout ONNX của DeepStream-Yolo khác bản Ultralytics |
| 2026-10-27 | **Kênh live:** `ds_app` bind ZMQ PUB :5555 (PerceptionFrame) và PULL :5556 (alert từ analytics → smart record + chữ trên OSD); `src/analytics_live.py` dùng cùng lớp `Analytics` với runner offline, ghép frame bằng `LiveSyncer` | Giữ D4: live và replay cho cùng alert (có test). Clip bằng chứng [−pre_s, +post_s] quanh lúc kích |
| 2026-10-27 | `ds_app` đặt `GIO_USE_PROXY_RESOLVER=dummy` | ≥ 2 nguồn RTSP crash SIGSEGV trong libproxy của GIO (gdb), cả bản trước đó |
| 2026-10-29 | **Đánh giá C2: tune trên session val, báo cáo trên session test** (`configs/c2_split.yaml`, `eval/run_c2.py`). Ngưỡng tune **cho từng phương pháp** bằng cùng thủ tục (coordinate descent, mục tiêu F1 PPE val, `eval/tune_c2.SPACE`); hiệu chỉnh p / v_learned fit lại trên C2 val, bộ phận thiếu mẫu (< 50 mỗi lớp) giữ fit SH17/CHV; báo kèm bảng chưa tune | Công bằng giữa baseline và P (§14.6); tránh báo số trên tập đã tune |
| 2026-10-29 | CROWD gán nhãn theo zone (`gt_id` trống), ghép theo loại + thời gian | Tụ tập không thuộc về một người |
| 2026-10-29 | `PPEEngine.step = decide(observe())`; eval cache phần observe (`run(obs_cache=)`) | Observe không phụ thuộc policy và mục `evidence/alert/baselines`; test chứng minh cùng output; toàn chuỗi C2 chạy thử 100 s |
| 2026-10-29 | Snapshot alert live lấy từ clip smart record + overlay vẽ từ metadata (`src/alerts/snapshot.py`) | Clip smart record không có OSD; offset chính xác khoảng 1 GOP |
| 2026-10-29 | **Không thêm cờ loại cam nhìn từ trên xuống** cho fuse fall; giữ `fall.fusion: vmean` (người dùng xác nhận) | Hệ thống và C2 không có cam nhìn từ trên xuống (cam đặt thấp, nhìn ngang/chéo). Vấn đề chỉ gặp ở UR Fall cam1 (trần); ghi vào Limitations |
| 2026-10-31 | **Bỏ ghép người giữa các camera** (người dùng quyết định): `association.mode: none` trong `configs/zones.yaml`, mỗi `(cam, track_id)` là một người riêng. Code ground-plane (D3) giữ lại, bật bằng `mode: ground_plane` | Không fuse đa cam theo người nữa: fusion C1 chỉ còn theo thời gian trong từng cam; B2–B4, thí nghiệm 1/2/3 cam, đếm CROWD và fuse fall đa cam cần xem lại. `eval_association` vẫn đo phương pháp ground-plane |
| 2026-10-31 | Khi `association.mode: none`: **eval chấm từng camera riêng** (`run_baselines`/`tune_c2`/`run_sensitivity --per-cam auto`, bỏ B2/B3 và thí nghiệm số cam); **CROWD đếm theo cam thấy nhiều người nhất** trong zone thay vì cộng mọi `global_id` | Không ghép người thì mỗi cam là một hệ thống độc lập; cộng `global_id` giữa các cam đếm trùng một người k lần |
| 2026-10-31 | **Cam thật:** 2 × Tiandy TC-C131N (1080p H264 25 fps, RTSP `/1/1`, substream `/1/2`) qua Ethernet, `configs/cameras_lan.yaml`; tài khoản cam trong `.env` (`CAM_USER`/`CAM_PASS`), `uri` dùng `${VAR}` và `load_cameras` thế + URL-encode | Không commit mật khẩu; mật khẩu có `@` phải encode trong URL. IP link-local 169.254.x có thể đổi khi cam khởi động lại. Ống kính méo cong rõ: cần xử lý trước homography |
| 2026-10-31 | **Bài chỉ xử lý từng camera** (người dùng quyết định, bài chủ yếu là detection): C1 = tích lũy theo thời gian có trọng số `v` + UNKNOWN trên từng camera; baseline B0, B1, B4 (bỏ B2/B3, bỏ thí nghiệm số cam); C2 = video 2 cam Tiandy, 11 take; không chạy thêm WILDTRACK; `crowd.K` 4 → 3 (T10 ≥ 4 người phải báo) | Sửa §2, D3, §4, §7.2, §7.5–7.7, §9, §10, §16, §17 cho khớp |
| 2026-10-31 | **Calibration cam thật bằng sàn gạch caro + mô hình méo ống kính** (`src/geometry/floor_grid.py`, `calibrate_homography.py grid / fit --undistort`): `cam*.yaml` có thêm `K`, `dist`; `to_floor` khử méo rồi mới nhân H. Chỉ fit k1, k2 (f, tâm ảnh cố định). Gạch 0.67 m. **cam1 (hành lang) không có homography**; vùng cấm Z2 / vùng rủi ro Z3 trong khung cam2 (người dùng chọn cam2) | Ống kính Tiandy méo thùng: cam2 LOO 0.156 m → 0.039 m khi khử méo. Một mặt phẳng không tách được f với k1, fit K đầy đủ không ổn định. Hành lang chỉ có một hàng góc gạch thẳng hàng |
| — | Có Jetson Orin không: `TODO` | |

---

## 16. Câu hỏi mở (hỏi người dùng khi cần)

- Hạn nộp chính xác? (phiếu giao đề chưa ghi ngày)
- Phần cứng: GPU dev là gì? Có Jetson Orin không?
- Có quay được ở môi trường công nghiệp thật không, hay dựng kịch bản trong lab/xưởng trường?
- GVHD muốn so sánh với "phương pháp hiện có" cụ thể nào (paper nào)? Hay chỉ cần B0, B1, B4?
- Giày bảo hộ: chấp nhận scope là "có giày kín" hay bắt buộc phân biệt giày bảo hộ với giày thường?

---

## 17. Limitations (ghi sẵn để không quên đưa vào thuyết minh)

- Không ghép người giữa các camera: mỗi camera quyết định độc lập, một người có thể bị báo ở nhiều camera (alert trùng), và không bổ sung bằng chứng chéo khi một cam bị che mà cam khác thấy rõ.
- Không phân biệt được giày bảo hộ với giày thường ở khoảng cách xa (giới hạn của dataset SH17).
- Dữ liệu té ngã là té giả, có thể khác té thật.
- Phát hiện té ngã giả định cam nhìn ngang/chéo. Cam nhìn từ trên xuống thấy rõ nhưng không phân biệt được đứng với nằm (UR Fall cam1); hệ thống không hỗ trợ cam kiểu này.
- Méo ống kính chỉ ước lượng từ góc gạch trên một mặt sàn (k1, k2; f cố định): ngoài vùng có góc gạch (sát mép ảnh) toạ độ sàn là ngoại suy. cam1 (hành lang) không calib được nên không phát hiện xâm nhập/tụ tập ở cam1.
- Người bị cắt chân ở mép dưới ảnh (đứng sát cam đặt thấp) có điểm sàn sai khoảng 0.6–0.8 m (đo trên WILDTRACK), ảnh hưởng vùng cấm / vùng rủi ro.
- Offline backend (Ultralytics) không đạt real-time cho đa luồng (khoảng 16 fps tổng ở 1080p trên RTX 4060 Laptop). DeepStream đạt khoảng 80 fps tổng: real-time 25 fps tối đa 3 cam 1080p trên GPU laptop này (4 cam còn 19.8 fps/cam).
- Clip bằng chứng live (smart record) là stream gốc của camera, không có OSD (snapshot có overlay vẽ lại từ metadata, lệch tối đa khoảng 1 GOP); chỉ hoạt động với nguồn RTSP.
- Bộ tự thu có quy mô nhỏ, chưa phải môi trường công nghiệp thật (nếu đúng như vậy).
