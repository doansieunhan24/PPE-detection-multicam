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

### C1 — Visibility-Aware Multi-View PPE Evidence Fusion (đóng góp chính)

**Vấn đề:** Hệ thống PPE truyền thống làm theo luật *"không detect được mũ ⇒ không đội mũ"*. Nhưng thường là do **camera không nhìn thấy** đầu: bị che, quay lưng, ở quá xa, bị mờ. Đây là nguồn **false alarm** lớn nhất khi triển khai thật.

**Đề xuất:**
- Mỗi camera, với mỗi người và mỗi bộ phận (head / torso / feet), xuất 2 giá trị:
  - `p` = xác suất có PPE trên bộ phận đó
  - `v` = **độ quan sát được** (visibility) của bộ phận đó trong camera đó
- Fuse bằng chứng giữa các camera có trọng số `v`, tích lũy theo thời gian.
- Trạng thái có **3 mức**: `COMPLIANT` / `VIOLATION` / **`UNKNOWN`**. Nếu chưa đủ bằng chứng quan sát thì trả về `UNKNOWN` và **không báo động**.
- Gán PPE vào bộ phận bằng pose: mũ phải nằm ở vùng đầu. Mũ cầm trên tay (gần cổ tay) ⇒ **không** tính là đang đội.

**Claim mục tiêu:** giảm X% false alarm/giờ so với baseline mà recall vi phạm không giảm đáng kể.

### C2 — Bộ dữ liệu đánh giá PPE đa góc nhìn (tự thu)

Các dataset PPE public (SH17, SHWD, CHV) đều là ảnh đơn góc nên không đánh giá được fusion đa camera. Bộ tự thu gồm 3 camera đồng bộ, có kịch bản che khuất, quay lưng, cầm mũ, té ngã giả. Nhãn là **khoảng thời gian trạng thái theo từng người** (interval labels).

### C3 — Hệ thống hoàn chỉnh real-time trên DeepStream (đóng góp kỹ thuật)

Gồm PPE, vùng cấm, té ngã/bất động, tụ tập và alert phân cấp. Phần này **không claim novelty**, chỉ cần chạy ổn định và có benchmark.

---

## 3. Quyết định kiến trúc đã chốt

| # | Quyết định | Lý do |
|---|---|---|
| D1 | **Giữ tracking trong từng camera** (`nvtracker`, NvDCF hoặc IOU) | Cần track ID để tích lũy theo thời gian, có 1 alert/người, chuỗi skeleton cho té ngã. Bỏ tracking = quay về detect từng frame. |
| D2 | **KHÔNG dùng appearance Re-ID cross-cam** | Công nhân mặc đồng phục giống nhau làm Re-ID hỏng. Tốn thêm model + công train. |
| D3 | Ghép người giữa các camera bằng **Ground-Plane Association** (homography → bản đồ mặt sàn) | Không cần train, tận dụng homography đã dùng cho vùng cấm. Giả định: mặt sàn phẳng, các cam nhìn chung vùng sàn. |
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
│ association: ground-plane matching giữa các cam → global_id                  │
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

- Foot point: nếu cả 2 ankle có `c > 0.5` thì lấy trung điểm 2 ankle, ngược lại lấy bottom-center của bbox.
- Nếu bbox chạm mép dưới ảnh (`truncated_bottom`) thì foot point không tin được: hạ trọng số association, đánh dấu `floor_conf` thấp.
- `H` (3×3) map ảnh → sàn (mét), tính bằng `cv2.findHomography` từ ≥ 4 cặp điểm (khuyến nghị 6–10 điểm, RANSAC). Lưu reprojection error vào YAML. **Báo cáo sai số này trong thuyết minh.**
- Mỗi cam có `fov_floor_polygon`: vùng sàn mà cam nhìn thấy. Chỉ ghép cặp người khi điểm nằm trong vùng giao FOV.

### 7.2 Ground-plane association (`association/`)

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
- Không có box nào thì `p_part = p_neg` (mặc định 0.05). Giá trị này phải calibrate trên tập val.
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

Với người `g`, part `k`, tại thời điểm `t`:

```
e_c   = v_c · logit(clip(p_c, 0.01, 0.99))          # bằng chứng từ cam c
E_t   = Σ_c e_c                                     # tổng bằng chứng
W_t   = Σ_c v_c                                     # tổng độ quan sát
λ     = exp(−Δt / τ)                                # τ mặc định 2 s
S_t   = λ·S_{t−1} + E_t
Ω_t   = λ·Ω_{t−1} + W_t
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
| B0 | Single-cam, per-frame: không detect được PPE thì vi phạm ngay (cách làm phổ biến) |
| B1 | Single-cam + persistence: vi phạm liên tục ≥ T_alert mới báo |
| B2 | Multi-cam OR: người được coi là tuân thủ nếu bất kỳ cam nào thấy PPE |
| B3 | Multi-cam majority vote theo frame + persistence |
| B4 | Multi-cam weighted bằng **detector conf** (không có v) + persistence |
| **P** | **Proposed:** v-weighted log-odds + UNKNOWN + temporal |

### 7.7 Behavior (`behavior/`)

- **Fall:**
  - Baseline rule: tỉ lệ khung hình w/h của bbox, góc thân (vector vai→hông so với phương thẳng đứng), vận tốc dọc của hông.
  - Learned: chuỗi skeleton 30–60 frame (chuẩn hoá theo bbox) → ST-GCN (hoặc 1D-CNN/LSTM nhẹ).
  - **Multi-view:** fall score fuse theo `v` của toàn thân (cùng tinh thần C1). Báo L3 khi score fuse vượt ngưỡng.
- **Immobility:** dịch chuyển keypoint trung bình < ε trong ≥ `T_immobile` **và** tư thế nằm/ngồi bệt **và** nằm trong vùng rủi ro, hoặc ngay sau một sự kiện fall. Phân biệt với người đứng yên làm việc bình thường.
- **Intrusion:** `floor_xy` nằm trong polygon vùng cấm ≥ `T_intr` (mặc định 1 s). Dùng toạ độ sàn đã fuse, không dùng ROI ảnh, để không bị lệch phối cảnh.
- **Crowd:** số `global_id` trong zone rủi ro > K trong ≥ T giây. Đếm theo global_id để không đếm trùng giữa các cam.

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
- **RTSP simulation:** `mediamtx` + `ffmpeg -re -stream_loop -1 -i camX.mp4 -c copy -f rtsp rtsp://localhost:8554/camX` (script ở `tools/rtsp_sim.sh`).
- **Đo FPS:** dùng `GST_DEBUG` hoặc probe đếm frame/giây theo từng source. Đo latency = `ts output alert − ts frame capture`.

---

## 9. Dữ liệu

| Mục đích | Dataset | Ghi chú |
|---|---|---|
| Train PPE detector | **SH17** (17 class: person, head, helmet, safety-vest, shoes, foot, gloves, ...) | Tạo **tập crop người** (`prepare_sh17_person_crops.py`) cho khớp với phân phối input của SGIE. Lưu ý: class "shoes" **không phân biệt giày bảo hộ với giày thường**, phải ghi rõ ở phần Limitations. |
| Bổ sung helmet | SHWD, CHV | Merge class cẩn thận (map tên class). |
| **Đã có (2026-10-21)** | SH17: 8,099 ảnh / 75,994 instance (khớp README), chỉ 927 helmet + 530 safety-vest. CHV: 1,330 ảnh, person/vest/4 màu mũ (3,887 person, 1,784 vest, 3,538 helmet), không nhãn head/shoes/foot → pseudo-label bằng model SH17 (chỉ train/val) | Chi tiết số crop/instance: `data/processed/*/stats.json`, `docs/experiments.md` |
| Fall (đơn góc) | Le2i, UR Fall, CAUCAFall | Train/eval module fall. |
| Fall (đa góc) | Multiple Cameras Fall Dataset (Univ. Montréal, 8 cam) | Đánh giá fusion fall đa camera. |
| **Đánh giá C1** | **Bộ tự thu (C2)** | Chỉ dùng để **eval** (hoặc chia theo session), **không train detector trên bộ này**. |

> **Kiểm tra license và link chính thức của từng dataset trước khi dùng, ghi vào `docs/experiments.md`.** Không tự bịa số lượng ảnh hay class. Đọc từ file của dataset.

### 9.1 Kế hoạch thu bộ tự thu (C2)

- 3 camera (IP cam hoặc điện thoại chạy app RTSP), đặt 3 góc nhìn chung một vùng sàn khoảng 6×8 m. Đặt marker trên sàn để calibrate homography.
- Đồng bộ: vỗ tay hoặc bật đèn flash ở đầu mỗi session, rồi chạy `tools/sync_offsets.py`.
- **Kịch bản** (mỗi cái 30–90 s, lặp với 2–5 người):
  1. Đủ PPE, đi lại bình thường
  2. Không đội mũ
  3. **Cầm mũ trên tay**
  4. **Quay lưng hoặc bị che bởi vật cản** (đủ PPE), kịch bản quan trọng nhất cho C1
  5. Đứng xa cam (người nhỏ)
  6. Cởi áo phản quang giữa chừng
  7. Đi vào vùng cấm
  8. Té ngã giả (có đệm), nằm bất động
  9. Tụ tập ≥ K người
- **Nhãn:** interval `(person_gt_id, item, state, t_start, t_end)` qua `tools/label_intervals.py`. Kèm GT person ID cho mỗi `(cam, track)` để chấm association. Nếu có thể, thêm bbox keyframe trên CVAT để đo detector.
- Tổng mục tiêu: khoảng 15–30 phút mỗi cam.

---

## 10. Đánh giá

### 10.1 Metric

| Mức | Metric |
|---|---|
| Detector | mAP50, mAP50-95 theo class (riêng helmet/vest/shoes), trên SH17 test |
| Association | Tỷ lệ ghép đúng cross-cam (pairwise precision/recall), IDF1 global (nếu có GT) |
| **PPE compliance (C1)** | Event P/R/F1, **False Alarms / giờ**, time-to-detect, **% thời gian UNKNOWN** (coverage) |
| Behavior | Event P/R/F1 cho fall/immobility/intrusion/crowd, FA/giờ |
| Hệ thống | FPS theo từng stream, số stream tối đa @ real-time, latency end-to-end, GPU util/mem |

### 10.2 Event matching (`eval/metrics.py`)

- Alert dự đoán khớp với GT interval nếu cùng người (qua mapping global_id ↔ gt_id), cùng type, và `t_raise ∈ [t_start − δ, t_end + δ]` (δ mặc định 2 s).
- Mỗi GT interval khớp tối đa 1 alert. Alert không khớp với GT nào là **false alarm**.
- FA/giờ = số false alarm / tổng thời lượng video (giờ).

### 10.3 Thí nghiệm bắt buộc

1. B0–B4 vs P trên bộ tự thu, bảng chính của luận văn.
2. Ablation của P: bỏ `v`, bỏ UNKNOWN, bỏ temporal, `v` thủ công vs `v_learned`, log-odds vs Dempster–Shafer.
3. Số camera: 1 / 2 / 3 cam, FA/giờ và recall thay đổi thế nào.
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

**Trạng thái hiện tại:** đã có khung repo (schemas, config loader, JSONL I/O, configs mặc định, test). Tiếp theo: P0 (Docker DeepStream) song song P1 offline backend.

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
| — | DeepStream version: `TODO` | |
| 2026-10-21 | GPU dev: RTX 4060 Laptop 8 GB, Windows 11 + WSL2 Ubuntu-22.04 | Máy hiện có. DeepStream phải chạy qua WSL2/Docker hoặc máy Linux — TODO(verify) hỗ trợ WSL2 |
| 2026-10-21 | Thư mục repo thực tế là `ppe_de/` (package name `ppe-multicam`) | Tên trong mục 5 chỉ là tên package |
| 2026-10-21 | **§7.5 chuẩn hoá theo Δt** (`dt_weighting`, mặc định bật): `S_t = λS + Δt·E`, `Ω_t = λΩ + Δt·W` → Ω hội tụ ≈ W·τ (visibility-giây), không phụ thuộc FPS. **Chờ người dùng xác nhận** | Công thức gốc cộng theo frame ⇒ Ω ≈ W·fps·τ: ở 25 fps v=0.05 đã vượt θ_obs=1 → UNKNOWN gần như không bao giờ xảy ra. Có test minh hoạ (`tests/test_fusion.py`). Có thể tắt cờ để so sánh |
| 2026-10-21 | Schema thêm `PersonDet.ppe_checked` | Phân biệt "detector không chạy" (không có bằng chứng) với "chạy mà không thấy PPE" (p_neg) |
| 2026-10-21 | Pose PGIE offline: `yolo11m-pose`, imgsz 960 | Trên vtest.avi (người ~70 px) imgsz 640 bỏ sót gần hết; 960 tăng ~3.7× số người phát hiện |
| 2026-10-21 | Hiệu chỉnh p theo prior trung tính (`configs/p_calibration.yaml`, fit SH17 val) thay `p_neg=0.05` + conf thô; `theta_neg` −1.0 → −0.5; baseline B0–B3 dùng cờ "có box" (không dùng p đã hiệu chỉnh) | Đo trên SH17: không-box chỉ ≈ −1 log-LR, có box ≈ +3; p_neg 0.05 phóng đại bằng chứng âm. Tune lại trên C2 |
| 2026-10-21 | Bổ sung CHV cho helmet/vest; pseudo-label head/shoes/foot | SH17 quá ít helmet/vest |
| — | Số cam / loại cam / có Jetson Orin không: `TODO` | |

---

## 16. Câu hỏi mở (hỏi người dùng khi cần)

- Hạn nộp chính xác? (phiếu giao đề chưa ghi ngày)
- Phần cứng: GPU dev là gì? Có Jetson Orin không?
- Có quay được ở môi trường công nghiệp thật không, hay dựng kịch bản trong lab/xưởng trường?
- GVHD muốn so sánh với "phương pháp hiện có" cụ thể nào (paper nào)? Hay chỉ cần B0–B4?
- Giày bảo hộ: chấp nhận scope là "có giày kín" hay bắt buộc phân biệt giày bảo hộ với giày thường?

---

## 17. Limitations (ghi sẵn để không quên đưa vào thuyết minh)

- Ground-plane association giả định mặt sàn phẳng và các cam có vùng nhìn chung. Sai khi người bị che chân hoặc đứng trên bậc, giàn.
- Không phân biệt được giày bảo hộ với giày thường ở khoảng cách xa (giới hạn của dataset SH17).
- Dữ liệu té ngã là té giả, có thể khác té thật.
- Bộ tự thu có quy mô nhỏ, chưa phải môi trường công nghiệp thật (nếu đúng như vậy).
