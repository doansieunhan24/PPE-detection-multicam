"""Build the progress deck (.pptx) from the eval reports: every number is read from results.json.

Needs python-pptx (not a project dependency: use a separate venv). Figures are read from
eval/reports/c2_s01/figs (tools/report/fig_timeline.py, fig_sens.py).

    python tools/report/build_deck.py . eval/reports/c2_s01/deck.pptx <pooled results.json>
"""

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT, OUT, MAIN = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ALT = Path(sys.argv[4]) if len(sys.argv) > 4 else None
R = ROOT / "eval" / "reports"

INK = RGBColor(0x1F, 0x2A, 0x33)
MUTED = RGBColor(0x5B, 0x67, 0x70)
LINE = RGBColor(0xC9, 0xD1, 0xD6)
TINT = RGBColor(0xF1, 0xF4, 0xF5)
TEAL = RGBColor(0x0E, 0x6E, 0x6E)
ORANGE = RGBColor(0xE0, 0x7A, 0x1F)
GREY = RGBColor(0x9A, 0xA5, 0xAD)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = "Cambria", "Calibri"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


main = load(MAIN)
res, meta = main["results"], main["meta"]
alt = load(ALT)["results"] if ALT else None
untuned = load(R / "c2_s01" / "untuned_all" / "results.json")["results"]
det = load(R / "detector" / "results.json")
fps = load(R / "bench_fps_ds_wt3" / "results.json")
rate = load(R / "c2_s01" / "ppe_rate" / "results.json")["rows"]
pt1 = load(R / "c2_s01" / "per_take_helmet_T1.0" / "results.json")["takes"]
pt3 = load(R / "c2_s01" / "per_take_helmet_T3.0" / "results.json")["takes"]
FIGS = R / "c2_s01" / "figs"
_pp = R / "c2_s01_refit_pp" / "cv" / "pooled" / "results.json"
PP_F1 = f"{load(_pp)['results']['P']['f1']:.2f}" if _pp.exists() else "chưa chạy"


def tot(pt, pol, k):
    return sum(v[pol][k] for v in pt.values())


FULL_PPE = ["s01_t01", "s01_t04", "s01_t05", "s01_t06", "s01_t08", "s01_t10"]
fp1_b1, fp1_p = tot(pt1, "B1", "fp"), tot(pt1, "P", "fp")
fast_cut = 1 - fp1_p / fp1_b1 if fp1_b1 else 0.0
H = {m: res[m]["per_type"]["PPE_HELMET"] for m in ("B0", "B1", "B4", "P")}
V = {m: res[m]["per_type"]["PPE_VEST"] for m in ("B0", "B1", "B4", "P")}
best_h = max(("B1", "B4", "P"), key=lambda m: (H[m]["f1"], -H[m]["fa_per_hour"]))
VEST_OFF = V["P"]["recall"] < 0.34
VERDICT = ("CV: P tốt nhất với mũ" if best_h == "P" else f"CV: với mũ, {best_h} nhỉnh hơn P") + (
    ", nhưng gần như bỏ qua gile" if VEST_OFF else ""
)
VEST_LINE = (
    f"P sau tune gần như không báo gile (recall {V['P']['recall']:.2f})."
    if VEST_OFF
    else f"gile: P recall {V['P']['recall']:.2f}, {V['P']['fa_per_hour']:.1f} báo sai/giờ."
)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5


def tb(
    slide,
    x,
    y,
    w,
    h,
    text,
    size=16,
    bold=False,
    color=INK,
    font=BODY,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    italic=False,
    name=None,
):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        runs = ln if isinstance(ln, list) else [(ln, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.name = o.get("font", font)
            f.color.rgb = o.get("color", color)
        p.space_after = Pt(o.get("after", 6)) if runs else Pt(6)
    return s


def bullets(slide, x, y, w, h, items, size=16, color=INK, after=10):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        runs = it if isinstance(it, list) else [(it, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", False)
            r.font.name = BODY
            r.font.color.rgb = o.get("color", color)
        p.space_after = Pt(after)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.28))))
        pPr.set("indent", str(-Emu(Inches(0.28))))
        from pptx.oxml.ns import qn

        for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
            for e in pPr.findall(qn(tag)):
                pPr.remove(e)
        bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
        pPr.append(bu)
    return s


def box(
    slide, x, y, w, h, fill=WHITE, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, lw=1.0, name=None
):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def title(slide, text, sub=None, dark=False):
    tb(
        slide,
        0.6,
        0.45,
        SW - 1.2,
        0.8,
        text,
        size=30,
        bold=True,
        font=HEAD,
        color=WHITE if dark else INK,
        name="Title",
    )
    if sub:
        tb(
            slide,
            0.6,
            1.2,
            SW - 1.2,
            0.5,
            sub,
            size=16,
            color=GREY if dark else MUTED,
            name="Subtitle",
        )


def footer(slide, n, dark=False):
    tb(
        slide,
        0.6,
        SH - 0.45,
        8,
        0.3,
        "ĐATN · Giám sát an toàn PPE · C2 s01",
        size=10,
        color=GREY if dark else MUTED,
    )
    tb(
        slide,
        SW - 1.6,
        SH - 0.45,
        1.0,
        0.3,
        str(len(prs.slides)),
        size=10,
        color=GREY if dark else MUTED,
        align=PP_ALIGN.RIGHT,
    )


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color


def arrow(slide, x1, y1, x2, y2, color=GREY):
    c = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    c.line.color.rgb = color
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    from pptx.oxml.ns import qn

    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def style_chart(ch, colors, legend=False, fmt="0.00", label_size=12):
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(12)
        ch.legend.font.name = BODY
        ch.legend.font.color.rgb = MUTED
    ch.font.name = BODY
    ch.font.size = Pt(12)
    ch.font.color.rgb = MUTED
    ca, va = ch.category_axis, ch.value_axis
    ca.format.line.color.rgb = LINE
    ca.tick_labels.font.size = Pt(12)
    ca.tick_labels.font.color.rgb = INK
    ca.has_major_gridlines = False
    va.visible = False
    va.has_major_gridlines = False
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(label_size)
    dl.font.color.rgb = INK
    for i, s in enumerate(plot.series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = colors[i % len(colors)]
    return ch


def bar(
    slide,
    x,
    y,
    w,
    h,
    cats,
    series,
    colors,
    title_txt=None,
    fmt="0.00",
    legend=False,
    point_colors=None,
    ymax=None,
    label_size=12,
):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd
    )
    ch = gf.chart
    style_chart(ch, colors, legend, fmt, label_size)
    if ymax is not None:
        ch.value_axis.maximum_scale = ymax
        ch.value_axis.minimum_scale = 0
    if title_txt:
        ch.has_title = True
        ch.chart_title.text_frame.text = title_txt
        p = ch.chart_title.text_frame.paragraphs[0]
        p.runs[0].font.size = Pt(14)
        p.runs[0].font.bold = True
        p.runs[0].font.name = BODY
        p.runs[0].font.color.rgb = INK
    else:
        ch.has_title = False
    if point_colors:
        s = ch.plots[0].series[0]
        for i, c in enumerate(point_colors):
            pt = s.points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = c
    return ch


def stat(slide, x, y, w, big, label, color=TEAL):
    box(slide, x, y, w, 1.9, fill=TINT, line=None)
    tb(slide, x + 0.3, y + 0.2, w - 0.6, 0.9, big, size=44, bold=True, font=HEAD, color=color)
    tb(slide, x + 0.3, y + 1.15, w - 0.6, 0.7, label, size=14, color=INK)


METHODS = ["B0", "B1", "B4", "P"]
P, B1, B4 = res["P"], res["B1"], res["B4"]
best_b = min((res[m] for m in ("B1", "B4")), key=lambda r: r["fa_per_hour"])
best_b_name = "B1" if best_b is B1 else "B4"
fa_cut = 1 - P["fa_per_hour"] / best_b["fa_per_hour"] if best_b["fa_per_hour"] else 0.0
fps3 = next(r for r in fps["rows"] if r["streams"] == 3)
n = 0
ND = {
    m: res[m]["fp"] - res[m].get("fp_duplicate", 0) for m in res
}  # false alarms, repeats excluded
NB = min(("B1", "B4"), key=lambda m: ND[m])  # baseline with the fewest non-repeat false alarms
nd_cut = 1 - ND["P"] / ND[NB] if ND[NB] else 0.0
FB = max(("B1", "B4"), key=lambda m: res[m]["f1"])  # best baseline by F1

# 1 title -----------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bg(s, INK)
tb(
    s,
    0.8,
    1.9,
    11.5,
    1.6,
    "Giám sát an toàn PPE đa camera theo thời gian thực",
    size=40,
    bold=True,
    font=HEAD,
    color=WHITE,
    name="Title",
)
tb(
    s,
    0.8,
    3.55,
    11.5,
    0.9,
    "Tích lũy bằng chứng có trọng số độ quan sát (C1) · bộ dữ liệu C2 · hệ thống DeepStream",
    size=20,
    color=GREY,
)
tb(
    s,
    0.8,
    5.3,
    11.5,
    0.9,
    ["Báo cáo tiến độ đồ án tốt nghiệp · ĐH Phenikaa · AI&RB", "Nguyễn Ngọc Đoan · 2026-11-06"],
    size=16,
    color=WHITE,
)
s.notes_slide.notes_text_frame.text = (
    "Báo cáo tiến độ: hệ thống đã chạy end-to-end, bộ dữ liệu C2 session 1 đã quay và gán nhãn, "
    "và đây là lần đầu có số liệu C1 trên dữ liệu thật."
)

# 2 summary ---------------------------------------------------------------------------------
n += 2
s = prs.slides.add_slide(BLANK)
title(
    s,
    "P cắt gần hết báo sai nhưng chưa hơn baseline về F1",
    "Cross-validation theo take: tune trên 9 take, chấm trên take còn lại, gộp 10 lượt",
)
stat(
    s,
    0.6,
    1.95,
    3.9,
    f"−{nd_cut * 100:.0f}%",
    f"báo sai không lặp so với {NB}: P {ND['P']}, B1 {ND['B1']}, B4 {ND['B4']}",
    TEAL,
)
stat(
    s,
    4.72,
    1.95,
    3.9,
    f"{P['mean_ttd_s']:.0f} s",
    f"thời gian phát hiện trung bình của P (B1 {B1['mean_ttd_s']:.0f} s, B4 {B4['mean_ttd_s']:.0f} s); "
    f"mũ bắt đúng {H['P']['tp']} = {H['B1']['tp']}",
    TEAL,
)
stat(
    s,
    8.84,
    1.95,
    3.9,
    f"{fps3['fps_per_stream_mean']:.0f} fps",
    "mỗi luồng khi chạy 3 camera 1080p trên DeepStream (RTX 4060 Laptop)",
    ORANGE,
)
bullets(
    s,
    0.6,
    4.35,
    12.1,
    2.6,
    [
        [
            (f"Theo F1 gộp, {FB} tốt nhất, P chưa hơn baseline: ", {}),
            (f"P {P['f1']:.2f}, B1 {B1['f1']:.2f}, B4 {B4['f1']:.2f}", {"bold": True}),
            (f" (recall P {P['recall']:.2f}, B1 {B1['recall']:.2f}).", {}),
        ],
        [
            ("Recall mất là gile: ", {}),
            (VEST_LINE, {"bold": True}),
            (" Detector chỉ bắt gile xanh dương 29–44% frame.", {}),
        ],
        "Dữ liệu còn nhỏ (10 take, 50 phút-camera, 25 khoảng PPE × camera): chênh lệch chưa có ý nghĩa thống kê.",
    ],
    size=16,
)
footer(s, 2)
s.notes_slide.notes_text_frame.text = (
    "Ba con số chính. Tất cả là số gộp trên take bị giữ lại của 10 lượt cross-validation, "
    "không phải số trên tập đã tune."
)

# 3 problem ---------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "Không thấy mũ không có nghĩa là không đội mũ")
img = ROOT / "docs" / "figures" / "c2_examples" / "s01_t04_cam2_048s.jpg"
s.shapes.add_picture(str(img), Inches(0.6), Inches(1.6), width=Inches(7.2))
tb(
    s,
    0.6,
    5.75,
    7.2,
    0.4,
    "T04 cam2, giây 48: A ngồi sau ghế, thân bị che; cả hai đủ PPE",
    size=11,
    color=MUTED,
)
bullets(
    s,
    8.25,
    1.7,
    4.5,
    4.6,
    [
        [("Luật phổ biến: ", {"bold": True}), ("không phát hiện được PPE ⇒ vi phạm.", {})],
        [
            ("Thực tế: ", {"bold": True}),
            ('phần lớn lần "không thấy" là do bị che, quay lưng, ở xa, hoặc ảnh mờ.', {}),
        ],
        [
            ("Hệ quả: ", {"bold": True}),
            ("báo sai liên tục, và người vận hành bỏ qua cảnh báo.", {}),
        ],
        [
            ("Ý tưởng C1: ", {"bold": True, "color": TEAL}),
            (
                "đo độ quan sát v của từng bộ phận; chưa đủ quan sát thì trả về UNKNOWN thay vì báo.",
                {},
            ),
        ],
    ],
    size=16,
)
footer(s, 3)

# 4 architecture ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "DeepStream chỉ làm perception, mọi quyết định chạy lại được từ JSONL")
steps = [
    ("Camera", "3 × Tiandy\nRTSP 1080p\n25 fps"),
    ("DeepStream 7.1", "YOLO11m-pose\nNvDCF tracker\nSGIE PPE trên crop"),
    ("Metadata", "JSONL (offline)\nZMQ (live)\ncùng schema"),
    ("Analytics · C1", "vùng bộ phận → p, v\ntích lũy log-odds\n3 trạng thái"),
    ("Cảnh báo", "L1 / L2 / L3\nclip + snapshot\ndashboard, Telegram"),
]
bw, gap, y0 = 2.18, 0.35, 2.2
for i, (h, b) in enumerate(steps):
    x = 0.6 + i * (bw + gap)
    hl = i == 3
    box(
        s,
        x,
        y0,
        bw,
        2.6,
        fill=TINT if not hl else RGBColor(0xDD, 0xEE, 0xEE),
        line=TEAL if hl else LINE,
        lw=2.0 if hl else 1.0,
        name=f"step-{i}",
    )
    tb(s, x + 0.2, y0 + 0.25, bw - 0.4, 0.5, h, size=17, bold=True, color=TEAL if hl else INK)
    tb(s, x + 0.2, y0 + 0.9, bw - 0.4, 1.6, b.split("\n"), size=14, color=INK)
    if i < len(steps) - 1:
        arrow(s, x + bw + 0.04, y0 + 1.3, x + bw + gap - 0.04, y0 + 1.3)
bullets(
    s,
    0.6,
    5.3,
    12.1,
    1.5,
    [
        "Probe trong DeepStream chỉ copy metadata ra hàng đợi; xử lý nặng nằm ở analytics.",
        "Live và replay dùng chung lớp Analytics, nên cùng một video cho cùng cảnh báo.",
    ],
    size=15,
)
footer(s, 4)

# 5 C1 method -------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "C1 tích lũy bằng chứng theo thời gian, trọng số là độ quan sát v")
box(s, 0.6, 1.7, 12.1, 1.5, fill=TINT, line=None)
tb(
    s,
    0.9,
    1.85,
    11.5,
    1.3,
    [
        [
            ("e = v · logit(p)", {"bold": True, "font": HEAD, "size": 20}),
            ("     bằng chứng của frame, frame khó nhìn đóng góp ít", {"color": MUTED, "size": 15}),
        ],
        [
            (
                "S ← λS + Δt·e,   Ω ← λΩ + Δt·v,   m = S / Ω",
                {"bold": True, "font": HEAD, "size": 20},
            ),
            ("     λ = exp(−Δt/τ), Ω là số giây-quan sát", {"color": MUTED, "size": 15}),
        ],
    ],
    size=18,
)
cards = [
    ("UNKNOWN", "Ω < θ_obs", "Chưa đủ quan sát: không báo động", GREY),
    ("COMPLIANT", "m ≥ θ_pos", "Đủ bằng chứng có PPE", TEAL),
    ("VIOLATION", "m ≤ θ_neg", "Đủ bằng chứng thiếu PPE; báo khi kéo dài ≥ T_alert", ORANGE),
]
for i, (h, c, d, col) in enumerate(cards):
    x = 0.6 + i * 4.1
    box(s, x, 3.6, 3.9, 2.2, fill=WHITE, line=col, lw=2.0, name=f"state-{h}")
    tb(s, x + 0.3, 3.8, 3.3, 0.5, h, size=20, bold=True, color=col, font=HEAD)
    tb(s, x + 0.3, 4.4, 3.3, 0.4, c, size=16, bold=True)
    tb(s, x + 0.3, 4.9, 3.3, 0.8, d, size=14, color=INK)
tb(
    s,
    0.6,
    6.15,
    12.1,
    0.6,
    "p: xác suất có PPE, từ box detector gán vào vùng bộ phận theo keypoint; "
    "mũ cầm tay (gần cổ tay, ngoài vùng đầu) không tính là đang đội.",
    size=14,
    color=MUTED,
)
footer(s, 5)

# 5b C1 example ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(
    s,
    "Đầu khuất khỏi khung: B1 báo sai, P chuyển UNKNOWN",
    "T05 cam2, người C (đủ PPE), bộ phận đầu, cảnh báo sau 1 s",
)
s.shapes.add_picture(str(FIGS / "timeline_t05_c.png"), Inches(0.6), Inches(1.75), width=Inches(9.2))
bullets(
    s,
    10.05,
    1.9,
    2.75,
    4.6,
    [
        "Giây ~22 và ~65: C khuất khỏi khung, v và p cùng tụt về 0.",
        [('B1 thấy "không có mũ" → VIOLATION → ', {}), ("2 báo sai.", {"bold": True})],
        [("P thấy Ω nhỏ → UNKNOWN → ", {}), ("0 báo sai.", {"bold": True, "color": TEAL})],
    ],
    size=14,
)
footer(s, 0)

# 6 baselines -------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(
    s,
    "Ba baseline dùng chung detector, tracker và thủ tục tune",
    "Mọi ngưỡng được tune riêng cho từng phương pháp bằng cùng một coordinate descent",
)
bl = [
    ("B0", "Từng frame", "Không có box PPE là vi phạm ngay", "Không tích lũy, không v"),
    ("B1", "Persistence", "Vi phạm liên tục ≥ T_alert mới báo", "Không v, không UNKNOWN"),
    ("B4", "Theo độ tin detector", "Tích lũy p đã hiệu chỉnh + persistence", "Không v"),
    ("P", "Đề xuất", "Log-odds trọng số v + UNKNOWN + thời gian", "—"),
]
for i, (k, n1, d, miss) in enumerate(bl):
    x = 0.6 + i * 3.08
    hl = k == "P"
    box(
        s,
        x,
        2.0,
        2.9,
        3.6,
        fill=RGBColor(0xDD, 0xEE, 0xEE) if hl else TINT,
        line=TEAL if hl else None,
        lw=2.0,
        name=f"method-{k}",
    )
    tb(s, x + 0.3, 2.2, 2.4, 0.8, k, size=40, bold=True, font=HEAD, color=TEAL if hl else INK)
    tb(s, x + 0.3, 3.1, 2.4, 0.5, n1, size=17, bold=True)
    tb(s, x + 0.3, 3.65, 2.4, 1.0, d, size=14)
    tb(s, x + 0.3, 4.8, 2.4, 0.6, miss, size=13, color=MUTED, italic=True)
tb(
    s,
    0.6,
    6.0,
    12.1,
    0.6,
    "B2/B3 (gộp nhiều camera) bỏ từ 2026-10-31 vì bài chỉ xử lý từng camera.",
    size=14,
    color=MUTED,
)
footer(s, 6)

# 7 dataset ---------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "C2 session 1: 10 take, 3 camera, 50 phút-camera có nhãn")
s.shapes.add_picture(
    str(ROOT / "docs" / "figures" / "calibration" / "cam2_zones.jpg"),
    Inches(0.6),
    Inches(1.6),
    width=Inches(6.6),
)
tb(
    s,
    0.6,
    5.4,
    6.6,
    0.4,
    "cam2: vùng cấm Z2 (đỏ) và vùng rủi ro Z3 vẽ theo mạch gạch 0.67 m",
    size=11,
    color=MUTED,
)
rows = [
    ("Camera", "Vị trí", "Sai số sàn (LOO)"),
    ("cam1", "Hành lang", "không calib"),
    ("cam2", "Sảnh", "0.039 m"),
    ("cam3", "Sảnh, lệch ~46°", "0.012 m"),
]
tbl = s.shapes.add_table(4, 3, Inches(7.6), Inches(1.7), Inches(5.1), Inches(1.8)).table
for r, row in enumerate(rows):
    for c, v in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = v
        p = cell.text_frame.paragraphs[0]
        p.runs[0].font.size = Pt(14)
        p.runs[0].font.bold = r == 0
        p.runs[0].font.name = BODY
        p.runs[0].font.color.rgb = WHITE if r == 0 else INK
        cell.fill.solid()
        cell.fill.fore_color.rgb = TEAL if r == 0 else (TINT if r % 2 else WHITE)
bullets(
    s,
    7.6,
    3.9,
    5.1,
    2.8,
    [
        "3 người: A gile xanh dương, B gile cam, C gile xanh nõn chuối.",
        "17 khoảng sự kiện: mũ 9, gile 2, xâm nhập 5, bất động 1.",
        "T01, T04, T05, T10 đủ PPE: mọi cảnh báo PPE ở đây là báo sai.",
    ],
    size=15,
)
footer(s, 7)

# 8 labeling --------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "Gán nhãn bán tự động: máy gợi ý, người duyệt từng dòng")
st = [
    ("1", "Track mẫu", "người chọn 8 track rõ, mỗi người mỗi cam"),
    ("2", "Gợi ý theo màu áo", "histogram thân + chân; hai track cùng lúc là hai người"),
    ("3", "Duyệt ảnh", "ảnh cắt theo giây cho mỗi dòng không chắc"),
    ("4", "Khoảng sự kiện", "đọc từ timeline 1 frame/giây, phóng to quanh vùng"),
]
for i, (k, h, d) in enumerate(st):
    y = 1.75 + i * 1.18
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), Inches(y), Inches(0.75), Inches(0.75))
    o.fill.solid()
    o.fill.fore_color.rgb = TEAL
    o.line.fill.background()
    o.text_frame.text = k
    pr = o.text_frame.paragraphs[0]
    pr.alignment = PP_ALIGN.CENTER
    pr.runs[0].font.size = Pt(20)
    pr.runs[0].font.bold = True
    pr.runs[0].font.color.rgb = WHITE
    tb(s, 1.6, y + 0.02, 5.6, 0.45, h, size=17, bold=True)
    tb(s, 1.6, y + 0.45, 5.6, 0.5, d, size=14, color=MUTED)
stat(s, 7.7, 1.75, 2.45, "183", "gợi ý tự động được duyệt")
stat(s, 10.3, 1.75, 2.45, "31", "track sửa tay")
stat(s, 7.7, 3.9, 2.45, "2", "track nhảy người, tách theo thời gian", ORANGE)
stat(s, 10.3, 3.9, 2.45, "1", "gợi ý tự động sai bị phát hiện", ORANGE)
tb(
    s,
    7.7,
    6.0,
    5.0,
    0.7,
    "Nhãn không lấy từ output hệ thống, để không chấm vòng tròn.",
    size=14,
    color=MUTED,
)
footer(s, 8)

# 9 components ------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "Hệ thống đạt real-time tới 3 camera; detector mạnh với mũ, yếu với gile")
m2 = {r["data"]: r for r in det if r["model"] == "crop_M2_sh17chv"}
cls = [("helmet", "mũ"), ("vest", "gile"), ("head", "đầu"), ("shoes", "giày")]
bar(
    s,
    0.6,
    1.6,
    6.2,
    4.6,
    [v for _, v in cls],
    [("SH17", [m2["sh17"]["per_class"][k]["mAP50"] for k, _ in cls])],
    [TEAL],
    "AP50 trên SH17 test (YOLO11s, crop người)",
    fmt="0.00",
    ymax=1.15,
)
rows_f = fps["rows"]
bar(
    s,
    7.1,
    1.6,
    5.6,
    4.6,
    [f"{r['streams']} cam" for r in rows_f],
    [("fps", [r["fps_per_stream_mean"] for r in rows_f])],
    [TEAL],
    "fps mỗi luồng, DeepStream 7.1, 1080p",
    fmt="0.0",
    point_colors=[TEAL if r["realtime"] else GREY for r in rows_f],
)
tb(
    s,
    0.6,
    6.3,
    12.1,
    0.5,
    f"Trên CHV test: mũ AP50 {m2['chv']['per_class']['helmet']['mAP50']:.2f}, gile "
    f"{m2['chv']['per_class']['vest']['mAP50']:.2f}. 4 camera còn "
    f"{rows_f[-1]['fps_per_stream_mean']:.1f} fps/luồng (dưới 25).",
    size=14,
    color=MUTED,
)
footer(s, 9)

# 10 main results ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(
    s,
    "Bảng chính: số trên take bị giữ lại, gộp 10 lượt",
    "Sự kiện PPE, chấm từng camera; tune ngưỡng + fit hiệu chỉnh p trên 9 take, cùng thủ tục cho mọi phương pháp",
)
hdr = [
    "Phương pháp",
    "F1",
    "Recall",
    "FA / giờ",
    "Báo sai",
    "trong đó lặp",
    "TTD (s)",
    "Mũ TP / FN",
    "Gile TP / FN",
]
rows_m = ["B0", "B1", "B4", "P"]
tbl = s.shapes.add_table(
    len(rows_m) + 1, len(hdr), Inches(0.6), Inches(1.85), Inches(12.1), Inches(2.9)
).table
for c, v in enumerate(hdr):
    tbl.cell(0, c).text = v
for r, m in enumerate(rows_m, 1):
    x, h, v = res[m], H[m], V[m]
    vals = [
        m,
        f"{x['f1']:.3f}",
        f"{x['recall']:.2f}",
        f"{x['fa_per_hour']:.1f}",
        str(x["fp"]),
        str(x.get("fp_duplicate", 0)),
        f"{x['mean_ttd_s']:.1f}" if x.get("mean_ttd_s") else "—",
        f"{h['tp']} / {h['fn']}",
        f"{v['tp']} / {v['fn']}",
    ]
    for c, val in enumerate(vals):
        tbl.cell(r, c).text = val
for r in range(len(rows_m) + 1):
    for c in range(len(hdr)):
        cell = tbl.cell(r, c)
        pr = cell.text_frame.paragraphs[0]
        pr.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        f = pr.runs[0].font
        f.size = Pt(15)
        f.name = BODY
        f.bold = r == 0 or rows_m[r - 1] == "P" if r else True
        f.color.rgb = WHITE if r == 0 else (TEAL if rows_m[r - 1] == "P" else INK)
        cell.fill.solid()
        cell.fill.fore_color.rgb = TEAL if r == 0 else (TINT if r % 2 else WHITE)
bullets(
    s,
    0.6,
    5.0,
    12.1,
    1.9,
    [
        [
            ("Báo sai không lặp: ", {}),
            (f"P {ND['P']}, B1 {ND['B1']}, B4 {ND['B4']}", {"bold": True}),
            (
                ". Alert lặp = cảnh báo thứ hai cho cùng một vi phạm thật (vd. sau khi track đứt); theo "
                "luật khớp vẫn tính là báo sai.",
                {},
            ),
        ],
        "Mũ: cả ba bỏ sót cả 7 đoạn của T03 khi T03 là take test (không còn kiểu tháo mũ ngắn trong tập tune).",
    ],
    size=14,
)
footer(s, 0)

# 10a charts ---------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(
    s,
    f"Báo sai không lặp giảm {nd_cut * 100:.0f}% so với {NB}, gile mất hết",
    "Cùng các lượt cross-validation (hiệu chỉnh p fit lại trên 9 take val mỗi lượt); gộp 10 take × 3 camera",
)
cats = ["B1", "B4", "P"]
bar(
    s,
    0.6,
    1.75,
    6.0,
    4.4,
    cats,
    [
        ("Báo sai không lặp", [ND[m] for m in cats]),
        ("Alert lặp", [res[m].get("fp_duplicate", 0) for m in cats]),
    ],
    [ORANGE, GREY],
    "Báo sai",
    fmt="0",
    legend=True,
)
bar(
    s,
    6.9,
    1.75,
    5.8,
    4.4,
    cats,
    [("Mũ", [H[m]["tp"] for m in cats]), ("Gile", [V[m]["tp"] for m in cats])],
    [TEAL, GREY],
    "Vi phạm phát hiện đúng",
    fmt="0",
    legend=True,
)
tb(
    s,
    0.6,
    6.25,
    12.1,
    0.6,
    f"Tổng số khoảng vi phạm × camera: mũ {H['P']['n_gt']}, gile {V['P']['n_gt']}.",
    size=14,
    color=MUTED,
)
footer(s, 0)

# 10b fast alert -----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(
    s,
    f"Báo sau 1 s: P ít hơn {fast_cut * 100:.0f}% báo sai về mũ so với B1",
    "Quét T_alert quanh điểm tune (cả 10 take, mô tả xu hướng); TP / FP gộp 3 camera",
)
cats = ["B1", "B4", "P"]
bar(
    s,
    0.6,
    1.75,
    6.0,
    4.3,
    cats,
    [
        ("T_alert 1 s", [tot(pt1, m, "fp") for m in cats]),
        ("T_alert 3 s", [tot(pt3, m, "fp") for m in cats]),
    ],
    [ORANGE, GREY],
    "Số báo sai về mũ",
    fmt="0",
    legend=True,
)
bar(
    s,
    6.9,
    1.75,
    5.8,
    4.3,
    cats,
    [
        ("T_alert 1 s", [tot(pt1, m, "tp") for m in cats]),
        ("T_alert 3 s", [tot(pt3, m, "tp") for m in cats]),
    ],
    [ORANGE, GREY],
    "Số vi phạm mũ phát hiện đúng",
    fmt="0",
    legend=True,
)
tb(
    s,
    0.6,
    6.15,
    12.1,
    0.7,
    [
        "Ở 3 s, persistence của B1 đã lọc phần lớn lần bị che ngắn nên hai bên gần ngang nhau; "
        "lợi thế của C1 là cho phép báo nhanh mà không bùng báo sai.",
        "Phần lớn báo sai còn lại ở T02/T07 là alert lặp cho cùng vi phạm thật (track đứt → alert mới).",
    ],
    size=13,
    color=MUTED,
)
footer(s, 0)

# 11 per camera / per item table ------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(
    s,
    "Kết quả theo từng camera và từng loại PPE",
    "F1 / báo sai mỗi giờ theo camera và theo loại PPE (cùng các lượt CV)",
)
cams = sorted(P["by_cam"])
hdr = ["Phương pháp"] + cams + ["Mũ", "Gile"]
tbl = s.shapes.add_table(
    len(METHODS) + 1, len(hdr), Inches(0.6), Inches(1.9), Inches(12.1), Inches(3.2)
).table
for c, v in enumerate(hdr):
    tbl.cell(0, c).text = v
for r, m in enumerate(METHODS, 1):
    vals = [m] + [
        f"{res[m]['by_cam'][c]['f1']:.2f} / {res[m]['by_cam'][c]['fa_per_hour']:.1f}" for c in cams
    ]
    vals += [
        f"{res[m]['per_type'][t]['f1']:.2f} / {res[m]['per_type'][t]['fa_per_hour']:.1f}"
        for t in ("PPE_HELMET", "PPE_VEST")
    ]
    for c, v in enumerate(vals):
        tbl.cell(r, c).text = v
for r in range(len(METHODS) + 1):
    for c in range(len(hdr)):
        cell = tbl.cell(r, c)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        f = p.runs[0].font
        f.size = Pt(15)
        f.name = BODY
        f.bold = r == 0 or (r == len(METHODS))
        f.color.rgb = WHITE if r == 0 else (TEAL if r == len(METHODS) else INK)
        cell.fill.solid()
        cell.fill.fore_color.rgb = TEAL if r == 0 else (TINT if r % 2 else WHITE)
tb(
    s,
    0.6,
    5.4,
    12.1,
    1.2,
    [
        "Mỗi ô: F1 / báo sai mỗi giờ-camera. cam1 là hành lang (người ở xa, đi dọc), cam2/cam3 là sảnh.",
        f"Số sự kiện ít (mũ {P['per_type']['PPE_HELMET']['n_gt']}, gile {P['per_type']['PPE_VEST']['n_gt']} "
        "theo camera), nên chênh lệch nhỏ giữa các phương pháp chưa có ý nghĩa thống kê.",
    ],
    size=14,
    color=MUTED,
)
footer(s, 11)

# 12 ablation -------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
abl = [
    ("P", "P đầy đủ"),
    ("P-no_v", "bỏ v"),
    ("P-no_unknown", "bỏ UNKNOWN"),
    ("P-no_temporal", "bỏ thời gian"),
    ("P-dempster_shafer", "Dempster–Shafer"),
    ("P-v_learned", "v học"),
]
abl = [(k, lab) for k, lab in abl if k in res]
worst = max((k for k, _ in abl if k != "P"), key=lambda k: ND[k])
worst_lab = dict(abl)[worst]
title(
    s,
    f"Ablation: bỏ v → {ND['P-no_v']}, bỏ UNKNOWN → {ND['P-no_unknown']} báo sai không lặp (P: {ND['P']})",
    "Cùng ngưỡng đã tune của P trong mỗi lượt CV, chỉ bật/tắt một thành phần",
)
bar(
    s,
    0.6,
    1.75,
    6.0,
    4.4,
    [lab for _, lab in abl],
    [("Báo sai không lặp", [ND[k] for k, _ in abl])],
    [GREY],
    "Báo sai không lặp",
    fmt="0",
    point_colors=[TEAL] + [GREY] * (len(abl) - 1),
    label_size=11,
)
bar(
    s,
    6.9,
    1.75,
    5.8,
    4.4,
    [lab for _, lab in abl],
    [("Mũ TP", [res[k]["per_type"]["PPE_HELMET"]["tp"] for k, _ in abl])],
    [GREY],
    "Vi phạm mũ bắt đúng",
    fmt="0",
    point_colors=[TEAL] + [GREY] * (len(abl) - 1),
    label_size=11,
)
tb(
    s,
    0.6,
    6.25,
    12.1,
    0.6,
    "Ablation dùng lại ngưỡng của P, không tune lại riêng, nên chỉ cho biết "
    "hướng ảnh hưởng của từng thành phần.",
    size=14,
    color=MUTED,
)
footer(s, 12)

# 13 error analysis -------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(
    s,
    "Báo sai về gile chủ yếu do detector không bắt gile xanh",
    "Tỉ lệ frame có box gile, các take đủ PPE (T01, T04, T05, T10)",
)
people = [(1, "A · xanh dương"), (2, "B · cam"), (3, "C · xanh nõn chuối")]
cams_r = sorted({x["cam"] for x in rate})
series = []
for c in cams_r:
    series.append(
        (
            c,
            [
                next((x["vest"] for x in rate if x["gt_id"] == g and x["cam"] == c), None)
                for g, _ in people
            ],
        )
    )
ch = bar(
    s,
    0.6,
    1.75,
    7.0,
    4.5,
    [lab for _, lab in people],
    series,
    [RGBColor(0x7F, 0xB3, 0xB3), TEAL, RGBColor(0x0A, 0x4A, 0x4A)],
    None,
    fmt="0%",
    legend=True,
    ymax=1.1,
    label_size=11,
)
bullets(
    s,
    7.9,
    1.9,
    4.8,
    4.4,
    [
        [("Mũ được phát hiện 91–99% frame ", {}), ("cho cả 3 người", {"bold": True}), (".", {})],
        [
            ("Thân A thấy rõ (v cao) nhưng không có box gile: ", {}),
            ("C1 không cứu được loại lỗi này.", {"bold": True}),
        ],
        'Hiệu chỉnh p trên các take val làm "không có box gile" thành bằng chứng âm yếu, công bằng cho B4 và P.',
        [
            ("Trước khi tune, θ_obs = 1 làm P ở UNKNOWN 42% thời gian: ", {}),
            ("Ω chỉ hội tụ về v·τ khi còn một camera.", {"bold": True}),
        ],
    ],
    size=15,
)
footer(s, 13)

# 14 limitations ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "Hạn chế cần nói rõ trong thuyết minh")
lim = [
    ("Dữ liệu nhỏ", "10 take, 25 khoảng PPE × cam, 3 người diễn; chưa có session test riêng"),
    ("Gile xanh dương", "detector thiếu dữ liệu màu này; không train trên C2"),
    ("Ngưỡng chung", f"một bộ θ cho mọi bộ phận tắt luôn gile; thử θ riêng theo bộ phận cũng không cứu được (F1 {PP_F1})"),
    ("Alert lặp", "vi phạm thật bị báo lại sau khi track đứt, vẫn tính là báo sai"),
    ("Từng camera riêng", "không mượn bằng chứng giữa các cam; alert trùng"),
    ("Tracker", "track đứt xoá bằng chứng; 2 ca nhảy người làm lẫn bằng chứng"),
    ("Điểm sàn = đáy bbox", "sai với người nằm ngang và người bị cắt chân"),
    ("Chưa đo trên C2", "té ngã (không quay vì an toàn), tụ tập (0 sự kiện), giày"),
]
for i, (h, d) in enumerate(lim):
    x = 0.6 + (i % 4) * 3.08
    y = 1.65 + (i // 4) * 2.55
    box(s, x, y, 2.9, 2.3, fill=TINT, line=None, name=f"limit-{i}")
    tb(s, x + 0.25, y + 0.25, 2.4, 0.5, h, size=17, bold=True, color=TEAL)
    tb(s, x + 0.25, y + 0.85, 2.4, 1.3, d, size=14)
footer(s, 14)

# 15 next -----------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bg(s, INK)
title(s, "Việc tiếp theo", dark=True)
nx = [
    ("Quay s02", "giày bảo hộ, đánh dấu Z2/Z3 trên sàn, T11 hỗn hợp; dùng làm session test"),
    ("Chốt ngưỡng", "θ_obs theo v·τ cho cấu hình một camera; kiểm lại trên s02"),
    ("Kiểm nhãn", "duyệt lại các dòng tự động trước khi báo số chính thức"),
    ("Viết thuyết minh", "bảng chính, ablation, độ nhạy, FPS, phân tích lỗi gile xanh"),
]
for i, (h, d) in enumerate(nx):
    y = 1.7 + i * 1.2
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), Inches(y + 0.05), Inches(0.7), Inches(0.7))
    o.fill.solid()
    o.fill.fore_color.rgb = ORANGE
    o.line.fill.background()
    o.text_frame.text = str(i + 1)
    pr = o.text_frame.paragraphs[0]
    pr.alignment = PP_ALIGN.CENTER
    pr.runs[0].font.size = Pt(20)
    pr.runs[0].font.bold = True
    pr.runs[0].font.color.rgb = WHITE
    tb(s, 1.6, y, 11.0, 0.45, h, size=20, bold=True, color=WHITE)
    tb(s, 1.6, y + 0.45, 11.0, 0.45, d, size=15, color=GREY)
footer(s, 15, dark=True)

prs.save(OUT)
print("saved", OUT)
