from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent
BURGUNDY = RGBColor(122, 32, 44)
INK = RGBColor(30, 25, 28)
MUTED = RGBColor(105, 96, 101)
ROSE = RGBColor(247, 241, 239)
PALE = RGBColor(250, 245, 246)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(34, 113, 84)
LINE = RGBColor(222, 192, 184)
SANS = "Source Han Sans SC"
SERIF = "Source Han Serif SC"
COMMIT = "398e9fc88129cde9c06e1f8cb07f6e4c3785711e"

prs = Presentation()
prs.slide_width = Inches(13.333333)
prs.slide_height = Inches(7.5)


def text(slide, x, y, w, h, value, size, color=INK, bold=False, font="Arial", align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]
    p.alignment = align
    p.space_before = p.space_after = Pt(0)
    run = p.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def rect(slide, x, y, w, h, fill, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    return shape


def new_slide(number, section, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    text(slide, 0.78, 0.29, 11.8, 0.28, section, 10.5, BURGUNDY, True)
    text(slide, 0.78, 0.70, 11.9, 0.56, title, 26, INK, True, SERIF)
    text(slide, 0.78, 1.31, 11.9, 0.33, subtitle, 11.5, MUTED, font=SANS)
    text(slide, 11.96, 6.99, 0.6, 0.20, str(number), 10, BURGUNDY, True, align=PP_ALIGN.RIGHT)
    return slide


def table(slide, x, y, widths, row_height, headers, rows, font_size=10.5, aligns=None, highlight=None):
    shape = slide.shapes.add_table(
        len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_height * (len(rows) + 1))
    )
    tbl = shape.table
    for index, width in enumerate(widths):
        tbl.columns[index].width = Inches(width)
    for r in range(len(rows) + 1):
        tbl.rows[r].height = Inches(row_height)
        values = headers if r == 0 else rows[r - 1]
        for c, value in enumerate(values):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BURGUNDY if r == 0 else (ROSE if r % 2 == 0 else WHITE)
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.015)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = (aligns[c] if aligns else (PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER))
            p.space_before = p.space_after = Pt(0)
            p.font.name = SANS if c == 0 else "Arial"
            p.font.size = Pt(font_size)
            p.font.bold = r == 0 or (highlight is not None and r == highlight)
            p.font.color.rgb = WHITE if r == 0 else (BURGUNDY if highlight is not None and r == highlight else INK)
    return tbl


def footer(slide, artifact):
    line = rect(slide, 0.78, 6.48, 11.78, 0.012, RGBColor(212, 165, 154))
    line.line.fill.background()
    text(slide, 0.78, 6.57, 11.5, 0.17, f"source commit {COMMIT}", 7.8, MUTED)
    text(slide, 0.78, 6.77, 11.5, 0.17, artifact, 7.8, MUTED)


slide = new_slide(
    20,
    "20 · FOLDNAR PHASE 0A / FROZEN MODEL-DEVELOPMENT PANELS",
    "冻结开发面板：多环境计数与泄漏边界",
    "42,564 条 TR1c 原始记录 · split 在任何这些 ID 的训练前冻结 · 外部测试集不参与开发",
)
table(
    slide, 0.78, 1.82, (2.52, 1.48, 1.25, 1.84, 1.39, 1.30), 0.36,
    ("Panel", "train seq", "dev seq", "dev groups", "dev pairs", "route strata"),
    (
        ("80% cluster", "33,421", "9,143", "3,104 clusters", "247,013", "6"),
        ("Rfam family", "19,519", "4,846", "346 families", "114,673", "22"),
        ("Legacy VL0", "control only", "196", "23 overlap clusters", "4,618", "0"),
    ),
    font_size=10.2,
)
text(slide, 2.05, 3.42, 4.9, 0.25, "多环境分层 · 序列 / pairs（环境可重叠）", 12.5, BURGUNDY, True, SANS)
table(
    slide, 0.78, 3.75, (2.48, 2.20, 2.20), 0.34,
    ("Environment", "cluster dev", "family dev"),
    (
        ("length 64–127", "5,534 / 118,302", "2,282 / 46,129"),
        ("topology: multiloop", "3,821 / 148,355", "1,633 / 54,470"),
        ("pair span 8–15", "7,940 / 56,120", "3,920 / 23,234"),
        ("isolated pair", "4,441 / 10,206", "2,450 / 5,549"),
        ("loop: hairpin", "9,114 / 22,343", "4,840 / 10,576"),
    ),
    font_size=9.8,
)
rect(slide, 7.93, 3.43, 4.63, 2.46, PALE, RGBColor(205, 161, 170))
text(slide, 8.16, 3.57, 4.17, 0.26, "泄漏解释边界", 13.5, BURGUNDY, True, SANS)
text(
    slide, 8.16, 3.93, 4.17, 1.70,
    "• 仅指定 exact / 80%-cluster / Rfam 检查未检出；不等于 clean。\n"
    "• 更广 homology、结构相似、预训练暴露、时间/标签来源仍 UNCHECKED。\n"
    "• family panel 排除 18,199 条：18,172 未映射/歧义 + 27 train-cluster overlap。\n"
    "• VL0 仅 legacy control，不可独立选路。",
    10.5, INK, font=SANS,
)
text(slide, 0.78, 5.99, 6.9, 0.25, "route strata 同时通过样本量与家族集中度门；不是性能结果。", 9.5, MUTED, font=SANS)
footer(slide, "artifact /teams/HoulabData_1788502323/liucunyu/foldnar/model_dev_v1/independent_verification.json")

slide = new_slide(
    21,
    "21 · FOLDNAR PHASE 0C / RAW BASELINE RECOVERY",
    "bpRNA-new 同序列基线：仅 raw 非去污口径",
    "5,388 / 5,388 coverage · frozen project GT · strict exact-pair · zero-based inclusive",
)
table(
    slide, 0.78, 1.85, (1.70, 0.93, 1.12, 1.02, 1.02, 1.16, 1.16, 1.20, 1.47), 0.48,
    ("System", "coverage", "TP", "FP", "FN", "micro P", "micro R", "micro F1", "macro F1"),
    (
        ("MXfold2", "100%", "100,227", "64,290", "34,957", "0.60922", "0.74141", "0.66885", "0.68195"),
        ("EternaFold", "100%", "104,470", "65,197", "30,714", "0.61574", "0.77280", "0.68538", "0.69970"),
    ),
    font_size=9.8,
    highlight=2,
)
rect(slide, 0.78, 3.57, 5.69, 1.28, ROSE, LINE)
text(slide, 1.00, 3.70, 5.25, 0.25, "MXfold2 · official v0.1.2 / TrainSetAB", 12.5, BURGUNDY, True)
text(
    slide, 1.00, 4.02, 5.25, 0.62,
    "官方 wheel、权重与冻结预测元数据哈希一致；6 条 CPU deterministic replay 的 pair sets 全部一致。",
    10.5, INK, font=SANS,
)
rect(slide, 6.68, 3.57, 5.88, 1.28, ROSE, LINE)
text(slide, 6.90, 3.70, 5.44, 0.25, "EternaFold · EternaFoldParams.v1", 12.5, BURGUNDY, True)
text(
    slide, 6.90, 4.02, 5.44, 0.62,
    "source 87b9aac · 官方 serial CPU / default MEA；同一 6 条 replay 的 pair sets 全部一致。",
    10.5, INK, font=SANS,
)
rect(slide, 0.78, 5.08, 11.78, 1.08, PALE, RGBColor(205, 161, 170))
text(slide, 1.00, 5.20, 11.32, 0.24, "PROJECT_RAW_GT_ONLY · EternaFold 在此 raw 面板高 +0.016537 micro / +0.017749 macro F1", 11.5, BURGUNDY, True, SANS)
text(
    slide, 1.00, 5.52, 11.32, 0.44,
    "非 clean / 非同源控制 / 非 CHANRG-protocol 可比。两者完整训练全集及与 bpRNA-new 的 homology overlap 未重建；不作 SOTA 排名，不用于模型或阈值开发。",
    10.5, INK, font=SANS,
)
footer(slide, "artifact /teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v1/phase0c_verification.json")

slide = new_slide(
    22,
    "22 · FOLDNAR PHASE 0B–D / GEC NUMERIC CORE",
    "GEC：数值原语与四个控制已验证",
    "假设：首轮精确非交叉后验 + 区间/竞争统计，以 stop-gradient 条件化第二轮残差 pair potential",
)
rect(slide, 0.78, 1.78, 11.78, 0.69, ROSE, LINE)
text(slide, 1.00, 1.87, 11.32, 0.26, "S1_ij = S0_ij + Δθ(H_ij, sg P0_ij, sg M0_ij, g_ij)", 16, BURGUNDY, True)
text(slide, 1.00, 2.16, 11.32, 0.17, "P + endpoint / crossing / inside / enclosing / outside masses · M 是期望计数，不是 union probability", 9.5, MUTED, font=SANS)
controls = (
    (0.78, 2.72, "matched_local", "仅 sigmoid(S⁰)；无精确后验区间"),
    (6.68, 2.72, "zero", "零上下文；隔离 residual 通路"),
    (0.78, 3.61, "destroy_interval_shuffle", "同 span 内置换完整特征；破坏 pair 对齐"),
    (6.68, 3.61, "independent_context", "独立 sigmoid 特征；破坏 exact ensemble 耦合"),
)
for x, y, name, description in controls:
    rect(slide, x, y, 5.88, 0.72, PALE, RGBColor(225, 206, 209))
    text(slide, x + 0.20, y + 0.09, 5.48, 0.22, name, 11.5, BURGUNDY, True)
    text(slide, x + 0.20, y + 0.36, 5.48, 0.20, description, 10, INK, font=SANS)
rect(slide, 0.78, 4.61, 11.78, 1.45, ROSE, LINE)
text(slide, 1.00, 4.73, 3.15, 0.37, "1,280 exhaustive cases", 20, GREEN, True)
text(slide, 1.00, 5.16, 3.15, 0.54, "长度 4 + 5 的全部 ACGU 序列\nlogZ / marginals 对穷举一致", 10.5, INK, font=SANS)
text(slide, 4.44, 4.73, 3.45, 0.37, "CPU ↔ CUDA parity", 18, GREEN, True)
text(slide, 4.44, 5.16, 3.45, 0.54, "CPU 12 passed · CUDA 6 passed\natol / rtol 2e-10；无 CPU/NumPy fallback", 10.5, INK, font=SANS)
text(slide, 8.17, 4.73, 3.85, 0.37, "stop-gradient · 5 / 5 modes", 17, GREEN, True)
text(slide, 8.17, 5.16, 3.85, 0.54, "context / marginals / logZ detached\nscores.grad = None；独立数值测试 5 passed", 10.5, INK, font=SANS)
text(slide, 0.78, 6.17, 11.5, 0.22, "NUMERICS ONLY · 未启动训练或 pilot · 不声称准确率提升、热力学校准或长序列性能。", 10.5, BURGUNDY, True, SANS)
footer(slide, "artifact /teams/HoulabData_1788502323/liucunyu/foldnar/gec_numeric_v1/manifest_v1.json")

path = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
prs.save(path)
print(path)
