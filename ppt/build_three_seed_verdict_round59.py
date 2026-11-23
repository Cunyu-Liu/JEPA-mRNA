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
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(34, 113, 84)

prs = Presentation()
prs.slide_width = Inches(13.333333)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = WHITE


def text(x, y, w, h, value, size, color=INK, bold=False, font="Arial", align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def rect(x, y, w, h, fill, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    return shape


text(0.78, 0.34, 11.8, 0.28, "15 · THREE-SEED CAPACITY DECISION", 11, BURGUNDY, True)
text(0.78, 0.77, 11.9, 0.58, "三种子否决容量扩张：raw 提升未转化为结构收益", 27, INK, True, "Source Han Serif SC")
text(0.78, 1.39, 11.9, 0.32, "同 seed control_matched 配对差值 · VL0 196 条 · 6 个新 run · 独立复核与交接终审均 PASS", 12, MUTED, font="Source Han Sans SC")

rows = [
    ("seed0", "head_deep", "+0.0775", "+0.0165", "+0.0141", "+0.0277", "+0.0049", "PASS"),
    ("seed0", "repr_head", "+0.1093", "+0.0203", "+0.0077", "+0.0310", "+0.0086", "PASS"),
    ("seed1", "head_deep", "+0.0324", "−0.0060", "−0.0096", "+0.0044", "−0.0087", "FAIL"),
    ("seed1", "repr_head", "+0.0756", "−0.0578", "−0.0785", "−0.0423", "−0.0592", "FAIL"),
    ("seed2", "head_deep", "+0.0793", "−0.0224", "−0.0342", "−0.0277", "−0.0411", "FAIL"),
    ("seed2", "repr_head", "+0.1808", "−0.0660", "−0.0850", "−0.0887", "−0.1187", "FAIL"),
]
headers = ("seed", "臂", "ΔAUPRC", "Δmicro F1", "Δmacro F1", "Δmicro Stem", "Δmacro Stem", "门控")
table = slide.shapes.add_table(7, 8, Inches(0.78), Inches(1.93), Inches(11.78), Inches(2.5)).table
for index, width in enumerate((0.82, 1.72, 1.38, 1.46, 1.46, 1.56, 1.56, 1.82)):
    table.columns[index].width = Inches(width)
for r in range(7):
    table.rows[r].height = Inches(0.357)
for c, value in enumerate(headers):
    cell = table.cell(0, c)
    cell.fill.solid()
    cell.fill.fore_color.rgb = BURGUNDY
    cell.text = value
for r, row in enumerate(rows, 1):
    for c, value in enumerate(row):
        cell = table.cell(r, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = ROSE if r % 2 == 0 else WHITE
        cell.text = value
for r in range(7):
    for c in range(8):
        cell = table.cell(r, c)
        cell.margin_left = cell.margin_right = Inches(0.04)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.font.name = "Arial"
        p.font.size = Pt(10.5)
        p.font.bold = r == 0 or (c == 7 and r > 0)
        p.font.color.rgb = WHITE if r == 0 else INK
        if r > 0 and (cell.text.startswith("−") or cell.text == "FAIL"):
            p.font.color.rgb = BURGUNDY
        elif r > 0 and cell.text == "PASS":
            p.font.color.rgb = GREEN

rect(0.78, 4.72, 5.66, 1.22, ROSE, RGBColor(222, 192, 184))
text(1.0, 4.86, 5.22, 0.28, "一致反例", 14, BURGUNDY, True, "Source Han Sans SC")
text(1.0, 5.17, 5.22, 0.58, "seeds 1/2 的 AUPRC 均提高，但 pair-F1 均下降；高支持/高 FN 子集同样恶化，逐条剔除任一序列后负向不变。", 12, INK, font="Source Han Sans SC")
rect(6.67, 4.72, 5.89, 1.22, RGBColor(250, 245, 246), RGBColor(205, 161, 170))
text(6.89, 4.86, 5.45, 0.28, "停止裁决", 14, BURGUNDY, True, "Source Han Sans SC")
text(6.89, 5.17, 5.45, 0.58, "不再增加 blocks / hidden / d_z；不选冠军 seed，不开放外部评估、严格 OOD 或 SOTA。结构头容量不是稳健的单一修复。", 12, INK, font="Source Han Sans SC")

line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.78), Inches(6.34), Inches(11.78), Pt(1))
line.fill.solid(); line.fill.fore_color.rgb = RGBColor(212, 165, 154); line.line.fill.background()
text(0.78, 6.52, 11.3, 0.25, "STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED · freeze 6f435ceaca…071d · code c289384", 9, MUTED)
text(11.96, 6.91, 0.6, 0.22, "16", 10, BURGUNDY, True, align=PP_ALIGN.RIGHT)

path = OUT / "RNA-JEV_三种子容量裁决_20261124.pptx"
prs.save(path)
print(path)
