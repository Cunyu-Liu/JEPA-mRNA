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


text(0.78, 0.34, 11.8, 0.28, "16 · STRUCTURED RE-DECODING DECISION", 11, BURGUNDY, True)
text(0.78, 0.77, 11.9, 0.58, "SRD 未超过当前 MEA：独立分支停止", 27, INK, True, "Source Han Serif SC")
text(0.78, 1.39, 11.9, 0.32, "冻结 seed0 checkpoint · VL0 196 条 · 相同 pair scores · 独立重算最大差值 0.0", 12, MUTED, font="Source Han Sans SC")

rows = [
    ("Nussinov MAP", "0.8002", "0.8048", "0.7782", "0.5368", "4.666"),
    ("当前 MEA", "0.8257", "0.8058", "0.8535", "0.5685", "19.870"),
    ("nested matching", "0.8002", "0.8048", "0.7782", "0.5368", "4.648"),
    ("nested SRD stacking", "0.8002", "0.8048", "0.7782", "0.5370", "3.989"),
    ("best-of-N", "0.8002", "0.8048", "0.7782", "0.5368", "0.034"),
]
headers = ("Decoder", "micro F1", "macro F1", "micro P", "micro Stem", "总秒数")
table = slide.shapes.add_table(6, 6, Inches(0.78), Inches(1.93), Inches(11.78), Inches(2.28)).table
for index, width in enumerate((3.48, 1.62, 1.62, 1.62, 1.74, 1.70)):
    table.columns[index].width = Inches(width)
for r in range(6):
    table.rows[r].height = Inches(0.38)
for c, value in enumerate(headers):
    cell = table.cell(0, c)
    cell.fill.solid(); cell.fill.fore_color.rgb = BURGUNDY; cell.text = value
for r, row in enumerate(rows, 1):
    for c, value in enumerate(row):
        cell = table.cell(r, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = ROSE if r % 2 == 0 else WHITE
        cell.text = value
for r in range(6):
    for c in range(6):
        cell = table.cell(r, c)
        cell.margin_left = cell.margin_right = Inches(0.06)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        p.font.name = "Source Han Sans SC" if c == 0 else "Arial"
        p.font.size = Pt(11)
        p.font.bold = r == 0 or r in (2, 4)
        p.font.color.rgb = WHITE if r == 0 else (GREEN if r == 2 else (BURGUNDY if r == 4 else INK))

rect(0.78, 4.55, 5.66, 1.34, ROSE, RGBColor(222, 192, 184))
text(1.0, 4.70, 5.22, 0.28, "等价性与成本", 14, BURGUNDY, True, "Source Han Sans SC")
text(1.0, 5.02, 5.22, 0.68, "同候选、同目标下，精确 DP 与穷举求解器等价；QUBO 无新增价值。SRD 与 MAP 在 195/196 条结构上相同，约束满足 100%，零失败、零 fallback。", 12, INK, font="Source Han Sans SC")
rect(6.67, 4.55, 5.89, 1.34, RGBColor(250, 245, 246), RGBColor(205, 161, 170))
text(6.89, 4.70, 5.45, 0.28, "停止裁决", 14, BURGUNDY, True, "Source Han Sans SC")
text(6.89, 5.02, 5.45, 0.68, "SRD−MEA：micro F1 −0.02550，macro F1 −0.00092，micro Stem −0.03150。不运行多模型重组，不搜索解码参数，不开放外部评估或 SOTA。", 12, INK, font="Source Han Sans SC")

line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.78), Inches(6.26), Inches(11.78), Pt(1))
line.fill.solid(); line.fill.fore_color.rgb = RGBColor(212, 165, 154); line.line.fill.background()
text(0.78, 6.44, 11.3, 0.25, "STOP_SRD · summary d0147bcd…e5bb · verification 10551f3d…fd7f · code 03f9afe", 9, MUTED)
text(11.96, 6.91, 0.6, 0.22, "17", 10, BURGUNDY, True, align=PP_ALIGN.RIGHT)

path = OUT / "RNA-JEV_SRD_VL0停止裁决_20261124.pptx"
prs.save(path)
print(path)
