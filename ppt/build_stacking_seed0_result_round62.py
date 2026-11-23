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
slide.background.fill.solid(); slide.background.fill.fore_color.rgb = WHITE


def text(x, y, w, h, value, size, color=INK, bold=False, font="Arial", align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame; frame.clear()
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]; p.alignment = align
    run = p.add_run(); run.text = value
    run.font.name = font; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
    return box


def rect(x, y, w, h, fill, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill; shape.line.color.rgb = line or fill
    return shape


text(0.78, 0.34, 11.8, 0.28, "18 · STACKING FORM / SEED0 DECISION", 11, BURGUNDY, True)
text(0.78, 0.77, 11.9, 0.58, "Stacking 局部有效，但 seed0 否决稳健机制", 27, INK, True, "Source Han Serif SC")
text(0.78, 1.39, 11.9, 0.32, "四臂同参数 · 9,600 steps · VL0 196 条 · 独立 raw-score 重算差值 0.0", 12, MUTED, font="Source Han Sans SC")

rows = [
    ("local control", "0.8429", "0.8309", "0.6172", "0.6900", "0.9482"),
    ("stacking closure", "0.8469", "0.8253", "0.6162", "0.6706", "0.9547"),
    ("destroy offset", "0.8479", "0.8292", "0.6210", "0.6775", "0.9494"),
    ("destroy shuffle", "0.8458", "0.8234", "0.6242", "0.6819", "0.9594"),
]
headers = ("Arm", "micro F1", "macro F1", "micro Stem", "macro Stem", "目标 recall")
table = slide.shapes.add_table(5, 6, Inches(0.78), Inches(1.93), Inches(11.78), Inches(1.96)).table
for index, width in enumerate((3.20, 1.65, 1.65, 1.72, 1.76, 1.80)):
    table.columns[index].width = Inches(width)
for r in range(5):
    table.rows[r].height = Inches(0.39)
for c, value in enumerate(headers):
    cell = table.cell(0, c); cell.fill.solid(); cell.fill.fore_color.rgb = BURGUNDY; cell.text = value
for r, row in enumerate(rows, 1):
    for c, value in enumerate(row):
        cell = table.cell(r, c); cell.fill.solid(); cell.fill.fore_color.rgb = ROSE if r % 2 == 0 else WHITE; cell.text = value
for r in range(5):
    for c in range(6):
        cell = table.cell(r, c)
        cell.margin_left = cell.margin_right = Inches(0.06); cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        p.font.name = "Source Han Sans SC" if c in (0, 5) else "Arial"; p.font.size = Pt(11)
        p.font.bold = r == 0 or r == 2
        p.font.color.rgb = WHITE if r == 0 else (BURGUNDY if r == 2 else INK)

rect(0.78, 4.28, 5.66, 1.50, ROSE, RGBColor(222, 192, 184))
text(1.0, 4.45, 5.22, 0.28, "局部信号确实存在", 14, GREEN, True, "Source Han Sans SC")
text(1.0, 4.82, 5.22, 0.72, "候选−control：micro F1 +0.00399，目标边界 recall +0.00647，AUPRC +0.02719。固定 1,700 个目标 pairs，分母不变。", 12, INK, font="Source Han Sans SC")
rect(6.67, 4.28, 5.89, 1.50, RGBColor(250, 245, 246), RGBColor(205, 161, 170))
text(6.89, 4.45, 5.45, 0.28, "停止：macro 与 destroy 门失败", 14, BURGUNDY, True, "Source Han Sans SC")
text(6.89, 4.82, 5.45, 0.72, "macro F1 −0.00564，macro Stem −0.01943，macro P −0.00659；destroy_offset 的 micro F1 更高。不启动 seeds 1/2，不追加模块，不开放外部评估。", 12, INK, font="Source Han Sans SC")

line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.78), Inches(6.18), Inches(11.78), Pt(1))
line.fill.solid(); line.fill.fore_color.rgb = RGBColor(212, 165, 154); line.line.fill.background()
text(0.78, 6.38, 11.3, 0.25, "STOP_STACKING_CLOSURE_FORM_SEED0 · verdict 1e6bc621…c03c · recheck 36a77e6c…c54e · code dde6641", 9, MUTED)
text(11.96, 6.91, 0.6, 0.22, "19", 10, BURGUNDY, True, align=PP_ALIGN.RIGHT)

path = OUT / "RNA-JEV_Stacking_seed0停止裁决_20261124.pptx"
prs.save(path)
print(path)
