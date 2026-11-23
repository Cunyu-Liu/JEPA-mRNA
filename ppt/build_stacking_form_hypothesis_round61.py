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


text(0.78, 0.34, 11.8, 0.28, "17 · NEW FORM HYPOTHESIS / NOT TRAINED", 11, BURGUNDY, True)
text(0.78, 0.77, 11.9, 0.58, "新形式准入：显式 stacking closure，尚未训练", 27, INK, True, "Source Han Serif SC")
text(0.78, 1.39, 11.9, 0.32, "三个冻结 control · VL0 196 条/seed · FN 环境独立复核 PASS · 不是根因结论", 12, MUTED, font="Source Han Sans SC")

cards = [
    (0.78, "60.01%", "helix 边界 FN", "1,789 / 2,981 FN"),
    (4.80, "500", "有正确邻居的边界 FN", "目标子群 · 三个 seed 均存在"),
    (8.82, "10.70%", "孤立 pair FN", "recall 低，但不属本假设"),
]
for x, value, label, note in cards:
    rect(x, 1.93, 3.74, 1.36, ROSE, RGBColor(222, 192, 184))
    text(x + 0.22, 2.07, 3.30, 0.48, value, 25, BURGUNDY, True)
    text(x + 0.22, 2.57, 3.30, 0.26, label, 12, INK, True, "Source Han Sans SC")
    text(x + 0.22, 2.88, 3.30, 0.22, note, 10, MUTED, font="Source Han Sans SC")

rect(0.78, 3.63, 5.66, 2.15, ROSE, RGBColor(222, 192, 184))
text(1.0, 3.80, 5.22, 0.28, "唯一科学变量：closure 拓扑", 14, BURGUNDY, True, "Source Han Sans SC")
text(1.0, 4.18, 5.22, 0.60, "Cᵢⱼ = [Hᵢⱼ, Hᵢ₊₁ⱼ₋₁, Hᵢ₋₁ⱼ₊₁,\nHᵢⱼ⊙Hᵢ₊₁ⱼ₋₁, Hᵢⱼ⊙Hᵢ₋₁ⱼ₊₁]", 15, INK)
text(1.0, 4.89, 5.22, 0.66, "对称化输出，canonical/min-loop 外置零；替换一个既有模块，不增加 blocks / hidden / d_z。物理对象仅为连续碱基 stacking 与 helix 末端。", 11.5, INK, font="Source Han Sans SC")

rect(6.67, 3.63, 5.89, 2.15, RGBColor(250, 245, 246), RGBColor(205, 161, 170))
text(6.89, 3.80, 5.45, 0.28, "证伪门：全局终点＋目标子群", 14, BURGUNDY, True, "Source Han Sans SC")
text(6.89, 4.19, 5.45, 1.32, "同参数非 stacking control；固定 offset 与 channel shuffle 两个 destroy 对照。\n\n必须改善目标边界 recall，同时通过 pair-F1 / stem-F1 全部门且优于 destroy。raw-only、单 seed-only 或 destroy 同样改善，立即停止。", 11.5, INK, font="Source Han Sans SC")

line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.78), Inches(6.20), Inches(11.78), Pt(1))
line.fill.solid(); line.fill.fore_color.rgb = RGBColor(212, 165, 154); line.line.fill.background()
text(0.78, 6.39, 11.3, 0.25, "HYPOTHESIS ONLY · FN audit 72cc68cb…742d · verification 5d34f46b…7527 · code 6d38f40", 9, MUTED)
text(11.96, 6.91, 0.6, 0.22, "18", 10, BURGUNDY, True, align=PP_ALIGN.RIGHT)

path = OUT / "RNA-JEV_Stacking形式假设_20261124.pptx"
prs.save(path)
print(path)
