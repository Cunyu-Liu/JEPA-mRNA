import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt


def set_east_asian_font(run):
    properties = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        element = properties.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            properties.append(element)
        element.set("typeface", "微软雅黑")

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "交接文档/PRISM_组会汇报_20261107_完整版_副本.pptx"
DECISION = ROOT / ".round56_work/pilot_v1_decision.json"
BLUE = RGBColor(43, 91, 143)
TEAL = RGBColor(47, 127, 125)
ORANGE = RGBColor(186, 96, 54)
DARK = RGBColor(35, 48, 61)
GRAY = RGBColor(100, 112, 123)
LIGHT = RGBColor(239, 244, 248)
WHITE = RGBColor(255, 255, 255)


def textbox(slide, x, y, w, h, text, size=14, color=DARK, bold=False,
            align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = valign
    frame.margin_left = frame.margin_right = Inches(0.04)
    frame.margin_top = frame.margin_bottom = Inches(0.02)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = 1.5
    run = paragraph.add_run()
    run.text = text
    run.font.name = "微软雅黑"
    set_east_asian_font(run)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def card(slide, x, title, value, color):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(1.65), Inches(3.75), Inches(1.15))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT
    shape.line.color.rgb = color
    shape.line.width = Pt(1.2)
    textbox(slide, x + 0.18, 1.78, 3.35, 0.28, title, 11, GRAY)
    textbox(slide, x + 0.18, 2.09, 3.35, 0.48, value, 23, color, True)


def main():
    decision = json.loads(DECISION.read_text())
    assert decision["status"] == "STOP_PILOT_NEGATIVE_DEV_ONLY" and not decision["pilot_passed"]
    results = decision["results"]
    prs = Presentation(DECK)
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    textbox(slide, 0.65, 0.35, 8.0, 0.3, "DUAL-VIEW RNA-JEPA  |  ROUND 56", 10, TEAL, True)
    textbox(slide, 0.65, 0.73, 11.8, 0.55, "双视图 JEPA：pilot 未通过", 27, DARK, True)
    textbox(slide, 0.65, 1.28, 11.8, 0.28,
            "单 seed=42 · VL0 · 全部 best@100 · 仅机制验证，不是外部精度结论", 11, GRAY)
    card(slide, 0.65, "H0  无 JEPA 宿主", f"{results['H0']['macro_f1']:.6f}", BLUE)
    card(slide, 4.80, "H2  MSE + target SIGReg", f"{results['H2']['macro_f1']:.6f}", TEAL)
    card(slide, 8.95, "H2 − H0  macro-F1", f"{decision['h2_minus_strongest_control']:+.6f}", ORANGE)
    textbox(slide, 0.65, 3.02, 12.0, 0.3, "全部七臂（独立计数复核）", 12, DARK, True)
    table_shape = slide.shapes.add_table(8, 4, Inches(0.65), Inches(3.38), Inches(12.0), Inches(2.62))
    table = table_shape.table
    table.columns[0].width = Inches(2.1)
    table.columns[1].width = Inches(3.2)
    table.columns[2].width = Inches(3.2)
    table.columns[3].width = Inches(3.5)
    headers = ["臂", "macro-F1", "micro-F1", "停止"]
    for col, value in enumerate(headers):
        cell = table.cell(0, col)
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE
        cell.text = value
    for row, arm in enumerate(("H0", "H1", "H2", "H3", "H4", "H5", "H6"), start=1):
        values = [arm, f"{results[arm]['macro_f1']:.6f}", f"{results[arm]['micro_f1']:.6f}",
                  f"early stop @ {results[arm]['steps_completed']}"]
        for col, value in enumerate(values):
            cell = table.cell(row, col)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if row % 2 else WHITE
            cell.text = value
    for row in range(8):
        table.rows[row].height = Inches(0.327)
        for col in range(4):
            cell = table.cell(row, col)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.LEFT if col == 0 else PP_ALIGN.CENTER
                paragraph.line_spacing = 1.5
                for run in paragraph.runs:
                    run.font.name = "微软雅黑"
                    set_east_asian_font(run)
                    run.font.size = Pt(10.5 if row else 11)
                    run.font.bold = row == 0 or (row == 3 and col == 0)
                    run.font.color.rgb = WHITE if row == 0 else DARK
    textbox(slide, 0.65, 6.20, 12.0, 0.36,
            "裁决：H2 未超过 H0、shuffle、随机编码器、参数匹配 MLP 或全局均值；停止三种子与正式评测。",
            12, ORANGE, True)
    textbox(slide, 0.65, 6.63, 12.0, 0.42,
            "保留：P1 工程门通过、全部 checkpoint 与失败日志。不得以 micro-F1、dev 或单 seed 宣称精度突破。",
            10.5, GRAY)
    textbox(slide, 0.65, 7.08, 12.0, 0.18,
            "manifests/p2/pilot_v1_decision.json  ·  SHA256 b6b47fb4…211c41", 8.5, GRAY)
    notes = slide.notes_slide.notes_text_frame
    if notes is not None:
        notes.text = (
            "这一页报告负结果。P1 的掩码、梯度、CUDA、resume 和序列-only 部署均已通过，"
            "但工程正确不等于结构模态有效。七臂都在第400步早停，best均为100步。"
            "H2的VL0 macro-F1为0.844264，低于H0的0.851460，也未超过其他对照，因此按合同停止三种子。"
            "H2 micro略高不能替代预注册macro判据，也不能把单seed dev结果写成外部精度突破。"
        )
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "deck": str(DECK), "status": decision["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
