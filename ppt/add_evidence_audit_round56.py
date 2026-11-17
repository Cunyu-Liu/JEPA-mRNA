import copy
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "交接文档/PRISM_组会汇报_20261107_完整版_副本.pptx"
WORK = ROOT / ".round56_work"
BLUE = RGBColor(43, 91, 143)
TEAL = RGBColor(47, 127, 125)
ORANGE = RGBColor(186, 96, 54)
DARK = RGBColor(35, 48, 61)
GRAY = RGBColor(100, 112, 123)
LIGHT = RGBColor(239, 244, 248)
WHITE = RGBColor(255, 255, 255)


def east_asian(run):
    properties = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        element = properties.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            properties.append(element)
        element.set("typeface", "微软雅黑")


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
    east_asian(run)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def blank(prs, section, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    textbox(slide, 0.65, 0.34, 8.8, 0.28, section, 10, TEAL, True)
    textbox(slide, 0.65, 0.73, 12.0, 0.55, title, 26, DARK, True)
    textbox(slide, 0.65, 1.27, 12.0, 0.3, subtitle, 11, GRAY)
    return slide


def card(slide, x, y, width, title, value, color=BLUE, value_size=22):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(width), Inches(1.18))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT
    shape.line.color.rgb = color
    shape.line.width = Pt(1.2)
    textbox(slide, x + 0.16, y + 0.13, width - 0.32, 0.28, title, 10.5, GRAY)
    textbox(slide, x + 0.16, y + 0.45, width - 0.32, 0.5, value, value_size, color, True)


def table(slide, data, x, y, widths, row_height=0.52, font_size=11):
    shape = slide.shapes.add_table(len(data), len(data[0]), Inches(x), Inches(y),
                                   Inches(sum(widths)), Inches(row_height * len(data)))
    result = shape.table
    for index, width in enumerate(widths):
        result.columns[index].width = Inches(width)
    for row, values in enumerate(data):
        result.rows[row].height = Inches(row_height)
        for column, value in enumerate(values):
            cell = result.cell(row, column)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE if row == 0 else LIGHT if row % 2 else WHITE
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.LEFT if column == 0 else PP_ALIGN.CENTER
                paragraph.line_spacing = 1.5
                for run in paragraph.runs:
                    run.font.name = "微软雅黑"
                    east_asian(run)
                    run.font.size = Pt(font_size)
                    run.font.bold = row == 0
                    run.font.color.rgb = WHITE if row == 0 else DARK
    return result


def set_notes(prs, slide, text):
    if slide.notes_slide.notes_text_frame is None:
        source = next(other.notes_slide.notes_placeholder for other in prs.slides
                      if other.notes_slide.notes_text_frame is not None)
        slide.notes_slide.shapes._spTree.insert_element_before(copy.deepcopy(source._element), "p:extLst")
    slide.notes_slide.notes_text_frame.text = text


def main():
    audit = json.loads((WORK / "data_exposure_audit.json").read_text())
    initialization = json.loads((WORK / "initialization_report.json").read_text())
    motif = json.loads((WORK / "bpfold_decision.json").read_text())
    assert audit["status"] == "PASS_AUDIT_WITH_EXPLICIT_LIMITS"
    assert initialization["status"] == "REPORT_INITIALIZATION_CONFOUNDING_ONLY"
    assert motif["status"] == "STOP_BPFOLD_DEV_NEGATIVE" and not motif["external_evaluation_started"]
    prs = Presentation(DECK)
    if len(prs.slides) != 20:
        assert len(prs.slides) == 22
        set_notes(prs, prs.slides[19], "这一页报告双视图pilot负结果。工程通过不等于结构模态有效；H2低于H0及其余对照，按合同停止三种子，不以micro-F1、dev或单seed宣称突破。")
        prs.save(DECK)
        print(json.dumps({"slides": len(prs.slides), "status": "NOTES_QA_ONLY"}, ensure_ascii=False))
        return
    slide = blank(prs, "DATA & MECHANISM AUDIT  |  ROUND 56",
                  "数据审计：不能把重叠与暴露风险隐藏掉",
                  "来源、独立性与预训练暴露优先于精度叙事 · 审计通过 ≠ clean-OOD 证明")
    card(slide, 0.65, 1.75, 3.75, "TR1c 唯一序列 / 80% 簇",
         f"{audit['tr1c']['n_unique_sequences']:,} / {audit['tr1c']['n_80pct_clusters']:,}")
    rfam = audit["evaluation_exact_overlap"]["rfam_fam.jsonl"]
    card(slide, 4.80, 1.75, 3.75, "Rfam ∩ TR1c 精确 / 簇",
         f"{rfam['exact_overlap_with_tr1c']} / {rfam['cluster_overlap_with_train']}", ORANGE)
    removed = audit["ncrna_clean"]["splits"]["test.csv"]["removed_exact_tr1c_rows"]
    kept = audit["ncrna_clean"]["splits"]["test.csv"]["output_n"]
    card(slide, 8.95, 1.75, 3.75, "ncRNA test 去重叠 / 保留",
         f"{removed} / {kept:,}", TEAL, 20)
    textbox(slide, 0.65, 3.14, 12.0, 0.28, "初始化是必须区分的协议因素（VL0 macro-F1）", 12, DARK, True)
    warm = initialization["arms"]
    table(slide, [
        ["初始化", "H0", "H2", "H2 − H0", "解释边界"],
        ["Plan-A warm-start", f"{warm['H0']['warm_plan_a_macro_f1']:.6f}",
         f"{warm['H2']['warm_plan_a_macro_f1']:.6f}",
         f"{warm['H2']['warm_plan_a_macro_f1'] - warm['H0']['warm_plan_a_macro_f1']:+.6f}", "model-dev 负结果"],
        ["RiNALMo 原生 · step500", f"{warm['H0']['native_rinalmo_macro_f1']:.6f}",
         f"{warm['H2']['native_rinalmo_macro_f1']:.6f}",
         f"{initialization['native_h2_minus_native_h0']:+.6f}", "未收敛；不作机制证据"],
    ], 0.65, 3.52, [2.65, 1.6, 1.6, 1.65, 4.5], 0.55, 10.5)
    textbox(slide, 0.65, 5.53, 12.0, 0.5,
            "仍阻断：RiNALMo 全预训练语料未冻结；原优化器延续 checkpoint 损坏。\ncanonical-nested 不是完整三维结构；80% 簇数不等于独立家族数。", 11, ORANGE, True)
    textbox(slide, 0.65, 6.27, 12.0, 0.48,
            "机制测量：单工程 batch 梯度 cosine = −0.052939；完整宿主 forward/backward = 199,494,385,321 FLOPs。\n仅 profiler 支持范围，不含精确 DP；不是普遍机制结论。", 10, GRAY)
    textbox(slide, 0.65, 7.07, 12.0, 0.18,
            "manifests/v2/data_audit  ·  ee1b7df5…c7e82   |   initialization_report  ·  325f79f1…bf29", 8.5, GRAY)
    set_notes(prs, slide, (
        "这一页先讲数据边界。TR1c有四万二千多条唯一序列，但只有一万五千多个80%簇，二者不能混称独立家族。"
        "Rfam仍有精确和簇重叠，ncRNA测试集也实际删除了304条TR1c精确重叠。"
        "原生初始化的小幅正差不能解读为结构模态有效，因为两臂都没收敛；warm-start的负结果也不做全称否定。"
        "预训练暴露和优化器延续仍是明确阻断项。"
    ))

    slide = blank(prs, "INDEPENDENT PHYSICS CONTROL  |  ROUND 56",
                  "BPfold motif：独立 dev 未显示收益",
                  "官方 key.energy · MIT · 单 seed model-dev · 原 Turner 与精确 DP 保持不变")
    values = motif["results_macro_f1"]
    card(slide, 0.65, 1.78, 2.75, "P0  无 motif", f"{values['P0_no_motif']:.6f}", BLUE, 20)
    card(slide, 3.75, 1.78, 2.75, "P1  inner / outer", f"{values['inner_outer']:.6f}", TEAL, 20)
    card(slide, 6.85, 1.78, 2.75, "P2  shuffled", f"{values['shuffled']:.6f}", ORANGE, 20)
    card(slide, 9.95, 1.78, 2.75, "P3  Turner = 0", f"{values['turner0']:.6f}", ORANGE, 20)
    textbox(slide, 0.65, 3.22, 12.0, 0.3, "同一 VL0 · macro-F1 · 独立计数复核", 12, DARK, True)
    table(slide, [
        ["比较", "差值 / 结果", "裁决"],
        ["P1 − P0", f"{motif['p1_minus_p0']:+.6f}", "未改善"],
        ["P1 − shuffled", f"{motif['p1_minus_shuffled']:+.6f}", "未超过随机结构对照"],
        ["P4：能量图，无 attention bias", f"{values['energy_only']:.6f}", "与 P0 相同"],
    ], 0.65, 3.63, [4.2, 2.7, 5.1], 0.62, 11)
    textbox(slide, 0.65, 6.16, 12.0, 0.4,
            "STOP_BPFOLD_DEV_NEGATIVE：不启动外部测试，不用物理先验叙事补偿精度。", 12, ORANGE, True)
    textbox(slide, 0.65, 6.68, 12.0, 0.3,
            "范围：官方 BRIQ 能量表输入实验，不等于完整 BPfold 复现；Turner / RIPE 不混称 BRIQ。", 10.5, GRAY)
    textbox(slide, 0.65, 7.08, 12.0, 0.18,
            "manifests/v2/bpfold_decision.json  ·  SHA256 55d2dfad…927b  ·  key.energy b657c553…ce8b", 8.5, GRAY)
    set_notes(prs, slide, (
        "这一页是独立物理对照，不是旧Turner或RIPE改名。我们核验了官方MIT能量表和哈希，"
        "在保持宿主、Turner和精确解码的条件下比较inner/outer、shuffle、Turner置零以及仅能量图输入。"
        "P1和P0完全相同，也没有超过shuffled，所以按合同停止，不访问外部测试。"
        "这不是对BPfold原论文的否定，也不是完整BPfold复现。"
    ))
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "deck": str(DECK), "status": "ADD_VERIFIED_AUDIT_AND_BPFOLD_ONLY"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
