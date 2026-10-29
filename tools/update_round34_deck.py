#!/usr/bin/env python3
"""Apply verified Round-33 figure replacements and rebuild Fig. 6 as native vectors."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

SRC = Path("/private/tmp/rnajev_audit/RNA-JEV_论文逻辑重构_20261026_导师讨论_preview_draft.pptx")
OUT = Path("/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261028_03/RNA-JEV_论文逻辑重构_20261028_导师讨论_round34_working.pptx")
FIG = Path("/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261028_03/figure_assets")
FONT = "微软雅黑"
C = {
    "ink": "17243A", "muted": "667085", "faint": "98A2B3", "blue": "0F4D92",
    "blue2": "3775BA", "teal": "0E7C86", "green": "287A5B", "red": "B64342",
    "gold": "C98A22", "surface": "F5F7FA", "rule": "DDE3EA", "white": "FFFFFF",
    "paleBlue": "EAF1F8", "paleTeal": "E9F5F4", "paleGold": "FBF4E4", "dark": "101B31",
}


def rgb(key):
    return RGBColor.from_string(C.get(key, key))


def remove(shape):
    shape._element.getparent().remove(shape._element)


def text(slide, value, x, y, w, h, size=10, color="ink", bold=False,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE, margin=0.02):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = Inches(margin)
    frame.margin_top = frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    frame.word_wrap = True
    lines = str(value).split("\n")
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.line_spacing = Pt(size * 1.5)
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(0)
        run = paragraph.add_run()
        run.text = line
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = rgb(color)
    return shape


def box(slide, x, y, w, h, fill="white", line="rule", radius=True, width=0.7):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line)
    shape.line.width = Pt(width)
    return shape


def labeled_box(slide, label, x, y, w, h, fill="white", line="rule", size=9,
                color="ink", bold=False, align=PP_ALIGN.CENTER):
    box(slide, x, y, w, h, fill, line)
    return text(slide, label, x + 0.06, y + 0.03, w - 0.12, h - 0.06, size, color, bold, align)


def connector(slide, x1, y1, x2, y2, color="muted", width=1.1, arrow=True):
    shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width)
    if arrow:
        ln = shape._element.spPr.get_or_add_ln()
        tail = OxmlElement("a:tailEnd")
        tail.set("type", "triangle")
        ln.append(tail)
    return shape


def matrix(slide, x, y, cell=0.16, n=6, palette=("paleBlue", "blue2", "teal")):
    for i in range(n):
        for j in range(n):
            key = palette[0] if i == j else palette[1] if (i + j) % 3 == 0 else palette[2] if i < j else "surface"
            box(slide, x + j * cell, y + i * cell, cell - 0.025, cell - 0.025,
                key, "white", radius=False, width=0.25)


def replace_picture(slide, image_path):
    pictures = [shape for shape in slide.shapes if shape.shape_type == 13]
    assert len(pictures) == 1, len(pictures)
    image_part, r_id = slide.part.get_or_add_image_part(str(image_path))
    pictures[0]._element.blipFill.blip.rEmbed = r_id


def text_frames(container):
    for shape in container.shapes:
        if getattr(shape, "has_text_frame", False):
            yield shape.text_frame
        elif getattr(shape, "has_table", False):
            for row in shape.table.rows:
                for cell in row.cells:
                    yield cell.text_frame


def replace_run_text(container, old, new, expected):
    count = 0
    for frame in text_frames(container):
        for paragraph in frame.paragraphs:
            for run in paragraph.runs:
                if old in run.text:
                    run.text = run.text.replace(old, new)
                    count += 1
    assert count == expected, (old, count, expected)


def update_speed_provenance(prs):
    replacements = {
        5: [
            ("主图 2｜r2d 单模型 124.78 seq/s，F1 逐位不变",
             "主图 2｜历史 head-only 峰值 124.78 seq/s，禁止跨口径比较", 2),
            ("RTX 4090、TS0、端到端含 frozen encoder + 9.3M head + C decode。",
             "RTX 4090、TS0；JEV 梯图为 cached-embedding/head-only，812,375 head；非 raw-sequence 端到端。", 1),
            ("124.78 属于 r2dtr1c-s0（F1=0.667971），不与 xens2 的 0.8039 拼接；A100 与 forward-only 参照分别标注。",
             "124.78 属于 r2dtr1c-s0 的历史 head-only 峰值；当前 raw-sequence B=1 为 22.28 seq/s（F1=0.668181），二者不得混用。", 2),
        ],
        9: [
            ("主图 5｜真实帕累托集：A100 上是 RNAformer 与 UFold",
             "主图 5｜历史 A100 帕累托集；4090 共同协议重测中", 2),
            ("同节点、同 split、strict micro-F1；r2d 在 A100 被 RNAformer 同时以更高 F1 和吞吐支配。",
             "当前图仅作 A100 历史参照；正式版使用同一 4090、TS0、raw-sequence 端到端 B=1/2/4 结果。", 1),
            ("4090 的 r2d=124.78 seq/s 仅作硬件更新，不混入 A100 前沿；不虚构 xens2 的吞吐或联合最优点。",
             "旧 124.78 为 cached-embedding/head-only，不能混入 raw-sequence 前沿；当前仅四个系统 B=1 已通过，完整网格待验收。", 2),
        ],
        15: [
            ("r2d 在 4090 达 124.78 seq/s，批大小改变不改预测",
             "4090 raw-seq B=1：r2d 22.28 seq/s；旧 head-only 峰值 124.78", 1),
            ("A100 r2d 不在 Pareto 前沿；不把该速度与 xens2 F1 拼成联合点",
             "共同 batch 网格未闭环；禁止跨口径拼接速度或 xens2 F1", 1),
        ],
        17: [
            ("附录 B / 速度协议", "附录 B / 历史 head-only 速度协议", 1),
            ("速度测试串行执行；不把跨硬件或 forward-only 读数写成同口径。",
             "本页为历史 head-only 阶梯；不得与 raw-sequence 或 forward-only 读数混用。", 1),
            ("配置", "历史 head-only 配置", 1),
            ("终值；峰值", "历史峰值", 1),
            ("124.78 属于 r2dtr1c-s0，不是 xens2；F1=0.667971。",
             "124.78 属于 r2dtr1c-s0 的历史 head-only 峰值，不是 xens2；F1=0.667971。", 1),
            ("端到端范围", "计时范围", 1),
            ("含 frozen RiNALMo 前向、9.3M 头与 C 解码；不是 encoder-cache 口径。",
             "含 cached embedding、812,375 head 与 C 解码；不包含 raw-sequence 编码器前向。", 1),
        ],
        20: [
            ("r2d 在 4090 达 124.78 seq/s",
             "raw-seq B=1：r2d 22.28 seq/s", 1),
            ("A100 前沿为 RNAformer/UFold", "旧 head-only 峰值 124.78；共同 batch 网格待验收", 1),
        ],
    }
    for slide_number, items in replacements.items():
        slide = prs.slides[slide_number - 1]
        for old, new, expected in items:
            replace_run_text(slide, old, new, expected - 1 if expected == 2 else expected)
            if expected == 2:
                replace_run_text(slide.notes_slide, old, new, 1)


def rebuild_fig6(slide):
    for index, shape in enumerate(list(slide.shapes)):
        if index < 6:
            continue
        if hasattr(shape, "text") and shape.text.startswith("与后文关系"):
            continue
        remove(shape)

    # Three coordinated panels with one continuous left-to-right data flow.
    panels = [(0.55, 3.72, "paleBlue"), (4.52, 4.28, "surface"), (9.05, 3.73, "paleTeal")]
    for x, w, fill in panels:
        box(slide, x, 1.55, w, 4.96, fill, "rule")
    text(slide, "A  FROZEN SEQUENCE REPRESENTATION", 0.76, 1.73, 3.30, 0.32, 10.3, "blue", True)
    text(slide, "B  PAIRWISE 2D DECISION", 4.76, 1.73, 3.80, 0.32, 10.3, "blue", True)
    text(slide, "C  CONSTRAINED DECODING + CALIBRATION", 9.25, 1.73, 3.32, 0.32, 9.2, "teal", True)

    # A: sequence, frozen encoder and embedding tensor.
    text(slide, "RNA sequence · length L", 0.82, 2.06, 2.90, 0.22, 8.6, "muted")
    for index, token in enumerate(["A", "U", "G", "C", "…", "U"]):
        labeled_box(slide, token, 0.78 + index * 0.48, 2.38, 0.38, 0.38,
                    "white", "rule", 9.2, "ink", True)
    connector(slide, 2.41, 2.76, 2.41, 3.20, "blue")
    box(slide, 0.92, 3.20, 3.02, 0.92, "blue", "blue", True, 0.6)
    text(slide, "RiNALMo-giga · 650M", 1.08, 3.30, 2.70, 0.28, 12, "white", True, PP_ALIGN.CENTER)
    text(slide, "33 Transformer blocks · 1280-d output", 1.08, 3.62, 2.70, 0.24, 8.8, "white", False, PP_ALIGN.CENTER)
    text(slide, "frozen backbone · 0 trainable parameters", 1.08, 3.87, 2.70, 0.24, 8.8, "white", False, PP_ALIGN.CENTER)
    connector(slide, 2.41, 4.12, 2.41, 4.46, "blue")
    matrix(slide, 1.32, 4.46, 0.18, 7, ("paleBlue", "blue2", "teal"))
    text(slide, "h: L × 1280", 2.76, 4.62, 1.34, 0.34, 9.6, "blue", True, PP_ALIGN.CENTER)
    text(slide, "one embedding per nucleotide", 2.76, 5.00, 1.34, 0.42, 8.6, "muted", False, PP_ALIGN.CENTER)
    labeled_box(slide, "cached or recomputed on raw sequences", 0.91, 5.73, 3.02, 0.42,
                "white", "rule", 8.6, "muted")
    text(slide, "Output: contextual sequence features", 0.91, 6.20, 3.02, 0.22, 8.6, "blue", True, PP_ALIGN.CENTER)

    # B: symmetric pair representation, pair tensor and 2D scorer.
    labeled_box(slide, "h_i", 4.82, 2.22, 0.62, 0.38, "white", "rule", 10, "blue", True)
    labeled_box(slide, "h_j", 5.55, 2.22, 0.62, 0.38, "white", "rule", 10, "blue", True)
    text(slide, "sum  h_i + h_j\nproduct  h_i ⊙ h_j\ndistance  |h_i − h_j|", 6.36, 2.16, 2.10, 0.66,
         8.6, "ink", False, PP_ALIGN.LEFT)
    text(slide, "zᵢⱼ = W_z [sum ; product ; distance]\n+ relative-offset(i,j)",
         4.82, 2.84, 3.68, 0.52, 8.6, "muted", False, PP_ALIGN.CENTER)
    matrix(slide, 4.95, 3.45, 0.18, 6, ("paleBlue", "blue2", "teal"))
    text(slide, "L × L × 128", 4.86, 4.62, 1.34, 0.24, 8.6, "muted", False, PP_ALIGN.CENTER)
    connector(slide, 6.03, 3.96, 6.60, 3.96, "blue")
    labeled_box(slide, "ResNet2D scorer\n1×1 stem → 4 residual blocks (3×3)\n→ 1×1 head",
                6.60, 3.35, 1.92, 1.18, "blue2", "blue2", 8.6, "white", True)
    labeled_box(slide, "symmetric L × L pair logits\nsingle forward · no autoregression",
                4.84, 5.03, 3.66, 0.60, "white", "rule", 8.8, "blue", True)
    labeled_box(slide, "NLL + KL distillation + RLCD + soft-ECE",
                4.84, 5.78, 3.66, 0.38, "paleGold", "paleGold", 8.6, "ink", False)
    text(slide, "decision head: 812,375 trainable parameters", 4.84, 6.22, 3.66, 0.22,
         8.8, "blue", True, PP_ALIGN.CENTER)

    # C: prior injection, constrained DP and two output branches.
    labeled_box(slide, "scoreᵢⱼ + w_turner × TurnerPriorᵢⱼ\nvalid-pair mask: canonical pairs + min-loop",
                9.34, 2.20, 3.16, 0.68, "white", "rule", 8.6, "ink", True)
    connector(slide, 10.92, 2.88, 10.92, 3.24, "teal")
    labeled_box(slide, "non-crossing max-product DP\nC-batched exact decode · legal by construction",
                9.34, 3.24, 3.16, 0.86, "teal", "teal", 9.0, "white", True)
    text(slide, "O(L³) exact decode; no partition-function computation",
         9.36, 4.17, 3.12, 0.30, 8.6, "muted", False, PP_ALIGN.CENTER)
    labeled_box(slide, "STRUCTURE\ndot-bracket + legal base pairs",
                9.34, 4.78, 1.50, 0.88, "white", "rule", 8.5, "teal", True)
    labeled_box(slide, "CONFIDENCE\nsigmoid → VL0 affine → SC / ECE",
                11.00, 4.78, 1.50, 0.88, "white", "rule", 8.5, "blue", True)
    text(slide, "structure branch", 9.36, 5.76, 1.46, 0.24, 8.5, "muted", False, PP_ALIGN.CENTER)
    text(slide, "calibration branch", 11.00, 5.76, 1.50, 0.24, 8.5, "muted", False, PP_ALIGN.CENTER)
    labeled_box(slide, "legal RNA secondary structure + per-structure confidence",
                9.34, 6.06, 3.16, 0.42, "paleTeal", "paleTeal", 8.6, "teal", True)

    # Cross-panel tensor hand-offs, drawn above the panels.
    connector(slide, 4.28, 3.82, 4.50, 3.82, "blue", 1.4)
    text(slide, "h", 4.28, 3.55, 0.22, 0.22, 8.5, "blue", True, PP_ALIGN.CENTER)
    connector(slide, 8.80, 3.82, 9.03, 3.82, "teal", 1.4)
    text(slide, "logits", 8.69, 3.55, 0.45, 0.22, 8.5, "teal", True, PP_ALIGN.CENTER)


def main():
    prs = Presentation(SRC)
    replace_picture(prs.slides[4], FIG / "fig02_speed_benchmark_3840x1920.png")
    replace_picture(prs.slides[5], FIG / "fig03a_ablation_matrix_3840x1920.png")
    replace_picture(prs.slides[6], FIG / "fig03b_ood_signature_3840x1920.png")
    replace_picture(prs.slides[7], FIG / "fig04_ece_confidence_3840x1920.png")
    replace_picture(prs.slides[8], FIG / "fig05_accuracy_speed_pareto_3840x1920.png")
    for shape in list(prs.slides[5].shapes):
        if hasattr(shape, "text") and shape.text.startswith("那么这部分我有一些建议"):
            remove(shape)
    for slide_index, prefix in ((8, "更新了，到 4090"), (11, "Splice 的没放")):
        for shape in list(prs.slides[slide_index].shapes):
            if hasattr(shape, "text") and shape.text.startswith(prefix):
                remove(shape)
    rebuild_fig6(prs.slides[9])
    update_speed_provenance(prs)
    prs.save(OUT)
    print(f"ROUND34_DECK_WORKING_WRITTEN {OUT}")


if __name__ == "__main__":
    main()
