#!/usr/bin/env python3
"""Add slides 28-33: the paper's six main figures (user-requested figure
system). Each slide: teal title, hero figure (16:9 aspect preserved), one
conclusion band, footer + page number. Follows the deck's established
typography (微软雅黑, 1.5x spacing, 0E7C86/0F1B2D/333A45/8A93A0/1F3A5F)."""
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor

P = "/Users/liucunyu/Documents/课题idea 数据 结论/浙江大学/RNA 相关模型/RNA-JEV/ppt/RNA-JEV_项目进展_20260924.pptx"
FIGD = "/tmp/ppt_work/figures/main"
p = Presentation(P)
lay = p.slides[-1].slide_layout

L, W = 493776, 9825280
# slide usable area: title 310896..; figure band 1200000..5300000; concl 5450000; footer 6419088
FIG_Y, FIG_H = 1210000, 3960000
FIG_W = int(FIG_H * 3057 / 1200)  # preserve aspect; center horizontally
FIG_X = L + (W - FIG_W) // 2

TEAL, DARK, BODY, GREY, NAVY = "0E7C86", "0F1B2D", "333A45", "8A93A0", "1F3A5F"

def tb(slide, name, x, y, w, h, text, size, color, bold=False):
    box = slide.shapes.add_textbox(x, y, w, h)
    box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    para = tf.paragraphs[0]
    para.line_spacing = 1.5
    r = para.add_run(); r.text = text
    f = r.font
    f.name = "微软雅黑"; f.size = Pt(size)
    f.color.rgb = RGBColor.from_string(color); f.bold = bold
    return box

figs = [
    ("fig1_main_benchmark.png",
     "主图 1 · 主 benchmark：8 板对比",
     "7/8 超最强实测参照（TS2 −0.044 如实写成范式边界）；单模型 5/8、xens2 集成 7/8。"),
    ("fig2_speed_benchmark.png",
     "主图 2 · 速度 benchmark：吞吐阶梯",
     "B=1→B=256 阶梯 21.4→124.78 seq/s；解码 47.3→2.9 ms/seq；同节点 RiNALMo-ft 14.5 = 8.6×。"),
    ("fig3_confidence.png",
     "主图 3 · 置信度 vs 下游分数",
     "结构级置信度与逐序列 F1 强相关（Spearman 0.709，n=1281，p<1e-195）；重标定 ECE 0.160→0.031。"),
    ("fig4_ece.png",
     "主图 4 · ECE 校准全景",
     "8-split 全板：大样本族 4/4 干净迁移；小样本族过矫如实报告（负迁移可见）。"),
    ("fig5_acc_speed.png",
     "主图 5 · 精度 × 速度联合前沿",
     "联合前沿属于决策头范式：F1×吞吐综合分 3.7× 于 UFold、8.5× 于 RiNALMo-ft。"),
    ("fig6_architecture.png",
     "主图 6 · Jev 架构与先验栈",
     "冻结 RiNALMo-giga + 平面决策头 + 先验栈；消融显示 OOD 由结构先验承载（DP +0.25 / 物理 +0.37 / 2D 头 +0.16）。"),
]

for i, (fn, title, concl) in enumerate(figs):
    s = p.slides.add_slide(lay)
    n = 28 + i
    tb(s, "T1", L, 310896, W, 250000, title, 15, TEAL, bold=True)
    s.shapes.add_picture(f"{FIGD}/{fn}", FIG_X, FIG_Y, FIG_W, FIG_H)
    tb(s, "C1", L, 5320000, W, 800000, concl, 11, DARK)
    tb(s, "F1", L, 6419088, 8000000, 280000,
       "RNA-JEPA 决策模型线 · 2026-10-19 · 提交 358f19e", 10, GREY)
    tb(s, "F2", 8769096, 6419088, 500000, 280000, str(n), 11, NAVY)

p.save(P)
print("slides 28-33 added; total", len(p.slides._sldIdLst))
