#!/usr/bin/env python3
"""Add slide 14: TS2 integration axes exhausted + preprint v3.26 submission-ready.
Content mirrors slide 13's proven layout (title bar / red headline / thin rule /
conclusion band / footer+page no), all numbers pre-checked (§15.83/§15.85,
checkers 8/8 + 9/9, JSON triple-verified)."""
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import copy

P = "/Users/liucunyu/Documents/课题idea 数据 结论/浙江大学/RNA 相关模型/RNA-JEV/ppt/RNA-JEV_汇报_精简版_20261016.pptx"
p = Presentation(P)
src = p.slides[-1]  # slide 13 as template

# clone the slide via layout then copy shapes from 13
lay = src.slide_layout
new = p.slides.add_slide(lay)

TITLE = "TS2 集成轴双向穷尽 · 预印本 v3.26 投稿就绪"
HEAD = "plana 桶 2/3/4-seed 0.8604/0.8581/0.8526 单调递减 ｜ 集成轴关闭 ｜ 6+1 战线终判 −0.044"
CONCL = ("结论\u3000plana 桶种子多样性轴穷尽（板协议 2-seed 即最优：精确率型种子稀释召回端，FN 141→146）；"
         "至此 TS2 集成轴双向关闭（r2d 成员侧 + plana 桶侧），0.8604 终判成立（七战线如实报告）。"
         "预印本 v3.25/v3.26：plana 轴入稿 + Data/Code availability 块 + 附录 D 图注，"
         "checker 8/8+9/9 全过。交付文件已同步本地。台账 §15.83–15.86，GitHub 0b49874。")

# clone shapes from src (order preserved), swapping texts
def clone_shape(sh):
    el = copy.deepcopy(sh._element)
    new.shapes._spTree.append(el)
    return new.shapes[-1]

title_done = head_done = foot_done = pg_done = concl_done = False
for sh in src.shapes:
    if not sh.has_text_frame:
        continue
    t = sh.text_frame.text
    if t.startswith("三种子家族成立"):
        c = clone_shape(sh); c.text_frame.text = TITLE; title_done = True
    elif t.startswith("新 s2 仅"):
        c = clone_shape(sh); c.text_frame.text = HEAD; head_done = True
    elif "2026-10-18" in t and "RNA-JEV" in t:
        c = clone_shape(sh); c.text_frame.text = "RNA-JEV · 浙大 · 2026-10-19"; foot_done = True
    elif t.strip() == "13":
        c = clone_shape(sh); c.text_frame.text = "14"; pg_done = True
    elif t.startswith("结论"):
        c = clone_shape(sh); c.text_frame.text = CONCL; concl_done = True

assert title_done and head_done and foot_done and pg_done and concl_done, \
    (title_done, head_done, foot_done, pg_done, concl_done)

p.save(P)
print("slide 14 added, saved", len(p.slides.__iter__.__self__._sldIdLst))
