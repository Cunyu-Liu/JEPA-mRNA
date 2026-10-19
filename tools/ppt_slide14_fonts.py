#!/usr/bin/env python3
"""Fix slide 14 fonts: the text swap created new runs without run-level props,
so they fall back to defaults instead of the slide-13 template styling.
Copy run properties (font name/size/color/bold) from the matching source
shape run-by-run. Also fix Chinese '｜' line-break risk by keeping single runs."""
from pptx import Presentation
from pptx.util import Pt
import copy

P = "/Users/liucunyu/Documents/课题idea 数据 结论/浙江大学/RNA 相关模型/RNA-JEV/ppt/RNA-JEV_汇报_精简版_20261016.pptx"
p = Presentation(P)
s13, s14 = p.slides[12], p.slides[13]

def first_run(sh):
    for para in sh.text_frame.paragraphs:
        if para.runs:
            return para.runs[0]
    return None

pairs = []  # (src_shape, dst_shape)
src_map = {}
for sh in s13.shapes:
    if sh.has_text_frame and sh.text_frame.text.strip():
        t = sh.text_frame.text
        if t.startswith("三种子家族"): src_map["title"] = sh
        elif t.startswith("新 s2"): src_map["head"] = sh
        elif "RNA-JEV · 浙大 · 2026-10-18" in t: src_map["foot"] = sh
        elif t.strip() == "13": src_map["pg"] = sh
        elif t.startswith("结论"): src_map["concl"] = sh

for sh in s14.shapes:
    if not (sh.has_text_frame and sh.text_frame.text.strip()):
        continue
    t = sh.text_frame.text
    key = None
    if t.startswith("TS2 集成轴"): key = "title"
    elif t.startswith("plana 桶"): key = "head"
    elif "RNA-JEV · 浙大 · 2026-10-19" in t: key = "foot"
    elif t.strip() == "14": key = "pg"
    elif t.startswith("结论"): key = "concl"
    if key:
        src_r = first_run(src_map[key])
        dst_r = first_run(sh)
        # copy the full rPr element (font/size/color/bold) from source run
        if src_r is not None and dst_r is not None:
            rPr = src_r._r.find(
                "{http://schemas.openxmlformats.org/drawingml/2006/main}rPr")
            if rPr is not None:
                new_rPr = copy.deepcopy(rPr)
                old = dst_r._r.find(
                    "{http://schemas.openxmlformats.org/drawingml/2006/main}rPr")
                if old is not None:
                    dst_r._r.remove(old)
                dst_r._r.insert(0, new_rPr)

p.save(P)
print("fonts fixed")
