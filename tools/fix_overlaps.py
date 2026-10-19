#!/usr/bin/env python3
"""Fix box-level overlaps on slides 26/27 (new slides only — user-edited
pages untouched):
- T1 height shrink to single-line (15pt*1.5 ≈ 285750 EMU > gap), so T1 top
  stays but height becomes 250000 and T2 moved down to 700000 for clear
  separation band.
- T5 (footer) width shrink to 8000000 so it no longer overlaps page number.
Slide 16/20 overlaps are pre-existing (user's own pages with text auto-fit)
— not touched per user instruction."""
from pptx import Presentation
from pptx.util import Emu

P = "/Users/liucunyu/Documents/课题idea 数据 结论/浙江大学/RNA 相关模型/RNA-JEV/ppt/RNA-JEV_项目进展_20260924.pptx"
p = Presentation(P)

s26 = p.slides[25]
for sh in s26.shapes:
    if sh.name == "T1":
        sh.height = Emu(250000)
    elif sh.name == "T2":
        sh.top = Emu(700000)
    elif sh.name == "T5":
        sh.width = Emu(8000000)

s27 = p.slides[26]
for sh in s27.shapes:
    if sh.name == "T1":
        sh.height = Emu(250000)
    elif sh.name == "T2":
        sh.top = Emu(700000)
    elif sh.name == "F1":
        sh.width = Emu(8000000)

p.save(P)
print("overlaps fixed on slides 26/27")
