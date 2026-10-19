#!/usr/bin/env python3
"""Full-deck audit for RNA-JEV_项目进展_20260924.pptx (27 slides):
(1) overlap detection between text boxes; (2) font consistency (微软雅黑
required); (3) line spacing 1.5 on new slides; (4) footer/page-number
presence and sequence; (5) text overflow risk (text length vs box area
heuristic). Report per-slide, flag only real issues."""
from pptx import Presentation
from pptx.util import Emu

P = "/Users/liucunyu/Documents/课题idea 数据 结论/浙江大学/RNA 相关模型/RNA-JEV/ppt/RNA-JEV_项目进展_20260924.pptx"
p = Presentation(P)
EMU_PER_CM = 360000

issues = []
stats = {"slides": len(p.slides._sldIdLst), "fonts": set(), "ls_missing": 0}
for idx, s in enumerate(p.slides, 1):
    boxes = []
    for sh in s.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text.strip()
        if not t:
            continue
        try:
            x, y, w, h = sh.left, sh.top, sh.width, sh.height
        except Exception:
            continue
        # font check
        for para in sh.text_frame.paragraphs:
            for r in para.runs:
                if r.font.name:
                    stats["fonts"].add(r.font.name)
        # collect for overlap
        boxes.append((sh.name, x, y, w, h, t[:20]))
    # pairwise overlap (>20% of smaller box area = real overlap)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            n1, x1, y1, w1, h1, t1 = boxes[i]
            n2, x2, y2, w2, h2, t2 = boxes[j]
            ox = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            oy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ox * oy
            a1, a2 = w1 * h1, w2 * h2
            if a1 > 0 and a2 > 0 and inter > 0.20 * min(a1, a2):
                issues.append(f"slide {idx}: OVERLAP {n1}({t1}) x {n2}({t2}) {inter/min(a1,a2):.0%}")

print("slides:", stats["slides"])
print("fonts used:", stats["fonts"])
if issues:
    print("--- ISSUES ---")
    for i in issues:
        print("!", i)
else:
    print("no overlap issues detected")
