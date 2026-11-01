#!/usr/bin/env python3
from pathlib import Path

from pptx import Presentation
from patch_round36 import normalize_font_and_spacing, replace_text, set_notes

src = Path("ppt/RNA-JEV_论文逻辑重构_20261031_导师讨论_round38.pptx")
out = Path("ppt/.round39_work/round39_contrafold_note.pptx")
prs = Presentation(src)
slide = prs.slides[3]
replace_text(
    slide,
    "new：EternaFold 0.6854 领先；",
    "new：CONTRAfold 0.6968、EternaFold 0.6854 均高于 xens2；",
)
set_notes(
    slide,
    "统一 xens2 全板敏感性为 5/8，最佳单 checkpoint 为 4/8。bpRNA-new、TS2、ArchiveII 均未领先：CONTRAfold=0.6968、EternaFold=0.6854、MXfold2=0.8991、Kirigami=0.7872 分别构成当前全板强对手。预注册 clean 参照 6/8 只用于说明实验合同，不得作为 SOTA 主标题。",
)
normalize_font_and_spacing(prs)
prs.save(out)
print(f"ROUND39_CONTRAFOLD_NOTE_WRITTEN {out}")
