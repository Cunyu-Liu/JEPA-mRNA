#!/usr/bin/env python3
import argparse
from pathlib import Path

from pptx import Presentation
from pptx.util import Pt

from patch_round36 import normalize_font_and_spacing, replace_text, set_notes, set_shape_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prs = Presentation(args.deck)
    assert len(prs.slides) == 23

    slide = prs.slides[3]
    title = next(shape for shape in slide.shapes if shape.has_text_frame and shape.text.startswith("主图 1｜"))
    set_shape_text(
        title,
        "主图 1｜统一 xens2：全板敏感性 5/8；最佳单 checkpoint 4/8；bpRNA-new、TS2、ArchiveII 未领先。",
        size=18.5,
        bold=True,
    )
    note = next(shape for shape in slide.shapes if shape.has_text_frame and shape.text.startswith("new："))
    set_shape_text(
        note,
        "new：EternaFold 0.6854 领先；TS2：MXfold2 0.8991 领先；ArchiveII：Kirigami 0.7872 > xens2 0.7760。预注册 clean 参照 6/8 仅作实验合同附注。",
        size=8.0,
        bold=False,
    )
    set_notes(
        slide,
        "统一 xens2 全板敏感性为 5/8，最佳单 checkpoint 为 4/8。bpRNA-new、TS2、ArchiveII 均未领先：EternaFold=0.6854、MXfold2=0.8991、Kirigami=0.7872 分别构成当前全板强对手。预注册 clean 参照 6/8 只用于说明实验合同，不得作为 SOTA 主标题。",
    )

    table = next(shape.table for shape in prs.slides[15].shapes if shape.has_table)
    table.cell(1, 1).text = "统一 xens2 全板敏感性 5/8；最佳单 checkpoint 4/8"
    table.cell(1, 2).text = "预注册 clean 参照 6/8 仅作合同附注；bpRNA-new、TS2、ArchiveII 未领先"

    replace_text(
        prs.slides[21],
        "xens2 预注册参照 6/8；最佳单 4/8",
        "xens2 全板敏感性 5/8；最佳单 4/8；bpRNA-new、TS2、ArchiveII 未领先",
    )
    replace_text(
        prs.slides[21],
        "TS2 未胜；旧 7/8 是混合组合",
        "预注册参照 6/8 仅作附注；旧 7/8 是混合组合",
    )
    replace_text(
        prs.slides[22],
        "统一 xens2 胜预注册参照 6/8，不等于全板 6/8 SOTA。",
        "统一 xens2 全板敏感性 5/8；最佳单 checkpoint 4/8；bpRNA-new、TS2、ArchiveII 未领先。",
    )
    normalize_font_and_spacing(prs)
    for run in title.text_frame.paragraphs[0].runs:
        run.font.size = Pt(18.5)
        run.font.bold = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.out)
    print(f"ROUND39_TITLE_DECK_WRITTEN {args.out}")


if __name__ == "__main__":
    main()
