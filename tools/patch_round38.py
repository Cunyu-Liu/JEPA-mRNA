#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

from patch_round36 import SPLITS, normalize_font_and_spacing, replace_text, set_notes, set_shape_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    assert source["status"] == "active_leakage_audit_and_clean_subset_table"
    prs = Presentation(args.deck)
    assert len(prs.slides) == 23

    slide = prs.slides[3]
    replace_text(slide, "主图 1｜统一模型口径：最佳单 checkpoint 4/8，xens2 为 6/8",
                 "主图 1｜xens2 胜预注册 clean 参照 6/8；不等于全板 6/8 SOTA")
    note = next(shape for shape in slide.shapes if shape.has_text_frame and shape.text.startswith("r2d_tr1 仅 3 格"))
    set_shape_text(note,
                   "new：EternaFold 0.6854 > MXfold2 0.6688 > eFold 0.6640 > UFold 0.6106；TS2：MXfold2 0.8991、ViennaRNA-mfe 0.8980，均高于 xens2。UFold 仅是预注册参照。",
                   size=8.2, bold=False)
    note.top, note.height = Inches(6.76), Inches(0.34)
    set_notes(slide,
              "6/8 只表示统一 xens2 对预注册 clean 参照的胜场，不是全板 SOTA。bpRNA-new 全板最高为 EternaFold 0.6854，UFold 0.6106 仅为预注册参照；TS2 的 MXfold2 为 0.8991，ViennaRNA-mfe 为 0.8980，xens2 为 0.8604。")

    for index, pending in [(4, "五模型扩展测速进行中；未完成前"), (8, "五模型扩展测速进行中；未完成前")]:
        slide = prs.slides[index]
        for shape in slide.shapes:
            if shape.has_text_frame and "UFold 未测" in shape.text:
                replace_text(slide, "UFold 未测", "eFold/UFold/BPfold/E2Efold/EternaFold 扩展测速中")
        set_notes(slide, prs.slides[index].notes_slide.notes_text_frame.text +
                  "\n新增五模型只接受官方权重、同机同数据、raw-sequence 端到端三次完整测速；当前不得把缺失项当成已验收结果。")

    slide = prs.slides[17]
    replace_text(slide, "附录 A2 / 泄露处理", "附录 A2 / 泄露审计")
    replace_text(slide, "泄露处理全板：污染格剔除，TR0 系 TestSetB 统一 clean-181",
                 "泄露审计与 clean 子集复算：仅已复算格可称去重叠")
    replace_text(slide,
                 "19 系统 × 8 split，与附录 A 原始全板对应；×=污染剔除，—=缺测/未复算，均不按 0 分。",
                 "19 系统 × 8 split；逐模型完整去泄露复测尚未完成，沿用原板的格子不得解释为已去污染。")
    table = next(shape.table for shape in slide.shapes if shape.has_table)
    assert len(table.rows) == len(source["rows"]) + 1
    for row_index, row in enumerate(source["rows"], start=1):
        if row.get("model_version"):
            name_runs = table.cell(row_index, 0).text_frame.paragraphs[0].runs
            name_runs[0].text = f"eFold {row['model_version']} (Sci Adv 26)"
            for run in name_runs[1:]:
                run.text = ""
        for col_index, split in enumerate(SPLITS, start=1):
            target = table.cell(row_index, col_index)
            runs = target.text_frame.paragraphs[0].runs
            if runs:
                runs[0].text = row["cells"][split]["display"]
                for run in runs[1:]:
                    run.text = ""
            else:
                target.text = row["cells"][split]["display"]
        source_cell = table.cell(row_index, 9)
        statuses = {cell["status"] for cell in row["cells"].values()}
        if row["system"] in {"ViennaRNA mfe", "ViennaRNA centroid", "ViennaRNA mea", "Nussinov+Turner"}:
            label = "非序列拟合"
        elif "clean_subset_recomputed" in statuses:
            label = "部分复算"
        else:
            label = "原板○"
        source_runs = source_cell.text_frame.paragraphs[0].runs
        source_runs[0].text = label
        for run in source_runs[1:]:
            run.text = ""
        for col_index, cell in enumerate(table.rows[row_index].cells):
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(7.2 if row_index == 19 and col_index == 0 else 8)
    note = next(shape for shape in slide.shapes if shape.has_text_frame and shape.text.startswith("†=TestSetB"))
    set_shape_text(note,
                   "†=clean 复算（RNAformer TS1/TS2/TS3/hard：n=25/11/4/8；TSB：n=181）；×=重叠未复算；—=缺测/未复算。○=沿用原板、未逐模型去污染；‡=特殊 checkpoint/审计下限。本页不是完整去泄露复测。",
                   size=8.2, bold=False)
    note.top, note.height = Inches(6.38), Inches(0.44)
    set_notes(slide,
              "本页不是逐模型完整去泄露复测。RNAformer-bprna 的 TS1/TS2/TS3/hard 已用原始 DBN 对精确训练重叠剔除后的子集重算，n=25/11/4/8，F1=0.7830/0.8506/0.8296/0.6933。ViennaRNA 没有训练拟合的序列模型，TS2 高值不构成泄露证据。其他沿用原板的格子仍需逐模型审计，不能以高分推定污染。")

    replace_text(prs.slides[13],
                 "xens2 OOD mean 0.4434 vs OFoldNA-SS 0.4410，仅写并列；GenF=0.3066 是弱项，不用均值掩盖。",
                 "GenA=0.6443 已领先；GenC=0.3793、GenF=0.3066 是短板，OOD mean=0.4434 仅并列；20.8%/39.2% 为零正确配对率，无新提升证据。")
    replace_text(prs.slides[18], "UFold 未测，不宣称绝对 SOTA", "五模型扩展测速中，不宣称绝对 SOTA")
    replace_text(prs.slides[20],
                 "MRL 与 HEK fold0–2 TE/EL、fold3 TE 已 QA，均不显著为正；fold3 EL 续训中，fold4–9 待跑；pc3 fold1 暂停（证据已归档）。",
                 "MRL 与 HEK fold0–3 TE/EL 已 QA，均不显著为正；HEK 仅 4 折，pc3 暂停，未满十折不进图。")
    replace_text(prs.slides[21], "xens2 6/8；最佳单 4/8", "xens2 预注册参照 6/8；最佳单 4/8")
    replace_text(prs.slides[21], "UFold 未测，不作绝对 SOTA", "五模型扩展测速中，不作绝对 SOTA")
    replace_text(prs.slides[21], "GenF 与 >400 nt 仍需改进", "GenC/GenF 零正确配对率 20.8%/39.2%；>400 nt 仍需改进")
    replace_text(prs.slides[21], "21", "22")
    replace_text(prs.slides[-1], "统一 xens2=6/8；历史混合集成不再冒充单一模型。",
                 "统一 xens2 胜预注册参照 6/8，不等于全板 6/8 SOTA。")
    replace_text(prs.slides[0], "2026-11-01 · Round 37", "2026-11-01 · Round 38")
    replace_text(prs.slides[-1], "RNA-JEV / ROUND 37", "RNA-JEV / ROUND 38")
    normalize_font_and_spacing(prs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.out)
    print(f"ROUND38_DECK_WRITTEN {args.out}")


if __name__ == "__main__":
    main()
