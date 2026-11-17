import hashlib
import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

from add_evidence_audit_round56 import DECK, WORK, TEAL, ORANGE, GRAY, blank, card, table, textbox, set_notes


def fix_existing_layout(prs):
    slide = prs.slides[-1]
    tables = [shape for shape in slide.shapes if shape.has_table]
    assert len(tables) == 1
    tables[0].top = Inches(3.10)
    for row in tables[0].table.rows:
        row.height = Inches(0.39)
    positions = {"下游去重叠": 6.36, "nt 历史快照": 6.77, "exact_overlap.json": 7.13}
    for prefix, top in positions.items():
        matches = [shape for shape in slide.shapes if shape.has_text_frame and shape.text.startswith(prefix)]
        assert len(matches) == 1
        matches[0].top = Inches(top)
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "status": "LAYOUT_QA_FIXED"}, ensure_ascii=False))


def main():
    report = json.loads((WORK / "exact_overlap.json").read_text())
    assert report["status"] == "PASS_EXACT_OVERLAP_WITH_NT_ENSEMBL_PENDING"
    central = report["sources"]["rnacentral_22_active"]
    rfam = report["sources"]["rfam_14_9"]
    prs = Presentation(DECK)
    if len(prs.slides) == 25:
        fix_existing_layout(prs)
        return
    assert len(prs.slides) == 24
    slide = blank(prs, "PRETRAINING EXPOSURE AUDIT  |  ROUND 56",
                  "预训练精确暴露：不能称为未见集合",
                  "RNAcentral 22.0 · Rfam 14.9 · 官方版本文件逐序列核对 · 阳性接触已成立")
    entries = [("VL0", "bprna_vl0"), ("bpRNA-new", "bprna_new"), ("Rfam", "rfam_fam"),
               ("ncRNA test", "ncrna_test.csv"), ("界面 holdout", "interface_holdout")]
    for index, (label, key) in enumerate(entries[:3]):
        item = central["queries"][key]
        card(slide, 0.65 + index * 4.15, 1.74, 3.75, f"{label}  精确重叠",
             f"{item['exact_overlap_unique']:,} / {item['unique_sequences']:,}", ORANGE, 21)
    rows = [["集合", "RNAcentral", "Rfam 14.9", "解释边界"]]
    for label, key in [("TR1c", "bprna_tr1c"), *entries, ("TS0", "bprna_ts0")]:
        left = central["queries"][key]
        right = rfam["queries"][key]
        rows.append([label,
                     f"{left['exact_overlap_unique']:,} / {left['unique_sequences']:,}  ({left['exact_overlap_unique'] / left['unique_sequences']:.1%})",
                     f"{right['exact_overlap_unique']:,} / {right['unique_sequences']:,}  ({right['exact_overlap_unique'] / right['unique_sequences']:.1%})",
                     "精确序列接触，不是家族推断"])
    table(slide, rows, 0.65, 3.10, [2.0, 3.5, 3.5, 3.0], 0.39, 10)
    textbox(slide, 0.65, 6.36, 12.0, 0.36,
            "下游去重叠 ≠ RiNALMo 预训练去污染。原始分数可保留，但不能称为 clean-OOD 或独立泛化证明。", 11, ORANGE, True)
    textbox(slide, 0.65, 6.77, 12.0, 0.31,
            "nt 历史快照、完整语料与 80% 同源/家族审计仍未闭环；阳性精确匹配不需要等待这些工作即可成立。", 10, GRAY)
    textbox(slide, 0.65, 7.13, 12.0, 0.12,
            f"exact_overlap.json  ·  SHA256 {hashlib.sha256((WORK / 'exact_overlap.json').read_bytes()).hexdigest()[:12]}…  |  后续按污染感知评估", 8, GRAY)
    set_notes(prs, slide, (
        "这一页是项目最重要的数据边界。我们直接扫描了官方RNAcentral22.0和Rfam14.9文件，"
        "发现VL0有184/196、bpRNA-new有5303/5388、Rfam有9989/10790条精确序列重叠。"
        "ncRNA test和界面holdout也有大量接触，因此不能把这些集合称为未见集合。"
        "阳性精确匹配本身就足以证明预训练接触，不需要等待nt或同源聚类完成。"
        "下游去重叠仍然有价值，但不能替代预训练去污染。原始模型结果保留为污染感知评估，"
        "新架构的未见家族精度主张目前不成立。后续会补齐同源和家族分层，不用部分来源反推无接触。"
    ))
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "deck": str(DECK), "status": "ADD_EXACT_PRETRAIN_EXPOSURE"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
