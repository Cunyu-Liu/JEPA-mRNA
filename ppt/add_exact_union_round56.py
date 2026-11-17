import hashlib
import json
from pathlib import Path

from pptx import Presentation

from add_evidence_audit_round56 import DECK, WORK, ORANGE, GRAY, blank, card, table, textbox, set_notes


def main():
    union = json.loads((WORK / "exact_union_summary.json").read_text())
    initial = json.loads((WORK / "exact_overlap.json").read_text())
    ensembl = json.loads((WORK / "ensembl_exact_overlap.json").read_text())
    assert union["status"] == "PASS_EXACT_UNION_WITH_NT_AND_HOMOLOGY_PENDING"
    assert initial["status"] == "PASS_EXACT_OVERLAP_WITH_NT_ENSEMBL_PENDING"
    assert ensembl["status"] == "PASS_ENSEMBL_EXACT_OVERLAP_WITH_NT_AND_HOMOLOGY_PENDING"
    central = initial["sources"]["rnacentral_22_active"]["queries"]
    rfam = initial["sources"]["rfam_14_9"]["queries"]
    ensembl_queries = ensembl["sources"]["ensembl_109_and_bacteria_56"]["queries"]
    prs = Presentation(DECK)
    if len(prs.slides) == 26:
        assert prs.slides[-1].shapes[1].text == "三源精确去重：仍不能称为未见集合"
        print(json.dumps({"slides": 26, "status": "ALREADY_PRESENT"}, ensure_ascii=False))
        return
    assert len(prs.slides) == 25
    slide = blank(prs, "PRETRAINING EXPOSURE AUDIT  |  ROUND 56",
                  "三源精确去重：仍不能称为未见集合",
                  "RNAcentral 22.0 · Rfam 14.9 · Ensembl 109/56 · 同名不同序列按源 FASTA 复核")
    entries = [("VL0", "bprna_vl0"), ("bpRNA-new", "bprna_new"), ("Rfam", "rfam_fam")]
    for index, (label, key) in enumerate(entries):
        card(slide, 0.65 + index * 4.15, 1.74, 3.75, f"{label}  三源去重",
             f"{union['queries'][key]['exact_overlap_unique_any_known_source']:,} / {union['queries'][key]['unique_sequences']:,}", ORANGE, 21)
    rows = [["集合", "RNAcentral", "Rfam 14.9", "Ensembl", "三源去重"]]
    for label, key in [("TR1c", "bprna_tr1c"), *entries, ("ncRNA test", "ncrna_test.csv"),
                       ("界面 holdout", "interface_holdout"), ("TS0", "bprna_ts0")]:
        rows.append([label,
                     f"{central[key]['exact_overlap_unique']:,}",
                     f"{rfam[key]['exact_overlap_unique']:,}",
                     f"{ensembl_queries[key]['exact_overlap_unique']:,}",
                     f"{union['queries'][key]['exact_overlap_unique_any_known_source']:,} / {union['queries'][key]['unique_sequences']:,}"])
    table(slide, rows, 0.65, 3.10, [1.75, 2.6, 2.5, 2.35, 2.8], 0.39, 10)
    textbox(slide, 0.65, 6.36, 12.0, 0.36,
            "联合值按完整序列去重，不是三列相加；同源搜索与家族映射尚未齐套，剩余记录仍为未知而非 clean。", 11, ORANGE, True)
    textbox(slide, 0.65, 6.77, 12.0, 0.31,
            "同名不同序列直接回查源 FASTA；nt 历史快照未确定，长序列还受训练随机裁剪影响。", 10, GRAY)
    textbox(slide, 0.65, 7.13, 12.0, 0.12,
            f"exact_union_summary.json  ·  SHA256 {hashlib.sha256((WORK / 'exact_union_summary.json').read_bytes()).hexdigest()[:12]}…  |  仅精确暴露，不是完整污染率", 8, GRAY)
    set_notes(prs, slide, (
        "这一页在上一页单源审计基础上做三源去重。VL0精确重叠193/196，bpRNA-new为5381/5388，"
        "Rfam为10327/10790，TS0为1253/1288，ncRNA test为24912/25038。"
        "表格同时保留RNAcentral、Rfam和Ensembl各自的命中数，联合值不是三列相加。"
        "发现plk等名称对应不同序列后，我们没有按名称强行合并，而是直接回查源FASTA并逐源对账。"
        "这些只是已核验公开源的精确匹配，不是完整预训练污染率；nt、完整同源和家族分析仍未闭环。"
        "原始模型结果保留为污染感知评估，不能称为未见家族或独立泛化证明。"
    ))
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "deck": str(DECK), "status": "ADD_EXACT_UNION_EVIDENCE"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
