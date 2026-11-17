import hashlib
import json
from pathlib import Path

from pptx import Presentation

from add_evidence_audit_round56 import DECK, WORK, BLUE, TEAL, ORANGE, GRAY, blank, card, table, textbox, set_notes


def main():
    rfam = json.loads((WORK / "rfam_homology_summary.json").read_text())
    ensembl = json.loads((WORK / "ensembl_homology_summary.json").read_text())
    families = json.loads((WORK / "rfam14_9_exact_sequence_family_mapping_summary.json").read_text())
    rfam_qa = json.loads((WORK / "rfam_homology_independent_qa.json").read_text())
    ensembl_qa = json.loads((WORK / "ensembl_homology_independent_qa.json").read_text())
    family_qa = json.loads((WORK / "rfam14_9_family_mapping_qa.json").read_text())
    verified = {"rfam_homology_summary.json": rfam_qa,
                "ensembl_homology_summary.json": ensembl_qa,
                "rfam14_9_exact_sequence_family_mapping_summary.json": family_qa}
    for filename, qa in verified.items():
        assert qa["status"].startswith("PASS")
        assert hashlib.sha256((WORK / filename).read_bytes()).hexdigest() == qa["summary_sha256"]
    prs = Presentation(DECK)
    if len(prs.slides) == 27:
        assert prs.slides[-1].shapes[1].text == "同源与家族证据：不能把未知写成未见"
        print(json.dumps({"slides": 27, "status": "ALREADY_PRESENT"}, ensure_ascii=False))
        return
    assert len(prs.slides) == 26
    slide = blank(prs, "HOMOLOGY & FAMILY AUDIT  |  ROUND 56",
                  "同源与家族证据：不能把未知写成未见",
                  "MMseqs2 80% identity＋双向80%覆盖 · 官方Rfam精确序列家族映射 · 独立逐查询复核")
    card(slide, 0.65, 1.73, 3.75, "Rfam 同源检出 / 全部查询", "155,635", BLUE, 22)
    card(slide, 4.80, 1.73, 3.75, "Ensembl 同源检出 / 全部查询", "61,092", TEAL, 20)
    card(slide, 8.95, 1.73, 3.75, "Rfam ∩ TR1c 可映射家族", "120", ORANGE, 23)
    rows = [["集合", "Rfam 同源", "Ensembl 同源", "唯一官方家族", "与 TR1c 共享"]]
    for label, key in [("Rfam", "rfam_fam"), ("bpRNA-new", "bprna_new"), ("TS0", "bprna_ts0"),
                       ("VL0", "bprna_vl0"), ("ncRNA test", "ncrna_test.csv"), ("界面 holdout", "interface_holdout")]:
        left = rfam["queries"][key]
        right = ensembl["queries"][key]
        family = families["datasets"].get(key)
        rows.append([label,
                     f"{left['detected_homology_unique']:,} / {left['n_queries']:,}",
                     f"{right['detected_homology_unique']:,} / {right['n_queries']:,}",
                     f"{family['mapped_unique_sequences']:,}" if family else "—",
                     str(len(families["mapped_family_overlap_with_tr1c"][key])) if family else "—"])
    table(slide, rows, 0.65, 3.13, [1.65, 2.4, 2.4, 2.75, 2.8], 0.43, 9.5)
    textbox(slide, 0.65, 6.24, 12.0, 0.32,
            "bpRNA-new 的可映射家族交集为 0，仅支持下游划分，不否定预训练暴露。", 10.5, ORANGE, True)
    textbox(slide, 0.65, 6.62, 12.0, 0.36,
            "Rfam 与 TR1c 有 120 个可映射共享家族；未检出不等于无同源，RNAcentral 与 nt 仍未闭环。", 10.5, ORANGE, True)
    textbox(slide, 0.65, 7.01, 12.0, 0.27,
            "界面 holdout 在长度过滤后为 198 条；缺元数据与多家族歧义不强行赋标签。", 10, GRAY)
    textbox(slide, 0.65, 7.29, 12.0, 0.12,
            f"family_mapping_qa  ·  {family_qa['summary_sha256'][:12]}…  |  非同源联合、非精度结论", 8, GRAY)
    set_notes(prs, slide, (
        "这一页分别报告Rfam和Ensembl的同源检出，不做跨来源联合，也不当作模型精度结果。"
        "Rfam有155635条检出，Ensembl有61092条；未检出不等于没有同源。"
        "官方精确序列家族映射中，Rfam与TR1c共享120个家族，bpRNA-new交集为0。"
        "这个0只说明可映射的下游标签划分，不能抵消前面展示的预训练精确暴露。"
        "缺元数据、歧义和长度过滤的记录均保留，RNAcentral同源与nt历史快照仍未闭环。"
    ))
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "deck": str(DECK), "status": "ADD_HOMOLOGY_AND_FAMILY_EVIDENCE"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
