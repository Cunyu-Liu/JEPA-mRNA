import hashlib
import json
from pathlib import Path

from pptx import Presentation

from add_evidence_audit_round56 import DECK, WORK, ORANGE, GRAY, blank, card, table, textbox, set_notes


def main():
    union = json.loads((WORK / "homology_union_summary.json").read_text())
    qa = json.loads((WORK / "homology_union_independent_qa.json").read_text())
    assert union["status"] == "PASS_KNOWN_SOURCE_HOMOLOGY_UNION_NT_AND_FAMILY_LIMITS_REMAIN"
    assert qa["status"] == "PASS_HOMOLOGY_UNION_INDEPENDENT_QA"
    assert hashlib.sha256((WORK / "homology_union_summary.json").read_bytes()).hexdigest() == qa["summary_sha256"]
    assert union["n_queries"] == 165854 and union["detected_by_any_source"] == 162489
    assert union["not_detected_by_any_source"] == 3365
    prs = Presentation(DECK)
    if len(prs.slides) == 28:
        assert prs.slides[-1].shapes[1].text == "三源同源联合：未检出也不等于未见"
        print(json.dumps({"slides": 28, "status": "ALREADY_PRESENT"}, ensure_ascii=False))
        return
    assert len(prs.slides) == 27
    slide = blank(prs, "KNOWN-SOURCE HOMOLOGY UNION  |  ROUND 56",
                  "三源同源联合：未检出也不等于未见",
                  "165,854 条唯一查询 · 80% identity＋双向80%覆盖 · 逐查询去重与独立复核")
    card(slide, 0.65, 1.74, 3.75, "三源联合检出", "162,489 / 165,854", ORANGE, 20)
    card(slide, 4.80, 1.74, 3.75, "VL0 至少一源检出", "196 / 196", ORANGE, 22)
    card(slide, 8.95, 1.74, 3.75, "bpRNA-new 至少一源检出", "5,388 / 5,388", ORANGE, 19)
    rows = [["集合", "RNAcentral", "Rfam 14.9", "Ensembl", "三源去重"]]
    for label, key in [("TR1c", "bprna_tr1c"), ("VL0", "bprna_vl0"), ("bpRNA-new", "bprna_new"),
                       ("Rfam", "rfam_fam"), ("ncRNA test", "ncrna_test.csv"),
                       ("界面 holdout", "interface_holdout"), ("TS0", "bprna_ts0")]:
        item = union["queries"][key]
        rows.append([label, f"{item['detected_by_source']['rnacentral_22_active']:,}",
                     f"{item['detected_by_source']['rfam_14_9']:,}",
                     f"{item['detected_by_source']['ensembl_109_and_bacteria_56']:,}",
                     f"{item['detected_by_any_source']:,} / {item['n_queries']:,}"])
    table(slide, rows, 0.65, 3.10, [1.75, 2.6, 2.5, 2.35, 2.8], 0.39, 10)
    textbox(slide, 0.65, 6.36, 12.0, 0.36,
            "3,365 条未检出仍是未知，不是 clean；nt 历史快照未确定，不能用已知来源反推无暴露。", 11, ORANGE, True)
    textbox(slide, 0.65, 6.77, 12.0, 0.31,
            "Rfam 与 TR1c 有 120 个可映射共享家族；bpRNA-new 的 0 仅指下游家族标签，不抵消预训练暴露。", 10, GRAY)
    textbox(slide, 0.65, 7.13, 12.0, 0.12,
            f"homology_union_summary.json  ·  SHA256 {qa['summary_sha256'][:12]}…  |  同源审计，不是模型精度结果", 8, GRAY)
    set_notes(prs, slide, (
        "这一页是三来源80%同源联合，不是精确匹配或模型精度结果。"
        "全部165854条查询中162489条至少在一个来源检出，VL0和bpRNA-new都全部至少在一个来源有同源匹配。"
        "Rfam与TR1c有120个可映射共享家族；bpRNA-new的家族交集0只支持下游标签划分，不抵消预训练暴露。"
        "未检出的3365条仍是未知，不是clean。MMseqs2是启发式搜索，且每条查询最多保留300个target，"
        "nt历史快照未确定，家族标签也不等于序列同源性，因此不能形成完全未见或完整独立泛化结论。"
    ))
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "deck": str(DECK), "status": "ADD_KNOWN_SOURCE_HOMOLOGY_UNION"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
