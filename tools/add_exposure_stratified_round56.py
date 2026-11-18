import hashlib
import json

from pptx import Presentation

from add_evidence_audit_round56 import DECK, WORK, ORANGE, BLUE, TEAL, GRAY, blank, card, table, textbox, set_notes

EXPECTED_SHA256 = "c1c39477b84c3a139a52a091f1011548b1841c50bf6f5e572dc56b75318007ec"
TITLE = "冻结预测的暴露分层：没有同源未检出对照"


def main():
    artifact = WORK / "xens2_exposure_stratified.json"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == EXPECTED_SHA256
    audit = json.loads(artifact.read_text())
    assert audit["status"] == "PASS_FROZEN_EXPOSURE_STRATIFIED_WITH_NO_HOMOLOGY_UNKNOWN_COMPARISON"
    assert audit["n_sequences"] == 5388 and audit["n_skipped"] == 0
    assert audit["statistical_test"] is None and not audit["primary_comparison"]["available"]
    assert audit["homology_strata"]["not_detected"]["n"] == 0
    prs = Presentation(DECK)
    if len(prs.slides) == 29:
        assert prs.slides[-1].shapes[1].text == TITLE
        print(json.dumps({"slides": 29, "status": "ALREADY_PRESENT"}, ensure_ascii=False))
        return
    assert len(prs.slides) == 28
    slide = blank(prs, "FROZEN PREDICTION EXPOSURE AUDIT  |  ROUND 56",
                  TITLE,
                  "原 xens2 · P0 历史/重放逐序列一致 · bpRNA-new 5,388 条全量、无跳过 · 无重新训练")
    card(slide, 0.65, 1.74, 3.75, "全量 macro-F1", f"{audit['independent_full_metrics']['macro_f1']:.6f}", BLUE)
    card(slide, 4.80, 1.74, 3.75, "精确暴露组  n=5,381", f"{audit['exact_overlap_strata']['detected']['macro_f1']:.6f}", TEAL)
    card(slide, 8.95, 1.74, 3.75, "精确未检出组  n=7", f"{audit['exact_overlap_strata']['not_detected']['macro_f1']:.6f}", ORANGE)
    rows = [["分层", "n", "micro-F1", "precision", "recall", "解释"]]
    for label, group, explanation in [
        ("全量", audit["independent_full_metrics"], "冻结预测复核"),
        ("精确暴露", audit["exact_overlap_strata"]["detected"], "已检出，不等于完整暴露"),
        ("精确未检出", audit["exact_overlap_strata"]["not_detected"], "未知，不是 clean"),
        ("同源未检出", audit["homology_strata"]["not_detected"], "无样本，不能比较"),
    ]:
        rows.append([label, group["n"], f"{group['micro_f1']:.6f}",
                     f"{group['precision']:.6f}", f"{group['recall']:.6f}", explanation])
    table(slide, rows, 0.65, 3.15, [1.65, 0.75, 1.55, 1.55, 1.45, 5.05], 0.62, 10.5)
    textbox(slide, 0.65, 6.31, 12.0, 0.36,
            "5,388 / 5,388 均至少在一个已知来源检出 ≥80% 同源；同源未检出 n=0，不做显著性或因果结论。", 11, ORANGE, True)
    textbox(slide, 0.65, 6.75, 12.0, 0.31,
            "7 条精确未检出仅是探索性描述；nt 历史快照未固定。原 xens2 仍是历史基线，不是精度突破。", 10, GRAY)
    textbox(slide, 0.65, 7.13, 12.0, 0.12,
            f"xens2_bprna_new.json  ·  SHA256 {EXPECTED_SHA256[:12]}…07ec  |  只读分层，不选模型、不改阈值", 8, GRAY)
    set_notes(prs, slide, (
        "这一页是对原xens2冻结预测的只读暴露分层，不是新训练或精度提升。"
        "三个checkpoint的实际哈希和P0一致，历史与重放逐序列完全一致；5388条全量没有跳过。"
        "全量macro-F1为0.542209，精确暴露组为0.542424，精确未检出只有7条，macro-F1为0.377253。"
        "7条仍是未知而不是clean，组间序列构成不同，不做显著性检验或因果推断。"
        "更关键的是，所有查询至少在一个已知来源有80%同源匹配，同源未检出组为0，"
        "因此现有bpRNA-new无法构造同源未检出精度对照；nt历史快照仍未固定。"
    ))
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "deck": str(DECK), "status": "ADD_FROZEN_EXPOSURE_STRATIFICATION"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
