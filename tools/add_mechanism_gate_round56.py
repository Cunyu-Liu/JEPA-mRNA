import hashlib
import json

from pptx import Presentation

from add_evidence_audit_round56 import DECK, WORK, ORANGE, BLUE, TEAL, GRAY, blank, card, table, textbox, set_notes

TITLE = "机制裁决：没有合格的新结构学习机制"
DECISION_SHA256 = "c4e77a4fa78a45b284eba7f4a2fd72bb04d04b98b824273964c3bcc9fbe44050"


def main():
    decision = WORK / "structure-mechanism-gate-v3.md"
    assert hashlib.sha256(decision.read_bytes()).hexdigest() == DECISION_SHA256
    assert "STOP_NO_JUSTIFIED_NEW_MECHANISM" in decision.read_text()
    pilot = json.loads((WORK / "pilot_v1_decision.json").read_text())
    motif = json.loads((WORK / "bpfold_decision.json").read_text())
    assert pilot["status"] == "STOP_PILOT_NEGATIVE_DEV_ONLY"
    assert motif["status"] == "STOP_BPFOLD_DEV_NEGATIVE" and not motif["external_evaluation_started"]
    prs = Presentation(DECK)
    if len(prs.slides) == 30:
        assert prs.slides[-1].shapes[1].text == TITLE
        print(json.dumps({"slides": 30, "status": "ALREADY_PRESENT"}, ensure_ascii=False))
        return
    assert len(prs.slides) == 29
    slide = blank(prs, "MECHANISM NOVELTY GATE  |  ROUND 56",
                  TITLE,
                  "直接配对监督已存在 · 独立新信息才值得试验 · 不拿换包装当创新")
    card(slide, 0.65, 1.74, 3.75, "H0 无 JEPA 宿主 · macro-F1", f"{pilot['results']['H0']['macro_f1']:.6f}", BLUE)
    card(slide, 4.80, 1.74, 3.75, "H2 双视图 · macro-F1", f"{pilot['results']['H2']['macro_f1']:.6f}", ORANGE)
    card(slide, 8.95, 1.74, 3.75, "motif − shuffle", f"{motif['p1_minus_shuffled']:+.7f}", TEAL, 19)
    table(slide, [
        ["候选做法", "为何不启动新 pilot"],
        ["换 mask / 查询 token / EMA / 损失权重", "仍是同一潜在预测与相同标签，不构成新机制"],
        ["Turner / motif / 旧 JEPA / 投票", "已有负结果或明确停止边界，不改名恢复"],
        ["扩数据 / 扩审计 / 换蒸馏对象", "改变规模或形式，不能证明独立结构收益"],
        ["三维 / 假结 / 复合物环境", "canonical-nested 标签不足，不能虚构标签范围"],
    ], 0.65, 3.16, [4.35, 7.65], 0.56, 10.5)
    textbox(slide, 0.65, 6.12, 12.0, 0.42,
            "STOP_NO_JUSTIFIED_NEW_MECHANISM：不是结构无效的全称否定，而是没有独立、可检验的新机制。", 11, ORANGE, True)
    textbox(slide, 0.65, 6.63, 12.0, 0.37,
            "预训练审计只约束未见/OOD，不阻塞同初始化 pilot；下游去重与标签 QA 仍是前置。", 10, GRAY)
    textbox(slide, 0.65, 7.11, 12.0, 0.14,
            f"structure-mechanism-gate-v3.md  ·  SHA256 {DECISION_SHA256[:12]}…4050  |  当前不启动新训练", 8, GRAY)
    set_notes(prs, slide, (
        "这一页说明为什么不继续堆结构模块。H0本身已有配对监督、ResNet2D、Turner和精确DP，"
        "所以新辅助目标必须提供独立的新信息或归纳偏置，而不是同一标签换一种形式。"
        "H2为0.844264，低于H0的0.851460；motif也没有超过shuffle，不能用调参或新数据挽救。"
        "换mask、查询token、EMA、蒸馏对象、扩数据和旧JEPA都不构成新机制。"
        "预训练审计只约束未见/OOD表述，不阻塞同初始化pilot；下游去重与标签QA仍是前置。"
        "未来只有独立数学命题、可证伪假设、匹配标签范围、同初始化对照、序列-only推理和停止审计扩张全部满足时才重评。"
    ))
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "deck_sha256": hashlib.sha256(DECK.read_bytes()).hexdigest(),
                      "status": "ADD_MECHANISM_NOVELTY_GATE"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
