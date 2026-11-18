from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = "rna-mechanism-gate-v3"

V3 = """> **v3 执行覆盖（2026-11-18：机制创新裁决优先；活跃清单 `rna-mechanism-gate-v3`）**
>
> 本覆盖依据用户对研究优先级和投入产出的复核，停止继续扩展三源/时间外同源审计与新数据搜索。已完成结果只用于限制披露，不按结果挑样，不再无限补全；731条最终标签序列的三源搜索输出保留，不继续扩展审计或作为新架构精度证据。
>
> **新的硬顺序**：
>
> 1. **下游去重与标签 QA 是算法实验前置。**必须保留 train/dev/test 身份、序列/簇隔离、标签来源、坐标—序列映射、canonical-nested 配对守恒和排除原因。测试标签不得进入训练、视图构造、选臂、早停、校准或阈值选择。
> 2. **预训练审计只约束“未见/OOD/独立泛化”表述，不阻塞同初始化算法 pilot。**RiNALMo 只输入序列并不意味着结构标签泄露，但相同/同源序列接触会削弱“未见家族”解释。它不改变同 backbone、同初始化、同数据、同主损失比较的基本逻辑，也不得以继续审计掩盖模型负结果。不得宣称 clean-OOD；同初始化 pilot 不需要先穷尽预训练语料重叠。
> 3. **机制创新是任何后续 pilot 的硬门。**先用可检验的数学输入、表示、目标函数和推理路径说明与已否决双视图 JEPA 的实质区别，再决定是否实验。仅更换名称、mask 比例、损失权重、查询 token、训练预算、骨干、数据规模或解码方式，不构成新机制。D1 通用扩增、RIPE、旧局部 Pair-JEPA、19,000 nt beam 和三 R2D 投票不得改名恢复。
> 4. **不用新数据掩盖架构负结果。**原双视图 H2=0.8442637614 低于 H0=0.8514596293，BPfold motif 独立 dev 也为负；它们只覆盖已测配方，但不得通过新数据、调参、换指标或不确定叙事挽救为突破。原生欠训练、H2 别名和结构臂 OOM 的边界继续保留。
> 5. **证据等级不变。**`ENGINEERING_PASS`、model-dev 结果和冻结外部结果分开；真实 CUDA、同初始化、自然收敛、独立复算和负结果保留仍是硬要求。没有合格机制时停止，不为了 GPU 占用、训练时长或预印本叙事启动实验。
>
> v2 的完整预训练资产审计不再是后续 pilot 的阶段门，而是外部泛化声明的限制证据。以下 v2/v1 原文保留为历史设计与身份约束；凡与上述优先级冲突，以 v3 为准。
>
"""

TASK = "> | T-A235 | 结构机制创新裁决与后续 pilot 放行 | 数学机制与已否决双视图有实质区别；同初始化 pilot 不被完整预训练审计阻塞；无换名恢复 | 待裁决，未启动新 pilot。停止三源/时间外审计扩张和新数据搜索；已有731条标签序列及三源输出只作限制披露。没有真正不同的结构学习机制就不做，不用新数据掩盖H2与motif负结果 |\n"
CHECK_DONE = "> - [x] T-A235a 按用户裁决停止三源/时间外同源审计扩张及新数据搜索；已有结果仅作限制披露，不继续无限补全或据此挑样。\n"
CHECK_PENDING = "> - [ ] T-A235b 任何新 pilot 先证明数学机制不同于已否决双视图 JEPA；仅换名/调参/换数据不得放行。下游去重与标签QA仍为前置，完整预训练审计不再阻塞同初始化pilot，只约束未见/OOD表述。\n"
SPEC = "> **v3 优先级覆盖（2026-11-18）**：停止继续扩展三源/时间外同源审计和新数据搜索，已有结果只作限制披露。下游去重与标签QA仍是算法前置；预训练暴露只约束“未见/OOD/独立泛化”表述，不阻塞同初始化、同backbone、同数据、同主损失pilot。后续实验先证明数学机制与已否决双视图有实质区别，否则不做；不得用新数据或调参掩盖H2/motif负结果。完整历史审计不再作为算法阶段门，任何clean/OOD主张仍禁止。\n>\n"
CLAIM = """## v3 优先级裁决（2026-11-18）

停止继续扩展三源/时间外同源审计和新数据搜索，已有输出只作限制披露。预训练暴露影响的是“未见/OOD/独立泛化”解释，不是结构标签泄露，也不应阻塞同初始化算法比较。后续 pilot 必须先通过真正的机制创新门；没有不同于已否决双视图的数学机制就不做。下游去重与标签QA仍是前置，原xens2基线、H2与motif负结果、H2别名、欠训练和OOM边界全部保留，不用新数据或叙事挽救。

"""
LOG = """

## §15.962（2026-11-18）用户裁决：停止审计扩张，重置机制创新门

- 用户指出预训练只输入序列，未见二级结构标签，并质疑持续审计的投入产出。复核后明确：重叠分析关注的是相同/同源序列接触是否削弱“未见家族”解释，而不是结构标签泄露；它不应阻塞同初始化、同backbone的算法比较，也不能改变既有负结果。
- 停止继续扩展三源/时间外同源审计和新数据搜索。已有结果只作为限制披露，不再无限补全，不按结果挑样。731条最终标签序列的三源搜索输出已落盘，不再继续扩展审计或作为新架构精度证据。其精确暴露结果247/731，484未检出仍为未知；TR1c精确重叠44、VL0=0，仅保留边界，不用于选臂或调参。
- 新合同以`rna-mechanism-gate-v3`为当前执行覆盖：下游去重和标签QA仍是算法前置；预训练审计只约束“未见/OOD/独立泛化”表述，不阻塞同初始化pilot。旧D0完整预训练审计不再作为阶段门，clean/OOD措辞仍禁止。
- 后续pilot先要求数学输入、表示、目标函数和推理路径与已否决双视图存在实质区别。仅换名、换mask/损失权重、调预算、换骨干、扩数据或改编码方式不算新机制；D1、RIPE、旧Pair-JEPA、19,000 nt beam和投票不得改名恢复。没有真正不同的结构学习机制就不做，不为GPU占用或预印本叙事启动实验。
- 原双视图H2=0.8442637614低于H0=0.8514596293，BPfold motif也判负；原生欠训练、H2别名和结构臂OOM继续保留。不得用新数据掩盖负结果。工程通过、model-dev和外部结论继续分开，真实CUDA、自然收敛、独立复算和负结果保存不变。
"""


def insert_after_anchor(path, anchor, text, marker):
    content = path.read_text()
    if marker in content:
        return
    assert anchor in content, (path, anchor)
    path.write_text(content.replace(anchor, anchor + text, 1))


def main():
    contract = ROOT / "docs/plans/2026-11-16-dual-view-jepa-accuracy-contract.md"
    content = contract.read_text()
    if ACTIVE not in content:
        assert content.startswith("# 双视图")
        contract.write_text(content.replace("\n\n> **v2 执行覆盖", "\n\n" + V3 + "> **v2 执行覆盖", 1))
    for name in ["tasks.md", "spec/tasks.md", "spec_sync/tasks.md"]:
        path = ROOT / name
        text = path.read_text().replace("污染感知的数据与机制证据；活跃清单 `rna-evidence-v2`", "机制创新裁决优先；活跃清单 `rna-mechanism-gate-v3`")
        if "T-A235" not in text:
            lines = text.splitlines(keepends=True)
            index = next(i for i, line in enumerate(lines) if "T-A234" in line) + 1
            lines.insert(index, TASK)
            text = "".join(lines)
        text = text.replace("精确/同源复核进行中，未启动模型推理", "精确命中247/731、484未知；三源输出已归档，按v3停止扩展，未启动模型推理")
        path.write_text(text)
    for name in ["checklist.md", "spec/checklist.md", "spec_sync/checklist.md"]:
        path = ROOT / name
        text = path.read_text().replace("当前执行覆盖（2026-11-18：rna-evidence-v2）", "当前执行覆盖（2026-11-18：rna-mechanism-gate-v3）")
        if "T-A235a" not in text:
            anchor = "> **当前执行覆盖（2026-11-16：双视图 RNA-JEPA 精度主线"
            assert anchor in text
            text = text.replace(anchor, CHECK_DONE + CHECK_PENDING + "\n" + anchor, 1)
        text = text.replace("最终序列精确/同源门未完成，不启动模型推理。", "精确命中247/731，484未知；三源输出仅归档，按v3停止扩展，不启动模型推理。")
        path.write_text(text)
    for name in ["spec.md", "spec/spec.md", "spec_sync/spec.md"]:
        path = ROOT / name
        text = path.read_text().replace("污染感知的数据与机制证据；活跃清单 `rna-evidence-v2`", "机制创新裁决优先；活跃清单 `rna-mechanism-gate-v3`")
        if "v3 优先级覆盖（2026-11-18）" not in text:
            anchor = "> **D0 状态**"
            assert anchor in text
            text = text.replace(anchor, SPEC + anchor, 1)
        text = text.replace("精确/同源复核进行中，未启动模型推理", "精确命中247/731、484未知；三源输出已归档，按v3停止扩展，未启动模型推理")
        path.write_text(text)
    claim = ROOT / "paper/rna_evidence_v2_claim_matrix.md"
    text = claim.read_text()
    if "v3 优先级裁决（2026-11-18）" not in text:
        claim.write_text(CLAIM + text)
    log = ROOT / "records/DECISION_TRAINING_LOG.md"
    content = log.read_text()
    assert "## §15.895" in content and "## §15.961" in content
    if "## §15.962" not in content:
        log.write_text(content.rstrip() + LOG + "\n")
    for name in ["spec.md", "tasks.md", "checklist.md"]:
        assert (ROOT / name).read_bytes() == (ROOT / "spec" / name).read_bytes() == (ROOT / "spec_sync" / name).read_bytes(), name
    assert "v3 执行覆盖" in contract.read_text()
    print("PASS: v3 priority and mechanism gate synchronized; historical content preserved")


if __name__ == "__main__":
    main()
