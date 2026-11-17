import hashlib
import json
from pathlib import Path

from pptx import Presentation

from add_evidence_audit_round56 import DECK, WORK, BLUE, TEAL, ORANGE, GRAY, blank, card, table, textbox, set_notes


def main():
    qa = json.loads((WORK / "interface_label_qa_summary.json").read_text())
    attention = json.loads((WORK / "attention_chunk_validation.json").read_text())
    structure = json.loads((WORK / "structure_feature_gate.json").read_text())
    assert qa["status"] == "PASS_INDEPENDENT_INTERFACE_LABEL_QA"
    assert attention["status"] == "PASS_RINALMO_QUERY_CHUNK_ENGINEERING"
    assert structure["status"] == "BLOCKED_STRUCTURE_FEATURE_FULL_LENGTH_CUDA_OOM"
    assert not qa["train_holdout_group_overlap"] and not qa["train_holdout_pdb_overlap"]
    assert qa["train_holdout_exact_sequence_overlap"] == 0
    prs = Presentation(DECK)
    if len(prs.slides) != 22:
        assert len(prs.slides) == 23
        print(json.dumps({"slides": len(prs.slides), "status": "ALREADY_PRESENT"}, ensure_ascii=False))
        return
    slide = blank(prs, "INTERFACE DATA & ENGINEERING GATE  |  ROUND 56",
                  "RNA–蛋白界面：标签先通过，性能仍待验证",
                  "原始 mmCIF · polymer / label_seq · modified-parent 精确映射 · 不混用 RNA–RNA 配对")
    train = qa["splits"]["train"]
    holdout = qa["splits"]["holdout"]
    card(slide, 0.65, 1.75, 3.75, "可用 train / holdout 复合物链",
         f"{train['n_valid']} / {holdout['n_valid']}", BLUE, 23)
    card(slide, 4.80, 1.75, 3.75, "观测残基 train / holdout",
         f"{train['n_observed_residues']:,} / {holdout['n_observed_residues']:,}", TEAL, 18)
    card(slide, 8.95, 1.75, 3.75, "PDB / 序列簇 / 精确序列重叠",
         "0 / 0 / 0", ORANGE, 23)
    textbox(slide, 0.65, 3.15, 12.0, 0.3, "接触与负样本：只来自真实 RNA–肽链坐标", 12, TEAL, True)
    table(slide, [
        ["项目", "定义与验收"],
        ["界面正样本", "deposited model 1 中，观测 RNA 残基到肽链重原子最近距离 < 5 Å"],
        ["同链负样本", "最近距离 ≥ 5 Å；未观测残基不生成负样本"],
        ["序列与标签", "完整 polymer 对齐；修饰残基按哈希绑定的亲本映射，labels.json 复核一致"],
        ["实验方法", f"train：电镜 {train['experimental_methods'].get('ELECTRON MICROSCOPY', 0)}，X-ray {train['experimental_methods'].get('X-RAY DIFFRACTION', 0)}；分辨率元数据完整"],
    ], 0.65, 3.52, [2.3, 9.7], 0.55, 10.5)
    pair = attention["pair_summary"]
    textbox(slide, 0.65, 6.36, 12.0, 0.34,
            f"工程通过：VL0 {pair['n_sequences']} 条 pair 集合与 macro-F1 不变；7,224 / 19,000 nt 前向有限（非 beam）。", 11, TEAL, True)
    textbox(slide, 0.65, 6.73, 12.0, 0.34,
            "仍阻断：7,224 nt 原 ResNet2D pair 矩阵 OOM，结构辅助臂不启动；不截断、不换成近似结构。", 10.5, ORANGE, True)
    textbox(slide, 0.65, 7.07, 12.0, 0.28,
            "三 seed probe 运行中，当前没有界面性能结论；PDB＋簇隔离不等于 RiNALMo 预训练去污染。", 10, GRAY)
    textbox(slide, 0.65, 7.31, 12.0, 0.12,
            f"label_qa_summary  ·  {hashlib.sha256((WORK / 'interface_label_qa_summary.json').read_bytes()).hexdigest()[:12]}…   |   attention_validation  ·  {hashlib.sha256((WORK / 'attention_chunk_validation.json').read_bytes()).hexdigest()[:12]}…", 8, GRAY)
    set_notes(prs, slide, (
        "这一页只讲数据与工程门，不展示模型性能。旧界面提取把同名蛋白链误认为RNA，已经撤销并隔离。"
        "新流程直接读取原始mmCIF，区分label_asym、label_seq和polymer，修饰残基按哈希绑定的亲本表映射。"
        "最终804条训练链和202条holdout链全部通过独立距离复算，PDB、序列簇和精确序列重叠都为零。"
        "未观测残基不作为负样本。编码器查询分块通过了完整VL0的pair一致性检查，"
        "但原ResNet2D在7224nt的结构矩阵仍OOM，所以结构辅助臂没有启动。预训练暴露仍需单独审计。"
    ))
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "deck": str(DECK), "status": "ADD_INTERFACE_DATA_GATE_ONLY"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
