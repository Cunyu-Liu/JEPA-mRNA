import hashlib
import json
from pathlib import Path

from pptx import Presentation

from add_evidence_audit_round56 import DECK, WORK, BLUE, TEAL, ORANGE, GRAY, blank, card, table, textbox, set_notes


def update_existing(prs):
    assert len(prs.slides) in (24, 25)
    slide = prs.slides[23]
    prefix = "链间组成差异使 shuffle"
    matches = [shape for shape in slide.shapes if shape.has_text_frame and shape.text.startswith(prefix)]
    assert len(matches) == 1
    matches[0].text_frame.paragraphs[0].runs[0].text = (
        "链间组成差异使 shuffle 也达 0.6292，须分层解读；RiNALMo 预训练精确暴露已证实，完整 F2 未通过。"
    )
    set_notes(prs, slide, (
        "这一页是冻结表征的界面probe，不是二级结构主线的翻身结果。H0的micro-AUPRC为0.7715，"
        "高于RiNALMo原生表征和组成对照，但H2与H0的编码器完全相同，不能解释为JEPA收益。"
        "结构辅助臂因长链全配对矩阵OOM没有完成，因此完整F2没有通过。"
        "标签shuffle也达到0.629，说明链间正样本率差异本身有信息，不能把高AUPRC简单等同于相互作用理解。"
        "长度100到499nt仍是弱桶。所有阈值只由训练集内部PDB和序列簇隔离验证选择，"
        "18个run的指标已由sklearn独立复算，旧AUPRC结果保留并已修正同分排序问题。"
        "RiNALMo预训练已确认直接序列暴露，界面分数不能称为未见复合物或独立泛化证据。"
    ))
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "status": "EXPOSURE_BOUNDARY_UPDATED"}, ensure_ascii=False))


def main():
    summary = json.loads((WORK / "converged_baseline_summary.json").read_text())
    strata = json.loads((WORK / "interface_stratified_summary.json").read_text())
    qa = json.loads((WORK / "independent_metric_qa.json").read_text())
    assert summary["status"] == "PASS_SEQUENCE_ONLY_INTERFACE_PROBES_WITH_H2_ALIAS_AND_STRUCTURE_BLOCKED"
    assert strata["status"] == "PASS_DESCRIPTIVE_INTERFACE_STRATA"
    assert qa["status"] == "PASS_INDEPENDENT_SKLEARN_INTERFACE_METRIC_QA" and qa["n_runs"] == 18
    assert qa["max_abs_metric_difference"] < 1e-12
    prs = Presentation(DECK)
    if len(prs.slides) != 23:
        update_existing(prs)
        return
    slide = blank(prs, "FROZEN REPRESENTATION INTERFACE PROBE  |  ROUND 56",
                  "H0 表征优于组成对照，但不是 JEPA 证明",
                  "18 个自然收敛 run · seed=42/2026/3407 · 阈值仅由训练内部 PDB＋序列簇验证选择")
    modes = summary["modes"]
    card(slide, 0.65, 1.73, 3.75, "H0  micro-AUPRC",
         f"{modes['h0']['micro_auprc_mean']:.4f} ± {modes['h0']['micro_auprc_sd']:.4f}", BLUE, 22)
    card(slide, 4.80, 1.73, 3.75, "RiNALMo  micro-AUPRC",
         f"{modes['rinalmo']['micro_auprc_mean']:.4f} ± {modes['rinalmo']['micro_auprc_sd']:.4f}", TEAL, 20)
    card(slide, 8.95, 1.73, 3.75, "H0  二分类 macro-F1",
         f"{modes['h0']['macro_f1_binary_mean']:.4f}", ORANGE, 23)
    rows = [["模型 / 对照", "micro-AUPRC", "二分类 macro-F1", "解释"]]
    labels = {"rinalmo": "RiNALMo 原生表征", "h0": "H0 结构宿主表征", "handcrafted": "局部序列手工特征",
              "random": "链内标签 shuffle", "length": "长度", "gc": "GC 含量"}
    for mode in ("rinalmo", "h0", "handcrafted", "random", "length", "gc"):
        item = modes[mode]
        rows.append([labels[mode], f"{item['micro_auprc_mean']:.4f} ± {item['micro_auprc_sd']:.4f}",
                     f"{item['macro_f1_binary_mean']:.4f} ± {item['macro_f1_binary_sd']:.4f}",
                     "独立表征" if mode in ("rinalmo", "h0") else "对照"])
    table(slide, rows, 0.65, 3.13, [3.3, 2.55, 2.85, 3.3], 0.43, 10)
    short = strata["modes"]["h0"]["mean"]["length:100-499"]
    long = strata["modes"]["h0"]["mean"]["length:>=1500"]
    textbox(slide, 0.65, 6.24, 12.0, 0.32,
            f"长度弱桶：H0 在100–499 nt 的 AUPRC={short['micro_auprc']:.4f}（n={short['n_chains']}）；≥1500 nt 为 {long['micro_auprc']:.4f}（n={long['n_chains']}）。", 10.5, ORANGE, True)
    textbox(slide, 0.65, 6.62, 12.0, 0.36,
            "H2 与 H0 共享冻结编码器，独立 JEPA 效果不可估计；结构辅助臂因 7,224 nt 全 pair 矩阵 OOM 保持 blocked。", 10.5, ORANGE, True)
    textbox(slide, 0.65, 7.01, 12.0, 0.27,
            "链间组成差异使 shuffle 也达 0.6292，须分层解读；RiNALMo 预训练精确暴露已证实，完整 F2 未通过。", 10, GRAY)
    textbox(slide, 0.65, 7.29, 12.0, 0.12,
            f"converged_baseline_summary  ·  {hashlib.sha256((WORK / 'converged_baseline_summary.json').read_bytes()).hexdigest()[:12]}…  |  sklearn QA 18/18", 8, GRAY)
    set_notes(prs, slide, (
        "这一页是冻结表征的界面probe，不是二级结构主线的翻身结果。H0的micro-AUPRC为0.7715，"
        "高于RiNALMo原生表征和组成对照，但H2与H0的编码器完全相同，不能解释为JEPA收益。"
        "结构辅助臂因长链全配对矩阵OOM没有完成，因此完整F2没有通过。"
        "标签shuffle也达到0.629，说明链间正样本率差异本身有信息，不能把高AUPRC简单等同于相互作用理解。"
        "长度100到499nt仍是弱桶。所有阈值只由训练集内部PDB和序列簇隔离验证选择，"
        "18个run的指标已由sklearn独立复算，旧AUPRC结果保留并已修正同分排序问题。"
        "RiNALMo预训练已确认直接序列暴露，界面分数不能称为未见复合物或独立泛化证据。"
    ))
    prs.save(DECK)
    print(json.dumps({"slides": len(prs.slides), "deck": str(DECK), "status": "ADD_INDEPENDENT_INTERFACE_PROBE_RESULTS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
