# Pair/helix 覆盖瓶颈：无训练诊断合同 v1

日期：2026-11-19

## 停止边界

FoldNAR、Tropical 执行器、全局生成模型及 E1–E5 保持停止。本合同不讨论也不实现这些算法轨迹，不启动 B0–B4 或任何训练、微调、蒸馏。允许的计算仅为数据审计、冻结 checkpoint 前向推理、解码与独立复核。

核心问题：在标签和评估可信的前提下，为什么基础 pair/helix 覆盖不足；瓶颈属于数据/任务定义、训练目标与表征，还是解码。

## 指标阅读约束

pair、stem、junction、topology 的定义、匹配容差和分母不同，不横向比较其 F1 大小，不将 topology-F1 高于 pair-F1 解释为 topology 学得更好。首要指标为真实 pair 的 TP/FP/FN、precision、recall、F1 与 stem 的一致变化。

## 诊断 A：标签与合法 mask 天花板

对注册表内 FROZEN 数据逐项统计：原始/重复/越界 pair、canonical、nested、min-loop、合法 mask 覆盖、crossing/pseudoknot、partner 冲突、非 ACGU 残基与无法回溯的缺失残基映射。

- `mask_theoretical_max_pair_recall = 合法 canonical 且 span>3 的 GT pair 数 / 全部唯一有效 GT pair 数`。
- crossing 与 partner 冲突另列，不将合法 mask 上限冒充 nested DP 的精确最优值。
- 当前 JSONL 无父序列/缺失残基映射时，历史删除量记 `NOT_OBSERVABLE_FROM_FROZEN_JSONL`；本审计自身不删除残基。
- 若 mask 覆盖率低于 100%，先修数据或任务定义，不用训练弥补被强制排除的 GT。

## 诊断 B：raw score 与 DP 解码天花板

模型身份固定为 Plan-A seed0、Plan-A seed1、Plan-A 等权平均、R2D 和 xens2（`w_plana=0.7`，Plan-A 内部 `s1_weight=0.5`）。不更换 checkpoint。

开发集为完整 VL0（196 条）。外部敏感性面板为八个主要 split 各 64 条，按 `sha256(dataset_id|name|sequence_sha256)` 排序取最小值，选择前不查看模型误差；外部只应用 VL0 冻结的参数一次。面板只用于方向复核，不替代全 split 结论。

比较：

- 合法候选上的 raw pair AUPRC、recall@precision≥0.5/0.7；
- 独立 sigmoid，阈值仅在 VL0 选择；
- 当前 Nussinov MAP；
- 精确 inside–outside marginal；受计算限制时按长度明确报告 skipped，不用近似冒充精确值；
- MEA：最大化 `2*gamma*sum(p_pair) + sum(q_unpaired)`，等价 pair 权重 `2*gamma*p_ij-q_i-q_j`；gamma 网格 `{0.5,1,2,4,8}`；
- centroid：最小期望 symmetric difference，等价 pair 权重 `2*p_ij-1`；
- marginal 温度网格 `{0.5,1,2,4}`（按最新任务合同扩展；在任何正式 raw/DP 结果产生前冻结）。正温度对纯 MAP 的 argmax 不变，不重复扫描 MAP 温度；
- Turner/prior 默认与关闭，以及各成员和 xens2 集成。参数仅在 VL0 选择。

判读：raw 排名好而 DP 差则优先解码；集成 recall 下降则检查成员互补与融合；raw 本身差则进入数据、训练目标与表征排查。小面板不用于宣称 SOTA。

## 诊断 C：训练—测试 pair 环境覆盖

以 TR1c 为训练参照，对 VL0 和八个主要 split 的全量 GT pair 与 xens2 FN 分桶：pair type、span、hairpin loop/直接子 helix 环境、helix 长度与 register、序列长度、GC、family/clan、标签来源及可获得的实验元数据。

采用分层 key，分别报告 exact key 在 TR1c 的出现次数（0、1–9、10–99、100+）及对应 FN rate，区分“训练未见”与“训练大量出现但仍学错”。无法获得的 family/clan/实验元数据保持未知，不补造。

## 路线裁决

以下数值门槛在正式 raw/DP 结果产生前冻结：若某非 MAP 解码在 VL0 的 micro pair-F1 相对 MAP 提升至少 0.02，且在八个外部面板中至少 6 个 micro 与 macro F1 同向改善、至少 6 个 precision 下降不超过 0.05，则称为“解码主效应证据”；较小的提升只记为次要选择效应。mask<0.99 记为数据/任务定义贡献因素。粗粒度局部键在 TR1c 出现 100 次以上的 pair 仍有至少 20% FN，记为“高频已见但仍学错”；若该现象在至少 6 个外部集合出现且解码主效应证据不成立，则优先排查训练目标、优化与表征，而不是继续补同类数据或更换全局架构。

- 数据/任务定义：mask 排除 GT、标签冲突/来源异质性或训练未覆盖主导 FN。
- 解码：raw 排序与独立候选较好，但 MAP/集成后明显损失，且 VL0 选择的解码在外部同向改善。
- 优化/训练目标/表征：合法且训练高频环境中的 FN 仍高，raw 排序差，或训练误差与泛化误差显示优化失败。

每项结论必须绑定逐序列/逐 pair artifact、代码与数据哈希，并由不导入主分析器的第二实现复核。若证据不足，路线为 `INSUFFICIENT_EVIDENCE`，不自动进入训练矩阵。

## 交付

代码留在 `/home/rna_jev_xens2_dualview_round56`；大型逐 pair 分数、缓存和 artifacts 放在 `/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/pair_coverage_diagnostics_v1/`。确认后提交并推送 GitHub，再将简洁结论追加到既有组会 PPT。
