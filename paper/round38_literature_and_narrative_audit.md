# Round 38 文献、补图与论文叙事审计

## 一、优先补图

| 优先级 | 图件 | 必须展示的实验现象 | 能支撑的叙事与边界 |
|---|---|---|---|
| P0 | 主图 2/5 的完整基线端到端速度 | 同一 4090、TS0 n=1288、raw-sequence、B=1/2/4、三次重复；eFold/UFold/BPfold/E2Efold/EternaFold 逐项列出；CPU 方法不冒充 GPU 时间；缺官方权重标 blocked | 只有在完整、公平的测量后才能判断是否速度 SOTA。不得混入 A100、head-only 或论文中 Titan Xp/RTX3090 的数值 |
| P1 | 置信度 risk–coverage / abstention 曲线 | TS0、bpRNA-new、GenA/GenC/GenF 分面；横轴保留样本比例，纵轴剩余样本 micro-F1 或错误率；标 n 与置信度分箱；阈值只由 VL0 确定 | 比单独 ECE/Spearman 更直接地证明置信度是否能筛掉失败结构。小 PDB split 单列，不把大样本结论外推 |
| P1 | GenC/GenF 家族级失败归因 | 每家族 n、预测/真值配对密度、零正确配对率、F1；突出 RF03981、RF03874、RF03662，但保留全部家族 | 证明 OOD mean 掩盖了弱家族；空输出很少，但 GenC/GenF 分别有 20.8%/39.2% 的序列没有任何正确预测配对，不能把该指标写成零输出率 |
| P1 | 泄露敏感性与 clean 子集复算矩阵 | 每模型区分精确重叠数、clean 重算 n、未审计、特殊 checkpoint；RNAformer 同时列原文 GT 与项目 GT | 证明结论对数据卫生敏感，但不能把部分复算称为完整去污染复测 |
| P2 | 置信度分层的代表性结构案例 | 预先按置信度中位数及 10%/90% 分位选样，画 GT、预测、FP/FN；不按最好看结果挑样本 | 补足机制直觉，不能替代全板统计 |
| P2 | 长度与长程配对联合分层 | 现有长度桶进一步区分短程/长程配对，报告 n、F1、运行时间与显存 | 与 E2Efold 的长序列限制一致，明确当前 O(L³) DP 的部署边界 |

## 二、原文对本项目的直接约束

- **UFold**：原文报告约 160 ms/seq，硬件为 Titan Xp；输入为 `17×L×L`，后处理包含 100 步迭代。不能把原文速度写进 RTX 4090 图，也不能测速时省略后处理。原文还使用 CD-HIT-EST 80% 去相似，因此本项目的精确重叠审计只是较弱口径，不能声称覆盖同源性去污染。
- **BPfold**：原文以 base-pair motif energy 支撑未知家族泛化，并用 confidence index 与 F1 的相关性支撑质量分层。其 CI 系数是在当时可用测试数据上经验选择的；若借鉴该思路，不能直接复制系数或用测试集拟合我们的阈值。正式 BPfold 基线还需包含六个官方 checkpoint、必要 motif 输入与后处理。
- **E2Efold**：原文 Table 4 的 0.40 s/seq 不是当前硬件结果；其 README 明示 600–1800 nt 的长序列模型表现不好。长序列应作为限制与分层实验，而不是只给 pooled F1。
- **RNAformer**：原文做了 80% similarity cutoff 与 BLASTN 同源剔除，并区分 intra-family、inter-family 和 biophysical distillation。固定 bprna、inter-family 特殊微调和 biophysical 权重必须分列；原文全文配对 GT 与项目 canonical-nested GT 不可互换。
- **BPfold/RNAformer 的共同启示**：只报精确序列重叠不足以证明公平。后续应补训练/测试的同源性与家族级敏感性，不可用“没有精确重复”替代完整去污染。

## 三、必须修正的表述

| 旧或易误导读法 | 应改为 |
|---|---|
| “xens2 在主 benchmark 实现 6/8 SOTA” | “统一 xens2 胜预注册 clean 参照 6/8；这不是全板 6/8 SOTA。” |
| “UFold 是 bpRNA-new 的 SOTA” | “UFold=0.6106 是预注册参照；当前原板 EternaFold=0.6854、MXfold2=0.6688、eFold=0.6640 均更高。” |
| “ViennaRNA 在 TS2 异常高，说明有泄露” | “ViennaRNA 没有训练拟合的序列模型，高值不是泄露证据；MXfold2=0.8991 还略高于 ViennaRNA-mfe=0.8980。” |
| “RNAformer 原文 TS2 与我们的复测不一致，所以复测错误” | “原文 0.802、inter-family 项目 GT 复测 0.9043、固定 bprna 复测 0.8590 属不同 checkpoint/GT 口径，必须并列而非互换。” |
| “第 18 页是完整泄露处理表” | “这是泄露审计与 clean 子集复算表；只有实际重算的格子可称去重叠，其余沿用原板并显式标记。” |
| “4090 导致主性能变化” | “八 split 的最大绝对 F1 差为 0.0000903，没有明显精度变化；变化主要在速度与运行环境。” |
| “我们已是速度 SOTA” | 在扩展基线未闭环前，只能说“当前已测共同协议中的精度-吞吐前沿”；禁止绝对 SOTA。 |
| “GenA/GenC 都能通过再训练提升” | “GenA=0.6443 已领先；GenC=0.3793、GenF=0.3066 是短板。已有 GT/混合语料/种子实验未超过 zero-shot xens2，新提升必须由新证据支持。” |
| “ECE 和 Spearman 证明置信度可部署” | 还需 risk–coverage、跨 split 稳定性及小样本限制；不能从相关性直接推出部署有效。 |
| “模型已适合长序列部署” | 长序列性能仍衰减，精确 DP 为 O(L³)，当前只界定适用范围。 |

## 四、论文写作状态与硬门

1. 当前根目录 `preprint_draft.md` 仍含旧模型性能、旧参照、训练中状态和“速度不声称”等历史段落，不能只改几句就作为新稿。应从已验收源 JSON 重建主结果、速度、泄露、OOD 和 Limitations，而不是沿用旧数字。
2. 每条主叙事按“现象 → 对照 → 限制”展开。没有对照实验的机制解释只写 hypothesis，不写 conclusion。
3. 新增图必须与源 JSON 同源；全文 SOTA、clean、end-to-end、macro/micro-F1、checkpoint 和 GT 术语保持一致。
4. 负结果 bpRNA-new、TS2、GenC/GenF、长序列与下游负迁移不得被平均值掩盖。
5. 在五模型测速、逐模型泄露复测和 OOD 新证据完成前，不升级速度 SOTA、完整去污染或 OOD 提升结论。
