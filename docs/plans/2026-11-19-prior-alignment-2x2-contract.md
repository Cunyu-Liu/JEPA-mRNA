# Plan-A 热力学先验 2×2 对齐合同（冻结版）

机器可读合同：`configs/prior_alignment_2x2_v1.json`。本合同在新训练启动前冻结；旧外部面板不参与 pilot、解码参数或路线选择。

## 研究问题

在相同 Plan-A 初始化、数据顺序、训练预算和解冻规则下，Turner 先验在训练与推理中何时帮助或抑制 pair/stem 学习？FoldNAR、Tropical 执行器和全局生成模型不在本合同范围内。

## 两条训练轨迹、四个评估身份

| 身份 | 训练轨迹 | 推理 Turner | 解码 |
|---|---|---|---|
| C0 | train-on | on | MAP |
| C1 | train-on | off | MEA |
| C2 | train-off | off | MEA |
| C3 | train-off | on | MEA |

每个 seed 只训练 train-on 和 train-off 两次。C0/C1 必须读取同一 checkpoint；C2/C3 必须读取另一同一 checkpoint。该矩阵使用单个 Plan-A 成员，不把 R2D 或 xens2 融合混入对照。

## 初始化和目标

- 每个 seed 从 RiNALMo-giga 与新初始化的 Plan-A ResNet2D head 建立唯一共享 init 文件，两臂加载并核验同一 SHA256。
- 不从历史 Turner-on Plan-A checkpoint 继续训练，否则 train-off 会被既有训练历史污染。
- 仅优化 length-normalized CRF NLL；distill、RLCD、calibration、JEPA 和 pair-recall 辅助项均为 0。若以后单独加入 recall 项，只能使用 inside–outside marginals。
- TR1c 42,564 条，batch=4，长度分桶，`default_rng((seed, epoch))`；两臂逐 batch 的序列哈希必须一致。
- head/backbone LR=`1e-4/1e-5`，head warmup=1,600 steps，之后每 800 steps 解冻 2 个 block；不早停。

## Pilot 与完整矩阵

- 单 seed=0 pilot 固定 3,200 steps，使用终端 checkpoint；在 step 2,400/3,200 应分别解冻 2/4 个 block。
- pilot 只用于工程有效性与 VL0 开发集可行性，不运行旧外部面板，也不作生产或 SOTA 决策。
- pilot 全部验收通过后，完整矩阵才能运行 seeds=0/1/2；每 seed 两条 20,000-step 轨迹，共 6 次训练、每 seed 四个评估身份。

## 解码和指标

- C0 的 MAP temperature=1；C1/C2/C3 各自只在 VL0 选择 MEA temperature 与 gamma。
- temperature grid=`[0.5,1,2,4]`，gamma grid=`[0.5,1,2,4,8]`。
- 选择顺序：micro pair-F1 → macro pair-F1 → micro precision；并列时取较低 temperature、较低 gamma。
- 同时报告 micro/macro pair precision、recall、F1 与 exact-stem F1；不跨层级比较 F1 数值。

## 解释边界

- C2>C1：支持训练—推理一致的无先验候选方向。
- C3 最佳：支持 Turner 作为 off-trained 表示的推理正则项。
- C0 最佳：不支持替换受控 on/on MAP 路线，需复核初始化与分布条件。
- 任何结论都要求三种子一致方向、precision 不崩溃且 stem-F1 同步；VL0 单独不能支持生产修复或 SOTA。
- 既有八组外部面板即使将来冻结应用一次，也只是非独立敏感性证据；严格 OOD 需另用官方或时间外划分。
