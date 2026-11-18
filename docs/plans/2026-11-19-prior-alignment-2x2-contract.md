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

## 执行、验收与存储

- 代码和 Git 提交留在 `/home/rna_jev_xens2_dualview_round56`；共享初始化、checkpoint、raw scores 和报告全部写入 `/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_v1/`。
- 端到端入口：`bash tools/run_prior_alignment_stage.sh smoke 0`、`bash tools/run_prior_alignment_stage.sh pilot 0`；pilot 全部验收通过后才可执行 `bash tools/run_prior_alignment_stage.sh full SEED`，其中 SEED 为 0/1/2。
- 每个阶段顺序执行：共享初始化核验 → train-on/train-off 两次训练 → 同一 checkpoint 的 on/off raw 导出 → VL0 四身份解码 → 独立工程与身份验收 → 独立科学复核。脚本不读取旧外部面板。
- 训练逐步写入 `batch_order.jsonl`；验收器逐 step 比较两臂序列哈希，检查共享初始化、终端 checkpoint、有限 loss/gradient、解冻计划、纯 length-NLL、四身份来源和 `/teams` 存储边界。
- `stage_verification.json` 只证明工程与身份有效，不自动选择科学路线，不构成生产修复或 SOTA 结论。
- `tools/verify_prior_alignment_science.py` 从逐序列网格明细独立重算 micro/macro 指标与冻结选择规则，再从原始 scores 独立重跑四个身份的终端解码、pair/exact-stem 计数和汇总；结果写入 `scientific_verification.json`。该文件仍只限 VL0 开发集，不选择生产路线。
- 当前 pilot 启动于科学复核器加入之前，因此完成后必须补跑该复核器并生成 `scientific_verification.json`，不能只凭旧启动脚本的退出状态宣告验收完成。

## 两项定向数据检查

- `tools/analyze_chanrg_task_truncation.py` 分开报告任务原生冻结 GT 与统一 canonical-nested 两种分母，并保存逐序列、逐 family 的删除原因和理论 recall 上限。
- `tools/analyze_high_frequency_fn.py` 仅应用已冻结的 `xens2-TurnerOff-MEA(T=2,gamma=4)`，对高频已见环境 FN 分析 posterior、Turner-on MAP 四状态、长度/loop/helix/register/source 分桶和保守的 exact-sequence 标签冲突；旧外部结果只作非独立诊断。
- `tools/verify_targeted_diagnostics.py` 独立重算 CHANRG 聚合、512 条外部 raw 的 candidate/baseline 解码、逐 pair 状态与 posterior，并核对冻结 route summary。复核通过前不得写入 PPT。
