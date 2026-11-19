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
- pilot 全部验收通过后，完整矩阵才能运行 seeds=0/1/2；每 seed 两条 20,000-step 轨迹，共 6 次训练、每 seed 四个评估身份。当前不再追加 pilot。
- 主终点只比较固定终端 checkpoint 的 C2−C1：三种子 macro pair-F1 平均提升至少 `+0.01`，micro pair-F1 平均不得下降，micro/macro pair precision 平均降幅均不得超过 `0.005`。本次采用三种子 macro pair-F1 方向全部为正的一致性判据，不在运行后改选配对区间替代，也不挑选冠军 seed。
- 次要终点：C2−C1 的 micro/macro exact-stem F1 平均非劣界限均为 `−0.005`；micro/macro recall 必须同步报告，方向不一致时写“混合证据”。
- 保留 C2 vs C1、C2 vs C0、C2 vs C3 三组比较；所有臂共享初始化、数据顺序、训练预算、MEA 网格与 VL0 选择规则。

### 解冻轨迹与最终裁决

- 固定 step 20,000 是唯一主裁决 checkpoint；保留 step 3,200（4/33 blocks）、9,600（20/33）、16,800（33/33）和 20,000（33/33）的模型快照。
- 每个快照 sidecar 记录实际解冻数、backbone 权重哈希和 checkpoint 文件 SHA256。轨迹 best 只能用于收敛分析，不能替代三种子终端主终点。
- 轨迹分析回答：C2 优势随解冻扩大/稳定/消失，stem 分歧是否集中于少数序列或长度桶，C2 是否仅较早达到好 checkpoint，以及 C0 充分训练后是否反超。
- C2 达主终点且 stem 非劣：选定无 Turner 对齐候选，之后只允许一次冻结外部评测；pair 提升但 stem 越界：不选生产路线，转向 pair→stem 目标不一致诊断。
- C0 终端更强：停止无 Turner 主线；种子方向不一致：不选路线、先查训练稳定性；C3 再次显著较弱：只允许形成“本设置下 Turner 不宜仅作推理补救”的有限结论。
- 既有八组外部面板只有在前述候选成立时才可冻结应用一次，且仍为非独立敏感性证据；严格 OOD 需另用官方或时间外划分。

## 解码和指标

- C0 的 MAP temperature=1；C1/C2/C3 各自只在 VL0 选择 MEA temperature 与 gamma。
- temperature grid=`[0.5,1,2,4]`，gamma grid=`[0.5,1,2,4,8]`。
- 选择顺序：micro pair-F1 → macro pair-F1 → micro precision；并列时取较低 temperature、较低 gamma。
- 同时报告 micro/macro pair precision、recall、F1 与 exact-stem F1；不跨层级比较 F1 数值。

## 解释边界

- 所有路线判断均以上述冻结的主要终点、次要终点和最终裁决规则为准；VL0 单独不能支持生产修复或 SOTA。
- 运行后不得放宽阈值、改用有利指标、用轨迹 best 替代终端结果，或从三个 seed 中挑冠军。

## 执行、验收与存储

- 代码和 Git 提交留在 `/home/rna_jev_xens2_dualview_round56`；共享初始化、checkpoint、raw scores 和报告全部写入 `/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_v1/`。
- 端到端入口：`bash tools/run_prior_alignment_stage.sh smoke 0`、`bash tools/run_prior_alignment_stage.sh pilot 0`；pilot 全部验收通过后才可执行 `bash tools/run_prior_alignment_stage.sh full SEED`，其中 SEED 为 0/1/2。冻结的完整矩阵由 `bash tools/run_prior_alignment_full_matrix.sh` 顺序执行 seeds=0/1/2，并在全部阶段验收后生成 `full_matrix_verdict.json`。
- 每个阶段顺序执行：共享初始化核验 → train-on/train-off 两次训练 → 同一 checkpoint 的 on/off raw 导出 → VL0 四身份解码 → 独立工程与身份验收 → 独立科学复核。脚本不读取旧外部面板。
- 训练逐步写入 `batch_order.jsonl`；验收器逐 step 比较两臂序列哈希，检查共享初始化、终端 checkpoint、有限 loss/gradient、解冻计划、纯 length-NLL、四身份来源和 `/teams` 存储边界。
- full 模式关闭周期性模型快照，只保存四个合同里程碑；resume 仍每 500 steps 覆盖保存，用于容错，不作为科学 checkpoint。每个里程碑 `.pt.json` sidecar 的文件 SHA 由阶段验收器复核。
- `stage_verification.json` 只证明工程与身份有效，不自动选择科学路线，不构成生产修复或 SOTA 结论。
- `tools/verify_prior_alignment_science.py` 从逐序列网格明细独立重算 micro/macro 指标与冻结选择规则，再从原始 scores 独立重跑四个身份的终端解码、pair/exact-stem 计数和汇总；结果写入 `scientific_verification.json`。该文件仍只限 VL0 开发集，不选择生产路线。
- 当前 pilot 启动于科学复核器加入之前，因此完成后必须补跑该复核器并生成 `scientific_verification.json`，不能只凭旧启动脚本的退出状态宣告验收完成。

## 两项定向数据检查

- `tools/analyze_chanrg_task_truncation.py` 分开报告任务原生冻结 GT 与统一 canonical-nested 两种分母，并保存逐序列、逐 family 的删除原因和理论 recall 上限。
- `tools/analyze_high_frequency_fn.py` 仅应用已冻结的 `xens2-TurnerOff-MEA(T=2,gamma=4)`，对高频已见环境 FN 分析 posterior、Turner-on MAP 四状态、长度/loop/helix/register/source 分桶和保守的 exact-sequence 标签冲突；旧外部结果只作非独立诊断。
- `tools/verify_targeted_diagnostics.py` 独立重算 CHANRG 聚合、512 条外部 raw 的 candidate/baseline 解码、逐 pair 状态与 posterior，并核对冻结 route summary。复核通过前不得写入 PPT。
