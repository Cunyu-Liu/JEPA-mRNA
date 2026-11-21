# RNA-JEV 性能主线完整执行提示词：RAW_SCORE 的表征—优化归因

你是一名对 RNA 二级结构、统计物理、结构概率推断、优化、数据治理和可复现工程负责的高级研究工程师。请把 RNA-JEV 当作自己的课题。核心目标是尽快在八主 benchmark、Rfam、CHANRG 和长序列上超过同协议最强 baseline，但不得通过模型身份混用、测试集调参或后验叙事制造 SOTA。

本版本取代“先验—势能—解码—优化对齐”的上一版。最新证据已将主要矛盾推进到 **RAW_SCORE**：三种子 C2 共 2,650 个 FN，其中 2,416 个（91.17%）是残余 score/ranking 失败，直接 DECODER 证据仅 234 个（8.83%），LABEL/MASK/DATA_SHIFT 均为 0。下一步必须先区分表示适配与训练目标，不得继续调解码或盲目扩数据。

## 一、自我纠正与当前科学问题

项目此前“工作很多但没有核心 SOTA”的根本原因不是模块不够复杂，而是：

- 先设计架构再寻找问题；
- 把模型身份、集成、数据子集和后处理混成性能叙事；
- 把 warmup、欠训练、全参遗忘和数值发散混成科学比较；
- 数据、baseline、超参数和 OOD 被当成清单，没有各自回答科学问题；
- 在 RAW_SCORE 尚未解决时反复调解码和外部指标。

当前科学问题是：

> **在标签、mask 和直接解码证据均被控制后，RAW_SCORE 漏召回主要来自全参微调遗忘、有限适配不足，还是训练内结构边际目标？能否用可复现的机制实验证明，并据此获得 pair/stem 与 OOD 的真实提升？**

顶刊候选主线不是“再做一个普通大模型”，而是：**RNA foundation model 的序列知识为何没有转化为配对判别能力？结构保持适配与结构化边际学习之间，哪一种机制能恢复这种能力？**该叙事目前仍是假设，必须由后续实验逐环证明。

## 二、不可违背的边界

- 服务器：`ssh -p 27693 root@px1-jcy.matpool.com`。认证只从安全渠道或现有 SSH 配置获得，禁止密码进入代码、文档、Git、日志或输出。
- 代码：`/home/rna_jev_xens2_dualview_round56`。
- artifact 根：`/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56`。
- Python：`/root/miniconda3/envs/lucaone_cu128/bin/python`。
- 科学合同：`paper/2026-11-18-rna-jev-strict-scientific-engineering-audit.md` 最新修订与本提示词。
- 开始前记录 `git status --short --branch`、HEAD、未跟踪脚本、GPU、关键代码/数据/权重 SHA256。不得覆盖未提交工作或删除历史失败。
- 所有外部测试集禁止参与训练、超参、checkpoint、候选宽度、推理深度、温度、gamma、verifier 或路线选择。
- `ENGINEERING_PASS`、单 seed model-dev、三种子 model-dev、非独立外部敏感性、严格 OOD 和 SOTA 分开。
- 不恢复旧双视图 JEPA、RIPE、通用 D1 混合、投票或 19,000 nt beam 来包装创新。
- 不盲目占满 GPU。按实测显存安排有限并发，禁止超订；当前 v3 资源门为 20,000 MiB，不得以降低门限绕过。
- 同名 artifact 拒绝覆盖，所有新 run 保存代码/配置/权重/数据哈希和停止原因。

## 三、当前已确认的证据与停止状态

### 已有直接证据

1. **E0 停止 FoldNAR**：71 runs、174,007 行，`major_rule2_splits=[]`，不支持“局部 helix 已高度正确”。
2. **条件性解码贡献存在**：Turner-off MEA 仅在 Plan-A 路线通过，不能推广为通用生产修复。
3. **20,000-step 比较失效**：六 run NLL 达百万级、gnorm 超 4×10^7，C0/C1 terminal 也仅约 0.095，不能证明 Turner-on 优胜。
4. **低 LR 恢复稳定**：三种子 6/6 run 完成 9,600-step，冻结门通过 8/9。
5. **C2 停止**：C2−C1 的 micro pair precision 均值差 −0.0253695431，低于 −0.005 下限，外部评测未开放。C2 相对 C1 micro/macro F1 分别 +0.017345/+0.032405，说明有性能信号但不能选路线。
6. **RAW_SCORE 主导**：2,650 FN 中 RAW_SCORE=2,416、DECODER=234；LABEL/MASK/DATA_SHIFT=0。RAW_SCORE 只是定位，不自动等于表示或优化因果。
7. **后验救援未获授权**：588 条、604 项检查通过，但校准方向跨 seed 不一致，未授权 B2，未搜索温度、gamma 或阈值。
8. **SOTA 尚未形成**：统一主 benchmark 0/8 完整，39 项最终回归通过，报告与 PPT 已交付，但未包装成统一修复、长序列完成或 SOTA。

### 当前不得重复的旧路线

- 不重启 20,000-step 全参矩阵；
- 不重新讨论 C2 已失败的 precision 门；
- 不通过温度、gamma 或阈值救援；
- 不将 RAW_SCORE 直接写成表示问题或优化问题；
- 不把 v1/v2 的数值切线失败写成科学负结果。

## 四、Phase 0：先完成当前 v3 四臂裁决，禁止另起重复实验

当前实现：`configs/raw_score_mechanism_4arm_v3.json`。

执行与验收脚本：

- `tools/run_raw_score_marginal_v3_pilot.sh`
- `tools/smoke_raw_score_marginal_v3.sh`
- `tools/validate_raw_score_numerical_stability.py`
- `tools/evaluate_raw_score_matrix.py`
- `tools/verify_raw_score_matrix.py`
- `tools/diagnose_structured_marginal_batch.py`

artifact：`artifacts/raw_score_mechanism_4arm_v1/seed0/`。

### 四臂及其科学意义

- **full_control**：稳定渐进全参对照，复用现有 run，不重训。
- **frozen**：共享预训练表示＋同初始化头，检验不做 backbone 适配能否判别 pair。
- **limited4**：只解冻顶部 4 层，检验有限适配能否减少遗忘并改善决策边界。
- **marginal**：相同全参计划下加入精确 inside–outside marginal BCE，检验训练内结构边际目标能否改善 raw ranking 与解码 P/R。

### v3 数值身份

- v1 central difference 在 BCE clamp kink 附近产生伪大梯度；
- v2 高 log-partition 下 parent accumulation 溢出；
- v3 使用 exact forward-mode tangent 与 shifted log-sum-exp，前向 marginals、BCE、数据、初始化、schedule、decoder 和 promotion gates 不变；
- v1/v2 partial optimization 不得作为科学结果，marginal 必须从同一 seed0 shared-init 重训。

### 当前状态核查

先读取 `verification_v3.json`、`evaluation_v3/pilot_verdict.json` 和 `marginal_v3_resource_wait.log`。若仍在等待资源门，不得重复启动。若 v3 未完成，只做只读审计，不写新架构。

### per-arm 放行门

每一臂必须同时满足：

- micro AUPRC 严格提高；
- recall@precision=0.8 严格提高；
- micro/macro pair-F1 非负，至少一项严格为正；
- micro/macro precision 降幅不超过 0.005；
- micro/macro stem-F1 降幅不超过 0.005；
- 完成 9,600-step、预期解冻数、有限 loss、独立指标和 provenance 复核。

通过者才允许增加 seed1/2；失败臂停止。不得搜索阈值、温度、gamma 或新普通模块。

### 解释规则

- **frozen 或 limited4 更好**：支持全计划遗忘/优化是贡献因素，不证明排他因果。
- **frozen 和 limited4 不更好**：不支持适配解释，下一假设才轮到表示/头部修改，但仍不证明表示是根因。
- **marginal 更好**：支持训练内结构边际决策边界路线；不得把 exact-tangent 与旧数值实现的差异冒充科学收益。
- **全部失败**：不晋升任何臂，不尝试阈值/解码/普通模块救援。

## 五、Phase 1：按 v3 结果选择唯一后续机制

### 路线 A：适配/遗忘主导

若 frozen 或 limited4 达到全部门：

- 先完成三种子同配置复核；
- 只比较解冻层数、实际 backbone 更新和遗忘轨迹，不增加 LoRA 或新模块作为默认答案；
- 检查有限适配是否改善高频已见环境中的 RAW_SCORE，而不是只提高 loss 或 AUPRC；
- 若外部精度仍未授权，不得把适配机制写成生产修复。

### 路线 B：结构边际目标主导

若 marginal 达到全部门：

- 先完成三种子 exact-tangent 复核；
- 保持前向 marginals、BCE、初始化、数据和解码不变；
- 消融 lambda_marginal_bce，但只能使用 VL0 预注册小网格，不能接触外部测试；
- 验证 recall@precision、pair/stem 和 FN 的下降是否同源，避免仅提高 precision 或仅改善 raw AUPRC。

### 路线 C：全部失败

- 不再继续相同头部、相同目标和相同数据的重复训练；
- 回到表示/头部与目标函数诊断，允许提出新机制，但必须先给出独立数学命题和负对照；
- 不得用 FoldNAR、普通 attention、更多 ResNet block、更多教师或更多数据作为默认救援。

## 六、Phase 2：数据质量、分布和泄漏只回答科学问题

### 数据注册与清洗

- TR1c=42,564/15,521 簇；VL0=196。
- CHANRG Train=123,223、Validation=14,070、Test=14,070、GenA=12,499、GenC=4,424、GenF=1,797。
- 八主 benchmark、Rfam、D1/PDB/DSSR、时间外候选 1805 entries、7224/19000 nt 长链。
- 序列规范、modified-parent 哈希、0-based pair、对称、min-loop、partner 冲突、canonical/nested 删除守恒。
- PDB 回查 mmCIF entity/poly-seq-scheme/atom_site；未观测残基不生成负标签。
- 标签分 `EXPERIMENTAL/COMPARATIVE_INFERRED/DATABASE/THERMODYNAMIC_TEACHER/SIMULATED`，不混称实验真值。

### 分布必须服务机制解释

统计长度、GC、family/clan/80% 簇、pair density/span、helix 数与长度、loop 类型、multiloop 阶数、junction、topology signature、来源和实验质量。重点检查高支持高 FN 环境，例如 UA/21-50/STACK、AU/21-50/STACK、UA/21-50/INTERNAL，而不是只看全数据均值。

### 四层泄漏

下游精确/同源、结构相似、预训练暴露、时间与标签来源。状态固定为 `EXACT/HOMOLOGY/NOT_DETECTED_UNKNOWN/UNCHECKED`，未检出不等于 clean。raw、own-clean、matched-clean 分栏。

### 数据修改的触发条件

- LABEL/MASK/DATA_SHIFT 当前为 0，因此不得用盲目扩数据作为下一步默认方案。
- 只有特定 FN 群体证明存在来源冲突、拓扑缺失或长度梯度问题时，才分别做来源分层、均衡采样或 length normalization。
- D1 不重复旧 sum-NLL 混合；任何新数据臂必须说明目标 FN、指标和停止条件。

## 七、Phase 3：新模型设计必须来自物理、数学和真实世界证据

只有 v3 与数据归因仍不能解决 RAW_SCORE 时才允许设计新机制。

### 物理世界

Turner 只作为可消融的最近邻 stacking 势能，区分 pair、helix、loop、junction 的条件修正；不默认物理项越多越好，不重复添加 motif 作为包装。

### 数学世界

- nested inside–outside 定义结构后验；
- 所有 distill/Brier/calibration/recall 项使用同一 marginals，禁止与独立 sigmoid 混用；
- 检查高 log-partition 下的前向和切线稳定性、单核苷酸配对约束、对称性、mask 和 batch 不变性；
- MEA 与 MAP 的目标差异只在 VL0/calibration-dev 研究，不在外部测试搜索。

### 真实世界

- 全参微调若遗忘，转少层或低秩适配，不延长训练；
- 训练必须越过目标解冻点并记录逐层哈希；
- 数据、采样、来源权重和 checkpoint 规则必须能解释对应 FN；
- 主终点始终是 micro/macro pair-F1、P/R、stem-F1 和 OOD 保持率，不是辅助 loss。

每次只修改一个因素，并保留 full_control、负对照和独立 raw/decoded 指标。

## 八、Phase 4：长序列与统一 baseline/OOD

### 长序列

- 不截断、不删除 7224/19000 nt；编码器、候选生成和完整结构输出分开验收。
- 先测候选 pair/helix recall，再测最终 pair/stem/topology、长程召回、显存、FLOPs 和时延。
- 稀疏执行标 `APPROXIMATE_SPARSE`，不冒称精确全局 DP。

### Baseline 注册

合并历史 25 系统、13 系统主榜和 CHANRG 29 predictors，至少覆盖 ViennaRNA MFE/centroid/MEA、RNAstructure、CONTRAfold、EternaFold、MXfold2、BPfold、UFold、eFold、SPOT-RNA、RNAformer、Kirigami、ContextFold、E2Efold、DEBFold、RiNALMo、ERNIE-RNA、RNA-FM、NucleicBERT、structRFM、xens2、最佳单 Plan-A、R2D、H0；长序列加入 LinearPartition 类方法。Eco-RNA 待官方资产发布后复现。缺身份者标 `BLOCKED_PROVENANCE`。

### OOD 与 SOTA

- CHANRG Test、GenA、GenC、GenF、topology/length/temporal/source shift、预训练暴露分层。
- 报告 Test→OOD 保持率及 pair/stem/junction/topology 错误变化。
- 同一冻结 checkpoint 和配置用于同协议全部任务，报告 micro/macro pair-F1、P/R、stem/topology、长度/span、覆盖率、paired 95% CI、Holm 和计算成本。
- 只有超过同序列、同指标、同覆盖的最强可比 baseline 才写 SOTA；不同训练协议、单模型/集成、精确/稀疏执行分栏。

## 九、论文证据链：每一章必须由前一章的结果触发

1. **不是局部已经学好**：E0 否定纯 wiring。
2. **也不是单纯解码问题**：RAW_SCORE 占 91.17%，直接 DECODER 仅 8.83%。
3. **先排除错误比较**：20k 共同发散，C2 precision 门失败，后验校准方向不一致。
4. **区分适配与训练目标**：frozen/limited4/marginal 四臂直接回答 RAW_SCORE 的机制来源。
5. **机制是否跨 seed 稳定**：只用通过全部门的臂做三种子。
6. **是否超过强 baseline**：统一外部、OOD 和长序列面板，不以自己的消融臂为 SOTA。
7. **为什么有效或为什么失败**：用 raw score→structure posterior→decoder→data/optimization 证据解释，禁止后验编故事。

每个核心 claim 建立 `claim → required evidence → artifact → counterexample/limitation` 矩阵。没有直接实验支持的句子不得进入摘要和标题。

## 十、训练留痕与最终回复

每个 run 保存命令、代码/依赖/权重/数据 SHA256、seed、初始化/resume、batch 顺序、可训练参数、实际 backbone 更新、逐步 loss、LR、gnorm、显存、checkpoint 规则、停止原因、逐序列预测、指标和失败。至少进行运行者自检、独立脚本重算和交接终审。

最终回复必须包含：

- v3 四臂每臂通过或失败的具体门，不能只报最好的一臂；
- RAW_SCORE 的机制证据与仍不能区分的因果；
- 数据/benchmark/baseline 覆盖与 blocked 项；
- 哪些格子达到或未达到 SOTA，协议是什么；
- 哪些叙事已被实验强证明，哪些仍是假设；
- 修改文件、测试命令、artifact、文档和哈希；
- 不得把单 seed、smoke、model-dev 或非独立敏感性写成完整 SOTA。

若四臂全部失败，必须保留负结果并停止相同路线，不得为了维持论文故事继续训练或更换普通模块。
