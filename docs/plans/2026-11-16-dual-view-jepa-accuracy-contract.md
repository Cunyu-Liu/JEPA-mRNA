# 双视图 RNA-JEPA、物理先验与功能下游统一研究合同（2026-11-16）

> **v3 执行覆盖（2026-11-18：机制创新裁决优先；活跃清单 `rna-mechanism-gate-v3`）**
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
> **v2 执行覆盖（2026-11-17：数据与机制证据优先；活跃清单 `rna-evidence-v2`）**
>
> v1 的 P0/P1 工程证据和 `STOP_PILOT_NEGATIVE_DEV_ONLY` 全部保留，不改写。v1 只证明该单 seed、Plan-A warm-start、VL0 配方未通过，不是结构模态的全称否定，也不是外部泛化结论。v2 不恢复三种子、D1、RIPE、旧 Pair-JEPA 或 19,000 nt beam，不通过调权、换指标或测试集选臂挽救 v1。
>
> **硬顺序与停止门**
>
> 1. **D0 数据与暴露审计**：逐来源核验序列数、独立 80% 簇、family/clan、标签来源与置信度、canonical/noncanonical/pseudoknot 删除守恒、PDB 分辨率/NMR/缺失残基、train/dev/test 精确与近似重叠、RiNALMo 预训练暴露和下游测试身份。没有证据的字段标 `UNKNOWN/BLOCKED`，不得把 Rfam 名称当实验质量证明，不得把 unique sequence 当独立家族。ncRNA test 与 TR1c 已知的 304 条精确重叠必须去除并重新审计，测试标签不得进入训练。
> 2. **D1 机制测量补全**：在 model-dev/工程样本上补测主损失与辅助损失梯度 cosine，不能以 `encoder_frozen_warmup` 的 None 冒充已测。完整宿主 FLOPs 必须覆盖 RiNALMo、原 ResNet2D/Turner、dual-view 与必要解码，并明确 profiler 支持范围、batch/length、是否 forward/backward 和显存；与 synthetic 模块口径分开。
> 3. **D2 初始化语义**：RiNALMo 原生初始化、已训练 Plan-A warm-start、优化器新开/延续、仅头训练/逐步解冻必须分开命名。任何新机制比较只能在 D0/D1 通过后于 model-dev 进行；不得用 warm-start 遗忘或原生欠训练代替结构模态结论。
> 4. **证据等级**：`ENGINEERING_PASS`、`MODEL_DEV_NEGATIVE/positive`、`FROZEN_EXTERNAL_RESULT` 分开。工程通过不证明机制有效，单 seed/dev 不证明外部泛化，外部结果必须由冻结 checkpoint、逐序列预测和独立复算支持。
> 5. **独立资产任务**：BPfold motif 仅在官方 MIT 能量表、BRIQ 身份和哈希复核后按 P0/P1/P2/P3/P4 单变量评估，Turner/RIPE 不得冒充。RNA–蛋白界面仅在真实接触、partner、距离、实验方法、负样本和 PDB＋序列簇隔离 QA 后执行；未通过标 blocked，不以 RNA–RNA 配对代替。二者都不是 v1 的抢救叙事。
> 6. **预印本与汇报**：先写 claim–evidence–limitation，优先可靠数据、机制现象和负结果，不预设精度突破。只有独立数据与机制证据支持时才形成相应主张；PPT 只收录复核后的结果。
>
> 每个阶段保存配置、代码/数据/权重哈希、seed、best/resume、停止原因、峰值显存、逐序列预测和独立复算。CUDA 不可用或 CPU 静默降级立即停止。测试集不得用于调参、早停、选臂、校准、视图构造或下游阈值。以下 v1 原文继续作为历史设计与身份约束。
>
> **预训练精确暴露门（2026-11-18）**：RNAcentral 22.0 与 VL0、bpRNA-new、TS0、Rfam、ncRNA test、界面 holdout 的精确序列重叠分别为 184/196、5303/5388、1098/1288、9989/10790、21452/25038、86/202。阳性精确匹配足以证明预训练接触，不需要等待 nt/Ensembl 或同源聚类完成；不得把这些集合称为完全未见、clean-OOD 或独立泛化证明。下游去重叠不等于 RiNALMo 预训练去污染。原始精度可保留为污染感知结果，不删除历史模型；完整预训练语料、80%簇与独立家族审计仍未完成，不能用部分来源反推无接触。证据 `/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit/exact_overlap.json`，SHA256=`6cd88fc5e3b6385ac15f179c225bf575f79c1cec308c1328ce1af31baf558877`。
>
> **三源精确联合补充（2026-11-18）**：RNAcentral 22.0、Rfam 14.9 和 Ensembl 109/56 已完成公开源核验及精确序列去重。联合命中为 TR1c=37870/42564、VL0=193/196、bpRNA-new=5381/5388、TS0=1253/1288、Rfam=10327/10790、ncRNA test=24912/25038、界面 holdout=86/202；同名不同序列直接回查源 FASTA，不按名称合并。证据 `exact_union_summary.json`，SHA256=`d441239311605a2572b3f561357531689b0ea24a19068e1ac417dca75148baa4`。Rfam 14.9 的 165854 条逐查询 80% identity＋双向 80% 覆盖搜索已完成，检出 155635 条，但未检出不等于无同源。Ensembl 同源检索及独立 QA 已完成，检出 61092/165854；Rfam 精确序列家族映射的 58742 行复核通过，Rfam 与 TR1c 有 120 个可映射共享家族，bpRNA-new 交集 0 仅支持下游标签划分。RNAcentral 同源检索及独立 QA 已完成，检出 151835/165854；三源同源联合检出 162489/165854，3365 条未检出不等于无同源，证据 SHA256=`69a0b13b2ef0cb061807e10f50efca3a6da7889ee006cd032e2e72bca2197131`。nt 历史快照未固定，禁止把三源精确或同源联合写成完整污染率或家族独立性证明。
>
> **冻结预测暴露分层裁决（2026-11-18）**：原 xens2 三个 checkpoint 的实际 SHA256 与 P0 一致，历史和重放的 bpRNA-new 预测逐序列一致。全量5388条无跳过，macro-F1=0.5422090871138705；精确暴露n=5381、macro-F1=0.5424236738973192，精确未检出n=7、macro-F1=0.3772531611514662。精确未检出仍未知，不能作clean或显著性/因果结论。全部5388条至少在一个已知来源检出80%同源，同源未检出n=0，故现有bpRNA-new不能形成同源未检出精度对照；不得把空组的0指标解释为性能。只读分析不选模型、不改阈值、不启动新训练，证据`xens2_bprna_new.json` SHA256=`c1c39477b84c3a139a52a091f1011548b1841c50bf6f5e572dc56b75318007ec`。旧替代预测的385条完整记录差异保留，不与P0身份混用。
>
> **全量标签排除审查（2026-11-18）**：1805入口完整QA得到2930接受链/731唯一序列，canonical-nested=485745；9KFX scheme19行与polymer18不一致、9P4K DSSR内部断言保留为不可用入口，不以猜测修复。`label_collection_qa.json`保持`BLOCKED_TEMPORAL_LABEL_COLLECTION_ERRORS`，另由`label_exclusion_review.json`通过`PASS_EXPLICIT_ENTRY_EXCLUSIONS_LABELS_REVIEWED`后才允许使用其余1803入口，所有候选仍保留在审计分母。QA SHA256=`5b57831a…c1c6`，排除审查=`e60c0aaa…9874`，最终序列=`f03efde1…d333`。最终731条标签序列的精确/80%同源复核完成前不得启动模型推理。
>
> **H2 编码器身份限制（2026-11-18）**：H0/H2 best step100 的编码器逐张量完全相同，warmup_head_steps=1600，两个 split 的缓存也一致。故 H2 当前只是 H0 的共享冻结编码器别名，不能用于独立 JEPA 迁移或功能增益结论。保留 seed42 及身份文件 `h2_encoder_identity.json`，停止其余重复 H2 run，汇总不计算独立 H2−H0 效果；不把这一发现改写成新的正/负精度结论，也不通过调参恢复主线。
>
> **界面 QA 更正（2026-11-17）**：此前仅验证距离阈值与计数的 QA 不充分，旧实现未证明 mmCIF 坐标与 RNA 序列逐残基对应，并混入同名蛋白链。旧标签、500 步未收敛探索基线及误标 macro-F1 的数值全部撤销为失败证据，不进入论文/PPT。新标签必须直接由原始 mmCIF 的 entity、poly-seq-scheme、atom_site 建立，逐条核对 polymer、label_seq、完整参考序列、肽链伙伴和未观测残基；缺失坐标不得当负样本。由不共用提取代码的验证器重算全部距离、标签及 PDB＋序列簇隔离后，才可运行冻结表征界面 probe。下游 probe 无固定步数上限，使用训练集内部 PDB＋簇隔离验证自然早停，阈值不得接触 holdout；下游 split 隔离不替代预训练审计，RiNALMo 预训练精确接触已证实，禁止 clean-OOD 叙事。
>
> **界面数据补充**：modified residue 只能在与 mmCIF 哈希绑定的 parents map 下转为亲本碱基，并要求完整 polymer 与原始 labels.json 的序列及哈希完全一致；一基 canonical pair seq_id 不得误用为接触标签。未观测坐标不生成负样本。H0 原 ResNet2D 在 7,224 nt 全配对表示上已真实 CUDA OOM，结构辅助臂未启动并记 `BLOCKED_STRUCTURE_FEATURE_FULL_LENGTH_CUDA_OOM`；不得截断、删除长链、偷换近似模型或把该工程阻断写成结构无效。序列-only probe 与结构辅助结果分开验收，结构臂未闭环时不得宣称完整 F2 通过。

> **状态：研究与实验设计已完成，尚未按本合同启动训练。**
>
> 本文件是当前唯一新实验执行覆盖，替代旧 Pair-JEPA、RIPE、D1 通用复合物扩增和“多数投票即创新”的执行路线。历史结果保留，但不混作新架构证据。执行者必须先完成 P0 资产、数据和接口验收，再按阶段门执行；不得为了赶进度把设计门跳过。

## 1. 中心科学问题与论文目标

中心问题是：

> **实验/可靠二级结构能否作为训练期的潜在模态，通过序列—结构双向预测提高序列表征的未见家族泛化，并且推理时完全不依赖结构标签？**

目标是以**单 checkpoint、单序列输入的新架构取得可统计验证的精度突破**，不是以集成或投票包装创新。原 xens2 的 Plan-A 两成员＋R2D 加权融合是主要历史精度基线，BPfold 是强外部基线。三 R2D 多数投票只保留为辅助部署/消融对照，不再作为新架构主线。

论文叙事不是“老旧投票法也能提升”，而是：

> **结构不仅是要预测的标签，也是可以指导序列表征学习的潜在变量。训练期让序列与结构互相预测，部署时只保留序列到结构的映射。**

不把 Mol-JEPA 的 world model 名称照搬到 RNA，不宣称动力学模拟、三维结构理解或复合物环境建模已经实现。

### 主要判定目标

- 主终点：bpRNA-new 的逐序列 macro-F1 相对原始 xens2 的配对提升，目标效应量至少 +0.01，且配对 bootstrap 95% CI 排除 0、Holm 校正后 p<0.05；这是验收目标，不是预设结果。
- 同步检查：TS0、ArchiveII、Rfam、CHANRG-GenC，不能以主终点改善换取这些集合的明显退化。
- 对照必须包含原始 xens2、同初始化/同数据/同训练预算的无 JEPA 宿主、BPfold 和必要的参数匹配 MLP 控制。
- 主要对照使用 strict micro-F1 与 macro-F1，INF/P/R 同报；raw、own-clean、matched-clean 分开，band=128 只能作为近似补充，不能进入主精确结论。
- 三训练 seed=42/2026/3407，最终均值与配对区间齐全。主比较采用预注册的配对 bootstrap 和 Holm 校正，不以单 seed、smoke 或 dev 结果宣称突破。
- 若新单模型不能超过原始 xens2，明确记为未达到精度突破，不能用不确定性故事掩盖失败。

## 2. 已核实的文献与实现依据

### 2.1 Mol-JEPA

- 论文：*Mol-JEPA: A multimodal molecular world model based on Joint Embedding Predictive Architecture*，arXiv:2608.22642v4。
- 官方仓库：https://github.com/Boehringer-Ingelheim/mol-jepa，核对分支 `master`，commit=`27f0065b69c73d18eec7a32ecc413b6dc3f33d90`。
- 许可证：仓库 README 标注 **CC BY-NC 4.0**，只作非商业研究借鉴，保留作者、论文、仓库、commit 和修改说明，不复制分子数据或模型声称为本项目结果。
- 可借鉴的核心：整模态掩码、可用模态到被遮蔽模态的潜在预测、Transformer predictor、SIGReg 抗坍塌、每类模态均有预测责任。
- 官方实现中 `ratio_mask`、`predict_mask_cls` 和 `prediction_samples` 的索引语义必须独立复核，不能照抄后假定 loss 一定落在被遮蔽模态上。新实现必须用单元测试证明这一点。
- Mol-JEPA 的消融提示：SIGReg 放在 target 端比放在 predicted/CLS 端更稳定；无额外 projection head 更优。此处只作待验证设计依据，不保证 RNA 上同样成立。
- 本地研究材料：`.round56_work/mol_jepa.txt`、`moljepa_jepa.py`、`moljepa_prediction_head.py`、`moljepa_sigreg.py`、`moljepa_README.md`。

### 2.2 BPfold

- 本地论文：`论文/BPfold_2025_NatCommun.pdf`。
- BPfold 的核心不是简单 Turner 或 RIPE 距离偏置，而是：
  1. 三类三邻居 base-pair motif，总计 75,990 个 motif；
  2. 用 BRIQ 建模并计算、归一化 motif 能量；
  3. 形成 inner/outer 两张 \(L\times L\) 能量图；
  4. 通过卷积和 base-pair attention 将能量图加入注意力。
- 本项目此前已完成 Turner stacking 和 RIPE 试验，但 RIPE 已退化并停止，**不能声称已完成 BPfold motif/BRIQ 实验**。
- 旧 `e2a_motif_round54`、`e2b_thermo_round54` 的资产和试验身份必须在 P0 重新核验；未核验前不得写成 BPfold 复现，也不得与新 motif 实验混为单变量对照。
- BPfold 的长序列和非规范配对限制仍然存在，本项目不得借其名义声称解决这两类问题。
- 本地提取文本：`.round56_work/bpfold.txt`。

## 3. 当前证据的保留与回退

### 保留

- 原 xens2 的 Plan-A＋R2D 互补表征、ResNet2D、Turner stacking 和精确非交叉 DP。
- C 精确 DP、流式评测、逐序列预测/分数保存、数据与 checkpoint 哈希、独立复算。
- 预测集合分歧作为不确定性研究的有效现象，但它不是精度突破的替代品。
- 三 R2D 多数投票及其耗时结果，仅作为部署消融和历史参照。
- 时间外集合剔除 33/436 条 TR1c 完全重合序列，保留 403 条；仍需披露同源和 RiNALMo 预训练接触风险。

### 停止或回退

- **D1 通用复合物扩增立即停止。**不把 D1 链加入本合同 JEPA 训练或模型选择；全长负结果保留。
- **RIPE 立即停止。**不得以调整距离偏置参数继续消耗主训练窗口。
- **旧局部 Pair-JEPA 停止。**它是同源特征上的局部掩码预测，不是本合同的序列—结构双视图。更早的 J1-block-EMA 低秩高相关、J2 不收敛和 J3 OOM 均作为历史失败身份保留；本合同是重新设计，不是首次尝试 JEPA，也不能跳过抗坍塌和掩码语义验收。
- **三 R2D 投票退出新架构冠军位置。**恢复原 xens2 作为精度基线，不以较弱 backbone 的校准提升代替原始精度。
- pair-level Platt 不再被当作通用结构校准；多数外部集上的结构 ECE 退化已保留。
- 不恢复已停止的 19,000 nt beam 实验，不用全长可运行冒充长序列有效。
- 不把物理回退、LinearDesign 或不确定性作为本阶段精度主线的替代叙事。

## 4. 新架构：RNA 双视图 JEPA

### 4.1 两个视图

**序列视图**

- 输入：原始 RNA 序列及局部序列上下文。
- 表征：原 xens2 的序列编码器/初始化，输出 \(H\in\mathbb{R}^{L\times d}\)。
- 保留原 ResNet2D、Turner stacking 和精确 DP 的结构输出头，便于同宿主消融。

**结构视图**

- 输入仅限训练期的 canonical-nested 配对、螺旋和 stacking 邻接；不把 RNA–蛋白接触误当 RNA–RNA 配对。
- 将连续 stacking 配对聚合为 helix token，记录 span、长度、闭合配对类型、stacking 数、局部 bulge/loop 和邻接关系。
- 结构编码器 \(E_{struct}\) 输出螺旋级潜在表示。再用固定数量的**学习查询 token**聚合为固定维结构模态 \(Z\in\mathbb{R}^{K\times d}\)，避免向 predictor 泄露被遮蔽 helix 的真实位置或数量。
- 结构视图不是完整三维坐标模型，也不承诺假结、非规范配对或复合物界面。

建议初始 \(d=256\)。只有资源与 pilot 证明需要时，才在 model-dev 上单独比较 \(d=512\)，不得默认扩大容量。

### 4.2 预测器与遮蔽

预测器使用轻量 Transformer，分别支持：

1. **整结构模态遮蔽**：由序列视图预测结构模态 \(Z\)；
2. **部分 helix 遮蔽**：由序列和未遮蔽结构 token 预测被遮蔽结构 token；
3. **序列 span/残基遮蔽**：由结构模态和未遮蔽序列 token 预测被遮蔽的序列表示。

每个样本至少保留一个视图。被遮蔽 helix 的真实 span、位置和数量不得作为 predictor 的输入。学习查询 token 只用于结构编码器聚合，不携带目标标签身份。

### 4.3 目标函数

令 \(M_s\) 和 \(M_z\) 分别为序列与结构遮蔽集合：

\[
L_{pred}=\operatorname{MSE}(\hat H_{M_s},H_{M_s})
+\operatorname{MSE}(\hat Z_{M_z},Z_{M_z})
\]

\[
L=L_{pred}+\lambda\left[L_{SIGReg}(H)+L_{SIGReg}(Z)\right]
\]

- 初始终态预测用 MSE，不引入负样本对比损失。
- SIGReg 默认作用于 target 端，参考 Mol-JEPA，不默认增加 projection head。
- EMA 不是必要组件，部署时 predictor、结构编码器和 mask 全部移除。
- 初始 \(\lambda\) 可在 model-dev 预注册小网格内选择，不得根据外部测试集调节。
- 主结构监督、Turner 和 DP 保持与宿主一致，新增辅助目标必须单独记录权重和梯度贡献。
- 推理只接受序列，不输入 helix、接触图、结构 token 或任何测试标签。

### 4.4 必须否定的旧实现缺陷

- 不允许只预测同一网络自身产生的局部 pair feature 就称双视图 JEPA。
- 不允许把 raw pair label 直接喂给部署模型。
- 不允许 loss 实际落在未遮蔽 token 上却写成 masked prediction。
- 不允许只用 target/prediction 标量 MSE 判定表示有益。
- 不允许把结构 token 的真实位置泄露给 predictor，制造容易但无效的目标。
- 不允许仅因辅助损失下降就宣称结构语义学习成功。

## 5. 数据角色与质量门

### 当前允许的数据

- 主训练：原 xens2 对应的 TR1c 及相同初始化、数据顺序和训练配方。
- model-dev：用于早停、机制选择和超参选择，身份与哈希固定；不得用外部测试代替。
- 结构视图仅从同一训练样本的 canonical-nested 标签构造，不能从测试标签构造。
- 原 xens2、BPfold 和历史评测资产只作对照，不改为当前训练结果。

### 当前禁止的数据操作

- 不把 D1 的 804/202 复合物链作为本合同的通用扩增训练/验证数据。
- 不把 plk、RFAM 等名称直接当作标签质量等级；experimental_pretrain 与 bpRNA 重叠必须保留。
- 不把 unique sequence 数当独立家族数；TR1c 的 42,564 条和 15,521 个 80% 同源簇必须分开描述。
- 不把 canonical-nested 标签称为完整实验结构；假结和非规范配对的删除必须保留在数据卡中。
- 不把 RiNALMo 下游去污染等同于其预训练无接触。
- 不用长度截断、删除失败序列或按模型效果挑样本改善结果。

### 必须新增的数据审计

- 训练/dev/测试的序列、PDB、80% identity 和可用 family/clan 分组审计。
- 结构视图的 helix 数、span、长度、loop、配对密度、来源分层分布。
- 缺失结构、空螺旋、非法配对、重复序列和跨分区视图泄漏检查。
- 每个视图的 `source_id/split/group/operator/params/seed` 可追踪。
- 在生成任何视图之前先固定分区；测试视图不得回流到训练。

## 6. 分阶段实验与停止门

### P0：文献、资产与身份复核

- 固定 Mol-JEPA commit、许可证和引用，保存上述本地研究文件。
- 核对原 xens2 checkpoint、原初始化、数据顺序和主评测身份。
- 核验 E2A/E2B 与 BPfold motif/BRIQ 的关系，未核实不得开始 motif 训练。
- 检查 BPfold 官方 motif 能量表的可用性、许可证和来源哈希。没有合法表格时，不冒充 BRIQ 复现。
- 确认下游 ncRNA family 数据的标签、split 和预训练暴露风险。
- 新工作区 `/home/rna_jev_xens2_dualview_round56`，旧 `/home/rna_jev` 与 `/home/rna_jev_xens2_plus` 的未提交状态不得覆盖或混用。

### P1：双视图工程门

- helix 构造、固定查询聚合、双向 mask、缺失视图 fallback、target 身份和形状测试。
- 证明 masked loss 确实计算在被遮蔽 token 上；防止官方索引歧义移植。
- 验证结构位置不泄漏、推理无结构输入、teacher/predictor 部署调用为零。
- 检查固定/随机/shuffled target、无 SIGReg、无 JEPA 的梯度覆盖和参数初始化。
- 真实 CUDA 小样本 forward/backward、空 helix、单 helix、非法输入、24 GB 显存峰值和数值有限性。
- 测试主要落在现有框架的 `tools/` 与对应 `src/rnajepa/` 模块，不引入 Mol-JEPA 的 torch-geometric 依赖。

### P2：小规模 pilot

先使用受限训练预算验证机制，不直接进入三种子长训：

| 臂 | 唯一变化 | 目的 |
|---|---|---|
| H0 | 同初始化、同数据、同主损失，无 JEPA | 宿主基线 |
| H1 | 加双视图 MSE，不加 SIGReg | 双视图预测是否有效 |
| H2 | H1＋target SIGReg | 抗坍塌是否有独立贡献 |
| H3 | H2 的结构标签 shuffle | 排除仅增加参数/损失项 |
| H4 | 固定随机结构编码器 | 排除随机特征也有效 |
| H5 | 参数量匹配的辅助 MLP | 排除仅增加容量 |
| H6 | 全局均值结构向量替代 helix token | 排除旧区域均值思路 |

- pilot 记录主任务指标、masked 各向损失、target/prediction 方差、有效秩、跨位置/跨序列相关性、CKA 和主损失/辅助梯度 cosine。
- 只有双视图优于 H0、shuffle/random/MLP，并且主任务不退化，才进入三种子正式训练。
- 如果 H2≈H3/H4 或 H2≤H5，停止并撤回结构模态有效的主张；不通过调权掩盖。

### P3：正式单变量训练

- 只冻结一个主候选，不与 motif、D1 或 RIPE 同时改动。
- 三训练 seed=42/2026/3407，保持数据、初始化、主损失和解码一致，保存 best/resume、完整后段曲线和停止原因。
- 先做未加 JEPA 的同条件宿主，再做冻结双视图候选；不得用旧不同训练配方代替直接对照。
- 记录训练与推理参数量、FLOPs、显存和额外结构编码开销。推理无结构编码器和 predictor 前向。

### P4：冻结后统一评测

- 主八 split＋Rfam；CHANRG 只作外部 OOD，不参与调参、早停、校准或选臂。
- complex 只作补充且必须标 canonical-nested、band/exact、长度和失败边界，不进入主精度冠军判定。
- 三个训练 seed 各自三次确定性推理，报告 mean/SD；SD 不代表 seed 方差。
- 主结果含 micro/macro-F1、INF/P/R、空输出、长度、span、raw/own-clean/matched-clean 和独立计数复核。
- 检查未见家族、长程配对和主失败簇，不只报总均值。
- 最终只冻结一次；若因新结果返回修改架构，必须重新标记为开发迭代，不能继续称同一次独立测试。

## 7. BPfold motif 物理先验独立实验

### 资产门先行

1. 优先核验官方 motif 能量表及许可，记录是否 BRIQ、归一化、类别和 SHA256。
2. 若使用 Vienna/CONTRAfold 等近似能量，必须命名为 surrogate motif energy，不能写 BRIQ 或 BPfold 复现。
3. 若需要重新计算 75,990 个 motif，先评估资源、合法性和标签一致性，不默认本机可承担，也不以简化表冒充完整表。
4. 核对 E2A/E2B 是否已有同配方证据；旧负结果不能直接排除新输入方式，但不得重复改名。

### 单变量矩阵

在已冻结的宿主或已证明有效的双视图宿主上，分别执行：

- P0：无 motif，保留 Turner；
- P1：inner/outer motif energy 作为 pair/attention 输入；
- P2：shuffled motif，保持张量形状和参数；
- P3：Turner=0 消融；
- P4：只增加能量图但不加 attention bias，检验输入方式；
- 可选：与已通过的双视图 JEPA 组合一次，必须有独立贡献才保留，不默认叠加。

能量图按 tile 计算，避免为长链额外生成巨大的完整多头矩阵。保留 Turner 与非交叉 DP，不把 motif 输入等同于 Vienna MFE。

只有 P1 在同条件 dev 和外部测试上均超过 P0/P2，才保留物理先验贡献。否则停止，不再以物理叙事补偿精度不足。

## 8. 功能下游任务

下游任务在结构主模型冻结后执行，禁止用下游测试标签参与 JEPA 预训练或主模型选择。优先以下两个方向，不同时铺开更多任务。

### F1：ncRNA 家族/功能分类

- 目的：检验双视图表示是否不仅改善配对，也改善序列到生物功能的迁移。
- 数据：`ncrna_rfamnovel_train/val/test` 及原始标签，先核验字段、家族分组和预训练暴露。
- 比较：RiNALMo-only、原结构宿主、双视图 JEPA、无结构模态和 shuffle 对照。
- 协议：只用 train 拟合分类器，val 选择，test 一次；多 seed，报告 macro-F1、混淆矩阵、必要时 AUROC/AUPRC。
- 不把 Rfam 预训练接触写成绝对 clean OOD。

### F2：RNA–蛋白界面/接触预测

- 目的：把“复合物来源标签”升级为真正的复合物环境任务，检验结构表示是否能迁移到相互作用预测。
- 数据门：从 mmCIF/DSSR 明确提取 RNA–蛋白接触及界面残基，保留 partner 类型、距离阈值、实验方法和分辨率；禁止把 RNA–蛋白接触写成 RNA–RNA 配对。
- 划分：PDB entry＋序列簇联合隔离，负链/非界面残基必须与正样本同来源分层，不能简单随机负采样造成组成捷径。
- 比较：sequence-only、structure-aware、双视图 JEPA、组成匹配随机和长度/GC 对照。
- 指标：以 AUPRC、macro-F1、界面残基召回和分层性能为主，不用大体量非界面残基制造高准确率。
- 若接触图和负样本定义不能通过 QA，任务标 blocked，不能虚构标签或退回普通二级结构充数。

### 暂缓

- LinearDesign/设计重排、结构置信度自动回退、更多功能预测不在本阶段关键路径。只有 F1/F2 或主精度形成稳定结果后再评审，不以湿实验效果、表达提高或免疫原性作任何外推。

## 9. 不确定性与物理回退的后续位置

分歧校准是已证实的正向现象，但降为辅助证据：

- 后续只在冻结的新架构上比较 raw confidence 与 disagreement-aware calibration。
- 物理回退必须分别称 ViennaRNA 热力学回退和 BPfold 神经备用模型，不能混称。
- 必须比较 always-xens2、always-Vienna、always-BPfold、confidence-routing、random-routing 和 length-routing，报告 coverage、risk、真实时间和成本。
- 阈值只在独立 calibration-dev 拟合，不能根据 genC 等结果反向调整。
- 如果 routing 不能超过两条单一路径，只保留人工复核/难例挖掘价值，不写性能优势。

## 10. 论文可发表性与叙事边界

主故事应是**结构模态引导的序列表示学习**，不是投票包装。论文成立至少需要：

1. 双视图相对同宿主具有独立精度贡献；
2. shuffle/random/MLP 对照排除伪收益；
3. 外部未见家族和多种子统计支持；
4. 推理无结构输入且计算成本明确；
5. motif 或功能下游只作为经过验收的补充证据。

若只有 calibration、投票或 dev 提升，不能写“新架构取得显著精度突破”。若双视图失败，保留负结果，不继续通过叙事掩盖。当前不把 SOTA、长序列、复合物理解、三维结构预测、完整物理模拟或湿实验效果写成既定事实。

## 11. 工程与复现要求

- 真实 CUDA，24 GB 显存优先；d、helix token 数、batch、tile 和 checkpoint 以实测为准，不为赶进度缩小科学对照。
- 禁止 CPU 静默降级、测试集调参、旧 checkpoint 名称冒充新机制、静默忽略 missing/unexpected keys。
- 每次训练保存配置、初始化、数据/代码/权重哈希、seed、最佳/恢复点、停止原因、峰值显存和梯度覆盖。
- 逐序列保存预测与置信度，长矩阵不跨序列累积；主结果必须独立复算。
- 正式 raw-sequence 计时独占 GPU，cached embedding 不冒充 raw-sequence。
- 代码修改不得覆盖原工作区或未提交状态；文档更新不代表实验已完成。
- 不提交密码、密钥、`.env` 或私有凭据；Mol-JEPA、BPfold 和本项目许可证分别记录。

## 12. 执行顺序

```text
文献与资产身份复核
→ 原 xens2/数据/接口 P0
→ 双视图 helix/mask/SIGReg 工程测试
→ H0–H6 pilot 与停止裁决
→ 同宿主三 seed 正式训练
→ 主八＋Rfam＋外部 OOD 冻结评测
→ BPfold motif 独立实验（资产许可通过才执行）
→ ncRNA 功能分类与 RNA–蛋白界面（标签门通过才执行）
→ 可选不确定性/物理回退
→ 独立复核、论文与归档
```

任何阶段未通过，不进入下一阶段的科学结论。文档已先于训练完成；当前没有按本合同启动新模型训练。


## 13. 当前数据审计的可复现命令（不是精度训练）

```bash
cd /home/rna_jev_xens2_dualview_round56
PY=/root/miniconda3/envs/lucaone_cu128/bin/python

# 已完成的独立标签、部署、界面指标与精确暴露复核
$PY tools/verify_interface_labels.py
$PY tools/validate_rinalmo_attention_chunk.py
$PY tools/aggregate_interface_results.py
$PY tools/analyze_interface_strata.py
$PY tools/verify_interface_metric_qa.py
$PY tools/audit_pretrain_exact_overlap.py
$PY tools/combine_pretrain_exact_overlap.py

# Ensembl 资产、精确重叠与同源 target
$PY tools/verify_ensembl_download_manifest.py
$PY tools/audit_ensembl_exact_overlap.py
$PY tools/prepare_ensembl_homology_target.py
$PY tools/verify_ensembl_homology_target.py

# 同源查询准备与全量搜索
$PY tools/prepare_pretrain_homology_queries.py
$PY tools/run_pretrain_homology_search.py --source rnacentral_22_active --threads 8 --split-memory-limit 50G
$PY tools/aggregate_pretrain_homology.py --source rnacentral_22_active
$PY tools/run_pretrain_homology_search.py --source rfam_14_9 --threads 8 --split-memory-limit 50G
$PY tools/aggregate_pretrain_homology.py --source rfam_14_9
$PY tools/run_pretrain_homology_search.py --source ensembl_109_and_bacteria_56 --threads 8 --split-memory-limit 50G
$PY tools/aggregate_pretrain_homology.py --source ensembl_109_and_bacteria_56
$PY tools/combine_pretrain_homology.py

# 官方家族映射与 nt 历史快照边界
$PY tools/audit_rfam_family_by_sequence.py
$PY tools/audit_nt_snapshot_provenance.py
```

已有 `homology_round56/<source>` 目录时 runner 会拒绝覆盖，这是历史保护而不是失败。全量重搜前必须先按哈希归档同名目录，禁止删除旧结果或把旧失败目录改成成功。上述命令只复核数据、标签、暴露和工程证据，不能推出新模型精度、完整历史去污染或独立泛化。


## 14. 时间外 RNA 候选审计的复现命令（仍非精度实验）

```bash
cd /home/rna_jev_xens2_dualview_round56
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
T=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/temporal_structure

# 固定日期与完整候选元数据；不得用模型结果修改 cutoff
$PY tools/audit_temporal_pdb_availability.py --cutoff 2025-07-01 --output "$T/availability.json"
$PY tools/freeze_temporal_pdb_metadata.py --cutoff 2025-07-01 --availability "$T/availability.json" --output "$T/metadata_frozen.json"
$PY tools/verify_temporal_pdb_metadata.py --availability "$T/availability.json" --metadata "$T/metadata_frozen.json" --output "$T/metadata_independent_qa.json"

# 精确暴露：三源预训练来源与 TR1c/VL0 分开报告
$PY tools/audit_temporal_exact_exposure.py --metadata "$T/metadata_frozen.json" --output "$T/exact_exposure.json" --fasta "$T/eligible_unique.fasta"
$PY tools/audit_temporal_downstream_overlap.py --metadata "$T/metadata_frozen.json" --train /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl --dev /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl --output "$T/downstream_exact_overlap.json"

# 同源：同一 MMseqs2 17.b804f、identity 与双向覆盖口径
$PY tools/run_temporal_homology_search.py --metadata "$T/metadata_frozen.json" --eligible-fasta "$T/eligible_unique.fasta" --output-root "$T/homology" --prepare-only
for source in rnacentral_22_active rfam_14_9 ensembl_109_and_bacteria_56; do
  $PY tools/run_temporal_homology_search.py --metadata "$T/metadata_frozen.json" --eligible-fasta "$T/eligible_unique.fasta" --output-root "$T/homology" --source "$source" --threads 16 --split-memory-limit 50G
done
$PY tools/aggregate_temporal_homology.py --fasta "$T/homology/queries.fasta" --root "$T/homology" --metadata "$T/metadata_frozen.json" --output "$T/homology_union.json"

# 完整 mmCIF/DSSR 标签与独立 QA；不得用四条 smoke 代替全量
$PY tools/collect_rna_complex_temporal.py --ids-json "$T/candidate_entry_ids.json" --root "$T/full_labels_v1" --workers 12
$PY tools/verify_temporal_label_collection.py --root "$T/full_labels_v1" --ids "$T/candidate_entry_ids.json" --metadata "$T/metadata_frozen.json" --output "$T/label_collection_qa.json"
$PY tools/review_temporal_entry_exclusions.py --root "$T/full_labels_v1" --label-qa "$T/label_collection_qa.json" --metadata "$T/metadata_frozen.json" --output "$T/label_exclusion_review.json"

# 仅在上述标签QA或显式排除审查通过后：以最终亲本恢复序列重新做暴露审计
$PY tools/prepare_temporal_evaluation_sequences.py --root "$T/full_labels_v1" --metadata "$T/metadata_frozen.json" --label-qa "$T/label_exclusion_review.json" --output "$T/evaluation_sequences.json" --fasta "$T/evaluation_unknown.fasta"
$PY tools/audit_evaluation_exact_exposure.py --evaluation "$T/evaluation_sequences.json" --output "$T/evaluation_exact_exposure.json" --fasta "$T/evaluation_input.fasta"
N=$($PY -c "import json; print(len(json.load(open('$T/evaluation_sequences.json'))['records']))")
$PY tools/run_temporal_homology_search.py --metadata "$T/metadata_frozen.json" --eligible-fasta "$T/evaluation_input.fasta" --output-root "$T/evaluation_homology" --prepare-only --expected-count "$N"
for source in rnacentral_22_active rfam_14_9 ensembl_109_and_bacteria_56; do
  $PY tools/run_temporal_homology_search.py --metadata "$T/metadata_frozen.json" --eligible-fasta "$T/evaluation_input.fasta" --output-root "$T/evaluation_homology" --source "$source" --threads 16 --split-memory-limit 50G --expected-count "$N"
done
$PY tools/aggregate_temporal_homology.py --fasta "$T/evaluation_input.fasta" --root "$T/evaluation_homology" --metadata "$T/metadata_frozen.json" --output "$T/evaluation_homology_union.json" --expected-count "$N"
# 若出现精确阳性短序列漏检，直接回查源FASTA后再生成独立resolved文件
$PY tools/resolve_temporal_exact_search_misses.py --union "$T/evaluation_homology_union.json" --fasta "$T/evaluation_input.fasta" --output "$T/evaluation_homology_resolved.json"
```

所有新输出目录均拒绝覆盖。必须保存 `availability.json`、`metadata_frozen.json`、`metadata_independent_qa.json`、`exact_exposure.json`、`downstream_exact_overlap.json`、`homology_union.json`、`label_collection_qa.json` 以及原始 mmCIF/DSSR/parent-map 和逐来源 `search_manifest.json`。标签和同源门均通过前，不启动冻结模型推理；时间外、精确未检出或同源未检出均不能单独证明完整未见或独立家族。
