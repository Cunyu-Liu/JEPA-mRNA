# RNA-JEV 性能主线完整执行提示词：全局结构系综学习与统一 SOTA 证据链

你是一名对 RNA 二级结构、统计物理、结构化预测、数据治理和可复现工程负责的高级研究工程师。请把 RNA-JEV 当作自己的课题，而不是执行一连串孤立的模块实验。核心目标是尽快在八主 benchmark、Rfam、CHANRG 各划分和长序列任务上超过同协议最强 baseline，并形成可投稿顶刊的正向科学故事。

“不要提交负结果”在本合同中意为：**不得以负结果汇总作为最终交付，不得把失败路线包装成论文主线；核心目标未完成时必须继续从模型、数据和评测资产中寻找有依据的解决方案。**但内部必须完整保留失败门、负对照和适用范围，必要消融如实报告，禁止删除不利指标、挑选种子、放宽门限或伪造 SOTA。

## 一、先纠正工作方向：为什么很多工作没有变成核心性能

当前问题不是工程数量不足，而是科学主线与最终目标之间出现了断层：

1. 多轮工作主要证明“某条路线不该继续”，尚未形成可外部验证的正向候选。
2. 反复优化 AUPRC、r@P0.8 或局部 FN，却没有稳定转化为 micro/macro pair-F1 与 stem-F1。
3. 模型选择长期依赖 196 条 VL0，无法覆盖 family、topology、长度、来源和标签质量偏移；单一对照提升距离八主 benchmark SOTA 很远。
4. 局部容量和局部 stacking 假设占据主线，但孤立 pair、长程 pair 和全结构竞争关系需要非局部信息。
5. 统一 baseline 资产尚未补齐：63 个注册 baseline 中 62 个仍有 provenance 阻塞，导致无法形成同协议全面排名。
6. 训练、验收和论文叙事有时围绕“完成阶段”而非“超过谁、在哪个数据切片、提升多少、为何提升”组织。

必须用绝对差距校准研究目标：bpRNA-new 上统一 xens2 micro-F1=0.5531420736，而 MXfold2=0.6688、EternaFold=0.6854，差距分别为 0.1156579264 和 0.1322579264。局部 control 上 0.004 量级且 macro 不保持的变化，不能回答这个核心性能缺口。每个候选在开发启动时都要写明目标 benchmark、当前最强可比系统、需要弥补的绝对差距和预期机制，不得只写“优于 control”。

后续所有工作必须归入三条互相支撑的工作流：**全局结构系综模型、宏泛化数据治理、统一 SOTA 证据链**。不允许再生成与这三条主线无关的模块试验。

## 二、最新实现与不可重走的路线

当前核对到的 GitHub/px2 HEAD 为 `ab9bb10d58380d86b6eb5c1051b7209a08d4e3fa`。px2（`ssh -p 29874 root@px2-jcy.matpool.com`）是指定执行服务器；认证只从安全渠道或现有 SSH 配置获得，禁止密码进入代码、文档、Git、日志或输出。若连接失败，记录真实阻塞并使用经哈希核验的镜像，不得把旧镜像结果冒充 px2 新结果。

### 已经存在、不能冒充新创新的实现

- `src/rnajepa/rlcd.py::combined_loss` 的 `nll` 已调用精确结构化 `negative_log_likelihood`，梯度来自 inside–outside marginals。
- 当前 pilot 已采用 `lambda_nll=1`、`lambda_marginal_bce=0`、`nll_normalization=length`、9,600 steps、终态解冻 20/33 blocks。
- 因此“把 BCE 换成 CRF”“首次引入 inside–outside”“使用结构化 NLL”都不是新的可行提案。
- `src/rnajepa/decision_head.py` 已有 pair representation、Turner residual、flat/2D head 和历史 cascade；新路线必须说明当前实际训练路径用到了什么，不能引用未参与训练的旧模块冒充结果。

### 已关闭的路线

- 容量三种子：`STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED`。不得继续增加 `blocks/hidden/d_z`，不得启动 seeds 1/2。
- SRD：`STOP_SRD_NOT_BETTER_THAN_CURRENT_MEA_OR_CONSTRAINT_FAILURE`。nested SRD micro pair-F1=0.800210，当前 MEA=0.825713；不得重开 stacking bonus、temperature、gamma、阈值或多模型重组搜索。
- 局部 stacking closure：`STOP_STACKING_CLOSURE_FORM_SEED0`。候选 micro pair-F1 相对 local_control +0.0039898，但 macro pair-F1 −0.0056406、macro stem-F1 −0.0194306，且未优于 destroy 对照；不得启动 seeds 1/2 或追加局部模块。
- 当前 `candidate_selected=false`，统一完整 SOTA=0/8。历史 6/8 仅是特定参照口径，不得写成当前统一 SOTA。

这些结果只用于内部排除重复路线，不作为最终论文的核心贡献。

## 三、核心科学问题：局部 pair score 如何获得全结构竞争信息

建议主线聚焦一个明确且有普遍意义的问题：

> **RNA 配对并不是独立的二维分类。一个孤立或长程 pair 是否正确，取决于它与竞争性配对、嵌套区间和整个折叠系综的关系。显式利用全局结构系综上下文，能否把预训练序列表示转化为更准确的非局部配对，并提高跨 family、跨 topology 和跨长度泛化？**

该问题与已有证据紧密衔接：局部 stacking 只能看到 \((i\pm1,j\mp1)\)，无法表达远端竞争与替代折叠；raw 排序提升但 macro/stem 不保持，说明局部分数变化不必然形成更好的全结构。三种子 VL0 描述性审计中，孤立 pair 只有 372 个真值却产生 319 个 FN，recall=0.1424731183；它们占全部 FN 的 10.70%。相比之下，helix boundary/interior recall=0.7057/0.8821。孤立 pair 是明确的高错误环境，但样本量有限，必须在多环境开发面板确认，不能把 VL0 描述直接外推为普遍规律。

当前结构化 NLL 已提供全局监督，但**输入 pair score 的表示仍缺少由全局系综计算得到的上下文**。这正是尚未被排除的表示形式问题。研究选择采用全局系综上下文，而不是只做数据重加权或通用迭代精修：前者不能凭空创造非局部竞争信息，后者若不携带结构分布语义就只是普通模块叠加。二者仍可分别作为数据臂和匹配计算对照，但不能取代主假设。

论文主假设 H-GEC（Global Ensemble Context）：

> 在相同序列表示、参数预算、训练步数与结构化 NLL 下，引入由非交叉结构分布计算的 ensemble context，将主要改善孤立/长程 pair 和全结构 macro 指标；打乱区间上下文或退回独立 sigmoid 上下文后，该优势应消失。

注意：这是待检验假设，不是已成立结论。

## 四、Phase 0：正向路线启动前的四项并行准备

### A. 建立多环境 model-dev 面板

不得继续只用 VL0 决定科学路线。只使用允许用于开发的训练/验证数据，建立冻结的多环境面板：

- VL0=196 保留为既有对照，但不作为唯一选择集。
- 从 TR1c/官方 Train/Validation 的合法开发部分构建 family、80% 簇、长度、pair span、孤立 pair、multiloop/junction 和 topology 分层。
- **进入任何梯度更新的序列及其近邻不得再作为该模型的开发评价样本。**优先使用既有未训练 validation；若需新划分，必须在训练前按 family/80% 簇 group-held-out 冻结，并保存 train/dev 双方 ID 与同源性检查，不能用训练内预测冒充泛化。
- CHANRG Train/Validation 可用于开发；Test/GenA/GenC/GenF 不参与架构、采样、超参或 checkpoint 选择。
- 每个开发子集登记序列 ID、来源、划分依据、标签质量和泄漏状态；禁止从外部测试反选开发样本。

输出 `model_dev_registry.json`，固定主指标、目标子群和每种环境的最低样本数。小样本层只作诊断，不作路线选择；孤立 pair 这类少样本环境须同时报告序列数、pair 数、paired CI 和 family 集中度，避免单个 family 驱动结论。

### B. 补齐前沿与直接前作对照

阅读原始论文和官方实现，不接受只按模块名堆砌。至少覆盖：

- 结构化预测、energy-based learning、neural algorithmic reasoning、iterative refinement；
- RNA 热力学系综、inside–outside、上下文相关折叠与长序列近似推断；
- 最新 RNA foundation model 与二维结构模型中表示传播、全局约束和长程序列建模的方法。

为每篇工作记录 `可迁移数学对象 / 与当前实现差异 / 直接对照 / 不可借鉴的协议`。若全局系综上下文已有直接同构工作，必须准确定位增量，不能把标准 CRF 改名。

### C. 并行解除 baseline 资产阻塞

以 `manifests/foldnar_v1/baseline_registry.json`、`data_registry.json`、`prediction_registry.json` 为起点，逐项恢复官方权重、版本、训练协议和逐序列预测。优先级为每个任务当前最强可比系统，而不是只跑容易复现的弱 baseline。

`BLOCKED_PROVENANCE` 必须记录具体缺项和尝试过的官方来源；不得填 0、删除弱格或把 summary-only 当 frozen prediction。模型训练与 baseline 恢复可以并行，不得互相等待数周。

### D. 冻结合同与身份

开始前记录 `git status --short --branch`、HEAD、未跟踪脚本、GPU、代码/依赖/数据/权重 SHA256。所有新 run 使用新 artifact 目录，保存真实停止原因；不得覆盖历史工作或修改旧合同制造通过。

## 五、Phase 1：Global Ensemble Context Refinement 最小模型实验

### 1. 数学形式

以当前 pair score 为第一遍能量场：

\[
S^{(0)}=f_\theta(H),\qquad
(P^{(0)},M^{(0)})=\operatorname{InsideOutside}(S^{(0)}),
\]

其中 \(P^{(0)}_{ij}\) 是非交叉结构系综中的配对边际，\(M^{(0)}_{ij}\) 是预注册的区间/竞争上下文，例如合法 inside/outside 质量、互斥竞争者总质量和区间占据特征。不得随意加入无法解释的张量。

第二遍只增加由这些全局量定义的上下文残差：

\[
S^{(1)}_{ij}=S^{(0)}_{ij}+\Delta_\theta(H_{ij},P^{(0)}_{ij},M^{(0)}_{ij},g_{ij}),
\]

其中 \(g_{ij}\) 仅包含 span、loop/区间边界等合法结构坐标。训练终点仍为现有精确结构化 NLL；第一遍是否使用辅助 NLL 必须在合同中固定，最小实验优先保持与对照完全相同的终点，不以新增损失混淆表示变量。

第一轮固定使用 \(\operatorname{stopgrad}(P^{(0)},M^{(0)})\)，把“全局上下文作为输入是否有用”与“通过第一遍 DP 反向传播是否有用”分开；所有对照采用相同 stop-gradient 规则。若第一轮成立，才可把可微反馈列为独立后续假设，不得在同一 pilot 同时改变。数值验收至少检查 \(P_{ij}=P_{ji}\)、\(0\le P_{ij}\le1\)、每个核苷酸配对质量不超过 1、canonical/min-loop mask、logZ 有限、非法位置为零，以及短序列相对穷举的 logZ/marginal 误差；合同须给出浮点精度与容差。

该设计的科学增量不是 inside–outside 本身，而是：**把全结构分布计算产生的非局部上下文重新注入 pair representation，使第二遍能量能够表达竞争配对的协调。**

### 2. 最小对照矩阵

所有臂使用同初始化、数据、batch、步数、参数/近似 FLOPs、终态解冻与 MEA：

- `matched_control`：相同额外参数和计算，但上下文通道置零或使用不含全局信息的匹配局部量。
- `ensemble_context`：真实非交叉 inside–outside 上下文。
- `destroy_interval_shuffle`：在合法样本/区间内按冻结规则打乱上下文，保留值分布但破坏 pair 对应。
- `independent_context`：以独立 sigmoid 分数替代结构系综边际，区分“多一次非线性处理”与“全局结构信息”。

先验证 DP 数值、对称性、canonical/min-loop、梯度守恒和短序列穷举等价，再做 2-step smoke，最后才允许 seed0 pilot。不得未经 pilot 直接启动三种子。

### 3. 目标子群与放行门

主要科学终点：

- 孤立 pair recall 与 precision；
- 长 span pair recall，按预注册分桶；
- 竞争环境中的 FN 改善；
- micro/macro pair-F1、P/R 与 micro/macro stem-F1。

seed0 必须同时满足：

- micro pair-F1 非负且 macro pair-F1 严格提高；
- micro/macro precision 和 stem-F1 降幅均不超过 0.005；
- 孤立或长程目标子群严格改善，且不能只由单一 family/序列贡献；
- 相对两个 destroy/independent 对照，在 micro/macro pair-F1 与 stem-F1 上保持预注册优势；
- raw AUPRC/r@P0.8 只作机制辅助，不抵消结构失败。

通过后才允许预注册三种子和多环境确认。若失败，最终回复必须报告真实门，但不得把本次任务收缩成“负结果已完成”；应回到错误分解，提出不同数学机制的下一候选，而不是调整门限或堆叠模块。

### 4. 超参数策略

第一轮只改表示形式，沿用已冻结的 9,600-step、batch、head/backbone LR、warmup、20/33 解冻与 MEA 配置，确保因果清晰。若优化不稳定，只允许对**所有臂共同**做预注册的小规模稳定性校准；后续超参优化必须给 control 相同搜索预算。

使用开发面板的 successive halving/等预算选择，不根据外部测试调参。不得搜索 decoder temperature、gamma、阈值来救援模型。保存所有 trial，而非只保存最佳 trial。

## 六、Phase 2：宏泛化数据治理，独立验证数据价值

连续出现 micro 与 macro 分歧，说明必须检查数据加权与困难环境覆盖，但不能直接认定“类别不平衡就是根因”。

### 1. 数据资产与质量审计

- TR1c=42,564 条/15,521 簇；VL0=196。
- CHANRG Train=123,223、Validation=14,070、Test=14,070、GenA=12,499、GenC=4,424、GenF=1,797。
- 注册八主 benchmark、Rfam、D1/PDB/DSSR、时间外候选 1805 entries 和 7224/19000 nt 长链。
- 检查重复 ID、modified-parent、0-based pair、对称、min-loop、partner 冲突、canonical/nested、crossing/pseudoknot 和标签来源。
- PDB 回查 mmCIF entity/poly-seq-scheme/atom_site；未观测残基不生成负标签。
- 标签分为 `EXPERIMENTAL/COMPARATIVE_INFERRED/DATABASE/THERMODYNAMIC_TEACHER/SIMULATED`，不得混称实验真值。

当前 `LABEL/MASK/DATA_SHIFT=0` 只表示既有 FN 归因规则未触发，不代表数据已证明 clean。

### 2. 分布与泄漏

统计长度、GC、family/clan/80% 簇、pair density/span、helix 数与长度、孤立 pair、loop 类型、multiloop 阶数、junction、topology、来源和质量。

四层泄漏必须覆盖：下游精确/同源、结构相似、预训练暴露、时间与标签来源。状态只用 `EXACT/HOMOLOGY/NOT_DETECTED_UNKNOWN/UNCHECKED`；未检出不等于 clean。raw、own-clean、matched-clean 分栏。

### 3. 单变量数据实验

只有当开发面板证实某环境样本不足、来源冲突或宏平均持续受损时，才开放对应数据臂：

- `source_family_balanced`：按预注册 family/80% 簇/来源有效样本量重加权；
- `topology_longspan_balanced`：补足孤立、长程、复杂 junction 环境，但不复制测试近邻；
- `quality_filtered`：只剔除有明确规则与回查证据的冲突标签，报告删除守恒。

每个数据臂与相同模型 control 使用同总优化步数和等预算采样。主终点是 macro pair/stem、目标环境保持率以及 micro 不被牺牲；不得把数据清洗、采样和新架构同时改变后归因给单一机制。

## 七、Phase 3：从多环境开发到外部 OOD 与 SOTA

### 放行条件

- 候选通过 seed0、三种子和冻结多环境开发门；不只靠 VL0。
- 模型、数据、checkpoint、decoder 和全部超参在外部读取前冻结。
- raw 导出、逐序列预测和独立指标重算一致。
- 同协议最强 baseline 的预测或合法可复现资产已就绪；缺项必须显式列出。

外部结果只能用于最终评价，不得根据弱格返回改模型后仍称同一次独立评估。若需修改，必须产生新版本并重新经过开发门。

### 完整 benchmark 与 baseline

覆盖八主 benchmark、Rfam、CHANRG Test/GenA/GenC/GenF、topology/length/temporal/source shift、预训练暴露分层和长序列。至少比较：

- ViennaRNA MFE/centroid/MEA、RNAstructure、CONTRAfold、EternaFold、MXfold2；
- BPfold、UFold、eFold、SPOT-RNA、RNAformer、Kirigami、ContextFold、E2Efold、DEBFold；
- RiNALMo、ERNIE-RNA、RNA-FM、NucleicBERT、structRFM；
- xens2、最佳单 Plan-A、R2D、H0 和当前冻结候选；
- 长序列加入 LinearPartition 类方法；Eco-RNA 待官方资产可复现后纳入。

同一冻结候选用于同协议全部任务，报告 micro/macro pair-F1、P/R、stem/junction/topology、长度/span、覆盖率、paired 95% CI、Holm 校正和计算成本。只有超过同序列、同指标、同覆盖的最强可比 baseline 才写 SOTA；单模型/集成、不同训练协议、精确/稀疏执行分栏。

## 八、长序列必须从模型假设延伸到计算假设

- 精确 inside–outside 为 \(O(L^3)\)，不得因在 VL0 有效就宣称支持 19,000 nt。
- 短序列先证明全局上下文机制；长序列再检验保留非交叉区间语义的块稀疏/候选区间近似。
- 不截断、不删除 7224/19000 nt；编码器、候选生成和完整结构输出分开验收。
- 先测候选 pair/helix recall，再测 pair/stem/topology、长程召回、显存、FLOPs 和实际时延。
- 近似执行标 `APPROXIMATE_SPARSE`，不得冒称精确全局 DP；与 LinearPartition 类方法比较覆盖率与精度。
- 梯度检查点只改变显存，不写成数学或架构创新。

## 九、正向论文故事：每一段必须由实验承接

最终论文不按“做过哪些模块”组织，而按以下推理链组织；只有获得正向证据的段落才进入主结论：

1. **科学现象**：孤立、长程和竞争环境中的配对错误，揭示局部 pair 表示与全结构决策之间的缺口。
2. **科学问题**：全局结构系综信息能否反哺 pair 表示，而不仅用于训练梯度或最终解码？
3. **方法回答**：Global Ensemble Context Refinement 用非交叉 inside–outside 产生上下文，再形成第二遍协调能量。
4. **机制证明**：真实上下文优于匹配 control、interval shuffle 和 independent context；收益集中在预期的孤立/长程环境，并转化为 pair/stem。
5. **数据回答**：宏泛化是否还受 family/topology/source 分布约束？只用单变量数据实验回答。
6. **泛化回答**：同一冻结模型跨 family、topology、长度、时间与来源偏移保持收益。
7. **性能回答**：在八主 benchmark、Rfam、CHANRG 和长序列上，与同协议最强 baseline 做完整比较。
8. **计算回答**：说明精确与近似的边界，证明长序列增益不是截断或降低覆盖率得到。

为每个 claim 建立 `claim → phenomenon → intervention → expected observation → artifact → limitation`。没有直接实验支持的句子不得进入摘要或标题；不得把历史失败清单当作最终论文交付。

## 十、工程留痕与最终交付

每个 run 保存命令、代码/依赖/权重/数据 SHA256、seed、初始化/resume、batch 顺序、可训练参数、实际 backbone 更新、逐步 loss、LR、gnorm、显存、checkpoint 规则、停止原因、逐序列预测和指标。至少进行运行者自检、独立脚本重算和交接终审。

最终回复必须优先给出：

- 相对 xens2 与同协议最强 baseline 的逐任务提升，而不是只相对自己的 control；
- 八主 benchmark、Rfam、CHANRG、长序列的完整覆盖、缺失项和 SOTA 状态；
- 全局系综上下文对孤立/长程 pair、macro pair/stem 的直接证据及 destroy/independent 对照；
- 数据分布、清洗、泄漏、采样与标签质量对结论的影响；
- 哪些叙事已被实验强证明，哪些仍未证实；
- 修改文件、真实测试命令与结果、artifact、文档和哈希。

若本轮没有超过 baseline，不得宣称核心目标完成，也不得以“负结果已交付”结束课题。必须指出当前最强候选、距离最强 baseline 的数值差距、下一项有独立科学依据的干预及其预期可观察结果；同时如实列出失败门，禁止用 raw-only、单 seed、smoke、单外部弱格或协议混用制造正向结论。
