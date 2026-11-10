# RiNALMo 全量重跑、结构对比与 LinearDesign 扩展执行合同（2026-11-10）

> 本文件是用户最新指令的统一执行覆盖，优先级高于此前 structRFM 路线及旧停止记录。先完成本文件、执行提示词覆盖、spec/tasks 和训练台账同步，再启动新训练。当前只完成文档与只读核对，未启动新实验。

## 1. 不可变边界与历史隔离

1. 后续所有新训练、诊断、评测和下游候选统一使用 **RiNALMo** 骨干，最终仍为单骨干、单任务头、单 checkpoint、无集成。训练期辅助 teacher、对比投影头、predictor 和 mask 不得进入部署前向。
2. structRFM 的所有 checkpoint、run、日志、预测、QA、容量裁决和历史结论原样保留，统一标为 **historical_structRFM_diagnostic**。不得删除、覆盖、恢复混入新 run，也不得与新 RiNALMo 对照拼成单变量结论或跨骨干种子均值。
3. 新实验使用独立命名空间 `round52_rinalmo_*` 和独立 artifacts 清单。旧 RiNALMo 数字可作历史参照，但不冒充本轮重新训练结果。每个科学臂重新训练 seed=42/2026/3407，steps=0、每500步 VL0、patience=6、best VL0，保存完整后段曲线并自然早停。
4. 常规结构精度只评 **主 benchmark 八 split + Rfam**。不以 CHANRG 或其他测试集调参、早停、校准、选臂；是否恢复额外 OOD 评测等待用户后续明确指令。LinearDesign 是独立下游任务，不计入主八 split。
5. 所有训练/GPU 验证使用真实 CUDA，禁止 CPU 静默降级。非计时阶段按实测峰值并行独立 seed/合法臂，最大化有效显存和 GPU 利用率，但不超订、反复 OOM、修改科学配置或用无资格任务填卡；正式 raw-sequence 测速及独占 QA 必须独占 GPU。
6. 测试集不得参与调参、早停、校准、阈值或负样本构造。smoke、proxy、训练集、单 seed 和未齐套结果不得写成最终结论。每个实现、训练、评测、统计、提交和 PPT 步骤继续执行实现合同、真实 CUDA/artifact、独立复算三轮检查。

## 2. 方案取舍

- **采用：核心优化先行，结构对比条件放行，LinearDesign 独立下游。** 先解决 loss/optimizer 和容量归因，再验证真正的 Pair-JEPA；结构对比作为单变量辅助目标，LinearDesign 只评价冻结模型，避免混改。
- **不采用：立即并行 A0–A5、D0/D1、JEPA 和物理项。** 这会破坏单变量归因，也无法判断收益来自优化、数据、目标还是算力。
- **不采用：照搬调研方案的第二宿主和未验证增强。** 用户已固定 RiNALMo，因此不引入 UFold/RNAformer 第二宿主，不作跨架构方法主张；反向、互补、未验证补偿突变、任意次优结构不得默认作为正对。

## 3. RiNALMo 核心重跑矩阵

### 3.1 R0 与 R-replay

先建立 **R0-RiNALMo** 新控制：使用 RiNALMo 官方预训练初始化、当前 R6 对应的 head/scorer/RIPE 配置、NLL-sum、原四目标、原分层 LR/解冻 schedule 和合法 batch/数据顺序。不得直接读取 structRFM R6 optimizer 或把旧 structRFM replay 数值换成 RiNALMo 名义。

R0 三种子自然收敛后，固定 seed42 best 权重+终态 optimizer/scheduler/stream，seed2026 复核，按保存队列重放 NLL-sum、NLL-length 和 shadow Adam；不更新权重。记录逐项梯度、cosine、m/v、update RMS、update/weight 和长度斜率。若状态/批次不能恢复，停止机制归因。

### 3.2 L-length 与 O-LR

- L-length：仅将 NLL sum 改为 length，其余同 R0。
- O-LR：NLL 保持 sum，仅将 backbone LR 从原值降至预注册低值；正式运行前按 R0 配置明确记录，不沿用 structRFM 数值而不核对。
- 两臂独立三种子自然收敛后才允许主八+Rfam 评测。只有两个单因子均有 VL0/稳定性依据才评审交互，不默认做 L+O 或 U-ramp。

### 3.3 容量诊断

先分析同配置训练/VL0 误差、冻结表征探针和优化稳定性。PairMLP 容量旋钮以真实代码为准，不把无效 `hidden` 当宽度。只有训练欠拟合证据支持时，才分别做 C-wide（latent 490→1024）与 C-local（两层轻量局部分离卷积）；不得混改。记录参数量、FLOPs、峰值显存、VL0 和长度/span 分桶。训练拟合改善但 VL0 不改善即停止扩参。

### 3.4 Pair-JEPA

顺序固定为 J-gate→J-mlp→J1-block-EMA→J1-fixed→J-random：

- J-gate：对称/非空 mask、masked 直接信息断裂、零 masked 输入梯度、stop-gradient、EMA 仅在 optimizer step 后更新、推理 teacher_calls=0/student_score_calls=1；包括失败注入和真实 CUDA。
- J-mlp：RiNALMo 同 student、同初始化、同 CRF/distill/RLCD/cal 与部署 scorer 的必要对照；旧 structRFM J-mlp 不复用为本轮对照。
- J1-block-EMA：结构化 sequence-block cross-product、候选 helix/反对角线 block 与少量随机 pair 的对称 mask，轻量局部 mixer；target 读取完整 pair grid 并 stop-gradient，直接 scorer/DP/Turner 不变。
- fixed/random 分别只改变 target 机制。每25步记录 target/prediction 总方差与逐通道方差、均值范数、有效秩、跨位置/跨序列相关性、teacher-student 距离、direct NLL 和梯度 cosine。泄漏、明显坍塌或不超 J-mlp 时不作 JEPA 有效主张。

### 3.5 D0/D1 与 motif/长序列

D0 复用届时按 VL0 选出的 RiNALMo 稳定基座；D1 从同一 RiNALMo 官方初始化，只新增审计通过的 canonical-nested 复合物链，不能从 D0 续训后仍称纯数据单变量。去泄露对象改为 RiNALMo 下游训练、VL0、主八+Rfam 及 RiNALMo 预训练语料风险；structRFM 预训练审计不再作为新模型的替代证明。

D1 数据门包括 exact+80% identity/query-coverage≥0.8、PDB entry+序列簇连通分量联合隔离、raw→canonical_nested 守恒、冲突/多构象、教师、manifest 和 SHA256。RNA–蛋白接触不得转成 RNA–RNA 配对。主八+Rfam 分表，complex holdout 独立报告。RiNALMo 原文明确预训练来源为 RNAcentral、nt、Rfam、Ensembl，36M唯一序列经MMseqs `--min-seq-id 0.7 -c 0.8`形成17M簇；因此Rfam必须披露预训练接触风险，不得写成绝对无预训练污染的clean OOD。

P-motif-input 只新增 outer/inner motif energy 输入，保留 Turner 与 DP，对照 PairMLP、shuffled-motif 和 Turner=0。长序列分别验证位置边界、完整覆盖、span≥30/≥100、O(L²) 显存、训练长度分布与 DP；只有长度不足经审计成立才单独做 L-balance。RiNALMo原文训练输入上限1024 tokens，微调对超过1022 nt的序列随机裁剪；新实验禁止沿用该裁剪来制造全长结果，超过当前支持边界必须标PARTIAL。

## 4. 调研目录的可借鉴内容：结构感知对比扩展

来源：`RNA-SSP-对比学习-开题调研/spec.md`、`tasks.md`、`checklist.md` 与17页开题汇报。它们提供待验证方案而非本项目结果；A0–A5 在该目录中未执行。

### 4.1 数据与样本关系前置门

在任何对比训练前完成：

1. TR1c/VL0/主八+Rfam 的 exact、80% identity、Rfam family/clan 审计，形成可追溯 `tr1c-v2`；不得因旧精确去重通过而跳过同源审计。
2. 先 split/group 后生成视图。每个视图保存 `source_id/split/group/operator/params/seed`，增强不得跨分区构造正负对。
3. 同序列 mask/dropout 是模型视图而非生物结构不变断言；简单裁剪默认禁用，避免破坏长程配对。
4. 二核苷酸保持 shuffle 只生成候选困难负样本。以训练参考标签的结构距离排除过近假阴性，负池不得包含 VL0 或测试序列。
5. 长度、GC、配对密度、标签来源和最近邻分布必须报告；用匹配子集/分层分析区分家族效应和浅层组成效应。

### 4.2 单变量对比矩阵

以已通过 Pair-JEPA 阶段裁决的 RiNALMo 稳定单模型为宿主，直接结构监督、部署 scorer、数据顺序和解码保持不变：

| 臂 | 唯一新增内容 | 主要否定问题 |
|---|---|---|
| CL0 | 无，复用冻结宿主 | 同条件基线 |
| CL1 | 仅同序列输入视图，无对比项 | 增强本身是否有效 |
| CL2 | 普通表示 InfoNCE：视图正对+随机负 | 任意对比正则是否有效 |
| CL3 | 仅将负样本换成审计通过的组成匹配困难负 | 结构样本关系是否超过随机负 |
| CL4 | 仅增加距离筛选/困难加权，其他同 CL3 | 假阴性控制是否有独立贡献 |
| CL5 | 仅增加可微固定维输出表示项，其他同 CL4 | 输出空间是否值得保留 |

每臂独立 seed=42/2026/3407、自然收敛；投影头仅训练期存在。λ/τ/ε/δ 仅在 VL0 或 train-dev 内预注册扫描，一次只改一个参数。记录表示范数、相似度分布、主任务损失和坍塌风险。必须同时报告同 step 与同 GPU-hour/FLOPs 对照，不能把多视图额外算力当方法收益。

主终点为 bpRNA-new 逐序列 macro-F1 相对同条件宿主的配对增量，strict micro、P/R、INF 同报；Rfam 独立报告，不合并均值。目标增量可参考调研的 0.01，但不是预设结果。若 CL2≈CL3，撤回结构困难负样本的特异主张；CL4 无独立收益则不声称距离控制有效；CL5 无收益或损害主任务即停止输出空间路线。固定 RiNALMo 单宿主时不作跨架构普遍有效结论。

## 5. LinearDesign 下游扩展合同

原文：Zhang et al., Nature 621, 396–403 (2023)，DOI=`10.1038/s41586-023-06127-z`。本地28页论文 SHA256=`2d8e7b5188a46ca6372f52e3abf785ae38f86af3b6394d52f51c91d3c46bfa9c`。原文算法针对编码区，以 DFA/lattice parsing 联合优化 `MFE – λ|p|log(CAI)`；生物实验结果只能作为文献背景，不是本项目结果。

### 5.1 LD0：既有工程资产与差异归因

保留官方 commit=`f0126ca89a8b853088b4bccfd2cc8c378d3678be`、编译和三条官方示例，不重复已通过的工程验收。官方代码/库继续只在 third_party，不违反再分发限制。

正式扩展前必须解释生成器 MFE 与 ViennaRNA 2.7.2 独立复算差异：逐项固定 Vienna 版本、dangling 参数、终止密码子、能量模型和输入序列；至少对照原文 ViennaRNA 2.4.14、`-d0` 与当前环境。差异未闭合不得声称 MFE 一致。

### 5.2 LD1：计算设计空间与对照

1. 固定论文114条 UniProt 蛋白（78–3333 aa）的可获得清单、figshare/补充材料版本和 SHA。按蛋白 ID 预先划分 design-dev/design-test，测试蛋白不得调 λ/beam/约束。
2. 分别执行 exact search 与 beam search（原文参考 b=500），λ=0 的 MFE-only 与联合优化分开；λ 网格只在 design-dev 锁定。原文 exact 在实用长度近二次、beam 近线性和 spike 约1.2%能量差距均为待复核结果，不写成已复现。
3. 对照至少包含野生型（若有序列）、确定性的 max-CAI、同义密码子随机（seed42/2026/3407）及 GC/组成匹配随机。OptimumGene/GeneOptimizer 若没有合法可执行资产，只列 quoted/缺失，不用自制规则冒充。
4. 区分 unconstrained 核心算法与论文约束版。论文约束包括5′前15 nt的单独枚举/低结构选择及避免≥33 bp长双链；不得把约束版和无约束版混为单变量。UTR、密码子表、宿主偏好、遗传密码和修改碱基分别记录。
5. spike 与 VZV gE 的 A–H/A–E 等设计只有从论文补充材料或 figshare 恢复到原序列并核验后才可复评；当前28页主文/Extended Data不包含全部补充序列，不得凭图读数伪造。

### 5.3 LD2：冻结 PRISM 的独立下游评价

- 输入蛋白必须逐位不变，无内部终止，末端终止、CDS长度、密码子映射、CAI、GC全部独立复算。
- 输出 LinearDesign 的 MFE/结构、ViennaRNA 独立 MFE/结构和 PRISM 预测分开。PRISM 对参考结构只报告**计算模型间一致性**，不得当实验 GT 或真实结构恢复率。
- 评价 MFE、CAI、GC、配对比例、最长 helix、长程配对、PRISM 置信度及风险排序；confidence 使用既有 VL0 冻结校准，不能对 design-test 再校准。
- UTR–CDS 交互按原文用 ViennaRNAfold `-d0` 独立评价；CDS-only 与 UTR+CDS 分开，不得把 UTR 相互作用归因给 PRISM 的训练目标。
- 所有设计序列必须全长输入。超过 RiNALMo/任务头支持或显存边界者标 PARTIAL/unsupported，禁止截断 spike 等长 CDS、只评短链后宣称整体优势。
- 生成器单线程 CPU 时间与 PRISM 独占 GPU raw-sequence 时间分开；exact/beam 的运行时间、内存和能量差距独立复算，不把训练加速写成设计速度优势。

### 5.4 禁止的生物学外推

无湿实验时，不写半衰期延长、表达提高、免疫原性增强、疫苗有效性或真实热力学稳定性证明。LinearDesign 输出不得作为真实标签反向蒸馏 PRISM，也不得用于一般非编码 Rfam RNA。

## 6. 统一阶段门与验收

```text
文档与 RiNALMo 资产门
→ R0/R-replay → L-length/O-LR → 容量诊断
→ J-gate/J-mlp/J1-block-EMA/fixed/random
→ 结构对比数据门 → CL1/CL2 → CL3/CL4 → 条件 CL5
→ D0/D1 → motif-input/长序列
→ 冻结候选的主八+Rfam、置信度与独占 raw-sequence 验收
→ LinearDesign LD0/LD1/LD2（使用冻结候选）
→ 独立复核、PPT副本、归档
```

LinearDesign 的算法资产审计、补充数据下载和 MFE 参数差异诊断可与训练并行，但不能提前用未冻结模型产生最终设计结论。结构对比若未获支持，不阻塞已经独立批准的 D0/D1，但必须完整保留负结果。

最终结构评测仅主八 split×3 seed×3 repeat+Rfam，含 greedy/Turner=0、micro/macro-F1、P/R/INF、population SD、共同支持 paired bootstrap 和主八 Holm；新增支持单列。概率、结构排序、长度/span 和 raw-sequence 速度按原执行提示词验收。所有 negative/PARTIAL/空桶均保留，三轮检查与独立复核前不得写 PPT。

## 7. 来源指纹与当前状态

- LinearDesign PDF：`2d8e7b5188a46ca6372f52e3abf785ae38f86af3b6394d52f51c91d3c46bfa9c`。
- 对比学习 `spec.md`：`8a2c36c826f076c2a39122e18a441edbc9a1e8f72a8c081a89e83fc0145a4976`。
- 对比学习 `tasks.md`：`d265c6777f0c88fef9f83cc6a3cbd5fe681f20b9ea3fd81e4a6d37d1ddfb5aae`。
- 对比学习 `checklist.md`：`665852a3dfe854ac212fb30e7687507609e80d3284badbc8ed044238d241f471`。
- 对比学习开题 PPT：`1799d2e91837549f4820bb0f6683253a58a0d5e9dc66310dda33ae9a470a293d`。

当前只完成论文/调研阅读、旧合同核对和本计划写入；未启动 R0、J-gate、CL、D0/D1 或 LD1/LD2，未修改 PPT。下一阶段是只读核验 RiNALMo 权重、旧 R6 对应配置、数据/预训练污染证据和可恢复训练接口，形成 R0 实际命令后再做三轮工程预检。
