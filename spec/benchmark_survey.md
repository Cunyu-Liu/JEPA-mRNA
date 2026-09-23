# RNA 二级结构预测：Benchmark 与先行工作调研报告

> **用途**：为「把决策模型范式迁移到 RNA 二级结构预测」的预印本（一周内投稿）提供可核查的评测集、基线、协议与先行工作清单。
>
> **两个待验证的核心贡献**：
> - **C1 免 DP 的校准（DP-free calibration）**：决策头单次前向直接输出**已校准**的碱基配对概率矩阵 p̂_ij，省掉 McCaskill 配分函数/inside-outside。
> - **C2 层级决策级联（hierarchical decision cascade）**：L0 块粗筛 → L1 螺旋级 → L2 碱基级细化，把 O(L²) 降到近似 O(L/w + w²)，在匹配平均 FLOPs 下优于纯精算/纯近似。
>
> **方法与可信度声明**：本报告的规模、划分、URL 与基线数字，均尽量取自论文原文 / 官方仓库 / Europe PMC / arXiv 一手来源，并逐条给出链接。凡本报告未能从一手来源确认的字段，一律显式标注「**未核实**」，不做推测。**不同论文报告的 F1 不可直接横比**（测试集、去冗余、指标定义、是否含假结/非规范配对均不同），详见第三节。
>
> **本报告完成日期**：2026-09-24。

---

## 一、主评测集（in-distribution）

### 1. ArchiveII

- **规模**：**3,975 条**结构，来自 **10 类 RNA**（原始定义）；在社区再整理版本中常被记为 3,966 条（如 BPfold 论文，见下）。来源论文为 Sloma & Mathews 2016（RNA 22:1808–1818）。原始 3975 这一数字被 E2Efold/UFold 等广泛引用为「来自 10 种 RNA 类型的 3975 个 RNA 结构」。
  - 来源：[E2Efold 解读（含 3975/10 类型表述）](https://cloud.tencent.com/developer/article/1589184)；[BPfold, Nat Commun 2025, 明确写 "archiveII (n = 3966 RNAs)"](https://doi.org/10.1038/s41467-025-60048-1)
- **来源论文与下载入口**：
  - 原始描述：Sloma, M. F. & Mathews, D. H. *Exact calculation of loop formation probability identifies folding motifs in RNA secondary structures.* RNA 22, 1808–1818 (2016)。（**原文 URL 未核实**）
  - 实际下载入口：多随训练框架分发，例如 UFold 数据仓库 [github.com/uci-cbcl/UFold](https://github.com/uci-cbcl/UFold) 与 [RiNALMo benchmark 的 Zenodo 打包](https://zenodo.org/records/13821093)。**ArchiveII 官方独立托管 URL 未核实。**
- **许可**：**未核实**（条目源自 RNA STRAND v2.0 / PDB 等，各条许可需逐条确认）。
- **标准 split**：**ArchiveII 本身是纯测试集，没有官方 train/valid**。训练集通常用 RNAStralign 或 bpRNA-1m；评测时把 ArchiveII 作为 held-out 测试。
- **常用指标**：碱基对层面的 F1 / precision / recall（= PPV / SEN）。
- **已知陷阱**：
  1. 含同源冗余序列（家族不均衡，tRNA/rRNA 类占比高，容易拉高平均分）。
  2. 假结通常被排除或单独统计。
  3. 不同论文对「ArchiveII」的再整理版本长度/条数略有差异（3975 vs 3966），**跨论文比较前必须核对具体文件**。
- **去冗余版本**：社区常对 ArchiveII 做 CD-HIT 去冗余后评测，但**阈值（80%/90%）、执行者与首个引用论文未核实**。可确定的同类做法是：SPOT-RNA 系列对 PDB 派生集统一用 **CD-HIT-EST 80% + BLAST-N（e-value 10）** 去冗余（见 §1.5）。

### 2. bpRNA-1m

- **规模**：**超过 102,318 条**单分子（single-molecule）RNA 二级结构，来自 **7 个既有数据库来源**，由 bpRNA 工具统一注释；论文自述「over 100,000 single-molecule, known secondary structures」。
  - 来源：[Danaee P, Rouches M, Wiley M, Deng D, Huang L, Hendrix D. *bpRNA: large-scale automated annotation and analysis of RNA secondary structure.* Nucleic Acids Res 46(11):5381–5394 (2018)](https://doi.org/10.1093/nar/gky285)（[PMC6009582](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6009582/)，开放获取）
  - 代码：[github.com/hendrixlab/bpRNA](https://github.com/hendrixlab/bpRNA)
- **标签是怎么来的（关键）**：bpRNA-1m 是一个**元数据库（meta-database）**，把 7 个来源的**「已知」二级结构**汇编后用 bpRNA 工具统一解析、注释成 dot-bracket。**它并不是用 RNAfold 做机器标注得到的**——原文表述为「known secondary structures」。这 7 个来源的具体清单**未逐一核实**（常见涉及 Rfam、RNA STRAND、PDB 等，但请勿在正式稿中直接断言）。
  - **对训练/评测可信度的影响（务必在论文中写明）**：
    1. 这些标签大多来自**比较序列分析/协方差模型等计算注释**，而非逐条直接实验测定，因此**存在系统性噪声**。
    2. **SPOT-RNA 原文明确指出**：bpRNA 数据「slightly noisy」，导致模型 precision 的**上界约为 96%**（"the slightly noisy data in bpRNA lead to an upbound around 96% for the precision"）。
       - 来源：[SPOT-RNA, Nat Commun 10:5407 (2019)](https://www.nature.com/articles/s41467-019-13395-9)
    3. 结论：**在 bpRNA-1m 上测得的 F1 上限被人为压低**，且训练/测试同源时会高估真实泛化；因此必须补充实验标签集（PDB 派生）与跨家族集（bpRNA-new）。
- **TR0 / TS0 / VL0 的定义与规模**（由 SPOT-RNA 定义，被 UFold/MXfold2 等大量沿用）：
  - 构建流程：bpRNA-1m (v1.0) 共 **102,348 条** → 用 CD-HIT-EST 去除序列相似度 >80% → 14,565 条 → 去除含 PDB 结构者、并限制长度 ≤500 nt → **13,419 条** → 随机划分。
  - **TR0 = 10,814（train），VL0 = 1,300（valid），TS0 = 1,305（test）**；三者平均长度约 130 nt。
  - 来源：[SPOT-RNA, Nat Commun 10:5407 (2019), Methods/Datasets](https://www.nature.com/articles/s41467-019-13395-9)
  - 文件级 train/valid/test 清单：见 SPOT-RNA / UFold / RiNALMo benchmark 仓库（[UFold](https://github.com/uci-cbcl/UFold)、[rna-llm-folding](https://github.com/sinc-lab/rna-llm-folding)）。

### 3. RNAStralign（亦写作 RNAStrAlign）

- **规模**：**37,149 条**，来自 **8 类 RNA**；来源论文 Tan et al. 2017（TurboFold II 相关）。常被用作**训练集**。
  - 来源：[E2Efold 解读（37149/8 类型）](https://cloud.tencent.com/developer/article/1589184)；[UFold 训练数据描述](https://github.com/uci-cbcl/UFold)
- **拼写变体**：`RNAStralign`（文献常用）与 `RNAStrAlign`（亦见于部分代码库），指向同一数据集。
- **常用子集 / split**：UFold 把训练集记为「RNAStralign train + TR0 + 增强数据 + PDB train」，测试为「ArchiveII, TS0, bpRNA-new, PDB test(ts1/ts2/ts3)」。
  - 来源：[UFold, NAR 50(2):e14 (2022)](https://doi.org/10.1093/nar/gkab1074)；[UFold README](https://github.com/uci-cbcl/UFold)
- **许可 / 官方下载 URL**：**未核实**（随 UFold / Mathews lab 分发）。

### 4. bpRNA-new（跨家族泛化测试集）

- **来源**：由 **SPOT-RNA2** 构建，用于**家族交叉（family-wise / cross-family）泛化**评测。
  - 来源：[Singh J, Paliwal K, Zhang T, Singh J, Litfin T, Zhou Y. *Improved RNA secondary structure and tertiary base-pairing prediction using evolutionary profile, mutational coupling and two-dimensional transfer learning.* Bioinformatics 37(17):2589–2600 (2021)](https://doi.org/10.1093/bioinformatics/btab165)；代码：[github.com/jaswindersingh2/SPOT-RNA2](https://github.com/jaswindersingh2/SPOT-RNA2)
- **为什么它是跨家族测试集**：从 **Rfam** 构建，**只保留 bpRNA-1m 及任何其他数据集中未出现过的 RNA 家族**，因此训练/测试之间**家族不重叠**，直接检验对未见家族的泛化。
  - 来源（家族排除策略）：[UFold 相关描述：bpRNA-new 源自 Rfam，"包含来自约 1500 个新 RNA 家族"，"bpRNA-1m 或任何其他数据集中出现的 RNA 家族被排除"](https://blog.csdn.net/cc0319cc/article/details/139984725)
- **Rfam 版本**：**存在分歧，未核实**——部分二手来源写 Rfam 14.1，部分写 14.2。**投稿前请以 SPOT-RNA2 原文为准。**
- **规模**：常见说法约 **1,500 个新家族**、序列数约 **1,539 条**；**确切条数未核实**。
- **哪些论文用了它**：UFold、MXfold2、RiNALMo benchmark、ERNIE-RNA、RNADiffFold、BPfold、DEBFold 等（均可在 Europe PMC 检索到 "bpRNA-new" 命中）。
  - 例：[RiNALMo benchmark, Brief Bioinform 26(2):bbaf137 (2025)](https://doi.org/10.1093/bib/bbaf137)；[ERNIE-RNA, Nat Commun 16:10076 (2025)](https://doi.org/10.1038/s41467-025-64972-0)；[RNADiffFold, Brief Bioinform 26(1):bbae618 (2024)](https://doi.org/10.1093/bib/bbae618)

### 5. PDB 衍生集 ts1 / ts2 / ts3

- **ts1**：**67 条** X 射线高分辨（**<3.5 Å**）**非冗余单链** RNA。
  - 构建流程：2019-03-02 从 PDB 下载 <3.5 Å 的 RNA → CD-HIT-EST 80% 去冗余得 226 条 → 随机分 TR1=120 / VL1=30 / TS1=76 → 去除与 TR0 相似度 >80% 者降为 69 → 再用 BLAST-N（e-value 10）降为 **67**。标签由 **DSSR** 从 3D 结构导出。
  - 来源：[SPOT-RNA, Nat Commun 10:5407 (2019), Methods/Datasets](https://www.nature.com/articles/s41467-019-13395-9)
- **ts2**：**39 条** NMR 解出的结构（从 641 条 NMR 结构中，用 CD-HIT-EST 80% + BLAST-N 对 TR0/TR1/TS1 过滤得到）；平均长度约 51 nt。
  - 来源：同上（[SPOT-RNA](https://www.nature.com/articles/s41467-019-13395-9)）
- **ts3**：由 **SPOT-RNA2** 引入（UFold 明确写「按照 SPOT-RNA2 中使用的分区，将 PDB 序列分为 ts1/ts2/ts3」）。**ts3 的精确定义（同一性阈值 / 分辨率过滤 / 是否含复合物）与规模，未核实。**
  - 来源（ts3 归属 SPOT-RNA2）：[UFold 解读](https://blog.csdn.net/cc0319cc/article/details/139984725)
- **三档的意义**：均为**实验/高分辨 3D 派生**标签，质量高于 bpRNA-1m；因此常被用作「精确标签」评测与迁移学习目标。**陷阱**：ts1/ts2 与训练集 TR0/TR1 之间用 CD-HIT 80% + BLAST-N 过滤，但 **80% 是最宽松的阈值**，仍可能有远端同源残留。

---

## 二、OOD / 泛化 / 特殊集

### 6. RNA-Puzzles

- **性质**：社区**盲测**（blind prediction），原为独立赛事，后并入 CASP。最新一轮 **RNA-Puzzles Round V**：**23 个 RNA 结构**、**18 个团队**参与。
  - 来源：[Bu F, … Westhof E, Miao Z. *RNA-Puzzles Round V: blind predictions of 23 RNA structures.* Nat Methods 22(2):399–411 (2025)](https://doi.org/10.1038/s41592-024-02543-9)
- **是否适合二级结构评测**：**主要评估对象是 3D 结构**；二级结构可作中间产物评估，但**不是官方主指标**。适合作为「真实盲测 / 时序外推（temporal hold-out）」的**加分项**，不建议作为唯一主集。
- **常用评测方式 / 工具**：RNA-Puzzles toolkit 提供基准数据集、结构操作与评估工具（含 3D 指标）。
  - 来源：[Magnus M, et al. *RNA-Puzzles toolkit.* Nucleic Acids Res 48(2):576–588 (2020)](https://doi.org/10.1093/nar/gkz1108)
- **轮次**：Round I–V 均已发表；**各轮靶标数未逐轮核实**。

### 7. CASP15 / CASP16 RNA 靶标

- **CASP15（2022）首次纳入 RNA 结构预测**（承接原 RNA-Puzzles 内容），2022-12 公布结果。
  - 官方评估论文：[Das R, Kretsch RC, Simpkin AJ, … Westhof E. *Assessment of three-dimensional RNA structure prediction in CASP15.* Proteins 91(12):1747–1770 (2023)](https://doi.org/10.1002/prot.26602)
  - 结果解读（中文）：[旗思妙想：CASP15 比赛结果详解](https://www.szbl.ac.cn/info/1019/4435.htm)
- **2024–2026 使用 CASP15 RNA 的论文**：
  - [Nithin C, Pilla SP, Kmiecik S. *When does molecular dynamics improve RNA models? Insights from CASP15 and practical guidelines.* Comput Struct Biotechnol J 27:4201–4211 (2025)](https://doi.org/10.1016/j.csbj.2025.10.003)
  - RhoFold+ 在 RNA-Puzzles / CASP15 天然 RNA 靶点上做回顾性评估（[Shen T, et al. Nat Methods 21(12):2287–2298 (2024)](https://doi.org/10.1038/s41592-024-02487-0)）。
- **具体靶标清单 / 二级结构层面的评测方式**：**未核实**（官方主评 3D）。**CASP16（2024）的 RNA 靶标与评估**：**未核实**。

### 8. PseudoBase++（假结）

- **性质**：PseudoBase 的扩展版（假结二级结构数据库，Leiden 维护），支持检索/格式化/可视化。PseudoBase 收录过去约 25 年间的**数百条**假结记录。
  - 来源（含 "over 250 records" 表述）：[PseudoBase++ 摘要转述](https://m.zhangqiaokeyan.com/open-access_resources_thesis/0100037208616.html)
- **规模**：**未核实**（不同版本条数不同；PseudoBase++ 相对 PseudoBase 有扩充，确切条目数需查原文）。
- **是否被当作假结评测标准**：**部分被用作假结评测来源**，但并非社区统一标准；假结评测更多以「PDB 派生集中含假结的子集」或「按可去除最小配对数定义的 pseudoknot pairs」进行（如 SPOT-RNA 对 TS1 中 40 条含假结 RNA 的单独评测）。
  - 来源：[SPOT-RNA, Nat Commun 10:5407 (2019), Table 3](https://www.nature.com/articles/s41467-019-13395-9)

### 9. 长链基准（16S / 23S rRNA 等）

- **常用做法**：以 **16S rRNA（约 1.5 kb）、23S rRNA（约 2.9 kb）** 等 rRNA 以及长 lncRNA 作为长链压力测试。LinearFold / LinearPartition 系列面向长序列，报告速度随长度近线性。
  - 来源：[LinearPartition, Bioinformatics 36(Suppl_1):i258–i267 (2020)](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i258/5870487)；[LinearFold, Bioinformatics 35(14):i295–i304 (2019)](https://doi.org/10.1093/bioinformatics/btz375)
- **公认的长链基准集与长度分桶**：**未核实**（未见统一标准；BPfold 论文亦指出「现有数据集序列长度多 <600 nt，长序列问题尚未解决」）。
  - 来源：[BPfold, Nat Commun 16:5856 (2025), Discussion](https://doi.org/10.1038/s41467-025-60048-1)

### 10. 跨家族 / 跨长度 / 跨 GC 外推衰减的专门评测

- **跨家族（最成熟）**：
  1. **bpRNA-new**（SPOT-RNA2）：家族不重叠的 family-wise 测试集（见 §1.4）。
  2. **RiNALMo benchmark**：显式设计「**increasing generalization difficulty**」的四级基准（ArchiveII、bpRNA、bpRNA-new、PDB-RNA），并做 **cross-family partitions** 与 **homology-aware** 划分；结论是「**low-homology 场景下泛化仍是重大挑战**」。
     - 来源：[Zablocki LI, et al. Brief Bioinform 26(2):bbaf137 (2025)](https://doi.org/10.1093/bib/bbaf137)；[arXiv:2410.16212](https://arxiv.org/abs/2410.16212)；[代码与数据 github.com/sinc-lab/rna-llm-folding](https://github.com/sinc-lab/rna-llm-folding)
  3. **BPfold**：以 `Rfam12.3–14.10`（**10,791 条**）作为 family-wise 数据集，专门检验未见家族的泛化。
     - 来源：[Zhu H, et al. *Deep generalizable prediction of RNA secondary structure via base pair motif energy.* Nat Commun 16:5856 (2025)](https://doi.org/10.1038/s41467-025-60048-1)
- **跨长度 / 跨 GC 外推的专门协议**：**未发现明确的、被广泛复用的标准协议**。常见做法是「按长度分桶报告 F1 随长度衰减曲线」（如 SPOT-RNA 报告 F1 随长度下降，长链 >1000 nt 时纯 ML 方法不如热力学方法）。
  - 来源：[SPOT-RNA, Nat Commun 10:5407 (2019), Discussion & Supplementary Fig. 1](https://www.nature.com/articles/s41467-019-13395-9)
- **建议**：C2 若要主打「近似 O(L/w + w²)」，**跨长度外推曲线（F1 vs. L、FLOPs vs. L）应由本项目自建**并作为核心图，因为社区尚无现成协议可复用。

---

## 三、横向对比模型表（核心交付物）

> **⚠️ 数字可比性警告（务必写进论文）**：下表中的 F1 **不可跨行直接比较**。原因：(a) 测试集不同（ArchiveII / TS0 / bpRNA-new / ts1-3 各异）；(b) 去冗余阈值不同；(c) 是否含假结/非规范配对、是否在碱基对层面 vs 序列层面统计不同；(d) 部分方法只在特定子集上报数。**只有在「同一测试集文件 + 同一去冗余 + 同一指标定义」下，数字才可比。** 本表把「可核实的具体数值」与「仅核实到来源、未核实具体数值」分开标注。

| 方法 | 年份 | 类别 | 测试集与报告指标 | 需配分函数/DP? | 推理复杂度 | 代码/权重 |
|---|---|---|---|---|---|---|
| **RNAfold (ViennaRNA)** | 2011 | 热力学 DP（MFE / 配分函数） | 常作通用基线；ArchiveII 等上的具体 F1 **未逐项核实** | **需要**（MFE 与配分函数均 O(L³)） | 三次 | [开源](https://doi.org/10.1186/1748-7188-6-26) |
| **CONTRAfold** | 2006 | 判别式（CRF）+ DP，**输出碱基配对概率** | TS1 上 ensemble defect = **0.24**（SPOT-RNA 原文） | **需要**（DP 算后验） | 三次 | [论文](https://doi.org/10.1093/bioinformatics/btl246) |
| **RNAstructure** | 2010 | 热力学 DP | 常作基线；具体 F1 **未核实** | **需要** | 三次 | [开源](https://doi.org/10.1186/1471-2105-11-129) |
| **LinearFold** | 2019 | 线性时间**近似** DP（5'→3' + beam search） | 主打速度；近似 MFE | 不做配分函数（beam 近似 DP） | **近似线性** | [开源](https://github.com/LinearFold/LinearFold) · [论文](https://doi.org/10.1093/bioinformatics/btz375) |
| **LinearPartition** | 2020 | **首个线性时间**配分函数 + 碱基配对概率（beam 近似） | 输出 BPP 矩阵；精度/速度具体数 **未核实** | 近似配分函数（beam） | **近似线性** | [开源](https://github.com/LinearFold/LinearPartition) · [论文](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i258/5870487) |
| **UFold** | 2021 | 判别式深度学习（图像化 FCN） | ArchiveII / TS0 / bpnew / TS1-3；**比热力学提升约 10–30%**，比其他学习方法最高 **+27%** | **不需要** | 单次前向，~**160 ms/序列（≤1600 bp）** | [开源+权重](https://github.com/uci-cbcl/UFold) · [论文](https://doi.org/10.1093/nar/gkab1074) |
| **SPOT-RNA** | 2019 | 判别式深度学习（ResNet + 2D-BLSTM 集成） | **TS1：F1 ≈ 0.69**；TS1 ensemble defect = **0.19**；与次优 mxfold 比 MCC +9%、F1 +10%+ | **不需要** | 单次前向；CPU 62 条/540 s，GPU 62 条/39 s | [开源+权重](https://github.com/jaswindersingh2/SPOT-RNA) · [论文](https://www.nature.com/articles/s41467-019-13395-9) |
| **SPOT-RNA2** | 2021 | 判别式深度学习 + 进化信息/突变耦合 | ts1/ts2/ts3、bpRNA-new；相对次优对齐法 +10%/+2%/+9%（原文表述） | **不需要** | 单次前向 | [开源](https://github.com/jaswindersingh2/SPOT-RNA2) · [论文](https://doi.org/10.1093/bioinformatics/btab165) |
| **MXfold2** | 2021 | 深度学习 + 热力学整合（**仍用 DP**） | TS0 / bpRNA-new 等；具体 F1 **未核实** | **需要**（整合热力学打分） | 三次 | [论文](https://doi.org/10.1038/s41467-021-21194-4)（仓库 URL **未核实**） |
| **E2Efold** | 2020 | 端到端深度学习（unrolled 算法 + 合法性后处理） | ArchiveII / RNAStralign；原文称在多指标上显著优于传统方法 | 不需显式配分函数（用 unrolled + 后处理） | 训练 O(L³)；推理含后处理 | [开源](https://github.com/ml4bio/e2efold) · [论文 arXiv:2002.05810](https://arxiv.org/abs/2002.05810) |
| **RNA-FM** | 2022 | RNA 基础模型（微调） | 下游结构预测；统一 benchmark 中**非最优**（RiNALMo 基准结论） | 取决于下游头 | O(L²) 注意力 | [开源+权重](https://github.com/ml4bio/RNA-FM) · [arXiv:2204.00300](https://arxiv.org/abs/2204.00300) |
| **RiNALMo** | 2024 | RNA 基础模型（650 M，33 层） | 统一 benchmark 中**两个明显更优的模型之一** | 取决于下游头 | O(L²) | [开源+权重](https://github.com/lbcb-sci/RiNALMo) · [arXiv:2403.00043](https://arxiv.org/abs/2403.00043) |
| **RNA-MSM** | 2024 | MSA-based RNA 基础模型（96 M） | 统一 benchmark；**非最优** | 取决于下游头 | O(L²) | [开源](https://github.com/yikunpku/RNA-MSM) · [NAR 52(1):e3](https://doi.org/10.1093/nar/gkad1031) |
| **mRNABERT** | 2025 | mRNA 语言模型（偏 mRNA 设计/功能） | 面向 mRNA 序列设计；**非二级结构标准基线** | — | — | [论文](https://doi.org/10.1038/s41467-025-65340-8) |
| **RNAformer** | 2024 | 深度学习（axial attention，homology-aware） | homology-aware 评测；具体 F1 **未核实** | **不需要** | O(L²) 单次前向 | [预印本](https://doi.org/10.1101/2024.02.12.579881)（仓库 URL **未核实**） |
| **trRosettaRNA** | 2023 | 深度学习（1D/2D 几何预测）+ 能量最小化 → 3D | RNA-Puzzles / CASP15（**3D**） | 不需要 | O(L²) | [论文](https://doi.org/10.1038/s41467-023-42528-4) |
| **RhoFold+** | 2024 | RNA 语言模型 + 深度学习 → 3D | RNA-Puzzles / CASP15 天然 RNA 靶点（**3D**） | 不需要 | — | [论文](https://doi.org/10.1038/s41592-024-02487-0) |
| **CDPfold** | 2019 | **CNN + 动态规划（DP）** | 3 个家族基准；原文声称约 **+30%** 预测成功率 | **需要 DP**（名字即「CNN + DP」） | CNN + O(L³) DP | **是否开源未核实** · [论文](https://doi.org/10.3389/fgene.2019.00467) |
| **RNADiffFold** | 2024 | 生成式（**离散扩散**） | ArchiveII / TS0 / bpRNA-new；具体 F1 **未核实** | 不需配分函数 | 多步采样 | [论文](https://doi.org/10.1093/bib/bbae618)（仓库 **未核实**） |

**同一测试集上「相对可比」的数字（可谨慎引用）**：
- **TS1 上**：SPOT-RNA 原文给出了 SPOT-RNA / mxfold / CONTRAfold / CentroidFold 等的 **precision–recall 曲线**与 **ensemble defect（0.19 / 0.24 / 0.25）**，这些是同一测试集、同一指标定义下的直接对比（[来源](https://www.nature.com/articles/s41467-019-13395-9)）。
- **统一 LLM benchmark 上**：RiNALMo benchmark 在同一架构、同一数据划分下比较了 RNABERT / RNA-FM / RNA-MSM / ERNIE-RNA / RNAErnie / RiNALMo，**这是少见的「同一测试集可比」对比**（[来源](https://doi.org/10.1093/bib/bbaf137)）。

---

## 四、概率与校准的先行工作（对 C1 最关键）

### 11. 有没有人系统评测过「碱基配对概率的校准」（ECE / 可靠性图 / Brier / NLL）？

- **结论：未发现**。多次检索（含 "calibration + base pairing probability + RNA"、"ECE + RNA secondary structure"、"reliability diagram + RNA structure"）**均未找到**对 RNA 碱基配对概率做**系统性校准评测（ECE / 可靠性图 / Brier score / NLL）**的论文或数据集/协议。
- **最接近的先行工作（务必在论文中引用并区分）**：
  1. **Ensemble defect**：衡量概率系综与真实结构的期望偏差（0 = 完美）。SPOT-RNA 用它比较了 **SPOT-RNA 0.19 vs CONTRAfold 0.24 vs CentroidFold 0.25**（TS1）。**但 ensemble defect 衡量的是系综质量，不是概率校准（calibration）**——一个模型可以 ensemble defect 低但概率系统性偏乐观/悲观。
     - 来源：[SPOT-RNA, Nat Commun 10:5407 (2019)](https://www.nature.com/articles/s41467-019-13395-9)
  2. **BPfold 的 "confidence index"（2025）**：用「网络原始 contact map 与约束精修后 contact map 的余弦相似度」构造置信度，并与 F1 做相关（ArchiveII 上 Pearson **0.728**，Rfam12.3–14.10 上 **0.692**）。**这是「置信度/可靠性估计」，不是概率校准**（没有可靠性图，也没有 ECE/Brier 意义下的校准目标）。
     - 来源：[Zhu H, et al. Nat Commun 16:5856 (2025)](https://doi.org/10.1038/s41467-025-60048-1)
- **对本项目的意义（强信号）**：**C1 的「校准」卖点在 RNA 二级结构领域基本是空白**。审稿人大概率没见过 ECE/可靠性图，因此你需要**明确定义**：以「每对碱基是否为真实配对的二值标签」为 ground truth，对 p̂_ij 计算 **ECE、可靠性图、Brier、NLL**，并与配分函数法（ViennaRNA/LinearPartition）的概率做同口径对比。**这是 C1 成立的关键，也解释了为什么这条线此前没人做。**

### 12. CONTRAfold 与 LinearPartition 输出的碱基配对概率质量如何？有没有评测其「概率可靠性」（而非仅 F1）？

- **未发现**对二者做**校准层面**评测的工作。已知的相关证据：
  - CONTRAfold 的概率被 SPOT-RNA 用 **PR 曲线**和 **ensemble defect（TS1 上 0.24）**评估过——属于「概率/系综质量」，**非校准**。
    - 来源：[SPOT-RNA, Nat Commun 10:5407 (2019)](https://www.nature.com/articles/s41467-019-13395-9)
  - **LinearPartition 的概率是 beam-search 近似**：它用 beam size（默认 100）近似配分函数，因此 BPP 是**近似后验**，理论上会系统性偏离精确 McCaskill 后验。**但该偏差是否被评测过、偏差幅度多少，未核实。**
    - 来源：[LinearPartition README（beam size 默认 100，"first linear-time partition function and base pair probabilities"）](https://github.com/LinearFold/LinearPartition)
- **对本项目的意义**：这给 C1 一个**天然的对照组**——「LinearPartition 的近似 BPP 是否比你的单次前向校准概率更差（在 ECE/Brier 上）？」如果能证明你的 p̂ **既省算力又更校准**，C1 的叙事就完整了。

### 13. **CDPFold 重点核查（单点风险）——结论与用户假设相反**

- **核查结论**：**CDPfold 不是条件扩散，也不绕开 DP。** 它是一个 **2019 年的「CNN + 动态规划」方法**，标题即 *A New Method of RNA Secondary Structure Prediction Based on **Convolutional Neural Network and Dynamic Programming***。
  - 来源：[Zhang H, Zhang C, Li Z, Li C, Wei X, Zhang B, Liu Y. Front Genet 10:467 (2019)](https://doi.org/10.3389/fgene.2019.00467)
- **它到底怎么做**：用 CNN 预测**每个碱基的配对概率**（并引入高斯权重考虑茎区两侧配对），**再对得到的概率施加增强的 DP 求最优二级结构**。也就是说：**它输出概率，但最终结构仍靠 DP 得到；它并没有「用扩散直接生成配对概率矩阵」，也没有绕开 DP。**
  - 来源（原文摘要与 Data/Methods）：[Front Genet 10:467 (2019)](https://doi.org/10.3389/fgene.2019.00467)
- **其输出概率是否被评测过校准**：**未发现**任何对 CDPfold 概率做校准评测的工作（它本身也未把校准作为卖点）。
- **代码/权重**：**未核实**是否公开（原文与检索均未见明确的官方代码仓库）。
- **对 C1 的风险评估（重要）**：
  - 用户担心的「CDPfold 已经用扩散直接出概率、且概率已校准」**不成立**——CDPfold 既非扩散、也仍需 DP、且无校准证据。
  - **真正需要盯防的不是 CDPfold，而是 §11 提到的 BPfold confidence index 与 SPOT-RNA/UFold 这类「单次前向直接输出配对概率矩阵」的判别式模型**：它们确实**单次前向就产出 L×L 概率/得分矩阵**，只是**没有把「校准」当卖点、也没有做过 ECE 评测**。这意味着 C1 的**新颖性应定位在「校准」而非「单次前向出概率」**（后者已有先例）。

### 14. 有没有工作把「免 DP 但概率校准」当卖点？有没有「单次前向出概率矩阵」的既有方法？

- **「免 DP 但概率校准」作为卖点**：**未发现**。这是 C1 的机会窗口。
- **「单次前向出概率矩阵」的既有方法（有，且不少）**：
  - **SPOT-RNA / SPOT-RNA2**：最后一层 sigmoid，直接输出 L×L 的**配对概率**（再阈值化得到结构），**不跑配分函数/DP**。
    - 来源：[SPOT-RNA, Nat Commun 10:5407 (2019)（"sigmoid function converts the output into the probability of each nucleotide being paired"）](https://www.nature.com/articles/s41467-019-13395-9)
  - **UFold**：图像化 FCN，单次前向输出配对得分矩阵（~160 ms/1600 bp）。
    - 来源：[UFold, NAR 50(2):e14 (2022)](https://doi.org/10.1093/nar/gkab1074)
  - **CDPfold**：CNN 输出配对概率（但随后用 DP）。
    - 来源：[Front Genet 10:467 (2019)](https://doi.org/10.3389/fgene.2019.00467)
- **结论**：C1 必须**明确与上述区分**——它们能「单次前向出概率」，但**从未被证明是校准的**；C1 的新意 = **首次给出免 DP 且经 ECE/Brier/可靠性图验证的校准概率**。建议在论文中直接把 SPOT-RNA/UFold 的概率拿来做 ECE 对照实验。

---

## 五、评测协议细节（可直接照做）

### 15. 二级结构评测指标的标准定义

- **碱基对层面（base-pair level）**：把预测结构与真实结构的**配对集合**逐对比较。
  - **TP**：预测且真实存在的碱基对；**FP**：预测但不存在；**FN**：真实但未预测。
  - **Precision = PPV = TP / (TP + FP)**（正例预测正确率）。
  - **Recall = Sensitivity = SEN = TP / (TP + FN)**。
  - **F1 = 2·P·R / (P + R)**（精确率与召回率的调和平均）。
  - 注意：这些指标**只统计碱基对，不直接统计配对是否满足嵌套（nesting）约束**；因此预测出非法（交叉）配对时需先做合法性处理或单独报告。
- **F1 的两种聚合口径**（论文中必须写明）：**micro（把所有序列的 TP/FP/FN 汇总后再算）** vs **macro（每条序列算 F1 再平均）**，两者结果不同。
- **SEN / PPV**：即上面 Recall / Precision 的同义写法，在 RNA 领域常成对出现（源于 RNA-Puzzles / 3D 评测传统）。
- **INF（Interaction Network Fidelity，相互作用网络保真度）**：
  - 定义：在「相互作用（interactions）」层面（通常 = 碱基对 **+** 碱基堆积/stacking，即 3D 语境下的相互作用集合）计算 **SEN 与 PPV 的调和平均**。0 最差、1 最好。
  - 用途：主要用于 **RNA 3D / 2D 结构比较**（RNA-Puzzles、CASP 评估体系），是 3D 主指标之一；二级结构论文一般不主用 INF，但在与 RNA-Puzzles/CASP 对齐时会用到。
  - 来源：[RNA-Puzzles toolkit（提供 3D 结构比较与评估工具，含 INF 等指标）Nucleic Acids Res 48(2):576–588 (2020)](https://doi.org/10.1093/nar/gkz1108)；[CASP15 RNA 评估, Proteins 91(12):1747–1770 (2023)](https://doi.org/10.1002/prot.26602)（**INF 的精确公式与「interactions 是否含 stacking」的原文定义，未逐字核实**）
- **长链是否用不同指标**：**未见统一约定**。常见做法是沿用同一套 F1/SEN/PPV，但**按长度分桶报告**，或补充「最长可处理长度」「显存/时间随 L 的变化」。
  - 参考：[SPOT-RNA 按长度报告 F1 衰减](https://www.nature.com/articles/s41467-019-13395-9)

### 16. 速度 / 延迟评测的标准协议

- **现状：社区没有公认的统一速度基准协议。** 各论文自报口径差异大，常见要素如下（**需在你的论文中全部写明**）：
  - **硬件**：CPU 型号/核数或 GPU 型号（例：SPOT-RNA 用单线程 32 核 Intel Xeon E5-2630v4 CPU 与单张 Nvidia GTX TITAN X GPU 分别报告）。
    - 来源：[SPOT-RNA, Nat Commun 10:5407 (2019), Discussion](https://www.nature.com/articles/s41467-019-13395-9)
  - **batch size**、**是否含后处理 DP**（对 C1/C2 尤其关键：你的卖点是「免 DP」，所以必须**明确区分**「仅前向」与「前向+后处理」的耗时）。
  - **长度分桶**：按序列长度分桶报告（如 <200 / 200–500 / 500–1000 / >1000 nt），报告每桶平均延迟或吞吐。
  - **参考点**：UFold 报「~160 ms/序列（≤1600 bp）」（[来源](https://github.com/uci-cbcl/UFold)）；LinearFold/LinearPartition 报相对 ViennaRNA 的加速比（[LinearPartition](https://github.com/LinearFold/LinearPartition)、[LinearFold](https://doi.org/10.1093/bioinformatics/btz375)）。
- **建议（针对 C2）**：在**匹配平均 FLOPs** 下与「纯精算（ViennaRNA/LinearPartition 精确模式）」和「纯近似（大 beam 线性法）」比较，同时报告 **wall-clock 与 FLOPs 两条曲线 vs. 长度 L**，并明确后处理是否含 DP。**由于无公认协议，你自建协议时务必把上述所有要素写全，才具备可复现性。**

### 17. 论文里常被 reviewer 质疑的评测缺陷（及规避做法）

| 缺陷 | 表现 | 规避做法 |
|---|---|---|
| **dev == test** | 验证集与测试集是同一批序列 | 严格三分（train/valid/test），并在附录给出文件哈希/清单 |
| **测试集与训练集同源** | 同家族/同源序列同时出现在训练与测试 | 用 CD-HIT（80% 或更严）+ BLAST-N 过滤；优先用 **family-wise**（bpRNA-new）划分 |
| **只报单一 seed** | 结果不可复现、方差未知 | 至少 **3–5 个随机种子**，报告均值 ± 标准差，并做显著性检验 |
| **标签来自预测而非实验** | 在 bpRNA-1m（计算注释）上刷分，掩盖真实精度 | 必须补充**实验/高分辨派生集**（ts1/ts2/ts3）与 **family-wise** 结果；明确 bpRNA 的 ~96% precision 上限 |
| **指标定义/聚合口径不一** | 与他法数字不可比 | 统一 micro/macro、统一是否含假结/非规范配对，并**只与同口径结果对比** |
| **用测试集做模型选择** | 早停/超参在 test 上调 | 只用 valid 选择；test 只在最后评一次 |
| **忽略假结/非规范配对** | 报高分但漏掉难例 | 单独报告假结子集与非规范配对的 F1（当前 SOTA 非规范配对 F1 仅约 0.22） |
| **长序列上结论外推** | 在短序列上训练却在长序列上宣称有效 | 按长度分桶报告；长链单独给结果与失败分析 |

- 相关来源：[RiNALMo benchmark 明确强调 homology-aware / cross-family 与「低同源泛化困难」](https://doi.org/10.1093/bib/bbaf137)；[BPfold 指出长序列与非规范配对的评测短板](https://doi.org/10.1038/s41467-025-60048-1)；[机器学习 RNA 2D 结构在实验数据上的基准（Justyna M, Antczak M, Szachniuk M. Brief Bioinform 24(4):bbad153, 2023）](https://doi.org/10.1093/bib/bbad153)

---

## 六、结论与建议

### 18. 推荐的 benchmark 套件（区分「必须做」与「加分项」）

**A. 必须做（主集 + 跨家族 + 速度 + 统计）**

1. **主集（in-distribution）**：
   - **TS0**（bpRNA-1m 的 1,305 条测试集）——社区最常用、可与 UFold/MXfold2/SPOT-RNA 直接对齐（**理由**：文献最多、划分明确）。
   - **ArchiveII**（3,975 / 3,966）——传统基线通用测试集（**理由**：热力学方法都在此报数）。
   - **训练集**：用 **RNAStralign(37,149)** 或 **TR0(10,814)**（与所选主集配套）。
2. **实验标签集（精确标签）**：**ts1（67）+ ts2（39）**（**理由**：标签来自 3D 高分辨结构，能对冲 bpRNA 的标签噪声；ts3 待你核实其定义后加入）。
3. **跨家族（OOD）**：**bpRNA-new**（family-wise）（**理由**：唯一被广泛复用的跨家族集；直接支撑 C1/C2 的泛化主张）。
4. **速度/复杂度基准**：**自建**，按长度分桶，报告 **wall-clock 与 FLOPs 两条曲线**，明确「是否含后处理 DP」，并给出「匹配平均 FLOPs」下的对比（**理由**：无公认协议，必须自建且写全）。
5. **统计检验**：**≥3–5 个随机种子**；报告 mean ± std；对主集与 OOD 做 **配对检验**（paired t-test / Wilcoxon），并报告效应量（**理由**：SPOT-RNA 原文即用 paired t-test 报显著性，可作为先例）。

**B. 加分项（有则显著增强说服力）**

1. **长链外推**：16S（~1.5 kb）/ 23S（~2.9 kb）rRNA 与长 lncRNA，报告 F1/FLOPs 随 L 的衰减曲线（直接支撑 C2 的 O(L/w + w²) 主张）。
2. **跨长度/跨 GC 外推**：自建分桶实验，报告衰减。
3. **校准专项（C1 的核心证据）**：在 ts1/ts2/TS0 上对 p̂_ij 计算 **ECE、可靠性图、Brier、NLL**，并与 **ViennaRNA（精确配分函数）**、**LinearPartition（近似）**、**SPOT-RNA/UFold（单次前向概率）** 同口径对比。
4. **假结/非规范配对**：单独报告子集 F1（对标当前约 0.22 的非规范配对上限）。
5. **真实盲测**：在 **RNA-Puzzles Round V（23 个靶标）** 或 **CASP15 RNA 靶标** 上做回顾性评估。

### 19. 数据可获取性（准确入口 vs 需申请/失效）

**可公开下载（给准确 URL）**：
- bpRNA 工具 + bpRNA-1m：[github.com/hendrixlab/bpRNA](https://github.com/hendrixlab/bpRNA)（论文 [NAR 46(11):5381–5394](https://doi.org/10.1093/nar/gky285)）
- SPOT-RNA / ts1、ts2、TS0 等：[github.com/jaswindersingh2/SPOT-RNA](https://github.com/jaswindersingh2/SPOT-RNA)（数据镜像 `http://sparks-lab.org/jaswinder/server/SPOT-RNA/`；**该镜像是否仍在线未核实**）
- SPOT-RNA2（bpRNA-new 相关）：[github.com/jaswindersingh2/SPOT-RNA2](https://github.com/jaswindersingh2/SPOT-RNA2)
- UFold 数据与权重：[github.com/uci-cbcl/UFold](https://github.com/uci-cbcl/UFold)（权重在其 Google Drive 链接中，见 README）
- RiNALMo benchmark 的**整理版基准数据集 + 各 RNA-LLM embedding**：[github.com/sinc-lab/rna-llm-folding](https://github.com/sinc-lab/rna-llm-folding)、[Zenodo record 13821093](https://zenodo.org/records/13821093)
- 各 RNA 基础模型权重：RNA-FM [ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM)、RiNALMo [lbcb-sci/RiNALMo](https://github.com/lbcb-sci/RiNALMo)、RNA-MSM [yikunpku/RNA-MSM](https://github.com/yikunpku/RNA-MSM)、RNABERT [mana438/RNABERT](https://github.com/mana438/RNABERT)、ERNIE-RNA [Bruce-ywj/ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA)、E2Efold [ml4bio/e2efold](https://github.com/ml4bio/e2efold)
- LinearFold / LinearPartition：[github.com/LinearFold/LinearFold](https://github.com/LinearFold/LinearFold)、[github.com/LinearFold/LinearPartition](https://github.com/LinearFold/LinearPartition)
- ViennaRNA：[https://www.tbi.univie.ac.at/RNA/](https://www.tbi.univie.ac.at/RNA/)（论文 [Algorithms Mol Biol 6:26](https://doi.org/10.1186/1748-7188-6-26)）

**需申请 / 许可待确认 / 可能失效**：
- **ArchiveII 官方独立下载页**：**未核实**（建议改用 UFold / RiNALMo benchmark 的打包版本，并注明出处）。
- **RNAStralign 官方原始 URL**：**未核实**（随 UFold / Mathews lab 分发）。
- **PseudoBase++**：**入口与许可未核实**。
- **RNA-Puzzles / CASP15 靶标结构**：结构本身在 [PDB](https://www.rcsb.org/) 公开，但**赛事评估所需的对照文件与官方评估脚本的获取方式未核实**。
- **bpRNA-1m 的 7 个来源中部分原始数据库**：**未核实**是否仍全部在线。

---

## 未能核实的条目

> 以下为本报告**未能从一手来源确认**、需人工核对的字段。**投稿前请勿在这些点上做确定性表述。**

1. **bpRNA-1m 的 7 个来源数据库的具体清单**（原文称 "extracted from seven different sources"，清单未逐一核实）。
2. **bpRNA-1m 条目数的精确值**：论文摘要写 "over 100,000"，正文写 "over 102,318"，SPOT-RNA 引用写 102,348；**三个数字并存，需以原文/版本为准**。
3. **bpRNA-new 的 Rfam 版本**：二手来源分别写 **Rfam 14.1** 与 **14.2**，冲突未解。
4. **bpRNA-new 的精确序列条数**（常见说法约 1,539 条）与**家族数**（约 1,500）——均未从一手来源核实。
5. **ts3 的精确定义与规模**（同一性阈值、分辨率过滤、是否含复合物、条数）。
6. **ArchiveII 的「去冗余版本」**：去冗余阈值、定义者、首个引用论文均未核实。
7. **ArchiveII 原始论文（Sloma & Mathews 2016）的准确 URL/DOI** 及**官方下载页**。
8. **RNAStralign（Tan et al. 2017）的原始论文 DOI 与官方下载 URL**；以及 37,149 条的 train/test 官方划分文件。
9. **PseudoBase++ 的规模与访问入口/许可**。
10. **RNA-Puzzles 各轮（I–V）的靶标数与二级结构层面的官方评测方式**。
11. **CASP16（2024）RNA 靶标清单与评估细节**；以及 CASP15 的**具体靶标列表**。
12. **长链基准（16S/23S rRNA）的公认数据集与长度分桶标准**（未见统一协议）。
13. **跨长度 / 跨 GC 外推的既有专门评测协议**（未发现明确标准，建议自建）。
14. **INF 的逐字公式定义**（尤其 "interactions" 是否包含 base stacking）与其标准出处（Parisien et al. 2009 的原始定义未逐字核实）。
15. **MXfold2、RNAformer、RNADiffFold、CDPfold 的官方代码/权重仓库 URL**（论文可确认，仓库地址未核实）。
16. **CONTRAfold / LinearPartition 输出概率的校准层面评测**：确认「未发现」，但**无法排除存在未被检索到的小众工作**。
17. **各基线在 ArchiveII / TS0 上的具体 F1 数值**：本报告仅核实了 SPOT-RNA 在 TS1 上的 F1≈0.69、ensemble defect（0.19/0.24/0.25）、UFold 的 ~10–30% 提升与 ~160 ms、CDPfold 的 ~30% 声称；**其余方法的逐项 F1 未核实**，请以各原文表格为准。
18. **各数据集/模型权重的具体许可协议**（多数未核实）。
19. **`sparks-lab.org` 上 SPOT-RNA 数据镜像是否仍然在线**。
20. **RNA-Puzzles / CASP 官方评估脚本与对照文件的获取方式**。
