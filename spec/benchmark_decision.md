# Benchmark 决策记录（评测集 / 协议 / 横向对比）

> 冻结时间：**2026-09-24**（本文件早于任何正式训练运行）
> 配套调研全文：`spec/benchmark_survey.md`（含逐条来源链接与 20 条「未能核实」条目）
> 本文件是**可执行决策**；`benchmark_survey.md` 是**证据底稿**。两者冲突时以本文件为准，但本文件的每条选择都必须能回溯到证据底稿或实测记录。

---

## 0. 一页结论

| 问题 | 决策 |
|---|---|
| 主集（in-distribution） | **TS0（1,305）** 为主 + **ArchiveII** 为辅；训练集用 **bpRNA TR0（10,814）**，并在 **RNAStrAlign（37,052）** 上做规模扩展实验 |
| 跨家族 OOD（主） | **bpRNA-new（5,401）** |
| 跨家族 OOD（补充） | **Rfam12.3–14.10（10,791）**（家族级）+ **Rfam14.10–15.0**（时间级，BPfold 发布） |
| 实验标签集 | **PDB ts1/ts2/ts3（60/38/18）** + **PDB_669（669）** |
| 速度基准 | **自建协议**（社区无公认协议），长度分桶，同时报 wall-clock 与 FLOPs，并**显式声明是否含后处理 DP** |
| 统计 | **≥5 seed**；主集与 OOD 均报 mean ± std；配对 Wilcoxon + bootstrap CI；Holm-Bonferroni 校正 |
| C1 的头号对比对象 | **不是 CDPFold**。是 **SPOT-RNA / UFold**（同为单次前向直接出 `L×L` 概率矩阵）与 **ViennaRNA / LinearPartition**（同为"概率"路线，但需 DP） |

---

## 1. 勘误：CDPFold 的定位被写错了（**必须修正**）

**原 spec 的说法（§0.8 问题 ⑥ / §0.9.4 / §7.2.1 / Task 10.8）**：CDPFold 是"条件扩散模型，直接出配对概率矩阵，已免 DP"，因此威胁 C1 新颖性，是**全局单点风险**。

**实测核查后的正确事实**：

| 项 | 核查结果 | 来源 |
|---|---|---|
| 方法类别 | **CNN + 动态规划（DP）**，标题即 *...Based on Convolutional Neural Network and Dynamic Programming* | Front Genet 10:467 (2019) |
| 是否条件扩散 | **否** | 同上 |
| 是否免 DP | **否**——它用 CNN 出每个碱基的配对概率，**再对概率施加增强 DP** 求最优结构 | 同上 |
| 是否被评测过校准 | **未发现**任何校准评测（它本身也未把校准当卖点） | 检索未命中 |
| 代码/权重 | **未核实**是否公开 | — |

**结论与影响**：
1. 原"CDPFold 单点风险"是**基于错误前提的风险**。Task 10.8 的原始判据（"若 CDPFold 校准与我们相当则 C1 崩塌"）**作废**。
2. **真正需要盯防的对象**是 **SPOT-RNA / SPOT-RNA2 / UFold**：它们**确实**单次前向直接输出 `L×L` 配对概率矩阵（sigmoid 输出），**不跑配分函数**。也就是说，**"单次前向出概率"本身已有先例，不能作为 C1 的卖点**。
3. **因此 C1 的新颖性必须且只能锚定在"校准"上**：既有方法能出概率，但**从未有人验证过这些概率是否校准**（见 §3）。这是 C1 的立论基础，也是 C1 唯一可辩护的表述。
4. CDPFold 降级为**普通基线**（且因权重公开性未核实，可能连基线都做不了），不再承担"决定论文骨架"的角色。

> **写作红线（新增）**：不得出现"我们首次实现单次前向输出配对概率矩阵"这类表述——SPOT-RNA/UFold 已做过。只能说"首次给出**经校准验证**的免 DP 配对概率"。

---

## 2. 数据现状：实测盘查（**非计划，全部为实测计数**）

### 2.1 已落盘（集群，可直接用于训练/评测）

数据根：`/mnt/cunyuliu/BPfold_data`（BPfold 发布包）+ `/mnt/cunyuliu/rna_ss_data`（本次新增）

| 数据集 | 路径 | 实测文件数 | 用途 | 状态 |
|---|---|---|---|---|
| bpRNA **TR0** | `BPfold_data/bpRNA/TR0` | **10,814** | 训练 | 已落盘 |
| bpRNA **TS0** | `BPfold_data/bpRNA/TS0` | **1,305** | **主集（测试）** | 已落盘 |
| bpRNA **VL0** | `BPfold_data/bpRNA/VL0` | **198** | 验证 | 存疑，见 §2.4 |
| **RNAStrAlign** | `BPfold_data/RNAStrAlign_bpseq` | **37,052** | 训练（规模扩展） | 已落盘 |
| **PDB_669** | `BPfold_data/PDB_669/PDB` | **669** | 实验标签训练 | 已落盘 |
| **Rfam12.3–14.10** | `BPfold_data/Rfam12.3-14.10` | **10,791** | **家族级 OOD** | 已落盘 |
| **ArchiveII** | `BPfold_data/archiveII/archiveII` | **0** | 主集（测试） | **空目录** |
| **bpRNA-new** | `rna_ss_data/bpfold/extracted/.../bpRNAnew/bpRNAnew.nr500.canonicals` | **5,401** | **跨家族 OOD 主集** | 本次下载 |
| **PDB ts1 / ts2 / ts3** | `rna_ss_data/bpfold/extracted/.../PDB_test/{TS1,TS2,TS3}` | **60 / 38 / 18** | 实验标签测试 | 本次下载 |
| **Rfam14.10–15.0** | `rna_ss_data/bpfold/BPfold_test_results/`（需解包） | 待计数 | **时间级 OOD** | 本次下载 |
| **ArchiveII（CSV 版）** | `rna_ss_data/rinalmo/ArchiveII.csv` | **3,864** 条记录 | 主集替代 | 本次下载 |
| bpRNA（CSV 版） | `rna_ss_data/rinalmo/bpRNA.csv` | 18,820 条记录 | 交叉核对 | 本次下载 |
| PDB-RNA（CSV 版） | `rna_ss_data/rinalmo/PDB-RNA.csv` | 203 条记录 | 交叉核对 | 本次下载 |

**ArchiveII 的处理决定**：
- 集群上原有的 `archiveII/archiveII/` 是**空目录**（同批 0 字节的 `archiveII.lst`、`data_index_archive.yaml` 表明那次解包被截断）。
- BPfold 的 release 包 `BPfold_data.tar.gz`（v0.2，159 MB）**不含 archiveII**——已实测解包确认，只有 `PDB_669 / PDB_test / bpRNAnew / test_data`。
- 因此 **ArchiveII 采用 RiNALMo benchmark 的整理版 `ArchiveII.csv`（3,864 条，含 sequence / structure / base_pairs / len）**，并**同时提供 family-fold 与 k-fold 两套划分**（`ArchiveII_famfold_splits.csv`、`ArchiveII_kfold_splits.csv`）。
- **必须在论文中写明 ArchiveII 版本与条数**（社区 3,975 / 3,966 / 3,864 三个数字并存），并说明我们用的是 RiNALMo 整理版；**不得笼统称"ArchiveII 3975"**。

### 2.2 本次新增下载（含来源与大小）

| 资产 | 来源 | 大小 | 校验 |
|---|---|---|---|
| `BPfold_data.tar.gz` | `https://github.com/heqin-zhu/BPfold/releases/download/v0.2/` | 166,902,534 B | 下载字节数 = release API 报告 size |
| `BPfold_test_results.tar.gz` | 同上 | 8,279,584 B | 同上 |
| `model_predict.tar.gz` | 同上 | 178,585,408 B | 同上 |
| `model_reproduce.tar.gz` | 同上 | 148,327,411 B | 同上 |
| RiNALMo benchmark 7 个 CSV | `raw.githubusercontent.com/sinc-lab/rna-llm-folding/main/data/` | 共约 13 MB | HTTP 200 + 行数已核 |

**BPfold 权重（`model_predict` / `model_reproduce`）的用途**：BPfold 是 2025 年 Nat Commun 的 SOTA 工作，且有**可运行的官方权重**。它因此是**我们能实际复现的最强基线**（相比 SPOT-RNA/UFold 的 Google Drive / Dropbox 权重在中国网络下不可达）。

### 2.3 网络实测（集群侧，2026-09-24）

| 主机 | 结果 |
|---|---|
| `github.com` | 间歇可达（tarball 下载成功；`git clone` 曾超时 129 s） |
| `api.github.com` / `raw.githubusercontent.com` / `codeload.github.com` | **稳定可达** |
| `bprna.cgrb.oregonstate.edu` | 可达 |
| `zenodo.org` | **不可达**（DNS 可解析，连接返回 000） |
| `www.dropbox.com` / `app.nihaocloud.com` / `drive.google.com` | **不可达** |

**推论**：SPOT-RNA / UFold 的官方权重与数据集走 Dropbox / Google Drive / NihaoCloud，**在集群上拿不到**；走 **GitHub Releases / raw.githubusercontent** 的资产（BPfold、RiNALMo benchmark）**能拿到**。基线选择必须服从这一约束。

### 2.4 存疑项（**不得当作事实引用，须在运行前核实**）

1. **bpRNA VL0 条数**：集群实测 **198**，而 SPOT-RNA 原文记为 **1,300**。差异未解释。→ 训练/验证划分若使用 VL0，必须先核实，否则**改用 TS0 内部切分或改用 RiNALMo `bpRNA_splits.csv`**。
2. **archiveII 原始 bpseq 版**：未取得。若后续需要与 UFold/SPOT-RNA 论文数字严格对齐，需另寻镜像。
3. **bpRNA-new 的 Rfam 版本**（14.1 vs 14.2）与**精确条数**：二手来源冲突，未从一手核实。
4. **ts3 的精确定义**（同一性阈值 / 分辨率过滤）：未核实。
5. **bpRNA-1m 标签来源**：**不是 RNAfold 机器标注**，而是 7 个来源的"已知结构"汇编（多为比较分析/协方差衍生）。SPOT-RNA 原文指出其噪声使 **precision 上界约 96%**。→ **必须在论文中写明**，且这正是**必须补实验标签集（PDB ts1/ts2/ts3）的理由**。

---

## 3. 校准：C1 的立论基础（本节是 C1 的命门）

### 3.1 先行工作检索结论

**未发现**任何对 RNA 碱基配对概率做**系统性校准评测（ECE / 可靠性图 / Brier / NLL）**的工作。

最接近的两条，**必须引用并显式区分**：

| 先行工作 | 它做了什么 | **与"校准"的区别** |
|---|---|---|
| **Ensemble defect**（SPOT-RNA 用其对比，TS1：SPOT-RNA 0.19 / CONTRAfold 0.24 / CentroidFold 0.25） | 衡量**概率系综**与真实结构的期望偏差 | 系综质量 ≠ 概率校准。一个模型可以 ensemble defect 低、但概率系统性偏乐观 |
| **BPfold confidence index**（2025，与 F1 的 Pearson 0.728） | 用网络 contact map 与精修 contact map 的余弦相似度构造**置信度**，与 F1 相关 | 置信度估计 ≠ 概率校准（无可靠性图、无 ECE/Brier 意义下的校准目标） |

### 3.2 因此 C1 的口径（**写作必须严格遵守**）

> **C1 = 首次给出「免 DP 且经 ECE / 可靠性图 / Brier / NLL 验证的校准配对概率」。**
> 不主张"首次单次前向出概率"（SPOT-RNA/UFold 已有）。

**C1 的判据（重定义，替代原 Task 10.8）**：
- **C1-a（存在性）**：我们的 System-1 头在 TS0 / ArchiveII / PDB ts1 上，ECE 与 Brier **优于或持平** SPOT-RNA / UFold 的 sigmoid 概率（若能取到其权重/预测）。
- **C1-b（对照）**：与 **ViennaRNA 精确配分函数概率**、**LinearPartition 近似 BPP** 做**同口径** ECE / Brier 对比。
- **C1-c（免 DP 代价）**：System-1 头的 ECE 与精确边际 `p̂^exact` 的 ECE 之差 **≤ 0.02**（原 S7 门限保留）。

### 3.3 指标口径必须先定义（领域空白，无现成约定）

- **ground truth**：`(i,j)` 是否为真实碱基对 → 二值标签（`L(L-1)/2` 个样本，按序列分组）。
- **ECE**：等宽/等频分箱 + 软分箱两版都报（软分箱可微，用于训练；硬分箱用于报告）。
- **Brier / NLL**：按序列聚合后取平均，并报 bootstrap CI。
- **必须报告**：配对间相关性对 ECE 的影响（同一序列内配对不独立），并**补充结构层面校准**（整条结构在模型下的后验对数似然）——这是应对审稿质疑 Q3 的必要动作。

---

## 4. 评测协议（可直接照做）

### 4.1 指标（七类）

1. **配对层面**：Precision(=PPV) / Recall(=SEN) / **F1** / MCC
   - **micro 与 macro 两种聚合口径都必须报**（先把所有序列的 TP/FP/FN 汇总再算 = micro；每条序列算 F1 再平均 = macro）。**论文必须写明用的是哪种**。
2. **结构层面**：INF
3. **校准**：ECE / 可靠性图 / Brier / NLL / 边际校准 / 结构层面校准（§3.3）
4. **合法性**：非法结构率（目标 0）、最小发夹环违规率（目标 0）
5. **速度**：分长度桶延迟（ms）、吞吐（seq/s）、峰值显存、GPU 小时/千条、**相对 McCaskill 配分函数加速比**
6. **跨家族泛化**：bpRNA-new F1 + Rfam12.3–14.10 F1（**与同源评测分列**）
7. **回退行为**：低置信回退比例、回退对 F1 的边际贡献

### 4.2 长度分桶（自建，须写全协议）

`<100 / 100–200 / 200–400 / 400–600 / 600–1000 / >1000` nt。每桶报条数与均值。
**速度测量必须声明**：硬件型号、batch size（1 与 32 分列）、warmup 步数、CUDA events 计时、**是否含后处理 DP**（我们的卖点是免 DP，所以必须把"仅前向"与"前向+后处理"分开报）。

### 4.3 统计严谨性（硬要求）

- **≥5 seed**（主集与 OOD 都要），报 mean ± std
- 配对显著性检验：**Wilcoxon signed-rank**（逐序列配对）+ **bootstrap CI**（1000 次重采样）
- 多重比较校正：**Holm-Bonferroni**
- 测试集**不参与任何超参选择**（早停与选模型只用 valid）

### 4.4 已知缺陷的规避清单（写进论文的 Limitations）

| 缺陷 | 规避做法 |
|---|---|
| `dev == test` | 严格三分，附录给文件哈希 |
| 测试集与训练集同源 | CD-HIT-EST 80% + BLAST-N 过滤；优先 family-wise（bpRNA-new / Rfam12.3–14.10） |
| 只报单一 seed | ≥5 seed + 显著性检验 |
| 标签来自计算注释而非实验 | **必须**补 PDB ts1/ts2/ts3（3D 高分辨派生）+ 明示 bpRNA 的 96% precision 上界 |
| 指标聚合口径不一 | 统一报 micro + macro，只与同口径结果比 |
| 忽略假结/非规范配对 | 单独报子集 F1 |
| 长序列外推 | 按长度分桶报，长链单独给失败分析 |

---

## 5. 横向对比模型（含可比性警告）

> **不同论文报告的 F1 不可跨行直接比较**（测试集、去冗余阈值、是否含假结/非规范配对、micro/macro 口径均不同）。只有"同一测试集文件 + 同一去冗余 + 同一指标定义"下的数字才可比。**论文中引用他人数字时必须标注其原始测试集与口径。**

### 5.1 分层对比策略（按"能否在集群上真实复现"分层）

| 层级 | 模型 | 能否真实复现 | 处置 |
|---|---|---|---|
| **A. 必须自跑** | **ViennaRNA（RNAfold + 配分函数）**、**LinearPartition**、**Nussinov + Turner 堆叠能（`MLP_T` 置零对照物）** | 可（pip/源码编译） | 自己跑，同硬件同协议 |
| **A. 必须自跑** | **BPfold**（2025 Nat Commun） | **可**（权重已下载到集群） | 自己跑 —— **这是我们能拿到的最强 SOTA 基线** |
| **B. 尽量自跑** | **UFold** | 权重在 Google Drive（不可达）；`~/rna_baselines_src/UFold-main` 有源码 | 尝试；失败则**引用其论文数字并标注测试集** |
| **C. 只能引用** | SPOT-RNA / SPOT-RNA2 / MXfold2 / E2Efold / RNA-FM / RiNALMo / RNA-MSM / mRNABERT / RNAformer / RNADiffFold | 权重源不可达 | 引用原文数字 + 显式标注测试集与口径 |
| **C. 只能引用** | **CDPFold** | 权重公开性未核实 | 引用原文数字；**且必须写清它是 CNN+DP，不是扩散、不免 DP** |
| **D. 统一基准（强）** | **RiNALMo benchmark 的既有结果** | 已有 `results/`（各模型在 ArchiveII 同划分上的 metrics） | **可直接引用**——这是少见的"同测试集同划分"可比数字 |

### 5.2 可比数字（可谨慎引用，均已核到来源）

| 测试集 | 数字 | 来源 |
|---|---|---|
| **TS1** | ensemble defect：SPOT-RNA **0.19** / CONTRAfold **0.24** / CentroidFold **0.25** | SPOT-RNA, Nat Commun 10:5407 (2019) |
| **TS1** | SPOT-RNA 碱基对 F1 ≈ **0.69** | 同上 |
| **ArchiveII / Rfam12.3–14.10** | BPfold confidence index 与 F1 的 Pearson **0.728 / 0.692** | BPfold, Nat Commun 16:5856 (2025) |
| 通用 | UFold 推理 **~160 ms/序列（≤1600 bp）**，相对热力学提升 ~10–30% | UFold, NAR 50(2):e14 (2022) |
| **ArchiveII（同划分）** | 6 个 RNA-LLM 的逐家族 metrics（RNABERT/RNA-FM/RNA-MSM/ERNIE-RNA/RNAErnie/RiNALMo） | RiNALMo benchmark, Brief Bioinform 26(2):bbaf137 (2025) |

**注意**：SPOT-RNA 在 **TS1** 上的 F1 不能与 UFold 在 **ArchiveII** 上的 F1 相比。

---

## 6. 缺口与待补（按优先级）

| 优先级 | 缺口 | 处置 |
|---|---|---|
| **P0** | ViennaRNA 未安装（全集群 5 个候选环境均无 `RNA` 模块） | pip 安装 + **版本锁定**，记录 Turner 参数版本 |
| **P0** | archiveII 原始 bpseq 版 | 先用 RiNALMo CSV 版；论文写明版本 |
| **P1** | bpRNA VL0 条数存疑（198 vs 1,300） | 训练/验证划分前核实；否则改用 RiNALMo `bpRNA_splits.csv` |
| **P1** | LinearPartition 未安装 | 源码编译 |
| **P2** | PDB ts1/ts2/ts3 的标签由 DSSR 从 3D 导出——需确认我们拿到的 bpseq 是否同源 | 抽样核对 |
| **P2** | RNA-Puzzles / CASP15-16 | 加分项；PDB 结构公开可取 |
| **P3** | PseudoBase++（假结） | 规模小；仅 T2 用 |
| **P3** | SHAPE/DMS 探测数据 | 仅 T3 用 |

---

## 7. 与 spec 其它章节的关系

- 本文件**替代** spec §7.3（基准集）的旧版本，并**修正** §0.8 问题 ⑥ / §0.9.4 / §7.2.1 中关于 CDPFold 的错误描述。
- Task **10.8 的原始判据作废**（见 §1），替换为 §3.2 的 C1-a/b/c 三条判据。
- Task **2 / 3 的状态从「受阻塞」改为「已完成（集群侧实测）」**，依据 §2。
- `spec/analysis_plan.md` 的 H2 / S7 门限**保留**（免 DP 的校准代价 ≤ 0.02），但**其对照物从"CDPFold"改为"精确边际 + ViennaRNA/LinearPartition + SPOT-RNA/UFold"**。
