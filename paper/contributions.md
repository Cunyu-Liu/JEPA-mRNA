# 贡献边界（Contributions）

> change-id: `build-rna-ss-decision-model`
> 依据：spec §2.1（真贡献）、§2.2（支撑性工作）、§2.3（实现手段，不主张）、§2.4（六条禁止表述）
> 本文档是**贡献句的唯一许可清单**。任何贡献句若引入本清单之外的项，即为不合格。

---

## 1. 真贡献（2 条，spec §2.1）

| ID | 贡献 | 为什么不是增量 |
|---|---|---|
| **C1** | **免 DP 的校准**（**重音在校准**）：单次前向输出配对概率，训练目标同时匹配**精确配分函数边际**（蒸馏）与**校准**（RLCD 式），使推理期**完全跳过 `O(L³)` 配分函数** | **"免 DP"本身不是空白**——CDPFold 用条件扩散直接出概率矩阵，UFold/SPOT-RNA 的接触图即概率矩阵。**空白在"免 DP 且概率经校准"**：上述方法均无校准目标、无 ECE 报告。**⚠️ 此贡献的成立以 CDPFold 校准对比为前提（阻塞项，见 spec §0.9.4）** |
| **C2** | **层级决策级联**（spec §5.0.2）：决策从 `O(L²)` 个碱基对降到 `O(L)` 个**螺旋级决策** + 局部细化。数学依据：非交叉匹配的弧图是**森林**，Nussinov 递推即其树分解；物理依据：螺旋是进化保守的功能单元 | 带状化**牺牲长程配对**（工程 hack）；扁平 `L×L` 头为 `O(L²)`。层级级联**保留长程配对**且降复杂度。**唯一不依赖他人工作的独立创新** |

**论文骨架 = C2（骨架） + C1（卖点）**。若 CDPFold 校准对比使 C1 崩塌，则骨架退为 **C2 + C4**（spec §0.9.4）。

---

## 2. 支撑性工作（2 条，spec §2.2；不单独主打）

| ID | 贡献 | 定位 |
|---|---|---|
| **C3** | **计算自适应折叠**：置信度门控的按需升级——只在低置信区域调用精确物理 | **C1 的下游应用**；校准不可信则门控无依据 |
| **C4** | **校准优先的评测协议**：把 ECE / 可靠性图 / 边际校准 / 后验对数似然 / **结构层面校准** / "合法化代价"作为一等指标引入本领域 | 领域缺口真实，但"提出指标"在顶刊偏弱；**适合作支撑** |

> **S5（可审计数据清洗管线）** 属工程规范，**不构成科学贡献**（spec §2.2），不得出现在贡献句中。

---

## 3. 实现手段（5 项，**明确不主张为贡献**，spec §2.3）

| # | 项 | 为何不主张 |
|---|---|---|
| 1 | **Gibbs 框架 / 精确似然** | CONTRAfold/CRF 已有；且推理期已不用它（仅作训练教师） |
| 2 | **Turner 残差先验（`MLP_T`）** | 合理工程技巧，非创新；零初始化只等价于 Nussinov + 堆叠能 |
| 3 | **构造性对称配对表示** | 正确做法，但审稿人不会视为贡献 |
| 4 | **隐式微分可微 DP** | E2Efold 等已用类似思路 |
| 5 | **多通道证据（SHAPE/DMS）** | 已有大量条件化工作 |

> **这些是方法章节的实现细节，写进方法即可，不得出现在"贡献列表"或摘要的贡献句中。**
> **允许出现的贡献表述仅限**：C1 免 DP 的校准 · C2 层级决策级联 · C3 计算自适应折叠 · C4 校准优先评测协议。

---

## 4. 六条禁止表述（spec §2.4）——投稿前逐条核查清单

| # | 禁止主张 | 原因 | 核查（稿件中不存在则打勾） |
|---|---|---|---|
| 1 | ❌ **不主张首创 Gibbs / 配分函数框架** | CONTRAfold（Do et al. 2006）与 CRF 谱系已具备：非交叉空间上的对数线性模型、DP 精确配分函数、条件似然训练、`(p̂−y)` 梯度、inside-outside 边际。**这些是教科书内容**（spec §0.7 问题 2） | [ ] |
| 2 | ❌ **不主张共转录决策顺序是创新** | 它只是重新表述 Nussinov 的填表顺序，不改变分布、不带来新能力。**已降级为消融项**（spec §0.7 问题 4） | [ ] |
| 3 | ❌ **不主张"非法结构率 = 0"是贡献** | 任何带约束解码的方法都能做到（spec §0.7 问题 5）；H1 已改为测量**合法化代价** | [ ] |
| 4 | ❌ **不主张 `MLP_T=0` 等价于 ViennaRNA** | Nussinov 只含堆叠项，不含环熵 / coaxial stacking / 终端错配（spec §0.7 问题 3） | [ ] |
| 5 | ❌ **不主张复现了 Jev 的 RLCD** | RLCD 算法未公开，本文为"**RLCD 启发式自设计目标**"（spec §0.3 / §5.8.4） | [ ] |
| 6 | ❌ **不主张在算法复杂度上优于 LinearFold** | LinearFold 为 `O(L)`；本方案 System-1 为 `O(L²d) + O(L·B²)`（spec §5.9） | [ ] |

**投稿前逐条核查**：上述 6 条禁止表述在稿件（含摘要、贡献句、图注）中**均不存在**，方可提交。
自动化核查：`python paper/check_manuscript.py --manuscript <稿件>`（检查项 a/b/c/d）。

---

## 5. 许可的贡献句模板（供写作时直接使用）

> **C1**：We show that a **single-forward-pass decision head can be trained to be as calibrated as the exact partition-function marginals**, so that inference skips the `O(L³)` partition function entirely (contribution C1: *DP-free calibration*).

> **C2**：We introduce a **hierarchical decision cascade** that replaces the `O(L²)` base-pair decision space with `O(L)` helix-level decisions plus local refinement, **without sacrificing long-range pairs** (contribution C2: *hierarchical decision cascade*).

> **C3**：We turn the calibrated confidence into a **compute-allocation controller**, upgrading only low-confidence regions to exact physics (supporting contribution C3: *compute-adaptive folding*).

> **C4**：We introduce a **calibration-first evaluation protocol** for RNA secondary-structure prediction, reporting ECE, reliability diagrams, marginal calibration, posterior log-likelihood and the *legalisation cost* as first-class metrics (supporting contribution C4).

> **谱系句（必写）**：Our work sits in the **CONTRAfold / CRF lineage**; the log-linear CRF framework and the partition function are **not our contribution**.
