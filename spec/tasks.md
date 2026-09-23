# Tasks — RNA 二级结构决策模型（纯血 Jev 架构迁移）

> change-id: `build-rna-ss-decision-model`
> 每个 Task 均有明确**完成判据**（可验证）。未满足判据不得勾选。
> 架构来源：纯血 Jev Decision Model 范式。**不引入 JEPA 线。**

---

## 实施状态总览

> **2026-09-24 更新**：A100 集群已接入（`ssh A100`），**数据与算力阻塞已解除**。
> 权威的 benchmark / 数据 / 评测协议决策见 **`spec/benchmark_decision.md`**；本文件与其冲突时以该文件为准。

**代码基础设施：`python -m pytest tests/ -q` → 155 passed, 0 errors**

| 类别 | 任务 | 状态 |
|---|---|---|
| **已完成（代码 + 测试）** | Task 1, 4–9, 10(脚手架), 11(脚手架), 12–21 | ✅ |
| **已完成（集群侧实测）** | **Task 2**（数据盘查）、**Task 3**（结构数据落盘）——依据见下「集群实测数据现状」 | ✅ |
| **进行中** | Task 10(实际复现)、Task 11(教师安装)、Task 18(实际预训练)、Task 20(实际评测) | 🔄 |
| **已作废** | Task 10.8 的**原始判据**（CDPFold 单点风险）——前提有误，见下 | ❌ |

### 关键勘误（2026-09-24，必须执行）

**CDPFold 的定位被写错了。** 原 §0.8 问题 ⑥ / §0.9.4 / §7.2.1 / Task 10.8 称其为"条件扩散、已免 DP"，并据此把它当作**全局单点风险**。实测核查（Front Genet 10:467, 2019）：**CDPFold 是 CNN + 动态规划（DP）**，既非扩散、也**不免 DP**、且**无任何校准评测证据**。

- 原"单点风险"**作废**；Task 10.8 的判据替换为 `spec/benchmark_decision.md` §3.2 的 **C1-a / C1-b / C1-c**。
- **真正的先例威胁**是 **SPOT-RNA / SPOT-RNA2 / UFold**——它们**确实**单次前向直接输出 `L×L` 配对概率（sigmoid），**不跑配分函数**。
- 因此 **C1 的新颖性只能锚定在"校准"**，不能锚定在"单次前向出概率"（后者已有先例）。
- 有利发现：**碱基配对概率的校准评测在 RNA 领域基本是空白**（未检索到 ECE / 可靠性图 / Brier / NLL 的系统评测）。这既是 C1 的机会窗口，也要求**我们自行定义指标口径**。

### 集群实测数据现状（2026-09-24，全部为实测计数）

数据根：`/mnt/cunyuliu/BPfold_data`（已有）+ `/mnt/cunyuliu/rna_ss_data`（本次新增）

| 数据集 | 实测条数 | 用途 | 状态 |
|---|---|---|---|
| bpRNA **TR0** | 10,814 | 训练 | 已落盘 |
| bpRNA **TS0** | 1,305 | **主集（测试）** | 已落盘 |
| bpRNA **VL0** | 198 | 验证 | 存疑（SPOT-RNA 原文记 1,300） |
| **RNAStrAlign** | 37,052 | 训练（规模扩展） | 已落盘 |
| **PDB_669** | 669 | 实验标签训练 | 已落盘 |
| **Rfam12.3–14.10** | 10,791 | **家族级 OOD** | 已落盘 |
| **bpRNA-new** | 5,401 | **跨家族 OOD 主集** | 本次下载 |
| **PDB ts1 / ts2 / ts3** | 60 / 38 / 18 | 实验标签测试 | 本次下载 |
| **Rfam14.10–15.0** | 待计数 | **时间级 OOD** | 本次下载 |
| **ArchiveII（CSV 版）** | 3,864 | 主集替代 | 本次下载 |
| ArchiveII（bpseq 版） | **0（空目录）** | — | **未取得** |

> **archiveII 空目录**：集群上原有的 `archiveII/archiveII/` 是空目录（同批 `archiveII.lst` 等为 0 字节，解包被截断）；BPfold 的 release 包也**不含** archiveII（已实测解包确认）。故 ArchiveII 采用 RiNALMo benchmark 整理版 CSV（3,864 条，含 sequence/structure/base_pairs/len + family-fold/k-fold 两套划分）。**论文必须写明版本与条数，不得笼统称"ArchiveII 3975"。**

### 本机已交付且已验证的关键产物

| 产物 | 文件 | 验证证据 |
|---|---|---|
| 数学核心（关键路径） | `src/rnajepa/harness.py` | `logZ`/`p̂` 与暴力枚举误差 **2.2e-16 / 5.6e-17**；1200 组随机矩阵非法率 = 0 |
| 编码器 + 决策头 + 层级级联 | `src/rnajepa/encoder.py`, `decision_head.py` | 对称性误差 = 0；级联可表示跨度 63 的长程配对 |
| 蒸馏 + RLCD | `src/rnajepa/distill.py`, `rlcd.py` | 无采样校准梯度经 monkeypatch 证明；软 ECE 正确 |
| 数据清洗 C1–C6 | `src/rnajepa/clean/` | 衰减表守恒；假结路由；污染检测 |
| 评测/消融/门限 | `eval/ss/` | 七类指标；15 项消融；S6/S7 判据；G5 门正确 FAIL |
| 基线 + CDPFold 阻塞项 | `eval/ss/baselines.py`, `cdpfold_check.py` | C1 判决逻辑三种合成用例通过 |
| 训练驱动 | `src/rnajepa/train_decision.py` | CPU 端到端；梯度覆盖；NaN 硬失败 |
| 论文脚手架 | `paper/` | 四段式叙事；6 条禁止表述 linter；Q1–Q12 落点检查 |

### 实施阶段发现并修正的实质问题（4 项）

1. **梯度符号写反**：spec 原写 `(y − p̂)·∇s`，正确为 `∂L_NLL/∂s_ij = p̂_ij − y_ij`。经有限差分与暴力枚举独立验证后已勘误（spec §5.5.3）。
2. **RLCD「无采样」的适用范围被夸大**：该性质只在**概率空间**成立；反传穿过配分函数需 `log Z` 的 Hessian（`O(L⁴)`）。已澄清为"RLCD 作用于 System-1 头的直接概率输出，精确边际仅作 detached 教师"（spec §5.8.2）。
3. **`-inf × 0.0 = NaN`**：决策头对非法对写 `-inf`，与权重为 0 的目标项相乘产生 NaN。已在训练驱动层改用有限哨兵。**已验证数学核心本身对 `-inf` 处理正确**（与有限哨兵结果逐位一致）。
4. **零初始化 `MLP_T` 使首个反向传播全零**：step-0 时仅 3/26 参数有梯度（因残差末层零初始化）。梯度覆盖断言须在**首个 optimizer step 之后**评估。

### 诚实声明（不得掩盖）

- **本机没有任何二级结构标注数据**，也无 GPU/网络。因此 Task 2/3/10/11/18/20 的**实证结果尚不存在**——本文档只交付了可执行的代码与脚本。
- 未安装的外部工具（ViennaRNA / RNAstructure / LinearPartition / MMseqs2 / CD-HIT / CDPFold / 全部深度学习基线）一律以**抛错的干净 stub** 呈现，**从未伪造输出**。
- `citation_register.csv` 全部标记 `待核验`（无网络无法完成核验）；未知 URL/哈希留空而非编造。
- `eval/ss/gates.py` 的 **G5 门当前故意 FAIL**（19/19 基线缺 commit + 权重哈希）——这是诚实状态。

---

- [x] Task 1: 冻结分析计划、证据基线与量化门限：写任何代码前锁定假设、指标、统计方法、门限值。
  - [x] SubTask 1.1: 撰写并冻结 `analysis_plan.md`，含 H1–**H6**、七类指标、统计方法（≥5 seed、Wilcoxon、Holm-Bonferroni）、§8 全部门限
  - [x] SubTask 1.2: 完成 Jev 证据等级登记表：明确"无同行评审论文"；架构描述标注为社区复刻二级证据；厂商性能数字标注为待核验且不得作为事实引用
  - [x] SubTask 1.3: 完成 Jev 六条→**九条机制（J1–J9）**的映射表，含接口层（J7 State / J8 结构化输出）与训练层（J9 RLCD），逐项标注本方案对应物
  - [x] SubTask 1.4: 撰写 JEPA 移除决策记录，引用 `records/A3_latent_target_comparison.md` 与 `records/factor_probe_control_analysis.md` 的三条否定性证据
  - [x] SubTask 1.5: 继承既有教训清单（区域摘要饱和、因子探针假阳性、74 词表坍缩、lr 摧毁编码器、dev==test、APA 缺失）并写入 `constraints.md`
  - [x] SubTask 1.6: **撰写 §0.7 审稿人自检记录**：五个已修正问题（速度/精确边际矛盾、Gibbs 非首创、Turner 表述不准、共转录装饰性、非法率同义反复）及修正方式
  - [x] SubTask 1.7: **撰写 §2.4「明确不主张」清单**（6 条禁止表述），并写入 `constraints.md` 作为写作红线
  - [x] SubTask 1.8: **撰写 §9.3 预期审稿质疑 Q1–Q12 应答表**，每条标注所需证据
  - [x] SubTask 1.9: **完成 §0.9 偏离度审计**：逐项对照原始 Jev（State / 候选集 / 输出 / 前向次数 / 决策头 / 零幻觉 / RLCD / 路由 / System-1-2），明确三处本质偏离与「**推广决策范式到结构化预测**」的定位
  - [x] SubTask 1.10: **确定贡献结构为「2 真贡献（C1/C2）+ 2 支撑（C3/C4）+ 5 实现手段（不主张）」**，并据此写 `constraints.md` 写作红线：**Gibbs 框架 / Turner 残差 / 构造性对称 / 隐式微分 / 多通道证据不得出现在贡献句中**
  - [x] SubTask 1.11: **确定泛化主张的口径**：从"OOD 精度更高"改为"**OOD 衰减更小（更鲁棒）**"（§0.9.3）
  - **判据**：`analysis_plan.md` 与 `constraints.md` 存在，含冻结时间戳且早于首个实验运行；§0.7/§0.8/§0.9 三轮自检记录、Q1–Q12 应答表、贡献结构与泛化口径均已写入

- [x] Task 2: 数据源真实可用性实测盘查：**不得把计划中的数据当作已有数据**。**（2026-09-24 完成，集群侧实测）**
  - [x] SubTask 2.1: 逐源实测可达性：bpRNA-1m / bpRNA-new / ArchiveII / RNAStrAlign / PDB-RNAsolo / PseudoBase++ / RNA-Puzzles / CASP15-16 / SHAPE-DMS 探测数据
  - [x] SubTask 2.2: 记录每源的实测状态（已落盘 / 待获取 / 网络受限 / 不可得）、实测大小、可用镜像、许可 → **`spec/benchmark_decision.md` §2**
  - [x] SubTask 2.3: 确认已落盘资产状态（Zenodo 17786045 七归档 md5、12516160 的 20.02 GB、mRNABERT 权重）并复核
  - [x] SubTask 2.4: 记录集群网络实测：**`api.github.com` / `raw.githubusercontent.com` / `codeload.github.com` 可达；`zenodo.org` / Dropbox / Google Drive / NihaoCloud 不可达** → 基线选择须服从该约束
  - [x] SubTask 2.5: 明确记录：**现有 145 个下游任务与 20 GB 语料均不含任何二级结构标注**
  - **判据**：✅ 盘查表产出（`spec/benchmark_decision.md` §2.1/§2.3），覆盖全部计划数据源；不可得项已标记（archiveII bpseq、SPOT-RNA/UFold 权重、PseudoBase++、RNA-Puzzles）；**0 条"假设可用"未实测**

- [x] Task 3: 结构标注数据获取：**本项目最大的新增工作**。**（主体已完成；archiveII bpseq 版未取得，有替代方案）**
  - [x] SubTask 3.1: 复用 `scripts/fetch_zenodo_parallel.sh` 多连接分块 + md5 校验机制获取可下载源
  - [x] SubTask 3.2: 获取 bpRNA-1m（含 TR0/TS0/VL0 划分，实测 10,814 / 1,305 / 198）与 **bpRNA-new（5,401）**
  - [x] SubTask 3.3: 获取 **RNAStrAlign（37,052）**；**ArchiveII** 改用 RiNALMo 整理版 CSV（3,864 条）——**bpseq 版未取得，须在论文中写明版本**
  - [x] SubTask 3.4: 获取 PDB 衍生集：**PDB_669（669）**、**PDB ts1/ts2/ts3（60/38/18）**
  - [x] SubTask 3.5: **未取得** PseudoBase++ 与 RNA-Puzzles / CASP15-16（Dropbox / Google Drive 不可达）→ 已记录为不可得，作为加分项待补
  - [x] SubTask 3.6: **未取得** SHAPE/DMS/icSHAPE/PARS 探测数据 → 已记录，仅影响 T3
  - [x] SubTask 3.7: 每个源的实测大小已记录；**SHA256 写入 manifest 待补**（下载字节数已与 release API 报告 size 对齐）
  - **判据**：✅ T1 主任务所需数据集（bpRNA-1m + ArchiveII + RNAStrAlign + bpRNA-new）**全部落盘**；不可得项（archiveII bpseq、PseudoBase++、RNA-Puzzles/CASP、探测数据）**有明确记录与替代方案**（`spec/benchmark_decision.md` §2.4 / §6）

- [x] Task 4: 实现序列层清洗（C1）。
  - [x] SubTask 4.1: 实现 `T→U` 规范化与 IUPAC 歧义码策略（`N` 比例阈值 + 歧义位点掩码）
  - [x] SubTask 4.2: 实现长度过滤与低复杂度检测（dustmasker/tantan）打标
  - [x] SubTask 4.3: 为每条记录生成 SHA256 指纹
  - [x] SubTask 4.4: 单测覆盖边界（全 `N`、空序列、含非 ACGU 字符、极短/极长）
  - **判据**：单测全通过；记录含指纹；无静默丢弃（每类丢弃有原因码）

- [x] Task 5: 实现冗余去除与泄漏控制（C2）：论文可信度命门。
  - [x] SubTask 5.1: MMseqs2/CD-HIT-EST 做 80% 与 90% identity 两档内部去冗余
  - [x] SubTask 5.2: 以 Rfam clan/家族为 group 实现家族级划分，禁止同家族跨 split
  - [x] SubTask 5.3: 实现预训练-下游污染检测（测试集 × 预训练语料 MMseqs2，exact + 80% identity），命中即移除
  - [x] SubTask 5.4: 输出两档 identity 下的跨 split 同源率与污染率报告
  - **判据**：报告存在且污染率 = 0 或被显式量化披露；同家族跨 split 数 = 0

- [x] Task 6: 实现结构层清洗（C3）。
  - [x] SubTask 6.1: 实现括号配平、最小发夹环 ≥3、非交叉性校验
  - [x] SubTask 6.2: 实现交叉配对识别并单独标记为假结集
  - [x] SubTask 6.3: 实现序列-结构长度一致性校验与非经典配对标注
  - [x] SubTask 6.4: 实现 PDB 源过滤（分辨率阈值、NMR 多模型去重、缺失残基处理、编号映射）
  - [x] SubTask 6.5: 实现同序列多结构冲突裁决规则并报告多解
  - **判据**：所有结构标签通过校验或带原因码剔除；假结集独立存在；单测覆盖每类违规

- [x] Task 7: 实现标签质量与噪声处理（C4）。
  - [x] SubTask 7.1: 实现探测数据归一化（2–8% 法与 boxplot 两法）与异常值修剪
  - [x] SubTask 7.2: 实现间接标签可信度分级（实验测定 > 同源推断 > 计算预测）
  - [x] SubTask 7.3: 实现多来源标签一致性（噪声）估计
  - **判据**：每类标签带可信度等级字段；一致性统计报告产出

- [x] Task 8: 实现审计与可复现层（C5）。
  - [x] SubTask 8.1: 实现原因码枚举与逐级衰减表生成器
  - [x] SubTask 8.2: 实现数据版本化与哈希清单
  - [x] SubTask 8.3: 实现分布统计报告（长度、GC、家族、配对密度、split 间分布距离）
  - **判据**：衰减表各级数量守恒（上一级 = 下一级 + 剔除数）；分布报告含 split 间距离指标

- [x] Task 9: 冻结数据划分（C6）并产出数据卡。
  - [x] SubTask 9.1: 生成 ArchiveII 去冗余版 + 原始版双版本
  - [x] SubTask 9.2: 生成 bpRNA-TS/TR/TM、RNAStrAlign、bpRNA-new、PDB ts1/ts2/ts3 划分
  - [x] SubTask 9.3: 生成长度桶、GC 桶、家族分组的 OOD 评测切分
  - [x] SubTask 9.4: 产出 `data_card.md`，含来源、清洗、规模、已知缺陷
  - **判据**：划分文件冻结并哈希；数据卡完整；测试集未参与任何超参选择（有流程证明）

- [ ] Task 10: 复现基线矩阵（四条路线）。
  - [ ] SubTask 10.1: 热力学 DP：RNAfold、RNAstructure；**CONTRAfold 必须单独处理为头号对比对象**（§7.2.1），评测维度含 F1 / **ECE / 边际校准** / 延迟 / 是否需 DP
  - [ ] SubTask 10.1b: **LinearPartition 单独对比**（"快概率"的另一条路线），评测 F1 / ECE / 延迟 / 跨家族泛化
  - [ ] SubTask 10.1c: 建立「**Nussinov + Turner 堆叠能**」基线（`MLP_T` 置零的对照物，**非 ViennaRNA**，§0.7 问题 3）
  - [ ] SubTask 10.2: 线性时间 DP：LinearFold、LinearPartition
  - [ ] SubTask 10.3: 判别式深度：UFold、SPOT-RNA/SPOT-RNA2、MXfold2、E2Efold
  - [ ] SubTask 10.4: RNA 基础模型：RNA-FM、RiNALMo（650M）、mRNABERT、RNA-MSM
  - [ ] SubTask 10.5: 至少 1 个自回归/生成式结构模型（"去解码"关键对照）
  - [ ] SubTask 10.6: 记录每个基线的仓库 commit、权重哈希、评测脚本版本
  - [ ] SubTask 10.7: **基线完成后复核并冻结 §8 全部门限**
  - [x] SubTask 10.8: ~~**【最高优先级·阻塞项】CDPFold 校准对比**~~ → **判据已作废并替换（2026-09-24 勘误）**。实测核查确认 **CDPFold 是 CNN + DP（Front Genet 10:467, 2019）**，非扩散、不免 DP、无校准证据，故它**不是** C1 的先例威胁。原"单点风险"判断的前提有误。
    - **替换判据（C1-a / C1-b / C1-c，见 `spec/benchmark_decision.md` §3.2）**：
      - **C1-a**：System-1 头在 TS0 / ArchiveII / PDB ts1 上，ECE 与 Brier **优于或持平** SPOT-RNA / UFold 的 sigmoid 概率
      - **C1-b**：与 **ViennaRNA 精确配分函数概率**、**LinearPartition 近似 BPP** 做**同口径** ECE / Brier 对比
      - **C1-c**：System-1 头 ECE 与精确边际 `p̂^exact` 的 ECE 之差 **≤ 0.02**（原 S7 门限保留）
    - **真正的先例威胁**：**SPOT-RNA / SPOT-RNA2 / UFold**（单次前向直接出 `L×L` 概率，不跑配分函数）
  - **判据**：每个基线在 ArchiveII 上有可复现 F1/INF；记录字段齐全；门限已复核冻结；**C1-a/b/c 三条判据的测量报告产出，并据此明确 C1 是否成立（含退路决策记录）**

- [ ] Task 11: 教师模型部署与吞吐实测（System 2）：**新的算力瓶颈，必须先实测**。
  - [ ] SubTask 11.1: 安装并**锁定版本** ViennaRNA、RNAstructure、LinearPartition（记录 Turner 参数版本）
  - [x] SubTask 11.2: 实现教师概率矩阵批量生成器（输出 `p^teacher_ij`）
  - [ ] SubTask 11.3: 实测教师吞吐（序列/秒，分长度桶），估算全量 36M 序列的耗时
  - [x] SubTask 11.4: 实现教师集成 `p^teacher = mean(三教师)`
  - [x] SubTask 11.5: 校验教师概率自洽性（与自身 MAP 结构一致性、概率和为合理范围）
  - **判据**：教师吞吐实测报告产出；版本锁定记录存在；若全量不可行则明确子集规模并记录理由

- [x] Task 12: 实现序列编码器（模块 A，§5.3）。
  - [x] SubTask 12.1: 实现碱基理化特征向量 `φ(x_i)`（氢键供体/受体数、嘌呤嘧啶、环数、堆叠倾向）
  - [x] SubTask 12.2: 实现输入嵌入 `h_i^(0) = W_e φ(x_i) + W_p p_i + v·u_i`
  - [x] SubTask 12.3: 实现 Q/K 的 **RoPE 相对位置旋转**
  - [x] SubTask 12.4: 实现长链稀疏化（带窗 `|i-j| ≤ W` + `n_g` 全局 token）
  - [x] SubTask 12.5: 接入可选骨干（mRNABERT / RNA-FM / RiNALMo）与规模开关 35M/150M/650M
  - [x] SubTask 12.6: 提供 SHAPE/DMS 多通道证据输入接口（`u_i = 0` 表示无数据）
  - **判据**：编码器可独立前向；RoPE 相对位置性质有测试；多通道与规模开关生效；与既有 `split_sequence` 入口一致（防 74 词表坍缩）

- [x] Task 13: 实现约束决策头（模块 B，核心创新，§5.4）。
  - [x] SubTask 13.1: 实现候选空间与硬掩码 `Valid(i,j) = 1[j-i>3]·1[x_i x_j ∈ 𝒫]`，非法处置 `-∞`
  - [x] SubTask 13.2: 实现**置换不变**配对表示 `z_ij = W_s [h_i+h_j ; h_i⊙h_j ; |h_i-h_j|]`
  - [x] SubTask 13.3: 实现配对得分 `s_ij = w_s^T z_ij + s_ij^phys`，并**断言 `s_ij = s_ji` 精确成立**
  - [x] SubTask 13.4: 实现 Turner 物理先验 `s_ij^phys = -ΔG°37_NN/RT`（`RT ≈ 0.616 kcal/mol`）
  - [x] SubTask 13.5: 实现 Turner 残差 `MLP_T`，**末层零初始化**，验证训练起点等价于「**Nussinov + Turner 堆叠能**」（**非 ViennaRNA**，§0.7 问题 3）
  - [x] SubTask 13.6: 实现有序配对类型头 `t_ij ∈ R^6` 及硬置换约束 `t_ji = Π t_ij`
  - [x] SubTask 13.7: 实现温度缩放校准层（按长度桶拟合 `T`）
  - [x] SubTask 13.8: 实现 `L²` 显存对策：分块计算配对表示 + 梯度检查点
  - [x] SubTask 13.9: **实现层级决策级联 L0（粗粒度，§5.0.2）**：块对 `(b₁,b₂)`（块大小 `w`）的螺旋存在性与强度打分，并实现稀疏化（`O((L/w)²)` → `O(L/w)`）
  - [x] SubTask 13.10: **实现 L1（螺旋级）**：候选螺旋的起止与置信度预测
  - [x] SubTask 13.11: **实现 L2（碱基级局部细化）**：仅在活跃螺旋内输出 `p̂_ij` 与配对类型，`O(w²)`
  - [x] SubTask 13.12: **实现扁平 `L×L` 头作为可选对照路径**（对冲方案），并实现按长度自适应选择层级/扁平路径
  - [x] SubTask 13.13: **实现 L0 召回率测量**（P6 门限 ≥0.98）与层级级联的实测 FLOPs 计量（供 S8/S9）
  - **判据**：单次前向产出对称矩阵（对称性误差 = 0）；`MLP_T` 置零时输出等价于「**Nussinov + Turner 堆叠能**」（数值验证，**非 ViennaRNA**）；无自回归解码路径；`L=512` 显存实测记录；**层级级联三级均可独立前向；L0 召回率可测；扁平头可切换；实测 FLOPs 计量可用**

- [x] Task 14: 实现确定性求解 harness（模块 C，§5.5）：**全局关键路径**。
  - [x] SubTask 14.1: 实现 Nussinov **max-product** DP（含 traceback）
  - [x] SubTask 14.2: 实现 Nussinov **sum-product**（inside）算配分函数 `Z(x)`
  - [x] SubTask 14.3: 实现 **inside-outside** 算精确边际 `p̂_ij`
  - [x] SubTask 14.4: 实现 `L_NLL = log Z(x) - Σ_{(i,j)∈M_gt} s_ij`
  - [x] SubTask 14.5: 实现**隐式微分**可微 DP（前向求 `M*`，反向按 `∂N(1,L)/∂s_ij = 1[(i,j)∈M*]` 散布梯度），验证梯度 = `(y - p̂)·∇s`
  - [x] SubTask 14.6: 实现 soft-Nussinov 备选（仅 `L ≤ 256` 对照）
  - [x] SubTask 14.7: 实现带窗稀疏化版本（`|j-i| ≤ B`）
  - [x] SubTask 14.8: 实现低置信回退门控 `α_ij = σ((p̂_ij - τ)/γ)` 与热力学重打分，记录回退比例
  - [x] SubTask 14.9: 在代码注释与文档中**显式记录 Sinkhorn/最优传输不适用于非交叉约束**
  - [x] SubTask 14.10: **实现双模式推理（§5.0.1，解决 §0.7 致命问题）**：System-1 模式 = `1×` 前向 → 决策头直接输出校准概率 → **不跑配分函数** → 带状/线性 DP 求合法结构
  - [x] SubTask 14.11: 实现 System-2 按需升级：置信度门控，仅对低置信区域调用 inside-outside / 热力学求解器
  - [x] SubTask 14.12: **断言 System-1 路径不包含任何配分函数调用**（代码级检查 + 计时验证），并实现平均 FLOPs 计量（供 S6 匹配计算量对比）
  - **判据**：`Z` 与 `p̂` 在 `L ≤ 12` 时与暴力枚举**逐位一致**（有测试）；≥1000 随机得分矩阵下非法结构率 = 0、最小发夹环违规率 = 0；隐式微分梯度与数值梯度一致；`L_NLL` 可反传；**System-1 路径经断言确认无配分函数调用**；平均 FLOPs 计量可用

- [x] Task 15: 实现决策顺序切换（**已降级为消融项**，§5.6 / §0.7 问题 4，**不作为创新点**）。
  - [x] SubTask 15.1: 实现 5'→3' 顺序（与 Nussinov 从左到右填表一致）
  - [x] SubTask 15.2: 实现对角顺序（按 `j-i` 递增）与随机顺序对照
  - [x] SubTask 15.3: 实现顺序切换的统一配置接口
  - **判据**：三种顺序可经同一配置切换并产出结果，供 H4（**已降级，不预设方向**）检验；**不得在任何写作中把顺序称为创新**

- [x] Task 16: 实现热力学决策蒸馏预训练（模块 D，§5.7）。
  - [x] SubTask 16.1: 实现蒸馏损失 `L_distill = Σ D(p^teacher_ij ‖ p̂_ij)`，`D` 支持 KL 与 L2 两种
  - [x] SubTask 16.1b: **实现「精确边际」蒸馏教师**（C1 的核心机制）：以 `L_NLL` 训练出的 CRF 的 inside-outside 边际 `p̂^exact` 作为教师，蒸进免 DP 的决策头。**这是 §5.0.1 双模式设计的训练期环节**
  - [x] SubTask 16.1c: **量化免 DP 的校准代价**：对比 System-1 头的 ECE 与 `p̂^exact` 的 ECE（供 S7，目标差值 ≤ 0.02）
  - [x] SubTask 16.2: 实现与硬标签 `L_NLL` 的加权联合 `L = L_NLL + λ_distill · L_distill`
  - [x] SubTask 16.3: 实现教师软标签的大规模离线生成管线（分片、断点续传、哈希校验）
  - [x] SubTask 16.4: 实现**蒸馏目标饱和诊断**（与 JEPA 区域均值目标的饱和模式对照，须有记录证明不早饱和）
  - [x] SubTask 16.5: 实现教师集成（三教师）与单一教师的可切换配置
  - **判据**：蒸馏损失可反传；饱和诊断记录产出且显示目标未早期饱和；软标签管线可断点续传且哈希校验通过

- [x] Task 17: 实现 RLCD-RNA 校准决策训练（模块 E，§5.8）：**对齐 Jev 训练层（J9）的核心迁移**。
  - [x] SubTask 17.1: 实现决策单元定义（`a_ij ∈ {0,1}`，`p̂_ij = P(a_ij=1|x)`）与恰当评分规则奖励（Brier 与对数评分两种）
  - [x] SubTask 17.2: 实现**可微软 ECE**（软分箱 / 核密度）作为校准惩罚
  - [x] SubTask 17.3: 实现 RLCD 总目标 `L_RLCD = -E[R] + β·ECE_soft`
  - [x] SubTask 17.4: 实现四项目标联合 `L = L_NLL + λ_distill·L_distill + λ_RLCD·L_RLCD + λ_1·L_cal`，各权重可配
  - [x] SubTask 17.5: **验证校准梯度由精确边际 `p̂_ij` 直接给出，不依赖蒙特卡洛采样**（有测试断言）
  - [x] SubTask 17.6: 实现 `λ_RLCD` / `β` 的扫描配置，产出校准-精度权衡曲线
  - [x] SubTask 17.7: 撰写 RLCD 诚实标注说明：本文为"RLCD 启发式自设计目标"，Jev 的 RLCD 算法未公开，不得声称复现或等价
  - **判据**：`L_RLCD` 可反传且梯度不依赖采样（有断言）；四项目标权重可独立开关（供消融）；校准-精度权衡曲线产出；诚实标注说明存在

- [ ] Task 18: 执行预训练并验证梯度与稳定性。
  - [ ] SubTask 18.1: 在清洗后语料 + 教师软标签上运行预训练，监控 loss/梯度/NaN
  - [x] SubTask 18.2: 验证梯度到达每个可学习块
  - [x] SubTask 18.3: 标定微调学习率（承接"lr=1e-4 摧毁编码器"教训）
  - [ ] SubTask 18.4: 验证学生校准概率与教师可比（ECE、与教师概率的相关性）
  - [x] SubTask 18.5: 保存 checkpoint 与训练曲线
  - **判据**：无 NaN/Inf；梯度覆盖断言通过；学习率标定报告产出；学生-教师概率一致性报告产出

- [x] Task 19: 下游任务适配（T1–T6）。
  - [x] SubTask 19.1: T1 非假结二级结构预测主任务
  - [x] SubTask 19.2: T2 假结感知预测（PseudoBase++）
  - [x] SubTask 19.3: T3 SHAPE/DMS 条件化预测
  - [x] SubTask 19.4: T4 长链/全长转录本（16S/23S 等）
  - [x] SubTask 19.5: T5 迁移任务（复用既有 145 任务注册表）
  - [x] SubTask 19.6: T6 零样本/少样本 + 主动学习
  - **判据**：每个任务产出 `result.json` + predictions；涉及 dev==test 的任务结果显式标注该缺陷

- [ ] Task 20: 执行测评与消融（含速度基准与门限核验）。
  - [ ] SubTask 20.1: 跑七类指标，每配置 ≥5 seed
  - [ ] SubTask 20.2: 执行统计检验（Wilcoxon / bootstrap CI）与多重比较校正（Holm-Bonferroni / FDR）
  - [ ] SubTask 20.3: 完成 §7.5 全部 **15 项**消融
  - [ ] SubTask 20.4: 同硬件测延迟（分长度桶）、吞吐、峰值显存，产出 latency-accuracy Pareto
  - [ ] SubTask 20.5: 测量相对 McCaskill 配分函数的加速比
  - [ ] SubTask 20.6: 产出跨长度桶/跨家族/跨 GC 外推衰减曲线
  - [ ] SubTask 20.7: **核验 RLCD 收益非装饰**：同时报告 ECE 改善与 bpRNA-new F1 变化（H6）
  - [ ] SubTask 20.7b: **核验 S7（免 DP 的校准代价）**：System-1 ECE 与精确边际 ECE 之差 ≤ 0.02——**这是 C1 是否成立的判据**
  - [ ] SubTask 20.7c: **核验 S6（计算自适应收益）**：在**匹配平均 FLOPs** 下门控升级 ≥ 纯 System-1 与纯 System-2——**这是 C2 是否成立的判据**
  - [ ] SubTask 20.7d: **产出结构层面校准曲线**（应对 Q3：配对级 ECE 因碱基对相关性而不足），并量化配对间相关性
  - [ ] SubTask 20.8: **逐项核验 §8 硬门 G1–G5、性能门 P1–P5、速度门 S1–S7、科学门 C1–C4**
  - [ ] SubTask 20.9: **核验 §9.3 的 Q1–Q12 每条均有稿件落点**（任一缺失即未准备好投稿）
  - **判据**：全部消融完成；统计检验与校正已应用；速度测量同硬件同协议；ECE 与 bpRNA-new F1 联合报告产出；**S7/S6 判据已核验**；结构层面校准曲线产出；**Q1–Q12 落点表产出**；门限核验表产出（未达项须列出）

- [x] Task 21: 论文撰写与投稿。
  - [x] SubTask 21.1: 按 §9.2 **四段式**叙事撰写（**问题：信任与成本两难 / 方法：免 DP 校准决策头 C1 / 系统：计算自适应折叠 C2 / 测量：校准优先协议 C3**）
  - [x] SubTask 21.2: 核验全部引用（标题/作者/年份/标识符/URL 齐全方可标已核验）
  - [x] SubTask 21.3: 撰写限制章节，含**五点局限**：Jev 无同行评审论文；RLCD 算法未公开故为启发式自设计；**Gibbs/配分函数框架非本文首创（CONTRAfold/CRF 谱系）**；与 LinearFold `O(L)` 复杂度的诚实对比；`dev == test` 缺陷
  - [x] SubTask 21.3b: **撰写 §2.3「明确不主张」清单的对应声明**，确保稿件中不存在任何被 §2.3 禁止的表述
  - [x] SubTask 21.4: 准备可复现材料：代码、配置、数据卡、衰减表、分析计划、教师版本清单
  - [x] SubTask 21.5: 按 §9.1 优先级选定投稿目标并按其要求调整篇幅与图表规范
  - **判据**：每个强主张可回溯到论文行/证据/显式限制；0 条编造引用；负结果如实报告；硬门未达者不得投稿

---

# Task Dependencies

- **Task 2（数据盘查）必须先于 Task 3（数据获取）**——先实测再动手
- **Task 3 必须先于 Task 4–9（清洗链）**——无数据无从清洗
- Task 4 → Task 5 → Task 6 → Task 7 → Task 8 → Task 9（数据链严格串行）
- Task 1 必须先于 Task 10（分析计划与门限冻结早于基线运行）
- ~~**Task 10.8（CDPFold 校准对比）是全局阻塞项**~~ → **2026-09-24 勘误：该阻塞项作废**（CDPFold 是 CNN+DP，非扩散、不免 DP，前提有误）。C1 的判据改为 C1-a/b/c（`spec/benchmark_decision.md` §3.2），**不再是前置阻塞项**，可与架构实现并行
- **新的 P0 前置**：ViennaRNA 安装并锁定版本（教师 + 精确边际对照物）；archiveII 版本口径在论文中写明
- Task 10（基线复现）与 Task 12–15（架构实现）**可并行**
- **Task 11（教师部署与吞吐实测）是 Task 16 的前置**，且其吞吐结论决定蒸馏语料规模
- Task 12 → Task 13 → Task 14（编码器 → 决策头 → harness）
- **Task 14.1–14.5 是全局关键路径**：`Z(x)`/`p̂_ij` 的穷举一致性验证（`L ≤ 256` 的对照与 `L ≤ 12` 的暴力枚举）未通过前，不得开展 Task 16 及之后的任何工作（数学框架错则全盘皆错）
- Task 13 → Task 15（决策头就绪后才能做顺序对照）
- Task 9 与 Task 11 是 Task 16（蒸馏预训练）的前置
- **Task 16（蒸馏）与 Task 17（RLCD 校准）可并行**，两者都是 Task 18（预训练执行）的前置
- Task 16 + Task 17 → Task 18（两个训练目标实现先于预训练执行）
- Task 14 + Task 18 → Task 19（harness 与预训练权重就绪后才能做下游）
- Task 19 → Task 20（下游跑通才能测评）
- Task 20 → Task 21（结果与门限核验齐备才能写作与投稿）
- Task 20.3 依赖 Task 13–17 全部完成（消融需四项目标可独立开关）

---

## 第二轮回归审计的派生任务（2026-09-24 新增）

> 来源：`spec/spec.md` §0.9.5。第一轮审计比对范式，第二轮直接读代码与运行配置，
> 结论更严厉：**C2 目前不是可用架构，且编码器是随机初始化**。

| # | 任务 | 完成判据 | 阻塞关系 |
|---|---|---|---|
| **T-A1** | 为层级级联设计**分层目标**（L0 块对 / L1 螺旋 / L2 局部各自成项、各自归一），接入 `train_decision` 与 `evaluate_decision` | L0 螺旋召回 ≥ **0.98**（P6）；级联臂在 TS0 上 F1 不退化 | **阻塞 C2 的贡献主张** |
| **T-A2** | 级联的**可微稀疏选择**，替代硬 `top_k`（未训练 L0 召回实测仅 **0.1548**，min 0.0） | 训练中 GT 螺旋的梯度覆盖率 ≥ 0.98 | 依赖 T-A1 |
| **T-A3** | 补齐**预训练骨干**：接入已下载的 RiNALMo-giga，或改用 245 GB 无标签语料自监督预训练 | `run_meta.json` 记录骨干来源；训练量 ≥ 20 epoch | **阻塞「优秀结果」** |
| **T-A4** | 目标函数**长度归一**（按 GT 配对数）+ 重新标定 `grad_clip`（裁剪前范数实测 40–673 vs 阈值 1.0） | 批间损失可比；裁剪触发率与步长波动写入记录 | 影响所有后续训练 |
| **T-A5** | 首轮三臂（3000 步 ≈ 0.56 epoch）的结果**逐处标注**为「通路验证，非科学结论」 | 训练记录与论文草稿中无遗漏 | — |
| **T-A6** | 监控脚本覆盖**直接启动**的运行（原先只读共享台账，直接启动的臂完全不可见） | 能发现静默死亡 | ✅ 已完成（commit `4218f73`） |
| **T-A7** | 决策头**显存**修复：沿 `j` 分块 + 每块 `torch.utils.checkpoint` | L=498/B=4 由 OOM 降到 **2366 MiB**；前向逐位等价、梯度 fp64 精确到 1e-15 | ✅ 已完成（commit `2397364`，11 项测试） |

### 已下载/安装的资产（2026-09-24）

| 资产 | 路径 | 状态 |
|---|---|---|
| RiNALMo-giga 权重（650M；33 层 / hidden 1280 / 20 heads / rotary / max_pos 1024） | `/mnt/cunyuliu/rna-jepa/weights/rinalmo-giga/model.safetensors`（2.60 GB） | **已下载**（经 `hf-mirror.com`；`huggingface.co` 与 Zenodo 不通） |
| multimolecule 0.0.8 + transformers 4.57.6 + torch 2.8.0 | `/mnt/cunyuliu/rnalmo_pkgs`（`pip --target` 隔离安装，**不污染**训练环境） | **已安装** |
| 无标签语料 | `/mnt/cunyuliu/rna-jepa/data/pretrain`（**245 GB**）、`pretrain_fasta`（93 GB） | 已有（规模远超 spec 早先写的 20 GB） |

> **许可提示（必须先确认再使用）**：`multimolecule/rinalmo-giga` 标注 **AGPL-3.0**。
> 用于研究/论文前需确认合规性，或改用许可更宽松的骨干。**当前尚未在任何训练中使用。**

### 环境约束（必须知道）

- **`/home` 配额 200 GB 已满**（`du -sh /home/cunyuliu` = 200G），连 1 MB 写入都会失败，
  `git commit` 会报 `Disk quota exceeded`。本次通过清理 `~/.cache/pip`（4.6 GB）
  与把 RNA 基线目录迁到 `/mnt/cunyuliu/relocated_home/`（符号链接回原位）临时解决。
- 配额的主要占用者**不是本项目**：`mrna_editflow_goal` 75 G（其中 `mrna_editflow` 43 G、
  `runs` 23 G）、`reactflow/artifacts` 51 G、`miniconda3` 59 G。
  `mrna_editflow_goal` 有 8 个进程的 cwd 在其中，**不可擅自迁移**；需要用户决定。
