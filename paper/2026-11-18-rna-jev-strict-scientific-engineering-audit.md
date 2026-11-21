# RNA-JEV 严格科研与工程终审：证据、SOTA 距离与一个月执行路线

## 2026-11-18 第二修订：从“结构辅助”转向“折叠算法学习”

> 本节结合 px1 最新 HEAD=`3c5106befa3ba96b1e7ddcf78960e226f9a50e0e`、现行 `harness.py/decision_head.py/rlcd.py`、最新机制门 `STOP_NO_JUSTIFIED_NEW_MECHANISM`，以及 2025–2026 年神经算法推理、迭代精修和结构提取工作重写研究主线。下文中的新架构、新实验和新预期均为**待检验设计**，不是既有结果。原审计的事实与负结果全部保留；凡旧 §3、§6、§9、§11–14 与本节冲突，以本节为新的执行版本。

### A. 任务边界与真正的科学问题

本项目所称“下游任务”只指 RNA 二级结构的下列评测，不再把 ncRNA 家族分类或 RNA–蛋白界面列为本轮目标：

1. 八个主 benchmark：TS0、bpRNA-new、ArchiveII、TS1、TS2、TS3、TS-hard、TestSetB；
2. Rfam；
3. CHANRG Test、GenA、GenC、GenF；
4. 长序列精度、长程配对、覆盖和计算性能。

目标是在统一协议下尽可能取得全部任务 SOTA，但不得通过每个测试集单独调参、选择 checkpoint、选择推理深度或拼接不同模型身份制造“全板领先”。

**顶刊级科学问题不再是“结构辅助是否有用”，而是：**

> **RNA 二级结构预测能否从学习家族相关的 pair 映射，转变为学习一个可跨家族、跨拓扑和跨长度复用的折叠算法？符号动态规划的算法监督与神经物理势能结合，能否同时提高真实结构精度、组合泛化和长序列执行能力？**

这个问题直接对应 CHANRG 已暴露的科学现象：模型可以恢复局部 helix，却错误组装 multiloop/junction；扩大 RiNALMo 对 Test 的帮助远大于 OOD。它研究的是**组合泛化与算法泛化**，不是普通模块容量问题。

核心主假设 H-A：

> 在相同序列表示、神经势能和训练预算下，接受 DP 中间状态、控制流和 traceback 监督的模型，比仅接受最终 pair 监督的模型更能执行未见长度与未见拓扑上的折叠计算，并获得更高的 stem/topology-F1 与长程召回。

核心主假设 H-B：

> 可学习的神经物理势能与共享折叠转移规则解耦后，模型可超过固定 Turner 势能，同时保留符号 DP 的跨长度泛化；收益不能只来自更大的 backbone、更多测试时采样或后处理。

### B. 为什么 xens2 简单却持续难打，以及为什么旧改造没有回答这个问题

1. **xens2 不是单个“物理先验 + ResNet”。**它是两个 650M RiNALMo Plan-A 成员与 R2D 的 0.7/0.3 集成，再使用 Turner 和精确 DP。集成降低方差并组合 precision/recall 偏好，不能拿来证明一个简单单模型不可超越。
2. **xens2 本身不是强基线天花板。**bpRNA-new 上统一 xens2 micro-F1=0.5531420736，P=0.6574、R=0.4776；MXfold2=0.6688、EternaFold=0.6854。低召回说明其保守 pair 决策仍有明显改造空间。
3. **H0–H6 没有完成表征学习检验。**七臂 best@100，而 `warmup_head_steps=1600`；H0/H2 编码器逐张量相同。该结果能否定当前头部配方，不能否定结构监督对 backbone 的作用。
4. **旧 JEPA 只复用最终标签。**masked MSE、SIGReg、查询 token、EMA 等仍从同一 canonical-nested pairs 构造目标，没有提供“如何执行折叠”的监督。它们属于最终答案的另一种编码，而不是算法学习。
5. **局部 pair 表征不保证全局组装。**ResNet2D 和 Turner 能描述 stacking，精确 DP 只执行给定分数，不会自动纠正神经势能中学错的 junction/wiring。
6. **训练与解码概率语义不一致。**现行 CRF NLL 在全局 nested 结构上定义概率，distill/RLCD/calibration 却使用独立 `sigmoid(scores)`。后者既不保证每个核苷酸至多配对一次，也不与 DP 的加权势能语义一致。
7. **数据增加伴随优化混杂。**D1 的 804 条中 649 条超过 1024 nt，在 length-sum NLL 下，少量长链可能主导梯度；旧扩充结果不能证明实验结构数据本身无用。

### C. 新主线 FoldNAR：算法蒸馏与双半环折叠推理

FoldNAR 是工作名，核心不是再增加一个 attention/scorer，而是把模型拆成两个可独立检验的对象：

- **势能模型 `E_theta`**：从序列和真实结构数据学习 pair、stacking、loop、junction 与全局 wiring 的条件偏好；
- **折叠处理器 `T_phi`**：从符号 DP 的执行轨迹学习共享转移规则，在 max-plus 或 log-sum-exp 半环上更新结构状态。

现行 px1 `harness.py` 已提供可直接使用的算法教师，不需要先发明新的标签：

- `_max_dp_reference` 产生最优 inside 表 `N`、决策 `dec` 和 split `kp`；
- `_traceback_max` 产生最终嵌套结构与 traceback；
- `nussinov_inside`、`_inside_outside_reference` 和现行 CUDA inside/outside 产生 `Z` 与 pair marginals；
- `decision_head.py` 继续提供 RiNALMo pair 表示、Turner 残差和合法 mask，作为势能初始化与对照。

MAP 分支的参考递推为：

`N(i,j)=max{N(i,j-1), N(i+1,j), N(i+1,j-1)+s_ij, max_k[N(i,k-1)+N(k+1,j-1)+s_kj]}`。

这对应 max-plus 半环。把 `max` 换成 `logsumexp`，同一结构分解对应配分函数和结构后验。**FoldNAR 学习的是两种半环上的执行与状态转移，不是把 Nussinov 改名为一个新模块。**

#### C1. 算法教师提供的新监督

每条算法轨迹包含：

- 当前区间 `[i,j]`、对角长度和全局执行指针；
- inside/outside 数值状态；
- pair/unpair/split 离散控制状态；
- split `k`、父子和兄弟区间关系；
- traceback 操作及最终 pairs；
- log-domain 配分状态与 pair marginal。

这些监督来自**算法**，不是 RNA 真值 pair 的再次编码。它约束模型如何组合局部子问题，属于新的归纳偏置，满足最新机制门“独立数学命题”的要求。

#### C2. 两个数据世界，避免有限结构标签过载

**算法世界：无限合成执行数据。**随机生成序列、合法 pair mask 和物理上可解释的扰动势能，运行符号 DP 获得逐步监督。训练长度、势能数值范围、稀疏性和拓扑复杂度可以独立控制。该过程不使用任何 benchmark 标签，不涉及预训练污染。

**真实世界：高质量结构势能数据。**TR1c、经 QA 的 PDB/DSSR 和其他允许训练的结构只用于学习真实势能与最终结构目标。标签来源、实验/推断身份、canonical 删除、缺失残基和 family/topology 划分继续保留。

这样可分别回答“算法是否学会”和“真实势能是否正确”，不再要求有限且有噪声的结构标签同时教会模型全部计算规则。

#### C3. 折叠处理器的实现，不是普通模块替换

第一版处理器按区间和对角执行共享更新：

- 连续数值通道表示 inside/outside 与势能；
- 离散通道表示 pair/unpair/split、当前区间和控制指针；
- parent–child、sibling 和 junction 边表示结构分解；
- max-plus 单元执行 MAP 组合，log-sum-exp 单元执行后验组合；
- 每步将离散控制投影回合法状态，防止连续隐状态绕过参考执行；
- 全局控制状态单独监督，避免只学习局部数值拟合。

2025 年 Tropical Attention 的 max-plus 电路、Hilbert 投影度量与长度/数值 OOD 结果，为处理器提供了一个有理论依据的候选实现。其公开代码为 MIT，本次核查 commit=`e3c12f3e7c401245b9b5577d7181922b0150efc6`。可以先复用并测试 `torch_maxplus_matmul` 参考实现，再评估可选 tropical-gemm 后端；**不得把“换 Tropical Attention”本身当作贡献，贡献是折叠状态、算法轨迹、双半环学习和执行验证。**

2026 年离散 neural insertion sort 证明：最终输出与中间 hint 准确仍可能出现 rush-to-output，离散控制和全局控制状态对长度泛化至关重要。因此 FoldNAR 必须包含执行捷径诊断，不能只报 F1。

2026 年 SDSM 工作证明训练—提取一致性的重要性。其 Sinkhorn 归一化只作为概率一致性对照，不替代 nested DP，也不成为 FoldNAR 主架构。公开 SDSM 代码核查 commit=`13bd5e07ce5f9f019cdc437de3a93d422223025f`，但仓库未发现 LICENSE，未复制其代码；只参考公开算法并独立实现对照。

2026 年 Eco-RNA 证明迭代精修可在固定参数下增加 pair recall，但它精修的是序列表征，不是 DP 状态与结构控制流。其论文明确写明权重和推理代码将在未来发布，因此当前只能实现同量级迭代对照，不能宣称已复现官方 Eco-RNA。

前沿来源与实现核查：

- [Tropical Attention / Neural Algorithmic Reasoning for Combinatorial Algorithms](https://arxiv.org/html/2505.17190)；[MIT 实现](https://github.com/Baran-phys/Tropical-Attention/)。
- [From Shortcut Learning to Discrete Neural Insertion Sort](https://arxiv.org/html/2609.31114)。
- [Differentiable RNA Secondary Structure Extraction for Deep Learning](https://arxiv.org/html/2609.30752)；[SDSM 实现](https://github.com/TylerIllman/RNA-Secondary-Structure-Extraction)。
- [Scaling RNA Foundation Models through Iterative Refinement / Eco-RNA](https://arxiv.org/html/2610.13106)。
- [CHANRG](https://arxiv.org/html/2603.22330)。

本次已在 px1 真实调用现行 `_max_dp_reference/_traceback_max/nussinov_inside`，确认可取得 `N/dec/kp`、最终 pairs 和有限 `logZ`；已对 Tropical MIT 实现执行独立 max-plus 矩阵计算，结果与广播穷举参考一致。这只证明算法教师与数值原语可用，不是 FoldNAR 精度结果。

### D. 神经物理势能：在正确算法之外解决真实 RNA 偏好

FoldNAR 不是只模仿旧 DP。若只复制 Turner，精度上限仍是旧模型。势能必须针对真实结构分解：

`E_theta = E_pair + E_stacking + E_helix + E_loop + E_junction + E_wiring`。

- `E_pair/E_stacking`：pair type、最近邻 stacking 与现有 Turner 初始化；
- `E_helix`：helix 长度、closing pairs、register 和局部 bulge；
- `E_loop`：hairpin、internal、bulge、external loop 的序列与长度；
- `E_junction`：multiloop 分支及子 helix 组合；
- `E_wiring`：loop–helix 图中的全局父子与 sibling 组装。

势能训练仍使用真实 pairs/structures 与全局 NLL；蒸馏、Brier、校准等概率项若保留，必须改用同一 inside/outside 结构后验，不能继续把独立 sigmoid 当结构概率。SDSM 仅作额外对照。

**关键硬对照是“同一神经势能 + 精确符号 DP”。**它把两种收益分开：

1. 神经势能是否比 Turner/H0 更好；
2. 学习处理器是否真正学会执行，而非依靠额外容量或近似解码碰巧提高分数。

### E. 顶刊级机制证据：不仅证明分数高，还证明模型在算什么

1. **合成精确执行**：在未见长度、未见势能范围、未见稀疏度上，与符号 DP 比较中间状态、split、traceback 和最终结构。
2. **跨长度自由运行**：训练只在短区间，测试 2×、4×、8×长度；teacher-forced 与自由运行分开，报告误差随步数传播。
3. **rush-to-output 检查**：用最终 pair decoder 探测每个中间状态；若模型在对应算法阶段前已输出最终答案且 hint 不真实，则判定存在捷径，不能宣称算法学习。
4. **反事实状态替换**：将某一步神经 inside/split 状态替换为教师状态，后续执行应恢复；人为破坏 junction 控制应产生可预测的全局 wiring 错误。
5. **半环不变量**：检查能量整体平移、温度缩放、非法 pair mask、对称性和 batch/padding 不变性；数值误差与结构决策误差分开。
6. **同势能 DP 对照**：学习处理器与精确 DP 使用完全相同势能，任何差异只能归因于执行与近似，不得混入新的 head。
7. **真实 OOD 组合泛化**：按 loop–helix topology signature 和 family 双重隔离，特别分析“局部 helix 正确、全局 junction 错误”的样本是否减少。

### F. 数据与长序列路线

- 对 TR1c/VL0 统计 family、80% 簇、helix 数、loop 类型、multiloop 阶数、topology signature 和 pair span；训练采样不能由少数家族/拓扑主导。
- CHANRG Train 只用于官方 CHANRG 协议训练；Test/GenA/GenC/GenF 不参与训练、轨迹筛选、checkpoint 或推理预算选择。与八 benchmark 的数据关系单独审计，不混用协议。
- PDB/DSSR 只作为真实结构势能数据，不作为算法轨迹的必要来源；算法轨迹可以从任意势能计算。
- D1 在 length normalization 后来回复核，避免 80.7% 长链通过 sum-NLL 支配训练。
- 长序列采用候选区间/稀疏 tropical 执行和分段层级组合，但不得未测量就宣称线性复杂度。必须报告候选 pair/helix recall、丢失的真值交互、显存、FLOPs 和实际时延。
- 7224 nt、19000 nt 不截断、不删除；编码器可运行、候选生成可运行、完整结构输出可运行是三个不同门。近似稀疏执行不得冒称精确全局 DP。

### G. 实验矩阵与不可绕过的停止门

| 臂 | 势能 | 执行器 | 监督 | 回答的问题 |
|---|---|---|---|---|
| A0 | 当前 H0 | 当前精确 DP | 最终 pairs | 历史宿主 |
| A1 | 同一神经势能 | 精确符号 DP | 最终 structures/NLL | 新势能是否有效 |
| A2 | 与 A1 完全相同 | FoldNAR | 仅最终输出 | 是否仍会走捷径 |
| A3 | 与 A1 完全相同 | FoldNAR | 完整算法轨迹＋最终输出 | 算法监督的独立贡献 |
| A4 | 与 A1 完全相同 | FoldNAR | 轨迹 shuffle/错误控制流 | 排除任意辅助监督 |
| A5 | Turner/随机扰动势能 | FoldNAR | 合成执行轨迹 | 跨长度/数值算法泛化 |

停止门：

1. **轨迹门**：教师导出与独立实现不一致，或合法结构不能 100% 往返，停止。
2. **算法门**：FoldNAR 在合成未见长度上不能接近教师执行，不得开始大规模 RNA 训练，更不得宣称长序列算法泛化。
3. **捷径门**：A3 不能显著优于 A2 的轨迹真实性，或全局控制 shuffle 不影响执行，说明机制未成立，停止该配方。
4. **势能门**：A1 在三种子 VL0 上不超过 H0，先停止真实性能扩张；不得以 FoldNAR 的近似误差掩盖势能失败。
5. **真实性能门**：A3 相对 A1/A2 只在训练分布提升，而未见 topology/长度无收益，则算法泛化假设不成立。
6. **SOTA 门**：任何测试集不得反向选择处理器、势能、T、候选宽度或 checkpoint；未冻结前不跑完整外部榜。

### H. 八主 benchmark、Rfam、CHANRG 与长序列的统一验收

- 主指标：micro/macro pair-F1、precision/recall、stem-F1、topology-F1/GED、pair-span 与长度分层；小 split 同时给 n 与逐序列结果。
- 算法指标：合成 exact execution、split accuracy、traceback accuracy、自由运行误差、跨长度保持率、反事实恢复率。
- 长序列指标：覆盖率、候选召回、长程 pair/stem recall、完整输出成功率、峰值显存、实际时延和精度—计算 Pareto。
- 强基线：xens2、最佳单 Plan-A、RiNALMo 官方头、BPfold、EternaFold、MXfold2、UFold、eFold、SPOT-RNA、RNAformer；Eco-RNA 待官方权重/代码发布后纳入，未发布前不把论文表格数字冒充同协议复现。
- 一个冻结模型、一套 VL0 选择的配置用于全部任务；CHANRG 官方训练协议与 TR1c 历史协议分栏。单模型与集成、精确与稀疏执行、不同训练数据分开报告。
- 顶刊故事成立需要同时满足：真实 SOTA 或明确优势、跨拓扑/长度泛化、执行机制证据、数据与模型身份可复现。只有 F1 而没有执行证据，不能支撑“学习折叠算法”的标题。

### I. 最短执行顺序

1. **先导出算法轨迹**：复用 `_max_dp_reference`、`_traceback_max`、`nussinov_inside` 和独立参考实现，完成合成数据生成器、合法性与双半环测试。
2. **小长度证明算法可学**：只训练 FoldNAR 处理器，先过合成精确执行、自由运行、rush-to-output 和反事实门。
3. **训练真实神经势能**：A1 与同初始化 H0 对比，修复 checkpoint 在 backbone 解冻前早停的问题；按 VL0 macro/stem/topology-F1 选择。
4. **组合 A2/A3/A4**：同势能、同数据顺序、同预算，验证算法监督的独立贡献。
5. **长序列稀疏执行**：先候选召回和全覆盖，再精度与计算；不恢复旧 19,000 nt beam 作为主架构。
6. **最后冻结评测**：八主 benchmark、Rfam、CHANRG 和长序列统一运行，公开全部逐格结果与失败。

这条路线允许较快淘汰错误实现，但不预设必然 SOTA。它的价值在于把性能目标、数据利用、物理约束、数学执行和机制解释收束到同一个可证伪科学问题，而不是继续寻找下一个普通模块。

---

## 2026-11-18 第三修订：问题驱动的科学故事与完整实验合同

> 本节纠正此前“先设计模型、后寻找问题”的叙事顺序。FoldNAR 只是检验组合泛化假设的候选工具，不是预定论文主角。后续执行从 E0 错误分解开始；若自己的模型并不存在明确的 junction/wiring 瓶颈，不得为了维护 FoldNAR 名称而继续训练。完整执行提示词见 `交接文档/docs/plans/2026-11-18-foldnar-scientific-story-execution-prompt.md`。

### 1. 一句话科学故事

**当局部螺旋正确而全局折叠失败：RNA 二级结构预测需要学习可迁移的组合规则，而不只是配对相关性。**

八主 benchmark、Rfam、CHANRG 和长序列是验证目标，不是科学问题本身。科学问题是：当前模型究竟缺少局部配对信息、训练拓扑覆盖，还是把局部 helix 组装成全局结构的计算规则？如果是最后一种，算法轨迹监督能否提供最终 pair 标签之外的组合归纳偏置，并真实改善 OOD 与长序列？

### 2. 现象—假设—实验—结论的连续推理链

1. **现象**：xens2 在传统 split 上有优势但召回不足；CHANRG 证明 foundation model 可以局部 helix 正确而全局 junction 错误，扩大 backbone 不能自动修复。由此提出“局部正确是否掩盖全局组装失败”。
2. **E0 错误归因**：先在自己的 xens2、Plan-A、H0 和强基线上分解 pair→stem→loop→junction→topology 误差，并联合数据拓扑覆盖。只有 wiring 是明确瓶颈，才进入 FoldNAR；若主要是 pair coverage 或训练拓扑缺失，分别转覆盖率/数据路线。
3. **E1 算法可学性**：在无限合成势能上学习 DP 中间状态、split、traceback 和控制流，测试未见长度、数值范围和稀疏度。若合成 DP 都学不会，FoldNAR 停止，不消耗真实 RNA 训练预算。
4. **E2 真实势能**：同一神经势能接精确符号 DP，与 H0/Turner 比较。若精确 DP 都不能改善，不得靠神经执行器掩盖势能失败。
5. **E3 算法监督独立贡献**：同势能、同初始化、同数据顺序、同预算比较仅最终输出、完整轨迹、shuffle 轨迹。只有完整轨迹在未见长度/topology 上占优才继续。
6. **E4 机制归因**：检查中间状态、自由运行、rush-to-output、反事实状态恢复和 junction 干预。只有结果支持组合机制，才能使用“学习折叠算法”的论文表述。
7. **E5 最终验证**：冻结后统一评测八主 benchmark、Rfam、CHANRG、时间外和长序列，检查局部精度、全局 wiring、OOD 保持率与计算成本是否共同改善。

论文图按同一顺序展开：E0 失败现象 → FoldNAR 科学假设与算法轨迹 → 合成精确执行 → 真实势能与主 benchmark → CHANRG/反事实机制 → 长序列和计算 Pareto。每张图回答上一张图留下的问题，不再按模块或工作日志编排。

### 3. 数据角色、分布与泄漏门

| 科学角色 | 数据与当前边界 |
|---|---|
| 算法学习 | 合成序列、合法 mask、扰动势能和 DP 轨迹；不使用 benchmark 标签 |
| 主训练 | TR1c=42,564 条/15,521 个 80% 簇；需补 family、loop、junction、topology 数据卡 |
| 开发选择 | VL0=196 条；193/196 与已知三源精确重叠，只能作污染感知 model-dev |
| CHANRG 官方协议 | Train=123,223、Validation=14,070、Test=14,070、GenA=12,499、GenC=4,424、GenF=1,797；仅 Train/Validation 参与拟合与选择 |
| 实验结构 | PDB/DSSR、D1；区分实验方法、分辨率、缺失残基、modified-parent 和 canonical 删除 |
| 长序列 | D1、7224/19000 nt 压力链与时间外 PDB；尚需形成统一可评测面板 |
| 时间外 | 1805 个候选 PDB 条目；全量合格链 QA 与冻结精度未闭环 |

数据清洗必须同时记录序列规范、pair 索引与冲突、noncanonical/pseudoknot 删除守恒、PDB 坐标—序列映射、未观测残基、标签来源和文件哈希。实验、比较推断、数据库注释、热力学教师与模拟轨迹不得混称实验真值。

分布分析除长度/GC 外，必须覆盖 family/clan/80% 簇、pair density/span、helix 数量与长度、loop 类型、multiloop 阶数、junction 和 loop–helix topology signature。若训练集本身缺乏目标拓扑，应判定为数据覆盖问题，不能把全部失败归因于算法能力。

泄漏分四层：下游精确/同源、结构相似、预训练暴露、时间。状态固定为 `EXACT/HOMOLOGY/NOT_DETECTED_UNKNOWN/UNCHECKED`；未检出不等于 clean。不同 baseline 的训练语料、官方默认训练列表和 clean 子集必须分别绑定，不能套用同一个 TR1c-clean。

### 4. Benchmark 与 baseline 完整性

- 八主 benchmark 回答真实精度；Rfam 回答大规模迁移；CHANRG Validation/Test/GenA/GenC/GenF 回答不同生物层级的 OOD；时间外回答时间独立；长序列回答共享计算与长程召回。它们不得互相替代。
- 建立项目历史 25 系统、已完成 13 系统主榜和 CHANRG 29 predictors 的并集注册表。按结构解码、直接神经、foundation model、本项目四类管理。
- 必须覆盖 ViennaRNA MFE/centroid/MEA、RNAstructure、CONTRAfold、EternaFold、MXfold2、BPfold、UFold、eFold、SPOT-RNA、RNAformer、Kirigami、ContextFold、E2Efold、DEBFold、RiNALMo、ERNIE-RNA、RNA-FM、NucleicBERT、structRFM、xens2、Plan-A、R2D、H0；长序列加入 LinearPartition 类方法，Eco-RNA 待官方资产发布后复现。
- 权重、训练集、版本无法核验者标 `BLOCKED_PROVENANCE`，不得填 0、删除弱格或把不同 clean 子集直接排名。

### 5. 超参数与 OOD 预注册

- 算法参数只在合成 train/dev 选择；势能参数只在 VL0 或 CHANRG Validation 选择；长序列候选/分段只在训练内部长链 dev 选择；解码温度/MEA gamma/候选数只在独立 calibration-dev 选择。
- 三训练种子、同数据顺序和预算；checkpoint 以 macro-F1 为主，同时要求 stem/topology 不退化。禁止外部测试参与任何超参、checkpoint、候选宽度、推理深度或 verifier 选择。
- OOD 分轴报告：GenA 架构、GenC clan、GenF genome-sparse、topology-holdout、length-holdout、temporal-holdout、pretraining-exposure 和 label-source shift。
- 除总分外必须报告 Test→OOD 保持率及 pair/stem/junction/topology 错误变化。pair-F1 提高但 junction 不改善，不支持核心组合假设；GenA 提高而 GenC/GenF 不提高，需优先检查长度或数据分布解释。

### 6. 最终结果如何回到开头

论文结论必须逐一回答开头的问题：当前失败是否主要来自 wiring；算法监督是否改善组合执行；真实势能是否提高精度；这些变化是否跨 family/topology/length；最终是否超过同协议最强基线。只有性能、组合泛化、机制干预和复现证据同时成立，才能形成“学习折叠算法”的顶刊故事。若任一环节失败，按预先定义的分支缩小或改变主张，不用后验叙事掩盖。

---

## 2026-11-20 第四修订：完整矩阵后的路线纠正与优化稳定性门

### 1. 原裁决的有效边界

三种子、六次 20,000-step 训练和 C0–C3 已全部完成，原 `full_matrix_verdict.json` 为 `STOP_TRAIN_OFF_MAINLINE_C0_FINAL_STRONGER`，外部评测仍禁止。该裁决只说明 **terminal C2 不可选**，不能解释为 Turner-on 优于 Turner-off，因为两条训练路线均发生灾难性退化。

| step | C0 macro pair-F1 | C1 macro pair-F1 | C2 macro pair-F1 | 解释 |
|---:|---:|---:|---:|---|
| 3,200 | 0.745236 | 0.633849 | 0.778484 | C2 尚有开发集优势 |
| 9,600 | 0.182019 | 0.182019 | 0.443427 | 双臂显著退化，C2 退化较缓 |
| 16,800 | 0.092092 | 0.092092 | 0.003653 | C2 接近零输出 |
| 20,000 | 0.095240 | 0.095240 | 0.005102 | 全部路线失效 |

轨迹分析 `unfreeze_trajectory_analysis.json` SHA256=`609b6859423fa1f79ef923df433d931177b6175c3ac646ee9d15da71f633cef9`，其 `terminal_matrix_verdict_modified=false`。不得用 trajectory-best 替换原 terminal endpoint，也不得宣称生产修复或 SOTA。

### 2. 共同优化失效证据

六个 run 都在约 6k–10k 步出现 NLL 从 <1 到 >10，随后达到百万级；记录到的最大梯度范数超过 4×10^7。历史配置为 head LR=1e-4、backbone LR=1e-5、length-normalized NLL、逐步全解冻，固定跑满 20,000，训练期间没有 best-checkpoint 选择。故当前主要矛盾是**优化稳定性与灾难性遗忘**，不是继续选择物理先验或延长训练。

原 `C0_final_stronger` 只能触发停止 train-off，不可进一步写成“Turner-on 有效”。完整矩阵后的新科学问题是：在相同初始化和 Turner 开关下，能否先获得数值稳定且不退化的训练，再讨论先验收益？

### 3. 已完成的工程复核

- 修复 `analyze_prior_alignment_unfreeze_trajectory.py` 的 terminal 原始分数目录：由不存在的 `raw_train_on` 改为实际 `export_train_on`，并加入三种子、四阶段、terminal 无长度字段的回归测试。
- 相关 17 项测试通过，轨迹聚合成功；后续加入可选发散保护、独立优化合同和验收器后，22 项测试通过。
- 发散保护默认关闭，不改变历史行为；启用时在 optimizer step 前检查 loss/gnorm，停止该步更新，run 标为 `diverged`，不生成可误用的完成 checkpoint。
- 2-step 双臂全链路 smoke 已完成：真实 CUDA、196 条分数导出、调参与 selected decode、manifest 均通过。
- 正式 seed0 优化诊断已完成：双臂均跑满 9,600-step 且未触发发散门，最大 loss=0.6254/0.5762，最大 gnorm=13.37/7.67。独立验收 `stability_pass=true`，196×4 行指标重算一致。
- seed0 C2 micro/macro pair-F1=0.827895/0.807065；相对 C1 macro 提升 0.017349、相对 C0 macro 提升 0.020805，但 micro 比 C1 低 0.000709，precision 降幅超过原 −0.005 门，故 seed0 不能选择生产路线。
- seed1/2 已按相同配置启动，三种子聚合仍使用原冻结的 macro/micro、precision、stem 和 C0 全部门；24 项相关测试通过。聚合完成前不读取旧外部面板、不作 SOTA 判断。

### 4. 当前唯一放行的新实验

仅允许 seed0、9,600-step 单变量优化诊断：

- backbone LR 从 1e-5 降为 1e-6；head LR、shared-init、TR1c、batch 顺序、NLL、Turner on/off、解冻计划和 VL0/MEA 网格均不变；
- 保留 3,200/6,400/9,600 快照；以旧 9,600-step 结果为对照；
- 启用 `loss>10` 或 `gnorm>100` 发散即停；
- artifact 独立为 `artifacts/prior_alignment_optimizer_v1/seed0_backbone_lr_1e-6`，旧 artifacts 与 verdict 不改；
- 先通过 2-step 双臂全链路 smoke，再启动正式优化诊断；smoke 与 pilot 都不是生产或 SOTA 证据。

若低 LR 仍发散，停止当前全参微调路线，转少层/低秩适配与更保守优化；若稳定，再比较同 step 的 pair/stem 和退化幅度，三种子复核前不得外部评测。

---

## 2026-11-21 第五修订：从“工作很多”到“性能证据链”

### 1. 为什么没有形成核心 SOTA

此前的问题不是工作量不足，而是科学归因失控：模型身份、数据子集、集成、Turner、MAP/MEA、warmup、全参遗忘和数值发散同时变化，造成结果无法可靠比较。FoldNAR 也一度先于证据成为默认答案，已被 E0 停止。当前不能把忙碌、模块数量或工程验收数当成核心性能。

新的主线是：**先恢复 pair coverage，再研究物理先验、学习势能、结构解码和优化如何共同决定泛化。**顶刊候选问题是“热力学先验何时帮助、何时抑制神经 RNA 折叠，神经势能、结构后验与真实数据分布应如何对齐”，不是再设计一个普通大模块。

### 2. 当前叙事与证据的逐项对应

| 叙事 | 直接证据 | 当前能说到什么程度 |
|---|---|---|
| 不是局部 helix 已学好只缺 wiring | E0 71 runs/174,007 行，`major_rule2_splits=[]` | 支持先查 pair coverage，暂停 FoldNAR |
| 部分漏召回被先验/解码压住 | Plan-A Turner-off MEA 通过；xens2-off+MEA VL0 +0.022278，外部 7/8 同向改善 | 只限条件性解码贡献，不是生产修复或 SOTA |
| 不能直接用 20k 比较先验 | 六 run NLL 百万级、gnorm 超 4×10^7，C0/C1 terminal 也仅约 0.095 | 原 verdict 仅停止 C2，不能证明 Turner-on 优胜 |
| 先修优化再谈架构 | seed0 低 LR 双臂完成 9600，max loss=0.6254/0.5762，max gnorm=13.37/7.67 | 数值稳定已证实，三种子仍需复核 |
| C2 可能恢复 pair/stem | seed0 C2 micro/macro=0.827895/0.807065，macro 超 C1 0.017349、超 C0 0.020805 | micro 与 precision 未过全部门，不能选生产路线 |
| 是否成为 SOTA | 三种子与外部统一 baseline 尚未闭环 | 完全未证明 |

### 3. 后续实验必须一环扣一环

1. 先完成低 LR 三种子，使用原冻结全部门，不根据 seed0 改判据。
2. 将 FN 分解为 LABEL/MASK/RAW_SCORE/DECODER/OPTIMIZATION/DATA_SHIFT/REPRESENTATION，不用“数据不足”概括一切。
3. 数据修改只针对明确的 FN 群体：长度梯度、来源冲突、拓扑覆盖或家族不均衡，不再盲目混合。
4. 模型修改只针对剩余瓶颈：Turner 角色、统一 inside–outside 后验、MEA 目标和保守适配，不再普通模块堆叠。
5. 最后才评统一 baseline、CHANRG/时间外/长序列与 OOD 保持率。

每一步都必须回答前一步留下的问题。若 micro/precision 不过门，转入后验校准与目标诊断；若种子方向不一致，先查训练稳定性；若绝对性能不超 C0，停止 train-off；若只赢自己的消融臂，不进入 SOTA 叙事。

### 4. 新版执行提示词

完整提示词已更新为 `交接文档/docs/plans/2026-11-18-foldnar-scientific-story-execution-prompt.md`，标题为“RNA-JEV 性能主线完整执行提示词：先验—势能—解码—优化对齐”。其中明确：不以 FoldNAR 为默认路线、不盲目占满 GPU、不删除负结果、不降低判据，并要求每个核心 claim 都绑定直接 artifact 和反例边界。

---

## 2026-11-22 第六修订：RAW_SCORE 的表征—优化归因与 v3 四臂门

### 1. 最终矩阵后的正确结论

三种子低 LR 的 6/6 run 均完成 9,600-step，冻结门通过 8/9。C2−C1 的 micro pair precision 均值差为 −0.0253695431，低于 −0.005 下限，因此 C2 停止，外部评测未开放。相对 C1，C2 micro/macro pair-F1 分别提高 0.017345/0.032405，说明存在真实性能信号，但不能用其他 8 个通过项掩盖 precision 失败。

当前统一主 benchmark 可声明完整 SOTA 的格子为 0/8。39 项最终回归通过和 PPT/报告交付只证明证据链可复核，不证明模型成功。

### 2. 主要矛盾已从解码转为 RAW_SCORE

三种子 C2 共 2,650 个 FN：RAW_SCORE=2,416（91.17%），直接 DECODER=234（8.83%），LABEL/MASK/DATA_SHIFT=0。后验重放 588 条、604 项检查通过，但校准方向跨 seed 不一致，未授权 B2，未搜索温度、gamma 或阈值。

因此下一步不是继续调解码或盲目扩数据，而是回答：**RAW_SCORE 漏召回主要来自全参遗忘、有限适配不足，还是训练内结构边际目标？**RAW_SCORE 只是误差定位，不自动证明表示或优化因果。

### 3. 当前唯一允许先完成的机制实验

最新合同为 `configs/raw_score_mechanism_4arm_v3.json`，四臂为：

- `full_control`：复用稳定全参对照，不重训；
- `frozen`：共享预训练表示＋同初始化头，检验不适配能否判别 pair；
- `limited4`：只解冻顶部 4 层，检验有限适配能否减少遗忘；
- `marginal`：相同全参计划下加入精确 inside–outside marginal BCE，检验训练内结构边际目标。

v1 central difference 在 BCE clamp kink 附近产生伪大梯度；v2 高 log-partition 下 parent accumulation 溢出；v3 改为 exact forward-mode tangent 和 shifted log-sum-exp。v1/v2 partial optimization 都不是科学结果，marginal 必须从同一 seed0 shared-init 重训，不能把数值实现差异写成科学收益。

当前 `verification_v3.json` 尚未生成，资源门要求 20,000 MiB，仍在等待，不得重复启动或降低资源门。

### 4. 每臂必须通过的全部门与分流

- micro AUPRC 和 recall@precision=0.8 严格提高；
- micro/macro pair-F1 非负，至少一项严格为正；
- micro/macro precision 降幅不超过 0.005；
- micro/macro stem-F1 降幅不超过 0.005；
- 9,600-step、预期解冻数、有限 loss、独立指标与 provenance 全部通过。

只有达到全部门的臂才能增加 seed1/2。frozen/limited4 更好只支持全计划遗忘是贡献因素；marginal 更好只支持训练内结构边际路线；全部失败则停止同路线，禁止阈值、解码或普通模块救援。

### 5. 新科学故事

**序列知识为何没有转化为配对判别能力？**先由 RAW_SCORE 证明主要矛盾在打分，再由 frozen/limited4/marginal 区分适配与训练目标，随后只让获胜机制做三种子，最后统一评估强 baseline、CHANRG、时间外和长序列。任何一步失败都不得跳级写 SOTA 或生产修复。

完整提示词已更新为 `交接文档/docs/plans/2026-11-18-foldnar-scientific-story-execution-prompt.md`，标题为“RNA-JEV 性能主线完整执行提示词：RAW_SCORE 的表征—优化归因”。提示词要求先完成 v3，不另起重复实验，并保留数据质量、四层泄漏、长序列和统一 baseline 的验收边界。

---

## 原审计（事实与历史执行方案保留）

- 日期：2026-11-18
- 审查对象：`px1-jcy.matpool.com:27693` 当前工作区 `/home/rna_jev_xens2_dualview_round56`，以及本地 `交接文档/` 现行文档、权威训练台账、历史结果表和证据稿。
- 远端 HEAD：`a29b75d202007650f42c19e724ba0a4e4c129c73`，分支 `dual-view-jepa-round56-20261116`。工作区另有多份未跟踪脚本，故 HEAD 不能单独定义实验身份。
- 证据标记：**[事实]** 可由代码、配置或原始 artifact 直接确认；**[推断]** 由现有证据合理推出；**[不足]** 当前材料不能回答；**[待验证]** 必须新增实验或文献核查。
- 审查边界：本轮没有启动训练、没有修改模型参数、没有补造实验结果。历史归档用于追溯，不与现行合同混用。公开文献用于确定必须比较的协议和基线，不把不同数据划分的数字横向拼接成排行榜。

---

## 1. 执行摘要

### 1.1 总结论

**当前项目不具备宣称“新架构在未见 RNA 家族上达到 SOTA”的证据，也尚不具备提交完整算法预印本的证据链。算法主线处于 A 阶段；数据污染审计、界面标签 QA 和受控负结果构成有价值的技术报告基础，但不能替代主科学问题。**

关键原因按优先级排列：

1. **[事实] 当前核心双视图假设在预定开发集终点上为负。** H0 macro-F1=0.8514596293，H2=0.8442637614，H2-H0=-0.0071958679；正式裁决为 `STOP_PILOT_NEGATIVE_DEV_ONLY`。这是单 seed、Plan-A warm-start、model-dev 负结果，不能全称否定结构学习，但足以禁止继续宣称当前 H2 有效。
2. **[事实] 现有外部测试无法支持 clean-OOD。** bpRNA-new 的 5388/5388 条序列均在已知 RiNALMo 预训练来源中检出 80% identity、双向 80% 覆盖同源匹配；精确未检出只有 7 条，且“未检出”仍是未知而非 clean。同源未检出对照样本数为 0。
3. **[事实] 当前常用强结果不是同一模型身份。** `xens2` 是两个 Plan-A 成员与 R2D 的 0.7/0.3 集成；`tables/sota_vs_ours.md` 自身也声明混合了不同模型身份和参照版本。集成结果不能作为单 checkpoint 或 JEPA 架构创新证据。
4. **[事实] 当前性能与强基线存在实质差距。** 在项目历史协议的 bpRNA-new 上，统一 xens2 micro-F1=0.5531420736，而 EternaFold=0.6854、MXfold2=0.6688、eFold=0.6640、UFold=0.6106；对应绝对差距分别为 0.132258、0.115658、0.110858、0.057458。该比较仍需统一 clean 子集、训练语料、预训练暴露和推理约束，不能直接称为最终公平排行榜。
5. **[事实] 可复现工程门并非全绿。** 全仓收集为 696 tests、12 collection errors；隔离运行 `tests/` 为 501 passed、1 skipped、8 failed。当前双视图、BPfold、界面、预训练暴露和时间外数据相关的 51 项定向测试在显式 `PYTHONPATH=src` 后全部通过，但这不等于全仓测试绿色。
6. **[事实] 时间外数据有进展但尚无模型结果。** 1805 个 PDB 条目目录中 1803 个具备 `labels.json/dssr.json/parents.json` 三件套，9KFX 缺 `labels.json`，9P4K 缺 `labels.json/dssr.json`；1803 个可解析 labels 文件合计 2930 个 chain records，但 578 个条目为 `chains=[]`。文件覆盖和 chain record 数都不是可评测独立 RNA 链数。现有正式 QA 仍是较早的 314 条部分进度快照，时间外集合没有冻结模型精度结果。独立文件清单 SHA256=`31ce8b1dcd674c9e2f6fbe104c5f6c8e9c82746fe15eb940b386ca3de9e67f65`。
7. **[事实] 一个月预印本的可行路径是先证据、后架构。** 立即完成时间外全量 QA、统一指标与基线、冻结模型评测，再用小规模受控实验决定保留、修改或放弃结构辅助路线。继续堆模块只会扩大混杂因素和复现负担。

### 1.2 对“一个月优秀结果”的现实定义

一个月内可以合理争取两类结果之一：

- **正向算法结果**：单模型在同一训练/开发协议下稳定超过 H0，并在时间外或结构 OOD 上至少一个预注册主终点取得可重复提升，随后再讨论是否达到统一协议 SOTA。
- **高可信负结果/基准结果**：若结构辅助仍无效，则交付“预训练暴露、结构划分和评价协议如何改变 RNA 二级结构泛化结论”的完整基准与受控负结果预印本。

**不能保证一个月内一定 SOTA；可以保证的路线是每周用不可绕过的证据门淘汰错误方向，避免在错误问题上继续消耗训练预算。**

---

## 2. 项目真实状态

### 2.1 完成度分层

下表严格区分“有代码”“能运行”“有可靠结果”“支持论文主张”。

| 模块 | 当前可确认状态 | 证据与限制 | 能否支持主论文主张 |
|---|---|---|---|
| 数据获取 | 有代码、已运行、部分可靠结果 | bpRNA、Rfam、ArchiveII、PDB/界面和时间外集合均有处理链；时间外仅文件覆盖接近完成 | 不能单独支持 |
| 数据清洗与预处理 | 有代码、部分强 QA | 界面新标签从 mmCIF 逐残基重建；modified residue 与 parents map 哈希绑定；时间外全量链级 QA 未闭环 | 界面可支持 probe；主任务仍不足 |
| 数据划分 | 有代码、已运行 | 有 PDB/group/家族/80% 簇隔离；但划分隔离不等于预训练去污染 | 不支持 clean-OOD |
| 模型实现 | 有代码、可运行 | 当前 RiNALMo Plan-A、双视图辅助、ResNet2D/Turner/精确 DP 均存在；旧 mRNABERT JEPA 是另一条模型线 | 支持“实现完成”，不支持创新有效 |
| 训练流程 | 有代码、已多次运行、有日志 | `tools/train_plan_a.py` 有 resume、渐进解冻、损失和身份记录；正式 H2 三种子与外部评测未启动 | 只支持开发集结论 |
| 推理流程 | 有代码、可运行、身份约束较强 | `evaluate_single_checkpoint.py` 禁止 dual-view 键和调用，记录哈希并要求重复前向 bitwise 一致 | 支持序列单模型身份，不证明精度 |
| 评估流程 | 有代码、部分可靠结果 | pair micro/macro-F1、覆盖、预测和哈希齐全；缺统一 stem/topology、多层级 OOD 与公平基线面板 | 不足以支撑 SOTA |
| 基线复现 | 部分代码、历史结果 | EternaFold、MXfold2、UFold、eFold、BPfold 等有历史值；训练数据、clean 子集和预训练暴露并不统一 | 不足以支撑公平 SOTA |
| 消融实验 | 有代码、已运行、开发集可靠负结果 | H0–H6 七臂、BPfold motif、Turner=0 等；多为单 seed model-dev | 支持否定当前具体路线，不支持普适结论 |
| 泛化实验 | 有历史外部测试、无合格 clean 对照 | bpRNA-new 同源未检出 n=0；时间外未评测 | 不支持 |
| 可解释性/机制 | 有 probe、梯度与表示诊断 | 单 batch 梯度 cosine、H0/H2 权重别名、界面 probe 均有明确边界；没有机制因果链 | 不支持“为什么有效” |
| 可复现性 | 有较好 artifact 习惯、工程债明显 | 哈希、manifest、逐序列预测丰富；但脚本未跟踪、同名测试、退役路线失败和环境依赖仍存在 | 部分支持技术报告 |
| 工程质量 | 核心路径可运行，全仓非绿色 | 51 项当前定向测试通过；全仓 12 收集错误和 8 失败需分类处理 | 不能写“全部测试通过” |
| 论文材料 | 有非投稿就绪证据稿和主张矩阵 | 3346 词、6 页 PDF 已复核；其定位是污染感知技术证据，不是新算法论文 | 不支持算法预印本提交 |

### 2.2 当前最可靠的结果

- **[事实] 冻结 xens2 在 bpRNA-new：** macro-F1=0.5422090871，micro-F1=0.5531420736；P0 历史与重放逐序列一致。证据 `xens2_bprna_new.json` SHA256=`c1c39477b84c3a139a52a091f1011548b1841c50bf6f5e572dc56b75318007ec`。
- **[事实] 双视图开发集：** H2 低于 H0，且没有超过 shuffle、随机结构编码器、参数匹配 MLP 或全局均值等关键对照。
- **[事实] 界面 probe：** H0 micro-AUPRC=0.7715063791±0.0052653141，高于 RiNALMo=0.7215875413、手工特征=0.5437812106、链内 shuffle=0.6291660948；但 holdout 存在预训练暴露，且选定 step100 的 H0/H2 编码器完全相同。
- **[事实] 长链工程：** 7224nt、19000nt 编码器前向可运行；这不能证明完整结构头可运行，更不能证明长链精度。结构辅助全 pair 模块曾在 7224nt OOM。
- **[事实] 速度：** xens2 正式独占结果约 13.5802 seq/s，自适应桶约 14.1949 seq/s，不支持速度 SOTA 主张。

### 2.3 代码、文档与结果的不一致

1. 历史表格曾给人“6/8 或 7/8 SOTA”的印象，但当前统一身份复核不支持把该说法用于单模型；raw 主榜约 5/8 领先、最佳单 checkpoint 约 4/8，且仍需统一协议。
2. `train_plan_a.py` 与未跟踪的 `train_plan_a_clean_head_p1.py` 不是同一入口；后者缺少双视图 H0–H6 参数。复现必须记录脚本 SHA256，不得只写“train_plan_a”。
3. `src/rnajepa/model.py` 的旧 mRNABERT 区域 JEPA 与当前 RiNALMo 双视图不是同一模型；论文若只写“RNA-JEPA”会造成身份混淆。
4. `src/rnajepa/encoder.py` 内“权重不在工作区”和 adapter `NotImplementedError` 属于旧自定义编码器路径，不能据此推断服务器没有 RiNALMo。
5. 时间外目录文件数已经增长，但正式 `partial_label_progress_qa.json` 仍是旧快照；不能把新文件覆盖写成已验收数据，也不能继续只引用旧的 314 条作为当前文件总量。

---

## 3. 核心科学问题审查

### 3.1 当前问题是否统一

当前合同的一句话问题是：

> **训练期实验/可靠二级结构能否通过序列—结构双向潜在预测，改善推理时仅使用序列的单 checkpoint 在未见家族上的结构预测精度？**

代码、H0–H6 对照和单模型评估器基本围绕这一问题展开；但历史集成、通用复合物扩充、界面 probe、长链 beam 和速度实验不断进入主叙事，导致科学问题被多个工程目标稀释。

### 3.2 任务、输入、输出与假设

- **目标任务**：单序列 RNA 二级结构预测；当前部署严格对应 canonical、nested base pairs，不覆盖 pseudoknot、非 canonical pair 或完整 3D 环境。
- **输入**：推理只输入 RNA 序列；训练可输入序列、真值 canonical nested pairs、教师概率和结构辅助目标。
- **输出**：合法 pair score matrix，经合法 pair mask 与精确 Nussinov nested DP 后输出碱基对集合。
- **核心假设**：结构视图提供序列主损失之外的互补监督，使 RiNALMo 表示更具可迁移结构信息。
- **希望超越的方法**：同骨干同头 H0、RiNALMo 官方结构头、BPfold、ERNIE-RNA、MXfold2、UFold、SPOT-RNA、EternaFold/CONTRAfold 等；历史表中的 PRISM 身份尚需核验，不能仅凭名称列为已确认公开 SOTA。
- **当前创新类型**：主要是**训练策略创新 + 工程整合**，不是已证实的新概念或新建模范式。双向 masked latent MSE、target SIGReg、结构 token、cross-attention、RiNALMo、ResNet2D、Turner 先验和 DP 均来自已有思想；新颖性必须落在特定组合及其可验证机制上。

### 3.3 建议重新定义的论文级问题

> **在显式记录预训练暴露、并按时间、序列同源和结构拓扑分层的条件下，训练期结构辅助能否让序列单模型在时间外/结构 OOD 的 pair、stem 和 topology 指标上稳定优于同骨干、同数据、同解码器的 H0，而不是仅提高同源插值分数？**

这个定义可证伪、与当前部署一致，也能容纳负结果。若最终回答为“不能”，论文贡献应转为“何种评估与结构监督不能带来可迁移收益，以及为什么”。

---

## 4. SOTA 距离分析

### 4.1 主任务分析表

| 项目 | 内容 |
|---|---|
| 目标任务 | 序列单模型 canonical nested RNA 二级结构预测；训练可用结构，推理禁用结构标签 |
| 主要数据集 | TR1c/VL0 仅作历史开发；TS0、bpRNA-new、ArchiveII、Rfam 作污染感知外部敏感性；时间外 PDB/DSSR 集合作优先独立测试；CHANRG 可作外部协议复现，不应与旧表数字直接拼接 |
| 数据划分 | 训练—测试按序列/80% 簇/家族隔离；另需时间切分、结构拓扑切分和 RiNALMo 预训练精确/80% 同源暴露分层 |
| 主要指标 | 每序列 macro-F1 与全体 micro-F1；新增 stem-F1、topology-F1/GED、长度与家族分层、覆盖率和置信区间；ECE/AUPRC 仅作辅助 |
| 强基线 | 同骨干 H0、RiNALMo 官方头、ERNIE-RNA、BPfold、MXfold2、UFold、SPOT-RNA、EternaFold/CONTRAfold；xens2 单列集成参照 |
| 当前结果 | H2-H0 model-dev macro-F1=-0.0071958679；xens2 bpRNA-new micro-F1=0.5531420736；旧 xens2 Rfam micro-F1 约 0.6514，扩充复合物后 0.6455382794 |
| 已知 SOTA | **不存在可直接套用到本项目的单一数字。** RiNALMo 原论文 ArchiveII 多数家族领先；统一 RNA-LLM benchmark 显示 ERNIE-RNA/RiNALMo 强但困难家族失效；CHANRG 预印本显示 Test 与 OOD 排名倒挂。必须在统一协议下重算 |
| 主要差距 | 独立测试不足、结构辅助未证明有效、历史结果低于多个强基线、局部 pair 指标与高层拓扑目标不一致、全 pair 头限制长链 |
| 数据泄漏风险 | 已确认高水平预训练暴露，不是假设风险；bpRNA-new 同源暴露 5388/5388，界面 holdout 也有暴露；nt 历史快照不可追溯，完整分母不可获得 |
| 公平性问题 | 参数量、RiNALMo 预训练、教师数据、集成、R2D、Turner、clean 子集和 DP 均可能不同；必须按“单模型/集成”“同训练集/不同训练集”“预训练暴露已知/未知”分栏 |

### 4.2 历史协议中的量化差距

| bpRNA-new 方法 | micro-F1 | 相对统一 xens2 的差值 | 证据边界 |
|---|---:|---:|---|
| 统一 xens2 | 0.5531420736 | 0 | 两个 Plan-A 成员 + R2D 集成，不是单模型 |
| UFold | 0.6106 | +0.057458 | 项目历史协议实测，公平性仍待统一 |
| eFold | 0.6640 | +0.110858 | 外部训练数据与规模不同 |
| MXfold2 | 0.6688 | +0.115658 | 热力学集成与训练协议不同 |
| EternaFold | 0.6854 | +0.132258 | 热力学/集成参照，不应简单归因于表征能力 |

Rfam 上历史 BPfold=0.7872 与旧 xens2≈0.6514 相差约 0.1358；同样只能作为“需要统一复现”的强警示，不能直接写成最终公平差距。

### 4.3 公开文献对项目叙事的约束

1. RiNALMo 原论文使用 650M 参数和 36M ncRNA 预训练，并在 ArchiveII 多数家族上表现强；其 t-SNE 只说明家族信息聚集，不能证明无污染泛化。[Nature Communications 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12219582/pdf/41467_2025_Article_60872.pdf)
2. 统一 RNA-LLM benchmark 显示 RiNALMo/ERNIE-RNA 在多个家族较强，但在 telomerase 等困难家族上表现很差，23S/16S 等也不是对所有基线占优；因此本项目必须加入 ERNIE-RNA 和传统结构解码器，而非只比较弱深度学习基线。[Briefings in Bioinformatics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11982019/pdf/bbaf137.pdf)
3. CHANRG 预印本报告 foundation-model 方法常规 Test 的优势在 GenA/GenC/GenF 上大幅下降，而 BPfold、EternaFold 等保留率更高。这直接支持本项目增加结构拓扑划分和 stem/topology 指标；但它是预印本、协议不同，数值不能移植为本项目 SOTA。[CHANRG, arXiv:2603.22330](https://arxiv.org/pdf/2603.22330)
4. BPfold 的核心是完整三邻居 base-pair motif 能量与混合 Transformer/CNN，不是本项目简单添加旧 Turner/RIPE 就能复现的路线。[Nature Communications 2025, DOI:10.1038/s41467-025-60048-1](https://doi.org/10.1038/s41467-025-60048-1)

### 4.4 差距来源排序

1. **科学评估问题（最高置信）**：把家族/PDB 隔离当预训练独立，把插值或 proxy 当 OOD。
2. **核心训练假设未成立（高置信）**：当前 H2 在预定开发集为负，继续加同类辅助损失没有依据。
3. **目标与结构归纳偏置不足（中高置信，待消融）**：局部 pair/ResNet 与 nested DP 主要保证合法结构，不一定学习高层 helix/topology；需要 stem/topology 消融验证。
4. **数据规模、质量与标签异质性（中置信）**：历史混合来源标签和通用复合物扩充未提高主结果，时间外标签尚待 QA。
5. **训练与 checkpoint 选择不一致（已确认存在，影响幅度未知）**：训练按组合 validation loss 选 best，科学终点按 macro-F1；NLL 默认按 length 不归一，辅助项按 pair 均值。
6. **推理/后处理混杂（已确认存在，非泄漏）**：Turner、可学习校准、精确 DP 和集成均影响最终指标；不能把提升全部归因于表征学习。
7. **纯粹表征容量不足（当前证据不足）**：650M RiNALMo 已很强，没有证据说明继续增大 backbone 是首要瓶颈。

---

## 5. 发表潜力评分

| 维度 | 分数（1–5） | 严格依据 |
|---|---:|---|
| 1. 科学问题重要性 | 5 | RNA 结构 OOD、预训练污染与可信评估均是领域关键问题 |
| 2. 现有方法局限是否明确 | 4 | 已有污染和公平划分证据；仍需用统一实验证明这些局限如何改变精度结论 |
| 3. 核心方法新颖性 | 2 | 当前主要是已有模块组合，H2 未显示开发集收益 |
| 4. 跨任务/跨领域意义 | 2 | 界面 probe 有补充价值，但主方法只覆盖 canonical nested 2D，跨任务机制未证明 |
| 5. 实验设计完整性 | 2 | 开发对照较认真；正式三种子、时间外、公平基线、stem/topology 和完整消融缺失 |
| 6. 结果可信度 | 3 | 哈希、逐序列预测和独立复核提高可信度；历史身份混用与测试非绿色降低可信度 |
| 7. 机制解释/科学发现 | 1 | 单 batch 梯度、表示图和 probe 不能构成机制因果链 |
| 8. 泛化能力 | 1 | bpRNA-new 无同源未检出对照，时间外无模型结果 |
| 9. 可复现性 | 3 | manifest/QA 较好；未跟踪脚本、环境依赖、失败测试和外部权重许可仍需整理 |
| 10. 论文故事完整性 | 2 | 证据稿逻辑较清楚，但“新算法提升未见家族精度”的核心链条在实验处断裂 |
| **合计** | **25/50** | 不是投稿就绪状态 |

### 阶段判断

- **算法主线：A——工程原型，尚未形成论文级科学贡献。**
- **数据审计支线：接近 B，但还缺“暴露如何影响结果”的完整实验，不宜自称已形成完整领域论文。**

进入 B 的必要条件：时间外全量 QA、统一评估协议、冻结模型与强基线面板、可复现测试门、单一核心主张。进入 C 的必要条件：三种子受控提升、至少一个真实时间外/结构 OOD 终点、完整消融和机制证据。只有这些完成后才讨论 D/E。

当前工作不能仅描述为“增加模块并提高指标”，因为连稳定提高主指标也尚未证明；目前最清晰的新认识是**许多看似独立的 RNA 结构测试在预训练来源层面并不独立**。这一认识有价值，但必须用时间外精度和结构分层结果完成证据闭环。

---

## 6. 模型架构问题

### 6.1 当前实际数据流

1. **输入表示**：`tools/rinalmo_preflight.py:55-58` 手工构造 `[CLS]+residues+[EOS]`；`load_encoder` 严格检查 checkpoint 键。RiNALMo-giga 为约 650M 参数、33 层、hidden=1280、20 heads、intermediate=5120。
2. **序列编码**：Plan-A 使用 RiNALMo 每残基 hidden states，而非旧自定义 RNAEncoder。`tools/train_plan_a.py:398-400` 默认 `warmup_head_steps=1600`，随后渐进解冻。
3. **pair 表示**：`src/rnajepa/decision_head.py:130-147` 构造 `sum/product/abs-diff` 的对称 pair feature；当前实现以该文件为准，不再用 README 中的简化描述代替。
4. **二维主干**：`FlatDecisionHead` 默认 `scorer=resnet2d`；`ResNet2DScorer` 用相对位置通道、64 hidden channels、卷积 residual blocks 和 `BatchNorm2d`（`decision_head.py:558-625`）。实际 pilot 配置另以 run manifest 为准，不能只凭默认 `resnet_blocks=4` 推断每次实验块数。
5. **先验与校准**：Turner residual、可学习 `prior_weight`、按长度桶的 temperature calibration 和合法 pair mask 在 `decision_head.py:413-450` 组合。
6. **PairTypeHead**：虽然被构建，但当前主 score/训练路径不消费类型输出，不能把它写成精度来源。
7. **结构辅助**：`src/rnajepa/dual_view.py` 将序列映射到 256 维，并从 canonical nested pairs 构造 helix/stacking token；固定 512 slots，learned-query cross-attention，双向 masked MSE，H2 加 target SIGReg。online/target 共用结构编码器，不是独立 3D 或复合物教师。
8. **训练目标**：`rlcd.py:374-436` 组合 CRF NLL、distill、RLCD-inspired Brier+ECE、calibration；NLL 默认 `sum`，辅助项是 per-pair 统计。`train_plan_a.py:1259-1311` 按 validation loss 选 best/早停。
9. **部署**：`evaluate_single_checkpoint.py:54-75` 禁止训练专用结构和 predictor 键/调用；`95-106` 先由序列计算 score/mask，再精确 `nussinov_map` 解码，最后才读真值计算指标。
10. **旧模型线**：`src/rnajepa/model.py` 是 mRNABERT student/EMA teacher、区域 descriptor、线性注意力 predictor 和正交因子，不应与当前 RiNALMo 双视图混写。

### 6.2 复杂度、长链与优化风险

- **pair 存储/卷积**：约 `O(L^2)` 内存与 `O(blocks × C^2 × k^2 × L^2)` 计算；完整结构头不能由编码器长链前向结果外推。
- **CRF NLL 与精确 nested DP**：通常为 `O(L^3)` 时间、`O(L^2)` 状态；block/chunk 只降低部分峰值显存，不改变渐近 DP 成本。
- **训练—推理差异**：训练有 dual-view、teacher 和辅助损失；部署只有 encoder/head。该隔离是优点，但训练专用目标只有在部署指标改善时才有价值。
- **目标尺度**：默认 NLL 按 sum 增长，辅助项按均值；不同长度下相对权重变化，可能造成长序列主导或梯度尺度不稳，必须做 length-normalization 消融。
- **BatchNorm**：二维头在可变长度和小 batch 上存在 train/eval 统计差异风险；已有状态恢复逻辑不代表该归纳偏置最适合 OOD。
- **checkpoint 选择**：validation loss 最优不等于 macro/stem/topology-F1 最优；不能用 test 选 checkpoint，应在开发集预注册选择规则并报告敏感性。
- **梯度证据边界**：已测单工程 batch 共享参数梯度 cosine=-0.0529394841，只能说明该 batch 存在弱负对齐，不能外推为普遍训练冲突。

### 6.3 必要性、冗余与上限

| 设计 | 判断 | 理由 |
|---|---|---|
| 严格 RiNALMo 权重加载、序列单模型身份、精确合法 mask/DP | 必要 | 保证任务定义和模型身份 |
| 同初始化 H0、shuffle/随机编码器/参数匹配对照 | 必要 | 是识别结构辅助真实收益的唯一可信方式 |
| ResNet2D 局部 pair 上下文 | 合理但证据不足 | 可学习 stem 局部连续性，但是否优于简单 MLP/官方头需同协议消融 |
| Turner 先验与校准 | 合理但混杂归因 | 有物理意义，也直接影响最终指标；需单独报告贡献 |
| PairTypeHead | 当前可能冗余 | 主路径不消费其输出 |
| 双视图固定 512 slots、共享结构编码器、target SIGReg | 缺乏有效支持 | H2 开发集为负，且表示只覆盖 canonical nested 结构 |
| 旧 Pair-JEPA、RIPE、通用复合物扩充 | 不应恢复为默认救援 | 已停止或无主榜收益，重新加入只会扩大搜索空间 |

### 6.4 训练目标与评价指标错位

存在：CRF likelihood、Brier/ECE 与最终 macro-F1/stem/topology 并非同一目标；validation loss 选 best 与 pilot macro-F1 终点也不同。精确 DP 是合法结构约束，不是数据泄漏，但它会掩盖 raw pair score 与最终结构之间的差异。论文必须同时报告 raw-pair、stem/topology 和解码后指标，禁止把全部提升归因于 JEPA 表征。

### 6.5 局部优化还是重做核心架构

**先不重做大架构。** 当前连“结构辅助有独立收益”和“真实 OOD 基线是多少”都未回答，直接替换核心架构无法判断成败。应先做 H0 头/损失/校准消融和时间外冻结评测；只有证据显示瓶颈在高层拓扑而非数据或优化，才进入下节两个最小架构方案。

---

## 7. 最不确定的三个问题

### 不确定性 1：观测到的性能究竟有多少来自预训练暴露，多少来自可迁移结构学习？

- **为何无法确认**：bpRNA-new 同源未检出 n=0；nt 历史快照不可追溯；时间外只有暴露统计，没有冻结模型精度。
- **影响**：直接决定“未见家族泛化”是否成立，也是最可能推翻整个项目主叙事的问题。
- **需要的证据**：时间外逐链 QA、训练/验证/预训练暴露 join、冻结逐序列预测、长度/家族匹配和配对区间。
- **最小实验**：不训练，完成时间外 QA 后运行 H0、H2、xens2（单列）、RiNALMo/BPfold/EternaFold 等冻结面板。
- **正面结果**：存在足够可评估的时间外链条，H2 或后续模型在预定主终点上稳定优于 H0/强基线。
- **负面结果**：没有足够独立样本，或提升只在暴露组/特定长度出现，或 H2 仍不优于 H0。
- **负结果处理**：放弃 clean-OOD 算法主张，转为污染感知基准与负结果；不再从旧测试集挑 7 条或少数未知样本包装结论。

### 不确定性 2：结构辅助是否提供独立于 RiNALMo、Turner 和训练头的信息？

- **为何无法确认**：H2 开发集为负；界面选定 step100 的 H0/H2 编码器逐张量相同；原生初始化两臂未收敛；缺少编码器确实发生独立更新的完整对照。
- **影响**：若回答为否，RNA-JEPA 的核心方法贡献不成立。
- **需要的证据**：训练前后逐张量哈希、有效 backbone 更新数、三种子 H0/H1/H2 与 shuffle/参数匹配对照、下游部署指标。
- **最小实验**：短 pilot 只用于工程检查；正式比较必须越过 warmup，证明 H0/H2 backbone 不再相同，并在相同 step/数据顺序下比较。
- **正面结果**：H2 在三种子开发集和时间外至少一个主终点优于 H0 与关键对照。
- **负面结果**：仅辅助损失下降，部署指标不升或下降。
- **负结果处理**：永久停止当前 dual-view 组合，不通过调权、换 micro-F1 或测试集选臂挽救。

### 不确定性 3：当前上限主要由 pair 表征、局部二维头、目标函数还是标签质量决定？

- **为何无法确认**：Turner、校准、CRF NLL、蒸馏、RLCD、ResNet2D 和 DP 同时存在，缺少同 checkpoint/同数据的最小分解；时间外标签尚未全量 QA。
- **影响**：决定应改训练目标、二维头、数据还是完全转向。
- **需要的证据**：NLL-only、length-normalized、no-Turner、no-calibration、MLP-vs-ResNet、raw-vs-DP、stem/topology 分解，以及分长度/来源误差。
- **最小实验**：同一 H0 初始化的小网格开发集消融，外部测试只运行一次冻结最佳配置。
- **正面结果**：某一因素在多个种子和长度桶一致成为瓶颈，可据此做单一架构修改。
- **负面结果**：各修改均无稳定收益，或提升只来自先验/后处理。
- **负结果处理**：停止“表征不足”叙事，优先数据/评估论文，不继续扩大 backbone。

**最可能推翻整个项目的是不确定性 1**：如果真实时间外/结构 OOD 上没有任何可重复收益，则即使在旧榜单通过模块组合提高数值，也无法支持原定的广泛价值。

---

## 8. 最大盲区

### 8.1 唯一最大盲区：独立性错觉（independence illusion）

**把训练—测试在家族、PDB 或 80% 簇上的隔离，误当作模型全部训练经历（包括预训练）与测试之间的独立；进一步把这种测试上的高分解释为未见家族泛化。**

### 8.2 仓库现象

- bpRNA-new 的 5388/5388 条均在已知预训练来源检出同源；精确未检出仅 7 条且仍未知。
- 界面 holdout 的 PDB/group 隔离为 0 重叠，但仍有 111/198 条已知同源暴露。
- 时间外 1131 条唯一查询中，合并精确或同源已知暴露 680，未检出 451；未检出仍不是 clean。
- 历史表格混合 xens2、单 checkpoint 和不同 clean 版本，容易把插值、集成和外部结果放在同一叙事中。

### 8.3 为什么容易忽视

团队通常只控制当前监督训练 split；预训练语料规模大、历史快照难追溯，且“未见家族”在字面训练标签层面看似成立。工程上已运行的榜单和漂亮图表又会形成沉没成本，使人倾向保留旧结论。

### 8.4 如何造成虚假性能或错误结论

模型可能复用预训练阶段见过的相同/近缘序列或家族信息，使分数反映记忆与插值，而非结构迁移。若只报告全体 micro-F1，长序列和高暴露家族还可能主导结果；若在少量未检出样本上挑表现好的模型，会产生选择偏差。

### 8.5 最小成本验证

无需先训练：完成时间外全量 QA，将已有冻结预测与暴露标签 join，报告 exact/homology/not-detected、长度、家族和拓扑分层。对配对样本使用序列/簇 bootstrap 和 Holm 校正；空组或小样本只报告描述，不做因果推断。

### 8.6 四层解决方案

- **模型**：H0/H2/集成身份分离；任何结构辅助必须证明部署时无标签调用。
- **数据**：时间切分 + 同源审计 + 结构拓扑去重；`UNKNOWN_NOT_CLEAN` 作为正式状态。
- **实验**：主终点在时间外/结构 OOD；旧榜单仅作敏感性；统一基线和模型身份。
- **论文**：把“未见 fine-tuning 家族”与“预训练未暴露”分开；不再使用无限定的 clean-OOD。

### 8.7 盲区消除判据

每条纳入主终点的序列都有版本化标签、时间、来源、暴露状态和模型预测；主结论在时间外与结构 OOD 上方向一致并有置信区间；所有未检出均写未知；单模型与集成、开发集与外部测试分栏。只有这些全部满足，才可认为该盲区已被控制。

### 8.8 次要盲区（最多三个）

1. **后处理归因盲区**：Turner、校准、集成和 DP 的贡献被写成表征学习收益。
2. **训练身份盲区**：warmup 导致 H0/H2 编码器别名、validation loss 与科学终点不一致、step cap 被误写为收敛。
3. **复现盲区**：未跟踪脚本、重复测试模块、退役测试失败和 cu128 环境依赖使“同一代码”并不明确。

---

## 9. 推荐解决方案

### 9.1 先执行的非架构方案

1. 冻结当前代码和 artifact 身份，整理现行/退役测试；禁止新训练覆盖历史目录。
2. 完成时间外全量 QA、统一指标和冻结基线面板；先得到可信 OOD 数字。
3. 在开发集做 H0 最小消融：NLL-only、length normalization、Turner、calibration、MLP/ResNet、raw/DP、stem/topology。
4. 只有在某一瓶颈被重复定位后，才从下面最多两个架构候选中选择一个进入正式三种子。

### 9.2 架构方案 A：pair→stem 对齐的轻量目标与头部改造（优先）

- **要解决的瓶颈**：最终错误可能集中在 helix 连续性和高层 wiring，而当前辅助损失主要优化独立 pair 概率或共享结构 latent。
- **架构修改**：保留 RiNALMo 和 nested DP；在现有 score matrix 上增加可微 stem 连续性/局部配对一致性目标，配合 length-normalized NLL；不新增大型 backbone。
- **理论/经验依据**：CHANRG 显示 pair-F1 与 topology 泛化可分离；当前局部 ResNet 与 pair Brier 不直接优化 stem/topology。
- **预计收益**：若瓶颈确为 stem 连续性，预期改善 stem-F1/topology，并间接改善 macro-F1；不预设必然提高 micro-F1。
- **主要风险**：过度奖励长 helix、降低非典型 loop/短 stem 召回；length normalization 改变长短序列权重；新损失引入另一组超参数。
- **最小实现版本**：一个 stem-consistency 辅助项 + 一个 length normalization 开关，复用现有 pair score，不加新网络。
- **对照实验**：同初始化 H0；NLL-only；仅 length norm；仅 stem loss；二者同时；相同 DP 与校准。
- **消融实验**：raw pair、解码后 pair、stem、topology、长度桶、Turner on/off。
- **成功标准**：三种子开发集 stem-F1 和 macro-F1 均相对 H0 提升至少 0.01，且时间外不下降超过 0.005； paired 95% CI 排除 0 后才进入 SOTA 面板。
- **失败处理**：若只改善 stem 而 pair/macro 明显下降，或时间外无收益，停止该辅助项，不调 test 权重。

### 9.3 架构方案 B：小规模全局 pair/axial-recycling 编码器（仅在 A 证明拓扑瓶颈后）

- **要解决的瓶颈**：局部 3×3 ResNet 即使有多个 block，也主要建模局部 pair 邻域；精确 DP 只保证 nested 合法性，不提供学习的全局拓扑表示。
- **架构修改**：将部分局部 ResNet block 替换为 1–2 个 pair-axial attention/recycling block，在行列方向传播长程 pair 信息；保留相同输入、先验和 DP。
- **理论/经验依据**：RNAformer/eFold 类二维建模提供经验依据；当前项目尚未证明其局部头足以处理结构 OOD 的高层 wiring。
- **预计收益**：目标是改善 GenA/时间外的 stem/topology，而非追求常规插值分数。
- **主要风险**：`O(L^2)` pair 激活与额外 attention 显存、优化不稳、参数/计算不公平、小数据过拟合；长链问题可能恶化。
- **最小实现版本**：单个 axial block、无大规模预训练、相同训练步数和接近的训练计算预算；只在限定长度 pilot。
- **对照实验**：H0、ResNet-only、参数量匹配 ResNet、计算量匹配 ResNet、axial 无 recycling。
- **消融实验**：行列注意力、recycling、相对位置、Turner、长度桶和拓扑分层。
- **成功标准**：开发集三种子 topology/stem-F1 提升至少 0.01，随后时间外至少一个共同主终点显著优于 H0；显存和覆盖达到预注册上限。
- **失败处理**：若收益只在暴露插值、计算不可接受或时间外无提升，删除该分支并回到证据/基准论文。

**不建议第三方案：** 当前不应再扩大 RiNALMo、恢复旧 Pair-JEPA/RIPE、引入通用复合物扩充或增加新投票器，因为这些方向没有先解决已识别的证据瓶颈。

---

## 10. 重构后的论文故事

### 10.1 逻辑链

**领域痛点**：RNA 二级结构模型需要在真正新家族和新时间上可靠工作。  
→ **现有关键局限**：常规 split 忽略预训练暴露和结构拓扑冗余，pair-F1 也不能代表高层结构迁移。  
→ **项目切入点**：建立污染感知、时间外、结构分层的单模型评估，并检验训练期结构辅助是否产生部署收益。  
→ **核心方法创新**：正向版本是经验证的轻量 pair→stem/全局 pair 设计；负向版本是可复现的暴露审计 + 严格对照 + 时间外评估框架。  
→ **为什么能解决局限**：模型身份、数据独立性和结构指标同时受控，避免把记忆、集成或后处理误认为迁移。  
→ **实验验证**：三种子 H0/H2/关键对照、统一强基线、时间外、GenA/结构 OOD、长度分层、raw/stem/topology、消融和区间。  
→ **新能力或认识**：要么获得真实 OOD 提升，要么证明当前结构辅助和常规榜单为何不能支持泛化结论。  
→ **广泛价值**：为 RNA foundation model 的结构迁移评估提供可复用协议，而不是只增加一个 benchmark 数字。

当前逻辑链的断裂点在“核心方法创新→部署收益→真实 OOD”两处；时间外 QA 和冻结评测是修复第一段逻辑的最短路径。

### 10.2 核心主张与次级主张

| 主张 | 需要证据 | 当前已有 | 当前缺失 | 禁止过度表述 |
|---|---|---|---|---|
| **核心（二选一）**：结构辅助在严格 OOD 上有效；或常规结构辅助/榜单不能证明 OOD | 预注册主终点、三种子、时间外、统一基线、消融、CI | H0–H6 开发负结果、暴露审计 | 时间外精度、正式三种子、公平基线 | 不得把单 seed dev 负结果全称否定所有结构学习 |
| 次级 1：预训练暴露会改变泛化解释 | 暴露分层预测、匹配分析、时间外敏感性 | 多集合暴露计数、bpRNA-new 无同源对照 | 足够独立样本的精度效应 | 不得把暴露直接写成因果性能增益 |
| 次级 2：pair 分数与高层结构迁移不同 | stem/topology/GED、GenA/长度分层 | 当前只有 pair 指标和文献依据 | 本项目结构指标 | 不得仅凭 CHANRG 文献替代本项目实验 |
| 次级 3：严格标签/身份 QA 可纠正伪结论 | 标签映射、撤回记录、冻结身份与独立重算 | 界面标签 QA、xens2 身份重放、测试审计 | 时间外全量 QA、公开复现包 | 不得把工程 QA 写成生物机制发现 |

### 10.3 两种可提交预印本的成稿条件

- **算法版**：核心方法在时间外/结构 OOD 有正向、可重复、公平的收益，标题和摘要以该方法为主。
- **基准/负结果版**：若方法门失败，则以“预训练暴露与结构划分重新评估 RNA 二级结构泛化”为主，双视图作为一个预注册负案例；不得把负结果藏到附录后仍以 SOTA 为标题。

---

## 11. Final Goal

### Final Goal

- **核心科学问题**：在控制 RiNALMo 预训练暴露、时间、序列同源和结构拓扑后，训练期结构辅助能否提升序列单模型的真实 OOD 二级结构预测？
- **核心假设**：只有当结构监督提供独立且与 pair/stem/topology 目标对齐的信息时，部署指标才会改善；常规插值分数不能验证该假设。
- **核心方法贡献**：优先交付污染感知时间外/结构 OOD 评估协议；算法贡献只保留经 H0 和强基线证明有效的轻量结构目标或二维头。
- **目标任务**：推理仅序列的 canonical nested RNA 二级结构预测；不宣称 pseudoknot、非 canonical 或 3D 能力。
- **主要数据集**：TR1c/VL0 开发；TS0/bpRNA-new/ArchiveII/Rfam 污染感知敏感性；时间外 PDB/DSSR 主测试；CHANRG 作为可选外部压力测试。
- **关键评价指标**：macro/micro pair-F1、stem-F1、topology-F1/GED、长度/家族/暴露分层、覆盖率、配对 95% CI；ECE/AUPRC 为辅助。
- **必须超过的基线**：首先同骨干 H0；随后统一协议的 RiNALMo 官方头、BPfold、EternaFold/CONTRAfold、MXfold2、UFold，尽可能加入 ERNIE-RNA/SPOT-RNA；xens2 仅作集成参照。
- **必须完成的泛化验证**：时间外冻结测试、至少一种结构拓扑/家族 OOD、长度分层、暴露敏感性；未检出只能写未知。
- **必须完成的机制或解释性分析**：raw pair→stem→DP 分解、Turner/校准/头/损失消融、编码器实际更新证明、长度与失败家族分析；不以 t-SNE 或单 batch 梯度代替机制。
- **论文级最终交付物**：完整预印本源稿和 PDF、claim–evidence 矩阵、统一结果表、逐序列预测、数据/模型/环境 manifest、三种子训练台账、消融和泛化图、复现脚本与限制说明。
- **项目成功标准**：算法版要求三种子开发集相对 H0 的 macro-F1 与 stem-F1 均至少 +0.01，并在时间外至少一个预注册共同主终点显著优于 H0且不劣于最强可比基线；SOTA 字样还要求在统一协议的全部主指标上超过最强可比公开方法。若未达到但证据链完整，基准版成功标准是时间外 QA、冻结面板、统一指标、受控负结果和复现包全部闭环。
- **项目终止或转向条件**：时间外没有足够合格样本；H2 正式三种子不优于 H0；两个候选架构均在时间外无收益；收益只在暴露组或由集成/后处理解释；无法核验基线身份。触发后停止算法 SOTA 路线，转基准/负结果，不继续加模块。

---

## 12. 分阶段 TODO

时间安排按 D1–D30 相对日执行；文档工作与新实验并行，但任何训练不得绕过数据和代码门。

### Phase 0：仓库、数据和实验真实性审计（D1–D3）

| 字段 | 内容 |
|---|---|
| 阶段目标 | 确定实际代码、可用数据、历史结果身份和测试基线 |
| 任务 | 冻结源码/未跟踪脚本哈希；分类现行与退役测试；重算时间外三件套、链数、映射和日期；核对 xens2/H0/H2 checkpoint |
| 涉及模块 | `tools/train_plan_a*.py`、`src/rnajepa/`、`tests/`、`tools/test_*`、`artifacts/v2/temporal_structure/`、历史 manifests |
| 前置依赖 | 当前服务器和本地交接副本只读访问 |
| 输出物 | `manifests/round57/code_inventory.json`、测试分层报告、时间外全量 QA、历史身份表 |
| 验收标准 | 每个当前 run 可绑定源码+配置+权重+数据哈希；现行测试独立通过；1805 条目均有纳入/排除原因；三次独立计数一致 |
| 风险 | 未跟踪代码改变结果；文件存在但标签无效；历史 checkpoint 错配 |
| 失败处理 | 身份不明结果降为历史参考，不重训补造；数据不合格则缩小主测试并报告覆盖 |
| 优先级 | P0 |

### Phase 1：可靠基线与统一评估协议（D2–D7）

| 字段 | 内容 |
|---|---|
| 阶段目标 | 在同一序列、同一指标、同一模型身份下回答“当前到底多少分” |
| 任务 | 预注册 macro/micro/stem/topology/长度/暴露指标；独立实现复核；冻结 H0/H2/xens2 和公开基线预测；记录预训练与计算预算 |
| 涉及模块 | `eval/ss/metrics.py`、`tools/evaluate_single_checkpoint.py`、`tools/xens2_eval.py`、新 `tools/evaluate_stem_topology.py` |
| 前置依赖 | Phase 0 数据和身份门 |
| 输出物 | 协议 YAML、指标单元测试、统一预测面板、基线公平性表 |
| 验收标准 | 合成指标测试与独立实现逐行一致；所有模型覆盖和失败原因公开；无 test 阈值选择 |
| 风险 | 基线数据结构定义不同；checkpoint/版本缺失；指标实现错误 |
| 失败处理 | 不可比结果单列，不进入 SOTA；缺身份则标 blocked |
| 优先级 | P0 |

### Phase 2：验证核心科学假设（D5–D12）

| 字段 | 内容 |
|---|---|
| 阶段目标 | 判断结构辅助是否有独立部署收益，以及 H0 的真实瓶颈 |
| 任务 | 时间外冻结 H0/H2；最小 H0 头/损失消融；必要时正式三种子 H0/H1/H2，证明 backbone 实际更新 |
| 涉及模块 | `tools/train_plan_a.py`、`src/rnajepa/dual_view.py`、`decision_head.py`、`rlcd.py` |
| 前置依赖 | Phase 1 协议和开发集冻结；训练前工程 smoke |
| 输出物 | 三种子 run manifests、逐层哈希、训练曲线、开发/时间外结果、go/no-go 决策 |
| 验收标准 | 同数据顺序/初始化/主损失；无 test 选臂；H0/H2 不再别名；三次独立指标重算一致 |
| 风险 | warmup 混淆、step cap、loss-F1 错位、负梯度、标签异质 |
| 失败处理 | H2 不优于 H0 则停止当前 dual-view；只允许一个由消融支持的架构候选进入 Phase 3 |
| 优先级 | P0 |

### Phase 3：模型架构迭代（D10–D20）

| 字段 | 内容 |
|---|---|
| 阶段目标 | 验证一个明确针对瓶颈的架构修改，而非模块堆叠 |
| 任务 | 优先 pair→stem 轻量方案；只有证明全局拓扑瓶颈才做单个 axial/recycling block |
| 涉及模块 | `decision_head.py`、`rlcd.py`、必要的新 stem loss/axial 模块及测试 |
| 前置依赖 | Phase 2 正交消融和 go 决策 |
| 输出物 | 一个候选分支、三种子开发结果、复杂度报告、失败记录 |
| 验收标准 | 达到 §9 的 +0.01 开发和时间外保护门；计算/显存/覆盖公开 |
| 风险 | 过拟合、显存、长短序列权衡、计算不公平 |
| 失败处理 | 立即删除候选主线资格，保留只读 artifact；转基准论文 |
| 优先级 | P1 |

### Phase 4：SOTA 对比、消融与泛化（D17–D25）

| 字段 | 内容 |
|---|---|
| 阶段目标 | 判断是否有资格使用“领先/SOTA”，并定位收益来源 |
| 任务 | 统一强基线、单模型/集成分栏、时间外/结构 OOD、长度和暴露分层、完整消融、配对 bootstrap/Holm |
| 涉及模块 | `eval/`、`tools/evaluate_*`、结果汇总与统计脚本 |
| 前置依赖 | Phase 3 成功候选或 Phase 2 最终 H0 证据面板 |
| 输出物 | 主结果表、OOD 表、消融表、置信区间、失败家族案例 |
| 验收标准 | 所有主表数字可由冻结预测一键重算；模型身份/数据/计算预算完整；无选择性报告 |
| 风险 | 多重比较、样本量不足、基线不公平 |
| 失败处理 | 不写 SOTA；改报效应量和限制，保留负结果 |
| 优先级 | P1（若冲算法则 P0） |

### Phase 5：机制分析、科学发现与论文叙事（D21–D27）

| 字段 | 内容 |
|---|---|
| 阶段目标 | 回答为什么有效或为什么无效，形成单一论文逻辑链 |
| 任务 | raw/stem/topology 分解、长度和失败家族、更新/梯度诊断、暴露敏感性、按两种成稿条件选择算法版或基准版 |
| 涉及模块 | 分析 notebooks/scripts、`paper/rna_evidence_v2_*`、新终审报告 |
| 前置依赖 | Phase 4 主表冻结 |
| 输出物 | 论文图、机制分析、claim–evidence 矩阵、摘要与标题 |
| 验收标准 | 每句核心主张都有直接证据；限制与负结果进入正文；无 t-SNE/proxy 代替机制 |
| 风险 | 过度解释、把相关当因果 |
| 失败处理 | 降低主张等级，不新增无依据叙事 |
| 优先级 | P1 |

### Phase 6：复现、代码整理与投稿准备（D25–D30）

| 字段 | 内容 |
|---|---|
| 阶段目标 | 第三方可按 manifest 重建全部主表和图 |
| 任务 | 清理当前测试分层、固化环境/许可、最小复现包、三次独立重建、预印本 PDF 文本与视觉 QA |
| 涉及模块 | 当前代码、配置、manifests、paper、复现脚本 |
| 前置依赖 | 论文结果冻结 |
| 输出物 | 预印本源稿/PDF、代码包、数据卡、模型卡、复现报告、最终交接 |
| 验收标准 | 干净目录主表/图重建一致；当前测试全绿、退役测试明确隔离；所有文件哈希和外部链接有效 |
| 风险 | 外部权重/数据许可、未跟踪依赖、表格或 PDF 错误 |
| 失败处理 | 不发布无法合法分发或无法重建的内容；提供获取脚本和精确版本 |
| 优先级 | P1 |

---

## 13. P0 立即执行清单

### 13.1 推荐串并行顺序

**串行主干：** `T00 代码身份 → T01 测试门 → T02 时间外 QA → T03 协议/指标 → T05 冻结时间外面板 → T06 假设裁决`。  
**可并行：** T00 后可并行 T01 与 T02；T03 的指标实现与 T04 基线环境准备可并行；论文框架整理可在 T05 后进行，但不能先写结论。  
**禁止并行：** 在 T02/T03 未通过前启动时间外调参；在 T06 未裁决前启动多个新架构；用外部 test 参与 checkpoint/阈值选择。

### 13.2 可直接交给代码智能体的任务

| ID/顺序 | 输入 | 动作与新增/修改文件 | 输出 | 必须测试/实验 | 可信判据 |
|---|---|---|---|---|---|
| **T00 / 1** | 远端 HEAD、未跟踪脚本、当前 checkpoints | 生成 `manifests/round57/code_inventory.json`；记录所有 `tools/train_plan_a*.py`、部署、双视图、界面和数据脚本哈希；不删除用户文件 | 唯一代码清单和 run→source 映射 | 重算哈希两次；抽查 3 个历史 run | 第三次独立脚本得到相同清单；身份不全者标 `LEGACY_NOT_REPRODUCIBLE` |
| **T01 / 2A** | `/tmp/round57_pytest_*`、当前 `tests/` | 修改 pytest import 配置/测试分层；新增 `manifests/round57/test_taxonomy.json`；退役 Pair-JEPA/RIPE/RNAformer 单独标记，不静默跳过当前失败 | 当前套件与 legacy 套件报告 | 当前套件、全仓收集、51 项定向测试分别运行 | 当前套件 0 收集错误/0 失败；legacy 失败有 owner 和原因；不能写全仓绿色除非真的全绿 |
| **T02 / 2B** | 1805 个时间外目录、原始 mmCIF/DSSR/parents、日期元数据 | 新增 `tools/verify_temporal_full_labels.py`、`tests/test_temporal_full_labels.py`；输出 `artifacts/v2/temporal_structure/full_qa_v1.json` 和序列 manifest | 可评测链、排除原因、日期/来源/标签哈希 | 三次独立计数；抽样回查原始 mmCIF；modified/unobserved/empty-chain 单测 | 1805 条目全部有状态；1803 三件套不自动等于纳入；空链/映射失败显式排除；QA 状态通过后才解冻评测 |
| **T03 / 3** | 冻结预测格式、`PairLevelMetrics` | 新增 `configs/eval/round57_protocol.yaml`、`tools/evaluate_stem_topology.py`、`tests/test_round57_metrics.py`；独立第二实现仅用于 QA | pair/stem/topology/长度/暴露统一指标 | 空结构、单 stem、pseudoknot 拒绝/分栏、对称性、padding、batch 不变性；双实现逐行比较 | 所有合成测试通过；真实预测计数与既有 micro/macro 完全一致；新指标独立重算一致 |
| **T04 / 3A（并行）** | 官方 RiNALMo、BPfold、EternaFold、MXfold2、UFold、ERNIE-RNA 可用权重/环境 | 新建 `manifests/round57/baseline_fairness.json` 和运行配置；记录参数、预训练、训练集、外部信息、版本、许可 | 可执行基线面板和不可比项清单 | 官方 smoke；相同样本覆盖；失败不填零 | 版本和命令可重放；不同训练数据/集成明确分栏；无法复现者不进 SOTA 主表 |
| **T05 / 4** | T02 合格链、T03 指标、H0/H2/xens2/T04 基线 | 新增 `tools/run_round57_frozen_panel.py`；只运行冻结推理，不训练；输出逐序列预测和 `frozen_metrics.json` | 第一版可信时间外结果 | 每模型身份/覆盖/重复前向；指标独立重算；空预测和失败链公开 | 三次汇总哈希一致；xens2 与单模型分栏；not-detected=未知；无阈值/ checkpoint 调优 |
| **T06 / 5** | T05 逐序列结果 | 新增 `tools/summarize_round57_paired.py`；按序列/簇 bootstrap、长度匹配、Holm；输出 go/no-go | 核心假设裁决报告 |  paired effect、CI、样本量、敏感性；禁止空组显著性 | 结论由预注册 macro/stem 主终点决定；小样本只描述；裁决写入训练台账 |
| **T07 / 6A（仅 T06 允许）** | 开发集、H0/H1/H2、最小消融配置 | 修改训练配置和日志，不改测试；证明 backbone 更新；运行三种子或停止 | 受控训练结果 | 同初始化/数据顺序、step cap 状态、逐张量哈希、三次指标复核 | H2 达标才保留；否则 `STOP_DUAL_VIEW_FORMAL`，不得换指标救援 |
| **T08 / 6B（并行写作）** | T00–T06 证据 | 更新证据稿、claim matrix、spec/tasks/checklist 和训练台账 | 可审查论文骨架 | 数字回链、链接检查、PDF 文本/视觉复核 | 所有主张有 artifact；所有限制保留；未生成结果不得写入 |

### 13.3 每次训练的强制留痕

每个 run 必须同时保存：命令、代码/依赖/权重/数据 SHA256、seed、初始化和 resume 来源、batch 顺序、可训练参数、实际 backbone 更新、逐步损失、学习率、梯度范数、显存、checkpoint 选择规则、停止原因、预测、指标、失败和结论。每次实验至少经历：

1. 运行者自检：配置、日志、artifact 完整；
2. 独立脚本重算：预测计数、指标、哈希；
3. 交接终审：科学边界、与历史身份一致性、是否支持主张。

---

## 14. 项目转向或终止条件

满足任一条件即触发，不允许通过继续加模块绕过：

1. **数据终止条件**：时间外全量 QA 后仍没有足够独立链条支撑预注册分析，或关键日期/标签映射无法追溯。转向小规模描述性技术报告，不做 OOD 精度结论。
2. **核心假设终止条件**：正式 H2 三种子不优于 H0，或提升不能超过 shuffle/参数匹配对照。永久停止当前 dual-view 精度路线。
3. **架构终止条件**：§9 两个候选分别在预定开发/时间外门失败；不再启动第三个救援架构，转基准/负结果。
4. **泛化终止条件**：收益只出现在预训练暴露或插值组，时间外/结构 OOD 无收益。不得使用“未见家族 SOTA”。
5. **归因终止条件**：收益主要由 R2D 集成、Turner、校准、test 选择或后处理解释。改写为工程集成结果，删除表征创新主张。
6. **公平性终止条件**：关键公开基线无法核验版本、数据或推理覆盖。可以报告本项目结果，但不得宣称 SOTA。
7. **复现终止条件**：当前主表无法在干净目录由冻结预测和 manifest 重建。暂停投稿，先修复现，不补写结果。
8. **范围终止条件**：若必须依赖 pseudoknot、非 canonical、3D 或实验结构才能讲清故事，而当前部署不支持，则重新定义任务，不把旧结果外推到这些能力。

---

## 最终直接回答

**当前最应该先完成“时间外全量 QA + 统一 pair/stem/topology 指标 + H0/H2/强基线的冻结评测”，因为它直接决定项目是否还有真实 OOD 科学问题、当前差距在哪里以及结构辅助是否值得继续；在这一答案出来前增加模型模块，只会把预训练暴露、训练目标、先验/DP 和模型身份进一步混杂，既无法证明 SOTA，也更难形成可信预印本。**
