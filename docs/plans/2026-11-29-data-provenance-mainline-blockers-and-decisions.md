# FoldNAR 数据谱系问题、主线正结果阻塞与后续决策

日期：2026-11-29  
状态：决策 1 继续执行；决策 2/3/4 已确认。当前不新增未经四问审查的 idea。

> **2026-11-29 用户统一口径更新**：RiNALMo 等无监督预训练暴露不再作为主线阻塞、排名否决或 clean-OOD gating。后续测评做好下游训练/评价去重、标签协议一致、覆盖率完整后正常排序。历史预训练暴露审计保留为记录，不用于 veto。权威政策见 `2026-11-29-evaluation-policy-user-override.md`。下文涉及“预训练暴露导致不能排名”的旧表述均按此政策降级为历史审计背景。

## 1. 核心结论

当前最大的问题不是实验数量不足，而是：

1. 还没有一个被结构终点、机制对照、泛化和外部协议共同支持的正向机制；
2. 旧基准标签存在预测派生/循环评价问题，修复后绝对性能差距更大；
3. 标签来源、覆盖率、下游泄漏和协议混用曾削弱 SOTA 叙事；无监督预训练暴露按用户最新决策不再作为排名否决项；
4. 工程中断和严格验收门进一步拉长了从候选到可交付正结果的路径。

因此，当前不能把局部 control 提升、raw 排序提升、validation loss 下降或旧投影协议高分写成主线正结果。

## 2. 数据上到底存在什么问题？

### 2.1 标签循环：用模型预测当“真值”，再评价同一个模型

已证实：

- bpRNA-new 旧项目 5,388 条标签来自 BPfold 官方 release 预测 BPSEQ 的过滤投影；
- ArchiveII clean 2,544 条旧标签也来自 BPfold 预测投影；
- 5,282/5,388 条 bpRNA-new 项目标签与官方参考标签不同；
- 官方 BPfold release 对项目派生标签得到 micro-F1 `0.9616708875`，这是 release→派生标签的循环一致性，不是独立精度。

含义：

> 如果“标准答案”本身来自某个模型的预测，那么该模型在这个答案上得高分，不能证明它预测真实 RNA 结构更准确。

处理：

- 旧派生标签全部标为 `DERIVED_FROM_BPFOLD_PREDICTIONS`；
- 旧 raw/clean 派生结果降为 `circular-audit-only`；
- 保留历史数值，但禁止作为独立 SOTA 证据；
- 使用官方参考标签重建面板。

### 2.2 投影协议：过滤后的“clean”不等于独立实验真值

项目旧面板对 min-loop、crossing 等做了投影过滤。过滤可以定义一个合法任务协议，但不能冒充未过滤官方协议或实验真值。

必须分栏：

- `projected-reference`：明确声明过滤投影协议；
- `official-reference`：未过滤官方参考；
- `raw-exposure descriptive`：存在训练/预训练暴露的描述性结果；
- `clean-OOD`：只有在完整暴露、同源、时间和标签谱系闭合后才可使用。

当前 TS0/TestSetB 的 projected 子集可以描述性比较，但官方全集覆盖不足时不能排名。

### 2.3 预训练暴露：历史审计背景，当前不作为排名否决项

当前候选 backbone 为 RiNALMo-giga。审计显示：

- TR1c cluster-dev：`EXACT=7,978 / 9,143`，`HOMOLOGY=539`，`NOT_DETECTED_UNKNOWN=626`；
- bpRNA-new：`EXACT=5,381 / 5,388`，其余 7 条为 HOMOLOGY；
- 完整官方预训练 manifest 仍 `UNCHECKED`。

含义：

> 即使下游 train/dev 按 family 或 80% cluster 严格划分，backbone 仍可能在预训练阶段见过评价序列。下游 cluster-held-out 不能消除预训练记忆。

当前用户决策：

- 无监督预训练暴露不再作为排名否决或主线阻塞；
- 测评做好下游训练/评价去重、标签协议一致和覆盖率完整后正常排序；
- EXACT/HOMOLOGY/UNKNOWN 可作为历史审计或补充描述，但不用于 veto；
- 不再要求完整官方预训练 manifest 闭合后才允许正常 benchmark 排名。

### 2.4 标签来源：数据库/比较推断不能统称实验真值

逐记录质量审计：

- TR1c 42,564 条标为 `DATABASE`；
- CHANRG 105,721 条标为 `COMPARATIVE_INFERRED`；
- 95,052 条含 noncanonical pair、10,250 条含 short-span pair，属于软复核，不自动删除；
- 硬冲突 0，实际删除 0；
- modified-parent、逐记录实验谱系、PDB/mmCIF 回查仍大量 `UNCHECKED`。

含义：

> 数据可用不等于标签是实验真值。论文必须说明标签来源和适用协议，不能把数据库推断结构写成实验测量结构。

### 2.5 PDB/Rfam：身份、观测残基和时间新颖性未闭合

PDB：

- 153 个原始候选中，62 条恢复唯一 PDB/entity/chain 并生成双协议参考；
- 91 条仍 `UNCHECKED`；
- 未观测残基只能 mask，不能生成负标签；
- TS-hard 与其他面板存在精确序列重叠，不能当独立难度切片。

Rfam：

- temporal 436 条中，269 条 `NOVEL_SEQUENCE_RELEASE_SUPPORTED`；
- 43 条旧序列、12 条 TR1c 排除、2 条跨面板排除、110 条 `UNCHECKED`；
- family first-release 不等于 sequence first-release。

含义：

> 不能把整个 PDB/Rfam 面板写成独立实验结构或时间外泛化；只能报告已证明的子集，并保留未知层。

### 2.6 覆盖率偏差：缺预测不能补 0，部分覆盖不能排名

例子：

- ArchiveII xens2 原 clean 身份只有 2,544/3,966；
- 新 GPU recovery composite 为 3,950/3,966，仍有 16 条协议阻断；
- 因此 xens2 不能参与 ArchiveII full3966 排名，也不能计算 full 差距；
- MaxExpect 已补齐 1,422/1,422，成为第 9 个 full3966 身份。

原则：

> 同序列、同标签、同覆盖、同指标，才允许排名。缺失预测不是 0 分，部分覆盖不是全集结果。

## 3. 为什么一直无法取得主线正结果？

### 3.1 科学机制尚未闭环

已有证据链：

1. 容量扩张三种子未复现，关闭；
2. SRD 同分数重解码未超过当前 MEA，关闭；
3. 局部 stacking closure macro/stem 不保持，关闭；
4. GEC seed0 真实上下文损害 isolated/long-span recall，科学 FAIL；
5. crossover 证明上下文输入有因果作用，但真实 ensemble 不是一致最优；
6. 残差审计显示 98.31% FN 为非正 score margin；
7. structured margin 是待检验候选，目前四臂未全部完成。

所以目前还没有“方法有效”的直接结构证据。

### 3.2 旧基准修复后，绝对差距变大

official-reference bpRNA-new：

- xens2 micro-F1 `0.4890166090`；
- BPfold replay `0.7745179151`，差距 `0.2855013061`；
- CONTRAfold `0.6296755882`，差距 `0.1406589792`。

ArchiveII official clean2544：

- xens2 `0.8156522070`；
- BPfold replay `0.8999331355`；
- RiNALMo-ft `0.8204522395`，paired macro CI 跨 0。

这说明局部 `+0.004` 量级 control 提升不能回答核心性能缺口。

### 3.3 严格门防止了“伪正结果”

当前合同禁止：

- 用 raw AUPRC 抵消结构失败；
- 用单 seed/单 family/单序列驱动结论；
- 用 destroy 较好包装成随机有益；
- 用旧投影标签高分冒充 SOTA；
- 用部分覆盖结果冒充全集结果；
- 科学 FAIL 后调门救援。

这些门让交付变慢，但避免了不可投稿的正结果。

### 3.4 工程与资源问题消耗时间

- `/teams` 配额两次导致 scheduler 退出；
- 活跃 checkpoint 被迫暂存 `/home`；
- 每 500 步完整 9,143 dev 验证约 38–40 分钟；
- 四臂受显存和配额限制串行/受限并行；
- 子任务服务多次 token/connection 失败；
- 官方权重下载和旧资产恢复存在网络/许可阻塞。

这些是执行效率问题，不是科学正结果本身。

## 4. 四问审查结论

1. **失效点**：局部通过。当前候选在 isolated/long-span truth pair 被非正 score margin 压掉的环境中有直接失败证据；但尚未证明领域普遍无解。
2. **问题独立性**：通过。序列表示如何形成全结构竞争一致的非局部配对，是独立于 GEC/margin 的科学问题。
3. **机制解释**：未完全通过。margin 动机合理，但 98.31% 是错误分类描述，不是因果证明；必须等四臂结构结果。
4. **可预测性**：已形式化，未验证。预测收益应集中在非正 margin 的 isolated/long-span/高竞争环境，而不是 MEA 竞争排斥或标签/暴露问题。

结论：当前只继续已冻结的 margin falsification，不新增 idea。

## 5. 已确认决策

### 决策 1：继续当前 margin 四臂

执行：

- matched/ensemble 已自然收敛；
- destroy/independent 继续至各自收敛；
- 完成后统一 prediction export、独立结构评估、暴露分层；
- 不调 decoder temperature/gamma/threshold；
- 不启动 seeds1/2，除非 seed0 通过。

### 决策 2：若 margin FAIL，回到错误分解提出不同数学机制

禁止：

- 继续堆局部模块；
- 调门救援；
- 把负结果包装成最终交付。

候选必须重新通过四问，并由新的直接实验现象支持。

### 决策 3：若 margin PASS，正结果门槛

必须依次满足：

1. seed0 全部结构门通过；
2. destroy/independent 对照排除普通精修/随机扰动；
3. 收益集中在预注册目标子群，且非单 family 驱动；
4. seeds1/2 复现；
5. 多环境开发面板确认；
6. 冻结后外部 official-reference 评价；
7. 与同协议最强 baseline 的绝对差距明确缩小。

只超过 matched control 不够。

### 决策 4：论文当前可写与不可写

可写：

- 局部 pair 表示与全结构决策之间的可测量缺口；
- GEC 失败与上下文因果敏感性；
- score-margin FN 错误分解；
- 标签谱系、暴露和覆盖率审计方法。

不可写：

- GEC 有效；
- margin 有效；
- clean-OOD SOTA；
- 完整八主 benchmark 超越；
- 长序列 GEC 支持。

## 6. 后续执行方案

### 阶段 A：完成当前裁决

- destroy/independent 训练至自然收敛；
- 四臂 best/stop、张量、ledger 独立审计；
- 固定 MEA prediction export；
- 16 门 + paired CI + family 集中度 + 暴露分层；
- margin vs qualified NLL 同 MEA 结构比较；
- 真实结果写入第 34–37 页并推送。

### 阶段 B：根据裁决分支

- PASS：预注册 seeds1/2 和多环境确认；
- FAIL：保留失败门，做新的错误分解，提出不同数学机制；
- 任何情况下都不把旧投影标签或覆盖不足结果升级为 SOTA；无监督预训练暴露不再作为否决项。

### 阶段 C：外部证据闭合

- 继续恢复缺失官方 baseline 资产；
- PDB/Rfam/CHANRG 剩余身份、观测 mask、时间新颖性闭合；
- 下游去重、标签协议和覆盖率闭合；历史预训练暴露仅作补充记录；
- 长序列 GEC approximate sparse 假设单独验证；
- 最终只报告同协议、同覆盖、同标签的绝对差距。

## 7. 当前最大风险

1. margin 四臂可能 FAIL，主线仍无正结果；
2. 即使 margin 涨点，也必须排除投影协议混用和覆盖不足；无监督预训练暴露不再作为否决项；
3. 外部官方参考和 baseline 资产未完全闭合，无法形成完整 SOTA 表；
4. 计算/配额/网络问题继续拖慢一个月交付窗口；
5. 叙事若按模块罗列，会失去科学故事吸引力。

应对：坚持单变量干预、结构终点、对照排除、下游去重、标签协议一致、覆盖率完整和绝对差距校准；所有正结论必须有直接 artifact 支撑。
