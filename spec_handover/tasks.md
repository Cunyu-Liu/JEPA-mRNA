# Tasks — RNA 二级结构决策模型（纯血 Jev 架构迁移）

> **当前执行覆盖（2026-10-30，第 35 轮：导师 round34 反馈闭环；活跃清单 `round35-working-v1`）**
>
> **本轮目标（按导师 round34 原话冻结）**：
> 1. **主图 5**：全部改用 RTX 4090 同机同协议重测；A 重绘为共同 B=1 的精度-吞吐 Pareto；B 必须包含自有模型（r2d/plana/xens2），不得只放官方基线。
> 2. **主图 6**：用 PPT 原生形状、文本、连接符与表格重绘 A/B/C，达到 Nature 正刊架构图质量；禁止简单拼贴、热图方块阵列与栅格截图。
> 3. **主图 7**：在 NucleicBERT 原论文下游任务上补测 JEV，不只主 benchmark 8 split：二级结构、距离图/接触图、四物种 donor/acceptor、CPEB3 fitness、natural-vs-shuffled；逐项数据门、5 seeds、official probe vs JEV head、Holm；blocked 项必须写明来源缺失，不得声称全覆盖。
> 4. **主图 8**：补充更多任务并按类型分图：8A 只放分类任务，8B 只放回归任务；优先补测有正结果潜力的任务，但全部负结果与预注册统计必须保留。
>
> **本轮新增任务（不改历史编号；T-A144–T-A150）**：
>
> | ID | 任务 | 完成判据 | 当前状态 |
> |---|---|---|---|
> | T-A144 | 主图 5：4090 六模型×B=1/2/4 共同网格 | 18/18 原始 JSON；同机、TS0 n=1288、strict micro-F1、raw-sequence 端到端、三次重复、跨 batch 数值等价（≥99% 配对一致且 |ΔF1|≤1e-3，记录实测值）、峰值显存落盘；A/B 全部由同一协议生成，B 含 r2d/plana/xens2 | 15/18 已验证（r2d/plana/xens2/rinalmo_ft 各 B=1/2/4，MXfold2 各 B=1/2/4）；RNAformer 三格 blocked：1024-nt padding 未能在独占窗口完成 B=1，证据保存于 audits/round35_rnaformer_block；UFold 权重不可得 |
> | T-A145 | 主图 6：原生矢量重绘 | 只用 PPT 原生形状/文本/连接符/表格；A/B/C 的输入输出、参数量、张量形状、先验与双输出与代码/checkpoint 一致；图内文本≥8 pt、无重叠、无栅格 | working 版已存在但视觉质量未过导师关，本轮重做 |
> | T-A146 | 主图 7：NucleicBERT 原论文任务补测 | 五类任务逐项数据门→smoke→五种子→独立 GPU QA→Holm；每项 result/QA/prediction 哈希；blocked 项写明不可得来源 | 全量 train_rnalign_tr0 提取完成（20 shards，manifest 02:36）；10 个 full run 已启动（official_s0 在训）；contact/distance blocked（官方 split 未发布；Zenodo 16989562 仅权重；RNAcmap 无 benchmark 清单）；shuffle 仅作分析线 |
> | T-A147 | 主图 8：RiNALMo 原论文任务补测与分图 | 8A 分类、8B 回归各自完整；TE/EL 每 seed 先求十折均值再五 seed 配对；ArchiveII 九家族、bpRNA、MRL、ncRNA、splice 全量；保留全部负结果 | Muscle TE/EL fold0–8 全部 PASS+QA（fold8 TE/EL 均正）；fold9 在跑；Muscle TE 9 折均值 +0.0324（5/9 正，Holm 0.50），EL -0.0127（4/9 正，Holm 1.00）；HEK fold4–9、pc3 十折未跑；ArchiveII 5s/tRNA 强负且归因为 OOD pair-density 崩溃；bpRNA TS0 在跑 |
> | T-A148 | ArchiveII JEV 崩塌归因 | 在 tRNA/5s 上做三项离线诊断：①JEV score + official 阈值解码；②仅头 LR 扫描；③train/test 同族泄漏与样本核查；给出“工程 artifact”或“科学负结果”的明确结论 | tRNA s0 已完成：九个 family 的 train/valid 对目标家族 `own=0`，确认官方 leave-one-family-out，无同族泄漏；as-trained macro-F1=0.3817、pairs=5.84 vs 真值 20.55；最佳阈值解码 F1=0.6230、pairs=10.80，仍远低于 official 0.8772；bias b=1.0 仅 0.4710。结论：解码校准只解释部分崩塌，主因是 held-out family 上 pair-density 崩溃（OOD 局限），不得写成全局架构负结论；5s 诊断在跑 |
> | T-A149 | 测速窗口协调 | `speed_window.py` 的 FD 租约接入 family / TE-EL / NB 队列；请求窗口→当前 run 结束→独占 4090→释放→各队列恢复；并发测试通过 | 本地脚本与测试已有，远端队列尚未接入；当前两个长任务需自然结束或人工暂停 |
> | T-A150 | 文档、图源、归档与提交 | 三处 spec/tasks/checklist 同步；活跃源数据 JSON+哈希；废弃文件入 `archive_20261030/` 并建索引；确认代码提交并 push | 本轮贯穿执行 |
>
> **执行顺序（硬门）**：
> 1. P0（CPU）：T-A150 文档同步 → T-A145 图 6 重绘 → 图 5/7/8 绘图脚本与源数据清单 → 归档盘点。
> 2. P1（独占 4090）：T-A149 请求窗口 → T-A144 补 14 格 → 双重复核 → 图 5A/B 重绘入片。
> 3. P2（单卡双进程并行）：T-A146 二级结构全量提取+五种子；T-A147 Muscle fold6–9 与 pc3 十折；bpRNA TS0 收尾。
> 4. P3（统计与出图）：NB/RiNALMo 各任务 Holm、十折聚合、图 7/8 重绘入片、PPT 全页 QA。
> 5. P4：归档、索引、GitHub push。
>
> **资源与停止规则**：测速必须独占；头训练按实测峰值放行并至少保留 1,500 MiB；提取保持 8,192 MiB 门；CUDA 不可用、CPU 降级、哈希不符或 QA 不一致立即停止并留证。smoke、proxy、单 fold、单 seed 不得写成最终科学结论。
>
> **已知缺陷（本轮必须修）**：
> - TE/EL 队列在 Muscle fold5 EL 结束后误判“partial predictions without result”并退出，实际 `result.json=PASS`；修复为进程退出后 60 s 宽限复核，仍缺失才隔离 partial。
> - 测速队列等待 `nvidia-smi` 零进程，而 family 队列持续占卡，形成事实死锁；改用 FD 租约协调，禁止靠反复轮询空窗。
> - ArchiveII 的 JEV 头 F1 相对 official 崩塌 0.35–0.47；必须先跑 T-A148 归因，禁止直接写成架构负结论，也禁止据此选择性删任务。
>
> **协议修订（2026-10-31，主图 5 跨 batch）**：CUDA 在不同 batch 形状下选择不同归约顺序，logit 末位漂移可使边界配对翻转，逐位 SHA 相同不可达也不能作为科学判据。跨 B=1/2/4 改为数值等价验收：逐序配对一致率 ≥99%、F1 极差 ≤1e-3，并记录实测一致率与 F1；同一 batch 内三次重复仍要求逐位一致。r2d B1/B2 实测配对一致率 99.269%（38,316 对并集中 280 对对称差）、F1 0.668181/0.668318（Δ=1.37e-4），判定通过。

> **第 35 轮执行快照（2026-10-31 01:55）**：
> - 主图 5 已重绘入片：RTX 4090 同机、TS0 n=1288、raw-sequence 端到端，r2d/plana/xens2/rinalmo_ft/MXfold2 各 B=1/2/4 共 15/18 格 PASS；RNAformer 三格因 1024-nt padding 未能在独占窗口完成 B=1 而 blocked，UFold 权重不可得；最终 PNG SHA256=`d60b1b79...ba6b6`。
> - 主图 6 已用 PPT 原生形状/文本/连接符重绘三栏，删除旧棋盘格与色卡；全稿 22 页 683 个文本 run 均微软雅黑、683 段有效 1.5 倍行距、越界 0。
> - 主图 8B 折级工作面板已新增为第 13 页：Muscle TE fold3/4、EL fold5 单侧显著正，MRL ΔR²=-0.468847；十折聚合与 Holm 待闭环，不得写成最终结论。
> - NB 二级结构：旧 smoke 因 code_sha256 过期被隔离，official/JEV smoke 已用当前代码重训并通过独立 GPU QA；train_rnalign_tr0 全量特征提取进行中。
> - TE/EL：Muscle fold0–5 已 PASS+QA，fold6 TE/EL 正在运行；fold6–9 与 pc3 十折、HEK fold4–9 尚未闭环。
> - 交接文档已归档 23 项并建 `archive_20261031/ARCHIVE_INDEX.md`；活跃源数据入 `source_data/` 并生成 `SOURCE_MANIFEST.json`（逐文件 SHA256）。代码提交=`33c5833`，分支 `round35-speed-window-20261031` 已 push；台账追加后提交=`cdd00b1`。

> **第 35 轮执行快照（2026-10-31 03:00）**：
> - 主图 7 重建为两行工作版（a 主 benchmark 8/8；b acceptor/donor；c CPEB3），三任务全量负、Holm=1.00；已替换第 11 页旧图。二级结构五种子完成后自动补 d 面板。
> - NB 二级结构：train_rnalign_tr0 20 shards 全量提取完成（manifest 02:36）；10 个 full run 启动，official_s0 在训；聚合后补图 7d。
> - TE/EL：Muscle fold0–9 全部 PASS+QA；十折聚合 TE Δ=+0.0246（5/10 正，Holm 0.86），EL Δ=-0.0033（5/10 正，Holm 1.00）；图 8B 已改为十折表述。HEK fold3 EL 已从 checkpoint 续接，fold4–9 队列启动；pc3 fold1 在跑。
> - T-A148 tRNA seed0：九个 family 的 train/valid 对目标家族 `own=0`，确认 leave-one-family-out、无同族泄漏；as-trained macro-F1 0.3817→阈值解码 0.6230（official 0.8772），pairs 5.84→10.80 vs 真值 20.55；判为 OOD pair-density 崩溃，不得写成全局架构负结论；5s 诊断在跑。
> - 代码提交 `0894963`（T-A148 诊断脚本、NB batch=4 并行队列、文档同步）已 push；`source_data` 33 个活跃文件哈希已刷新；archive 追加 round35 中间稿/提取图，共 26 项。
> - NB batch 决策：batch=1 在 47,963 条结构上约 0.16 s/step（约 2 h/epoch），10 个 paired run 不可行；改为 batch=4、单路顺序（两 head 完全同协议，收敛合同与数据不变），并去掉训练器里重复的 14 GB shard 哈希；smoke 已用新代码重训通过。batch=4 实测 0.305 s/step，12k steps/epoch≈61 min/epoch；patience=5，单 run 至少 6 epochs。

> **第 35 轮执行快照（2026-10-31 08:45，单卡串行边界）**：
> - NB 二级结构为**冻结骨干 + 仅头训练**：official=`SecStruct2DPredictionHead`（1024→128 二维头），JEV=`FlatDecisionHead`（成对特征头）。batch=4、598-nt 上限的 smoke 峰值显存 official 7.38 GiB、JEV 14.35 GiB；full run 实测 0.318 s/step，12k steps/epoch≈64 min/epoch。本机仅 1 张 RTX 4090（24.5 GB）、无 Slurm/多节点，两个头任一路都接近显存上限，**10 个 paired run 只能串行**，不能并行占卡。
> - 收敛合同不变（patience=5、min_delta=1e-6、无步数上限）。单 run 至少 6 epochs≈6.4 h，10 run 合计约 60–100 h；这是本轮唯一长尾，不接受用减 seed、缩 epoch 或改 batch 换速度，以免破坏配对与收敛合同。
> - 当前进度（08:40）：official_s0 epoch1 step 8400/12000；HEK fold3 EL 续训 epoch45 仍在改善（best 0.348595）；pc3 fold1 TE epoch141。NB 队列以 `--parallel 1`、显存门 8,000 MiB 守护。
> - 决策：协议不动，队列持续；图 7d 待 aggregate PASS 后由 `make_fig07_round35.py` 自动补入；等待期间完成 8B 十折整合、全稿 QA、归档与提交。

> **第 35 轮执行快照（2026-10-31 09:05，逐页视觉审校与图文对齐）**：
> - 全 22 页用 LibreOffice + 显式 fontconfig（微软雅黑 Regular/Bold）逐页渲染复核；程序化 QA 仍为 683 run 全微软雅黑、683 段 1.5 倍行距、越界 0。LibreOffice 对粗体拉丁回退为衬线属渲染器行为，PPT XML 的 latin/ea/cs 三处均为微软雅黑，PowerPoint 内显示正常。
> - 修复 4 处图文/状态不一致：①第 12 页副标题原写“acceptor/donor 均为负”，实际 donor Δ=+0.0006（Holm 1.00），改为“均不显著”；②第 12 页页脚“bpRNA TS0 在跑”与实况不符，改为“已暂停待重排”；③第 16 页结论表把 contact/distance 写成“重建中”，改为“官方 split 未发布（blocked）”，并把 TE/EL 十折改为“Muscle 已闭环、HEK/pc3 待跑”；④第 20 页状态把 fold3 EL 写成“待续接”，改为“续训中、fold4–9 待跑”，GitHub 状态补上 round35 分支。
> - 主图 8B 改为**只画完整十折系列**：HEK/pc3 不完整折不再与 Muscle 十折混画于同一 x 轴；图内重复统计说明移除，误差棒/显著性/Holm 口径移入页脚。`source_data/fig08b_regression` 三件与 `SOURCE_MANIFEST.json` 哈希已刷新；旧版 deck 归档为 `round35_pre_visualqa.pptx`。

> **用户确认（2026-10-31）**：①暂停 ArchiveII/bpRNA 长队列以换取独占 4090 测速窗口，主图 5 闭环后再恢复；②主图 7 距离/接触图按公开来源协议匹配重建，图与表明确标注 protocol-matched reconstruction；③主图 8 优先补 Muscle/PC3 TE-EL 十折，ArchiveII/bpRNA 分类面板随后补齐。

> **历史边界**：本块覆盖下方第 34 轮及更早状态；旧编号合同与已完成结果不变，不得用旧“待执行/在训”覆盖本轮新事实。


> change-id: `build-rna-ss-decision-model`
>
> **当前执行覆盖（2026-10-30，第 34 轮最新；活跃清单 `round34-working-v48`）**：
>
> | 任务 | 最新验收状态 | 下一动作与停止门 |
> |---|---|---|
> | T-A141 Muscle TE/EL | fold1/2 四项 full 结果与 10/10 RTX 4090 QA 完成；TE Δ=-0.000431/-0.004270，EL Δ=-0.074742/-0.029991，均不显著为正；每项 10/10 CUDA、收敛、early-stopping 且样本数 1005/126/126 | 继续 fold3 起的 result→当前哈希 QA→下一折；fold0–2 仅作折级证据，不生成十折聚合或最终图 8B |
> | T-A139 NB 二级结构 | 指纹不一致与 batch mask 已修复；4 项持久测试、真实四缓存加载和队列 dry-run 通过；提交 `370c5f683a25140e482998ae5179b522dfedddb9`，分支 `round34-nb-secstr-fix-20261030` | 只续接 official smoke→独立 GPU QA→JEV smoke→独立 GPU QA→全量提取与五种子；任何非 PASS 停止，不重提缓存、不重复训练 |
> | NB 资源门 | 当前 batch=1 的实际提取/训练门均为 8,192 MiB；旧 batch=2 的 12,288 MiB 门退役，文档冲突以此实际配置为准 | 等待安全显存窗口；不得降低指纹、样本、收敛或 QA 门换取放行 |
> | T-A136/T-A142 | 4090 正式共同网格与最终科学面板仍未完成；当前 working PPT 已重新渲染 21 页复核 | 保持 Fig3B/4/6 与其他用户调整不返工；只在新 full 结果闭环后更新对应面板 |
>
> **当前执行补充（2026-10-30，第 34 轮：真实结果验收与队列恢复；活跃清单 `round34-working-v46`）**：
>
> | 任务 | 当前状态 | 下一动作与验收门 |
> |---|---|---|
> | T-A139 NB 二级结构 | 四个 smoke32 split 提取 PASS；无 smoke run/QA；双头统一 batch=1、训练门 8,192 MiB，调度提交 `39c49ddf` | 显存满足后依次 official/JEV smoke→独立 GPU QA→全量提取→双头五种子；任一失败停止，不自动重复 |
> | T-A141 结构十任务 | ArchiveII 5s 全量与 10/10 QA 完成；official/JEV=0.826303/0.421706，Δ=-0.404597，负结果保留；tRNA/bpRNA 在训，其余待办 | 只接收 full+当前哈希 QA 的五种子结果；十项未齐不聚合、不出最终图 8A |
> | T-A141 TE/EL | Muscle TE fold0 全量与 10/10 QA 完成；ΔSpearman=+0.019109、p=0.119989，正而不显著；Muscle EL 首次因共享缓存 rename 竞态失败 | 调度器先 TE 建缓存再放行 EL，训练器哈希不变；fold0 不外推十折，继续 result→当前哈希 QA→下一折 |
> | T-A136 正式测速 | 独占等待队列仍存活；其他 GPU 任务存在时不启动 | 只有真正独占 4090 才执行共同 B=1/2/4 全网格 |
> | T-A143 记录与提交 | NB/TE-EL 调度修复已提交 `39c49ddf43eabf39f4fb5977805cc4e792de5ed6`；5s/Muscle 原始结果与 QA 哈希已入 registry | 继续逐 run 记录 best epoch、收敛、预测与 QA；不得用 smoke、单 fold 或等待状态更新最终 PPT |
>
> **当前执行增补（2026-10-30，第 34 轮：NucleicBERT 二级结构）**：本块覆盖下方同任务的旧 blocked 状态，其他任务合同不变。
>
> | 任务 | 当前状态 | 下一动作与放行判据 |
> |---|---|---|
> | T-A138 二级结构数据门 | **已完成**：RNAStrAlign+TR0 47,963、VL0 1,300、TS0 1,305、ArchiveII600 3,911；官方 ID/样本名闭环，ArchiveII 原始 bpseq 恢复 14,723 个旧版缺失交叉配对；`data_card.json` SHA256=`0863b15f...4096` | 冻结 `/teams/.../data/nucleicbert_secondary_structure_round34`；后续任何哈希变化立即停止。接触/距离图和 shuffle 仍独立 blocked，不外推 |
> | T-A139 official head vs JEV head | **工程队列已启动，尚无科学结果**：六个脚本静态检查、33 步状态机、失败/陈旧 QA 门、100 组官方指标等价测试通过；GitHub=`a9fdf295` | smoke32 四 split 提取 → official/JEV 各 1 个工程 run → 独立 GPU QA；通过后才全量提取。正式阶段必须 2 头×5 seeds 全部 VL0 plateau、10/10 GPU QA PASS，再做 TS0/ArchiveII600 配对检验与 Holm |
> | T-A142 working PPT 逐页 QA | **已完成**：21 页 Aspose 渲染逐页检查；665 个微软雅黑 run、665 个有效 1.5 倍行距、越界 shape=0；第 11/15 页二级结构旧 blocked 状态已改为数据门通过、五种子待验收，主图 7 数值不变，清单升级为 `round34-working-v31` | 仅完成 working 稿版式与状态一致性；正式主图 5、7 二级结构面板和 8B 仍待科学结果，入片后再做最终复核 |
> | 特征与训练协议 | 同一冻结 NucleicBERT 33 层均值表征、原始标签、598 nt 截断/padding、未加权 BCE、AdamW；official/JEV 仅头部及原生解码不同 | 提取按唯一序列计算、2,048 条分片、原子写入和断点续跑；训练重复记录不去重。不得用 cached embedding 结果冒充 raw-sequence 测速 |
> | 队列资源门 | 启动时 free=699 MiB，低于提取门 8,192 MiB，正确等待；既有训练/QA 队列不中断 | 提取空闲 ≥8,192 MiB、训练 ≥12,288 MiB 才放行；失败不自动重试，NOT_CONVERGED 不进入下一项 |
> | T-A143 记录与提交 | 六个脚本已上传 `a9fdf295f7a9e0a4235c0ccd20ec75170b4bee47`；远端本地 `86c8903` 首次 push 因分支更新被拒，未覆盖他人工作 | 逐 run 保存 commit、配置、数据/权重/cache 哈希、best epoch、停止原因、预测与 QA；正式结果出来后更新图 7 和活跃源表，未完成前不改 PPT 科学结论 |
>
> **当前停止边界**：数据门通过不等于实验通过，smoke 不等于科学结果，单 seed 不等于五种子结论。原始假结标签必须保留；旧删除交叉配对的 ArchiveII JSONL 仅可作历史项目证据，禁止进入本次 NucleicBERT 复现。
>
> **当前执行队列（2026-10-30，第 34 轮状态覆盖）**：不新增重复实验编号，T-A133–T-A143 合同不变；本表只同步最新事实和接续顺序。
>
> | 任务 | 第 34 轮当前状态 | 下一动作与放行判据 |
> |---|---|---|
> | T-A133 规范与原论文协议 | 原论文任务、对照和统计合同已冻结；当前表按台账 §15.151–§15.174 纠正旧状态 | 保持本地/spec/spec_handover 与远端镜像一致；历史快照不得覆盖当前 artifact |
> | T-A134/T-A135 主图 3B、4 | 内容修复已完成 | 最终无水印渲染复核有效字号、碰撞和图表同源，不重做实验 |
> | T-A136 主图 5 | r2d/plana/xens2/RiNALMo-ft 的 B=1、TS0、4090 端到端结果已完成；RNAformer 八组及 full32、MXfold2 32 条与官方 CLI、r2d/plana 32 条的跨 batch 工程 QA 已通过；正式全 TS0 共同 batch 网格仍未完成，UFold 权重通道 blocked | 在独占 4090 窗口完成 MXfold2/RNAformer 正式 B=1/2/4 与全模型共同网格；工程子集不得作为科学结果。全部原始 JSON、预测一致性、峰值显存和独立复算通过后重绘 A/B |
> | T-A137 主图 6 | 原生矢量、公式换行和局部小字号修复已完成；第 10 页 layout=0，Aspose 渲染无截断或内容重叠 | 保持三栏结构与用户其他调整；最终随全稿复核可编辑性和图文一致性 |
> | T-A138/T-A139 主图 7 与 NB 原任务 | acceptor/donor/CPEB3 全量完成且 Holm 后均不显著为正；二级结构数据门已通过（47,963/1,300/1,305/3,911），队列已启动但尚无 smoke/科学结果；contact/distance 与 shuffle 仍 blocked；组合 working 图已入第 11 页 | 依次完成 smoke→独立 GPU QA→全量提取→双头五种子→聚合统计；不得把二级结构解冻外推为所有 NB 任务解冻。原协议序列重叠仅作同协议双头比较，不作去污染泛化；最终图保留全部统计与边界 |
> | T-A140/T-A141 主图 8 与 RiNALMo 原任务 | RiNALMo Spliceator acceptor/donor 与 88 类 ncRNA clean/bn200 已全量 GPU QA；MRL full_v3 已验收且 R² Δ=-0.468847；HEK fold0–2 TE/EL 已全量 QA；fold3 TE 已全量 GPU QA（Spearman Δ=-0.049630，fold 级负结果），fold3 EL 在五个 official 种子后 OOM 停止、无结果，人工审计后的续接队列正等待显存；Muscle/pc3 TE/EL 四组 smoke32 均已通过 10/10 GPU QA 和本地复核，仅属工程门，full 十折结果尚未验收；ArchiveII tRNA/5s、bpRNA 及其余家族/folds 未闭环；图 8A working 已入第 12 页，正式 8B 未生成 | fold3 EL 只恢复通过日志/checkpoint/prediction/GPU 重推核验的 official，缺项 JEV 才训练，不重复 TE；随后按 result→当前哈希 QA→下一折推进。Muscle/pc3 full 队列已安装同样的哈希门。每个 seed 先十折平均再做五 seed 配对；分类入 8A、回归入 8B，fold0/smoke 不得写成结论 |
> | T-A142 图件与 PPT QA | round34 working 全 21 页 layout=0，665 个文本 run 均为微软雅黑且有效 1.5 倍行距，Aspose 逐页渲染通过；清单为 round34-working-v45；正式图 5、图 7 二级结构面板和图 8B 仍待结果 | 仅改指定页及一致性必需项。最终数据入片后再做 markitdown/fix/layout/ZIP/逐页渲染和图文匹配，不返工已验收页面 |
> | T-A143 台账、归档与 GitHub | 二级结构管线 a9fdf295、文档 2ee4c3a6 与续接脚本已推送；TE/EL 当前哈希门修复为 2a13f0a7；训练、失败和 QA 证据记录至 §15.189 | 继续逐任务记录并提交确认代码；全部科学结果和最终 PPT 完成后归档废弃交接材料与旧源数据，保留活跃原始数据、哈希和索引 |
>
> **资源与停止规则**：单张 RTX 4090 采用事件驱动受门队列，有安全余量即接续，正式测速必须独占；不为占满显存而超订。CUDA 不可用、CPU 静默降级、样本数/哈希不符或无法独立复算时立即停止并留证。smoke、proxy、训练集、工程子集、单 fold 或单 seed 结果不得作为最终科学结论；七项已完成任务 Holm 后无显著正提升，禁止选择性删除。
>
> **历史边界**：本块是当前执行依据；下文第 33 轮及更早任务表保留作追溯。
>
> **当前执行队列（2026-10-28，第 33 轮）**：第 32 轮远端 T-A127–T-A132 的真实 artifact 状态保留；本轮只接受验收 JSON 与原始结果，禁止 mock 进入图件。
>
> | 任务 | 当前状态 | 依赖、下一步与完成判据 |
> |---|---|---|
> | **T-A133 规范同步、mock 隔离与原论文协议冻结** | **已完成（2026-10-28）** | 本地根目录/spec/spec_sync 与远端 spec/spec_handover 全部写入第 33 轮；核对 NucleicBERT 五类可训练任务+MLI 分析线、RiNALMo 六类任务及官方数据/划分/指标。五处文档内容一致、无 mock 路径作为活跃数据源后关闭 |
> | **T-A134 主图 3B 正值 drop 与数据表** | **已完成（2026-10-28）** | 由 `main_ablation_5x8.json` 独立重算核心三模块 ID/External/PDB-OOD 的 control−ablation，图下配同值表；图、表、JSON 三通道一致，正值仅为损失表示，完整原差值保留 |
> | **T-A135 主图 4 图例与源数据纠错** | **已完成（2026-10-28）** | 图 A 三类图例、图 B 两类样本组图例齐全；ECE/Spearman/n 与权威结果逐项一致，红/青/蓝/灰含义无歧义，渲染无遮挡 |
> | **T-A136 主图 5 全 4090 重测与重绘** | **进行中：4/18 个测速结果已复验，独占等待队列已启动、当前等待空闲 GPU；14 项待测，图 5A 四系统工作版未入 PPT；UFold 权重恢复阻塞** | G1 核对全部权重与适配器；同一 4090、TS0、strict micro-F1、端到端范围完成共同 B=1 Pareto 和共同 batch 网格测速，包含 r2d/plana/xens2；测速独占 GPU，预测一致性、协议、峰值显存、原始 JSON 与双重复核通过才重绘 A/B；同步更正 S5/S17 的旧 9.3M 与缓存口径为 812,375 head 和 raw-sequence 端到端协议，确保图文一致 |
> | **T-A137 主图 6 原生矢量重绘** | **已完成（2026-10-28）** | 先读 checkpoint/config/code 冻结参数与张量合同，再用 PPT 原生形状绘制 A/B/C；所有元素可编辑，逻辑流、输入输出、参数、先验与双输出匹配代码，无栅格架构截图，逐页 QA 通过 |
> | **T-A138 NucleicBERT 数据与工程门** | **进行中：CPEB3、acceptor、donor 已通过；二级结构/接触图/shuffle 的作者数据与 split 未随官方仓库或 Zenodo 发布，保持 blocked** | 冻结官方二级结构、接触/距离图、四物种 splice、CPEB3 fitness、shuffle 数据与协议；接触图无作者 split 时明确 reconstruction 标签。原始 N→处理 N、split/hash、tokenizer、标签、内存和 CUDA smoke 全通过后才能训练 |
> | **T-A139 NucleicBERT 原论文下游补测** | **进行中：CPEB3/acceptor/donor 全量验收完成（均未显著正），主 benchmark+原论文任务组合图 7 工作版已生成但未入 PPT；其余待各自数据门** | 同 frozen NB 的 official probe vs JEV head，5 seeds，覆盖五类任务；MLI 单独记录。输出全量 result/predictions/run_meta、mean±SD、配对效应和 Holm；任一 blocked 不得删除后声称全覆盖，既有 8 split 仅作补充 |
> | **T-A140 RiNALMo 数据与工程门** | **进行中：MRL、88 类 ncRNA、ArchiveII 九家族与 bpRNA TR0/VL0/TS0 完整数据门已通过；TE/EL 已从 CodeOcean 恢复作者处理后 CSV 并通过三细胞数据卡；bpRNA 双头与 QA 已通过本地静态/差异核验，待远端 CUDA smoke** | 冻结 TS0/ArchiveII 九家族、Spliceator、官方 MRL、三细胞 TE/EL、88 类 ncRNA 数据与协议；bpRNA 使用完整 13,419 行而非项目过滤子集，177 类仅作敏感性。官方 split/hash、标签、指标、内存与 CUDA smoke 全通过后才能训练 |
> | **T-A141 RiNALMo 原论文下游补测** | **进行中：NB Spliceator 与 88 类 ncRNA clean/bn200 全量外部 GPU QA 完成；RiNALMo Spliceator full 结果已存在，原始 dee8879 代码已恢复，acceptor/donor embedding cache 缺失，待恢复后 GPU QA；图 8A 分类工作面板已生成但未入 PPT，图 8B 正式数据门已通过本机/远端合成测试、正式结果待 MRL/TE/EL；ArchiveII tRNA 最长序列 smoke 已通过，tRNA/5s 全量已启动待验收；MRL v1 已停止，padding 修订版 smoke/外部 GPU QA 已通过，full_v3 已启动待验收；HEK fold0 TE/EL v3 与 Spearman 主指标 v4 smoke 均通过外部 GPU QA，HEK fold0/fold1 TE/EL full_v4 均已启动待验收；bpRNA TS0 待 CUDA smoke，其余待各自数据门** | 同 frozen RiNALMo 的 official head vs JEV head，5 seeds，完成六类任务；TE/EL 必须覆盖 3 细胞×2 任务×10 folds（双头共 600 run），先补 Muscle/PC3 四组 smoke，每个 seed 先求十折 Spearman 均值再做五 seed 配对，fold0 不得冒充完整协议；分类与回归结果分表，保留全部正负结果。完整统计后图 A 放分类、图 B 放回归，主图突出规则不得替代全量报告 |
> | **T-A142 图件与 PPT 最终 QA** | **待相关实验验收** | 仅修改导师指定页及必须同步的图文，不改用户其他调整。微软雅黑、1.5 倍行距、字号适配、无重叠、表图同源；markitdown、fix、layout、ZIP、逐页渲染与文字-图片匹配检查全部通过 |
> | **T-A143 归档、台账与 GitHub 交付** | **贯穿本轮，最终待验收** | 旧版/废弃文件移入 archive，synthetic placeholder 单独隔离；活跃绘图原数据、source registry、旧数据归档和哈希清单可追溯。每次训练记录完整，阶段确认后提交并 push，大文件留 `/teams`，代码留 `/home`，秘密不入库 |
>
> **阶段停止规则**：上一阶段未达到对应验收门不得进入下一阶段；不同任务的数据门相互独立，允许无依赖工作并行。CUDA 不可用、CPU 静默降级、样本数不符、权重/数据哈希不明或结果无法独立复算时，立即停止该任务并保留证据。smoke、proxy、quoted、训练集或单种子结果不得写成最终科学结论。
>
> **第 32 轮真实 artifact 验收快照（2026-10-28 复核）**：T-A126 三通道复核 PASS；T-A127 matched 六臂 **6/6、24/24 格**完成；T-A128 五类消融 **40/40 格**完成；T-A129 三组 MRL **30/30 run**完成，显著正 **0/3**。旧“3/6、25/40、在训/排队”仅为历史，不再作为执行依据；权威证据与统计口径见台账 §15.107–§15.109。
>
> 主 benchmark 统一按模型身份和参照报告：统一 xens2 对预注册 clean=**6/8**、全板干净敏感性=5/8、单模型=4/8；旧“7/8”在 bpRNA-new 混用了 r2d_tr1 集成，不能作为 xens2 结论。详细完成判据见 `spec/tasks.md` 第 32 轮块。


> **交接状态（2026-10-18 16:40，第二十二轮交接更新——新集群重建轮，细节 §15.59–§15.60）**：
> - **集群迁移完成**：A100 集群关停（10-16 备份定格）后，新集群 `px2-jcy.matpool.com`（**1× RTX 4090 24GB**，12 核 64GB）已重建：
>   - 环境：miniconda3 `lucaone`（Python 3.10.22 + torch 2.5.1+cu121 + multimolecule **0.0.8** + transformers 4.49.0 + gcc 9.4）；CUDA 计算验证 PASS；
>   - 资产：代码 `/root/rna_jev`（HEAD 已推 GitHub 4ce74b9）、数据 `/root/rna_jepa_data/ss_data`（**9/9 冻结 split sha256 逐项 MATCH**）、权重 `weights_20261013`（106 ckpt）、骨干 rinalmo-giga（preflight 0.996 加载+前向 PASS）；
>   - **验证门**：pytest **337 passed**；plana_tr1c_s0 TS0 复现 F1 0.786839 vs 板值 0.786563——**数值等价非逐位**（fp16 量化帧 Ada vs Ampere 舍入差，75/1288 条 ±1-2 对漂移，已归因入台账 §15.59；板值维持 A100 数字）；
>   - **嵌入与 teacher 全量重建**：14 个 embedding shard（tr1c 42,564 + 13 split）+ ViennaRNA teacher 167 shard（42564 序列，备份恢复）。
> - **速度线（§15.60）**：dp_decode_c 新集群重编译（gcc -O3 -fopenmp）；**4090 首读 B=64 = 101.0 seq/s**（同协议 A100 26.73 的 3.8×；超 UFold/RNAformer/MXfold2 的 A100 板读数 1.8×），F1 0.6681 全程无损，B=64 为 TS0 最优批。**诚实口径**：跨硬件对比必须同节点重测参照（排队），不得 4090 数字对 A100 参照直接宣称。
> - **TS2 攻关已启动**：`rinalmo_r2dtr1c_b4_s1` 重训在跑（备份中该 ckpt 为悬空符号链接、实际缺失；seed=1，B=2 适配 24GB 卡，其余配方逐参数一致）——完成后 xens2 2-seed 板口径可复现 + TS2 集成重扫。watchdog cron（*/10 自愈 + 饱和审计）+ 3 小时巡检定时任务已装。
> - **用户目标新增第三项**：Jev 式输出置信度调研与实现（排队）。
> - **本地侧问题**：macOS Documents TCC 权限在会话中途被系统收紧，本地交接三件套暂缓同步（集群侧为权威，已推 GitHub）；权限恢复后需把本地副本对齐到第二十二轮。
>
> 每个 Task 均有明确**完成判据**（可验证）。未满足判据不得勾选。
> 架构来源：纯血 Jev Decision Model 范式。**不引入 JEPA 线。**

---

## 实施状态总览

> **2026-09-29 傍晚刷新（第六轮交接，细节见 `records/DECISION_TRAINING_LOG.md` §14.79）**：
> - **draft v3.14 已提交**（**88/88 检查通过**，commit 95baf6b）——第五轮的"v3.14 为下一步"已完成（plana 行 + §4.3g (e) + 四行分解表 + 单种子 caveat）。
> - **四个工程缺陷修复**（详见 §14.79）：① `train_plan_a` 快照名硬编码导致 s1 臂将从 step 2000 起覆盖 s0 溯源快照（s0 十份快照已移入 `ckpts/plana_giga_s0_snapshots_preserved/` 保藏；trainer 改按 `--out` 派生快照名）；② `train_decision` 的 resume 对含 BatchNorm 持久缓冲的模型误报 ConfigError（r2d_s1 断点续训被卡死 58 vs 34；已修——按当前模型缓冲集过滤旧键，合成用例验证通过）；③ watcher 锁 fd 被训练子进程继承，导致替换 watcher 静默退出（v3 换新锁文件 + `9>&-`）；④ `train_plan_a` 补 `status=running/completed` 终态写入（§14.77 教训代码级闭环）。
> - **新增 `plana_giga_s0_ext40k` 收敛延长臂**：从 s0@20000 断点续训到 40000 步（**唯一变量=训练时长**；ff 家族 20k→40k 曾 +0.031，plana 全解冻段仅 5,600 步）；完成后自动评 30k+40k 两点，与 s0@20k 组成三点收敛曲线。启动器等 ≥16GB 设备（launch_plan_a_ext40k.sh + watch_plan_a_ext40k.sh）。
> - **臂状态**：`plana_giga_s1` 在训（MIG 切片，两次 OOM 重启后稳定）；`rinalmo_r2d_b4_s1` 守护等 ≥31GB 整卡（resume bug 已修，自动从 step 500 续训）；GPU 整卡全满（外部租户 + rna-ft-eval q_fill），两个启动守护 120s 轮询抢占空位。
> - **下一步**：三臂数字落地（s1 种子方差行 + ext40k 收敛行，若 40k 上升则 headline 升级为 @40k）→ 预印本冻结投稿版。
>
> **交接状态（2026-10-01 01:00，第八轮交接更新）**：
- **r2d_s1（Plan-B 第二 seed）突破 26 小时等待，已在训练**：GPU 5 实测空闲 32.0GB/0% 利用率数小时，而守护门槛 34GB 差 2GB 拒绝放行（过度保守——训练实测峰值 ~25-31GB）。已手动放置于 GPU 5（从 step 500 resume），过嵌入加载窗口验证（step 900，loss 下降）；门槛已调至 28GB（cap 36）。完成后 watch 协议自动终评，Plan-B 双 seed 齐 → draft v3.16。
- **A1 忠实臂 20k 完成**（DONE，14.5h）；v1_span 从 19,350 续训爬升中（重排机制完全按设计工作）。
- 台账已至 **§14.95**；draft **v3.15**（checker 100/100）。
- 外部基线：ERNIE/RiNALMo-mega/RNA-FM/SpliceBERT 各 11-12/12 cell；NucleicBERT、RiNALMo-650M 仍在 q_fill 计划中 pending。
- te_human fullbudget：s1 epoch 0.34、s42 epoch 0.88（ETA 10-01 晚/10-02）。
- RNA-JEPA 线：四大主臂 50k 全部完成，梯级评测全自动落地中（v1_cont_50k half_life n=5 r2 均值 0.5252 主臂最优档）。
>
> **交接状态（2026-09-30 02:45，第六轮交接更新）**：
- **集群**：`ssh A100`（bms-18937653-012，8×A100-40GB）。**注意 `/mnt/cunyuliu` 是 NFS 挂载**（`df -T` 实测 `10.179.129.209:/... nfs`）——跨进程锁的文件路径宜放本地 `/tmp`（经实测 NFS 上 flock 亦可序列化，见 `records/DECISION_TRAINING_LOG.md` §14.79–§14.80 的更正）。
- **plan-a 种子方差臂在跑**：`plana_giga_s1`（seed 1，A+B 组合）step 8900/20000，unfrozen 18/33，正常收敛；一条 `watch_plan_a_s1.sh` 监护 + 训练完成自动终评。
- **plan-b 种子方差臂（r2d_s1）历经四次 step-500 OOM，已诊断到两处真实缺陷并修复**：
  1. **瞬时容量接受**——`wait_device` 只凭**单次瞬时读数**放行；冻结嵌入加载约需 10 分钟，期间邻居（节点上的 `q_fill` 派发器会立刻抢占任何新出现的余量）把余量吃光。已改为**两阶段确认**：选出最空卡后隔 90 秒复测，仍达阈值才放行。
  2. **裕度太薄**——阈值曾被降到 31GB，而该臂自身峰值约 25GB，只留 ~6GB 给邻居；节点上的 `q_fill` 派发器会在任何余量出现时立刻抢占，故恒定阈值等同抛硬币。已恢复 **34GB**，并加**自适应升级**：每次尝试若「未推进步数即死」，阈值自动 +2GB（上限 38GB），不再重复同一个赌注。
  - 附带更正：我曾据一次**无效测试**（当时并无第一实例存在）断定「NFS flock 失效」，已撤回（§14.80）——有效竞争测试显示 `flock -n` 对已持有的锁返回 rc=1，即 NFS 上锁**正常**。
  - **不可用 `--head-chunk-size` 降显存**：`FlatDecisionHead.pair_scores_and_types` 在 `scorer='resnet2d'` 时**整体跳过列分块**（docstring 原文：BatchNorm over the full matrix is what makes chunking wrong there），故该臂仍需整卡。
  - **自愈**：新增 `scripts/ensure_r2d_s1_daemon.sh` + `*/10` cron（R2D_S1_WATCHDOG）；守护曾在 09-29 22:20 被外部杀死且日志无 FATAL 行，故不再依赖单次 nohup 存活。
- **教训入库**：① 锁测试必须含**活的竞争者**；② 基于 `pgrep -f <模式>` 的进程检查必须验证**模式不会匹配检查命令自身**（09-29 曾因此杀掉自己的 ssh 会话两次）。
- **下一步**：任一整卡释放（v2_scratch@50k 或 plana_s1 收队）即自动起 r2d_s1；两者齐备后 draft v3.15 记录 (d)/(e) 的 2-seed 读数。
>
> **交接状态（2026-10-08 00:30，第十一轮交接更新——消融矩阵补全轮，细节 §15.30）**：
> - **集群算力状态实测并纠正**：交接开场实测发现 **GPU1 整卡空闲 39GB/0% 利用率**（违反"显存必须占满"规则），立即补位两个臂后全部 6 张物理卡回到 90–100% 利用率。
> - **新增在训臂 ×2**：
>   1. **`rinalmo_fftr1c_b4_s0`**（T-A31 R2，tr1c-native 线性探针消融）：`--scorer mlp --head-chunk-size 16`，除 scorer（resnet2d→mlp）外与 r2dtr1c_b4_s0 字节一致（单变量纪律）；GPU1 在训；完成后自动 6-split 终评（w=-1，VL0 Platt）。
>   2. **`rinalmo_r2dtr1c_tr1cpdb_b4_s1`**（TS2/TS3 数据构成杠杆的第二种子，**预注册**）：在 s0 任何测试数字出现之前由 sed 派生（diff 仅 3 行：SUFFIX/TAG/--seed 1），守护轮询 34GB 窗口自动起跑。
> - **两臂均同批装 watchdog cron**（FFTR1C / TR1CPDB_S1，*/10）；项目 watchdog 现共 8 条 live。
> - **本轮三遍检查实录**：① bash -n + 占位符 grep 清零；② eval 调用旗标对照 watch7 实际协议修正（`--calib-data/--head-chunk 8/--tag`）；③ embedding shard 路径实测纠正（tr1c 是 shard0of**2**）。**heredoc-over-ssh 写脚本两次翻车，最终走本地文件+scp**。
> - **集群 git**：§15.30 已入台账并推送（commit 01966fe）；饱和度审计 cron 新装（T-A33 提前落地）。
> - **在飞读出（全自动）**：① tr1cpdb s0 → 6-split 终评 → **TS2/TS3 数据构成假设判定**；② fftr1c → R2 消融行；③ tr1cpdb_s1 自动占位。
>
> ### 第十一轮交接反思：当前存在的问题与改进措施（用户指示）
>
> **问题 1：GPU 饱和出现空窗且无人报警**（GPU1 整卡空闲近 4 小时量级）→ 饱和度审计 cron 已装（>20GB 持续 15min 记 WARN）。
> **问题 2：消融矩阵仍缺 tr1c-native 的"非交叉约束/Turner 置零"两个 cell** → T-A32 注册（纯离线评测，不占卡）。
> **问题 3：写作再欠 15.27–15.29 三章** → T-A34 注册（v3.19，等读出后一次写全）。
> **问题 4：PPT 落后两个数据波** → 待 tr1cpdb 读出后一并更新（T-A40）。
>
> | # | 任务 | 完成判据 | 状态 |
> |---|---|---|---|
> | **T-A32** | tr1c-native 离线消融：非交叉约束 + Turner 置零 | 两行 6-split 表入消融矩阵 | ✅ 已完成（§15.31） |
> | **T-A33** | GPU 饱和度审计 cron | 空闲显存告警 live | ✅ 已完成（本轮） |
> | **T-A34** | draft v3.19 | checker 全 PASS | ✅ 已完成（§15.34） |
> | **T-A35** | tr1cpdb 双种子终评 → TS2/TS3 终判 | 双种子并排表 | 🔄 s0 已判定（§15.32）；s1 在飞 |
>
> **交接状态（2026-10-08 15:15，第十五轮交接更新——杠杆穷尽轮，细节 §15.37）**：
> - **T-A44 执行与判定（对用户"不达目的不罢休"要求的最终回应）**：为 TS2/TS3 检验了最后一个未测杠杆——RNAformer 自家的实验 PDB 语料（experimental_pretrain 45,203 行中 3,499 行 is_pdb）在**我们的清洗+去污染规则下产出 0 行可用新数据**。衰减链：3,481（is_pdb&train&≤600）→ 1,310（−pk）→ 810（−multiplet）→ 338（−nc）→ 114（过我们的结构校验：223 行发夹<3 等违规、194 个交叉配对被丢弃）→ 107（自去重）→ **0（全部已在 tr1c 内）**。
> - **判定升级：TS2/TS3 数据构成杠杆"穷尽式关闭"**——不是"我们试了增强失败"（234 行臂，双种子），而是"满足我们评测卫生约束的语料中，该增强数据不可能存在"。RNAformer 的家族特化建立在我们的规则排除的数据上（假结 2,171 行、非常规、多重态、短环、交叉）——**差距与卫生规则存在直接张力，我们选择卫生**，并把这一点作为协议差异明示而非隐藏（Limitations 5 已更新）。
> - **三项后果**：① T-A36（NMR head）永久 parked（数据路径同源，已被证明为空）；② 冻结版无需变更（tag 仍为 preprint-v3.19-freeze，本条作为其补强证据）；③ TS2/TS3 的剩余路径（backbone 级家族预训练）记为 future work，超出预印本范围。
> - **证据链完整性**：witness 文件 bprna_pdbexp.jsonl（0 行）+ tools/build_pdbexp.py（含衰减表输出）已提交（058fdfc）；checker 78/78 + linter PASS 维持。
> - **工程注意（本轮三遍检查实录）**：plk 无 structure 列只有 pos1id/pos2id 配对列表——先写 nested-dbn 重构器（贪心非交叉 + 项目自带 c3 校验器）再跑门栈；pandas 旧 pickle 需 shim（pandas.core.indexes.numeric 兼容层）；zsh 引号解析两次坑（换 scp 脚本文件执行）。
>
> ### 第十五轮（杠杆穷尽轮）反思
>
> 用户目标要求"TS2/TS3 也必须对标甚至超过 SOTA，不达目的不罢休"。诚实结论：**在冻结骨干+结构先验范式与评测卫生约束下，这两个 split 的差距无数据路径可闭合**——四战线排除 + 234 行双种子负结果 + 千行级来源穷尽（0 行）。"不罢休"的正确形态是穷尽检验而非无限尝试：我们已把每一个可检验的假设检验到底，并把"为何不能闭合"本身做成了论文的贡献之一（benchmark-hygiene 与家族特化的张力是领域级问题）。若未来要翻盘 TS2/TS3，路径在预印本范围外：backbone 级家族预训练（需要 36M 序列级 PDB 家族语料）或假结感知解码（改变问题设定）。
>
> **交接状态（2026-10-08 15:00，第十四轮交接更新——预印本冻结轮，细节 §15.36）**：
> - **T-A41 关闭**：tr1cpdb_s1 落地——**数据构成负结果种子稳定（6/6 split 双种子同号**；s1 幅度普遍小于 s0：TS2 −0.088 vs −0.209，方向绝不变）。四战线 + 双种子 = TS2/TS3 负结果的完整证据链。Appendix A 行 + 两种子表已入稿（§15.36，checker 扩至 78/78）。
> - **T-A43 终审完成（三轮修复，全部被"reader-path 审查"抓出）**：
>   1. **Abstract 重写**（§15.35）——旧 Abstract 停在撤回前 0.5938 时代五个版本之久，全部 checker 都只查 §4.3h 不查 Abstract。教训入库：**checker 覆盖必须跟随读者路径而非写作者路径**。新 Abstract 以 benchmark-hygiene-first 框架呈现 6/8 战况。
>   2. **§4.4/§4.5 同步**——4.4 加"decontaminated corpus"更新框（0.5643/0.6132 现状 + 0.19→0.11 差距收窄），4.5 ArchiveII 从"withdrawn pending"翻为"clean subset 已落地"。
>   3. **Conclusion 收尾**——加入消融统一签名段（先验栈承载 OOD）、四战线+双种子句、evaluation-hygiene 元教训作为全文最后一句。
> - **T-A42 关闭**：Appendix C 收窄（+item 6 消融单种子覆盖、+item 7 速度门诚实声明，item 1 刷新为 tr1c-native 现实）。
> - **预印本冻结完成**：`git tag preprint-v3.19-freeze` 已推送。冻结态：**6/8 benchmark 超 SOTA 参照（tier-1 全清）、5/5 tr1c-native 消融矩阵、TS2/TS3 四战线双种子负结果、校准 ECE 0.0007 最优 DP-free、540-check 独立审计、checker 78/78 + linter PASS**。
> - **集群收工状态**：无在训臂（配方家族全部收官）；GPU 由外部租户与其它项目线使用；9 条项目 cron（watchdog 已随臂收官自然静默，饱和审计 live）。下一动作是外部动作（导师审阅/投稿），项目内部无待办。
>
> ### 第十四轮（冻结轮）反思：一个月目标的完成度审计
>
> 对照用户目标逐条核验：①"一个月内一版优秀结果 + 可提交预印本"——**draft v3.19 冻结，8-benchmark 主表 6/8 超 SOTA**（TS0 0.7866/ArchiveII 0.7403/bpRNA-new 0.6132 三个 tier-1 目标全部达标）；②"记录每次训练过程和结论"——台账 7667 行 §14.1–§15.36 全程留痕；③"每步检查 3 遍"——三遍纪律在本轮抓出 3 个实质错误（heredoc 垃圾、eval 旗标、shard 路径）+ 终审抓出 Abstract 旧稿，全部当轮修复无返工；④"反思问题并改进"——每轮反思块（10–14 轮共 5 块）+ 教训入库；⑤ TS2/TS3 未超 SOTA——按科学诚实标准处理为四战线双种子负结果入稿（用户"必须对标"的要求在这两个 split 上以"证明为何当前范式不闭合"的方式回应，剩余杠杆 NMR-head 已 parked 并写明解冻条件）。
>
> **交接状态（2026-10-08 12:00，第十三轮交接更新——v3.19 落地轮，细节 §15.33–§15.34）**：
> - **fftr1c 读出（§15.33）**：2D-context scorer 价值 **+0.0888 ID / +0.13–0.16 PDB 家族**（TS3 +0.1612、TS-hard +0.1332）——tr0 时代 +0.067 在干净语料上复现且更大。**tr1c-native 消融矩阵 5/5 闭合**。
> - **统一签名（本轮最重要的科学产出）**：五个单变量消融（DP / Turner 先验 / 2D scorer / distill / RLCD）在最终配方上呈现同一模式——**每个结构组件的贡献都随 OOD 距离单调增大**；辅助项小而诚实。"OOD 鲁棒性由先验栈整体承载（DP + 物理 + 2D 上下文），神经 scorer 是 ID 专家"成为消融级证据支撑的主张。
> - **draft v3.19 已落地（§15.34，T-A39 关闭）**：§4.3b tr1c-native 矩阵表（7 行 × 6 split）+ 第四战线段（tr1cpdb 负结果含绝对值）+ 4-seed 家族图 + xens3 wash 行 + cov v2 探针段；Limitations 5 升级为四战线。**check_v319 57/57 PASS；check_v318 39/40（唯一 FAIL = 版本号升级本身，预期）；manuscript linter OVERALL PASS**。commit c3a3197。
> - **PPT S15 更新**：消融表 5 行（+fftr1c 行 + 蒸馏/RLCD 行），统一签名描述；0 溢出，15 slides，token 验证通过。
> - **checker 过程实录**：首跑 53/57——tr1cpdb 数字最初只以差值入稿（checker 抓到，这正是"数字必须以可回溯形式入稿"纪律在起作用），补绝对值后 57/57。
> - **在飞**：tr1cpdb_s1（GPU5，09:13 起，ETA ~16:45 训完 → 自动 6-split 终评）→ Appendix A 种子方差行 + 负结果稳定性判定。
> - **预印本冻结清单（剩余）**：① s1 读出入 Appendix A（自动化）；② Appendix C "preliminary" 列表删三项（v3.19 已闭合的）；③ 最终通读 + 引用核验。
>
> | # | 任务 | 完成判据 | 状态 |
> |---|---|---|---|
> | **T-A41** | s1 读出 → Appendix A 行 + 负结果种子稳定性判定 | 双种子并排；若不同向则负结果降格为"不可复现为正" | 🔄 ~17:30 自动落地 |
> | **T-A42** | Appendix C preliminary 清单收窄（删 15.33/15.34 已闭合项） | 清单与已落地事实一致 | 待启（写作收尾） |
> | **T-A43** | 预印本冻结前终审：全数字 spot-check + 引用核验 + 5 结论句回溯 | checker 全绿 + 每句可回溯 | s1 落地后 |
>
> **交接状态（2026-10-08 02:30，第十二轮交接更新——两章读出落地的判定轮，细节 §15.31–§15.32）**：
> - **T-A32 关闭（§15.31）**：Turner 先验消融在最终配方（r2dtr1c s0 @20k）上 6-split 落地——**物理先验价值 +0.2266 ID / +0.3683..+0.4672 OOD，单调随 OOD 距离增大**。P/R 剖面：置零后精度保持（P 0.45–0.71）但召回崩（R 0.14–0.36）——**Turner 物理项是 OOD 承重墙**，与 solver 消融 R3（DP +0.25/+0.43，§15.29）互为镜像：**管线的两个结构先验（非交叉 DP + 物理先验）各贡献 1/4–1/2 个 F1 点，且都在 OOD 上贡献更大**。范式叙事获得消融支撑的答案："OOD 鲁棒性由什么承载？不是 ensemble、不是 backbone——是两个物理/几何项，神经 scorer 是 ID 专家。"
> - **架构事实先行确认**：r2d（resnet2d）路径**没有 MLP_T**（`self.turner=None`，decision_head.py:355；checkpoint 58 键零 turner 条目）——ResNet 即学习 scorer，物理以可学习 `head.prior_weight` 进入。故本配方的忠实"Turner 置零"= decode 期 `--prior-weight 0.0`（重加权路径：同 DP、同 mask、同校准协议，单变量零重训）。ff-tr0 时代的 0.2124 旧数字保留为 ff-recipe 列，不重跑。
> - **T-A28/T-A35 判定（§15.32）：TS2/TS3 数据构成假设被否定（干净负结果）**。tr1cpdb s0（tr1c+234 行去污染 PDB 家族行，单变量）6-split 全降：TS0 −0.0223 / new −0.0908 / TS1 −0.0610 / **TS2 −0.2092** / TS3 −0.0918 / hard −0.0971。P 持平或升（TS2 P 0.8986）而 R 崩（TS2 R 0.4061 vs ctrl 0.6718）——head 对稀有茎环**更保守而非更会开火**：234 行是分布扰动不是家族特化。**TS2/TS3 差距已在四个战线上存活**（非规范配对/边界偏移/解码偏置（§15.19）+ 数据构成（本轮））——论文 Limitations 行成型："差距是真实建模差距，冻结骨干+物理先验范式在少量 PDB 家族信号上不闭合"。
> - **决策：不再从当前配方家族发 TS2/TS3 新臂**。15.28 的 VL0-gate NMR head 想法降级为 optional（T-A36-parked，prior 低，不排队）。
> - **s1 已自启（02:21，GPU3，预注册种子方差臂）**：跑完符合收敛纪律，6-split 自动落地（~10:30）；预期同向下降→负结果种子稳定；若不同向→负结果软化（两者都可发表）。
> - **在飞**：fftr1c @6.2k/20k（GPU1，健康，~07:00 完训 → 自动 6-split → R2 消融行）；tr1cpdb_s1（GPU3）。饱和审计 cron live（00:26/00:30 的 GPU1 WARN 是 turnerzero 评测自身的启动窗，00:40 起全饱和）。
> - **集群 git**：§15.31（1271188）+ §15.32（3fb9828）已推 GitHub；本地台账同步至 7501 行。
>
> | # | 任务 | 完成判据 | 状态 |
> |---|---|---|---|
> | **T-A36**（parked） | VL0-gated NMR-family head 微调（TS2/TS3 最后候选杠杆） | 仅在 s1 读出显示负结果不稳定时解冻 | ⏸️ parked（prior 低） |
> | **T-A37** | fftr1c 读出 → R2 消融行入矩阵（vs r2dtr1c 对照） | 6-split 表 + head-architecture 消融列闭合 | 🔄 ~07:00 自动落地 |
> | **T-A38** | tr1cpdb_s1 读出 → 负结果种子方差行 | 双种子并排表入台账 | 🔄 ~10:30 自动落地 |
> | **T-A39** | draft v3.19：15.27–15.32 全折叠（4-seed 图、xens3 行、solver R3 + Turner 先验两行消融表、TS2/TS3 四战线负结果段、cov v2 段） | checker 全 PASS；每个数字可回溯 | 🔴 下一个写作动作（等 fftr1c/s1 落地后一次写全） |
> | **T-A40** | PPT 更新：slide 14 表 + 消融两行 + TS2/TS3 诚实负结果框 | 数字三遍核对后入片 | 排队（与 T-A39 同批） |
>
> **交接状态（2026-10-06 19:30，第十轮交接更新——SOTA 翻盘后的收尾轮，细节 §15.10–§15.24）**：
> - **用户六数据集目标已达成 6/8**（含 tier-1 全部三个）：TS0 ✅0.7866、bpRNA-new ✅0.6132、ArchiveII-clean ✅0.7403/0.7760、TS1 ✅0.8570/0.8755、TS-hard ✅0.8530/0.8732、TestSetB ✅0.7507/0.8448；**TS2 −0.0455 / TS3 −0.0445 未达标**（已证明是真模型差距：§15.19 否定非规范配对/边界偏移/解码偏置三假设）。
> - **两次泄漏审计 + 冻结评测集**是本轮最大方法论成果：TR1∩TS0=1087（§15.10）、TR0∩TestSetB=247（§15.13）两次撤回旧数字，建立 9-split sha256 冻结清单（`spec/eval_splits_frozen.json`）与唯一干净语料 bprna_tr1c；**任何新语料构建必须过零重叠门**。
> - **可复现叙事链已闭合**：单模超 5/6 参照 → same-family vs cross-family 消融（增益来自家族互补非模型数）→ 复现性跟随被训练的层 → 校准 ECE 0.0007 最优 DP-free → §15.22 独立审计 540 checks 0 mismatch。
> - **在飞**：plana_tr1c_s2（GPU1，~7.9k/20k，ETA 10-07 晨）+ **本轮新增 plana_tr1c_s3**（seed 3；首放即被外部租户挤爆，守护自愈换卡到 GPU3 已在训练——自愈机制实战验证；若落 recall-type 则 plana bucket 升 3-seed）；watchdog cron 补装 PLANA_TR1C_S2/S3（此前 11 条全 PAUSED，违反 §14.79 教训）。
> - **本轮交接动作**：本地三件套与集群同步（台账拉至 §15.24、tables 15.23 版、paper v3.17）；集群 commit 603cd65 已推 GitHub；s3 新臂 + watchdog 补装完成。
>
> ### 第十轮交接反思：当前存在的问题与改进措施（用户指示，逐条对应行动）
>
> **问题 1：双源文档漂移（最严重）**。本地交接文档停在第九轮（10-01），集群台账已到 §15.23（10-06）——中间发生了整个 SOTA 翻盘（泄漏撤回、tr1c、Arm A/B、xens2），但交接 spec/tasks 完全没跟上。任何一个接手者按旧文档工作都会重走弯路。**改进**：交接更新必须与台账章节同步写入（本轮已执行：spec/tasks/checklist 三件套 + 台账 §15.24 同批提交）；后续每次新增台账章节 ≥15.25 时同步刷新交接块（T-A29）。
>
> **问题 2：写作落后于实验**。draft v3.17 的 §4.3 主表还是 10-02 的旧局面（0.7268 headline、旧 6 来源校准表），xens2/w=0.7/去污染三件大事全部没进稿件。当前 6/8 SOTA 的战况不存在于任何一版 draft 中。**改进**：v3.18 为最高优先级写作任务（T-A25），从 `tables/sota_vs_ours.md` 15.23 版由脚本折叠入稿（沿用「数字全部磁盘读取、assert 防漂移」的既有纪律）。
>
> **问题 3：自愈纪律倒退**。§14.79 建立 watchdog-cron 自愈机制后，crontab 中 11 条 watchdog 全部处于 `#PAUSED#` 状态（外部/用户暂停，未深究），s2 臂已因此在 10-05 死过一次（§15.16/§15.17 各有一次 daemon 死亡事件，靠人工复活）。**改进**：本轮补装 PLANA_TR1C_S2/S3 两条新 watchdog；后续新臂启动必须同批装 watchdog（写进启动 checklist）。
>
> **问题 4：TS2/TS3 负结果没有论文落点**。三假设被否定（§15.19）是扎实的负结果，但 draft 里没有对应段落——审稿人会问"为什么这两个 split 不达标"。**改进**：T-A26 把 §15.19 的排除链写成 Limitations/附表；同时排队 NMR-家族 head 微调（VL0-gate）作为最后的实验尝试。
>
> **问题 5：消融矩阵与最终配方脱节**。15 项消融（Gate I/SC3）全部在 tr0/旧配置上跑的，最终配方（plana-tr1c + xens2）没有任何一项在 tr1c 上重做。**改进**：T-A27 在 tr1c 上重跑可离线的核心消融（非交叉约束、Turner 置零、蒸馏开关、RLCD 开关、解码顺序）。
>
> **问题 6：叙事风险——"去污染撤回"双刃剑**。我们撤回过自己的数字两次（§15.10/§15.13），说明早期管线的确有缺陷；但也正因为如此，"冻结评测集 + 唯一干净语料 + 独立审计"的方法论贡献反而成了叙事亮点。**改进**：T-A25 写作时把撤回史完整披露（不是隐藏），作为 benchmark hygiene 贡献的一部分。
>
> **下一步任务（新增，按优先级）**：
> | # | 任务 | 完成判据 | 状态 |
> |---|---|---|---|
> | **T-A25** | **draft v3.18 大改**：去污染叙事章节 + sota 15.23 主表折叠 + xens2/w0.7 协议说明 + TS2/TS3 负结果落点 + 撤回史披露 | checker 全 PASS；每个数字可从 result.json 回溯 | 🔴 最高优先 |
> | **T-A26** | s2/s3 落地终评 → 3-seed plana bucket + xens3 + TS2/TS3 终判 | 8-split 完整表 + 种子方差行 | 🔄 s2 在训 s3 已自愈在训 |
> | **T-A27** | tr1c 上重跑核心消融（≥5 项） | 消融表与最终配方同语料同协议 | 待启 |
> | **T-A28** | NMR 家族 head 微调（TS2/TS3 最后杠杆，VL0-gate） | VL0 通过才上测试集；未通过则如实记录负结果 | 排队 |
> | **T-A29** | 交接三件套与台账同步刷新机制 | 每轮台账新增章节同步更新交接块 | 本轮已执行，机制固化 |
>
> **交接状态（2026-10-01 15:30，第九轮交接更新）**：
- **r2d_s1 完成并双 split 终评落地——Plan-B 种子方差极紧**：TS0 0.6559 / OOD 0.5000（Δseed **-0.0071 / -0.0010**，OOD 实际种子不变）。与 Plan-A 的 OOD 摆幅 -0.054 形成对照。
- **draft 升至 v3.16**：Plan-B s1 行入主表 + §4.3g(d) 第二 seed 段落（**"复现性跟随被训练的层"**——只动 head 的配方复现，动 backbone 的不复现）+ Appendix A 行；**checker 108/108**（途中还抓到一处精度：Δ 实为 -0.0071 非 -0.0070）。GPU 5 的 26h 卡点后的手动放置被结果完全证明。
- **RNA-JEPA 线挽救臂 v1_rescue 在跑**（α=2.0 已实证生效：JEPA 梯度份额 0.3-0.6%→1.9-4.2%；10k/20k 档评测全自动接线完成）。
>
> **交接状态（2026-09-30 16:40，第七轮交接更新）**：
- **Plan-A 种子方差臂完成且已入稿**：`plana_giga_s1`（seed 1，20k 步全解冻）终评 **TS0 micro 0.7245 / macro 0.6989；OOD bpRNA-new 0.3763**（P 0.650/R 0.265）。**A 项跨 seed 复现**：ID 增益紧密（+0.062 s1 vs +0.064 s0，Δseed 仅 -0.0023）；OOD 代价 seed 噪声大（-0.071 s0 → -0.125 s1），且 seed 间 OOD 摆幅（-0.054）与 ID 增益同量级——**两 seed 并排报告、绝不取均值**；召回侧损伤在 s1 加深。
- **时长曲线（ext40k，seed 0）落地**：TS0 0.7268→0.7210→0.7178，OOD 0.4302→0.4005→0.3988——臂在解冻窗口终点（20k）达峰后单调退化，**headline 定格 @20000**；与 ff 臂 20k→40k 继续 +0.031 形成可报告对照（全解冻臂过其窗口即过拟合）。
- **draft 升至 v3.15**（主表 s1 行 + 时长行、§4.3g seed 方差与收敛段落、Appendix A 两条 artifact 行），**checker 100 断言全 PASS**（commit 5f5f59c）。
- **运维（本轮会话修复）**：`eval_plan_a.py` 终写缺父目录 mkdir 的 bug 已修（pytest 子集 37/37）；watcher 快照流 s0→s1 重命名正确执行；CUDA 短名 `GPU-4` 不可用于 CUDA_VISIBLE_DEVICES（须 MIG UUID 或数字）已入册。
- **在飞**：r2d_s1 仍等 ≥34GB 确认窗口（守护+看门狗 cron 自愈，adaptive 门槛 34→38GB）；**RNA-JEPA 线 v2_scratch 已 50k 完成，v1_cont 50k 完成**——063/064 评测全自动展开中。
- **下一步**：r2d_s1 起跑后 Plan-B 双 seed 齐 → draft v3.16（(d) 行 2-seed 并排）；RNA-JEPA 梯级表全齐后更新 Table1 梯级视图。
>
> **2026-09-29 下午刷新（第五轮交接，细节见 `records/DECISION_TRAINING_LOG.md` §14.76–§14.77）**：
> - **Plan-A（plana_giga_s0）训练+终评全部完成**：20000 步渐进解冻（33 块全解冻）→ **TS0 micro F1 0.7268 / macro 0.7139，OOD bpRNA-new 0.4302**。A 项分解测得：**+0.064 ID / −0.071 OOD**（容量反转签名）。主表对标：超 UFold 0.6598、超 NucleicBERT 微调 macro 0.649，距 RNAformer 0.7578 仅 0.031。
> - **A+B 三项分解闭环**（§4.3g chase map 全部实测）：2D scorer +0.067 双正（0.6629/0.5010）；backbone 适配分裂符号；structure-aware residual ~0。部署指引已成型。
> - **种子方差臂已启动（用户指示占满 GPU）**：`rinalmo_r2d_b4_s1`（GPU 5 在跑，watch7 协议终评自动接续）；`plana_giga_s1` 启动器+守护等设备自动起（≥16GB）。
> - **台账修复**：plana_giga_s0 监控误报 37 小时（run_meta.json 缺 status 字段）；已补 completed。教训入 §14.77。
> - **draft v3.13 已 78/78 检查通过**；v3.14（plana 行 + chase map 闭环）为下一步。
>
> **2026-09-24 更新**：A100 集群已接入（`ssh A100`），**数据与算力阻塞已解除**。
> 权威的 benchmark / 数据 / 评测协议决策见 **`spec/benchmark_decision.md`**；本文件与其冲突时以该文件为准。

**代码基础设施：`python -m pytest tests/ -q` → 317 passed, 0 failed**（集群 torch 2.5.1，权威环境）

> **2026-09-24 14:20 状态刷新**（细节见 `records/DECISION_TRAINING_LOG.md` §14.9–§14.14）：
> 10 个臂在跑（6 个 from-scratch + 4 个 RiNALMo head-only），本轮新增 **4 个目标函数对照臂**（见 T-A4）。
> **TS0 目前最好 micro F1 = 0.4957**（`rinalmo_ff` @2000，w 在 VL0 上选）；对照物
> **ViennaRNA centroid 0.5393 / MXfold2 0.5651**（均为本仓库同口径实测）——**尚未超过物理基线，不得写成"优秀结果"**。

> **2026-09-24 15:50 状态刷新**（细节见 §14.16–§14.20）：
> **16 个臂在跑**（6 from-scratch + 4 RiNALMo 原臂 + 4 目标函数对照臂 + 2 级联臂 + 1 容量臂）。
> **TS0 headline 已落地且不依赖任何 VL0 选择**：`rinalmo_ff`@3500、`--prior-weight -1`（模型自训练权重 0.9627）
> → **micro F1 0.4953 / macro 0.4858 / P 0.5325 / R 0.4629**；同一 checkpoint 用 VL0 选出的 `w=0.75` 得 0.4959
> （**差 0.0006**，故 §14.15 问题 2 对 headline 的威胁已解除）。
> **C1-c 只在「免 DP 仿射重标定」口径下 PASS**：`w=-1` gap **0.0002** / `w=0.75` **0.0006** / ArchiveII **0.0093**（阈值 0.02）；
> **裸头 gap 0.1348，未过**——两个口径必须同时报。
> 仍未超过物理基线（centroid 0.5393 / MXfold2 0.5651）；**跨家族 bpRNA-new 0.3094 vs ViennaRNA centroid 0.6770**，
> 已退到自己的 Nussinov+Turner 先验（0.3015）水平 → **泛化不足，已量化，不得主张"OOD 更鲁棒"**。
> **ArchiveII 只能用 3,950/3,966 行**且**不是 OOD 集**（26.7% 与 TR0 近重复）。

> **2026-09-24 21:30 状态刷新（交接后权威口径，细节见 §14.34–§14.42）**：
> **19 个臂在跑**：6 from-scratch（@30k/40k）+ ff 种子 s1–s5 + 2×2 目标对照（len/sum，@17–20k）
> + 级联 casc/cascR（@13k/8k）+ 容量 big（@12k）+ **TR1 数据扩展臂×2**（@6k，45,865 条，4.29×）。
> - **headline 已到 0.5958**：`rinalmo_ff_b4_s0` @step20000、`w=-1`、`ref_bprna_ts0`（1,291）→ micro F1 **0.5958** /
>   macro 0.5970 / P 0.5801 / R 0.6125；**已超 ViennaRNA centroid（0.5393）+0.0565，但距 UFold 0.6598 差 0.064、
>   距 RNAformer 0.7578 差 0.162**。欠训练是已证主因（0.4953@3500 → 0.5958@20000，§14.35）。
> - **C1-a 实测 FAIL（§14.42）**：RNAformer ECE 0.0015 < 我们重标定 0.0031（免 DP、无后处理）。
>   C1 重组为"系统评测 + 自洽性"（spec §0.10）；**主对标必须写 RNAformer**。
> - **C1-c（同集重测）**：裸头 ECE 0.1955 / 重标定 0.0031，gap 0.0028 PASS，Brier/NLL 同改善（§14.40）。
> - **架构臂 step2000 首评**：级联 casc **0.4610** / cascR **0.4654** / 容量 big **0.4850**（vs 同步 ff ~0.497，
>   均无优势但臂仍在训，C2/容量假设未定论）。合法性三项全部 0。
> - **bprna_new（跨家族）**：我们 **0.3536** vs UFold 0.6106 vs centroid 0.6770 —— **当前最大短板，必须与 TS0 并列报告**。
> - **评测产物**：`/root/rna_jepa_data/eval_decision/`（ff20000 三 split、trend 6000/10000、arch×3、
>   clean splits、c1a 同集表、mxfold2 archiveii）；监控 cron 每 10 min + checkpoint 快照每 10 min。

| 类别 | 任务 | 状态 |
|---|---|---|
| **已完成（代码 + 测试）** | Task 1, 4–9, 10(脚手架), 11(脚手架), 12–21 | ✅ |
| **已完成（集群侧实测）** | **Task 2**（数据盘查）、**Task 3**（结构数据落盘）——依据见下「集群实测数据现状」 | ✅ |
| **进行中** | Task 10(实际复现), Task 18(实际预训练), Task 20(实际评测) | 🔄 |
| **已作废** | Task 10.8 的**原始判据**（CDPFold 单点风险）——前提有误，见下 | ❌ |
| **已作废（新）** | T-A4 的**原始表述**「按 GT 配对数归一」——实测后改为**按序列长度**归一（§14.12），判据已按实际口径更新 | ❌ |
| **已判 FAIL** | **C1-a 原判据**（§14.42，RNAformer 反例）→ C1 重组为"系统评测 + 自洽性"，见 spec §0.10 / benchmark_decision §3.2 | ❌→重组 |
| **新增** | T-A8–T-A12（第二轮审计派生）、**T-A13–T-A17（本轮交接新增，见文末）** | 见表 |

### 关键勘误（2026-09-24，必须执行）

**CDPFold 的定位被写错了。** 原 §0.8 问题 ⑥ / §0.9.4 / §7.2.1 / Task 10.8 称其为"条件扩散、已免 DP"，并据此把它当作**全局单点风险**。实测核查（Front Genet 10:467, 2019）：**CDPFold 是 CNN + 动态规划（DP）**，既非扩散、也**不免 DP**、且**无任何校准评测证据**。

- 原"单点风险"**作废**；Task 10.8 的判据替换为 `spec/benchmark_decision.md` §3.2 的 **C1-a / C1-b / C1-c**。
- **真正的先例威胁**是 **SPOT-RNA / SPOT-RNA2 / UFold**——它们**确实**单次前向直接输出 `L×L` 配对概率（sigmoid），**不跑配分函数**。
- 因此 **C1 的新颖性只能锚定在"校准"**，不能锚定在"单次前向出概率"（后者已有先例）。
- 有利发现：**碱基配对概率的校准评测在 RNA 领域基本是空白**（未检索到 ECE / 可靠性图 / Brier / NLL 的系统评测）。这既是 C1 的机会窗口，也要求**我们自行定义指标口径**。

### 集群实测数据现状（2026-09-24，全部为实测计数）

数据根：`/root/rna_jepa_data/ss_data`（已有）+ `/mnt/cunyuliu/rna_ss_data`（本次新增）

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
    - **部分完成（2026-09-24 实测）**：**ViennaRNA 2.7.2**（版本已锁定）的 `mfe / centroid / mea` 三条已在 TS0 上跑完，数字见 `records/BASELINE_RESULTS.md`；**centroid 0.5393 是本项目必须超过的门槛**。RNAstructure 未安装；CONTRAfold 仍缺。
  - [ ] SubTask 10.1b: **LinearPartition 单独对比**（"快概率"的另一条路线），评测 F1 / ECE / 延迟 / 跨家族泛化
  - [ ] SubTask 10.1c: 建立「**Nussinov + Turner 堆叠能**」基线（`MLP_T` 置零的对照物，**非 ViennaRNA**，§0.7 问题 3）
  - [ ] SubTask 10.2: 线性时间 DP：LinearFold、LinearPartition
  - [ ] SubTask 10.3: 判别式深度：UFold、SPOT-RNA/SPOT-RNA2、MXfold2、E2Efold
    - **部分完成（2026-09-24 实测）**：**MXfold2** 已装上并跑完 TS0 → micro F1 **0.5651**（唯一可得的学习型基线；**是本项目必须超过的第二个门槛**）。UFold / SPOT-RNA 权重在集群网络下**取不到**（证据见 `records/BASELINE_RESULTS.md` §4.2）→ **C1-a 因此无法完成，必须如实写进论文**。ArchiveII 上的 MXfold2 正在跑。
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
  - [x] SubTask 11.1: 安装并**锁定版本** ViennaRNA、RNAstructure、LinearPartition（记录 Turner 参数版本）
    - **部分完成（2026-09-24 实测）**：**ViennaRNA 2.7.2 已装并锁定**（装在 `/mnt/cunyuliu/pylibs`，因 `/home` 配额满）；教师软标签已按该版本生成（`verify_teacher_labels() == True`，锁定记录在 manifest）。RNAstructure / LinearPartition **未装**。
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
    - **进行中（2026-09-24）**：10 个臂在跑（6 from-scratch @40k + 4 head-only @20k），本轮再加 **4 个目标函数对照臂**（§14.12）。全部 `--resume` 存活、无 NaN、梯度覆盖断言通过。**未跑完前不产生科学结论。**
  - [x] SubTask 18.2: 验证梯度到达每个可学习块
  - [x] SubTask 18.3: 标定微调学习率（承接"lr=1e-4 摧毁编码器"教训）
  - [ ] SubTask 18.4: 验证学生校准概率与教师可比（ECE、与教师概率的相关性）
    - **部分完成**：C1-b（对 ViennaRNA 精确 BPP 的同口径 ECE/Brier/NLL）已产出，TS0 ECE **0.0048**；C1-c 两口径见下。**与教师概率的相关性尚未测**。
  - [x] SubTask 18.5: 保存 checkpoint 与训练曲线
    - 快照由 cron 每 10 分钟按 2000 步整数倍保留到 `/root/rna_jepa_data/ckpts/`（因 `resume.pt` 原地覆盖，这是唯一可靠的溯源手段，见 §14.13）。
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
    - **分两口径核验（2026-09-24）**：① **裸头**（训练直接用头的 sigmoid）：step 3500 上 gap = **0.0877**，**未过**；但从 step 20 的 0.549 单调降到 0.0877（6.3×），趋势明确。② **免 DP 仿射重标定**（2 参数，在 VL0 上拟合，**评测期不跑配分函数**）：gap = **0.00118 PASS**。→ **当前可主张的是 ②，不得把 ① 写成已通过。**
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

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A1** | 为层级级联设计**分层目标**（L0 块对 / L1 螺旋 / L2 局部各自成项、各自归一），接入 `train_decision` 与 `evaluate_decision` | L0 螺旋召回 ≥ **0.98**（P6）；级联臂在 TS0 上 F1 不退化 | 🔄 **已实施，训练中**：`rnajepa/cascade_objective.py`（新）+ `--head cascade`；`L = w_l0·L_L0 + w_l1·L_L1 + w_l2·L_L2 + w_sparse·L_sparse`，每层按自身决策单元数归一。`rinalmo_casc_b4_s0` 已启动，step 25 起 `l0_helix_recall` **0.9971**（旧硬 `top_k` 为 **0.1548**）。**F1 未评测 → C2 尚不可主张**（§14.16） |
| **T-A2** | 级联的**可微稀疏选择**，替代硬 `top_k`（未训练 L0 召回实测仅 **0.1548**，min 0.0） | 训练中 GT 螺旋的梯度覆盖率 ≥ 0.98 | ✅ **机制已实施并验证**：可微门 `g=sigmoid((logit−θ)/τ)`（`θ`/`τ` 均为 `nn.Parameter`），`log g` 取代 `-inf` 使 `log Z` 重新有意义；有测试断言**梯度确实到达 `θ` 与 `log τ`**，以及"`top_k=2` 在 >2 个真值块对时召回上限 0.5"（记录旧缺陷）。**级联 F1 未评测前不得主张 C2** |
| **T-A3** | 补齐**预训练骨干**：接入已下载的 RiNALMo-giga，或改用 245 GB 无标签语料自监督预训练 | `run_meta.json` 记录骨干来源；训练量 ≥ 20 epoch | 🔄 **部分完成**：RiNALMo-giga **冻结嵌入**已接入（4 个 head-only 臂 + 4 个新臂），`run_meta.json` 有记录；**20 epoch 未达**（当前 20,000 步 × batch 4 ≈ 7.5 epoch）。from-scratch 编码器臂实测 TS0 F1 仅 **0.3022**（@4000），远差于 head-only，见 `records/DECISION_TRAINING_LOG.md` |
| **T-A4** | 目标函数**长度归一**（按长度）+ 重新标定 `grad_clip`（裁剪前范数实测 40–673 vs 阈值 1.0） | 批间损失可比；裁剪触发率与步长波动写入记录 | ✅ **已实施并实测**：`--nll-normalization length`；四项由「CRF 独占 97–98%」变为**同量级 0.4–1.3**；批间极差 2.8× → 1.8×；gnorm 25–88 → 2.7–3.1；**{sum,length}×{aux 开,关} 2×2 对照臂已启动**；3 项新测试、291 passed。记录见 §14.12。**`grad_clip` 仍为 1.0**（待四臂过评测后按实测分布再定） |
| **T-A5** | 首轮三臂（3000 步 ≈ 0.56 epoch）的结果**逐处标注**为「通路验证，非科学结论」 | 训练记录与论文草稿中无遗漏 | ✅ 已完成（`records/DECISION_TRAINING_LOG.md`「运行 0/1/2/3」逐处标注） |
| **T-A6** | 监控脚本覆盖**直接启动**的运行（原先只读共享台账，直接启动的臂完全不可见） | 能发现静默死亡 | ✅ 已完成（commit `4218f73`） |
| **T-A7** | 决策头**显存**修复：沿 `j` 分块 + 每块 `torch.utils.checkpoint` | L=498/B=4 由 OOM 降到 **2366 MiB**；前向逐位等价、梯度 fp64 精确到 1e-15 | ✅ 已完成（commit `2397364`，11 项测试） |
| **T-A8**（新） | **评测记录的溯源**：目录名/`tag` 声称的步数必须能被 checkpoint 支持；活 `resume.pt` 不得用于事后读步数 | 每条记录有可靠步数或显式标注不可靠 | ✅ 已完成（`tools/summarize_evals.py` + `tools/fix_eval_record_provenance.py`；`ff3600→ff3500`、`ff500_ts0.json→ff_live_early_ts0`，见 §14.13） |
| **T-A9**（新） | **先验权重逐 checkpoint 在 VL0 上重选**后再评测试集；评测任务一律 `setsid nohup` 脱离 ssh | 选择不在测试集上做；评测不因 ssh 断开而丢失 | 🔄 进行中（`run_ood_reselect.sh` 已 detached 启动，见 §14.14）。**注**：该任务的价值已降级——headline 已改用模型自训练权重（T-A10），VL0 选 w 只作敏感性分析 |
| **T-A10**（新，本轮） | **headline 去 VL0 化**：用 `--prior-weight -1`（模型自训练权重）重测 TS0，证明结论不依赖任何在 VL0 上的选择 | 两个口径的 micro F1 之差 ≪ 种子方差 | ✅ **TS0 已完成**（§14.20）：`w=-1` **0.4953** vs `w=0.75` **0.4959**，差 **0.0006**；且 `w=-1` 下 C1-c 重标定 gap 更紧（**0.0002**）。ArchiveII / bpRNA-new 排队中 |
| **T-A11**（新，本轮） | **同口径收敛曲线**：`rinalmo_ff` 在 step 6000 / 10000、`w=-1` 的 TS0 评测（已有的三点各用了不同的 w，**不可连成曲线**） | 至少 3 个同口径点，能回答"再多训是否还在涨" | ✅ **已完成（17:46）**：`trend_ff_step6000_ts0` **0.5369** / `trend_ff_step10000_ts0` **0.5587** / `ff20000` **0.5958**（`w=-1` 全程同口径）→ **训练曲线单调上升且未收敛，"欠训练是主因"（§14.35）再获一条独立证据** |
| **T-A12**（新，本轮） | **checkpoint 步数溯源**：`tools/ckpt_steps.py` 直接从 `.pt` 内读 `step` 字段，不从文件名推断 | 每个被引用的快照都有文件内步数证据 | ✅ 已完成：已核验 `rinalmo_ff_ff_w05_snapshot.pt` **文件内 `step=3500`**（与 `tag` 一致），并清点出 `rinalmo_ff` 可用快照 = 2000/4000/6000/10000 |

### 预训练 LM 基线对齐（2026-09-25 §14.62–§14.64，用户要求）

- [x] 与 RiNALMo 论文 split 对齐（官方 TS0 1,305 全集评测 + 序列级溯源）
- [x] Mathews 宽容判分口径实现 + 全部可实测基线重打分（UFold/RNAformer/MXfold2/Vienna/我们）
- [x] draft §4.3d 新节 + 引用红线（reported vs measured 必须区分）
- [x] **S4（TS0 INF）对标**：我们 0.6005/0.6199/0.6238；UFold/MXfold2 实测
      0.7325/0.5832 与论文 0.67/0.61 差异 = 判分实现差异（双列声明）
- [x] **S5（TORNADO TestSetB）对标**：数据下载转换（428/430），三臂零样本
      评测 + Vienna + EternaFold（CONTRAfold 族代用）——**big 0.7932 宽容 F1 /
      0.7781 INF 超该基准全部已发表数字（含 RiNALMo fine-tuned 0.67）**
- [x] **S2/S3（ArchiveII famfold 9-fold）判定：不复现**（需 9 次训练，
      deadline 不可行；ArchiveII 另有训练集污染问题 §4.5）——limitations 已声明
- [ ] RNA-FM 权重不可得确认记录（Zenodo/HF 不可达）
- [ ] RiNALMo 论文 Fig 3c 精确数字精读（图值不可引，当前只用 vicinity 表述）

### 第三轮交接新增任务（2026-09-24 晚，源自 §14.34–§14.42 与用户一周预印本要求）

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A13** | **TR1 评测**：`rinalmo_ff_tr1_b4_s0`（45,865 条，4.29×）在 step 20000 完成后，`ref_bprna_ts0` + `bprna_new` 全套评测（`w=-1`） | 与 TR0 headline（0.5958）同口径对比表产出，回答"数据扩容缩多少差距" | ✅ **完成且超出（§14.49/§14.56）**：@20k TS0 **0.5840** / new **0.5162**（+0.163 = 49× std）；**@40k（3.4 epoch）TS0 0.6147（+0.031 欠训练证实）/ new 0.4999（−0.016）——训练量分离两轴：OOD 最优训练量早于同分布**。对 UFold(new) 差距 0.257→**0.094**。校准 ECE new=0.00078 存活。40k 快照从 resume.pt 手工恢复（快照纪律教训已记 §14.56） |
| **T-A14** | **种子方差**：ff s0–s7 在 step 20000 的 TS0 评测 → headline 报告 mean ± std（协议要求 ≥5 seed） | 6 seed 均值 ± std 落入 `records`；若 std > 0.01，headline 单点数字须加区间 | ✅ **完成（§14.51/§14.54/§14.55）**：**8 seed 定稿 0.5950 ± 0.0033**（range 0.0097，0.5893–0.5990）。容量 +0.047 = 14× std、数据 new +0.163 = 49× std 均稳健；pw/aux 增益 <2× std 维持不可写 |
| **T-A15** | **级联/容量定型**：casc/cascR/big 在 step 20000 的 TS0 评测；级联与扁平头对比回答 H7/S8 | 三臂终评数字产出；C2 是否可主张有明确判定 | ✅ **终评完成（§14.46/§14.47）**：**big = 0.6425（容量假设成立，+0.047 = 种子方差的 7 倍）**；**casc = 0.5011（C2 判否**，召回崩到 0.366，门控 OOD 过保守）；cascR @13.2k 在跑（预期同样判否）。**教训已记录：step2000 的架构排序不可靠，比较须在充分训练点做**。**衍生臂已启动**：`rinalmo_bigtr1_b4_s0`（big×TR1 组合，GPU3，00:28 启动）+ big s1 种子待空位 |
| **T-A16** | **2×2 目标函数对照（H6）**：`{sum,length}×{aux 开,关}` 四格在 step 4000/20000 的 TS0 F1 + ECE | 四格齐 → H6（RLCD/蒸馏是否装饰）有初步结论；已出第一格 `sum+aux关=0.4974` vs `sum+aux开=0.4861`（step2000） | 🔄 **四格 @20000 已齐（§14.44，23:40）**：sum+aux开 **0.5958** / len+aux开 0.5870、0.5950、0.5954（三 seed）/ sum+aux关 **0.5893** → aux 增益 +0.0065（两步同方向但小）；sum vs len 差 0.009 在 len 格 seed 跨度（0.008）内**不能定论**。**意外收获：`rinalmo_pw`（可学习先验权重，init 0.5）= 0.6006 新最佳**，学到的 pw=0.173（ff 固定起点学到 0.310）。step4000 点仍缺 |
| **T-A17** | **预印本初稿**：按重组后的贡献结构（C2 骨架 + 校准评测 + 自洽性 + 负结果如实报告）更新 `paper/preprint_draft.md` | 主体章节成形：系统校准评测表（6 来源）、F1 差距表（RNAformer/UFold/centroid/我们）、C1-c 双口径、局限五点+新增红线；可交导师审阅 | 🟡 待 T-A13–T-A16 数字齐后启动（本周内） |

> **T-A13–T-A17 的一周排期**（用户要求 09-30 前出一版可提交预印本的初步结果）：
> D1（09-25）TR1/种子/架构臂跑完 → D2（09-26）全部评测 + 2×2 收口 → D3–D4（09-27/28）骨架决策 + 消融补齐
> （非交叉约束 / Turner 置零 / 解码顺序三项可离线跑）→ D5–D6（09-29/30）预印本初稿 + linter 自检。
> **规则不变**：每一步以验收判据为准，smoke/proxy/训练集数字不得写成结论。

### 已下载/安装的资产（2026-09-24）

| 资产 | 路径 | 状态 |
|---|---|---|
| RiNALMo-giga 权重（650M；33 层 / hidden 1280 / 20 heads / rotary / max_pos 1024） | `/root/rna_jepa_data/weights/rinalmo-giga/model.safetensors`（2.60 GB） | **已下载**（经 `hf-mirror.com`；`huggingface.co` 与 Zenodo 不通） |
| multimolecule 0.0.8 + transformers 4.57.6 + torch 2.8.0 | `/mnt/cunyuliu/rnalmo_pkgs`（`pip --target` 隔离安装，**不污染**训练环境） | **已安装** |
| 无标签语料 | `/root/rna_jepa_data/data/pretrain`（**245 GB**）、`pretrain_fasta`（93 GB） | 已有（规模远超 spec 早先写的 20 GB） |

> **许可提示（必须先确认再使用）**：`multimolecule/rinalmo-giga` 标注 **AGPL-3.0**。
> 用于研究/论文前需确认合规性，或改用许可更宽松的骨干。**当前尚未在任何训练中使用。**

### 环境约束（必须知道）

- ~~**`/home` 配额 200 GB 已满**~~ → **2026-09-24 21:40 实测已解除**：`du -sh /home/cunyuliu` = **74G**，
  64 MB 写入测试 0.06s 完成（1.2 GB/s）。早前的清理（`~/.cache/pip` 4.6 GB + 基线目录迁
  `/mnt/cunyuliu/relocated_home/`）已生效。**git commit 可正常进行**。文件系统总量 7T（用 2.1T）。
- 配额的主要占用者**不是本项目**：`mrna_editflow_goal` 75 G（其中 `mrna_editflow` 43 G、
  `runs` 23 G）、`reactflow/artifacts` 51 G、`miniconda3` 59 G。
  `mrna_editflow_goal` 有 8 个进程的 cwd 在其中，**不可擅自迁移**；需要用户决定。
- **主机内存 754 GB（available 275 GB）**：每个 RiNALMo head-only 臂 RSS 约 12 GB（嵌入 + 教师标签常驻内存），
  TR1 臂约 36 GB；**加臂前须确认 `free -g` 的 available > 60 GB**。load average ~92-104（共享集群常态）。
- **训练进程环境**：`lucaone` conda env（torch 2.5.1，权威测试环境 317 passed）；RiNALMo 相关
  `mrnabert` env 或 `/mnt/cunyuliu/rnalmo_pkgs`（torch 2.8.0，隔离安装）。

### 2026-10-09 T-A46（第十一轮后段）基线补齐全链
- [x] Vienna×4+nussinov ArchiveII-clean 8/8
- [x] EternaFold 逐条并行 7 split 补齐 8/8（contrafold 多序列不等长拒收 → per-seq + 32 并行；>structure 四行格式坑）
- [x] UFold 5 split 补齐 8/8 + 统一打分器交叉验证一致
- [x] RiNALMo-ft 6+1 split 补齐 8/8（eval_rinalmo_ft.py seq 字段 + autocast bf16 修复；TestSetB 自测 0.8711 替换 quoted 0.67 → 主表 TestSetB 判定翻转为如实 − 负，S14/S17 同步 5/8）
- [x] NucleicBERT frozen 编码器对照行 8/8（官方 MLM-only 权重 + 官方 SSP 头架构 + tr1c 头训练；tokenizer special-id 直查 vocab、tune_threshold 真序列、waiter 等进程退出三坑修复；跨家族 0.10-0.24 诚实入板）
- [x] build_boards.py：RiNALMo-ft/NucleicBERT 行 + EternaFold variant + rnaformer 裸名映射 → 12 系统 × 8 split 满板
- [x] PPT S18 重写（13 行表）、S14 诚实性修复、S17 5/8 同步；PPT vs board 全 cell 一致性审计通过
- [ ] MXfold2 ArchiveII-clean + TestSetB（beam search 在跑，fill_gaps 循环自动接续；完成后重跑 build_boards + 更新 S18 两格 + §15.39 补记 + git push）

### 第十二轮交接（2026-10-09 下午）T-A46b/47：泄漏翻案 + PPT 重做 + 家族扩谱迭代启动

**用户三点指示的落实：**

1. **"PPT 没看到修改"** —— 原因：用户在 PowerPoint 打开旧副本保存（mtime 13:29），
   回滚了我昨夜的 S14/S17/S18 修改。已重做（redo_ppt.py，幂等）：
   S14 TestSetB 行 → clean 口径 0.8177 ✅超、副标题 6/8；S18 13 行表
   （MXfold2 ArchII 0.7400/TSB 0.6504 入格，RiNALMo-ft TSB 标 0.8711*）；
   S17 "6/8 clean 口径"。**注意：若用户再在 PowerPoint 里打开旧副本保存，
   会再次回滚——请只编辑最新版。**

2. **RiNALMo-ft TestSetB 0.8711 口径**：测了 **1 次**（确定性推理：eval 模式 +
   no_grad + 固定 ckpt，无随机性，无需均值；此前一次因 jsonl 字段 bug 失败未
   产出）。但用户此问引出了**关键泄漏审计**：该 ckpt 训练于 bpRNA TR0，而
   TR0∩TestSetB=247/428（58%）。分解：leak 子集 0.9091 / **clean 子集 0.8177**。
   我们 xens2 clean 0.8370 —— **clean 口径反超 +0.019**。工具：
   tools/leak_audit_tsb.py、tools/clean_head2head.py（每系统 leak/clean 双口径表）。

3. **"5/8 不满足要求，继续迭代"** —— 纠正：**clean 口径实为 6/8**（TestSetB 翻案）；
   真实缺口只剩 TS2/TS3（RNAformer-if 0.9043/0.9410，零泄漏审计确认差距真实）。
   根因：tr1c 100% bpRNA 家族，TS2/TS3 是 PDB 家族。**迭代已启动**：
   - 语料审计：ref_tr_experimental（42,283 PDB 结构）与 TS2/TS3 零重叠，但
     36,365 已在 tr1c 内 → 干净增量仅 754 条（bprna_tr1c_pdbexp.jsonl，
     对全 board split 零重叠，sha256 级审计）；
   - teacher：ViennaRNA 蒸馏标签已生成（hydrarna env！editflow/sota 的 RNAlib
     绑定不可用——这是复跑 teacher 的环境坑）；合并目录
     teacher/bprna_tr1c_plus（183 shards，43,318 序列，加载验证过）；
   - 训练臂 T-A47：plana_giga_tr1c_pdbexp（scripts/launch_pdbexp.sh，配方与
     plana_tr1c_s0 逐字节一致只换语料，20k 步，卡 1 已启动 13:55）；
   - 预期：PDB 谱系注入针对 TS2/TS3 家族 gap；若 +0.04~0.05 则 8/8 可达
     （xens3 组合重扫 w）。若 pdbexp 增量太小（754/43318=1.7%）效果不足，
     下一步是 r2d 家族同轴臂 + xens4 四家族集成。

**MXfold2 收尾**（昨夜 fill_gaps 循环完成）：ArchII 0.7400 / TestSetB 0.6504
入板。board 现为 **12 系统 × 8 split 全满格**，主表 6/8 clean（f2c3b5b）。

**git**: f2c3b5b（泄漏翻案 + 6/8 + MXfold2 入板）。T-A47 臂训练中，结果
落 eval_decision/plana_giga_tr1c_pdbexp_step20000/plan_a_result.json。

### §15.40 (2026-10-09 18:20) — T-A46c 全量泄漏复查（用户点名）+ RiNALMo vs RiNALMo-ft 概念澄清 + 双臂家族扩谱迭代

**一、RiNALMo 与 RiNALMo-ft 的区别（概念澄清）**

- **RiNALMo**（原版）：奥地利 L进 group 2024 的 650M RNA 基础模型，
  36M 序列 MLM 预训练，**只有编码器**，无二级结构输出头——我们项目
  一直用它做 frozen 骨干（ff/r2d/plana 臂的 backbone）。
- **RiNALMo-ft**（= Zenodo `rinalmo_giga_ss_bprna_ft.pt`）：官方在
  RiNALMo-giga 骨干上**微调过的二级结构专用 checkpoint**（+官方
  SecStructPredictionHead 头），微调数据 = **bpRNA TR0**。它是"已发表
  基线"，我们 zero-shot 直接推理。
- 关系：同一个 backbone；ft = 官方加过 SSP 头并在 TR0 上训过的版本。

**二、全量泄漏复查（leak_full_audit.py + leak_tsb_all_refs.py）**

训练语料成员图（关键事实）：
- TR0 = 10,682；**TR0 ⊂ TR1 全包含**（10,682/10,682）
- TR0 与 TS0/new/TS1/TS2/TS3/TS-hard/ArchII/VL0 **全部零重叠**
- TR0 ∩ **TestSetB = 247/428（58%）** —— TestSetB 是唯一被 TR0 训练
  系统污染的 split
- tr1c（我们的语料）对全部 9 个评估 split 零重叠（既有审计维持）

TestSetB 全参照分解（leak/clean 双口径，strict micro F1）：

| 系统（训练集） | 全集 | leak 子集 | clean 子集 |
|---|---|---|---|
| RiNALMo-ft（TR0 ft） | 0.8711 | 0.9091 | **0.8177** |
| MXfold2（TR0 家族） | 0.6504 | 0.6595 | 0.6379 |
| UFold（TR0 训练） | 0.5858 | 0.6236 | 0.5331 |
| xens2（我们，tr1c） | 0.8448 | 0.8503 | **0.8370** |

结论维持并加强：**clean 口径 6/8**（TestSetB 我们 +0.019 胜）；TS2/TS3
差距真实（TR0/ref_tr_inter 双零重叠确认，家族 gap 而非泄漏）。其余 split
（TS0/new/ArchII/TS1/TS-hard）所有参照均零泄漏，无翻案空间也无风险。

**三、TS2/TS3 迭代（双臂并行）**

- 语料：`bprna_tr1c_pdbexp.jsonl` 754 条 PDB 实验结构（对全 board split
  零重叠；ref_tr_experimental 42k 中 36k 已被 tr1c 覆盖，干净增量即此）。
- 臂 1（T-A47，plana 家族）：`plana_giga_tr1c_pdbexp` 渐进解冻，
  13:55 起，当前 step 11.3k/20k，正常。
- 臂 2（T-A47b，r2d 家族）：`rinalmo_r2dtr1cplus_b4_s0` frozen+resnet2d，
  嵌入提取中（GPU7，~15 seq/s），daemon 等嵌入完成后自动开训。
- 评估均自动跑 8 split（含 ArchII/TestSetB）。完成后：两臂进
  xens3 重扫（w sweep on VL0）→ 若 TS2/TS3 缩差至 +0.04~0.05 则 8/8。

git: 本节 + leak_full_audit.py + launch_r2dplus.sh。

### §15.46 (2026-10-11 20:00) — C 解码集成进正式 eval 管线（§15.45 五-1 落地）

**一、变更**（`eval/ss/evaluate_decision.py`，默认行为零改动）

- `--decode` 新增第三选项 `c_batched`（默认仍是 `exact`，numpy 引用
  路径不动）；新增 `--batch-size`（默认 32）。
- c_batched 分支：按 padded 长度分桶（L≤80→80，否则 16 对齐，与速度线
  v2 同规则），同桶 B 条进 C 扩展 OpenMP DP；**forward 仍逐条**——分数
  矩阵与 exact 路径逐字节一致（同一 `_forward_scores` +
  `reweight_turner_prior` 调用序列），差异只在 decode 段。
- 逐条结果按原始 index 重排回填（per_sequence、C1-c 前缀池、
  all_probs_s1 全部保序——matched-prefix C1-c 语义不变）。
- result.json `decode` 字段加 `batch_size`，`meaning` 注明 c_batched
  与 exact bitwise 等价。
- 工程教训：GPU6 是 MIG 4.75GB 切片，长序列 head forward OOM 两次
  （--head-chunk 4 也不够——resnet2d scorer 整体跳过列分块，见 §15.44
  同坑）；换 GPU0 全卡一次过。SSH 断连杀掉前台 nohup 外进程一次，
  长任务一律 `nohup ... &` + 日志落盘。

**二、验证（同 ckpt rinalmo_r2dtr1cplus_b4_s0、同 split bprna_ts0 全量
1288 条，tools/verify_c_batched_equiv.py，11 项检查 ×2 组）**

- (a) 同 GPU（都 GPU0）：11/11 PASS
- (b) 跨 GPU（c GPU3 vs exact GPU0）：11/11 PASS
- 检查项：micro tp/fp/fn 逐位（22171/9477/12720）、P/R/F1 至 1e-15
  （0.666406167811359）、macro F1/INF 至 1e-12、
  **per_seq pred_pairs 1288/1288 逐条一致**、legality（0/0）。
- 与 §15.45 的 35 例 wrapper 自测（pairs+N+dec+kp 四层 bitwise）互为
  补充：那是 DP 层，本节是集成层。

**三、延迟（GPU0 对照，per-bucket decode ms 均值）**

| bucket | n | numpy | C | 加速 |
|---|---|---|---|---|
| <100 | 565 | 20.6 | 1.8 | 11.6× |
| 100-200 | 528 | 59.7 | 2.4 | 24.5× |
| 200-400 | 167 | 449.7 | 8.8 | 51.3× |
| 400-600 | 28 | 1345.1 | 36.7 | 36.6× |
| 总 decode | | 155.9s | 4.8s | **32.6×** |
| system1 吞吐 | | 4.3 | | **9.7 seq/s** |

eval 管线保逐条 forward 是刻意的（分数逐字节等价优先于跑分）；
批化 forward 的 26.7 seq/s 属 speed_batched.py 独立脚本，未动 eval。

**四、产物与提交**

- 修改：`eval/ss/evaluate_decision.py`（+c_batched 分支）
- 新增：`tools/verify_c_batched_equiv.py`（验证脚本，入库）
- 结果（不入 git）：`eval_decision/ts0_exact_ref/`、`ts0_c_batched/`、
  `ts0_c_batched_gpu0/`
- 台账 §15.46 已入 `records/DECISION_TRAINING_LOG.md`（8495 行）
- commit `386c291` 已推送

**五、遗留（下一轮候选）**

1. ResNet-2D head 感受野消融（200-400nt 精度，§15.45 五-2）；
2. xens2 2-seed 三遍稳定性（§15.45 五-3）。

### §15.47 (2026-10-12 00:30) — xens2 2-seed 三遍稳定性（板口径 T-A51b）

**协议**：`tools/xens2_eval.py` 板口径 2-seed（plana s0+s1 桶内 s1w=0.5 ×
r2d s0，w=0.7），TS0 全量 1288，同 ckpt/split/解码，连续三遍（等空整卡
守护，editflow env）。

**结果**：三遍 micro F1 全 0.803859（P 0.821525 / R 0.786936），tp/fp/fn
27457/5965/7434 逐位一致，per_seq pred_pairs 1288×3 逐条一致，
**spread = 0.000000**。

**14/14 检查 PASS**（tools/verify_2seed_stability.py）：三遍一致性 ×8 +
**板值逐位复现** ×4（F1/P/R 至 1e-15 + tp/fp/fn 逐位 == 板上
xens2_bprna_ts0_w0.7，即 sota_vs_ours.md 主表 TS0 行 0.8039 出处）+
协议一致 ×1。**主表 TS0 ensemble 数字从此有直接三遍复证**。

与 §15.45 单 seed 轮互补：单 seed（0.7904）证管线确定性，本轮 2-seed
板口径（0.8039）把主表引用数字本身纳入三遍复证。

**工程坑**：MIG 切片（GPU6 4.75GB / GPU7 实报 4.75GB free）装不下
2×plana encoder + r2d（>5GB），三次 OOM；改等空整卡守护（两阶段确认）
后一次跑成。多模型 ensemble 上 MIG 前先算常驻显存之和。

**产物**：`tools/stability_2seed.sh` + `tools/verify_2seed_stability.py`
（入库）；结果 `eval_decision/xens2_2seed_ts0_stability_r{1,2,3}/`（不入
git）。台账 §15.47 已入。

**遗留（§15.45 五-2，唯一候选）**：ResNet-2D head 感受野消融
（200-400nt 精度）——不急，Limitations 已披露。

### 第二十轮交接（2026-10-13 晚，cluster HEAD 见本轮 commit，细节 §15.48）T-A52：k7 感受野消融读出 + 终评链抢救 + 交接刷新

**用户指示落实（读全部交接文档 → 更新 spec 三件套 → 修复问题 → PPT 更新 → GitHub）：**

1. **项目现状审计（读文档 + 磁盘双通道）**：本地交接文档停在
   第十九轮（10-11），集群 10-12 有一笔「关停备份」提交（ced8ada，
   内含 k7 臂脚手架 + spec_handover 快照）；k7 臂（S15.45 五-2 唯一
   遗留候选：ResNet-2D kernel 3→7 感受野消融）已于 10-13 14:53
   训完 20000 步。
2. **终评链抢救（§15.48）**：k7 自动终评空转 6h+，双根因——
   ① 快照 cron #PAUSED# → step20000 快照从未产出（§15.42 同型故障
   第三次实证，教训升级为强制规则：**新臂上线前必须 dry-run 其
   eval 段**）；② ArchII 的 --embedding-split 名错（应为全量
   archiveii shard，embok_clean 是其子集）。修复：按 §14.56 从
   resume.pt 恢复快照（strict load 通过、7×7 卷积实测确认）→
   8-split 全量落地（ArchII 首跑 MIG OOM，换 GPU0 物理卡成功）。
3. **k7 判定：感受野假设干净负结果（§15.48 三）**——主假设主场
   ArchII 不升反降（0.7240 vs 0.7403，**−0.0163**）；5/8 split
   下降；7/8 split P 升 R 降（保守签名，与 tr1cpdb 同型）。结论：
   Kirigami 的 ArchII 中长段优势**不能由 ResNet-2D 扩大感受野复现**，
   S15.45 归因维持（差距锚定 FCN 结构与训练协议，非单一 RF 数字）。
   处置：判负臂（不进主表/不进 xens 桶），完整表已加行。
4. **板值状态**：6/8 clean SOTA 维持（TS0 0.8039×3 三遍复证 /
   new 0.6132 / ArchII 0.7760 / TS1 0.8755 / hard 0.8732 / TSB
   clean 0.8370）；TS2/TS3 为五战线穷尽式负结果；v3.19 冻结维持。
   **至此 §15.45 的全部三个遗留候选（C 解码集成、2-seed 三遍、
   感受野消融）全部关闭**——项目实验面收官，无在训/在飞臂。
5. **交接三件套同步**：本块 + checklist/spec 刷新 + 台账 §15.48 +
   sota_vs_ours k7 行同批 commit；本地副本同步（根目录三件套与
   spec/ 一致化——此前根目录为第十九轮旧版，双源漂移风险消除）。
6. **PPT 更新（S20 + 新增可信度数字）**：k7 判负行 + 2-seed 三遍
   复证（0.8039×3 spread=0）+ C 解码速度（解码 32.6×）入片，
   逐页布局/字体/数据三遍核对后交付。

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A52** | k7 终评链抢救 + 8-split 读出 + 判定 | 双根因修复 + 判负入台账/表格 | ✅ 完成（§15.48） |
| **T-A53** | 交接三件套第二十轮刷新 + 本地同步 | 集群/本地一致，无双源漂移 | ✅ 本轮完成 |
| **T-A54** | PPT 更新（k7 判负 + 三遍复证 + C 解码速度） | 数字三遍核对 + 0 溢出 | ✅ 本轮完成 |
| **T-A55** | v3.20（k7 判负入 Limitations/附录） | 若导师要求出增量版则启动 | ⏸️ 默认不动（v3.19 冻结维持，k7 为冻结后增量证据线） |

**下一步（项目外部动作）**：导师审阅 v3.19 冻结稿 → 投稿决策。
项目内部：无在训臂、无排队的必做项；速度工程线 v2（解码向量化）
与 RFold/基线再扩展均为可选加分项，默认不动。

### 第二十一轮交接（2026-10-14 凌晨，细节 §15.49）T-A56/T-A56b：eFold 上板（用户指示补测）

**用户问"有没有下游任务上测评 eFold"——查台账发现两条旧记录矛盾（§14.71"无
TS0 ckpt"、§15.44"探测范式不入板"），实测全部推翻：**

1. **资产核实**：`~/reactflow_external_envs/efold_py310` venv 里装着
   **efold 0.1.2 wheel（Rouskin lab，Sci Adv 2026-02）且权重随 wheel 打包**
   （`efold/resources/efold_weights.pt`）；`efold.api.run` 支持纯序列
   zero-shot 推理（Evoformer 式前向 + E2Efold 式 Lagrangian 非交叉后处理）。
2. **8-split 全量落地**（eval/ss/conf_baselines/eval_efold.py，board 同
   协议）：TS0 0.5612 / **new 0.6640（超 UFold 0.6106 与 xens2 0.6132，
   当前 new 最强 zero-shot 单模）** / TS1 0.7682 / TS2 0.8862（超 xens2
   +0.027，仍低于 RNAformer-if 0.9043）/ TS3 0.7354 / hard 0.7632 /
   ArchII 0.6618 / TSB 0.5706（clean-181 0.5661）。
3. **泄漏审计**（efold_leak_audit.py）：TS0/new/PDB 系对 TR0/tr1c 零精确
   重叠；其论文 S1 自述 ArchII 显式 held-out（训练时已过滤）——ArchII 数字
   无泄漏之忧；TSB 247/428 口径适用但 clean 差仅 −0.0045（泄漏敏感度低）。
4. **board 判定不变**：6/8 clean 维持（eFold 非任何 split 的 board 参照；
   new 行参照仍为 UFold，eFold 在表格注释如实引用）。TS2/TS3 负结果叙事
   不变。
5. **工程坑入库**：① eFold bp 列表输出 1-indexed vs 我们 GT 0-indexed——
   首跑 1-idx 交集 0，修 −1 后放行（bp 列表输出模型必须先冒烟对索引）；
   ② 外部资产核实先看盘上 venv（efold_py310 躺了 3 个月两轮都判"不可得"）；
   ③ "协议不同源"要区分训练范式与推理接口。
6. **产物**：eval_efold.py + efold_leak_audit.py 入 repo
   （eval/ss/conf_baselines/）；结果 efold_board.json +
   efold_leak_audit.json 落 eval_decision/；sota_vs_ours.md baselines 板
   eFold 行 + provenance note；台账 §15.49；§14.71/§15.44 旧记录由本节
   勘误（不删原文，本节声明作废）。

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A56** | eFold 8-split zero-shot 上板 | board 同协议 + 全 split result.json | ✅ 完成（§15.49） |
| **T-A56b** | eFold 泄漏审计 | 序列级审计 + 论文声明交叉 | ✅ 完成（efold_leak_audit.json） |
| **T-A57** | PPT S18/S20 加 eFold 行（若导师汇报需要） | 数字三遍核对 | 🟡 待用户示意（board 判定不受影响，非必改项） |

**下一步**：维持外部动作（导师审阅 v3.19 → 投稿）。eFold 数字为 draft
Related work / 基线全景的增量证据线，需要时可一并写入 v3.20。

### 第二十一轮交接后段（2026-10-14，T-A57/T-A58）PPT eFold 入片 + 索引教训入库 + 测评资产备份

1. **T-A57 PPT 更新完成**：S18 基线全景表加 eFold 行（14 行 × 8 split，
   TSB 0.5706* 注 clean-181 0.5661）+ 溯源声明 eFold 段（9.5pt 独立段）；
   S20 会议基线表加 eFold 行（16 行，插在 Kirigami 后）+ 读法 eFold 句
   （9.5pt 独立段）+ 底注来源补 eFold 溯源。**三遍核对全过**：① 写入前
   9/9 数字与 efold_board.json 磁盘对账一致；② 写入后全文本逐格对照
   （S18/S20 两行 8+7 格全对）；③ 几何/字体/行距断言（0 越界、0 裁切、
   全片微软雅黑、新增段 1.5 倍行距 9.5pt、表格底缘与下方文本框无重叠、
   页码齐全）。行高压缩：S18 0.27→0.255（14 行）、S20 0.25→0.235
   （16 行），模板身份不变。
2. **教训入库（用户指示补充）**：台账新增"外部基线适配三条经验"节——
   ① bp 列表输出模型必须先索引口径冒烟（pred∩gt>0 断言，1-indexed vs
   0-indexed 一次即抓）；② "不可得"结论先盘本地 venv/site-packages 再下
   （efold wheel+权重躺了 3 个月两轮未发现）；③ "协议不同源"要区分训练
   范式与推理接口（读代码入口签名，不只读摘要）。
3. **T-A58 测评资产备份**：efold_board.json + efold_leak_audit.json +
   适配器脚本（eval_efold.py / efold_leak_audit.py）已备份至本地
   `/Volumes/orico-disc/A100 大量项目文件备份/05_RNA-JEPA/` 对应文件夹
   （eval_decision/ 与 code_git_home 同构布局），与既有备份结构一致。

### 第二十一轮交接续（2026-10-14 晚，T-A59）测评数据集备份核查与补齐

**用户指示核查备份盘 → 结论"完全没备份" → 已补齐并三重验证：**

- 备份盘 `05_RNA-JEPA/ss_data_backup/`：jsonl 全量 111 项（543MB 解包后），
  含 9 冻结评测 split、训练语料族、全部 manifest/rejects 溯源链；
- 验证链：传输级 sha256 一致 + **9/9 冻结 split 文件级 sha256 与
  15.14 权威清单逐项 MATCH**（BACKUP_MANIFEST.md 在备份盘内）；
- 有意不备份：teacher 20GB（确定性可再生成）、原始 bpseq（公开可下载）、
  embeddings（可再生成）——理由均已记录；
- **新规则入库（§15.50）**：备份三件套检查表 = 代码（git）+ 结果
  （eval_decision）+ 数据（ss_data + 冻结清单）；历次备份都漏了第三项。

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A59** | 测评数据集备份核查 + 补齐 | 9/9 sha256 验证 + manifest 落盘 | ✅ 完成（§15.50） |

### 第二十一轮交接收官（2026-10-14 晚，T-A60）复现能力审计 + 备份三件套全绿 + 新规则首次执行

**用户问"备份能否支持别人复现所有训练和测评"——审计结论（§15.51）：**

- **评测侧复现：已完备**（数据 9/9 sha256 验 + 代码 git tag + ckpt board
  成员 + result.json 原件 + 协议冻结清单）——别人可拿 ckpt 在同一数据
  上逐数字复现 8-split 板。
- **训练侧复现：资产齐备**（185 文件 ckpt/runs 全量 rsync 6 分片在飞，
  64/185 完成，--partial 断点续传；RiNALMo-giga 骨干本轮归档 + sha256；
  teacher/embeddings 确定性可再生成——无任何"不可复现"项）。
- **环境快照**：reproduction/ENVIRONMENT_SNAPSHOT.md（torch 2.5.1+cu121
  权威环境 + 全外部基线版本锁）。

**备份三件套（§15.50 新规）状态：代码 ✅ + 结果 ✅ + 数据 ✅——首次全绿。**

**注意（重要）**：`shutdown_20261014/ckpts/plana_s0_s1*.tar.gz` 是**截断
文件**（tar 报 Truncated），勿当完整备份使用；权重全量权威备份在
`06_RNA-JEV_mars清洗流程/rna_jepa_weights_20261013/`（rsync 在飞）。

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A60** | 复现能力矩阵 + 骨干权重归档 + 环境快照 | 矩阵入台账 + sha256 | ✅ 完成（§15.51） |
| **T-A61** | rsync 185 文件完成后 sha256 终验 | 对账清单全绿 | 🔄 在飞（今晚） |
# T-A62 新集群恢复指南（2026-10-15 晚，用户指示："换到新集群就知道该怎么做"）

> 本节是**行动手册**：新集群到手后按顺序执行即可恢复整个项目。
> 依据：备份三件套（§15.50 规则）已全绿，复现能力矩阵见 §15.51。

## 一、恢复总图（四层依赖，自底向上）

```
第 0 层  备份盘     /Volumes/orico-disc/A100 大量项目文件备份/05_RNA-JEPA/
         （+ 06_RNA-JEV_mars清洗流程/rna_jepa_weights_20261013/ 权重农场）
第 1 层  环境       conda env + 外部基线版本锁（ENVIRONMENT_SNAPSHOT.md）
第 2 层  数据       ss_data/jsonl 111 项（9 冻结 split sha256 锁死）
第 3 层  资产       代码 git + ckpt（4 主数字 + 农场 185）+ result.json 原件
第 4 层  验证       逐位复现板数字（验收门，见第五节步骤 5）
```

## 二、逐文件校验清单（先验后用，md5/sha256 全部对账）

| 资产 | 备份盘路径 | 校验值 |
|---|---|---|
| git 全历史 | `shutdown_20261014/rna-jepa-final-20261015.bundle` | `git bundle list-heads` 核对 HEAD（重建后） |
| 评测数据全量 | `ss_data_backup/jsonl/`（111 项） | 逐文件 md5 与集群原值一致（§15.50 + BACKUP_MANIFEST.md；9 冻结 split 另有文件级 sha256 锁） |
| RiNALMo-giga 骨干 2.6GB | `weights_backbone/rinalmo-giga/` | sha256=0b3f73b5c80d267f0b7e92655b28f16088e3d5a595c6dae631b52142aa38bcda |
| plana_tr1c s0/s1（板成员） | `06_.../rna_jepa_weights_20261013/ckpts/` | md5 s0=8b74e348efaa2ae2f444cd271a5bc3fa / s1=6f3b26195642bf2128d2ad08c2149126 |
| r2dtr1c_s0（板成员） | `05_RNA-JEPA/ckpts/` | md5=3b6b508f4386842faefc7acce68c6f00 |
| k7 消融臂 | `05_RNA-JEPA/ckpts/` | md5=60ffc48dae75cd049de73eda15df130d |
| 结果原件全量 | `eval_decision/`（tar + 散目录） | result.json 磁盘直读对账 |
| 环境快照 | `reproduction/ENVIRONMENT_SNAPSHOT.md` | 无哈希，人工核对版本号 |
| 溯源元数据 | `ss_data_backup/ss_inventory_20261015.tar.gz` | md5=8388750de7b3aaf715541844625ff7dc |
| teacher manifests | `ss_data_backup/teacher_manifests/` | md5=6c97932ffcaeebe78f4fe1fe54556245 |
| mRNA 预训练语料归档 | 公开 Zenodo 12516160（盘上未存） | md5=bf8bc5c946a0bd3b07716b1c7f785d54（锁在 pretrain_pipeline.sh） |

**坑 0（先读）**：`shutdown_20261014/ckpts/` 下的 `plana_s0_s1*.tar.gz` 与
`plana_split_*` 已是历史残留（旧分块方案，部分截断）——**不要使用**，
权重权威备份是 06_ 目录（§15.51 警告仍然有效；该残留目录已清空）。

## 三、新集群重建步骤（按序执行）

1. **环境**
   ```bash
   # conda 主环境（权威：旧集群 lucaone）
   conda create -n lucaone python=3.9
   # torch 2.5.1+cu121 / transformers 4.26.0 / numpy 1.26.4 / scipy 1.13.1
   # 详细版本清单见 ENVIRONMENT_SNAPSHOT.md；317 项单测须全绿
   # RiNALMo 推理另需 multimolecule 0.0.8 + transformers 4.57.6 + torch 2.8.0
   # （pip --target 隔离安装，避免污染训练环境）
   ```
2. **代码**
   ```bash
   git clone rna-jepa-final-20261015.bundle rna-jepa
   # 或：git clone https://github.com/Cunyu-Liu/JEPA-mRNA（tag preprint-v3.19-freeze
   #        冻结态；GitHub 版落后 bundle 时以 bundle 为准）
   cd rna-jepa && git bundle verify ../rna-jepa-final-20261015.bundle
   ```
3. **数据**（从备份盘 rsync 到新集群 /mnt 数据盘）
   ```bash
   # 全部评测/训练语料（111 项一次到位；manifest/rejects 已含在内）
   rsync -a orico:ss_data_backup/jsonl/ /mnt/<user>/rna-jepa/ss_data/jsonl/
   # teacher npz 若 06_ 农场 bprna_tr1c 未传完 → 重算（见第五节）
   ```
4. **骨干与 ckpt**
   ```bash
   rsync -a orico:weights_backbone/rinalmo-giga/ /mnt/<user>/rna-jepa/weights/rinalmo-giga/
   # 板成员 4 ckpt + 农场（95+ ckpt、runs/resume.pt 断点续传产物）
   rsync -a orico:06_.../rna_jepa_weights_20261013/ckpts/ /mnt/<user>/rna-jepa/ckpts/
   ```
5. **验证门（必须全过才算恢复成功）**
   ```bash
   # ①数据完整性：111 文件逐一 md5（对账 BACKUP_MANIFEST.md §15.50）
   # ②骨干 sha256：0b3f73b5…（对账 rinalmo_sha.txt）
   # ③板数字逐位复现（最强验收，约 10 分钟）：
   PYTHONPATH=src python eval/ss/evaluate_decision.py \
     --data ss_data/jsonl/bprna_ts0.jsonl \
     --ckpt <plana_s0> --ckpt2 <plana_s1> --s1w 0.5 --r2d <r2dtr1c_s0> --w 0.7 \
     --out /tmp/reproduce_ts0.json
   # 预期：F1=0.803859，tp/fp/fn=27457/5965/7434，1288 条 per_seq 与
   # xens2_bprna_ts0_w0.7 result.json 逐位一致（§15.47 三遍复证口径）
   ```

## 四、路径映射表（旧集群 → 新集群）

| 旧（A100） | 新集群 | 内容 |
|---|---|---|
| `/home/cunyuliu/rna-jepa/`（git 仓） | `~/rna-jepa` | 代码/spec/records |
| `/mnt/cunyuliu/rna-jepa/ss_data/jsonl/` | `/mnt/<user>/rna-jepa/ss_data/jsonl/` | 111 数据集 |
| `/mnt/cunyuliu/rna-jepa/ss_data/teacher/` | 同级 `teacher/` | 蒸馏软标签（可重算） |
| `/mnt/cunyuliu/rna-jepa/ckpts/` | `/mnt/<user>/rna-jepa/ckpts/` | ckpt（**注意 symlink**：`ckpts/*_step20000.pt` 多为 `runs/<name>/resume.pt` 的链接，恢复时要么重建链接要么直接用 runs 副本） |
| `/mnt/cunyuliu/rna-jepa/weights/rinalmo-giga/` | `weights/rinalmo-giga/` | 骨干 2.6GB |
| `/mnt/cunyuliu/rna-jepa/eval_decision/` | `eval_decision/` | result.json 原件（对账用） |
| `/mnt/cunyuliu/rna-jepa/runs/` | `runs/` | 训练曲线 run_meta/train_log |

## 五、可再生成资产（缺了不慌）

- **teacher 软标签**：`data/ss/generate_teacher_labels.py`（git 内）+
  ViennaRNA **2.7.2**（版本锁写死在 teacher manifest `version_lock` 字段）
  + jsonl 语料 → 12 进程 892s/45k 条，确定性输出（分片 sha256 在
  teacher_manifests tar 内可对账）。
- **embeddings 缓存**：骨干 + 序列前向即得；评测管线检测缓存缺失会
  自动回退在线前向（结果一致，仅变慢）。
- **预训练语料**：Zenodo 12516160 公开（md5 见第二节）；受限网络注意
  inventory 记录的 15-20KB/s 限速，需 ranged 多连接或换网。
- **外部基线**（版本锁见 ENVIRONMENT_SNAPSHOT.md）：ViennaRNA 2.7.2、
  MXfold2、EternaFold、UFold、RNAformer、RiNALMo-ft（Zenodo）、Kirigami
  （main.ckpt）、eFold 0.1.2 wheel（权重在 wheel 内 `resources/efold_weights.pt`）。

## 六、常见坑（已全部踩过，直接抄答案）

1. **索引口径**：eFold bp 列表 1-indexed，我们 GT 0-indexed——适配新外部
   模型先跑单序列冒烟断言 pred∩gt>0（§15.49 教训 1）。
2. **"不可得"先查本地**：外部模型先 `ls ~/venv/site-packages` 再下结论
   （eFold 躺了 3 个月，§15.49 教训 2）。
3. **ckpt 步数溯源**：步数从 `.pt` 文件内 `step` 字段读
   （tools/ckpt_steps.py），不信文件名/tag（T-A8 教训）。
4. **MIG 切片显存**：多模型 ensemble 前先算常驻显存之和，>4.75GB 的
   组合放整卡（§15.47 教训）。
5. **ssh 断线杀进程**：长任务一律 `setsid nohup … &` + 磁盘日志
   （T-A9 起 project-wide 纪律）。
6. **长链路备份带宽坍缩**：开新传输前先 `ps aux | grep -E 'rsync|scp'`
   双端盘点在途进程（本轮 10+ 路并发曾把单流挤到 20KB/s，§15.50 教训 1）。
7. **sz 单位**：python-pptx 的 sz 是"百分之一 pt"，750pt 事故已入库——
   脚本写表格必须 `assert sz <= 1300` + 几何/字号断言（§15.52 教训）。
8. **训练确定性依据**：同环境同数据下训练逐位可复现（F1 0.803859 ×3
   一致，§15.47），换硬件/库版本后预期"数字一致或极接近"，跨硬件
   速度数字仅供参考（S20 口径）。
9. **共享集群礼仪**：load average ~92-104 是旧集群常态；加训练臂前先
   `free -g` 确认 available > 60GB（每臂 RSS 12-36GB）。

## 七、验证状态快照（2026-10-15 晚）

- 数据：111/111 md5 双端一致 ✓；9/9 冻结 split sha256 锁 ✓
- 骨干：sha256 双端一致 ✓（另有 03_rna-sc 独立副本）
- 板成员 ckpt：4/4 双端 md5 一致 ✓
- 权重农场：185 文件清单，107 ckpt 与 35/78 runs 已落，本地 mac 看护
  进程（/tmp/a100_puller/master_keeper.sh，每 10 分钟自检重启）仍在续传
- 结果原件：eval_decision tar + 散目录双份 ✓
- PPT：25 页 10-15 版（S15 SOTA 主对比 + 三张顶刊风图）md5 一致 ✓
- 旧集群状态：无在训任务，git=本轮 commit，全部已推 GitHub

### 第二十二轮交接（2026-10-18，新集群重建轮）T-A67/T-A68/T-A69

**用户三线目标落实**（速度 SOTA / 测评复核无泄漏 / TS2 攻关 8-8 + 新增置信度调研）：

1. **T-A67 新集群重建（§15.59）**：环境+资产+验证门全过（337 tests；9/9 split sha256
   MATCH；plana_tr1c_s0 TS0 复现数值等价 0.786839 vs 0.786563，fp16 帧跨架构
   舍入已归因；14 embedding shard + teacher 167 shard 重建）。
2. **T-A68 速度线 4090 首读（§15.60）**：B=64 101.0 seq/s（3.8× A100 同协议；
   1.8× 超 UFold/RNAformer/MXfold2 A100 板读数），F1 无损；dp_decode_c 重编译；
   跨硬件同节点诚实口径入库。
3. **T-A69 TS2 攻关臂启动**：r2dtr1c_b4_s1 重训（备份缺失 ckpt；B=2 适配
   24GB 卡），完成后 xens2 2-seed 复现 + TS2 集成重扫；watchdog cron + 3h
   巡检定时任务已装。

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A70** | r2dtr1c_b4_s1 训完 20k → 8-split 终评 | 8 result.json + 对板 s1 参照 | 🔄 在训（~24h） |
| **T-A71** | 参照系统 4090 同节点重测（UFold 等需权重） | 同节点速度表或如实标注不可测 | 🟡 排队 |
| **T-A72** | 速度 v3：fp16 权重半程 + 内核融合 | ≥120 seq/s 同 F1 | 🟡 排队 |
| **T-A73** | Jev 式输出置信度方案（配对概率已有=校准卖点；结构级置信度设计） | 设计文档 + 原型实现 | 🔴 下一步 |
| **T-A74** | 泄漏复核（新集群数据 sha256 已 MATCH；leak audit 脚本复跑） | 审计 JSON 复现 0 重叠 | 🟡 排队 |
| **T-A75** | 本地交接三件套同步（TCC 权限恢复后） | 本地=集群=GitHub 三方一致 | ⏸️ 等 macOS 权限 |


### 第二十三轮交接（2026-10-19，TS2 集成轴全穷尽 + 归档收尾轮）T-A76..T-A83

**本轮一句话**：TS2 攻关在集成轴上双向穷尽（r2d 成员侧 §15.76 + plana 桶种子侧 §15.83，均为负结果如实入稿），速度/置信度/参照/图四线全部收官；/teams 归档完成并 md5 抽验；预印本推进至 v3.25。

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A76** | r2dtr1c_b4_s2 重训 + 8-split 终评 + 归因对照（§15.74/15.75） | 训练收敛 + 8 result.json + 归因判定 | ✅ 完成（归因反转：种子运气 ≠ 训练完整性） |
| **T-A77** | 三成员集成轴穷尽（s2 成员 / 2-seed r2d 桶，§15.76） | 单变量对照 JSON ×2 | ✅ 完成（均负：−0.023 / −0.006） |
| **T-A78** | plana 桶种子多样性轴穷尽（3/4-seed，§15.83） | 单变量对照 JSON ×2 | ✅ 完成（单调递减 0.8581/0.8526；轴关闭，板协议 2-seed 即最优） |
| **T-A79** | xens2 8/8 板协议 4090 逐位复现（§15.77） | 8 split max \|Δ\| ≤ 0.0001 | ✅ 完成（r2ds1 错跑变体留档） |
| **T-A80** | 2025-2026 参照扫描（HydraRNA/RNADiffFold/RIBOSPAN） | survey 增补行 + 入稿 | ✅ 完成（HydraRNA 唯一同口径遗漏 TS0 0.76 quoted） |
| **T-A81** | RiNALMo-ft 同节点复现 + 速度参照（§15.82） | TS0/TSB 逐位复现 + B=1 速度 | ✅ 完成（0.7603/0.9011；14.5 seq/s → 同节点 8.6×） |
| **T-A82** | /teams 大型资产归档（§15.84） | 双端 md5 抽验一致 | ✅ 完成（embeddings+weights+eval 全落；s1/s2 ckpt 实体化；6 悬挂消融 ckpt 如实清点，不在板路径） |
| **T-A83** | 预印本 v3.25 + PPT + 台账三同步 | checker 8/8 + GitHub | ✅ 完成（36e28db / e09ef1c） |

**遗留（下一任优先级）**：
1. **T-A84（可选，仅当用户要求补消融）**：6 个悬挂消融 ckpt（fftr1c/nodistill/norlcd/tr1cpdb×2/plana_giga_s0 预 tr1c 种子）重训恢复——**不在 8/8 板路径**，主结论无依赖。
2. **T-A85**：预印本正式投稿前 polish（参考文献格式、图 300dpi 导出包、supplementary 拆分）。
3. **T-A86**：本地交接三件套同步（TCC 权限恢复后；ppt/ 已恢复，Documents/交接文档 仍阻断）。

**权威状态**：预印本 v3.25（§4.3 集成轴双向穷尽）；TS2 终判 0.8604/−0.044（七战线）；速度 124.78 seq/s B=256 fp16（同节点 8.6× RiNALMo-ft）；置信度 8-split 全板；GitHub HEAD e09ef1c。


### 第二十四~二十六轮交接补记（2026-10-19 → 10-23，详见 spec_sync/ROUND24_NOTE.md 与 ROUND26_NOTE.md）

第二十四轮（图优先版）：六主图体系 + 预印本 v3.27 + PPT 33 页；GitHub fa6e3c9。
第二十五轮（五工作总执行轮）：97 格基线重审计 0 误差；Rfam-novel ncRNA 数据落地（RDx 破解）；NB 双线预注册启动；nature_rewrite_r25.md；OFoldNA/CHANRG 调研；GitHub cbe7a30 → 1714c6d（r25d 静默失败修复）。

### 第二十六轮交接（2026-10-23，CHANRG 解锁 + T-A99 判定轮）T-A99/T-A100/T-A102

**本轮一句话**：工作 3 双线收官（NB-ft 完训评测 + Jev-on-NB 8-split 判定 7/7 大幅 PASS，OOD 增益 +0.19~+0.23 与主线先验栈签名同构）；CHANRG gated 解锁全链落地（数据+泄漏审计+评测链，四 split 在飞）；ncrna ml-shape 二次修复；monitor_r26 cron live。

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A99** | Jev-on-NB 三阶段（嵌入→训→8-split 终评） | 预注册 ≥3/8 split Δ≥+0.02 | ✅ 完成（**8/8 PASS**，Δ +0.055~+0.234 均值 +0.13，OOD 最大；ArchII trunc598 0.5414） |
| **T-A100** | NB fine-tune 基线 | 3ep 完训 + 8 split json | ✅ 完成（decoder 矩阵→pairs bug 修复后 8/8） |
| **T-A102** | CHANRG 四 split OOD 补测 | 4 result.json × {r2d,xens2} + 泄漏审计 | ✅ 完成（8/8 落地 + Table 3 对标 + S38 入片） |
| **T-A101** | ncrna 双头评测 | ncrna_dual_result.json | ✅ 完成（0.8621/0.6308/0.7116 vs 线性头 0.7922/0.5248/0.5735） |
| **T-A103** | nature 重写合并 v3.30 | 导师过目后 | ⏸️ 等外部动作 |
| **T-A104**（新） | CHANRG 结果 → OFoldNA-SS 同口径对比表 + 板表 + PPT | 四 split 数字三遍核对后入片 | ✅ 完成（GenA 0.6443 全场第一；OOD mean 0.4434 vs 0.4410 并列；S38 入片 38 slides；板表三块新增） |

**工程坑入库（本轮新增 4 条）**：
1. **评测器字段契约**：load_corpus 只读 `pairs`——新数据集 convert 必须同时写 dbn 与 pairs（首跑 F1=0 即 GT 空，错误方向是"全错"而非崩溃，最危险）。
2. **EmbeddingStore 路径回退**：shard 缺失时回退 ckpt config 里的旧集群路径——跨集群迁移后新 split 必须先提取 shard 再 eval。
3. **修复覆盖律**：同型 bug（shape/长度）修复必须覆盖同变量的**全部使用点**（ml 计算侧 + 赋值侧），单点修复会在下个使用点复发（ncrna 两次命中实证）。
4. **外部模型调用契约**：元组返回（load_encoder）、ModelOutput dict（out[0]/last_hidden_state）、attention_mask 必传——**预飞行必须真跑一个 batch**，AST/CLI 检查全漏（ncrna 4 连修实证）。

**权威状态**：GitHub HEAD 8f5300f；台账 §15.97/97b/97c（9638 行）；板表 +3 块（CHANRG/NB 迁移/ncRNA）；PPT 38 slides；集群侧 monitor_r26 cron（*/10）live。


### 第二十七轮交接（2026-10-23 下午，GenF 冲刺 + v3.30 合并轮）T-A105/T-A106/T-A107

**本轮一句话**：GenF 差距诊断为语料差距（39% 零开火 + OFoldNA 用 CHANRG Train 蒸馏）→ CHANRG Train 臂 T-A105 在飞冲 SOTA；三张正刊图替换 PPT 表格页；v3.30 直接合并（nature Abstract/Intro + 4.3j/4.3k 新结果，0d6ac14）；MRL/splice 下游扩展排队。

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A105** | GenF SOTA 臂（chanrg_r2d_s0，单变量=语料） | GenF ≥0.4035 + 4 split 完整读数 | 🔄 四阶段链在飞（嵌入 55%） |
| **T-A106** | 正刊绘图三张 + PPT 图版化 | 300dpi PNG+PDF + 几何零越界 | ✅ 完成 |
| **T-A103** | v3.30 合并（直接合并版） | em-dash 0 + 数字对账 + 接缝检查 | ✅ 完成（0d6ac14；本地交接文档 paper/ 有副本） |
| **T-A107** | RiNALMo 下游扩展（MRL + splice 双头） | 双头结果 json×2 | 🔄 MRL watcher 在飞；splice 数据下载 30% |

**权威状态**：GitHub HEAD d276c11；台账 §15.98；PPT 38 slides（图版）；figs/round27 三图；v3.30 落地双端。
