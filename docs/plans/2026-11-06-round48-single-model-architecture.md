# Round 48：PRISM 独立单模型架构攻关合同（2026-11-06）

## 1. 目标与既有事实

本轮只优化用户指定的三条架构路线，顺序固定为：structRFM+JEV → RIPE 思想迁移 → JEPA 式 scorer。目标是独立单模型在主去泄露八 split 全板 SOTA，Rfam/CHANRG 尽可能多领先，raw-sequence 端到端速度 SOTA，结构质量排序与 risk–coverage 最优，并改善 200–400 nt 与 >400 nt 的长序列表现。目标不是预设结论；未达标必须保留负结果。

既有权威事实：

- 13 系统 own-clean 104/104 主榜已完成，xens2 领先 5/8；bpRNA-new、TS2、ArchiveII 未领先。xens2 是三个不同权重骨干的集成，不满足独立单模型。
- 单 plana 学生三种子已收敛，clean 8/8 已测但未达 SOTA；VL0 融合与混合 NLL 候选均被否决，不返工。
- 注册概率证据 128 格已完成，多个系统优于 xens2 的 ECE 或结构排序，不能宣称最佳置信度。
- xens2 自适应批次为 14.1949 seq/s，仅 1.045× 且非 SOTA；双 stream 已按用户决定暂停。
- Rfam：PRISM=0.6514、BPfold=0.7872；CHANRG：PRISM Test/GenA/GenC/GenF=0.5656/0.6443/0.3793/0.3066，OOD mean=0.4434。

## 2. 总体不变量与验收口径

1. 主模型必须是一个骨干、一个任务头、一个 checkpoint；不得集成不同权重、不同 seed 或测试时增强。辅助 EMA teacher 只允许存在于训练期，推理期不得保留第二次前向。
2. 训练、验证、模型选择只用 TR1c 与锁定 VL0=196；测试集不参与超参、早停、校准、融合或候选选择。
3. 主榜为 TS0、bpRNA-new、ArchiveII、TS1、TS2、TS3、TS-hard、TestSetB 的各自 own-clean 数据与 strict micro-F1；同名同序列同 GT 的 matched-clean 只作附加敏感性。
4. A/B/C 每个冻结候选均须评测：A=TS0/ArchiveII/TestSetB；B=bpRNA-new、Rfam12.3–14.10、CHANRG Test/GenA/GenC/GenF；C=TS1/TS2/TS3/TS-hard。Rfam 与 CHANRG 分表，PDB 小样本保留 n，不用无边界的总均值掩盖弱格。
5. 每个新训练候选至少三个训练种子，均须 validation plateau、best checkpoint、明确停止原因和独立 CUDA QA。smoke、proxy、训练集结果和单 seed 只作过程证据，不进入最终科学结论。
6. 所有训练与 GPU 验证强制 CUDA；发现 CPU 静默降级立即停止并保存环境、命令、日志和异常样本。正式训练不设固定步数终点，以验证集平台与 patience 早停为准。
7. 同一配置三次确定性推理；新训练模型以三个训练种子的 mean±SD 为主。报告 F1/INF/P/R、ECE/Brier/NLL、Spearman、AURC/risk–coverage、长度桶与峰值显存。
8. 速度只在精度候选冻结后测：RTX 4090 独占、全 TS0 raw-sequence、含 tokenizer/骨干/任务头/解码、三次无 resume；与 eligible baseline 同机同协议。并行开发测速只作 profile，不作 SOTA。
9. 长度主桶为 ≤100、100–200、200–400、>400 nt；同时报告长程配对跨度 ≥30/≥100 nt。不得截断、滑窗拼接或删除长序列来制造“长序列优势”。
10. 非计时任务按实测显存并行，目标高利用而不 OOM；正式速度测试例外，必须独占。每阶段完成的是“合同验收与明确裁决”，不是要求每条路线都涨点才能进入下一条。

## 3. 共用对照与 A/B/C 消融矩阵

每条路线先以 VL0 选择配置，再冻结并进行完整 A/B/C。至少包含：

| 层级 | 必需臂 | 回答的问题 |
|---|---|---|
| 宿主对照 | 同训练配方 RiNALMo 单模型+原 JEV 头 | 新骨干/scorer 是否优于既有单模型 |
| 主候选 | 用户指定架构+同一 JEV 物理先验、mask、DP 与损失 | 架构净效应 |
| 参数/算力对照 | 近参数量 MLP 或等 FLOPs 训练对照 | 是否只是容量或预算增加 |
| 机制消融 | 各路线预注册的通道开关与随机杀手对照 | 指定机制是否必要 |
| 解码对照 | 同一分数下 DP→greedy、Turner=0（无需重训） | 增益是否来自求解器或物理先验 |
| 最终单模型 | VL0 锁定的最佳单种子配置，另跑三种子 | 独立单模型主榜与泛化 |

统计以逐序列配对 bootstrap 95%CI 为主，主八 split 做 Holm 校正；小 split 只作描述性证据。架构筛选不得反复读取测试榜后改配置。

## 4. 路线一：structRFM+JEV（P1）

### 4.1 已核实机制与风险

- 官方 Nature 稿：12 层 BERT、hidden=768、12 heads，预训练最大长度 512；SGMLM 用配对引导 mask 与动态比例，MUSES 融合 ViennaRNA/CONTRAfold/BPfold。下游可使用 nucleotide-wise 与 pairwise 表征。
- 官方代码：`heqin-zhu/structRFM`，本轮固定 commit=`f0c1add19f8dfc786f16bc4abb7aa775dbadddbf`；本地既有 `tools/structRFM-master` 与该文件哈希一致，不重复改造两份代码。
- 官方预训练权重约 300 MB，只能从官方 release 加载并记录 SHA256 与严格张量加载结果。不得直接使用在 bpRNA-1m/RNAStrAlign 上已监督微调的 SSP checkpoint 作为干净主模型初始化。
- Nature 版预训练语料为 Zenodo 18272040，`RNAcentral_512_MUSES_connects.csv.tar.gz`，5,654,000,185 bytes，MD5=`651db6dc7b6d6aadece06a252831d1fe`。旧记录 16754363 是另一版本，不能混用。
- structRFM 预训练覆盖 RNAcentral v24，主测试序列可能存在预训练接触。必须对 21M 语料重新计算精确与 80% identity 重叠，生成 structRFM 专属 clean；TR1c downstream clean 不能替代预训练 clean。
- 原文已报告 23S 长序列退化与 512 nt 限制。全长评估不得直接扩大绝对位置 embedding；若 >512 输入无法严格加载，标为工程边界并进入 RIPE 路线解决，不能随机扩展后冒充官方模型。

### 4.2 实验臂

- S0：既有 RiNALMo 单模型+同一 JEV 头，复用已验收权重/数据；缺哪个协议才补哪个。
- S1：structRFM 冻结骨干+JEV 头，负责分离表征质量与训练成本。
- S2：structRFM 全参微调+JEV 头，同一 TR1c、VL0、目标、batch 顺序、DP/Turner 与早停规则。
- S3：近参数/等 FLOPs 对照，解释 S1/S2 是否仅由优化预算差异造成。
- **S3 冻结定义（在 S2 A/B/C 读出前）**：全参 structRFM+JEV，仅将 ResNet2D scorer 换为 PairMLP-490，不使用 EMA、latent 辅助损失或 RIPE；其余数据、batch 顺序、目标、解冻、学习率、VL0 和 patience=6 与 S2 相同。实测独立 scorer 参数量 PairMLP=304,547、ResNet2D=304,321，差 +226（+0.074%）；608,591 是包含共有组件的原 ResNet 整头参数，不能误作独立 scorer 参数。共有 pair representation、type head、校准和 Turner 参数保持一致；纯 PairMLP 不保存 EMA teacher，S3 整头 608,817，对照整头 608,591，差值同为 +226。S3 三个训练种子自然早停后复用 13 split×3 repeat 全参链。
- S3 只支持近参数 scorer 归因，不冒充全模型等参数：S1 仅头部可训练；S2/S3 解冻12/12 Transformer blocks，实际训练85,054,464个骨干参数，其余996,096个embedding/pooler等参数保持冻结，必须分开报告。保存各 run 的长度/步数账，并在冻结长度网格实测 forward FLOPs；训练期预算比较不得写成精度冻结后的独占速度结论。
- 若官方 pairwise attention 接入：只能作为 S4 独立臂，与 S2 分开，不与骨干替换混为一个归因结论。

### 4.3 出门条件

严格权重加载、>512 边界、预训练污染审计、CUDA smoke、三种子收敛和 A/B/C 全量结果全部完成。裁决为晋级、保留为对照或否决；即使未超 RiNALMo，也进入路线二，不重复换测试集调参。

## 5. 路线二：RIPE 思想迁移（P2）

### 5.1 迁移原则

迁移四通道思想而非照搬尚未在本任务验证的大模型训练配置：

1. `K_pair`：log 距离基覆盖多尺度，远端采用连续饱和平台，不把训练集配对距离分布写入初始化。
2. `K_helix`：{11,22} 双周期与可学习相位，提供有界长程几何先验。
3. `Phi_far`：`−ν log(1+Δ/s)`，ν>1，避免 ALiBi 远场过硬衰减；与 log-ALiBi 共用实现。
4. `C·G`：对称互补内容偏置，Δ≥4；学生实际可见 token 计算，特殊 token/PAD 为零，不能由“偏置对称”推断“配对互惠”。

优先注入 JEV pair representation/scorer，使原 BERT 的 512 位置限制不被随机扩展掩盖；若在 structRFM attention 内注入，必须单独作为另一臂并重新核验位置边界。

### 5.2 实验臂

- R0：路线一最佳单模型、无 RIPE。
- R-log：仅 log-ALiBi/Φ_far，排除“换 log 形态就够”。
- R1：K_pair+Φ_far。
- R2：R1+K_helix。
- R3：完整 RIPE。
- R5：R3+预注册随机周期，杀“任意周期都有效”。
- R6：R3+预注册对称随机互补表，杀“任意内容偏置都有效”。
- 等参数 MLP 对照；QK-norm 若改变，必须单列 QK-norm 对照，不能归给 RIPE。

合成门先验证边界连续、ν>1、Δ<4 内容项为零、随机表/周期可复现、>512/1024 无 NaN、梯度确实到达门控。通过后三种子训练并做 A/B/C；长序列与长程配对是主判定，短序列平均提升不能替代。

## 6. 路线三：JEPA 式 pair scorer（P3）

### 6.1 选型

采用 I-JEPA 思想的 **Pair-JEPA**，不是视频 V-JEPA，也不恢复旧区域均值 JEPA 预训练。目标是替代 ResNet2D 的配对打分，同时保留 JEV 的合法 mask、Turner 残差和 DP。上下文编码器预测被掩候选配对的 latent，训练期 EMA target encoder 提供 stop-gradient 目标；推理只保留一个轻量 pair scorer，不做 teacher 前向。

旧区域均值目标在 60–100 step 饱和，因此禁止全区域均值余弦作为唯一目标。新目标必须是被掩 pair/token 级 latent，并记录 target/prediction 方差、有效秩和随机 target 对照，防止常数坍塌。

### 6.2 实验臂

- J0：路线二最佳单模型+ResNet2D。
- J-mlp：近参数、无 2D ResNet 的对称 pair MLP/低秩双线性 scorer。
- J1：Pair-JEPA+直接配对 logit，训练期 EMA，推理单 scorer。
- J1-noEMA：同容量固定/在线 target 对照，分离 EMA 的贡献。
- J-random：随机或打乱 latent target，杀“任意辅助损失都有效”。
- 速度对照：J0/J-mlp/J1 同精度门下的 forward-only 与 raw-sequence profile；最终只以独占全 TS0 为准。

若 Pair-JEPA 不超 J-mlp，结论只能是“未优于简单 scorer”，不能宣称 JEPA 有效。若速度提升来自删去 2D CNN，应归因于轻量 scorer，不全部归给 EMA。

## 7. 执行顺序与资源

P0：三处 spec/tasks/checklist 同步、官方资产/训练身份、既有单模型对照与数据协议冻结。

P1：structRFM 权重与预训练污染审计 → S1/S2 CUDA smoke → 可并行的特征提取/全参训练 → 三种子收敛 → A/B/C → 速度/长度/置信度验收。

P2：RIPE 合成门 → R-log/R1/R2/R3/R5/R6 与等参数对照 → 三种子 → A/B/C。

P3：Pair-JEPA 防坍塌门 → J-mlp/J1/J1-noEMA/J-random → 三种子 → A/B/C 与速度。

P4：只把已独立复算的结果写入图源、PPT 副本和论文；完成代码提交、源数据归档、文件整理与 claim–evidence–limitation 终审。

单卡可并行独立下载、CPU 数据审计和适当显存的训练臂；正式测速与独占 QA 不与其他任务并发。不得为“占满显存”超订。

## 8. 训练记录、PPT、归档与论文叙事

- 每 run 记录：run ID、路线/臂、commit、seed、完整配置、数据/权重/预训练 clean 哈希、CUDA/GPU、batch、参数/FLOPs、best epoch、validation 曲线、早停原因、checkpoint 哈希、A/B/C 原始结果、独立复算哈希、结论与下一步。
- PPT 只修改 `PRISM_组会汇报_20261107_完整版_副本.pptx`，原稿及用户其他调整不动。新增或更新页须微软雅黑、1.5 倍行距、简洁清晰、无重叠、图文同源、逐页渲染复核；未验收结果不写入。
- 活跃绘图源 JSON/NPZ/预测与哈希必须保留；旧版只移动归档并写索引，不删除。根目录废弃脚本、日志、中间稿在科学结果稳定后归档，活跃代码和最新图源不移动。
- 叙事必须按证据决定：骨干表征是否带来增益 → RIPE 哪一通道改善长程/长序列 → 轻量 Pair-JEPA 是否同时改善精度与速度。任一环节不成立就收窄主张，不把模块堆叠写成机制证明。
