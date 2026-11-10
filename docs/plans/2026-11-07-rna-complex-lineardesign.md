# RNA 复合物数据扩充与 LinearDesign 下游设计合同（2026-11-07）

> **2026-11-10 最新覆盖**：后续统一改用 RiNALMo，structRFM 仅保留历史诊断；LinearDesign 的114蛋白计算基准、design-dev/test、对照、约束、UTR–CDS与全长评价以 `2026-11-10-rinalmo-full-replay-and-downstream-expansion.md` 为准。本文件保留为原始资产与初始合同。

## 1. 位置与目标

新增路线在既有 structRFM+JEV→RIPE→Pair-JEPA 之后依次执行，不中断当前 S3，也不改变已冻结候选。顺序调整为：P1 structRFM → P2 RIPE → P3 Pair-JEPA → **P4 RNA 复合物数据扩充** → **P5 LinearDesign 下游设计验证** → P6 独立复算、PPT、归档与论文。

P4 的首要问题是：真实 RNA 复合物中的链内结构监督能否改善 Rfam/CHANRG 等 B 组，而不损害 A/C。P5 的问题是：能否把 LinearDesign 作为编码区设计生成器，并用冻结的独立单模型和热力学基线进行可复现评价。两者都不是对既有失败实验的事后调参。

## 2. P4：RNA 复合物结构数据扩充

### 2.1 推荐方案与排除项

采用 **D1 纯数据扩充**：收集 RNA–蛋白质等复合物中的 RNA 链，使用真实链内二级结构标签训练；复合物元数据和界面接触单独保存。D0 为不加新数据的同配置对照。只有 D1 完成验收后，才可增加独立 D2 界面辅助任务，不把蛋白伴侣或界面接触转换成不存在的 RNA–RNA 配对。

排除两条捷径：不把近邻接触直接当二级结构标签；不把复合物序列未经审计混入 TR1c。当前模型只吃 RNA 序列，因此 D1 证明的是“数据分布扩充是否有效”，不是“模型理解了蛋白界面”。D2 若实施，必须有独立界面标签、输出和消融，不能与 D1 混为同一归因。

### 2.2 数据来源与标签

- 主源为 RCSB PDB 条目和 mmCIF；NDB 仅作补充，需独立记录来源与许可。全文检索只可作候选预筛，最终必须按条目/实体类型精确过滤，不能用全文命中数冒充复合物数。2026-11-07 已用 `rcsb_entry_info.polymer_entity_count_RNA>0` AND `polymer_entity_count_protein>0` 获取8,157个去重entry，清单SHA256=`9b0d5045d2b5cdabd5cbc769827e6cb037adf9de2dc21773524ef7b761e74f2d`；当前API属性查询使用 `service="text"`。该数仅为entry级候选，不代表可用RNA链数或训练样本数。
- 保存 PDB ID、entity/chain、实验方法、分辨率、发布日期、来源物种、伴侣类型、RNA 长度、原始 mmCIF 哈希和解析错误。大文件放 `/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rna_complex_round49`，代码留 `/home/rna_jev`。
- 链内配对使用固定版本 DSSR 的完整输出；记录软件版本、命令、Raw 配对、canonical-nested 子集和假 knot。DSSR/mmCIF 映射以完整 `nts` 清单中的链、实测残基顺序和碱基身份为桥接，不混用作者编号与实体序号。Raw 可能包含共享碱基、非经典配对与假结，不能作为当前嵌套结构解码器的可实现监督；canonical-nested 用确定性动态规划取最大基数的合法嵌套子集，不按测试表现挑配对。D1 训练及主 complex holdout 只使用 canonical-nested，Raw 与非嵌套部分保留用于分层分析，不把不可实现配对写入训练目标。D2 界面接触另存距离阈值与链映射，不进入 D1 的配对标签。
- DSSR 不可用、链映射不一致或标签无法独立复核时停止，不安装来路不明替代物，不用经验规则补标签。2026-11-07 后续已从官方 `docs.x3dna-dssr.org/download.php?pkg=linux` 固定安装 DSSR v1.9.10-2020apr23，ZIP SHA256=`602b2d0dfda969999204a570e848c3dfb0203c9d3b12436e7d0275a5b4ea78aa`，二进制 SHA256=`c7261c0ab099e1f70c508c51363c7aa6915e216a78bb6d65ccda533092d6c39a`；仅限非商业用途，不推送到项目仓库。1EHZ mmCIF工程验收输出34对，JSON SHA256=`0828a38b70938b170745e938c394b5ad2e18b01f5aeeffc2b0c0518ea413eda1`。P4仍须逐条目核验链映射和标签一致性。

### 2.3 去泄露与分桶

- 对 TR1c、VL0、13 个 A/B/C split 和 structRFM 预训练语料分别执行 exact + 80% identity/query-coverage 审计。使用 structRFM 初始化时，预训练接触不可忽略。
- 按 PDB entry 和序列簇联合切分，任一 RNA 链及其同源簇不得跨训练、验证和独立 holdout。保留删除清单、raw→clean attrition 和 SHA256；不得用删除长链的方式制造提升。
- D0/D1 均至少三种子自然收敛，VL0 只做早停和配置选择。每个候选执行 A/B/C、Rfam、CHANRG 和独立 complex holdout，保留长度、长程配对、假 knot 分层和单 checkpoint 推理。
- 主判定为 B 组改善且 A/C 不显著退化；主八 split 做配对 bootstrap/Holm，B 组单独报告，不以合并均值掩盖弱格。D2 若开展，另报界面 AUROC/AUPRC，不能用结构 F1 代替界面效果。

### 2.4 用户指定的旧 xens2 优先验证（2026-11-09）

在通用 D0/D1 前插入旧 xens2+扩充数据→Rfam验证，保留其集成身份，不替代独立单模型目标。三个训练种子固定 Plan-A=42、Plan-A=2026、R2D=3407；两个 Plan-A 桶内等权，Plan-A 桶与 R2D 按0.7/0.3融合。Plan-A保持四目标、33层逐步解冻，R2D保持frozen RiNALMo+ResNet2D与同一四目标；按用户新种子合同改变种子，训练均须自然收敛。Plan-A原batch=4在step1550反向阶段OOM，batch=2也在同一点达到约23.2GB，故从step1500以batch=1恢复，模型、数据与收敛门不变。R2D原batch=4在24GB RTX4090独占时于975nt pair-type张量处OOM，batch=2也在step75发生峰值OOM，故硬件适配为batch=1；模型结构、数据与收敛门不变，失败日志保留。

当前远端独立去泄露版本为805 train/201 holdout，旧647/359不得混用。原xens2共同长度上限1024nt，故仅150/805条train可直接用于本架构，655条超限明确排除；holdout可支持101条。训练集为TR1c 42564+扩充150=42714，使用各自真实canonical-nested GT与ViennaRNA 2.7.2教师，不按Rfam效果筛选链。先验证教师与RiNALMo嵌入完整，再训练三成员，最后在相同Rfam上做三次确定性推理和独立计数复核。该结果不能外推为长链或独立单模型结论。

## 3. P5：LinearDesign 参考的下游设计

### 3.1 算法边界

官方论文：Zhang et al., *Algorithm for optimized mRNA design improves stability and immunogenicity*, Nature 621, 396–403 (2023), DOI=`10.1038/s41586-023-06127-z`。算法以 DFA/lattice representation 和 lattice parsing 联合优化 MFE 与 CAI，目标为 `MFE – λ|p|log(CAI)`，原文只针对编码区。

推荐 **生成器+独立评价器**：LinearDesign 生成候选，冻结 PRISM 单模型与独立热力学工具评价。禁止把 LinearDesign 的输出当真实标签反向蒸馏 PRISM，也禁止用于没有蛋白编码约束的一般 Rfam RNA。

### 3.2 实施与验收

- 固定官方代码/权重或算法复现的 commit、输入蛋白、遗传密码表、宿主密码子偏好和 λ；先在小蛋白上验证氨基酸序列逐位不变、合法终止和密码子映射。官方实现已定位为 `LinearDesignSoftware/LinearDesign`，commit=`f0126ca89a8b853088b4bccfd2cc8c378d3678be`；只保留在 `third_party`，其许可禁止未经书面许可的代码再分发，不提交或镜像到本项目仓库。已用隔离 Python 2.7.18 编译成功，README 的 MNDTEAI（λ=0）和 testseq（λ=3）在序列、结构、MFE、CAI 上逐项一致；编译二进制 SHA256=`f1f22901967cb3e0f569ff2332c326800ac355727ab58e377b14514db4fe9688`，官方库 SHA256=`5707cd7d7950a711af74350c94956d1b0212eae37cc5bd04872967597cb3d330`。该工程示例不代表下游设计效果。
- λ=0 的 MFE-only 与联合优化分开；λ 网格只在 design-dev 锁定，测试蛋白不得参与调参。野生型、常规 codon-optimized 和 LinearDesign 候选分开报告。
- 评价 MFE/CAI/GC、PRISM 结构 F1/INF/置信度、长程配对、运行时间和合法性；独立 ViennaRNA/RNAfold 作热力学参照。PRISM 不替代实验稳定性、表达量或免疫原性证据。
- 最终输出设计序列、输入蛋白、参数、代码哈希、预测结构和独立复算。无湿实验时不写“半衰期延长、表达提高或免疫增强”。

## 4. 阶段门与交付

P4 开始前必须通过：RCSB 精确实体清单、mmCIF/DSSR 标签一致性、联合去泄露、非空 complex holdout、D0 配置冻结。P5 开始前必须完成 D0/D1 的 A/B/C 裁决。P6 只接收独立复核后的结果；任何负结果收窄主张，不为了预设 8/8 反复改测试集。

训练日志继续写 `records/DECISION_TRAINING_LOG.md`。PPT 副本、绘图源、归档和论文 claim–evidence–limitation 终审延至 P6；现有页面和用户调整不因此提前修改。
