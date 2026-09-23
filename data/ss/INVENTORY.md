# 二级结构数据盘查清单（实测状态，非计划）

- 生成时间（UTC）：`2026-09-23T17:54:41Z`
- 模式：**offline**（offline：仅使用 recorded 状态，未做任何探测）
- 数据根目录：`/mnt/cunyuliu/rna-jepa/data/raw`
- 状态类别（恰好四档）：已落盘 / 待获取 / 网络受限 / 不可得

> ## ⚠️ 关键发现（必须置顶）
> 
> 已落盘资产（Zenodo 17786045 / 12516160 / mRNABERT）**不含任何二级结构标注**。17786045 是 145 个 mRNA 功能任务，12516160 是无标签序列。二级结构预测需要一条全新的数据获取工作流。
>
> **已落盘资产含二级结构标注数 = 0。** 不得复用现有语料充当结构标签。

## 证据说明（recorded-fact vs live-probe）

| evidence | 含义 | 是否可称『已核验』 |
|---|---|---|
| `recorded_fact` | 已记录在案的实测事实（如已完成 fetch 的 md5/大小） | 否（本环境未复测） |
| `live_probe` | 本次进程真实探测（文件存在 / HTTP 可达 / 工具可导入） | 是 |
| `requires_live_probe` | 仅有记录性预期，**尚未探测** | 否 |

## 1. 已落盘资产（可直接使用，均无结构标注）

| key | 状态 | evidence | verified | 规模 | md5 | 许可 | SS 标注 | 备注 |
|---|---|---|---|---|---|---|---|---|
| `zenodo_17786045::full_length` | **已落盘** | recorded_fact | — | 281.2 KB | 3652178c257341010800e2d241a9c258 | 见 Zenodo 记录 17786045 | no | mRNA 功能任务包；不含二级结构标注 |
| `zenodo_17786045::Spliceator` | **已落盘** | recorded_fact | — | 24.2 MB | d80c393eec09728c57e2c66c361e90e9 | 见 Zenodo 记录 17786045 | no | mRNA 功能任务包；不含二级结构标注 |
| `zenodo_17786045::protein` | **已落盘** | recorded_fact | — | 30.7 MB | e8f6b277303959a2010d0047830b2c83 | 见 Zenodo 记录 17786045 | no | mRNA 功能任务包；不含二级结构标注 |
| `zenodo_17786045::te_ultra_full_length` | **已落盘** | recorded_fact | — | 32.6 MB | 939b495793687db362d4b9464a5df570 | 见 Zenodo 记录 17786045 | no | mRNA 功能任务包；不含二级结构标注 |
| `zenodo_17786045::CDS` | **已落盘** | recorded_fact | — | 35.6 MB | dbb145edd36a67b63e7184da04dab8c4 | 见 Zenodo 记录 17786045 | no | mRNA 功能任务包；不含二级结构标注 |
| `zenodo_17786045::5UTR` | **已落盘** | recorded_fact | — | 60.2 MB | 6d36a52b06b6d493e900d60590e881da | 见 Zenodo 记录 17786045 | no | mRNA 功能任务包；不含二级结构标注 |
| `zenodo_17786045::3UTR` | **已落盘** | recorded_fact | — | 64.1 MB | 6edf8dffcb2ee63560e276a65a5f5e9f | 见 Zenodo 记录 17786045 | no | mRNA 功能任务包；不含二级结构标注 |
| `zenodo_12516160` | **已落盘** | recorded_fact | — | 20.02 GB | bf8bc5c946a0bd3b07716b1c7f785d54 | 见 Zenodo 记录 12516160 | no | 无标签序列语料；不含二级结构标注 |
| `mrnabert_weights` | **已落盘** | recorded_fact | — | 待核验 | 待核验 | 见 HuggingFace 模型卡 | no | 权重哈希未记录 -> 待核验；不含二级结构标注 |

## 2. 待获取的结构标注数据（本项目必需）

| key | 状态 | evidence | verified | 规模 | md5 | 许可 | SS 标注 | 备注 |
|---|---|---|---|---|---|---|---|---|
| `bprna_1m` | **待获取** | recorded_fact | — | 待核验 | 待核验 | 待核验 | yes | recorded_from_spec §3.3；站点可用性未经实测；主训练集，须尽早实测 |
| `bprna_new` | **待获取** | recorded_fact | — | 待核验 | 待核验 | 待核验 | yes | 仅评测，不可用于训练；recorded_from_spec §3.3；站点可用性未经实测 |
| `archiveii` | **待获取** | recorded_fact | — | 待核验 | 待核验 | 待核验 | yes | 须去冗余并双版本（去冗余 + 原始）并列报告 |
| `rnastralign` | **待获取** | recorded_fact | — | 待核验 | 待核验 | 待核验 | yes | 与 ArchiveII 有重叠，须交叉去冗余 |
| `pdb_derived` | **待获取** | recorded_fact | — | 待核验 | 待核验 | 待核验 | yes | 需分辨率与方法过滤 |
| `pseudobase_pp` | **待获取** | recorded_fact | — | 待核验 | 待核验 | 待核验 | yes | 假结；规模小 |
| `rna_puzzles_casp` | **待获取** | recorded_fact | — | 待核验 | 待核验 | 待核验 | yes | 仅评测，绝不进训练 |
| `probing_data` | **待获取** | recorded_fact | — | 待核验 | 待核验 | 待核验 | partial | 间接/噪声标签，须分级使用，不可当作结构真值 |
| `rnacentral` | **网络受限** | recorded_fact | — | 待核验 | 待核验 | 待核验 | no | 规模（数十 GB）远超结构数据集 1-2 GB；须先做可达性与吞吐实测 |
| `rfam` | **网络受限** | recorded_fact | — | 待核验 | 待核验 | 待核验 | partial | 含一致性二级结构（covariance models），非逐碱基结构真值 |
| `ensembl_gencode` | **网络受限** | recorded_fact | — | 待核验 | 待核验 | 待核验 | no | 序列/注释，非二级结构标注 |

## 3. 教师模型依赖（非数据）

| key | 状态 | evidence | verified | 规模 | md5 | 许可 | SS 标注 | 备注 |
|---|---|---|---|---|---|---|---|---|
| `viennarna` | **待获取** | recorded_fact | — | 待核验 | 待核验 | ViennaRNA 许可（待核验具体条款） | n/a | System-2 教师 + 低置信回退求解器；安装需网络 |
| `rnastructure` | **待获取** | recorded_fact | — | 待核验 | 待核验 | 待核验 | n/a | 教师集成成员；安装需网络 |
| `linearpartition` | **待获取** | recorded_fact | — | 待核验 | 待核验 | 待核验 | n/a | 教师集成成员；安装需网络 |

## 4. 网络约束（实测）

| 约束 | 后果 |
|---|---|
| zenodo.org DNS 不可解析（返回 ::） | 必须硬编码 IP（scripts/fetch_zenodo_parallel.sh 已实现 --resolve） |
| 单连接吞吐被限速至 ~15-20 KB/s | 单流下载不可行；须多连接分块 |
| 168 并发实测约 4 MB/s（32 conn ~460 KB/s，64 conn ~950 KB/s） | 大文件走多连接 ranged download + md5 校验 |
| 本代码机无网络访问（大文件在集群 /mnt/cunyuliu，本机不可达） | 本仓库只产出代码/清单；不执行下载 |

## 5. 状态统计与断言

| 状态 | 数量 |
|---|---|
| 已落盘 | 9 |
| 待获取 | 11 |
| 网络受限 | 3 |
| 不可得 | 0 |

**断言（代码强制）**：每个 source 的状态必属四档之一，且必有 evidence 标签；offline 模式下未探测的条目 `verified=False`，**不存在『假定可用』的 source**。

