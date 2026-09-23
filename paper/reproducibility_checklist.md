# 可复现性清单（Reproducibility Checklist）

> change-id: `build-rna-ss-decision-model`
> 依据：spec §8.1 G5（可复现性：每个基线记录 commit + 权重哈希；0 条缺失）、§4 C5（审计与可复现）、§7.4（统计严谨性）、§10.1（资源）
> 原则：**每一条都必须能追溯到具体文件或命令**。未知值标 `待核验`，不得编造哈希 / DOI / URL。

---

## 1. 代码（Code）

| 项 | 要求 | 位置 / 状态 |
|---|---|---|
| 仓库 commit | 记录本工作全部产出的 commit hash | 运行台账写入 `run_meta.json` 的 `git_commit`（不可得时写 `待核验`） |
| 冻结模块 | `harness.py` / `encoder.py` / `decision_head.py` / `distill.py` / `rlcd.py` / `clean/` / `eval/ss/` 为已核验模块，本阶段**不修改** | `src/rnajepa/`、`eval/ss/` |
| 训练驱动 | `src/rnajepa/train_decision.py`（四项目标、梯度覆盖、NaN 守卫、LR 标定） | 已交付 |
| 提交脚本 | `scripts/run_pretrain.sh`、`scripts/submit_finetune.sh`、`scripts/queue_matrix.py`、`scripts/monitor.py` | 已交付 / 复用 |
| 教师部署 | `scripts/deploy_teachers.sh` | 已交付 |
| 教师吞吐 | `eval/ss/teacher_throughput.py` | 已交付 |
| 测试 | `python -m pytest tests/ -q` 全绿（0 errors） | 交付前实测 |
| 基线纳入 | **官方代码 + 官方权重**，逐条记录仓库 commit 与权重哈希 | 0 条缺失方可投稿（G5） |

---

## 2. 配置（Configs）

| 项 | 要求 | 位置 |
|---|---|---|
| 超参全量记录 | **每一个**超参写入 `run_meta.json`（`TrainConfig.as_dict()`） | 每个 run 目录 |
| 分析计划冻结 | 假设 H1–H7、七类指标、统计方法、§8 全部门限在**任何实验运行之前**冻结 | `spec/analysis_plan.md`（`freeze_status: FROZEN`） |
| 门限修订 | 只允许**追加**在「修订记录」表，注明修订前后数值、依据、时间戳 | `spec/analysis_plan.md` §8 |
| 消融矩阵 | 15 项消融，逐项可开关 | `eval/ss/ablations.py`（`ABLATION_KEYS`） |
| 冻结时间戳 | `freeze_timestamp_utc` 必须早于首个实验 run 的时间戳 | `spec/analysis_plan.md` §0（当前为占位值，**须人工复核替换**） |
| 任务注册表 | 145 任务 + 二级结构任务注册表 | `configs/tasks.yaml`、`configs/ss_tasks.yaml` |
| LR 标定 | 每个微调任务的 lr **逐任务标定并报告**，不得继承默认 `1e-4` | `configs/tasks.yaml` 的 `lr_status` / `lr_evidence`；`records/R3_GATE_AND_LR_CALIBRATION.md` |

---

## 3. 数据卡（Data Card）

| 项 | 要求 | 位置 / 状态 |
|---|---|---|
| 数据卡 | 每个数据集的来源、许可、规模、已知缺陷 | `src/rnajepa/clean/data_card.py` 产出 |
| 数据源状态登记 | 四档状态：已落盘 / 待获取 / 网络受限 / 不可得；**不得把计划中的数据当作已有数据** | `data/ss/inventory.json`、`data/ss/INVENTORY.md` |
| **关键发现（必须披露）** | 既有 20 GB 语料与 145 个下游任务**不含任何二级结构标注** | spec §3.2；数据卡 |
| 不可得项 | `apa_usage` **不在** Zenodo 17786045 内 → 标记不可得，不得静默替代 | spec §3.1 |
| `dev == test` | 23 个任务 `dev == test`（15 `full_*`/`utr5_*` + 8 Spliceator）→ 结论须显式标注 | `eval/ss/downstream.py` 的 `dev_equals_test` |
| 版本化 + 哈希清单 | 数据版本化 + manifest 哈希 | `data/build_manifest.py` |

---

## 4. 逐级衰减表（Attrition Table）

| 项 | 要求 | 位置 / 状态 |
|---|---|---|
| 逐级衰减表 | 清洗 C1–C6 每级列出**剔除数量 + 原因码**，且各级数量守恒 | `src/rnajepa/clean/pipeline.py`、`c5_audit.py` 产出 |
| 原因码 | 每条被剔除记录可追溯（配平 / 最小发夹环 / 交叉 / 长度不一致 / 低分辨率 / 缺失残基 / NMR 重复 / …） | `src/rnajepa/clean/records.py` 的 `ReasonCode` |
| 去冗余 | identity **80% 与 90% 两档分别报告** + family 级划分（同家族不得跨 split） | `src/rnajepa/clean/c2_redundancy.py` |
| 污染检测 | 下游测试集对预训练语料做 exact + 80% identity 搜索，命中即移除，报告污染率 | 同上 |
| 双版本报告 | ArchiveII **去冗余版 + 原始版双报告** | 评测表 |

---

## 5. 分析计划（Analysis Plan）

| 项 | 要求 | 位置 |
|---|---|---|
| 预注册 | 假设 / 指标 / 统计方法 / 门限在任何实验前冻结 | `spec/analysis_plan.md` |
| 重复性 | 每配置 **≥ 5 seed**，报告均值 ± std | `eval/ss/stats.py` 的 `aggregate_over_seeds`（`strict`） |
| 配对检验 | Wilcoxon signed-rank + bootstrap 95% CI（≥ 10,000 次重采样） | `eval/ss/stats.py` 的 `paired_test` |
| 多重比较校正 | Holm-Bonferroni（主）+ FDR / Benjamini-Hochberg（稳健性附报） | `eval/ss/stats.py` 的 `apply_corrections` |
| 红线 | **测试集不参与任何超参选择** | 全流程；`eval/ss/gates.py` 校验 |
| 负结果 | 如实报告，不选择性汇报 | `SC4` |

---

## 6. 教师版本清单（Teacher Version List）

| 项 | 要求 | 位置 / 状态 |
|---|---|---|
| 版本锁定文件 | ViennaRNA / RNAstructure / LinearPartition 的**固定版本 + 检测版本 + 是否核验** | `scripts/deploy_teachers.sh` 写出 `teachers/teacher_versions.json` |
| Turner 参数版本 | 参数集随版本变化；**实际使用的 `.par` 文件必须具名** | 同上，`turner_parameters` 段 |
| 教师集成定义 | `p^teacher = mean(p^ViennaRNA, p^RNAstructure, p^LinearPartition)` | `src/rnajepa/distill.py` 的 `assemble_teacher` |
| 吞吐实测 | 分长度桶 seq/s + 全语料外推（**basis 必须标注**） | `eval/ss/teacher_throughput.py` |
| 软标签可复现 | 分片 manifest + 逐片 SHA256 + 序列 SHA256 + 断点续跑 | `src/rnajepa/distill.py` 的 `generate_teacher_labels` / `verify_teacher_labels` |
| **诚实标注** | 本环境教师**未安装**；用 mock 教师产出的任何数字须标注为 `mock_teacher_throughput`，**不得当作真实工具吞吐** | `eval/ss/teacher_throughput.py` 的 `extrapolation.basis` / `warning` |

---

## 7. 运行台账（Ledger）

| 项 | 要求 | 位置 |
|---|---|---|
| 训练台账 | 每次 run 追加 start / completed / diverged 行 | `<run_dir>/ledger.jsonl` |
| 调度台账 | 每次提交尝试一行（status / device / attempt / batch） | `$ART/ledger.jsonl` |
| 断点续跑 | `resume.pt`（模型 / 优化器 / scheduler / step / 梯度覆盖报告） | `<run_dir>/resume.pt` |
| 训练曲线 | 逐步 loss + 分项 loss + lr + grad norm | `<run_dir>/train_log.jsonl` |
| 元数据 | 超参、数据版本、教师版本、LR 标定结果、梯度覆盖报告、诚实性说明 | `<run_dir>/run_meta.json` |

---

## 8. 投稿前逐项确认（不得留空）

- [ ] G1 非法结构率 = 0
- [ ] G2 最小发夹环违规率 = 0
- [ ] G3 `L ≤ 12` 时 `Z(x)` / `p̂_ij` 与暴力枚举逐位一致
- [ ] G4 跨 split 同源率 = 0；预训练污染率 = 0 或显式量化披露
- [ ] G5 每个基线记录 commit + 权重哈希，**0 条缺失**
- [ ] 分析计划 `freeze_timestamp_utc` 已人工复核替换（当前为占位值）
- [ ] 教师版本清单无 `待核验` 残留
- [ ] 引用登记表无未核验的强主张引用（`spec/citation_register.csv`）
