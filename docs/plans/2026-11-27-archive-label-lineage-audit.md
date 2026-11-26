# ArchiveII clean、bpRNA TS0、TestSetB 标签谱系独立审计

基线：`bf463ccb4b5d0a8da2a2d2fed1696fbf63a8809c`；bpRNA-new official-reference 修复 `e7ff4e76fe078178ab41d2052349627317b5a3e3` 已验证为基线祖先。

本审计只读外部 Test 标签、manifest 和预测登记元数据，不运行训练/推理、不使用 GPU、不修改共享 registry、不计算排名或选择模型。`PASS` 仅指审计执行/复算完成；证据不足保留 `UNCHECKED`。

## 总结与独立评价资格

| 数据集 | 项目标签谱系 | 项目标签上的既有分数 | 独立评价资格 |
|---|---|---|---|
| ArchiveII clean | `DERIVED_FROM_BPFOLD_PREDICTIONS` | `CIRCULAR_AUDIT_ONLY` | `NOT_ELIGIBLE_INDEPENDENT_ACCURACY` |
| bpRNA TS0 | `DERIVED_FROM_OFFICIAL_REFERENCE_PROJECTION` | `PROJECTED_REFERENCE_PROTOCOL_ONLY_NOT_UNFILTERED_OFFICIAL_PROTOCOL` | `ELIGIBLE_ONLY_EXPLICIT_PROJECTED_REFERENCE_PROTOCOL` |
| TestSetB | `DERIVED_FROM_OFFICIAL_REFERENCE_PROJECTION` | `PROJECTED_REFERENCE_PROTOCOL_ONLY_NOT_UNFILTERED_OFFICIAL_PROTOCOL` | `ELIGIBLE_ONLY_EXPLICIT_PROJECTED_REFERENCE_PROTOCOL` |

ArchiveII clean 的名称不能证明标签独立或训练无重叠。TS0/TestSetB 的参考来源独立不等于每个模型均无训练暴露；必须按模型训练路线另审。bpRNA 论文说明其是来源异质的 meta-database，CRW/Rfam 含比较分析谱系，PDB 经 RNAview 转换，因此逐记录实验谱系仍为 `UNCHECKED`。

## ID、sequence、pairs 与 SHA 全量核对

| 数据集 | 项目/官方参考 n | 项目配对精确等于来源投影 | 相对来源删除 min-loop/crossing | 项目/参考 empty | 参考包交叉核对 |
|---|---:|---:|---:|---:|---|
| ArchiveII clean | 2544 / 3966 | 2544 | 163 / 8002 | 0 / 5 | `EXACT` |
| bpRNA TS0 | 1288 / 1305 | 1288 | 56 / 0 | 0 / 1 | `EXACT` |
| TestSetB | 428 / 430 | 428 | 113 / 0 | 0 / 0 | `UNCHECKED_SINGLE_PACKAGE` |

逐记录证据见 `artifacts/archive_label_lineage_audit_20261127/records.jsonl.gz`，包含官方全集、项目排除记录、member SHA、sequence SHA、pairs SHA、参考包间比较和投影差分。TestSetB 的项目 ID 仅增加 `testsetb_` 前缀。BPSEQ 连续 1-based partner 转为 zero-based inclusive pairs。

ArchiveII 的 3950→2544 筛选只能证明是保留记录子集，未找到独立筛选 manifest，故该规则为 `UNCHECKED`。TS0 的 17 条和 TestSetB 的 2 条官方记录未进入项目面板，原因按 BPSEQ 字符证据分类；全集协议不得静默使用项目子集代替。

项目计分器 `eval/ss/metrics.py` 的 empty/empty 单序列 P/R/F1=1；Mathews rescore 也采用 empty/empty F1=1。micro 汇总不采用逐序列 empty/empty 加分。三套项目标签均无 empty 记录，但该约定仍须随结果登记。

官方参考为空但进入项目面板的记录数为：ArchiveII 4、TS0 0、TestSetB 0。ArchiveII 的这 4 条项目派生标签非空，不能把其预测派生标签上的非空表现当作官方参考上的 empty/empty 一致。

## 训练重叠（按路线，不作模型选择）

| 数据集 | 路线 | 精确序列重叠 | 冻结 homology-only 删除 | clean n | clean SHA 重建 |
|---|---|---:|---:|---:|---|
| archiveii_embok_clean | bprna_tr0 | 0 | 1450 | 1094 | EXACT |
| archiveii_embok_clean | bprna_tr1 | 1014 | UNCHECKED | UNCHECKED | UNCHECKED |
| archiveii_embok_clean | bprna_tr1c | 0 | 1437 | 1107 | EXACT |
| bprna_ts0 | bprna_tr0 | 0 | 96 | 1192 | EXACT |
| bprna_ts0 | bprna_tr1 | 1087 | UNCHECKED | UNCHECKED | UNCHECKED |
| bprna_ts0 | bprna_tr1c | 0 | 575 | 713 | EXACT |
| testsetb | bprna_tr0 | 247 | 65 | 116 | EXACT |
| testsetb | bprna_tr1 | 358 | UNCHECKED | UNCHECKED | UNCHECKED |
| testsetb | bprna_tr1c | 0 | 140 | 288 | EXACT |

精确重叠与 MMseqs2 homology 是不同口径。TR0/TR1c 的既有 clean 审计已逐索引并重建完整 clean JSONL SHA；TR1 只在本审计中复算精确序列重叠，homology 为 `UNCHECKED`。任何 `clean_n` 都不能消除 ArchiveII 的预测派生标签问题。

## 需重算的预测集合

目标均为匹配 ID 的**未过滤官方参考 BPSEQ**。`RESCORE_FROZEN_RECORD_LEVEL_PREDICTIONS` 表示无需新推理即可复算；其余动作表示当前登记资产不足，不能从 summary-only 数值补造记录级结果。官方全集中不在项目面板的 ID 需先做覆盖核算；本审计未补跑。


### archiveii_embok_clean

- `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED`（12）：`nussinov_turner/archiveii_embok_clean/repeat0`、`nussinov_turner/archiveii_embok_clean/repeat1`、`nussinov_turner/archiveii_embok_clean/repeat2`、`vienna_centroid/archiveii_embok_clean/repeat0`、`vienna_centroid/archiveii_embok_clean/repeat1`、`vienna_centroid/archiveii_embok_clean/repeat2`、`vienna_mea/archiveii_embok_clean/repeat0`、`vienna_mea/archiveii_embok_clean/repeat1`、`vienna_mea/archiveii_embok_clean/repeat2`、`vienna_mfe/archiveii_embok_clean/repeat0`、`vienna_mfe/archiveii_embok_clean/repeat1`、`vienna_mfe/archiveii_embok_clean/repeat2`
- `BLOCKED_REGISTERED_PREDICTION_MISSING`（3）：`best_plan_a/archiveii_embok_clean`、`h0/archiveii_embok_clean`、`rinalmo_2d_s0/archiveii_embok_clean`
- `RESCORE_FROZEN_RECORD_LEVEL_PREDICTIONS`（17）：`bpfold/archiveii_embok_clean`、`contrafold/archiveii_embok_clean`、`efold_0_1_3/archiveii_embok_clean`、`eternafold/archiveii_embok_clean`、`kirigami_fast/archiveii_embok_clean`、`mxfold2/archiveii_embok_clean`、`nucleicbert_frozen_official/archiveii_embok_clean`、`nucleicbert_ft/archiveii_embok_clean`、`rinalmo_ft/archiveii_embok_clean`、`rnaformer_biophysical/archiveii_embok_clean`、`rnaformer_bprna/archiveii_embok_clean`、`rnaformer_interfamily/archiveii_embok_clean`、`rnaformer_intrafamily/archiveii_embok_clean`、`rnastructure_fold/archiveii_embok_clean`、`rnastructure_maxexpect/archiveii_embok_clean`、`ufold/archiveii_embok_clean`、`xens2/archiveii_embok_clean`

### bprna_ts0

- `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED`（12）：`nussinov_turner/bprna_ts0/repeat0`、`nussinov_turner/bprna_ts0/repeat1`、`nussinov_turner/bprna_ts0/repeat2`、`vienna_centroid/bprna_ts0/repeat0`、`vienna_centroid/bprna_ts0/repeat1`、`vienna_centroid/bprna_ts0/repeat2`、`vienna_mea/bprna_ts0/repeat0`、`vienna_mea/bprna_ts0/repeat1`、`vienna_mea/bprna_ts0/repeat2`、`vienna_mfe/bprna_ts0/repeat0`、`vienna_mfe/bprna_ts0/repeat1`、`vienna_mfe/bprna_ts0/repeat2`
- `BLOCKED_REGISTERED_PREDICTION_MISSING`（2）：`best_plan_a/bprna_ts0`、`h0/bprna_ts0`
- `RESCORE_FROZEN_RECORD_LEVEL_PREDICTIONS`（19）：`bpfold/bprna_ts0`、`contrafold/bprna_ts0`、`efold_0_1_3/bprna_ts0`、`eternafold/bprna_ts0`、`kirigami/bprna_ts0`、`kirigami_fast/bprna_ts0`、`mxfold2/bprna_ts0`、`nucleicbert_frozen_official/bprna_ts0`、`nucleicbert_ft/bprna_ts0`、`rinalmo_2d_s0/bprna_ts0`、`rinalmo_ft/bprna_ts0`、`rnaformer_biophysical/bprna_ts0`、`rnaformer_bprna/bprna_ts0`、`rnaformer_interfamily/bprna_ts0`、`rnaformer_intrafamily/bprna_ts0`、`rnastructure_fold/bprna_ts0`、`rnastructure_maxexpect/bprna_ts0`、`ufold/bprna_ts0`、`xens2/bprna_ts0`

### testsetb

- `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED`（12）：`nussinov_turner/testsetb/repeat0`、`nussinov_turner/testsetb/repeat1`、`nussinov_turner/testsetb/repeat2`、`vienna_centroid/testsetb/repeat0`、`vienna_centroid/testsetb/repeat1`、`vienna_centroid/testsetb/repeat2`、`vienna_mea/testsetb/repeat0`、`vienna_mea/testsetb/repeat1`、`vienna_mea/testsetb/repeat2`、`vienna_mfe/testsetb/repeat0`、`vienna_mfe/testsetb/repeat1`、`vienna_mfe/testsetb/repeat2`
- `BLOCKED_REGISTERED_PREDICTION_MISSING`（2）：`best_plan_a/testsetb`、`h0/testsetb`
- `RESCORE_FROZEN_RECORD_LEVEL_PREDICTIONS`（18）：`bpfold/testsetb`、`contrafold/testsetb`、`efold_0_1_3/testsetb`、`eternafold/testsetb`、`kirigami_fast/testsetb`、`mxfold2/testsetb`、`nucleicbert_frozen_official/testsetb`、`nucleicbert_ft/testsetb`、`rinalmo_2d_s0/testsetb`、`rinalmo_ft/testsetb`、`rnaformer_biophysical/testsetb`、`rnaformer_bprna/testsetb`、`rnaformer_interfamily/testsetb`、`rnaformer_intrafamily/testsetb`、`rnastructure_fold/testsetb`、`rnastructure_maxexpect/testsetb`、`ufold/testsetb`、`xens2/testsetb`

ArchiveII 还可直接使用 BPfold v0.2 release 原始预测对官方参考复算，但必须与项目六 checkpoint replay 保持不同 prediction identity。所有外部 Test 复算只可用于后续审计，不得据此选 checkpoint、阈值或模型。

## 独立产物与复现

- `source_data/archive_label_lineage_audit_20261127/`：官方文档快照、source manifest 和输入身份。
- `artifacts/archive_label_lineage_audit_20261127/`：逐记录谱系、训练重叠、预测重算清单、summary 和各轮独立验证。
- `tools/build_archive_label_lineage_audit_20261127.py`：第一实现；`tools/verify_archive_label_lineage_audit_20261127.py`：不导入生成器的第二实现。
- `tests/test_archive_label_lineage_audit_20261127.py`：BPSEQ、投影、empty/empty、资格与 fail-closed 单元测试。

共享 `data_registry.json`、`prediction_registry.json`、`baseline_registry.json` 仅读取并记录 SHA，后续由统一流程登记，本分支不修改。
