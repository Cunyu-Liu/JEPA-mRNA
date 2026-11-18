# FoldNAR Phase 0 baseline 与预测身份合同（v3）

日期：2026-11-18  
分支：`foldnar-e0-20261118`

## 1. 补救结论

旧版 `baseline_registry.json` 的 18 行已完整备份，不作为本次充分性的证据。本次冻结结果为：

| 注册表 | 行数 | 口径 |
|---|---:|---|
| 数据 | 31 | 27 FROZEN，4 BLOCKED |
| 预测 | 276 | 162 FROZEN 逐序列预测，96 SUMMARY_ONLY，18 BLOCKED |
| Baseline | 63 | 54 个三来源核心并集 + 9 个合同/实测补充身份 |

25 个历史系统、13 个主榜身份和 29 个 CHANRG predictor 共形成 67 条来源成员记录；三种独立计数均得到 54 个核心去重身份。63 个 baseline 按结构解码、直接神经、foundation model、本项目四类管理，分别为 14、16、28、5 行；按身份层级分为 56 个具体 predictor 和 7 个 model family。

每一行均含 `official_weights`、`training_data`、`license`、`version_protocol`、`inference`、`comparability`、`prediction_ids` 及来源证据。无法核验的字段标为 `BLOCKED_PROVENANCE`，不填 0，不把框架支持某个 head 解释为 checkpoint 已发布，不把 `LEGACY_TABLE_ONLY` 解释为当前逐序列预测。

## 2. CHANRG 29 的证据等级

CHANRG 主文给出 29 predictors 和 canonical 17-model cohort。当前身份组成为：

- 8 个 structured decoders；
- 3 个 direct neural predictors；
- 6 个 foundation-model backbone × Linear / ResNet / U-Net 三类 contact head，共 18 个 predictor。

17 个 canonical 名称已从主文 Figure 2 核准；三类 head 由 MultiMolecule 0.2.3 源码确认。完整 29 的组成标为 `RECONSTRUCTED_FROM_MAIN_AND_MULTIMOLECULE_0_2_3_SOURCE`，不是 Extended Data 官方结果表的逐项核验。Extended Data 结果表、各 head 的 checkpoint、逐项训练配置和模型/权重许可仍为 blocked。

来源：

- https://arxiv.org/html/2603.22330
- https://pypi.org/project/multimolecule/
- https://github.com/DLS5-Omics/MultiMolecule

## 3. 新登记的预测资产

在原有 89 行（71 FROZEN、18 BLOCKED）基础上，新增 187 行：

- 91 行新的 FROZEN 逐序列预测：eFold 0.1.3（8）、CONTRAfold（8）、RNAstructure Fold/MaxExpect（16）、RNAformer 四 checkpoint（32）、NucleicBERT official/fine-tuned（16）、Kirigami/Kirigami-fast（10）、SPOT-RNA bpRNA-new（1）。
- 96 行 SUMMARY_ONLY：8 个主 split × 3 repeats × Vienna MFE/centroid/MEA/Nussinov。文件只有聚合指标，没有逐序列列表，不能计入 raw 预测。

新增资产全部标记 `NOT_IN_EXPANDED_E0_PHASE0_COMPLETION`；既有 71-run E0 的输入范围和 174,007 行结果未改变。

## 4. 强制身份边界

- eFold 主榜 0.1.2 与当前 raw 0.1.3 分行。
- NucleicBERT 项目训练 head、官方 released SSP checkpoint 和 fine-tuned checkpoint 分行。
- RNAformer bpRNA、inter-family、intra-family、biophysical 四 checkpoint 分行；biophysical 不替代 intra-family，也不作为 PDB 实验标签参照。
- RiNALMo-ft Zenodo checkpoint、R2D 项目 head 与 CHANRG RiNALMo-Giga-U-Net 分行。
- RNAstructure Fold 与 MaxExpect 分行；Kirigami 标准与 fast decoder 分行。
- 旧 generic `vienna`、`linearfold`、`nucleicbert_frozen` 身份均有显式处置，不作自动连接。

## 5. 验收

- 生成器内验收：PASS。
- 不导入生成器的独立复核：50 项检查，0 失败。
- 回归测试：34 passed。
- 旧 18 行 baseline 注册表备份：`phase0_baseline_v3/registry_backup/baseline_registry.json`。
- 独立复核初次将嵌套的 `final_decision.e1_allowed` 错读为顶层字段；失败报告保留为 `independent_registry_verification_v1_schema_error.json`，修正验收器后正式 PASS，科学结果未改。

复现命令：

```bash
cd /home/rna_jev_xens2_dualview_round56
PYTHONPATH=src /root/miniconda3/envs/lucaone_cu128/bin/python -m pytest -q \
  tests/test_foldnar_phase0.py tests/test_foldnar_phase1.py \
  tests/test_foldnar_e0.py tests/test_fold_structure.py

/root/miniconda3/envs/lucaone_cu128/bin/python tools/foldnar_phase0.py \
  --out-phase phase0_baseline_v3 --refresh-registries

/root/miniconda3/envs/lucaone_cu128/bin/python tools/foldnar_verify_phase0.py \
  --phase phase0_baseline_v3
```

## 6. 科学裁决与 SOTA 边界

`final_decision.e1_allowed=false`，执行分流仍为 `DO_NOT_ENTER_E1`，原因仍为 `INSUFFICIENT_EVIDENCE_ROUTE_PAIR_COVERAGE_AND_DATA`。E1–E5 均未运行；组合规则学习未测试；本轮没有产生 FoldNAR checkpoint 或最终 wiring 因果证据。

SOTA 仍按预注册协议：每个 predictor 固定模型/配置/推理协议，raw 与 baseline-specific clean 结果分面，项目 raw 与 CHANRG 外部协议分面；历史表格数字只能标 `LEGACY_TABLE_ONLY`。当前 CHANRG 文件数量与完整合同不一致，外部训练集同源与许可仍不完整，因此不作统一 SOTA 排名或 SOTA 宣称。

## 7. 代码、产物与哈希

代码提交：`3f2ddca`（注册表合同）、`68989f0`（独立验收器字段兼容修正）。

代码：

- `/home/rna_jev_xens2_dualview_round56/tools/foldnar_phase0.py`
- `/home/rna_jev_xens2_dualview_round56/tools/foldnar_baseline_contract.py`
- `/home/rna_jev_xens2_dualview_round56/tools/foldnar_verify_phase0.py`
- `/home/rna_jev_xens2_dualview_round56/tests/test_foldnar_phase0.py`

正式注册表：

- `manifests/foldnar_v1/data_registry.json`  
  SHA256 `4fe3f5376498b64ffcce758300067f3bcf1489868dbe3156d84b886f0c16fd69`
- `manifests/foldnar_v1/prediction_registry.json`  
  SHA256 `da2b6fadaf994496dc004375be96cda9fdaa024dba4ce0e465738f048010d1c9`
- `manifests/foldnar_v1/baseline_registry.json`  
  SHA256 `51878f149f7521ab714b8ed7e1db9cee8dab11f32639aa38a0f6a0fb65711c8a`

Artifact 根：`/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/foldnar_v1/`。

- `phase0_baseline_v3/environment.json`  
  SHA256 `23b1d7eebee047f93ced62a64ea10c428cb4d038444448237de67e97743db847`
- `phase0_baseline_v3/registry_verification.json`  
  SHA256 `c245a5e7f631dbab12dcdbbf5b31effdf2e607005ce980909e6dfefb185d7957`
- `phase0_baseline_v3/independent_registry_verification.json`  
  SHA256 `ab4d2f73ea1afa33cd09ccca1e67a854ab7a8a1211eb55c9c63c6b2a87208629`
- `phase0_baseline_v3_tests.log`  
  SHA256 `e6f0443ac923a25952c59d0a822bc1172620c56658dafb752be9a2c7d61b4fe4`
- `e0_expanded/final_scientific_decision.json`  
  SHA256 `c5db8b20f72575cee049dc9eecdea5d669f6be8cb71ce98793ecd780b13b641f`

组会 PPT 已追加第 36 页，在线整稿 36 页、revision 23，新增页 ID `png`；原 35 页 ID 与内容保持不变。最终回读 XML SHA256：`dc5ee8cd9390ed152a25258724f993b1fb5ab370fe14755283c116c8e74b5aa4`。
