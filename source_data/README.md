# 活跃绘图源数据索引

本目录只放当前 PPT 实际引用的源数据；废弃版本在 `../archive_20261031/`。

| 文件 | 说明 | 状态 | SHA256（前 12 位） |
|---|---|---|---|
| `fig03_ablation/fig03b_group_drops.json` | 主图 3B 核心三模块分组均值与 PDB/ID 比值 | active | `3e048a175cf8` |
| `fig05_speed/r2d_b1_profile.json` | round36 r2d B=1 全 TS0 分段测速原始结果 | active | `7ef14e1fe5a7` |
| `fig05_speed/r2d_b4_profile.json` | round36 r2d B=4 全 TS0 分段测速原始结果 | active | `db08a49ceb81` |
| `fig05_speed/plana_b1_profile.json` | round36 plana B=1 全 TS0 分段测速原始结果 | active | `e178b142da13` |
| `fig05_speed/plana_b4_profile.json` | round36 plana B=4 全 TS0 分段测速原始结果 | active | `9344b0d92ec8` |
| `fig05_speed/xens2_b1_profile.json` | round36 xens2 B=1 全 TS0 分段测速原始结果 | active | `d6fa8717a6e6` |
| `fig05_speed/xens2_b4_profile.json` | round36 xens2 B=4 全 TS0 分段测速原始结果 | active | `bc7c195d4c57` |
| `fig05_speed/r2d_b16_profile.json` | r2d B=16 fp32 扩展 batch 全量验收结果 | active | `289757842a95` |
| `fig05_speed/plana_b16_profile.json` | plana B=16 fp32 扩展 batch 全量验收结果 | active | `47865dc1f409` |
| `fig05_speed/xens2_b8_profile.json` | xens2 B=8 fp32 扩展 batch 全量验收结果 | active | `1e39284598ab` |
| `fig05_speed/r2d_b1--bf16-r2d-encoder_profile.json` | r2d B=1 bf16 独占负优化结果，不得入选最快配置 | active | `7f0886b1fb40` |
| `fig05_speed/r2d_b4--bf16-r2d-encoder_profile.json` | r2d B=4 bf16 负优化且等价门失败结果 | active | `660418765034` |
| `fig05_speed/head_only_b256_fp16_repro.json` | 4090 历史 head-only 复现（117.735 seq/s、历史计数一致） | active | `028b41138490` |
| `fig05_speed/speed_round36_profile_summary.json` | 六份基线 profile 的阶段耗时与占比汇总 | active | `84438d5f213d` |
| `round32_verified_source_data.json` | 主图 1 主 benchmark、主图 2 历史速度与泄露审计的 round32 已验证汇总源 | active | `7d0cb98abfad` |
| `fig04_calibration/fig04_ece_confidence_source.json` | 主图 4 ECE/Spearman/n 源数据 | active | `d12c6524197f` |
| `fig05_speed/fig05_accuracy_speed_4090_final_3840x1920.pdf` | 主图 5 最终矢量 PDF（文字放大版） | active | `1014ceade7e5` |
| `fig05_speed/fig05_accuracy_speed_4090_final_3840x1920.png` | 主图 5 最终 PNG（3840×1920，文字放大版） | active | `39e15b4795f3` |
| `fig05_speed/fig05_accuracy_speed_4090_final_source.json` | 主图 5 最终源数据（15 格、Pareto、跨 batch 等价） | active | `6ead896ef80f` |
| `fig05_speed/mxfold2_b1_full.json` | RTX 4090 测速原始 JSON | active | `74a31268f706` |
| `fig05_speed/mxfold2_b2_full.json` | RTX 4090 测速原始 JSON | active | `9e7e9d86c2e8` |
| `fig05_speed/mxfold2_b4_full.json` | RTX 4090 测速原始 JSON | active | `412551cc96f2` |
| `fig05_speed/plana_b1_full.json` | RTX 4090 测速原始 JSON | active | `2414b2cee920` |
| `fig05_speed/plana_b2_full.json` | RTX 4090 测速原始 JSON | active | `0491a2f9aeb8` |
| `fig05_speed/plana_b4_full.json` | RTX 4090 测速原始 JSON | active | `9927e19b3939` |
| `fig05_speed/r2d_b1_full.json` | RTX 4090 测速原始 JSON | active | `c9915ccc5b11` |
| `fig05_speed/r2d_b2_full.json` | RTX 4090 测速原始 JSON | active | `5118d9420da2` |
| `fig05_speed/r2d_b4_full.json` | RTX 4090 测速原始 JSON | active | `3dbe91160395` |
| `fig05_speed/rinalmo_ft_b1_full.json` | RTX 4090 测速原始 JSON | active | `a2a4b885f12f` |
| `fig05_speed/rinalmo_ft_b2_full.json` | RTX 4090 测速原始 JSON | active | `24b05d9f8878` |
| `fig05_speed/rinalmo_ft_b4_full.json` | RTX 4090 测速原始 JSON | active | `67189bffa7b7` |
| `fig05_speed/xens2_b1_full.json` | RTX 4090 测速原始 JSON | active | `18cbb013ba69` |
| `fig05_speed/xens2_b2_full.json` | RTX 4090 测速原始 JSON | active | `fe7b78f53e27` |
| `fig05_speed/xens2_b4_full.json` | RTX 4090 测速原始 JSON | active | `551a14149b63` |
| `fig06_architecture/fig06_architecture_contract.json` | 主图 6 架构参数与张量合同（代码/配置复核） | active | `543fc71def6a` |
| `fig07_nucleicbert/fig07_nucleicbert_combined_working_source.json` | 主图 7 主 benchmark + 原论文任务组合 working 源数据 | active | `06df2bc429d2` |
| `fig07_nucleicbert/fig07_nucleicbert_original_round35_3840x1920.pdf` | 主图 7 原论文任务工作版矢量 PDF | active | `f8d67d5aee82` |
| `fig07_nucleicbert/fig07_nucleicbert_original_round35_3840x1920.png` | 主图 7 原论文任务（acceptor/donor/CPEB3）工作版 PNG（3840×1920） | active | `489971c2ad11` |
| `fig07_nucleicbert/fig07_nucleicbert_original_tasks_working_source.json` | 主图 7 原论文任务 working 源数据 | active | `53346872a227` |
| `fig08a_classification/archiveii_diag_5s_s0.json` | T-A148 ArchiveII 5s 崩塌归因诊断（as-trained/偏置/阈值解码） | active | `41b02b3ce3e1` |
| `fig08a_classification/archiveii_diag_tRNA_s0.json` | T-A148 ArchiveII tRNA JEV 崩塌归因诊断（解码/偏置/泄漏） | active | `ca828f5ffe1b` |
| `fig08a_classification/fig08a_rinalmo_classification_round35_3840x1920.pdf` | 主图 8A 分类任务工作版矢量 PDF | active | `32a97ca369c7` |
| `fig08a_classification/fig08a_rinalmo_classification_round35_3840x1920.png` | 主图 8A 分类任务工作版 PNG（3840×1920） | active | `bda17a2dea7e` |
| `fig08a_classification/fig08a_rinalmo_classification_working_source.json` | 主图 8A 分类任务 working 源数据（5 seeds、Holm） | active | `ee033ff585e9` |
| `fig08b_regression/TE_EL_TENFOLD_AGGREGATE.json` | Muscle TE/EL 十折聚合（折级符号秩与 Holm） | active | `40bf2c2cc3b2` |
| `fig08b_regression/fig08b_rinalmo_regression_foldlevel_working_3840x1920.pdf` | 主图 8B 回归任务工作版矢量 PDF | active | `10239824a579` |
| `fig08b_regression/fig08b_rinalmo_regression_foldlevel_working_3840x1920.png` | 主图 8B 回归任务工作版 PNG（3840×1920） | active | `5167af18a163` |
| `fig08b_regression/fig08b_rinalmo_regression_foldlevel_working_source.json` | 主图 8B 回归任务 fold-level working 源数据 | active | `0849041bee44` |
| `appendix_leakage/leakage_aware_baseline_source.json` | 附录 A2 泄露处理完整基线表（19 系统、clean-181、排除格与审计边界） | active | `e5ca478111e0` |
| `appendix_leakage/rnaformer_protocol_audit.json` | RNAformer 原文/复测、固定/特殊 checkpoint 与双 GT 审计 | active | `e32d4875e601` |
| `appendix_leakage/main_benchmark_4090_reproduction.json` | xens2 八 split 4090/A100 精度复现（最大绝对差 0.0000903） | active | `e6f4f51f85f5` |
| `appendix_leakage/build_source.py` | 上述源表的可复现构建脚本 | active | `204d22a54a70` |

## 待补

- `fig05_speed/`：round36 分段 profile、无损优化与同机完整基线结果。
- `fig07_nucleicbert/`：二级结构五种子（batch=4 单路在训，测速窗口后恢复）。
- `fig08b_regression/`：HEK fold4–9 与 pc3 十折（目前仅 Muscle 十折完成；pc3 暂停）。
