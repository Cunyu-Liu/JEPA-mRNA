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
| `fig05_speed/plana_b32_profile.json` | plana B=32 fp32 全 TS0 三次原始结果与等价门 | active | `63b64d47d643` |
| `fig05_speed/r2d_b32_profile.json` | r2d B=32 fp32 全 TS0 三次原始结果与等价门 | active | `d4d70a477b26` |
| `fig05_speed/r2d_b16--tf32_profile.json` | r2d B=16 TF32 全 TS0 三次原始结果；通过但不优于 B=32 | active | `b74ef461f388` |
| `fig05_speed/xens2_b16_profile.json` | xens2 B=16 fp32 全 TS0 三次原始结果与等价门 | active | `419a26db79b3` |
| `fig05_speed/xens2_b32_profile.json` | xens2 B=32 fp32 全 TS0 三次原始结果与等价门 | active | `0919bf3fa171` |
| `fig05_speed/speed_round36_summary.json` | round36 基线、优化候选、最优 eligible 与 RNAformer 汇总 | active | `ba4cfd8ca1b5` |
| `fig05_speed/build_round36_profile_summary.py` | 六份基线 profile 汇总构建脚本 | active | `db9da4148df8` |
| `fig05_speed/build_round36_speed_comparison.py` | eligible 优化与参照模型汇总构建脚本 | active | `d07b0a8e8c61` |
| `fig05_speed/rnaformer_b1_full.json` | RNAformer 4090 B=1 全 TS0 三次 raw-sequence 原始 JSON | active | `22f42726ea0d` |
| `fig05_speed/rnaformer_b2_full.json` | RNAformer 4090 B=2 全 TS0 三次 raw-sequence 原始 JSON | active | `5ae1e83147d2` |
| `fig05_speed/rnaformer_b4_full.json` | RNAformer 4090 B=4 全 TS0 三次 raw-sequence 原始 JSON | active | `e49e5ba613c5` |
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
| `appendix_leakage/leakage_aware_baseline_source.json` | 附录 A2 泄露审计与 clean 子集复算表；eFold 旧板标 0.1.2 | active | `997d2f5d6846` |
| `appendix_leakage/rnaformer_protocol_audit.json` | RNAformer 原文/复测、固定/特殊 checkpoint 与双 GT 审计 | active | `e32d4875e601` |
| `appendix_leakage/main_benchmark_4090_reproduction.json` | xens2 八 split 4090/A100 精度复现（最大绝对差 0.0000903） | active | `e6f4f51f85f5` |
| `appendix_leakage/build_source.py` | 泄露审计源表构建脚本；修复 clean split 键映射与版本身份 | active | `fe732350b7bb` |
| `appendix_leakage/rnaformer_bprna_clean_recompute.json` | RNAformer 四 split clean 子集独立复算 | active | `8d2819355248` |
| `fig05_speed/round38_baseline_assets.json` | 两项正式测速与三项官方权重 blocked 的资产和版本证据 | active | `20bab1641c3e` |
| `fig02_speed/fig02_end_to_end_speed_4090_round38_3840x1920.pdf` | 主图 2 八系统端到端速度矢量 PDF | active | `780e64b34705` |
| `fig02_speed/fig02_end_to_end_speed_4090_round38_3840x1920.png` | 主图 2 PNG（PPT 嵌入同哈希） | active | `347a8cb5d06c` |
| `fig02_speed/fig02_end_to_end_speed_4090_round38_source.json` | 最优配置、范围、F1、版本与 blocked 边界 | active | `d9407460da16` |
| `fig05_speed/fig05_accuracy_speed_4090_round38_3840x1920.pdf` | 主图 5 精度-吞吐与共同 batch 矢量 PDF | active | `0d80fb5dbe57` |
| `fig05_speed/fig05_accuracy_speed_4090_round38_3840x1920.png` | 主图 5 PNG（PPT 嵌入同哈希） | active | `ce6425028ac0` |
| `fig05_speed/fig05_accuracy_speed_4090_round38_source.json` | 29 个配置及 eFold 新旧协议审计 | active | `e43cf0b793d8` |
| `fig05_speed/round38_speed_validation.json` | 数值等价、旧版 rounded F1 与计数差验收 | active | `baff5aea6754` |
| `fig05_speed/efold_b1.json` | eFold 0.1.3 B=1 全 TS0 三次原始结果 | active | `f0bdb75396ac` |
| `fig05_speed/efold_b2.json` | eFold 0.1.3 B=2 全 TS0 三次原始结果 | active | `96237cd5fbbf` |
| `fig05_speed/efold_b4.json` | eFold 0.1.3 B=4 全 TS0 三次原始结果 | active | `009809fde033` |
| `fig05_speed/eternafold_b1.json` | EternaFold 官方 serial CPU 全 TS0 三次原始结果 | active | `4eeba4000d35` |
| `fig05_speed/efold012_b1.json` | eFold 0.1.2 原生逐条状态重置三次版本核验 | active | `1e4f939ef692` |
| `fig09_ood/build_ood_failure_source.py` | GenC/GenF 逐家族独立重算脚本 | active | `448bbc0039d4` |
| `fig09_ood/round38_ood_failure_source.json` | 家族失败、零正确配对、配对密度与 OOD mean 边界 | active | `a6e4a343d7ff` |
| `appendix_leakage/round39_eternafold_raw_summary.json` | EternaFold 八 split 原始集三次复测及独立计数重算；不是逐模型 clean 结论 | active | `d1310c1386b8` |
| `appendix_leakage/round39_contrafold_raw_summary.json` | 原生 CONTRAfold 八 split 三次复测及独立计数重算；不是逐模型 clean 结论 | active | `d7cbdd7aa666` |
| `appendix_leakage/round39_eternafold_clean_tr1c_summary.json` | EternaFold 在 tr1c 80% identity clean 子集上的八 split 三次复测 | active | `4af21aeff828` |
| `appendix_leakage/round39_contrafold_clean_tr1c_summary.json` | 原生 CONTRAfold 在 tr1c 80% identity clean 子集上的八 split 三次复测 | active | `195274ef57b0` |
| `appendix_leakage/round39_ufold_raw_summary.json` | UFold 八 split 原始集三次复测；clean 训练并集 provenance 未闭环 | active | `e22f77455ae5` |
| `appendix_leakage/round39_clean_audits/INDEX.json` | TR0/tr1c 各八 split clean n、删除数、数量守恒与审计哈希索引 | active | `699c259064ac` |

## 后续边界

- `fig05_speed/`：eFold 0.1.3 与 EternaFold 正式三次测速完成；BPfold/UFold/E2Efold 官方权重 blocked，不作无限制速度 SOTA。round37 图件已归档。
- `fig07_nucleicbert/`：official_s3 已从 epoch5 checkpoint 恢复，尚未收敛；全部种子与独立 QA 总门未关闭。
- `fig08b_regression/`：Muscle 十折完成；HEK/pc3 未满十折不进图，本轮不新增任务。
- `fig09_ood/`：GenA 已领先，GenC/GenF 是短板；20.8%/39.2% 是零正确配对率而非零输出率。无新提升证据，不重复旧训练。
