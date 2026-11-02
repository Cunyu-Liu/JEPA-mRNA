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
| `appendix_leakage/round39_model_clean_audits.json` | 九套模型专属训练并集的八 split clean 审计（含 DEBFold、RNAformer-biophysical） | active | `cc3455b81232` |
| `appendix_leakage/round39_eternafold_clean_train_union_summary.json` | EternaFold 自身训练语料 clean 八 split 三次复测及独立计数重算 | active | `6f2846d15522` |
| `appendix_leakage/round39_mxfold2_raw_summary.json` | MXfold2 八 split 原始集三次复测；ArchiveII 为完整 2544 条 | active | `9d690e989f18` |
| `appendix_leakage/round39_mxfold2_clean_trainsetab_summary.json` | MXfold2 在 TrainSetA+B clean 子集上的八 split 三次复测及独立计数重算 | active | `0b3003292573` |
| `appendix_leakage/round39_ufold_raw_summary.json` | UFold 八 split 原始集三次复测；clean 训练并集 provenance 未闭环 | active | `e22f77455ae5` |
| `appendix_leakage/round39_vienna_raw_summary.json` | Vienna 三系统与 Nussinov-Turner 八 split 三次复测；非序列训练系统 | active | `5769a0dd04f4` |
| `appendix_leakage/round39_invalid_results.json` | UFold 通道错配、SPOT-RNA 空解码与 MXfold2 长度子集诊断索引 | active | `12b83fb4b77e` |
| `appendix_leakage/round39_clean_audits/INDEX.json` | TR0/tr1c 各八 split clean n、删除数、数量守恒与审计哈希索引 | active | `699c259064ac` |
| `appendix_leakage/round39_bpfold_clean_train_union_summary.json` | BPfold 官方六 checkpoint 在自身训练并集 clean 子集上的八 split 三次复算 | active | `4c071f0bc34f` |
| `appendix_leakage/round39_rnastructure_raw_summary.json` | RNAstructure Fold/MaxExpect 八 split 三次复算；固定取第一 MEA 结构 | active | `50d181fec340` |
| `appendix_leakage/round39_efold_raw_summary.json` | eFold 八 split 原始集三次复算；单线程修复后结果，clean provenance 未闭环 | active | `4168671fd757` |
| `appendix_leakage/round39_xens2_student_s0_qa.json` | xens2 单学生 seed0 三次独立 VL0 CUDA QA；单种子过程证据 | active | `b88f43b0d2ed` |
| `appendix_leakage/round39_xens2_distill_objective_comparison.json` | 旧四项目标与纯教师 KL 的 VL0 QA、教师保真度及 TS0 对照 | active | `b6ab2b0a12c6` |
| `appendix_leakage/round39_plana_distill_s0_qa.json` | 单 plana 学生 seed0 收敛、三次 VL0 CUDA QA 与 TS0 结果 | active | `4f41f4b04b74` |
| `appendix_leakage/round39_plana_distill_three_seed_vl0_ts0.json` | plana 三种子收敛、VL0 校准和 TS0 均值/SD；clean 主板待完成 | active | `938e1ae6e056` |
| `appendix_leakage/round39_plana_clean_three_seed_progress.json` | clean 8/8 三训练种子独立聚合；结果完整但未达 SOTA，旧版已归档 | active complete | `79520874a26c` |
| `appendix_leakage/round39_nucleicbert_ft_progress.json` | NucleicBERT fine-tuned 原 B=4 raw 与 clean 各 8/8；独立计数，旧 1/8 汇总已归档 | active complete | `0baaec44a935` |
| `appendix_leakage/round39_plana_vs_bpfold_matched_diagnostic.json` | 学生与 BPfold 同名、同序列、同 GT 的 clean 交集配对复算；8/8 三种子，旧版已归档 | active diagnostic | `29655670693c` |
| `appendix_leakage/round39_plana_calibration_three_seed_vl0_verification.json` | 三个 VL0 校准的原始数组独立复算；ECE 降低但排序退化，非跨 split 结论 | validation verified | `4c3a7071da3d` |
| `appendix_leakage/round39_fusion_clean_ref_pdb_ts2.json` | VL0 锁定融合在双 clean TS2 的三次结果；n=1，未领先 BPfold | active partial | `22339a903029` |
| `appendix_leakage/round39_fusion_clean_aggregate.json` | 融合在 6 个 clean split 上与 BPfold/student 的同源配对、校准与置信度聚合；旧 5/8 版已归档 | active partial | `1a1e2b0afc27` |
| `appendix_leakage/round39_fusion_candidate_rejection.json` | 融合 TS2/TS3 负格使 8/8 不可达；停止剩余推理，非完整全板 | active rejected candidate | `5e094ea460ef` |
| `appendix_leakage/round39_kirigami_fast_ts0_equivalence.json` | Kirigami C 解码 TS0 三次完整等价与 11.814 倍中位加速 | active engineering | `f6886f443c27` |
| `appendix_leakage/round39_kirigami_fast_bprna_new_equivalence.json` | Kirigami C 解码 bpRNA-new 三次完整等价与 5.813 倍中位加速 | active engineering | `87473ab44930` |
| `appendix_leakage/round39_kirigami_fast_archiveii_diagnostic.json` | ArchiveII 当前算法对照完成；历史 F1 差异仍保留 | active current algorithm verified | `adfdeb4981a1` |
| `appendix_leakage/round39_kirigami_fast_raw_clean_summary.json` | Kirigami C 解码 raw 与 clean 各八 split 独立聚合；当前算法对照完成，保留历史差异 | active complete with boundary | `890e2491b63d` |
| `appendix_leakage/round39_kirigami_archive_protocol_reconcile.json` | ArchiveII 历史逐序列计数与当前预测对账；忽略假结不足以解释 F1 差异 | active diagnostic | `59701fd72ac3` |
| `appendix_leakage/round39_kirigami_archive_200_400_original_vs_c.json` | ArchiveII 200–400 nt 全桶 728 条原 Python/C 配对完全一致 | active complete 728/728 | `891e2650344c` |
| `appendix_leakage/bpfold_vl0.json` | BPfold 全 VL0 三次官方推理与原始 utility 哈希；仅供模型选择 | validation only | `ea21f21814ab` |
| `appendix_leakage/round39_vl0_bpfold_fusion_fit.json` | VL0-only xens2+BPfold 非负 NLL 融合参数与对照；拟合集指标不作最终结论 | validation candidate | `bfde5279469f` |
| `appendix_leakage/round39_mixed_nll_data_audit.json` | 混合 NLL 数据标签清理和 VL0/八 split 去泄露审计 | active data gate | `ee6ddbb038d3` |
| `appendix_leakage/round39_mixed_nll_independent_verification.json` | 1657 条新增序列独立标签与九组零泄露复扫 | active independent verification | `a3c6acbe139d` |
| `appendix_leakage/round39_spotrna_raw_summary.json` | SPOT-RNA 转换权重 raw 八 split 三次独立聚合；clean 待完成 | active raw complete | `dd752681b49b` |
| `appendix_leakage/round39_rinalmo_ft_raw_summary.json` | RiNALMo-ft raw 八 split 三次 full-length 独立聚合；TR0-clean 待完成 | active raw complete | `04ac23a5369e` |
| `appendix_leakage/round39_spotrna_raw_clean_summary.json` | SPOT-RNA raw+clean 16 格独立聚合；TS3 为空，旧 6/8 已归档 | active complete | `dfb80f51e014` |
| `appendix_leakage/round39_spotrna_vl0_calibration.json` | SPOT-RNA 与三个学生 VL0 同掩码校准对照；置信度全部饱和，不作泛化结论 | active VL0 diagnostic | `534ddea6ea41` |

## 后续边界

- `fig05_speed/`：eFold 0.1.3 与 EternaFold 正式三次测速完成；BPfold/UFold 权重已恢复但正式速度待独占复测，E2Efold 权重仍 blocked，不作无限制速度 SOTA。round37 图件已归档。
- `fig07_nucleicbert/`：official_s3 已在 epoch13 完整 checkpoint 安全暂停，best validation BCE=0.011949524802079076，尚未收敛；全部种子与独立 QA 总门未关闭。
- `fig08b_regression/`：Muscle 十折完成；HEK/pc3 未满十折不进图，本轮不新增任务。
- `fig09_ood/`：GenA 已领先，GenC/GenF 是短板；20.8%/39.2% 是零正确配对率而非零输出率。无新提升证据，不重复旧训练。
- `appendix_leakage/`：九套模型专属 clean 审计已完成（含 DEBFold、RNAformer-biophysical）；EternaFold、MXfold2、BPfold 的自身 clean 八 split 三次复测均已闭环，UFold/eFold/CONTRAfold provenance 未闭环。plana 学生 clean TS0 已独立聚合，其他 7 split、独占测速和 VL0-only 校准未验收。
