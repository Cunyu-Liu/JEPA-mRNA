# bpRNA TS0 / TestSetB 官方全集与 projected-reference 双面板

面板构建仅对冻结及已恢复预测做 CPU 复算，不做训练、新推理、阈值调整或模型选择。TS0 的 17 条补充推理在独立恢复阶段完成；项目子集结果是 `DERIVED_FROM_OFFICIAL_REFERENCE_PROJECTION` 的显式 projected-reference 协议，不与未过滤官方全集混排。缺预测不补 0，部分覆盖不排名；共享 registry 未修改。

严格 zero-based inclusive 配对，不追加过滤；macro 是逐序列 P/R/F1 非加权平均，仅 empty/empty 时 P/R/F1=1。TS0 官方唯一 truth-empty 为 `bpRNA_SRP_192.bpseq`，只在 full1305 全覆盖身份中计分；部分覆盖身份仍不排名。TestSetB 无 truth-empty。

## bprna_ts0：官方全集 1305 结果

| Prediction identity | Model form | n | micro F1 | macro F1 | truth-empty | empty/empty |
|---|---|---:|---:|---:|---:|---:|
| `bpfold_official_release/bprna_ts0` | six_checkpoint_ensemble | 1305 | 0.6812363564 | 0.6574981350 | 1 | 0 |
| `eternafold/bprna_ts0` | deterministic_single_predictor | 1305 | 0.5391623606 | 0.5391319526 | 1 | 0 |
| `vienna_centroid/bprna_ts0/recovered_official17` | deterministic_single_predictor | 1305 | 0.5389168251 | 0.5281497484 | 1 | 0 |
| `contrafold/bprna_ts0` | deterministic_single_predictor | 1305 | 0.5380230780 | 0.5362908433 | 1 | 0 |
| `vienna_mea/bprna_ts0/recovered_official17` | deterministic_single_predictor | 1305 | 0.5237175605 | 0.5211055529 | 1 | 0 |
| `vienna_mfe/bprna_ts0/recovered_official17` | deterministic_single_predictor | 1305 | 0.5054624747 | 0.5080129860 | 1 | 0 |
| `nussinov_turner/bprna_ts0/recovered_official17` | deterministic_single_predictor | 1305 | 0.2112852889 | 0.2293670906 | 1 | 0 |

全集结果仅作 frozen baseline 描述性覆盖审计，不用于选模；部分覆盖身份不进入本表。

## bprna_ts0：projected 1288 子集结果

| Prediction identity | Model form | n | micro F1 | macro F1 | Δmicro vs xens2 | Δmacro vs xens2 |
|---|---|---:|---:|---:|---:|---:|
| `xens2/bprna_ts0` | three_checkpoint_ensemble | 1288 | 0.8032405715 | 0.7999193349 | +0.0000000000 | +0.0000000000 |
| `rnaformer_bprna/bprna_ts0` | single_model | 1288 | 0.7568974968 | 0.7444430454 | -0.0463430747 | -0.0554762895 |
| `bpfold/bprna_ts0` | six_checkpoint_ensemble | 1288 | 0.7501927623 | 0.7365779928 | -0.0530478092 | -0.0633413421 |
| `kirigami/bprna_ts0` | single_model | 1288 | 0.7296615755 | 0.7261263011 | -0.0735789960 | -0.0737930338 |
| `kirigami_fast/bprna_ts0` | single_model | 1288 | 0.7296615755 | 0.7261263011 | -0.0735789960 | -0.0737930338 |
| `rinalmo_ft/bprna_ts0` | single_model | 1288 | 0.7208173117 | 0.7241314670 | -0.0824232597 | -0.0757878679 |
| `bpfold_official_release/bprna_ts0` | six_checkpoint_ensemble | 1288 | 0.6804056016 | 0.6567174489 | -0.1228349698 | -0.1432018860 |
| `rinalmo_2d_s1/bprna_ts0` | single_model | 1288 | 0.6499831872 | 0.6603953076 | -0.1532573843 | -0.1395240273 |
| `rinalmo_2d_s2/bprna_ts0` | single_model | 1288 | 0.6429869240 | 0.6408665162 | -0.1602536475 | -0.1590528187 |
| `rnaformer_intrafamily/bprna_ts0` | single_model | 1288 | 0.6389316470 | 0.6437297979 | -0.1643089244 | -0.1561895370 |
| `rnaformer_interfamily/bprna_ts0` | single_model | 1288 | 0.6384644315 | 0.6418755620 | -0.1647761400 | -0.1580437729 |
| `mxfold2/bprna_ts0` | deterministic_single_predictor | 1288 | 0.5647599096 | 0.5695812107 | -0.2384806619 | -0.2303381242 |
| `efold_0_1_3/bprna_ts0` | single_model | 1288 | 0.5503063121 | 0.5674471035 | -0.2529342594 | -0.2324722314 |
| `eternafold/bprna_ts0` | deterministic_single_predictor | 1288 | 0.5396620336 | 0.5392145247 | -0.2635785379 | -0.2607048103 |
| `vienna_centroid/bprna_ts0/recovered_official17` | deterministic_single_predictor | 1288 | 0.5388863717 | 0.5285524883 | -0.2643541998 | -0.2713668466 |
| `contrafold/bprna_ts0` | deterministic_single_predictor | 1288 | 0.5388044914 | 0.5365088055 | -0.2644360801 | -0.2634105294 |
| `rinalmo_2d_s0/bprna_ts0` | single_model | 1288 | 0.5284031939 | 0.5579239135 | -0.2748373776 | -0.2419954214 |
| `vienna_mea/bprna_ts0/recovered_official17` | deterministic_single_predictor | 1288 | 0.5237635397 | 0.5215667726 | -0.2794770318 | -0.2783525623 |
| `rnastructure_maxexpect/bprna_ts0` | deterministic_single_predictor | 1288 | 0.5215778237 | 0.5188606058 | -0.2816627478 | -0.2810587291 |
| `rnaformer_biophysical/bprna_ts0` | single_model | 1288 | 0.5056427227 | 0.5094441970 | -0.2975978488 | -0.2904751379 |
| `vienna_mfe/bprna_ts0/recovered_official17` | deterministic_single_predictor | 1288 | 0.5052049705 | 0.5083536608 | -0.2980356010 | -0.2915656741 |
| `rnastructure_fold/bprna_ts0` | deterministic_single_predictor | 1288 | 0.5018261255 | 0.5067783348 | -0.3014144459 | -0.2931410001 |
| `nucleicbert_ft/bprna_ts0` | single_model | 1288 | 0.4606195215 | 0.5427560917 | -0.3426210500 | -0.2571632432 |
| `ufold/bprna_ts0` | single_model | 1288 | 0.4604204516 | 0.4976113824 | -0.3428201199 | -0.3023079525 |
| `nucleicbert_frozen_official/bprna_ts0` | single_model | 1288 | 0.4024960380 | 0.3674415140 | -0.4007445335 | -0.4324778209 |
| `nussinov_turner/bprna_ts0/recovered_official17` | deterministic_single_predictor | 1288 | 0.2122805736 | 0.2301621166 | -0.5909599979 | -0.5697572183 |

### 最强单预测器与集成（描述性，不用于选模）

- **最强集成/总体**：`xens2/bprna_ts0`，micro-F1=0.8032405714870655，macro-F1=0.7999193349056970；Δmicro=+0.0000000000000000，Δmacro=+0.0000000000000000，配对 macro Δ95% CI [+0.0000000000000000, +0.0000000000000000]。训练暴露栏：`HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN`（route=bprna_tr1c）。
- **最强单预测器**：`rnaformer_bprna/bprna_ts0`，micro-F1=0.7568974968024850，macro-F1=0.7444430454311176；Δmicro=-0.0463430746845805，Δmacro=-0.0554762894745794，配对 macro Δ95% CI [-0.0674523877056411, -0.0435507990639508]。训练暴露栏：`UNCHECKED`（route=UNCHECKED）。

### 官方独有 17 条与覆盖状态

官方比项目子集多 17 条；当前 official-full 全覆盖 7 个，部分覆盖不排名 19 个，blocked 14 个。逐 ID 恢复来源见 `ts0-official-coverage-recovery.md`（TS0）；未恢复身份不以空结构或 0 分补齐。

官方独有 ID（均为项目 bad-alphabet 拒绝，异常字符在括号内）：`bpRNA_CRW_25863.bpseq` (N)、`bpRNA_CRW_31704.bpseq` (N)、`bpRNA_CRW_43565.bpseq` (N)、`bpRNA_CRW_54323.bpseq` (N)、`bpRNA_CRW_54562.bpseq` (N)、`bpRNA_RFAM_10644.bpseq` (Y)、`bpRNA_RFAM_1072.bpseq` (N)、`bpRNA_RFAM_18822.bpseq` (RY)、`bpRNA_RFAM_18935.bpseq` (RY)、`bpRNA_RFAM_38651.bpseq` (N)、`bpRNA_SRP_192.bpseq` (N)、`bpRNA_tmRNA_252.bpseq` (N)、`bpRNA_tmRNA_397.bpseq` (N)、`bpRNA_tmRNA_643.bpseq` (N)、`bpRNA_tmRNA_651.bpseq` (N)、`bpRNA_tmRNA_686.bpseq` (N)、`bpRNA_tmRNA_693.bpseq` (N)。

逐预测 missing ID 全表见 `official_coverage_gaps.json`；summary-only/blocked 的 metrics=null，不补 0。

### 训练暴露分栏

以下为数据集级路线事实；TR1 homology 未检查。模型相关暴露另按 prediction identity 列出，不能用标签独立替代训练独立。

| 路线 | 精确序列重叠 | homology-only | clean n | clean SHA 重建 |
|---|---:|---:|---:|---|
| bprna_tr0 | 0 | 96 | 1192 | EXACT |
| bprna_tr1 | 1087 | UNCHECKED | UNCHECKED | UNCHECKED |
| bprna_tr1c | 0 | 575 | 713 | EXACT |

| Prediction identity | 训练语料/路线 | 精确重叠 | homology-only | clean n | 全项目面板暴露判定 |
|---|---|---:|---:|---:|---|
| `xens2/bprna_ts0` | bprna_tr1c | 0 | 575 | 713 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rnaformer_bprna/bprna_ts0` | official bprna_data.plk | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `bpfold/bprna_ts0` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `kirigami/bprna_ts0` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `kirigami_fast/bprna_ts0` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `rinalmo_ft/bprna_ts0` | bprna_tr0 | 0 | 96 | 1192 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `bpfold_official_release/bprna_ts0` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `rinalmo_2d_s1/bprna_ts0` | bprna_tr1c | 0 | 575 | 713 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rinalmo_2d_s2/bprna_ts0` | bprna_tr1c | 0 | 575 | 713 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rnaformer_intrafamily/bprna_ts0` | official intra-family training data | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `rnaformer_interfamily/bprna_ts0` | official inter-family training data | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `mxfold2/bprna_ts0` | mxfold2_trainsetab | 61 | 39 | 1188 | `AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED` |
| `efold_0_1_3/bprna_ts0` | bpRNA subset+RNAstralign+Ribonanza | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `eternafold/bprna_ts0` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `vienna_centroid/bprna_ts0/recovered_official17` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |
| `contrafold/bprna_ts0` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `rinalmo_2d_s0/bprna_ts0` | bprna_tr1c | 0 | 575 | 713 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `vienna_mea/bprna_ts0/recovered_official17` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |
| `rnastructure_maxexpect/bprna_ts0` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |
| `rnaformer_biophysical/bprna_ts0` | official biophysical_model_data.plk train=410408 | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `vienna_mfe/bprna_ts0/recovered_official17` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |
| `rnastructure_fold/bprna_ts0` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |
| `nucleicbert_ft/bprna_ts0` | bprna_tr1c | 0 | 575 | 713 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `ufold/bprna_ts0` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `nucleicbert_frozen_official/bprna_ts0` | rnastralign_tr0_style | 72 | 172 | 1044 | `AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED` |
| `nussinov_turner/bprna_ts0/recovered_official17` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |

外部模型的未核实训练语料保持 `UNCHECKED`；MXfold2/NucleicBERT 的审计子集证据不升级为完整训练暴露排除证明。逐模型 provenance 原文和 clean SHA 见 `system_metrics.json`。

## testsetb：官方全集 430 结果

| Prediction identity | Model form | n | micro F1 | macro F1 | truth-empty | empty/empty |
|---|---|---:|---:|---:|---:|---:|
| `mxfold2/testsetb` | deterministic_single_predictor | 430 | 0.6468595682 | 0.6692063276 | 0 | 0 |
| `eternafold/testsetb` | deterministic_single_predictor | 430 | 0.5816473147 | 0.5997317048 | 0 | 0 |
| `contrafold/testsetb` | deterministic_single_predictor | 430 | 0.5633544581 | 0.5793087893 | 0 | 0 |
| `vienna_centroid/testsetb/recovered_official2` | single_model | 430 | 0.5452759933 | 0.5547336489 | 0 | 0 |
| `vienna_mea/testsetb/recovered_official2` | single_model | 430 | 0.5365910503 | 0.5519406982 | 0 | 0 |
| `vienna_mfe/testsetb/recovered_official2` | single_model | 430 | 0.5202373304 | 0.5404191111 | 0 | 0 |
| `nussinov_turner/testsetb/recovered_official2` | single_model | 430 | 0.2158239143 | 0.2240284299 | 0 | 0 |

全集结果仅作 frozen baseline 描述性覆盖审计，不用于选模；部分覆盖身份不进入本表。

## testsetb：projected 428 子集结果

| Prediction identity | Model form | n | micro F1 | macro F1 | Δmicro vs xens2 | Δmacro vs xens2 |
|---|---|---:|---:|---:|---:|---:|
| `rinalmo_ft/testsetb` | single_model | 428 | 0.8670430034 | 0.8628944070 | +0.0267227911 | +0.0306466007 |
| `kirigami_fast/testsetb` | single_model | 428 | 0.8421929216 | 0.8343000298 | +0.0018727094 | +0.0020522235 |
| `xens2/testsetb` | three_checkpoint_ensemble | 428 | 0.8403202122 | 0.8322478063 | +0.0000000000 | +0.0000000000 |
| `rnaformer_bprna/testsetb` | single_model | 428 | 0.7951901076 | 0.7711776661 | -0.0451301046 | -0.0610701403 |
| `bpfold/testsetb` | six_checkpoint_ensemble | 428 | 0.7881461750 | 0.7949848207 | -0.0521740372 | -0.0372629856 |
| `rinalmo_2d_s1/testsetb` | single_model | 428 | 0.7285987151 | 0.7434270615 | -0.1117214972 | -0.0888207448 |
| `rinalmo_2d_s2/testsetb` | single_model | 428 | 0.7178103315 | 0.7146001239 | -0.1225098807 | -0.1176476824 |
| `rnaformer_interfamily/testsetb` | single_model | 428 | 0.6685137188 | 0.6637434237 | -0.1718064934 | -0.1685043826 |
| `rnaformer_intrafamily/testsetb` | single_model | 428 | 0.6650966452 | 0.6586049650 | -0.1752235670 | -0.1736428413 |
| `mxfold2/testsetb` | deterministic_single_predictor | 428 | 0.6473580571 | 0.6698447616 | -0.1929621551 | -0.1624030447 |
| `eternafold/testsetb` | deterministic_single_predictor | 428 | 0.5823802646 | 0.6004962870 | -0.2579399476 | -0.2317515193 |
| `contrafold/testsetb` | deterministic_single_predictor | 428 | 0.5638971236 | 0.5798955998 | -0.2764230886 | -0.2523522065 |
| `efold_0_1_3/testsetb` | single_model | 428 | 0.5545354109 | 0.5705247034 | -0.2857848013 | -0.2617231030 |
| `vienna_centroid/testsetb/recovered_official2` | single_model | 428 | 0.5456713963 | 0.5551500445 | -0.2946488160 | -0.2770977618 |
| `rinalmo_2d_s0/testsetb` | single_model | 428 | 0.5382570989 | 0.5531410266 | -0.3020631133 | -0.2791067797 |
| `vienna_mea/testsetb/recovered_official2` | single_model | 428 | 0.5369090838 | 0.5523298757 | -0.3034111284 | -0.2799179306 |
| `rnastructure_maxexpect/testsetb` | deterministic_single_predictor | 428 | 0.5269634739 | 0.5453195038 | -0.3133567383 | -0.2869283025 |
| `vienna_mfe/testsetb/recovered_official2` | single_model | 428 | 0.5202597501 | 0.5405667540 | -0.3200604621 | -0.2916810523 |
| `rnaformer_biophysical/testsetb` | single_model | 428 | 0.5111127968 | 0.5314199088 | -0.3292074154 | -0.3008278976 |
| `rnastructure_fold/testsetb` | deterministic_single_predictor | 428 | 0.5030067748 | 0.5222784016 | -0.3373134375 | -0.3099694047 |
| `nucleicbert_ft/testsetb` | single_model | 428 | 0.4842599509 | 0.5244210001 | -0.3560602613 | -0.3078268062 |
| `ufold/testsetb` | single_model | 428 | 0.4508291874 | 0.4781111939 | -0.3894910248 | -0.3541366124 |
| `nucleicbert_frozen_official/testsetb` | single_model | 428 | 0.3652332314 | 0.3126424606 | -0.4750869808 | -0.5196053458 |
| `nussinov_turner/testsetb/recovered_official2` | single_model | 428 | 0.2154385685 | 0.2237670841 | -0.6248816437 | -0.6084807223 |

### 最强单预测器与集成（描述性，不用于选模）

- **最强集成/总体**：`xens2/testsetb`，micro-F1=0.8403202122116432，macro-F1=0.8322478063381703；Δmicro=+0.0000000000000000，Δmacro=+0.0000000000000000，配对 macro Δ95% CI [+0.0000000000000000, +0.0000000000000000]。训练暴露栏：`HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN`（route=bprna_tr1c）。
- **最强单预测器**：`rinalmo_ft/testsetb`，micro-F1=0.8670430033503144，macro-F1=0.8628944070360038；Δmicro=+0.0267227911386712，Δmacro=+0.0306466006978335，配对 macro Δ95% CI [+0.0169091444489763, +0.0452470510286115]。训练暴露栏：`EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT`（route=bprna_tr0）。

### 官方独有 2 条与覆盖状态

官方比项目子集多 2 条；当前 official-full 全覆盖 7 个，部分覆盖不排名 17 个，blocked 14 个。逐 ID 恢复来源见 `ts0-official-coverage-recovery.md`（TS0）；未恢复身份不以空结构或 0 分补齐。

官方独有 ID（均为项目 bad-alphabet 拒绝，异常字符在括号内）：`testsetb_104.bpseq` (S)、`testsetb_36.bpseq` (N)。

逐预测 missing ID 全表见 `official_coverage_gaps.json`；summary-only/blocked 的 metrics=null，不补 0。

### 训练暴露分栏

以下为数据集级路线事实；TR1 homology 未检查。模型相关暴露另按 prediction identity 列出，不能用标签独立替代训练独立。

| 路线 | 精确序列重叠 | homology-only | clean n | clean SHA 重建 |
|---|---:|---:|---:|---|
| bprna_tr0 | 247 | 65 | 116 | EXACT |
| bprna_tr1 | 358 | UNCHECKED | UNCHECKED | UNCHECKED |
| bprna_tr1c | 0 | 140 | 288 | EXACT |

| Prediction identity | 训练语料/路线 | 精确重叠 | homology-only | clean n | 全项目面板暴露判定 |
|---|---|---:|---:|---:|---|
| `rinalmo_ft/testsetb` | bprna_tr0 | 247 | 65 | 116 | `EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT` |
| `kirigami_fast/testsetb` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `xens2/testsetb` | bprna_tr1c | 0 | 140 | 288 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rnaformer_bprna/testsetb` | official bprna_data.plk | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `bpfold/testsetb` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `rinalmo_2d_s1/testsetb` | bprna_tr1c | 0 | 140 | 288 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rinalmo_2d_s2/testsetb` | bprna_tr1c | 0 | 140 | 288 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rnaformer_interfamily/testsetb` | official inter-family training data | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `rnaformer_intrafamily/testsetb` | official intra-family training data | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `mxfold2/testsetb` | mxfold2_trainsetab | 0 | 55 | 373 | `AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED` |
| `eternafold/testsetb` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `contrafold/testsetb` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `efold_0_1_3/testsetb` | bpRNA subset+RNAstralign+Ribonanza | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `vienna_centroid/testsetb/recovered_official2` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |
| `rinalmo_2d_s0/testsetb` | bprna_tr1c | 0 | 140 | 288 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `vienna_mea/testsetb/recovered_official2` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |
| `rnastructure_maxexpect/testsetb` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |
| `vienna_mfe/testsetb/recovered_official2` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |
| `rnaformer_biophysical/testsetb` | official biophysical_model_data.plk train=410408 | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `rnastructure_fold/testsetb` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |
| `nucleicbert_ft/testsetb` | bprna_tr1c | 0 | 140 | 288 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `ufold/testsetb` | UNCHECKED | UNCHECKED | UNCHECKED | UNCHECKED | `UNCHECKED` |
| `nucleicbert_frozen_official/testsetb` | rnastralign_tr0_style | 247 | 65 | 116 | `AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED` |
| `nussinov_turner/testsetb/recovered_official2` | NOT_SEQUENCE_FITTED | UNCHECKED | UNCHECKED | UNCHECKED | `NOT_SEQUENCE_FITTED` |

外部模型的未核实训练语料保持 `UNCHECKED`；MXfold2/NucleicBERT 的审计子集证据不升级为完整训练暴露排除证明。逐模型 provenance 原文和 clean SHA 见 `system_metrics.json`。
