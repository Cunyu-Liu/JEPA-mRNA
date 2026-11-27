# TestSetB 官方 430 覆盖恢复

仅恢复最终 frozen baseline 覆盖，不训练、不选模。保留原冻结 project 428 预测，只对官方多出的 `testsetb_104.bpseq`（S）与 `testsetb_36.bpseq`（N）做显式 CPU 补充。原始字符不替换；缺预测不补 0，部分覆盖不排名。大额补充与归一化预测位于 `/teams`。

CPU 补充前对 6 条既有记录逐系统回放，9 个 decoder 均逐对精确匹配冻结预测；两条补充均做 3 次确定性重复。MXfold2 使用 `--gpu -1`，所有进程 `CUDA_VISIBLE_DEVICES=''`。

RNAstructure Fold/MaxExpect 官方工具拒绝 `testsetb_104.bpseq` 的原始 S 字符，二者仅补到 429/430，保持 `PARTIAL_COVERAGE_NOT_RANKED`；该条缺预测不补 0。

## 新增全覆盖身份

| Full430 prediction identity | 来源 |
|---|---|
| `contrafold/testsetb` | frozen 428 + official-only 2 CPU completion |
| `eternafold/testsetb` | frozen 428 + official-only 2 CPU completion |
| `mxfold2/testsetb` | frozen 428 + official-only 2 CPU completion |
| `nussinov_turner/testsetb/recovered_official2` | frozen 428 + official-only 2 CPU completion |
| `vienna_centroid/testsetb/recovered_official2` | frozen 428 + official-only 2 CPU completion |
| `vienna_mea/testsetb/recovered_official2` | frozen 428 + official-only 2 CPU completion |
| `vienna_mfe/testsetb/recovered_official2` | frozen 428 + official-only 2 CPU completion |

## 官方 full 430

按 identity 字典序列出，不按指标排序。full 部分覆盖只展示 present-only 指标，状态保持 `PARTIAL_COVERAGE_NOT_RANKED`。

| Prediction identity | 状态 | n | micro F1 | macro F1 | 训练暴露 |
|---|---|---:|---:|---:|---|
| `best_plan_a/testsetb` | BLOCKED_REGISTERED_PREDICTION_MISSING | 0 | null | null | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `bpfold/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.7881461750 | 0.7949848207 | `UNCHECKED` |
| `contrafold/testsetb` | PASS_FULL_COVERAGE | 430 | 0.5633544581 | 0.5793087893 | `UNCHECKED` |
| `efold_0_1_3/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.5545354109 | 0.5705247034 | `UNCHECKED` |
| `eternafold/testsetb` | PASS_FULL_COVERAGE | 430 | 0.5816473147 | 0.5997317048 | `UNCHECKED` |
| `h0/testsetb` | BLOCKED_REGISTERED_PREDICTION_MISSING | 0 | null | null | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `kirigami_fast/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.8421929216 | 0.8343000298 | `UNCHECKED` |
| `mxfold2/testsetb` | PASS_FULL_COVERAGE | 430 | 0.6468595682 | 0.6692063276 | `AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED` |
| `nucleicbert_frozen_official/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.3652332314 | 0.3126424606 | `AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED` |
| `nucleicbert_ft/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.4842599509 | 0.5244210001 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `nussinov_turner/testsetb/recovered_official2` | PASS_FULL_COVERAGE | 430 | 0.2158239143 | 0.2240284299 | `NOT_SEQUENCE_FITTED` |
| `nussinov_turner/testsetb/repeat0` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `nussinov_turner/testsetb/repeat1` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `nussinov_turner/testsetb/repeat2` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `rinalmo_2d_s0/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.5382570989 | 0.5531410266 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rinalmo_2d_s1/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.7285987151 | 0.7434270615 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rinalmo_2d_s2/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.7178103315 | 0.7146001239 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rinalmo_ft/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.8670430034 | 0.8628944070 | `EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT` |
| `rnaformer_biophysical/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.5111127968 | 0.5314199088 | `UNCHECKED` |
| `rnaformer_bprna/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.7951901076 | 0.7711776661 | `UNCHECKED` |
| `rnaformer_interfamily/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.6685137188 | 0.6637434237 | `UNCHECKED` |
| `rnaformer_intrafamily/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.6650966452 | 0.6586049650 | `UNCHECKED` |
| `rnastructure_fold/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 429 | 0.5033218177 | 0.5225221948 | `NOT_SEQUENCE_FITTED` |
| `rnastructure_maxexpect/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 429 | 0.5273032873 | 0.5455791703 | `NOT_SEQUENCE_FITTED` |
| `ufold/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.4508291874 | 0.4781111939 | `UNCHECKED` |
| `vienna_centroid/testsetb/recovered_official2` | PASS_FULL_COVERAGE | 430 | 0.5452759933 | 0.5547336489 | `NOT_SEQUENCE_FITTED` |
| `vienna_centroid/testsetb/repeat0` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_centroid/testsetb/repeat1` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_centroid/testsetb/repeat2` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mea/testsetb/recovered_official2` | PASS_FULL_COVERAGE | 430 | 0.5365910503 | 0.5519406982 | `NOT_SEQUENCE_FITTED` |
| `vienna_mea/testsetb/repeat0` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mea/testsetb/repeat1` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mea/testsetb/repeat2` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mfe/testsetb/recovered_official2` | PASS_FULL_COVERAGE | 430 | 0.5202373304 | 0.5404191111 | `NOT_SEQUENCE_FITTED` |
| `vienna_mfe/testsetb/repeat0` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mfe/testsetb/repeat1` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mfe/testsetb/repeat2` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `xens2/testsetb` | PARTIAL_COVERAGE_NOT_RANKED | 428 | 0.8403202122 | 0.8322478063 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |

## 项目 project 428

按 identity 字典序列出，不按指标排序。full 部分覆盖只展示 present-only 指标，状态保持 `PARTIAL_COVERAGE_NOT_RANKED`。

| Prediction identity | 状态 | n | micro F1 | macro F1 | 训练暴露 |
|---|---|---:|---:|---:|---|
| `best_plan_a/testsetb` | BLOCKED_REGISTERED_PREDICTION_MISSING | 0 | null | null | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `bpfold/testsetb` | PASS_FULL_COVERAGE | 428 | 0.7881461750 | 0.7949848207 | `UNCHECKED` |
| `contrafold/testsetb` | PASS_FULL_COVERAGE | 428 | 0.5638971236 | 0.5798955998 | `UNCHECKED` |
| `efold_0_1_3/testsetb` | PASS_FULL_COVERAGE | 428 | 0.5545354109 | 0.5705247034 | `UNCHECKED` |
| `eternafold/testsetb` | PASS_FULL_COVERAGE | 428 | 0.5823802646 | 0.6004962870 | `UNCHECKED` |
| `h0/testsetb` | BLOCKED_REGISTERED_PREDICTION_MISSING | 0 | null | null | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `kirigami_fast/testsetb` | PASS_FULL_COVERAGE | 428 | 0.8421929216 | 0.8343000298 | `UNCHECKED` |
| `mxfold2/testsetb` | PASS_FULL_COVERAGE | 428 | 0.6473580571 | 0.6698447616 | `AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED` |
| `nucleicbert_frozen_official/testsetb` | PASS_FULL_COVERAGE | 428 | 0.3652332314 | 0.3126424606 | `AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED` |
| `nucleicbert_ft/testsetb` | PASS_FULL_COVERAGE | 428 | 0.4842599509 | 0.5244210001 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `nussinov_turner/testsetb/recovered_official2` | PASS_FULL_COVERAGE | 428 | 0.2154385685 | 0.2237670841 | `NOT_SEQUENCE_FITTED` |
| `nussinov_turner/testsetb/repeat0` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `nussinov_turner/testsetb/repeat1` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `nussinov_turner/testsetb/repeat2` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `rinalmo_2d_s0/testsetb` | PASS_FULL_COVERAGE | 428 | 0.5382570989 | 0.5531410266 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rinalmo_2d_s1/testsetb` | PASS_FULL_COVERAGE | 428 | 0.7285987151 | 0.7434270615 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rinalmo_2d_s2/testsetb` | PASS_FULL_COVERAGE | 428 | 0.7178103315 | 0.7146001239 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |
| `rinalmo_ft/testsetb` | PASS_FULL_COVERAGE | 428 | 0.8670430034 | 0.8628944070 | `EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT` |
| `rnaformer_biophysical/testsetb` | PASS_FULL_COVERAGE | 428 | 0.5111127968 | 0.5314199088 | `UNCHECKED` |
| `rnaformer_bprna/testsetb` | PASS_FULL_COVERAGE | 428 | 0.7951901076 | 0.7711776661 | `UNCHECKED` |
| `rnaformer_interfamily/testsetb` | PASS_FULL_COVERAGE | 428 | 0.6685137188 | 0.6637434237 | `UNCHECKED` |
| `rnaformer_intrafamily/testsetb` | PASS_FULL_COVERAGE | 428 | 0.6650966452 | 0.6586049650 | `UNCHECKED` |
| `rnastructure_fold/testsetb` | PASS_FULL_COVERAGE | 428 | 0.5030067748 | 0.5222784016 | `NOT_SEQUENCE_FITTED` |
| `rnastructure_maxexpect/testsetb` | PASS_FULL_COVERAGE | 428 | 0.5269634739 | 0.5453195038 | `NOT_SEQUENCE_FITTED` |
| `ufold/testsetb` | PASS_FULL_COVERAGE | 428 | 0.4508291874 | 0.4781111939 | `UNCHECKED` |
| `vienna_centroid/testsetb/recovered_official2` | PASS_FULL_COVERAGE | 428 | 0.5456713963 | 0.5551500445 | `NOT_SEQUENCE_FITTED` |
| `vienna_centroid/testsetb/repeat0` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_centroid/testsetb/repeat1` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_centroid/testsetb/repeat2` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mea/testsetb/recovered_official2` | PASS_FULL_COVERAGE | 428 | 0.5369090838 | 0.5523298757 | `NOT_SEQUENCE_FITTED` |
| `vienna_mea/testsetb/repeat0` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mea/testsetb/repeat1` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mea/testsetb/repeat2` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mfe/testsetb/recovered_official2` | PASS_FULL_COVERAGE | 428 | 0.5202597501 | 0.5405667540 | `NOT_SEQUENCE_FITTED` |
| `vienna_mfe/testsetb/repeat0` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mfe/testsetb/repeat1` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `vienna_mfe/testsetb/repeat2` | BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED | 0 | null | null | `NOT_SEQUENCE_FITTED` |
| `xens2/testsetb` | PASS_FULL_COVERAGE | 428 | 0.8403202122 | 0.8322478063 | `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN` |

## 逐 ID 覆盖核查

| Official-only ID | 非 ACGU | 原冻结覆盖 | 最终全覆盖身份数 |
|---|---|---:|---:|
| `testsetb_104.bpseq` | S | 0 | 7 |
| `testsetb_36.bpseq` | N | 0 | 7 |

未恢复的神经模型没有 full430 原始资产，保持 428/430 或原阻塞状态；不使用 GPU，不根据外部测试指标选模。训练暴露列逐行保留，不能用标签独立替代训练独立。
