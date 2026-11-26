# ArchiveII 未过滤 official-reference 3966 / clean 2544 重算面板

本面板仅对冻结预测做 CPU 复算，不做训练、新推理、阈值调整或模型选择。旧 `DERIVED_FROM_BPFOLD_PREDICTIONS` 标签及旧分数只保留 circular-audit-only。共享 registry 未修改；`registry_recommendation.json` 供后续统一整合。

BPfold_data 与 MXfold2 的 3966 条官方参考 BPSEQ 已逐条验证 ID、sequence、pairs 和 member SHA 一致。严格 zero-based inclusive 配对计分，不追加 min-loop/crossing/canonical 过滤；truth-empty 记录全部计入 macro，仅 empty/empty 的单序列 P/R/F1=1，truth-empty 而预测非空仍按普通规则计分。官方全集 5 条、clean 交集 4 条 truth-empty；release 对这些记录均产生非空预测，因此实际 empty/empty=0。缺失预测不补 0。

## clean 2544 全覆盖结果（描述性，不用于选模）

| Prediction identity | Model form | n | micro P | micro R | micro F1 | macro P | macro R | macro F1 | Δmicro vs xens2 | Δmacro vs xens2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bpfold/archiveii_embok_clean` | six_checkpoint_ensemble | 2544 | 0.8979856958 | 0.9018890402 | 0.8999331355 | 0.9092643302 | 0.9264180927 | 0.9161611814 | +0.0842809284 | +0.0565024689 |
| `rinalmo_ft/archiveii_embok_clean` | single_model | 2544 | 0.8389606925 | 0.8027427956 | 0.8204522395 | 0.8692056329 | 0.8624298566 | 0.8630290509 | +0.0048000324 | +0.0033703384 |
| `xens2/archiveii_embok_clean` | three_checkpoint_ensemble | 2544 | 0.8814553121 | 0.7589913774 | 0.8156522070 | 0.8981631687 | 0.8345239006 | 0.8596587125 | +0.0000000000 | +0.0000000000 |
| `rnaformer_bprna/archiveii_embok_clean` | single_model | 2544 | 0.8957236530 | 0.7371440322 | 0.8087334342 | 0.8986936788 | 0.8189488396 | 0.8489915231 | -0.0069187728 | -0.0106671894 |
| `bpfold_official_release/archiveii` | six_checkpoint_ensemble | 2544 | 0.8245520942 | 0.7933472317 | 0.8086487346 | 0.8251730514 | 0.8437786893 | 0.8300397546 | -0.0070034724 | -0.0296189578 |
| `kirigami_fast/archiveii_embok_clean` | single_model | 2544 | 0.8357803684 | 0.7289964829 | 0.7787448396 | 0.8535512667 | 0.8049238885 | 0.8229724637 | -0.0369073674 | -0.0366862488 |
| `rinalmo_2d_s1/archiveii_embok_clean` | single_model | 2544 | 0.7583102591 | 0.7573746313 | 0.7578421564 | 0.8467911148 | 0.8384138298 | 0.8397663192 | -0.0578100506 | -0.0198923932 |
| `rnaformer_intrafamily/archiveii_embok_clean` | single_model | 2544 | 0.7734011114 | 0.6829121284 | 0.7253453236 | 0.7999981067 | 0.7970835583 | 0.7909518516 | -0.0903068834 | -0.0687068609 |
| `rnaformer_interfamily/archiveii_embok_clean` | single_model | 2544 | 0.7750448935 | 0.6763671432 | 0.7223515958 | 0.7985645814 | 0.7918907252 | 0.7867907946 | -0.0933006112 | -0.0728679179 |
| `rinalmo_2d_s2/archiveii_embok_clean` | single_model | 2544 | 0.8435754190 | 0.6156753460 | 0.7118291118 | 0.8667117010 | 0.7492814353 | 0.7926688575 | -0.1038230952 | -0.0669898550 |
| `mxfold2/archiveii_embok_clean` | deterministic_single_predictor | 2544 | 0.6890026121 | 0.7332085319 | 0.7104185561 | 0.7378670263 | 0.7719408334 | 0.7512771918 | -0.1052336510 | -0.1083815206 |
| `contrafold/archiveii_embok_clean` | deterministic_single_predictor | 2544 | 0.5681173193 | 0.6189088382 | 0.5924264227 | 0.6071409675 | 0.6616497845 | 0.6306631338 | -0.2232257844 | -0.2289955786 |
| `eternafold/archiveii_embok_clean` | deterministic_single_predictor | 2544 | 0.5417224059 | 0.6087616292 | 0.5732888147 | 0.5748638551 | 0.6370069140 | 0.6020168480 | -0.2423633924 | -0.2576418644 |
| `rnastructure_maxexpect/archiveii_embok_clean` | deterministic_single_predictor | 2544 | 0.5310051533 | 0.5728457568 | 0.5511324874 | 0.5606868512 | 0.5914913259 | 0.5731918597 | -0.2645197197 | -0.2864668527 |
| `efold_0_1_3/archiveii_embok_clean` | single_model | 2544 | 0.5241607243 | 0.5780434536 | 0.5497850242 | 0.5782932448 | 0.6410191844 | 0.6053761720 | -0.2658671828 | -0.2542825405 |
| `rnaformer_biophysical/archiveii_embok_clean` | single_model | 2544 | 0.5199000639 | 0.5710517359 | 0.5442767206 | 0.5534418284 | 0.6052712190 | 0.5754162898 | -0.2713754864 | -0.2842424227 |
| `rnastructure_fold/archiveii_embok_clean` | deterministic_single_predictor | 2544 | 0.5035592309 | 0.5708390061 | 0.5350925587 | 0.5459844933 | 0.5977234604 | 0.5684986641 | -0.2805596483 | -0.2911600484 |
| `nucleicbert_frozen_official/archiveii_embok_clean` | single_model | 2544 | 0.8121082356 | 0.3711566826 | 0.5094706924 | 0.7389580338 | 0.4279873745 | 0.5148706103 | -0.3061815146 | -0.3447881022 |
| `ufold/archiveii_embok_clean` | single_model | 2544 | 0.4997878232 | 0.4927388246 | 0.4962382926 | 0.6183172336 | 0.6194280730 | 0.6157846094 | -0.3194139144 | -0.2438741031 |
| `nucleicbert_ft/archiveii_embok_clean` | single_model | 2544 | 0.3917047720 | 0.5169261970 | 0.4456868961 | 0.5840391631 | 0.6742451676 | 0.6230066998 | -0.3699653110 | -0.2366520126 |

## 最强单预测器与集成

- **最强集成/总体**：`bpfold/archiveii_embok_clean`，micro-F1=0.8999331354520079，macro-F1=0.9161611813979388；相对 xens2 Δmicro=+0.0842809284069906、Δmacro=+0.0565024689359115，配对 macro Δ95% CI [+0.0511175356573565, +0.0619467954524887]。
- **最强单预测器**：`rinalmo_ft/archiveii_embok_clean`，micro-F1=0.8204522394549935，macro-F1=0.8630290508840547；相对 xens2 Δmicro=+0.0048000324099762、Δmacro=+0.0033703384220274，配对 macro Δ95% CI [-0.0012121956293115, +0.0078699689970171]。

BPfold release 与项目 replay 是不同 prediction identity；release 可覆盖全集 3966，replay 只覆盖 clean 2544。配对区间采用固定 seed=20261127 的 10,000 次序列级 percentile bootstrap，仅适用于 clean 2544 共同全覆盖。

## 官方全集 3966 覆盖与 present-only 指标

只有全覆盖身份可参与全集最强判断。部分覆盖指标只对实际存在的预测计分，不把 1422 条缺失记录补 0，不计算全集相对 xens2 差距，也不进入排名。

| Prediction identity | n / 3966 | coverage | status | present-only micro F1 | present-only macro F1 |
|---|---:|---:|---|---:|---:|
| `best_plan_a/archiveii_embok_clean` | 0 / 3966 | 0.0000000000 | `BLOCKED_REGISTERED_PREDICTION_MISSING` | UNCHECKED | UNCHECKED |
| `bpfold/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.8999331355 | 0.9161611814 |
| `contrafold/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.5924264227 | 0.6306631338 |
| `efold_0_1_3/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.5497850242 | 0.6053761720 |
| `eternafold/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.5732888147 | 0.6020168480 |
| `h0/archiveii_embok_clean` | 0 / 3966 | 0.0000000000 | `BLOCKED_REGISTERED_PREDICTION_MISSING` | UNCHECKED | UNCHECKED |
| `kirigami_fast/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.7787448396 | 0.8229724637 |
| `mxfold2/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.7104185561 | 0.7512771918 |
| `nucleicbert_frozen_official/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.5094706924 | 0.5148706103 |
| `nucleicbert_ft/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.4456868961 | 0.6230066998 |
| `nussinov_turner/archiveii_embok_clean/repeat0` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `nussinov_turner/archiveii_embok_clean/repeat1` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `nussinov_turner/archiveii_embok_clean/repeat2` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `rinalmo_2d_s0/archiveii_embok_clean` | 0 / 3966 | 0.0000000000 | `BLOCKED_REGISTERED_PREDICTION_MISSING` | UNCHECKED | UNCHECKED |
| `rinalmo_ft/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.8204522395 | 0.8630290509 |
| `rnaformer_biophysical/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.5442767206 | 0.5754162898 |
| `rnaformer_bprna/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.8087334342 | 0.8489915231 |
| `rnaformer_interfamily/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.7223515958 | 0.7867907946 |
| `rnaformer_intrafamily/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.7253453236 | 0.7909518516 |
| `rnastructure_fold/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.5350925587 | 0.5684986641 |
| `rnastructure_maxexpect/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.5511324874 | 0.5731918597 |
| `ufold/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.4962382926 | 0.6157846094 |
| `vienna_centroid/archiveii_embok_clean/repeat0` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `vienna_centroid/archiveii_embok_clean/repeat1` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `vienna_centroid/archiveii_embok_clean/repeat2` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `vienna_mea/archiveii_embok_clean/repeat0` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `vienna_mea/archiveii_embok_clean/repeat1` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `vienna_mea/archiveii_embok_clean/repeat2` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `vienna_mfe/archiveii_embok_clean/repeat0` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `vienna_mfe/archiveii_embok_clean/repeat1` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `vienna_mfe/archiveii_embok_clean/repeat2` | 0 / 3966 | 0.0000000000 | `BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED` | UNCHECKED | UNCHECKED |
| `xens2/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.8156522070 | 0.8596587125 |
| `bpfold_official_release/archiveii` | 3966 / 3966 | 1.0000000000 | `PASS_FULL_COVERAGE` | 0.7974584108 | 0.8200813623 |
| `rinalmo_2d_s1/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.7578421564 | 0.8397663192 |
| `rinalmo_2d_s2/archiveii_embok_clean` | 2544 / 3966 | 0.6414523449 | `PARTIAL_COVERAGE_NOT_RANKED` | 0.7118291118 | 0.7926688575 |
| `rinalmo_2d_s0/archiveii_embok_clean_trunc598` | 0 / 3966 | 0.0000000000 | `BLOCKED_PROTOCOL_MISMATCH_FIRST598_NOT_FULL_LENGTH` | UNCHECKED | UNCHECKED |

全集唯一全覆盖身份为 `bpfold_official_release/archiveii`（ensemble），micro-F1=0.7974584107660678，macro-F1=0.8200813622940291，empty/empty=0。全集没有其他全覆盖单预测器，因此最强单预测器为 `UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR`，不以部分覆盖排名替代。

## 阻塞与边界

R2D seed 0 全长 raw 仍缺失；first-598 资产标为 `BLOCKED_PROTOCOL_MISMATCH_FIRST598_NOT_FULL_LENGTH`，未冒充全长预测。seed 1/2 是独立 seed 的全长冻结资产，不用于替代 seed 0。

其余 summary-only 或缺失身份均保持 blocked、metrics=null、不补 0。完整清单见 `system_metrics.json` 和 `registry_recommendation.json`。
