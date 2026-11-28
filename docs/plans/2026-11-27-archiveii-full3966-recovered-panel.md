# ArchiveII full3966 恢复双面板

新预测仅来自官方 clean 2544 之外的 1422 条，原始 sequence/SHA 未改写；新 identity 为 clean-frozen + recovery composite。缺预测不补 0，部分覆盖不排名。xens2 已恢复 1406 条、16 条协议阻断，composite 为 3950/3966，故 full3966 的 xens2 差距仍保持 `UNCHECKED_XENS2_NOT_FULL_COVERAGE`。

| Baseline | checkpoint 完成 | PASS | coverage | full3966 可排名 |
|---|---:|---:|---:|---|
| contrafold | 1422 / 1422 | 1422 | 1.0000000000 | True |
| eternafold | 1422 / 1422 | 1422 | 1.0000000000 | True |
| mxfold2 | 1422 / 1422 | 1422 | 1.0000000000 | True |
| vienna_mfe | 1422 / 1422 | 1422 | 1.0000000000 | True |
| vienna_centroid | 1422 / 1422 | 1422 | 1.0000000000 | True |
| vienna_mea | 1422 / 1422 | 1422 | 1.0000000000 | True |
| rnastructure_fold | 1422 / 1422 | 1422 | 1.0000000000 | True |
| rnastructure_maxexpect | 1422 / 1422 | 1422 | 1.0000000000 | True |
| xens2 | 1422 / 1422 | 1406 | 1.0000000000 | False |

## full3966 全覆盖指标

| Prediction identity | Model form | n | micro F1 | macro F1 |
|---|---|---:|---:|---:|
| `bpfold_official_release/archiveii` | six_checkpoint_ensemble | 3966 | 0.7974584108 | 0.8200813623 |
| `mxfold2/archiveii_official_reference_3966/cpu_recovery_v1` | deterministic_single_predictor | 3966 | 0.7023050286 | 0.7457279553 |
| `contrafold/archiveii_official_reference_3966/cpu_recovery_v1` | deterministic_single_predictor | 3966 | 0.5796193275 | 0.6186547352 |
| `eternafold/archiveii_official_reference_3966/cpu_recovery_v1` | deterministic_single_predictor | 3966 | 0.5691609744 | 0.5994220587 |
| `vienna_centroid/archiveii_official_reference_3966/cpu_recovery_v1` | deterministic_single_predictor | 3966 | 0.5691313692 | 0.5943999687 |
| `vienna_mea/archiveii_official_reference_3966/cpu_recovery_v1` | deterministic_single_predictor | 3966 | 0.5597636145 | 0.5919867904 |
| `rnastructure_maxexpect/archiveii_official_reference_3966/cpu_recovery_v1` | deterministic_single_predictor | 3966 | 0.5500991323 | 0.5765659118 |
| `vienna_mfe/archiveii_official_reference_3966/cpu_recovery_v1` | deterministic_single_predictor | 3966 | 0.5388319123 | 0.5771613799 |
| `rnastructure_fold/archiveii_official_reference_3966/cpu_recovery_v1` | deterministic_single_predictor | 3966 | 0.5346280306 | 0.5725529991 |

## 新增 composite 在 clean 2544 的 xens2 差距

仅列 clean 全覆盖的新身份；Vienna 没有旧 clean 记录级预测，故不在此表补 0 或排名。

| Prediction identity | n | micro F1 | macro F1 | Δmicro vs xens2 | Δmacro vs xens2 |
|---|---:|---:|---:|---:|---:|
| `contrafold/archiveii_official_reference_3966/cpu_recovery_v1` | 2544 | 0.5924264227 | 0.6306631338 | -0.2232257844 | -0.2289955786 |
| `eternafold/archiveii_official_reference_3966/cpu_recovery_v1` | 2544 | 0.5732888147 | 0.6020168480 | -0.2423633924 | -0.2576418644 |
| `mxfold2/archiveii_official_reference_3966/cpu_recovery_v1` | 2544 | 0.7104185561 | 0.7512771918 | -0.1052336510 | -0.1083815206 |
| `vienna_mfe/archiveii_official_reference_3966/cpu_recovery_v1` | 2544 | 0.5411742135 | 0.5739987539 | -0.2744779936 | -0.2856599585 |
| `vienna_centroid/archiveii_official_reference_3966/cpu_recovery_v1` | 2544 | 0.5751153985 | 0.5944413500 | -0.2405368085 | -0.2652173625 |
| `vienna_mea/archiveii_official_reference_3966/cpu_recovery_v1` | 2544 | 0.5660990282 | 0.5921087656 | -0.2495531789 | -0.2675499469 |
| `rnastructure_fold/archiveii_official_reference_3966/cpu_recovery_v1` | 2544 | 0.5350925587 | 0.5684986641 | -0.2805596483 | -0.2911600484 |
| `rnastructure_maxexpect/archiveii_official_reference_3966/cpu_recovery_v1` | 2544 | 0.5511324874 | 0.5731918597 | -0.2645197197 | -0.2864668527 |
| `xens2/archiveii_official_reference_3966/gpu_recovery_v1` | 2544 | 0.8156522070 | 0.8596587125 | +0.0000000000 | +0.0000000000 |

## 最强结果

- clean_2544 strongest_ensemble: `bpfold/archiveii_embok_clean`，micro-F1=0.8999331354520079，macro-F1=0.9161611813979388。
- clean_2544 strongest_single_predictor: `rinalmo_ft/archiveii_embok_clean`，micro-F1=0.8204522394549935，macro-F1=0.8630290508840547。
- full_3966 strongest_ensemble: `bpfold_official_release/archiveii`，micro-F1=0.7974584107660678，macro-F1=0.8200813622940291。
- full_3966 strongest_single_predictor: `mxfold2/archiveii_official_reference_3966/cpu_recovery_v1`，micro-F1=0.7023050285738565，macro-F1=0.7457279552838411。

xens2 live queue=`COMPLETE_WITH_PROTOCOL_BLOCKS`，处理 1422/1422，pass=1406，protocol-blocked=16，预测 SHA256=`dd776f0be7f8470f37e57e939179088b14acf4f7369cb67f837c9534ffa5adf1`。恢复完成不等于 full coverage；3950/3966 仍不排名。

完整逐 ID coverage、指标、delta/CI 和 registry 建议见 `system_metrics.json`、`recovered_per_record_metrics.jsonl.gz`、`official_coverage_gaps.json` 与 `registry_recommendation.json`。
