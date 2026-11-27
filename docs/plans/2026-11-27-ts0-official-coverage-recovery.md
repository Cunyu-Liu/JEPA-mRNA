# TS0 官方 1305 覆盖恢复

仅恢复最终 frozen baseline 覆盖，不训练、不选模。BPfold 使用作者官方 release 的 1305 条 BPSEQ；Vienna/Nussinov、CONTRAfold、EternaFold 和 RNAstructure 保留原冻结 1288 条，只对官方多出的 17 条做 CPU 补充。其他神经模型没有 full1305 原始资产，保持部分覆盖或阻塞，不使用 GPU。

所有输出按官方未过滤 BPSEQ 面板统一 ID、序列和序列 SHA256。部分覆盖不排名、不补 0；仅 truth 与 prediction 同时为空时 P/R/F1=1。唯一 truth-empty 为 `bpRNA_SRP_192.bpseq`。RNAstructure 官方工具拒绝 3 条含 R/Y 的序列，Fold/MaxExpect 因此保持 1302/1305 部分覆盖，不改写碱基。

| Full1305 prediction identity | 来源 | 记录数 |
|---|---|---:|
| `bpfold_official_release/bprna_ts0` | AUTHOR_BPFOLD_TEST_RESULTS_V0_2_FULL1305 | 1305 |
| `vienna_mfe/bprna_ts0/recovered_official17` | FROZEN_RAW1288_PLUS_OFFICIAL17_CPU_COMPLETION | 1305 |
| `vienna_centroid/bprna_ts0/recovered_official17` | FROZEN_RAW1288_PLUS_OFFICIAL17_CPU_COMPLETION | 1305 |
| `vienna_mea/bprna_ts0/recovered_official17` | FROZEN_RAW1288_PLUS_OFFICIAL17_CPU_COMPLETION | 1305 |
| `nussinov_turner/bprna_ts0/recovered_official17` | FROZEN_RAW1288_PLUS_OFFICIAL17_CPU_COMPLETION | 1305 |
| `contrafold/bprna_ts0` | FROZEN_RAW1288_PLUS_OFFICIAL17_CPU_COMPLETION | 1305 |
| `eternafold/bprna_ts0` | FROZEN_RAW1288_PLUS_OFFICIAL17_CPU_COMPLETION | 1305 |
| `rnastructure_fold/bprna_ts0` | FROZEN_RAW1288_PLUS_OFFICIAL17_CPU_COMPLETION | 1302 |
| `rnastructure_maxexpect/bprna_ts0` | FROZEN_RAW1288_PLUS_OFFICIAL17_CPU_COMPLETION | 1302 |

## 逐 ID 资产核查

每条均先核查既有冻结预测与 BPfold 官方 release；`existing_frozen_coverage` 为空表示既有原始预测没有该 ID，不以空结构代替缺失。

| Official-only ID | 非 ACGU | truth-empty | 既有冻结覆盖 | BPfold release | 最终全覆盖身份数 |
|---|---|---:|---|---|---:|
| `bpRNA_CRW_25863.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_CRW_25863.bpseq | 9 |
| `bpRNA_CRW_31704.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_CRW_31704.bpseq | 9 |
| `bpRNA_CRW_43565.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_CRW_43565.bpseq | 9 |
| `bpRNA_CRW_54323.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_CRW_54323.bpseq | 9 |
| `bpRNA_CRW_54562.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_CRW_54562.bpseq | 9 |
| `bpRNA_RFAM_10644.bpseq` | Y | false | 0 | BPfold_test_results/bpRNA/bpRNA_RFAM_10644.bpseq | 7 |
| `bpRNA_RFAM_1072.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_RFAM_1072.bpseq | 9 |
| `bpRNA_RFAM_18822.bpseq` | RY | false | 0 | BPfold_test_results/bpRNA/bpRNA_RFAM_18822.bpseq | 7 |
| `bpRNA_RFAM_18935.bpseq` | RY | false | 0 | BPfold_test_results/bpRNA/bpRNA_RFAM_18935.bpseq | 7 |
| `bpRNA_RFAM_38651.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_RFAM_38651.bpseq | 9 |
| `bpRNA_SRP_192.bpseq` | N | true | 0 | BPfold_test_results/bpRNA/bpRNA_SRP_192.bpseq | 9 |
| `bpRNA_tmRNA_252.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_tmRNA_252.bpseq | 9 |
| `bpRNA_tmRNA_397.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_tmRNA_397.bpseq | 9 |
| `bpRNA_tmRNA_643.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_tmRNA_643.bpseq | 9 |
| `bpRNA_tmRNA_651.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_tmRNA_651.bpseq | 9 |
| `bpRNA_tmRNA_686.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_tmRNA_686.bpseq | 9 |
| `bpRNA_tmRNA_693.bpseq` | N | false | 0 | BPfold_test_results/bpRNA/bpRNA_tmRNA_693.bpseq | 9 |

CPU 补充前的有界回放索引为 `0, 17, 100, 501, 1000, 1287`，8 个 decoder 均逐对精确匹配原冻结预测；17 条补充均做 3 次确定性重复。面板重算本身不推理。
