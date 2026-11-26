# bpRNA-new official-reference 5388 协议修复面板

## 范围

本面板使用经认证 BPfold v0.2 `BPfold_data/test_data/bpRNAnew` 官方参考标签，按冻结项目 5,388 个 ID 的原顺序取交集。只进行 CPU 冻结预测复算，不做新推理、训练、阈值调整或模型选择。旧派生标签及历史数值保留，但其结果仅为 circular-audit-only，不再作为独立 SOTA 证据。0.9617 项目标签误配值不进入本面板或排名。

标签面板 SHA256：`4924e3512e89fd965233a1a14b327c3d87e6c583764b7682c2d293b1454159e4`；ID 顺序 SHA256：`ef573022b3344bcb5092c97fd40478a7ee7060958f18efdb081e1d4acf0cba6e`。每条记录含 ID、sequence、pairs、sequence/pair-set/record SHA256。

## 指标与覆盖

严格 zero-based inclusive 精确配对；不追加 crossing、canonical 或 min-loop 过滤。micro 汇总 TP/FP/FN；macro 为逐序列 P/R/F1 的非加权平均。真值与预测均空时单序列 P=R=F1=1。排名仅使用 5,388 全覆盖身份，按 micro-F1 描述冻结结果，不构成选模。

清单共 43 个身份：20 recovered + 23 additional registered；28 个全覆盖，1 个 5386 部分覆盖，14 个 blocked/summary-only。

| Prediction identity | Model form | n | micro P | micro R | micro F1 | macro P | macro R | macro F1 | Δmicro F1 vs xens2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bpfold/bprna_new` | six_checkpoint_ensemble | 5388 | 0.7058527846 | 0.8579821401 | 0.7745179151 | 0.7197046556 | 0.8805564148 | 0.7827906824 | +0.2855013061 |
| `bpfold_official_release/bprna_new_official_reference` | six_checkpoint_ensemble | 5388 | 0.5942107427 | 0.6793905687 | 0.6339521965 | 0.6035264394 | 0.7235105242 | 0.6472307861 | +0.1449355874 |
| `contrafold/bprna_new` | deterministic_single_predictor | 5388 | 0.5598105664 | 0.7194657684 | 0.6296755882 | 0.5849493111 | 0.7348931655 | 0.6424043796 | +0.1406589792 |
| `vienna_centroid/bprna_new/recovered_v4` | deterministic_single_predictor | 5388 | 0.5739461267 | 0.6936628545 | 0.6281512903 | 0.5928516030 | 0.7000228890 | 0.6303173655 | +0.1391346813 |
| `mxfold2/bprna_new` | deterministic_single_predictor | 5388 | 0.5511892388 | 0.7103242989 | 0.6207196323 | 0.5799138846 | 0.7182616666 | 0.6330391053 | +0.1317030232 |
| `eternafold/bprna_new` | deterministic_single_predictor | 5388 | 0.5426335115 | 0.7211890960 | 0.6192979447 | 0.5686499805 | 0.7365984730 | 0.6331692943 | +0.1302813357 |
| `vienna_mea/bprna_new/recovered_v4` | deterministic_single_predictor | 5388 | 0.5416086703 | 0.7132382892 | 0.6156863010 | 0.5678304290 | 0.7181511430 | 0.6255327479 | +0.1266696920 |
| `rnastructure_maxexpect/bprna_new` | deterministic_single_predictor | 5388 | 0.5414572034 | 0.7061804794 | 0.6129447880 | 0.5664048249 | 0.7106308751 | 0.6215347252 | +0.1239281790 |
| `vienna_mfe/bprna_new/recovered_v4` | deterministic_single_predictor | 5388 | 0.5200806645 | 0.7151496162 | 0.6022123865 | 0.5520024308 | 0.7204280122 | 0.6167327988 | +0.1131957774 |
| `rnaformer_biophysical/bprna_new` | single_model | 5388 | 0.5185153369 | 0.7092198026 | 0.5990564789 | 0.5503170248 | 0.7191051341 | 0.6150748687 | +0.1100398699 |
| `rnastructure_fold/bprna_new` | deterministic_single_predictor | 5388 | 0.5116526337 | 0.6973601755 | 0.5902438216 | 0.5426002269 | 0.7029741703 | 0.6042415414 | +0.1012272126 |
| `rnaformer_intrafamily/bprna_new` | single_model | 5388 | 0.6037770051 | 0.5740090866 | 0.5885168617 | 0.5919362599 | 0.5930714239 | 0.5791259575 | +0.0995002527 |
| `rnaformer_interfamily/bprna_new` | single_model | 5388 | 0.6070104398 | 0.5706877644 | 0.5882889685 | 0.5916752169 | 0.5904215027 | 0.5775316013 | +0.0992723594 |
| `efold_0_1_3/bprna_new` | single_model | 5388 | 0.4846566364 | 0.6626272912 | 0.5598382517 | 0.5022811274 | 0.6778536178 | 0.5681537055 | +0.0708216427 |
| `rinalmo_2d_s0/bprna_new` | single_model | 5388 | 0.5250339837 | 0.5052639825 | 0.5149593035 | 0.5485867532 | 0.5282194753 | 0.5274788803 | +0.0259426945 |
| `spot_rna_postprocessed/bprna_new` | single_model | 5388 | 0.4969186208 | 0.5052953157 | 0.5010719612 | 0.4960927689 | 0.5279989558 | 0.5016089946 | +0.0120553522 |
| `kirigami/bprna_new` | single_model | 5388 | 0.6015234096 | 0.4218862604 | 0.4959391517 | 0.5883240507 | 0.4468502886 | 0.4873753421 | +0.0069225427 |
| `kirigami_fast/bprna_new` | single_model | 5388 | 0.6015234096 | 0.4218862604 | 0.4959391517 | 0.5883240507 | 0.4468502886 | 0.4873753421 | +0.0069225427 |
| `xens2/bprna_new` | three_checkpoint_ensemble | 5388 | 0.5624306128 | 0.4325552248 | 0.4890166090 | 0.5738515065 | 0.4598143307 | 0.4972882639 | +0.0000000000 |
| `structrfm/bprna_new` | single_model | 5388 | 0.4718200469 | 0.5044179853 | 0.4875747710 | 0.4836062896 | 0.5079596442 | 0.4850647195 | -0.0014418381 |
| `structrfm_official_v0.0.8/bprna_new` | single_model | 5388 | 0.4718200469 | 0.5044179853 | 0.4875747710 | 0.4836062896 | 0.5079596442 | 0.4850647195 | -0.0014418381 |
| `ufold/bprna_new` | single_model | 5388 | 0.4095248460 | 0.5305890647 | 0.4622618654 | 0.4465127211 | 0.5603551244 | 0.4893648205 | -0.0267547437 |
| `rnaformer_bprna/bprna_new` | single_model | 5388 | 0.6113158940 | 0.3494595018 | 0.4447036190 | 0.5519794493 | 0.3712638219 | 0.4236984771 | -0.0443129901 |
| `rinalmo_ft/bprna_new` | single_model | 5388 | 0.4512102995 | 0.3321870594 | 0.3826569514 | 0.4634412771 | 0.3605041701 | 0.3918555985 | -0.1063596576 |
| `nucleicbert_ft/bprna_new` | single_model | 5388 | 0.2226455777 | 0.3785445715 | 0.2803814232 | 0.2521303997 | 0.4003919124 | 0.3029833378 | -0.2086351859 |
| `rnafm_project_frozen_encoder_crf/bprna_new` | single_model | 5388 | 0.2901423162 | 0.2611076296 | 0.2748603352 | 0.3325653607 | 0.2927742704 | 0.3057367156 | -0.2141562738 |
| `nucleicbert_frozen_official/bprna_new` | single_model | 5388 | 0.4101010101 | 0.1017703274 | 0.1630726748 | 0.3676579028 | 0.1194607182 | 0.1675804131 | -0.3259439342 |
| `spot_rna/bprna_new` | single_model | 5388 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | -0.4890166090 |

## 最强单预测器与集成

- **最强集成/总体**：`bpfold/bprna_new`，micro-F1=0.7745179151021447，macro-F1=0.7827906824085470；相对 xens2 Δmicro=+0.2855013060718978、Δmacro=+0.2855024184959545，配对 95% CI [+0.2784513110377516, +0.2925532575213283]。
- **最强单预测器**：`contrafold/bprna_new`，micro-F1=0.6296755882191630，macro-F1=0.6424043795760952；相对 xens2 Δmicro=+0.1406589791889161、Δmacro=+0.1451161156635027，配对 95% CI [+0.1379042374614799, +0.1524978902029485]。

配对区间为 10,000 次序列级 percentile bootstrap，固定 seed=20261127；仅适用于共同 5,388 全覆盖记录。每个系统相对 xens2 的 macro-F1 配对 CI 均在 `system_metrics.json`。

## 部分覆盖与不可评分

- `contrafold/bprna_new/clean_tr1c`：5386/5388，缺失 `['FP929046.1_562024-562125.bpseq', 'URS0000D67ADA_12908_1-55.bpseq']`；指标仅对现有记录复算，不进入排名。
- Blocked/summary-only（不补值、不排名）：`best_plan_a/bprna_new`、`h0/bprna_new`、`vienna_mfe/bprna_new/repeat0`、`vienna_centroid/bprna_new/repeat0`、`vienna_mea/bprna_new/repeat0`、`nussinov_turner/bprna_new/repeat0`、`vienna_mfe/bprna_new/repeat1`、`vienna_centroid/bprna_new/repeat1`、`vienna_mea/bprna_new/repeat1`、`nussinov_turner/bprna_new/repeat1`、`vienna_mfe/bprna_new/repeat2`、`vienna_centroid/bprna_new/repeat2`、`vienna_mea/bprna_new/repeat2`、`nussinov_turner/bprna_new/repeat2`。

## Registry 修复

- 项目旧标签标为 `DERIVED_FROM_BPFOLD_PREDICTIONS`，谱系指向 v0.2 release 预测及 min-loop/crossing 投影，旧 raw/clean 派生结果均为 circular-audit-only。
- BPfold 官方 release 与项目六 checkpoint replay 保持不同 prediction identity；官方 5,401 指标保留，新面板只添加同一官方标签的 5,388 交集结果。
- `data_registry.json`、`prediction_registry.json`、`baseline_registry.json` 和 recovered registry 只增加/修正谱系与比较边界，历史 metric 字段通过逐字段不变断言。

机器可读完整结果：`source_data/bprna_official_reference_5388/system_metrics.json`；独立第二实现：`tools/verify_bprna_official_reference_panel_20261127.py`。
