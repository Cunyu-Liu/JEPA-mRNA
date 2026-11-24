# FoldNar baseline recovery v2 — BPfold, UFold, ContextFold

Date: 2026-11-24
Scope: frozen `bpRNA-new` evidence only. All new computation was CPU-only parsing and scoring; no model inference, training, calibration, model selection, shared-registry edits, or running-artifact edits were performed.

## Verified raw metrics

Strict exact-pair scoring uses zero-based inclusive pairs. No crossing-pair or short-span filtering is added by the scorer. Micro precision/recall/F1 are computed from pooled TP/FP/FN; macro-F1 is the unweighted per-sequence mean, with F1=1 only when both truth and prediction have zero pairs.

| Prediction identity | Label scope | n / coverage | TP | FP | FN | Micro P | Micro R | Micro F1 | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen project BPfold replay | Project `bprna_new.jsonl` | 5388 / 100% | 106998 | 48176 | 28186 | 0.6895356180803485 | 0.7914989939637826 | **0.7370074184282851** | 0.7447389175263477 |
| Frozen UFold MultiMolecule conversion | Project `bprna_new.jsonl` | 5388 / 100% | 80572 | 84827 | 54612 | 0.48713716527911294 | 0.5960172801514972 | **0.5361048362681855** | 0.5589247295596474 |
| Official BPfold v0.2 release | Project labels (label-sensitive audit only) | 5388 / 100% | 135184 | 10776 | 0 | 0.92617155385037 | 1.0 | **0.9616708875167174** | 0.9584497822657716 |
| Official BPfold v0.2 release | Official `BPfold_data/test_data/bpRNAnew` | 5401 / 100% | 86933 | 59441 | 41092 | 0.5939101206498422 | 0.6790314391720367 | **0.633624758107719** | **0.6470223791166072** |
| Official BPfold v0.2 release | Official labels, project-name intersection | 5388 / 100% | 86731 | 59229 | 40929 | 0.5942107426692245 | 0.6793905686981043 | **0.6339521964768657** | **0.6472307861307366** |
| ContextFold | No verified predictions | 0 / not measurable | — | — | — | — | — | — | — |

The all-5401 official-label macro-F1 rounds to **0.647**, independently reproducing the rounded value in BPfold v0.2's official README. This validates the prediction/reference parsing and the distinction between official and project label scopes. The 0.9616708875 project-label result must **not** be presented as the official BPfold result or as a SOTA claim.

## Identity and coverage checks

- Frozen dataset: `/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl`, 5,388 unique names, SHA256 `271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9`.
- Both frozen BPfold and UFold artifacts contain all 5,388 names in dataset order with no duplicates, missing names, or extras. Every per-sequence TP/FP/FN/F1 and all three recorded repeat prediction hashes/metrics were reproduced.
- BPfold official release: 5,401 author-released BPSEQ files. All 5,388 project names match with exact sequences; 13 official-only records are retained in the normalized release JSONL but excluded from project metrics. No project names are missing.
- Official versus frozen BPfold replay: only 287/5,388 pair sets are identical; 5,101 differ. They are different prediction identities, not interchangeable replicates.
- Official versus project labels: only 106/5,388 same-name/same-sequence label sets are identical; 5,282 differ. This is the central comparability limitation behind the large label-scope score difference.
- ContextFold metrics and coverage remain null, not zero. Summary-only values were never converted into predictions.

## Provenance recovered

### BPfold

- Source commit `d37d6aa10cbca13e590ff83917fc4d63fec2ddbc`, package version 0.3.1, MIT source license.
- Official `model_predict.tar.gz` release asset identity: 178,585,408 bytes, archive SHA256 `7796e4f56101c8a662e19252c4120be941e832deb0c5b593585e607ae9ae9e7b`; all six checkpoint member hashes match the frozen prediction artifact.
- Frozen protocol: six-model `BPfold_predict`, raw full-length sequences, batch size 1, no cached embeddings, `connects` converted to upper-triangle pairs. Historical CUDA artifact was not rerun.
- Official v0.2 `BPfold_test_results.tar.gz`: 8,279,584 bytes, SHA256 `e8a3420e3458f7aad4c86b500f5d56d0e935b7da82b936675fefda64c5e03073`; normalized per-sequence predictions are saved under `predictions/bpfold_official_release/`.

### UFold

- Source commit `75bd9acc83826059682dfca9d3659df66b132cd1`, MIT source license.
- Local `ufold_train_alldata.pt` SHA256 `1b554e1c98d770ced93939f3cdc02da0d7daabfe5a5b8d5ccad277b59d75d4d3`; the author-hosted Drive folder timed out again, so this local file is not newly certified as an author-downloaded asset.
- Frozen predictions use a MultiMolecule conversion, not an original-author checkpoint distribution. `model.safetensors` SHA256 `d8bb850571c7ec8b21a7b44e83663e73f7fe36e6211068a0aea0471a4475fd21`; threshold 0.5; three recorded deterministic repeats verified.
- The official source default training list explicitly includes `bpnew` along with other benchmark sets. The exact effective training union was not reconstructed. Raw UFold results are therefore not clean, homology-controlled, or training-decontaminated comparisons.

### ContextFold

- PMID 22035327 / DOI 10.1089/cmb.2011.0184 confirms the paper and the official trained-model/source URL.
- Both `?negevcb/contextfold` and `~negevcb/contextfold` HTTP/HTTPS variants were unreachable. Targeted GitHub search returned zero repositories. Wayback attempts timed out or reset. No verified version, trained parameters, license, inference protocol, or per-sequence predictions were recovered.
- Detailed attempts and saved PubMed/GitHub responses are in `reports/official_provenance_and_source_attempts_v2.json` and `official_assets/contextfold/`.

## Remaining limitations

1. All reported project-label results are raw and non-decontaminated; they are not clean/generalization claims.
2. BPfold official-release scores are label-protocol-sensitive. Official labels and project labels must not be mixed within a comparison table without an explicit scope column.
3. BPfold author release and frozen project replay differ in source/package/checkpoint identity and predictions; no uncertainty or ensemble claim is inferred from their difference.
4. UFold's author-hosted distribution and exact effective training union remain unresolved; MultiMolecule conversion provenance is retained explicitly.
5. ContextFold remains blocked, with no metric or coverage imputation.

## Reproduction

```bash
R=/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2
/root/miniconda3/envs/lucaone_cu128/bin/python "$R/scripts/recover_bprna_new_baselines_v2_20261124.py" \
  --bpfold-official-archive "$R/official_assets/bpfold/BPfold_test_results_v0.2.tar.gz" \
  --bpfold-official-reference-archive /teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz
/root/miniconda3/envs/lucaone_cu128/bin/python -m unittest discover \
  -s "$R/scripts" -p 'test_*_v2_20261124.py' -v
```

Machine-readable metrics: `reports/bprna_new_raw_metrics_verification_v2.json`. Per-sequence TP/FP/FN/P/R/F1 rows are saved alongside it for all three project-label prediction arms.
