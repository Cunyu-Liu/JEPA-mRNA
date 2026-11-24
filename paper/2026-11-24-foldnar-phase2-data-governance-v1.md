# Phase 2 development-data governance audit

Scope: CPU-only TR1c and CHANRG Train/Validation. No Test/GenA/GenC/GenF asset is accessed. This is an observed-data audit, not a model result or a sampling decision.

## Source populations

| Asset | Rows | Duplicate ID rows | Pairs |
|---|---:|---:|---:|
| tr1c | 42564 | 3496 | 1157273 |
| chanrg_train | 95181 | 0 | 2891878 |
| chanrg_validation | 10540 | 0 | 326559 |

## CHANRG matched-clean accounting

Cross-split 80% clusters: 3250. Conservative both-side exclusion accounts for 16322 Train rows and 4668 Validation rows. These are metrics-only populations; no subset or ID file is produced.

## Raw leakage states

| Comparison | Exact | 80% homology | Exact topology | Pretraining | Temporal/label |
|---|---|---|---|---|---|
| tr1c_registered_train_vs_dev | NOT_DETECTED_UNKNOWN | NOT_DETECTED_UNKNOWN | EXACT | UNCHECKED | UNCHECKED |
| chanrg_raw_train_vs_validation | NOT_DETECTED_UNKNOWN | HOMOLOGY | EXACT | UNCHECKED | UNCHECKED |
| chanrg_matched_clean_accounting | NOT_DETECTED_UNKNOWN | NOT_DETECTED_UNKNOWN | EXACT | UNCHECKED | UNCHECKED |
| tr1c_vs_chanrg_train | EXACT | UNCHECKED | EXACT | UNCHECKED | UNCHECKED |
| tr1c_vs_chanrg_validation | EXACT | UNCHECKED | EXACT | UNCHECKED | UNCHECKED |

## Boundaries

- NOT_DETECTED_UNKNOWN is limited to the named exact/alignment/cluster check and never means clean.
- TR1c own-clean is a supplied exact-downstream-clean manifest label; downstream sets were not opened in this assignment.
- Chemical modified-parent mapping and per-record experimental quality are unavailable in the aggregated JSONL.
- Genomic accession parent groups in CHANRG are not chemical modified-parent mappings.
- Exact topology signatures are not structural-similarity searches.
- Pretraining exposure and temporal-label provenance remain UNCHECKED.
- Matched-clean populations are descriptive accounting views and authorize no sampling, training, or route decision.

Detailed distributions, pair QA, macro-risk strata, annotation identities, MMseqs commands, and file hashes are in the accompanying JSON artifacts.
