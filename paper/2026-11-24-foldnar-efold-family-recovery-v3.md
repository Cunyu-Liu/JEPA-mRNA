# FoldNar baseline recovery v3: eFold, E2Efold, and DEBFold

## Scope and evidence classes

Recovery was performed on px2 for `/home/rna_jev_xens2_dualview_round56`. The frozen project label scope is `/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl`, SHA256 `271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9`: 5,388 unique names and 135,184 reference pairs. Pair scoring uses exact zero-based inclusive `(i,j)` pair sets, rejects invalid indices and partner conflicts, and applies no crossing or short-span filtering. Empty-truth/empty-prediction F1 is 1; undefined per-sequence precision or recall is null.

Only eFold has a frozen bpRNA-new per-sequence prediction artifact. It is **project replay using official source and weights**, not an author-released prediction set. E2Efold and DEBFold have no eligible bpRNA-new predictions, so their metrics and coverage are null. No paper summary, third-party model, reference label, or zero score is substituted. Recovery used CPU parsing and downloads only; no model inference or GPU call was made.

## Exact bpRNA-new results

| System | Evidence class | Predictions | Coverage | TP | FP | FN | Micro precision | Micro recall | Micro F1 | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eFold 0.1.3 | Project replay, official source/weights | 5,388 / 5,388 | 1.0 | 100,583 | 73,955 | 34,601 | 0.5762813828507259 | 0.7440451532725766 | 0.6495050400036162 | 0.6556973548256184 |
| E2Efold | Blocked provenance | None | null | — | — | — | null | null | null | null |
| DEBFold | Blocked provenance | None | null | — | — | — | null | null | null | null |

The dashes denote unavailable counts, not zero. These are raw project-label results, not an own-training-clean or cross-protocol SOTA ranking.

## eFold replay verification

- Frozen prediction artifact: `predictions/efold_project_replay_bprna_new_frozen_v3.json`, SHA256 `17e2652ea8ea622574efc2ff8d4e277509d8d9d61e04040fb9c1c8a244a15e3e`.
- Official repository: `https://github.com/rouskinlab/eFold`, commit `4e146fe92c574104b2518d7350e0fa4a39f32030`, version 0.1.3.
- Official weights: `official_assets/efold/efold_weights.pt`, SHA256 `fcf06c6a7724501e0fb1c5f12bc21ab5bfe7d6131e643d41f139be776aa1dca9`.
- Recorded protocol: raw-sequence end-to-end, exact-length groups, official constraints, UFold postprocess and Hungarian decode, eval mode with a single model load, batch size 1. The historical replay ran on an RTX 4090; the present recovery only parsed its frozen output.
- All 5,388 recorded per-sequence TP/FP/FN/F1 values were reproduced. All three recorded repeats match the recomputed counts and micro/macro F1.
- Legacy replay prediction SHA256 `567520183e007c2b3a50c67884162cf556190c6dc46ec03bf51aaa7aba2a70a2` was reproduced using the original stable length-group ordering and `{name,pairs}` objects. A separate dataset-order name/pair identity SHA256 is `1b7e8a50c8ce1b1afdd34423d89c6c28e93f0901dfb4ee111c2e8aedfdcfd7f5`; the different scopes are intentional and explicit.
- The official GitHub releases API response is empty. No author-released bpRNA-new per-sequence prediction set was found.

Per-sequence recomputation is stored in `reports/efold_project_replay_bprna_new_per_sequence_v3.jsonl` (5,388 rows).

## E2Efold: concrete missing provenance

Official source is preserved at commit `2aab9e78326ddaa27d14c49eb34001c0003a1960` (`https://github.com/ml4bio/e2efold`). Its README documents RNAStralign/ArchiveII protocols, not bpRNA-new, so those scopes cannot be merged. The checkpoint directory contains only its README; the GitHub release list is empty.

On 2026-11-24, both the official Google Drive checkpoint folder (`1m038Fw0HBGEzsvhS0mRxd0U7cGXqLAVt`) and released-results endpoint (`1ASAiXmiH-d6zdg1wNB0if-o-d1TU9lpb`) timed out after 60 seconds. Curl logs are preserved under `official_assets/e2efold/`; no response body was treated as a model or prediction. Historical SharePoint/access audits are also preserved.

Missing items are: (1) bpRNA-new per-sequence predictions and hash, (2) an eligible official checkpoint and hash, and (3) a bpRNA-new inference/postprocessing record linked to that checkpoint. Therefore metrics and coverage remain null.

## DEBFold: concrete missing provenance

The official repository `https://github.com/cobisLab/DEBFold` was fixed at commit `ff693cdf4cb177e50d8a6a15c7a6292af802b75e`. Its complete Git tree contains only `README`, `TestSet_alpha.zip`, `TestSet_beta.zip`, `TestSet_gamma.zip`, and `TrainingValidationSet.zip`; the README points to a web service. The release list is empty. No local inference source, checkpoint, or bpRNA-new prediction release is present.

`TrainingValidationSet.zip` is preserved with SHA256 `20d4404ffa2c1d2b973f230de859bbfadd47cf3de46c3164e7b8c550b891c6a2`. Its CT structures are training-provenance labels only, never test predictions or project test labels. A partial codeload download is explicitly marked `.partial` and was not used as validated source; commit/tree/README API evidence was used instead.

Missing items are: (1) bpRNA-new per-sequence predictions and hash, (2) official local inference source and checkpoint with hashes, and (3) a batch bpRNA-new protocol linked to predictions or a checkpoint. Therefore metrics and coverage remain null.

## Files and verification

Main repository additions are limited to:

- `tools/recover_efold_family_baselines_v3_20261124.py`
- `tests/test_recover_efold_family_baselines_v3_20261124.py`
- `tools/verify_efold_family_recovery_v3_20261124.py`
- `reports/baseline_recovery_v3_efold_family_20261124.md`

All large assets, source snapshots, scripts, and reports are under `/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v3_efold_family/`. No shared registry, running config, evaluator, or PPT was edited; no commit or push was performed.

The seven unit tests cover exact TP/FP/FN arithmetic, empty-pair convention, name-scope mismatch, partner conflicts, out-of-range pairs, legacy hash ordering, and null blocked results. `delivery_verification_v3.json` records a second standalone recomputation, expected exact values, row counts, null-scope checks, and SHA256 for every core deliverable.
