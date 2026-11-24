# FoldNar living preprint claim–evidence registry

Snapshot: 2026-11-24T08:45:03Z. Internal scientific contract, not an abstract/title draft or final delivery. JSON source of truth: `2026-11-24-foldnar-preprint-claim-evidence-registry.json` in this directory. `SUPPORTED` never implies abstract/title eligibility; no performance, mechanism, OOD, long-sequence, or SOTA claim is currently authorized for either.

## Planned story claims

| ID | Claim and phenomenon | Intervention → expected observation | Evidence artifact(s) | Limitation | Status |
|---|---|---|---|---|---|
| C01 | Isolated/long-span/competitive pairs are reproducible local-score error modes. VL0 isolated recall is 0.1424731183 (319/372 FN), but this is a small descriptive stratum. | Frozen multi-environment error decomposition and non-local context → target P/R gains with paired sequence/family-cluster CIs, adequate counts, no single-family driver, and pair/stem preservation. | A01–A03 | VL0 n=196 cannot establish universality or causality; long-span/competitive effects remain unconfirmed. | **IN_PROGRESS** |
| C02 | GEC improves isolated/long-span pairs and whole-structure metrics through non-local ensemble feedback; H-GEC is still a hypothesis. | Four matched seed0 arms → all micro/macro pair/stem, precision, target-stratum, interval-shuffle, and independent-context gates pass before three seeds. | A04–A06, A14–A15 | Numerical/smoke checks are not performance evidence. Seed0 v2 failed at first validation serialization before evaluation; v1 also failed without evaluation. Neither falsifies H-GEC; raw ranking/loss cannot rescue structure. | **IN_PROGRESS** |
| C03 | A single-variable data intervention improves macro generalization without sacrificing micro. Audits show source heterogeneity and leakage risks, not intervention efficacy. | Diagnostics-motivated balanced/filter arm versus identical model/budget → macro pair/stem and target retention improve, micro is preserved, and sampling/deletion conservation holds. | A02, A07 | No sampling/cleaning/training arm has run; descriptive matched-clean accounting authorizes no route decision. | **UNSUPPORTED** |
| C04 | One frozen candidate preserves gains across family/topology/length/time/source/exposure shifts. Current panels cover only limited separations. | Pass development gates, freeze all choices before external reads, then evaluate one candidate → full-coverage pair/stem/junction/topology gains with paired 95% CIs and Holm correction. | A02, A08 | No selected candidate; external evaluation is closed. Unchecked exposure/lineage and blocked temporal/PDB assets remain. | **UNSUPPORTED** |
| C05 | FoldNar beats strongest same-protocol baselines on eight main benchmarks, Rfam, CHANRG, and long sequences. Raw bpRNA-new shows a large gap, not a ranking victory. | Recover official predictions/assets and run one frozen candidate → same-label/sequence/metric/coverage wins with uncertainty and compute reporting. | A08–A12 | Unified SOTA=0/8; 62/63 baselines have blocked provenance. Raw/clean/CHANRG/label scopes and summary-only values cannot be mixed. | **UNSUPPORTED** |
| C06 | Sparse approximate GEC gives accurate full-length 7,224/19,000-nt predictions with tractable compute. Exact DP is O(L³), so short-sequence success does not scale by assertion. | After exact GEC passes, preregister sparse intervals → full outputs, candidate recall, accuracy, coverage, memory/FLOPs/time, and LinearPartition-class comparison without truncation. | A08, A13 | Both long-chain assets are blocked; no sparse result exists. Use `APPROXIMATE_SPARSE`, never “exact global DP.” | **UNSUPPORTED** |
| C07 | The implemented GEC numerical context and four-arm path pass engineering checks. | Float64 CPU/CUDA, independent exhaustive, and real two-step resume tests → all focused checks pass with common initialization/order. | A04, A05 | Support is limited to 12 CPU + 6 CUDA + 5 independent tests, 1,280 exhaustive cases, and smoke; no accuracy, calibration, convergence, novelty, or scaling claim. | **SUPPORTED** |
| C08 | On frozen raw project bpRNA-new labels, xens2 has substantial exact micro-F1 gaps to MXfold2, EternaFold, and frozen BPfold replay. | Independent exact-pair scoring, 5,388/5,388 coverage → gaps below. | A08, A11, A12 | Raw non-decontaminated arithmetic only; unresolved training overlap blocks clean/OOD/SOTA interpretation. | **SUPPORTED** |

The execution prompt's eight story sections map as follows: phenomenon→C01; question/method/mechanism→C02 (method readiness only→C07); data→C03; generalization→C04; performance→C05 with raw calibration→C08; computation→C06.

## Absolute raw bpRNA-new gaps

Scope for every gap: `PROJECT_RAW_GT_ONLY`, frozen project `bprna_new.jsonl` SHA256 `271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9`, zero-based inclusive exact pairs, no added crossing/short-span filter, pooled strict micro-F1, 5,388/5,388 coverage, raw and non-decontaminated. xens2=0.5531420736.

| Comparator | Strict micro-F1 | Comparator − xens2 | Protocol scope |
|---|---:|---:|---|
| MXfold2 v0.1.2 TrainSetAB MixC | 0.6688466171284047 | **0.1157045435284047** | Official CPU `@TrainSetAB.conf`, frozen predictions; training overlap unresolved. |
| EternaFold Params.v1 | 0.685384007269125 | **0.132241933669125** | Official serial `predict`, default MEA, frozen predictions; training overlap unresolved. |
| BPfold frozen project replay | 0.7370074184282851 | **0.1838653448282851** | Six-model `BPfold_predict`, project labels; training overlap unresolved. |

BPfold identity warning: official-release predictions score 0.9616708875167174 on same-name project labels but 0.633624758107719 micro-F1 / 0.6470223791166072 macro-F1 on all 5,401 official-label records. The project-label value is a label-sensitive audit, not the official result or a valid mixed-label gap.

## GEC readiness and live seed0 boundary

- Numerical support: float64 atol/rtol=2e-10, mass atol=2e-09; 12 CPU passed/6 skipped with CUDA hidden, 6 CUDA passed/12 deselected, 5 independent exhaustive tests passed, 1,280 short-sequence cases. Four-arm two-step smoke passed with real resume and no CPU fallback. This supports C07 only.
- At 08:45:03Z, seed0 v2 `ensemble_context` had failed at step-500 validation serialization because `allow_nan=False` rejected a non-finite float; `matched_control` was terminated as its peer, and `destroy_interval_shuffle`/`independent_context` were not launched. No scientific evaluation or performance verdict exists. Seed0 v1 also stopped without evaluation. These preserved engineering failures do not falsify H-GEC, are not final delivery, and authorize no performance sentence; a new attempt first needs diagnosis and a finite-value engineering gate.

## Data-governance findings

- TR1c: 42,564 rows, 3,496 duplicate-ID rows beyond first, 1,157,273 pairs. Frozen cluster80 split: 33,421 train/9,143 dev sequences; family split: 19,519 train/4,846 dev. VL0=196 is legacy control only.
- Frozen CHANRG Train/Validation contain 95,181/10,540 rows versus contract counts 123,223/14,070: retain the subset/version mismatch. Raw Train/Validation has 80% `HOMOLOGY`; 3,250 shared clusters lead to descriptive exclusions of 16,322 Train and 4,668 Validation rows.
- TR1c has exact sequence overlap with CHANRG Train and Validation. Exact topology signatures are not structural-similarity searches. Pretraining exposure, temporal/label lineage, chemical modified-parent mapping, and per-record experimental quality remain unresolved. `NOT_DETECTED_UNKNOWN` is never “clean.”

## OOD, long-sequence, and SOTA release requirements

External evaluation stays closed until seed0, three-seed, and frozen multi-environment gates pass. Required scopes are the eight main sets (`archiveii_embok_clean`, `bprna_new`, `bprna_ts0`, `ref_pdb_ts1/2/3/hard`, `testsetb`), `rfam_temporal`/`rfam_fam`, CHANRG Test/GenA/GenC/GenF, and topology/length/span/temporal/source/pretraining-exposure strata. One frozen candidate must have raw exports, per-sequence predictions, independent recomputation, same-protocol strongest baselines, paired 95% CIs, Holm correction, and compute costs. Current unified complete SOTA is **0/8**.

Long-chain 7,224/19,000-nt and temporal-1,805 assets are blocked. No truncation or deleted coverage is allowed; accept encoder, candidates, and complete structures separately. Report candidate recall before accuracy and label sparse execution `APPROXIMATE_SPARSE`; compare with LinearPartition-class coverage/accuracy. Checkpointing is only a memory technique.

## Artifact ledger

Paths are absolute; hashes are full SHA-256. The JSON contains machine-readable claim→artifact references.

- **A00** `/home/rna_jev_xens2_dualview_round56/docs/plans/2026-11-18-foldnar-scientific-story-execution-prompt.md` — `2bb52d87c9c58d376881ad9324e0798d82c7d24e141143ec9e158c30fd41e852`
- **A01** `/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/pair_form_hypothesis_v1/stacking_fn_audit.json` — `72cc68cb6e53b3c91db07dfa84fcc0a6e9dfc6d7b1c5a1008346cb3eff74642d`
- **A02** `/home/rna_jev_xens2_dualview_round56/manifests/foldnar_v1/model_dev_registry.json` — `62ae08e06f1b3a291c2de2a9cf52d2e57ec55570396fc810cd3638612c774902`
- **A03** `/teams/HoulabData_1788502323/liucunyu/foldnar/model_dev_v1/environment_summary.json` — `857b96c33209d7a9ace76cf669aea8297976474613cd216f7a3952a677aab7d5`
- **A04** `/teams/HoulabData_1788502323/liucunyu/foldnar/gec_numeric_v1/manifest_v1.json` — `c98ede00d047b5d8db541fc2b18427ae7e4ee63d5e1ffbfdd472ca24bb59d649`
- **A05** `/teams/HoulabData_1788502323/liucunyu/foldnar/gec_smoke_v2/generation_02/smoke_verification_v2.json` — `59e95d5d7dc77f9be4eac1cf513dceda3e353fd2da0821a5ab02f68031ea50fe`
- **A06** `/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v2/production_launch.json` — `0f89518dc5e34648de001c49ffcb58fd302d8be680c0ab7bbdcbaf11f729bb9a`
- **A07** `/teams/HoulabData_1788502323/liucunyu/foldnar/data_governance_v1/audit_summary.json` — `a8af93d974c6c80797fb04ba8535df9250233f402ab429d3f239d62a6e75526c`
- **A08** `/home/rna_jev_xens2_dualview_round56/manifests/foldnar_v1/data_registry.json` — `4fe3f5376498b64ffcce758300067f3bcf1489868dbe3156d84b886f0c16fd69`
- **A09** `/home/rna_jev_xens2_dualview_round56/manifests/foldnar_v1/baseline_registry.json` — `51878f149f7521ab714b8ed7e1db9cee8dab11f32639aa38a0f6a0fb65711c8a`
- **A10** `/home/rna_jev_xens2_dualview_round56/manifests/foldnar_v1/prediction_registry.json` — `da2b6fadaf994496dc004375be96cda9fdaa024dba4ce0e465738f048010d1c9`
- **A11** `/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v1/phase0c_verification.json` — `3e29b2e2461fdeb14a1a7cfc041e749692ebf95e297e8fb0118bd216fbc28ef2`
- **A12** `/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2/reports/bprna_new_raw_metrics_verification_v2.json` — `6bb55b84a655d677b6c17c04adf8406b464df6ef25d060190b7840c59e92c8ca`
- **A13** `/home/rna_jev_xens2_dualview_round56/docs/plans/2026-11-24-phase0b-global-ensemble-context-direct-prior-audit-v1.md` — `97649b4f596075aa659c45c29143fc6889ac39c9d9ce4974e57f18e0f1ebc0a9`
- **A14** `/home/rna_jev_xens2_dualview_round56/logs/gec_seed0_production_v1.supervisor.log` — `45bac1f9e40abad433a81db6e1cd956e2e76bccf573d39ecdd1d1d6c840ab40c`
- **A15** `/home/rna_jev_xens2_dualview_round56/logs/gec_seed0_production_v2.supervisor.log` — `4308b0e5fce46b38fa07b1ee31350278cd82e785311f045e309b809c0adc038d`

## Update rule

Change `current_status` to `SUPPORTED` only with direct evidence for that exact claim and scope; keep `supported` and `abstract_title_eligible` consistent. Preserve failures and negative controls, add new artifacts rather than overwriting history, and continue the preregistered scientific route after failure instead of presenting failure as the paper. Validate with:

```bash
cd /home/rna_jev_xens2_dualview_round56
/root/miniconda3/bin/python3 tools/validate_foldnar_claim_evidence_registry.py \
  docs/plans/2026-11-24-foldnar-preprint-claim-evidence-registry.json \
  --markdown docs/plans/2026-11-24-foldnar-preprint-claim-evidence-registry.md \
  --self-test
```
