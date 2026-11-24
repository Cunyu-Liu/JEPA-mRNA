# FoldNar foundation-model baseline recovery v3

Date: 2026-11-24
Scope: existing frozen bpRNA-new predictions only. All new computation was CPU-only parsing, hashing, provenance download, and exact-pair scoring. No model inference, training, calibration, model selection, imputation, shared-registry/config/evaluator/PPT edits, commit, or push was performed.

## Independently verified raw metrics

The frozen project data are `/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl`, 5,388 unique sequences and 135,184 reference pairs, SHA256 `271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9`. Scoring uses zero-based inclusive exact pairs with no added crossing-pair or short-span filter. Micro P/R/F1 use pooled TP/FP/FN. Macro-F1 is the unweighted per-sequence mean; F1=1 only when truth and prediction both have zero pairs. All five available prediction identities cover **5,388/5,388 (100%)**, in dataset order, with no missing names, extras, duplicates, or invalid pair indices.

| Model and exact downstream identity | Origin / model form | TP | FP | FN | Micro P | Micro R | Micro F1 | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RiNALMo-giga author bpRNA fine-tune | Author checkpoint; single model | 51448 | 42537 | 83736 | 0.5474065010373996 | 0.38057758314593443 | **0.4489961556755058** | 0.44123738936817275 |
| NucleicBERT author `secstr.ckpt` | Author checkpoint; single model; **full fine-tune** | 15926 | 15754 | 119258 | 0.5027146464646465 | 0.11780979997632857 | **0.19088599098667178** | 0.18515300147695582 |
| NucleicBERT frozen encoder + project TR1c head | Project downstream checkpoint; single model | 60809 | 156240 | 74375 | 0.2801625439416906 | 0.4498239436619718 | **0.34527713189848763** | 0.3611124978084714 |
| structRFM author `structRFM_SSP_bpRNA1m.pt` replay | Author checkpoint; single model | 73864 | 62616 | 61320 | 0.5412075029308323 | 0.5463960231980116 | **0.5437893868896873** | 0.5338105702966888 |
| RNA-FM frozen 640-d encoder + project CRF/decision head | Project downstream checkpoint; single model | 37568 | 77317 | 97616 | 0.3270052661357009 | 0.27790271037992664 | **0.3004610727439227** | 0.3201747208074672 |
| ERNIE-RNA official bpRNA-new protocol | Official source/protocol recovered; full predictions absent | — | — | — | — | — | **null** | **null** |

ERNIE-RNA coverage is **0/5,388 usable full-project predictions (0%)**, not a zero-F1 result. Its official repository provides only a one-sequence bpRNA-new FASTA/CT demonstration, which was not promoted to a benchmark prediction set. No paper or summary value was substituted.

These are raw, non-decontaminated, project-label measurements—not clean-OOD, independent-family, uncertainty, or unified-SOTA claims. Author checkpoints and project downstream adaptations must not be collapsed into one row per backbone. The three recorded RiNALMo/NucleicBERT repeats are deterministic inference repeats, not training seeds or ensembles. Every listed system is a single prediction model, not a multi-checkpoint ensemble.

## Downstream protocol distinctions

- **RiNALMo:** author `rinalmo_giga_ss_bprna_ft.pt`, RiNALMo-giga plus two-block ResNet secondary-structure head, checkpoint-selected threshold, raw full-length B=1 replay. Official source: `lbcb-sci/RiNALMo`; current official main commit recovered as `2c2c5c14a5ae609d8c560a5d9ca32e51e0288955`. The frozen prediction artifact separately records project code commit `583488192e2da2c12d7abd40c20cfaa605916090`; these identities are not conflated.
- **NucleicBERT author arm:** official `secstr.ckpt`, epoch 99, global step 258000, threshold 0.5, B=2, first-598-nt protocol. Crucially, official `configs/secstr.yml` says `frozen: False`; this is full fine-tuning despite the historical directory/arm name `nucleicbert_frozen`. It must not be described as a frozen-encoder probe.
- **NucleicBERT project arm:** official pretrained encoder frozen, official `SecStruct2DPredictionHead` trained by the project on `bprna_tr1c` with BCE and `pos_weight=100`, VL0-selected threshold 0.55, original B=4 first-598-nt padding behavior. This is distinct from the author full-fine-tuned checkpoint.
- **structRFM:** historical replay records author `structRFM_SSP_bpRNA1m.pt`, bpRNA1m downstream fine-tuning, frozen LM embeddings into MixedFold CNN/LSTM pair scoring with Turner parameters and constrained DP decoding. `n_over_maxlen=0` at `max_len=512`. Official release v0.0.8 and source commit `f0c1add19f8dfc786f16bc4abb7aa775dbadddbf` were recovered.
- **RNA-FM:** frozen 640-d RNA-FM embeddings plus the project CRF/decision head, TR0 training for 20,000 steps with VL0 protocol, learned Turner-prior weight, exact max-product/Nussinov decode. This is **not** the official `RNA-FM-ResNet_bpRNA.pth` model. The official bpRNA config uses `pc-resnet_1_sym_first:r-ss` and threshold `0.47729617`; no frozen author prediction artifact for that official arm was found.
- **ERNIE-RNA:** official `dataset_name=bpRNA-new` selects `ERNIE-RNA_attn-map_frozen_ss_prediction_bpRNA-1m_checkpoint.pt`; bpRNA-new is not its downstream training set. Official inference uses attention-map constrained post-processing and CT output. Source main commit `43bc06de1088ed03ffd7de918ad4b2c2a3346a43` and author checkpoint links were recovered, but full project predictions were not available.

## Checkpoint and source verification

- **RiNALMo PASS:** local checkpoint is 2,604,809,354 bytes; MD5 `3688b049ac282fa2088fcd80854e34dd` matches Zenodo record 15043668. SHA256 `44e377cb0c92f7b9cff8db8a38e33e5c0363d2f9a19831c6342346bb788a5e55` matches the frozen prediction metadata.
- **NucleicBERT author PASS:** local `secstr.ckpt` is 4,970,204,510 bytes; MD5 `b5b3d40fa07adc3cc8c1942e6ebfe1a5` matches Zenodo record 16989562. SHA256 `90b9af474693db3879830a455f4c25662d672dc0f4e9cdc61b90055e9b6d5e07` matches the frozen metadata.
- **NucleicBERT project PASS:** official `pretrained.pt` MD5 `170a248d9e82dc9707c00ee7b4e0ac8d` matches Zenodo; SHA256 `f40630f4054b0bac255eb70fa1e2999d56749f4fc22919be041e5114bcbe11dc`. Project downstream checkpoint SHA256 `127aab7b07ff024a4d0eb14dcd8183934374c2119d5740f73c072c6f10dfb77f` matches the frozen metadata.
- **structRFM partial:** official v0.0.8 release identifies a 348,179,698-byte checkpoint with SHA256 `689e09e0ad3b51af19b03cb32ea3be5237384c3c9aa5716dff29989098e12145`. The historical local checkpoint is absent on px2. Three direct downloads timed out after partial transfers; partial bytes were deleted. Therefore the official expected identity is recovered, but the checkpoint used for the historical predictions is **not independently byte-verified here**.
- **RNA-FM partial:** official backbone name/source and downstream config were recovered. The historical pretrained backbone and project head checkpoint paths are absent on px2, so no new weight checksum is claimed. The complete frozen prediction file itself is hash-verified.
- **ERNIE-RNA partial:** official checkpoint links and source are verified as provenance; no local weight-byte verification or prediction-metric claim is made.

## Pretraining exposure and homology limitations

RiNALMo has a stronger existing audit, and it is unfavorable to a clean interpretation: the versioned RNAcentral 22.0/Rfam 14.9/Ensembl 109/56 union contains exact matches for **5,381/5,388** bpRNA-new sequences, and the known-source 80%-identity/bidirectional-coverage search detects homology for **5,388/5,388**. The seven exact-undetected sequences remain **UNKNOWN, not clean**. No metrics were imputed for empty clean strata. The exact summary SHA256 is `d441239311605a2572b3f561357531689b0ea24a19068e1ac417dca75148baa4`; homology summary SHA256 is `69a0b13b2ef0cb061807e10f50efca3a6da7889ee006cd032e2e72bca2197131`. These public sources are not a complete historical pretraining corpus; the nt snapshot and family independence remain unresolved.

For RNA-FM, NucleicBERT, structRFM, and ERNIE-RNA, model-specific bpRNA-new exact/homology clearance is **unresolved**. Official pretraining sources include RNAcentral100 (RNA-FM), MARS (NucleicBERT), and RNAcentral/BPfold-derived sequence-structure data (structRFM). structRFM also uses bpRNA1m downstream. Raw scores must not be called decontaminated merely because a downstream TR0/TR1c split was cleaned. ERNIE-RNA's explicit “bpRNA-new not trainset” statement addresses that downstream head only, not pretraining exposure or homology.

## Artifacts and reproduction

Artifact root: `/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v3_foundation/`.

- `reports/foundation_bprna_new_metrics_verification_v3.json`: machine-readable metrics, coverage, protocols, source/weight status, and exposure limitations.
- `predictions/<identity>/bprna_new_v3.jsonl`: normalized exact per-sequence predictions for all five full arms.
- `reports/<identity>_bprna_new_per_sequence_v3.jsonl`: independently recomputed per-sequence counts and P/R/F1.
- `official_assets/`: official API responses, decoded source/protocol files, checkpoint metadata, and README provenance.
- `scripts/`: standalone recovery, unit tests, and delivery verifier.

```bash
R=/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v3_foundation
/root/miniconda3/envs/lucaone_cu128/bin/python \
  /home/rna_jev_xens2_dualview_round56/tools/recover_foundation_baselines_v3_20261124.py
cd /home/rna_jev_xens2_dualview_round56
/root/miniconda3/envs/lucaone_cu128/bin/python -m unittest \
  tests.test_recover_foundation_baselines_v3_20261124 -v
/root/miniconda3/envs/lucaone_cu128/bin/python \
  tools/verify_foundation_recovery_v3_20261124.py
```
