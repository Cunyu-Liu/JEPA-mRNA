# RiNALMo-giga pretraining exposure audit v1

Scope: static, CPU-only audit of existing legal assets from FoldNAR-integrated base commit `79a2ed45879f1f4e81d6719ed54600bf7f8da807`. No model inference, training, external download, new homology search, or model selection was run. External Test sequences are used only for this final exposure/quality audit.

## Identity and corpus boundary

The current candidate backbone is the local RiNALMo-giga asset at `/root/rna_jepa_data/weights/rinalmo-giga/model.safetensors`, SHA256 `0b3f73b5c80d267f0b7e92655b28f16088e3d5a595c6dae631b52142aa38bcda` (33 layers, hidden size 1280, maximum position 1024). Direct byte identity between this MultiMolecule distribution and the Zenodo `rinalmo_giga_pretrained.pt` file is `UNCHECKED`; the audit does not infer it from similar names.

The official RiNALMo repository snapshot declares RNAcentral, Rfam, nt, Ensembl 109, and Ensembl Bacteria 56 as pretraining sources. It does not provide a complete historical sequence manifest. In particular, the nt URL is a live alias without a pinned training-date release or content hash. Therefore the complete official pretraining coverage state is **`UNCHECKED`**. The local public snapshots are evidence sources, not a renamed reconstruction of the exact historical training union.

| Known public source | Local version and integrity |
|---|---|
| RNAcentral | 22.0 active FASTA; 7,281,938,321 bytes; SHA256 `9698f99e05743be566b06f0885e5cae8e087bf781538f32af0950fc8e33c8f2d`; 30,191,670 records scanned |
| Rfam | 14.9 FASTA; 180,119,223 bytes; SHA256 `5909e198df80a0febf6a62cbe128654a2b6ffffafda9b77f258fd4557009cc5c`; 3,132,784 records scanned |
| Ensembl | 109 plus Bacteria 56; 31,046 files and 701,138,004 bytes; independently hash-bound manifest; 4,040,439 records scanned |
| nt | Historical training snapshot unavailable and unpinned; `UNCHECKED`; current nt was not downloaded as a substitute |

Normalization is uppercase plus T-to-U, preserving other symbols and requiring normalized full-sequence equality after the published 16–8192 nt source-length filter. Positive source membership proves exposure to that known source; it does not prove which random crop of a long sequence was used in every training step.

`EXACT` means a known-source full-sequence hit. `HOMOLOGY` means no exact hit but a hit in the existing 80% identity/bidirectional-coverage search. `NOT_DETECTED_UNKNOWN` means only non-detection in the declared known-source search. `UNCHECKED` means that coverage is unavailable or a homology search was not performed. Neither non-detection nor unchecked coverage means clean.

## TR1c cluster Train and development

| Population | Rows | EXACT | HOMOLOGY only | NOT_DETECTED_UNKNOWN | UNCHECKED homology |
|---|---:|---:|---:|---:|---:|
| TR1c raw | 42,564 | 37,870 | 2,464 | 2,230 | 0 |
| cluster80 Train | 33,421 | 29,892 | 1,925 | 1,604 | 0 |
| cluster80 Dev | 9,143 | 7,978 | 539 | 626 | 0 |

The exact Train/Dev ID files have SHA256 `3a5bdb60098c46a4db8923829f34b2827288be7bf54e795cba2627664c0584f3` and `2560ecadf85b2557de048f53c4c0dddbb7841b840feae1e90deed7aceb1f1ae6`, respectively. Cluster separation in downstream training does not remove backbone pretraining exposure. The 1,604 Train and 626 Dev `NOT_DETECTED_UNKNOWN` rows are not clean subsets.

## Registered external evaluations: final audit only

| Evaluation | Rows | EXACT | HOMOLOGY only | NOT_DETECTED_UNKNOWN | UNCHECKED homology |
|---|---:|---:|---:|---:|---:|
| ArchiveII | 2,544 | 2,048 | 462 | 34 | 0 |
| bpRNA-new | 5,388 | 5,381 | 7 | 0 | 0 |
| bpRNA-TS0 | 1,288 | 1,253 | 30 | 5 | 0 |
| PDB TS1 | 63 | 36 | 21 | 6 | 0 |
| PDB TS2 | 39 | 38 | 0 | 1 | 0 |
| PDB TS3 | 19 | 17 | 2 | 0 | 0 |
| PDB TS-hard | 28 | 19 | 5 | 4 | 0 |
| TestSetB | 428 | 416 | 6 | 6 | 0 |
| CHANRG Test | 10,597 | 3,868 | 0 | 0 | 6,729 |
| CHANRG GenA | 5,337 | 2,372 | 0 | 0 | 2,965 |
| CHANRG GenC | 3,936 | 1,458 | 3 | 0 | 2,475 |
| CHANRG GenF | 1,734 | 713 | 0 | 0 | 1,021 |
| Rfam temporal | 436 | 112 | 0 | 0 | 324 |
| Rfam family | 10,790 | 10,327 | 384 | 79 | 0 |
| ncRNA Rfam-novel Test (registered out-of-scope task) | 29,472 | 29,306 | 117 | 9 | 40 |

For CHANRG and Rfam temporal, no registered known-source homology union covers the whole population; those rows remain `UNCHECKED` in the joint exact-or-homology layer rather than being promoted to a negative homology result. Exact non-detection remains separately recorded as `NOT_DETECTED_UNKNOWN` in the machine-readable registry.

### bpRNA-new 5381/5388 verification and baseline separation

The new full-source scan independently reproduces the old union count: RNAcentral 5,303, Rfam 5,380, Ensembl 10, and a deduplicated union of **5,381/5,388 EXACT**. The remaining seven are `HOMOLOGY`, so there is no known-source homology-undetected comparison arm. They are not a clean seven-sequence subset and do not support a causal exposure/accuracy comparison.

This is exposure of the RiNALMo-giga backbone's declared pretraining sources. It is distinct from the official secondary-structure baseline `rinalmo_giga_ss_bprna_ft.pt`, SHA256 `44e377cb0c92f7b9cff8db8a38e33e5c0363d2f9a19831c6342346bb788a5e55`, whose registered supervised fine-tuning corpus is bpRNA TR0. That baseline's fine-tuning exposure is additional and must not be attributed to, or substituted for, the current candidate backbone's self-supervised pretraining audit.

## Registry and independent verification

- `manifests/pretrain_exposure_v1/asset_registry.json`: backbone, official declaration, source versions/hashes, nt gap, and baseline separation.
- `manifests/pretrain_exposure_v1/sequence_registry.jsonl.gz`: 157,227 population-row records with stable query ID, row index, sequence SHA256, normalized length, source record ID/header/file/count, layered states, and selection-use flag. Raw sequences and external labels are not copied into the repository.
- `manifests/pretrain_exposure_v1/summary.json`: row and unique-sequence counts by population and source.
- `manifests/pretrain_exposure_v1/independent_verification.json`: independent JSONL/FASTA parser, full source re-scan, per-row source evidence and state checks, summary reconciliation, and legacy-count reconciliation.

## Required SOTA wording limits

Supported wording is **exposure-aware descriptive performance on the registered benchmarks**. The results do not support `clean-OOD`, `pretraining-unseen`, `uncontaminated SOTA`, or superiority on unseen RNA families. Known-source positive exposure invalidates such wording for the affected populations; incomplete nt coverage prevents estimating a complete exposure rate or certifying an uncontaminated remainder.

Any leaderboard must retain the benchmark-specific exposure layer and must not convert `NOT_DETECTED_UNKNOWN` or `UNCHECKED` into clean labels. External Test exposure results must not select a model, arm, checkpoint, threshold, or hyperparameter. Downstream cluster cleaning and family-based splitting remain useful but are not substitutes for backbone pretraining decontamination.
