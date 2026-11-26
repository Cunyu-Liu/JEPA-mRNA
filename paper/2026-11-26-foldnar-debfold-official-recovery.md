# FoldNAR DEBFold official recovery — 2026-11-26

## Delivery status

**Not full-ready for inference or SOTA scoring.** The official public data, source snapshot, paper protocol, bpRNA-new 5,388-row I/O contract, and a fail-closed GPU launcher are ready. The official local implementation and weights are not published in the official repository or website download page. No official weight SHA can therefore exist. No predictions, coverage, or metrics are reported; these fields are null, not zero.

The worktree is `/home/rna_jev_debfold_recovery`, based on FoldNAR `906f554cd59454b09101cea447583836215be3e2`. Large/public assets are isolated under `/teams/HoulabData_1788502323/liucunyu/foldnar/debfold_official_recovery_20261126/`. Recovery performed downloads and CPU/static checks only; no GPU inference was launched and the occupied GPU was not disturbed.

## Official provenance

- Repository: `https://github.com/cobisLab/DEBFold`
- Frozen official commit: `ff693cdf4cb177e50d8a6a15c7a6292af802b75e` (`main`, complete three-commit history inspected).
- Complete tree: `README`, `TrainingValidationSet.zip`, `TestSet_alpha.zip`, `TestSet_beta.zip`, `TestSet_gamma.zip`.
- Web service/download: `https://cobis.bme.ncku.edu.tw/DEBFold/` and `https://cobis.bme.ncku.edu.tw/DEBFold/download`.
- Paper: Tzu-Hsien Yang, *DEBFold: Computational Identification of RNA Secondary Structures for Sequences across Structural Families Using Deep Learning*, JCIM 2024, DOI `10.1021/acs.jcim.4c00458`, PMCID `PMC11094721`.
- Releases/tags: both empty. GitHub license endpoint: HTTP 404. There is no repository/model/weight license; the paper's CC-BY status is not a substitute.
- Every website ZIP is byte-identical to its GitHub counterpart. Exact URLs, sizes, SHA256 values, retrieval evidence, and paper XML hash are recorded in `manifests/foldnar_v2/debfold_official_recovery_20261126.json` and the artifact-root reports.

The source statement in the paper says the implemented pipeline is available, but the actual public endpoints expose only the web prediction interface and data archives. This recovery does not invent a local implementation from the paper, use an unrelated same-name repository, create pseudo-weights, or substitute paper/summary values for predictions.

## Official training and inference protocol

Stage 1 input is an `l x 10` tensor: an `l x 4` RNA one-hot encoding plus six binary paired-location tracks from RNAfold, IPknot, MaxExpect, ProbKnot, RNAprob, and Fold. It produces a length-`l` folding-probability vector `Y`. Stage 2 uses `S = 1 - Y` as SHAPE-like soft constraints for RNAstructure `Fold` minimum-free-energy optimization, with default slope/intercept. Training pads to length 512 and prediction does not require padding. Methods suggests `<512 nt` as the most suitable regime; the Limitations section separately states that the current version is suitable only for sequences up to 512 bps. Therefore any project rows over 512 nt require an additional verified official protocol rather than silent extrapolation.

Training uses family-wise 25-fold cross-validation, 250 epochs, batch size 512, Adam, and summed per-base binary cross-entropy with positive weight 1.8. Learning rate follows cosine decay to `1e-13` for 227 epochs, then exponential decay with multiplicative factor `1e-10` for 23 epochs. Dropout is 0.1 in residual layers and 0.4 in convolution/transpose-convolution layers. The network has two self-attention blocks, 16 heads, Q/K/V dimension 512, and feed-forward dimension 1024. The paper reports an NVIDIA RTX 4080 for training; this is provenance, not an environment claim for px2.

The training-validation archive contains 8,782 CT structures and 8,384 unique sequences. It was constructed from bpRNA-1m/Rfam 12.2: 43,269 valid sequences in 2,125 families, 44 families reserved for TestSetα, 2,081 families for training-validation, and at most five sequences per family. TestSetα/β/γ contain 44/40/15 structures. These external test archives were audited only and were not used for tuning.

## bpRNA-new 5,388 contract and protocol distinction

Frozen project input: `/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl`, SHA256 `271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9`, exactly 5,388 unique names, with lengths 33–489 nt and zero rows over 512 nt. The launcher passes only `{name,seq}` to a future verified official adapter; reference pairs remain outside model input and are reserved for later evaluation.

Required prediction output is one JSONL object per sequence, `{name,seq,pred_pairs}`, with zero-based inclusive `i < j` pairs. Complete name and sequence identity over all 5,388 rows is mandatory. Summary-only output, partial coverage, duplicate/unknown names, sequence mismatches, invalid pairs, and duplicate pairs are rejected.

DEBFold's paper **does not evaluate the full project 5,388 set**. Its bpRNA-new-derived TestSetβ retains 481 families/5,265 eligible sequences after family/clan and alphabet filtering, then samples four families from each of ten length bins, yielding 40 sequences with CD-HIT 0.85 separation. TestSetβ-40 and project bpRNA-new-5,388 must not be merged or described as the same evaluation protocol.

## GPU launcher boundary

`tools/run_debfold_bprna_new_gpu_20261126.py` defaults to CPU-only preflight and currently exits with status 2 because no verified official adapter exists. A future adapter must identify the same official commit, hash every implementation/weights file under the artifact root, and provide an argument-list command template with input/output/implementation/weights placeholders. Shell command strings are forbidden. Actual execution additionally requires explicit `--execute`; predictions cannot be overwritten and are accepted only after full 5,388-row contract validation.

No official adapter manifest was created in this delivery because the necessary official assets and CLI are absent. Consequently this branch is **data/protocol/tooling-ready but not model-ready, prediction-ready, or unified-SOTA-score-ready**.

## Verification

Three rounds are logged under the artifact root `reports/`:

1. Download/source verification: fixed commit and tree, empty releases/tags, license 404, website/GitHub SHA equality, ZIP CRC, complete CT parsing and counts, and full paper XML.
2. Environment/static verification: Python byte-compilation and default fail-closed launcher preflight; no GPU call.
3. CPU unit tests: CT parsing, pair reciprocity/alphabet, strict dataset identity, complete per-sequence output, pair validation, and official-asset path/fail-closed boundaries.

`reports/delivery_verification_20261126.json` records final file hashes and the explicit `full_ready: false` result. No large asset is tracked by Git. The isolated unified `manifests/foldnar_v1/baseline_registry.json` DEBFold row and its generator now carry the official training-validation archive identity and SHA. The row still has no prediction IDs, no weight SHA, no model license, and blocked provenance; no metric or coverage was changed.
