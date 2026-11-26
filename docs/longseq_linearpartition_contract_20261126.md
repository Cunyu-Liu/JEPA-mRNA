# LinearPartition long-sequence acceptance contract (2026-11-26)

## Scope and verdict

Branch `longseq-linearpartition-contract-20261126` starts at FoldNAR integration HEAD
`81601249aefa2ea1ae343d5e9d2f6d382cf8455b`. This delivery performs CPU-only source
recovery, compilation, an official short smoke, input identity checks, and static
acceptance tests. Its initial static phase did not run 7224/19000 nt inference.
A later train-only 7224 nt resource pilot is documented in
`docs/linearpartition_7224_cpu_pilot_20261126.md`; the 19000 nt holdout remains unexecuted.
No phase creates or modifies a GPU job.

**LinearPartition is executable and short-smoke ready, but not long-sequence
full-ready.** Two train-only 7224 nt runs now have measured coverage, recall, memory and
latency; 19000 nt coverage/recall/resources and all long-chain FLOPs remain `NOT_MEASURED`. No exact global-DP implementation has qualified for these targets.
GEC support at 7224/19000 nt, exact cubic or otherwise, is not established or claimed.
Summary tables, logs, and the short smoke are not long-chain predictions.

## Frozen input identities

The executable freezer reads the D1 JSONL files registered by
`source_data/round53/rinalmo_full_d1_manifest.json`, verifies their full-file SHA256,
then verifies each original chain against its source `labels.json`. Sequence processing
is uppercase plus `T -> U` only. Reference `pairs` are zero-based inclusive; raw DSSR
pairs remain one-based inclusive. Full hashes and source file records are in
`manifests/foldnar_v1/longseq_linearpartition_targets.json`.

| Chain | PDB | Split / scientific role | Row (0-based) | Length | Sequence SHA256 | Canonical pairs | Raw DSSR pairs |
|---|---|---|---:|---:|---|---:|---:|
| 9MR4_5 | 9MR4 | train; not unbiased evaluation | 42894 | 7224 | `b549ef087290a901736193d7c0890b915620a194008b6cffd8111c4bdb4d6403` | 964 | 1583 |
| 9Q7Q_5 | 9Q7Q | train; not unbiased evaluation | 42911 | 7224 | `0eaa85f0f67b900a85b21850f31e672f9fd9d2ba21931cc6640ec3165e272c2d` | 934 | 1558 |
| 7AM2_1 | 7AM2 | holdout evaluation | 57 | 19000 | `6d21b52178a592b01c000d0e800d486cb540f7e3d869a0313326e244faa95a23` | 73 | 166 |

The two 7224 nt chains may be used for execution/coverage tests or explicitly labelled
training-internal analysis. They must not be presented as held-out accuracy evidence.

## Official implementation and provenance

- Repository: `https://github.com/LinearFold/LinearPartition.git`
- Pinned commit: `b450fb3e63189073b68d385589035f992080aa3a` (official master selection;
  tag `v1.0` is not silently substituted).
- Recovered source: `/teams/HoulabData_1788502323/liucunyu/foldnar/longseq_linearpartition_contract_20261126/upstream_linearpartition/`
- License: custom license, not assigned an SPDX identifier. Use/modification conditions,
  proper credit, fee-redistribution restriction, and commercial-product contact clause
  are recorded in the recovery report; the license itself is hashed.
- C uses embedded `feature_weight.h`; V uses embedded Vienna `energy_parameter.h`.
  Neither is described as parameter-free. No external model checkpoint is required.

`tools/recover_linearpartition_longseq.py` checks the origin, commit and tracked-source
cleanliness, builds both official binaries with `make -j2`, and runs the official
`ecoli_tRNA` input. The actual source sequence is **76 nt**, SHA256
`2af42dfab28438539d9cfeb851d53541bc11fdbc2ee87e1b5d97db4f5d1bdf5d`.
Both C and V passed with complete raw BPP and one balanced 76 nt MEA structure:
C emitted 767 pair rows and V emitted 183. These counts are short-smoke facts only.
Source, binaries, input, outputs, logs, commit/tree and license hashes are in
`recovery_v1/recovery_report.json` under the evidence root.

## Frozen execution columns

| Column | Admission rule | Current implementation |
|---|---|---|
| `EXACT_GLOBAL_DP` | `exact_global_dp=true`, BPP cutoff 0, and no beam, window, top-k or sparse pruning; requires implementation-specific evidence | None qualified; no long-sequence result |
| `APPROXIMATE_SPARSE` | Any non-exact/pruned execution; retain every emitted pair at the frozen cutoff and never interpret absent pairs as exact zeros | LinearPartition-C and V, beam 100, cutoff 0.0 |

C and V are separate protocols and must not be pooled. Exact and approximate results
must remain in separate tables/files. Beam 0 is not automatically relabelled exact
without a separate algorithmic and output-equivalence proof.

The future full-length command template (not executed in this delivery) is:

```bash
/root/miniconda3/envs/codonflow/bin/python linearpartition --fasta \
  -b 100 -c 0.0 -M --rewrite FULL_PATH.bpp < FULL_PATH.fasta
# Add -V for the separately registered LinearPartition-V run.
```

## Required raw output and coverage

Every expected chain appears exactly once as `COMPLETED` or `FAILED_EXPLICIT` with a
reason. A failed chain is not omitted and is not scored as zero. A completed record must
contain:

1. Matching input length and sequence SHA256; output length equals the untruncated input.
2. The complete in-memory emitted pair list and its raw `bpp_1based_tsv` artifact,
   including path, full-content SHA256, row count, format, and frozen cutoff/protocol.
   Every row is parsed and compared with the in-memory list. Mismatch, duplication,
   malformed probabilities, or coordinates outside the full sequence fail validation.
3. Full-length structure output when a structure is requested. BPP candidates may cross
   or share partners; structure-specific validity is a separate concern.
4. Measured resource fields described below. `summary_only=true`, estimated resources,
   missing pair artifacts, or metrics-only rows are rejected.

Coverage is reported separately as record coverage, completed-target coverage, and
completed-nucleotide coverage. Record coverage of 1.0 does not hide explicit failures.

## Pair and long-range recall

For reference pair `(i,j)`, span is `j-i` on zero-based inclusive coordinates. The
validator reports overall pair recall and micro recall in `0-49`, `50-199`, `200-999`,
and `>=1000` buckets, plus per-target values. Empty buckets are null, not zero. Micro
counts preserve target identity so identical coordinates on different chains do not
collapse. Candidate recall uses every emitted pair at the preregistered cutoff; no
post-hoc threshold is allowed. These are contract definitions, not measured long-chain
results in this delivery.

Run validation without invoking a model:

```bash
/root/miniconda3/envs/codonflow/bin/python tools/validate_longseq_predictions.py \
  --repo /home/rna_jev_longseq_contract \
  --predictions FULL_RAW_RECORDS.json --protocol FROZEN_PROTOCOL.json \
  --output VALIDATION_REPORT.json
```

## Memory, latency, and FLOPs

Each completed full-length target requires positive measured wall time and peak CPU RSS
(`/usr/bin/time -v` or equivalent). Actual CUDA runs additionally require measured peak
allocated and reserved GPU memory; CPU runs use null for those inapplicable fields, not
zero. FLOPs must be a non-negative measured value in `FLOPs`, with method recorded and
`estimated=false`, using hardware performance counters or instrumented operation counts
for that exact execution. Short-chain scaling formulas are not accepted.

The initial static report contained no long-chain resources. The subsequent two-chain
7224 nt train-only pilot measured memory and latency with GNU time; FLOPs could not be
measured because hardware counter access was denied. See the pilot document for observed
values and the separate 19000 nt authorization gate. No GPU task is needed for
LinearPartition. Any GEC long-sequence experiment requires a separate support proof,
resource review, and explicit authorization.

## Evidence and reproduction

Evidence root (outside Git):
`/teams/HoulabData_1788502323/liucunyu/foldnar/longseq_linearpartition_contract_20261126/`

- `recovery_v1/`: build log, official short input, raw C/V BPP, stdout/stderr, provenance.
- `static_acceptance_v1.json`: target identities and explicit `full_ready=false` matrix.
- `delivery_verification.json`: final tests, commits, branch and large-file audit.

Git contains only source, tests, documentation and the small identity manifest; binaries,
raw predictions, JSONL assets and logs remain outside Git.
