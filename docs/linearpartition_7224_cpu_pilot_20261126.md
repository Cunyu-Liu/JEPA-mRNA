# LinearPartition-C 7224 nt serial-CPU resource pilot

## Scope

This run executes the two frozen 7224 nt **train** chains, `9MR4_5` followed by
`9Q7Q_5`, with the official LinearPartition-C binary at commit
`b450fb3e63189073b68d385589035f992080aa3a`, beam 100, BPP cutoff 0.0 and one
CPU process. The column is `APPROXIMATE_SPARSE`. All results are **train-only execution
and descriptive recall**, not independent scientific evaluation. The 19000 nt holdout
was not executed.

Evidence root:
`/teams/HoulabData_1788502323/liucunyu/foldnar/longseq_linearpartition_contract_20261126/runs/linearpartition_c_7224_train_cpu_20261126/`

Each chain directory preserves the full input FASTA, raw BPP, standalone and stdout MEA,
stderr, GNU time report, run report, metrics and independent verification. `summary.json`
re-runs the independent verifier for both chains and pools train-only counts.

## Measured execution and coverage

| Chain | Wall seconds | Peak CPU RSS bytes | BPP rows | MEA pairs | Nucleotide coverage |
|---|---:|---:|---:|---:|---:|
| 9MR4_5 | 32.78 | 1,679,233,024 | 112,772 | 2,349 | 7224/7224 = 1.0 |
| 9Q7Q_5 | 31.78 | 1,683,169,280 | 110,655 | 2,377 | 7224/7224 = 1.0 |

GNU time 1.7 came from Ubuntu `time 1.7-25.1build1` package SHA256
`c46db7f84e8e92004ebb437e36560e3bfe656ff79d4d1a359a8a193a6c2521c2`.
It was extracted under the teams tools directory, not installed into the system.
Latency and maximum RSS are measurements from the raw `gnu_time_v.txt` files. GPU memory
fields are null because these were CPU executions.

## Train-only pair and span recall

Span is `j-i`; candidate BPP recall and final MEA recall are separate output levels.
The pooled values below combine reference-pair counts from two train chains and remain
descriptive, not independent evaluation.

| Output | Overall | 0-49 | 50-199 | 200-999 | >=1000 |
|---|---:|---:|---:|---:|---:|
| BPP candidate | 1798/1898 = 0.947313 | 923/927 = 0.995685 | 633/668 = 0.947605 | 234/295 = 0.793220 | 8/8 = 1.0 |
| MEA | 975/1898 = 0.513699 | 471/927 = 0.508091 | 392/668 = 0.586826 | 112/295 = 0.379661 | 0/8 = 0.0 |

Per-chain values and all output/hash records are in each `metrics.json` and
`independent_verification.json`. The independent implementation reparses raw BPP/MEA,
recomputes sequence and artifact hashes, checks GNU time values and recomputes every
coverage/span value without using the runner in-memory pair list.

## FLOPs measurement limitation

FLOPs are `NOT_MEASURED`, value null, and are not estimated. The host has no `perf` CLI;
`kernel.perf_event_paranoid=4`; a direct `perf_event_open` hardware-instructions probe
returned `Operation not permitted`. The probe log is hashed into both run reports. No
formula, short-chain scaling or latency substitution is presented as FLOPs.

## 19000 nt resource decision

The observed maxima are 32.78 seconds and 1,683,169,280 RSS bytes. At the 19000/7224
length ratio, simple linear planning gives about 86.22 seconds and 4,426,940,244 bytes.
These are **scheduling heuristics, not 19000 nt measurements**. The resource decision is:
a separately authorized serial-CPU holdout run is feasible with 8 GiB available RAM and
an initial 600 second timeout, preserving the same raw-output and independent-verification
protocol. No 19000 nt execution or scientific result is claimed here.

No GPU job was launched or modified by this pilot. A later separately authorized 19000 nt
holdout execution is documented in `docs/linearpartition_19000_holdout_cpu_pilot_20261126.md`.
