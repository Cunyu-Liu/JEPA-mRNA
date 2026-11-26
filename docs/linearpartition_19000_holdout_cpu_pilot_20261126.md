# LinearPartition-C 19000 nt holdout serial-CPU pilot

## Scope and execution

`7AM2_1` was executed once as a full-length 19000 nt **holdout descriptive** run with the
official LinearPartition-C binary at commit
`b450fb3e63189073b68d385589035f992080aa3a`, beam 100, BPP cutoff 0.0, one CPU process,
and a 600 second hard timeout. The execution column is `APPROXIMATE_SPARSE`. These results
are not used for model selection, calibration, early stopping or training.

Evidence directory:
`/teams/HoulabData_1788502323/liucunyu/foldnar/longseq_linearpartition_contract_20261126/runs/linearpartition_c_19000_holdout_cpu_20261126/7AM2_1/`

The resource gate observed `MemAvailable: 55936508 kB`, above the required 8 GiB. GNU time
was outside the timeout wrapper, so both successful and timeout paths have a resource
report. The run completed normally with exit status 0.

## Measured resources and raw outputs

- Wall time: **362.79 seconds**; user 351.76 seconds; system 6.00 seconds.
- Peak CPU RSS: **8,481,693,696 bytes** (about 7.90 GiB).
- Coverage: **19000/19000 = 1.0**, no truncation.
- BPP: 785,106 rows, 14,744,472 bytes, SHA256
  `6322acd284ee2216498441ac1ca14b31251696c13f9a6c8dd4e87a6cdd7745b4`.
- MEA: 2,614 pairs, 19000 nt, SHA256
  `5ca1b42bbaa4b938eef9b6d03d1a3eb98bb331137e3200a63482ccffd5d515ab`.
- GNU time SHA256:
  `11d20fa31262ea928eb7b5a59794c7fb775139c7c5c87b08b447508943dbc95a`.

Actual peak RSS was about five times the 7224 nt maximum, not the 2.63 times linear
planning heuristic. The 8 GiB planning allowance had little margin; future comparable runs
should reserve at least 10-12 GiB rather than treating the linear estimate as a bound.

FLOPs remain `NOT_MEASURED`, value null, with no estimate. The hashed capability probe
records the host denial of hardware counter access. GPU fields are null because this was a
CPU execution.

## Independently recomputed holdout descriptive metrics

The independent verifier reparsed raw BPP and MEA, checked all artifact hashes and GNU time,
and recomputed coverage and span recall. Span is `j-i`; an empty bucket is null, not zero.

| Output | Overall | 0-49 | 50-199 | 200-999 | >=1000 |
|---|---:|---:|---:|---:|---:|
| BPP candidate | 53/73 = 0.726027 | 41/42 = 0.976190 | 12/18 = 0.666667 | 0/13 = 0.0 | null, 0 reference |
| MEA | 8/73 = 0.109589 | 6/42 = 0.142857 | 2/18 = 0.111111 | 0/13 = 0.0 | null, 0 reference |

`independent_verification.json` records the complete recomputed counts and verified file
identities. These values describe this holdout chain only and do not select or rank models.
No GPU job was launched or modified.
