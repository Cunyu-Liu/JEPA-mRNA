# R3 baseline gate: status and the learning-rate finding

The gate requires three representative tasks (CDS mRFP regression, 3'UTR-RBP
classification, ultra-long TE) to reproduce the published values within 0.05, using the
protocol spreadsheet's hyper-parameters. This file records where that stands and, more
importantly, what had to be changed to get there.

## Gate tasks

| task | metric | ours | paper | delta | status |
|---|---|---|---|---|---|
| cds_mrfp | Spearman | 0.8663 | 0.89 | -0.024 | PASS at the official lr 1e-4 |
| rbp_0 | accuracy | 0.8250 | 0.786 | +0.039 | PASS, but only after calibrating lr |
| te_human | R2 | pending | 0.669 | - | running under a shortened protocol (see below) |

## The finding: the documented default learning rate is unsafe across task kinds

Running the official default (lr 1e-4) produced two *degenerate* results, and each one
cost a full investigation because a collapsed run looks like a modelling failure rather
than a hyper-parameter failure.

3UTR-RBP: at lr 1e-4 the model reaches accuracy 0.6675, f1_positive 0.0, MCC 0.0 — exactly
the majority-class rate — at 1, 5 and 10 epochs, and identically in fp32. Train loss falls
to 0.549 by step 60 and then *rises* back to 0.64 and stays there. A frozen encoder
reaches 0.7796 on the same split, so the features are fine; the encoder is being pushed out
of its useful regime in the first ~100 steps. Controls: a 3-mer logistic regression reaches
0.781, so the task is learnable and the pipeline was at fault.

protein_solubility: a **regression** task, which was assumed to be safe, gives Spearman
-0.0194 at lr 1e-4 versus 0.6488 at 5e-5. cds_fungal gives -0.0041 against a published
0.89. So the failure is not specific to classification, nor to large label counts.

## Calibration

A dev-set sweep over lr {1e-4, 5e-5, 1e-5} on the **reference model only**; the chosen
value is then applied to every model in the study. Tuning each model separately would let
the baseline and the candidate run different hyper-parameters, which would make the
head-to-head meaningless.

| task | 1e-4 | 5e-5 | 1e-5 | chosen | paper |
|---|---|---|---|---|---|
| cds_cov | 0.5382 | 0.8335 | 0.7869 | 5e-5 | 0.89 |
| cds_riboswitch_1 | 0.3980 | 0.4193 | 0.5050 | 1e-5 | 0.58 |
| full_in_cell_half_life | 0.4677 | 0.6688 | 0.4976 | 5e-5 | (R2 0.769) |
| protein_solubility | -0.0194 | 0.6488 | 0.6289 | 5e-5 | (R2 0.63) |
| rbp_0 | 0.6675 | - | - | 2e-5 | 0.786 |

Each chosen value is stored in `configs/tasks.yaml` with `lr_status: verified` and an
`lr_evidence` string naming the numbers, so the protocol is traceable to measurements
rather than to preference. Five further tasks still have `lr_status: selected` and are
queued for their own sweeps.

## Other protocol deviations, recorded rather than silent

- **ultra-long TE**: at the published 3066-token budget with batch 1 and 20 epochs the run
  is 178,440 optimizer steps and OOMs on the quadratic attention (450 MiB of scores per
  layer per sample). Reduced to batch 2 x 3 epochs, applied identically to every model;
  the delta stays valid, the absolute value is not comparable to the paper's 20-epoch one.
- **full-length family**: the whole family is ~233 sequences and the paper used
  cross-validation, so a 21-sequence holdout is not comparable; marked
  `not_comparable_to_paper` in the registry. The model-vs-model delta is still valid.
- **shared dev/test**: 23 tasks ship byte-identical dev and test files — the 15 previously
  documented (`full_*`, `utr5_*`) plus all 8 Spliceator tasks. Their test scores are
  development scores and are flagged in the table builder.

## Reproduction status of the wider subset

| task | ours | paper | note |
|---|---|---|---|
| cds_mrfp | 0.8663 | 0.89 | pass at the default lr |
| m6a_HEK293T_fold0 | 0.9620 | 0.966 | pass at lr 2e-5 |
| cds_ecoli | 0.5730 | 0.58 | pass |
| rbp_0 | 0.8250 | 0.786 | pass after calibration |
| rbp_1 | 0.8125 | 0.786 | pass |
| cds_cov | 0.8335 | 0.89 | after calibration; delta 0.057, just outside tolerance |
| cds_riboswitch_1 | 0.5050 | 0.58 | after calibration; delta 0.075, outside tolerance |
| cds_fungal | -0.0041 | 0.89 | collapse at lr 1e-4; sweep queued |
| protein_solubility | 0.6488 | (R2 0.63) | collapse at lr 1e-4; recovered |

Two tasks remain outside the 0.05 tolerance after calibration. Neither is used to support
a claim about absolute performance; both are kept in the head-to-head because the
model-vs-model comparison is still valid on the identical protocol. That distinction is
stated in the task registry and will be stated in the paper's protocol section.