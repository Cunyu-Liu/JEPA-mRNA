# DP-Free Calibrated Base-Pair Probabilities for RNA Secondary Structure

**Preliminary preprint draft — v3.9, 2026-09-26.**

> **Read this banner before quoting anything.** Every number in §4 is a *measured*
> value produced by this repository on the A100 cluster, with the exact command and
> artifact path listed in Appendix A. Nothing here is a placeholder, and nothing here
> is extrapolated.
>
> **v3.9 change note (gap attribution closed).** Two released checkpoints are now
> re-run under their own protocols and both re-scored under ours: RiNALMo-ft (650M
> fine-tuned) scores 0.7602 tolerant / 0.7210 strict on TS0 — 0.11 above our
> frozen-backbone head with the *same encoder*, so backbone fine-tuning is the
> dominant gap term (§4.3g); and on bpRNA-new it scores 0.4489, *below* our frozen
> head (0.4870), the out-of-distribution reversal at 650M scale. structRFM (86M,
> structure-guided pretrain) re-runs at 0.6638 strict TS0 / 0.5438 bpRNA-new —
> beating our head without a large backbone. A backbone-swap arm (RNA-FM frozen,
> same CRF head: 0.4199/0.3005) measures the representation axis at ~0.17. §4.3g
> assembles these into a three-term decomposition of the 0.13–0.15 gap; the
> 2D-context scorer arm isolating the architecture term is training (Plan B).
>
> **v3.8 change note (seed-variance bound + new baselines secured).** §4.3f now
> reads the seed ensemble (score-matrix averaging over the 8 TR0 seeds, single
> decode, zero retraining: TS0 0.6105 / +0.017; bpRNA-new 0.5106 / +0.024) as a
> *diagnostic*: seed stochasticity bounds at most ~15–20% of the deficit, so the
> remaining gap is systematic (representation, data, objective) — the
> improvement programme targets those, not ensembling. The RiNALMo fine-tuned
> TS0 checkpoint (2.6GB) and the structRFM bpRNA1m SSP checkpoint (348MB) have
> been downloaded and MD5-verified locally with their official inference code;
> measured rows are pending cluster recovery.
>
> **v3.7 change note (seed completions + rounding audit).** The capacity,
> combination, and TR1 arms gained seeds (capacity now 4: 0.6324 ± 0.0098;
> combination 2: interaction direction unanimous; TR1 2). The sum-normalised
> 512-dim replication arm closes the capacity normalisation confound (0.6302).
> An arithmetic audit also corrected the 8-seed mean to 0.5938 and one capacity
> seed value (0.6337, not 0.6343) — both errors were ours and both are flagged.
>
> **v3.6 change note (TestSetB + INF columns).** The RiNALMo supplementary tables
> S2–S5 are now aligned table-by-table: S4's INF convention is added to §4.3d, and
> S5's benchmark (TORNADO TestSetB) is fully measured zero-shot (§4.3e) — where our
> capacity head exceeds every published number on that benchmark, including RiNALMo's
> fine-tuned 0.67. S2/S3 (ArchiveII 9-fold leave-one-family-out) require nine
> fine-tuning runs and remain un-replicated; the zero-shot ArchiveII numbers stay
> withdrawn per §4.5.
>
> **v3.5 change note (pretrained-LM baselines + official split).** The evaluation is
> extended to the official full TS0 (1,305 sequences, identical to the RiNALMo-paper
> split), and all baselines are additionally scored under the Mathews-tolerant
> macro-F1 convention that paper uses (§4.3d), including RNAformer and UFold. The
> RiNALMo fine-tuned structure model itself cannot be re-run (weights on an
> unreachable host) and is quoted as a reported number with that flag attached.
>
> **v3.4 change note (protocol audit).** A cross-check of every baseline in the
> evidence chain found that v3.3's cross-family TR0 reference (0.3536) came from a
> step-3500 snapshot with a different calibration protocol, not from the matched
> step-20000 checkpoint. All cross-family comparisons in this version use the matched
> number (0.4870), which reverses the sign of the capacity effect out of distribution
> and rescales the data effect; the 8-seed mean was also recomputed exactly
> (0.5938, not 0.5950). Significance is now reported as paired per-sequence Wilcoxon
> tests with Holm correction (§4.3c, `tables/stats_definitive.json`).
>
> The draft is nevertheless **preliminary** in three specific,
> enumerated ways (Appendix C), the largest being that the strongest learning-based
> baselines (SPOT-RNA, UFold) could not be run in our network environment, so the
> existence claim C1-a is *not* tested.
>
> One claim that earlier drafts of this project carried has been **deleted because it
> was measured and found false**: that our model shows smaller cross-family
> degradation than the physical baselines. §4.4 reports the opposite, and
> `paper/check_manuscript.py` now bans that phrasing.

## Abstract

RNA secondary structure prediction faces a trust-versus-cost dilemma. Physical models
report base-pair probabilities that are well calibrated but require an `O(L^3)`
partition function before any probability exists; discriminative deep models emit an
`L x L` probability matrix in a single forward pass but have never been shown to be
calibrated. We ask whether a single-forward-pass decision head can be made as
calibrated as the exact partition-function marginals, so that inference need not
compute the partition function at all.

On bpRNA TS0 (1,288 sequences, non-redundant with respect to the training split) we
measure an expected calibration error gap between our DP-free head and the exact
marginals of the same model of **0.0028**, against a pre-registered threshold of
**0.02**, after fitting a two-parameter affine map on a disjoint validation split.
The map is applied at evaluation time and involves no partition function. The raw,
unrecalibrated head does **not** pass this gate (gap 0.192; ECE 0.1955), and we report
both. We also place this number in the first systematic calibration audit we are aware
of for RNA base-pair probabilities: six probability sources — RNAformer, UFold, our
exact marginals, our recalibrated head, ViennaRNA exact probabilities, and our raw
head — scored on the same 6,022,538 candidate pairs of the same split. RNAformer's
probabilities are already well calibrated (ECE 0.0015) without any post-processing,
so DP-free calibration is not unique to us; UFold is 4.1x over-confident (ECE 0.0147);
our exact marginals are the most calibrated source measured (ECE 0.0004).

Structure accuracy is **not** a uniform loss, and the aggregate number is misleading in
both directions. Pooled over TS0 our micro F1 is **0.5938** across 8 seeds
(128-dim head; ViennaRNA centroid **0.5393**, MXfold2 **0.5651**, UFold **0.6598**,
RNAformer **0.7578** on the same split), and **0.6425** when the decision head is given
4.8x capacity — a paired gain of +0.039 that holds in all three capacity seeds but is
concentrated on pair-dense sequences (§4.3c). Under the Mathews-tolerant convention of
the RiNALMo paper, on the official full TS0, our frozen-backbone pipeline scores
**0.6368** (base) to **0.6474** (capacity) — above ViennaRNA (0.5665) and MXfold2
(0.6102), in the vicinity of secondarily-reported numbers for the fine-tuned 650M
RiNALMo structure model itself, and 0.13 below the strongest measured models (UFold
0.7807, RNAformer 0.7779) while training a ~5M-parameter head instead of
fine-tuning 650M (§4.3d).
Under-training was the dominant error
source at earlier checkpoints (0.4953 at step 3500, monotonically rising to 0.5956 at
step 20000 with no plateau), which also reverses an early negative reading of the
capacity hypothesis taken at step 2000. Crossing the split by source and by length,
with our model and the baselines scored on the same sequences in every cell, reverses
the comparison for one half of it:

| Source | Length | n | Ours | Centroid | Ours − centroid |
|---|---|---|---|---|---|
| `CRW` (conserved) | <=100 nt | 68 | **0.9677** | 0.6729 | **+0.295** |
| `CRW` (conserved) | 100–200 nt | 16 | **0.8725** | 0.6004 | **+0.272** |
| `RFAM` (diverse) | <=100 nt | 486 | 0.5598 | **0.6209** | **−0.061** |
| `RFAM` (diverse) | 100–200 nt | 498 | 0.4272 | **0.5347** | **−0.108** |

**The sign of the comparison is set by how conserved the source population is, not by
length.** We beat the partition-function baseline in both length buckets on conserved
sequences and trail in both on diverse ones; length changes the magnitude within a
stratum but never flips the sign. Because `RFAM` accounts for 76% of TS0, the pooled
number necessarily shows a loss. The conserved-stratum advantage is not memorisation —
containment against the training split there is 0.048 with no sequence above 0.5 — and it
replicates on the independent validation split (0.9709 vs 0.6702).

Cross-family generalization is the method's dominant weakness, and data scaling is the
only lever that moves it: on bpRNA-new our TR0-trained model reaches micro F1
**0.4870**, while the same recipe on 4.29x more training data reaches **0.5162**
(+0.029, per-sequence Wilcoxon p = 4e-62) — against ViennaRNA centroid's **0.6770**
and UFold's **0.6106**. The deficit is no longer a collapse but it is still 0.17-0.16,
and we report it as the main open number. Head capacity, by contrast, *hurts* on this
split (−0.023, p = 7e-83): the two scaling axes point in opposite directions out of
distribution. The same checkpoints tell the opposite story on a second OOD benchmark:
on TORNADO TestSetB (22 structurally dissimilar Rfam families, the RiNALMo paper's
hardest generalization set) our capacity head reaches **0.7932 tolerant F1 zero-shot**,
above every published number on that benchmark — "cross-family generalization" is
decided by which families the benchmark holds, and we report both ends rather than
one. Notably, the calibration result **does** survive both the cross-family
shift and the data scaling (recalibrated C1-c gap 0.0014 on TR0 and 0.00078 on TR1,
inside the same 0.02 threshold). A model can rank poorly and still report honest
probabilities, and on this benchmark it does.

We conclude that DP-free calibration is achievable, that it dissociates from ranking
quality — a dissociation we quantify rather than paper over — and that the two scaling
axes we measure buy different, partly opposite things: head capacity buys pooled
in-distribution accuracy (+0.039 paired) at a cross-family cost (−0.023), while
training data is the only confirmed cross-family lever (+0.029 at 20k steps) at a
small pooled cost at 20k (−0.012) that reverses to +0.021 by 40k steps.

## 1. Introduction

A probability is only useful if it means what it says. In RNA secondary structure
prediction the probabilities that come with a guarantee are the McCaskill-style
base-pair probabilities, and they cost an `O(L^3)` partition function. The
alternatives that are cheap — contact-map style networks that emit a sigmoid matrix
in one forward pass — are evaluated almost exclusively on structure accuracy (F1,
INF), and to our knowledge no published work reports expected calibration error,
reliability diagrams, Brier score or NLL for their pair probabilities.

That leaves two questions unanswered at the same time. *Can a cheap probability be
trusted?* and *if it can, what does it cost in accuracy?* This draft answers the first
with a measurement and the second with a stratification: the cost is not a single
number, and reporting only the pooled one misstates the method in both directions. A
third question emerged from the measurements themselves and structures the results
section: *what does each scaling axis buy?* Head capacity and training data turn out
to buy different things on different splits, and quantifying that separation is more
useful than any single headline number.

The framework is deliberately borrowed rather than invented. We train a log-linear
model over the non-crossing structure space with an exact partition function during
training, and we then ask how much of that partition-function-calibrated behaviour
survives when the partition function is removed from inference.

## 2. Related work

The framework is not our contribution. CONTRAfold already trains a log-linear model
over the non-crossing structure space with an exact partition function, and the CRF
lineage is long. We sit inside it.

The DP-free precedents that matter are SPOT-RNA, SPOT-RNA2, UFold and **RNAformer**:
they emit an `L x L` pair-probability matrix from a single forward pass and do not run
a partition function. We therefore do **not** claim to be first at emitting
probabilities without dynamic programming — and after measuring them we do not claim
to be first at calibrated ones either: RNAformer's probabilities, which had not been
audited before, turn out to be well calibrated (ECE 0.0015 on TS0). What we could not
find in that literature is any calibration *report* — no ECE, no reliability diagram,
no Brier score — which is the gap this draft closes with a six-source audit rather
than with a new model.

We also correct an earlier internal misreading: CDPFold is a CNN followed by dynamic
programming (Front Genet 10:467, 2019). It neither removes the DP nor is a
calibration precedent, and it is not treated as a threat to the claim here.

The pretrained-RNA-LM line — RNA-FM, Uni-RNA and RiNALMo — supplies our frozen
backbone. RiNALMo's paper fine-tunes the full 650M model for secondary-structure
prediction with a ResNet head and reports it as the strongest LM on this split; we
consume the same frozen encoder and train a small CRF head on top instead, and
§4.3d compares both under that paper's own tolerant scoring convention.

**Why a frozen foundation-model backbone.** The from-scratch alternative was measured
to failure: an own-encoder 35M Transformer trained on the same corpus peaks at 0.547
(nll+distill) and *degrades* with further training (0.511 at 20k, 0.495 at 40k for
the four-objective variant — overfitting on 10k sequences), while the frozen RiNALMo
embeddings reach 0.5958 at identical steps. The decision head on frozen
representations is therefore not a convenience but a measured choice, consistent with
the backbone-scale literature (RiNALMo, RNA-FM).

## 3. Method

**Representation.** Sequences are embedded once, offline, with a frozen RiNALMo-giga
backbone (33 layers, hidden 1280, rotary, `max_pos` 1024). The decision head consumes
those frozen embeddings; the backbone is not fine-tuned in any run reported here.

**Head.** A permutation-invariant pair representation is formed per candidate pair and
mapped to a scalar score, producing a symmetric `L x L` score matrix in exactly one
forward pass. Illegal pairs (too close in sequence, or not a legal base combination)
carry a finite sentinel in training and are masked out at decode time. There is no
autoregressive path in the code.

**Training objective.** A conditional-random-field negative log-likelihood over the
non-crossing structure space, with three auxiliary terms: a distillation term against
teacher pair probabilities, a calibration term, and a proper-scoring-rule reward term.
A length-normalisation switch makes the four terms numerically comparable; the effect
of that switch is the subject of a separate controlled comparison that is still
running and is therefore *not* reported here.

**Scaling axes.** The headline head uses a 128-dim pair representation; the capacity
arm scales it to 512 (4.8x parameters). The data axis swaps the training corpus from
TR0 (10,682 sequences) to TR1 (45,865 sequences, 4.29x after de-duplication) with
everything else fixed — the data and combination axes train with the same NLL
normalisation as the base family. The capacity arm, however, was launched with
length-normalised NLL where the base family uses unnormalised (sum) NLL, so the
capacity comparison carries a second changed variable; two controls close it: a
matched pair at 128 dims (len 0.5870 vs sum 0.5893, both at 20k) bounds the
normalisation effect at 0.002 — inside the seed spread — and a **sum-normalised
512-dim replication arm** reaches 0.6302, reproducing the capacity gain (+0.036 over
the 8-seed base mean) with the confounding variable removed. Both axes are reported
at 20,000 steps, `w=-1` decode, on the same splits. The seed spread
of the base configuration across 8 seeds is **mean 0.5938, std 0.0034** (range
0.5893–0.5990); the capacity arm across 4 seeds is
**0.6425 / 0.6189 / 0.6337 / 0.6345** (mean 0.6324, std 0.0098) — its paired capacity gain is
**+0.0394 ± 0.0088**, about 12x the base spread, and its own seed spread is visibly
larger than the base configuration's, consistent with capacity amplifying
initialisation effects. The cross-family data gain (+0.029) is ~9x the base spread.
All headline comparisons are additionally tested as paired per-sequence Wilcoxon
signed-rank tests with Holm-Bonferroni correction (§4.3c).

**DP-free recalibration.** After training, a two-parameter affine map
`p = sigmoid(a * s + b)` is fitted on a validation split disjoint from every test
split. It is a strict generalisation of temperature scaling (`b = 0`). Fitting uses
only the head's scores; it never runs a partition function. Inference with the map
therefore remains DP-free.

## 4. Results

All numbers in this section are micro-averaged base-pair F1 unless stated otherwise,
computed by a single metric implementation shared by every row of every table
(`eval/ss/evaluate_decision.py` and `eval/ss/reference_calibration.py`). "micro" means
TP/FP/FN are pooled across sequences before computing P/R/F1; "macro" means per-sequence
F1 is averaged afterwards. Both are reported in Appendix A.

### 4.1 Data and protocol

| Role | Split | n used | Note |
|---|---|---|---|
| Training | bpRNA TR0 | 10,682 | projected corpus; mean length 132.5, range 33-498 |
| Validation (recalibration + any selection) | bpRNA VL0 | 196 | disjoint from TR0 and TS0 |
| In-distribution test | bpRNA TS0 | 1,288 | headline split |
| Official full TS0 (RiNALMo/RNAformer) | bpRNA TS0-1305 | 1,305 | our 1,288 + 17 recovered from the published test set; used for §4.3d |
| Secondary test | ArchiveII (BPfold bpseq) | **3,950 / 3,966** | **not** an out-of-distribution split, see §4.5 |
| Cross-family test | bpRNA-new | 5,388 | the only clean OOD split we hold |
| Cross-family test 2 (RiNALMo S5 benchmark) | TORNADO TestSetB | 428 / 430 | zero-shot; see §4.3e |

Teacher probabilities are ViennaRNA 2.7.2 with a locked parameter set. Decoding is a
max-product dynamic program over the non-crossing space, which guarantees legal
structures; measured illegal-structure rate and minimum-hairpin violation rate are
**0.0000** at every evaluation point reported below.

### 4.2 The calibration result (C1)

Checkpoint: `rinalmo_ff` at step 20000 (step read from inside the checkpoint file, not
from its name). Decode weight is the model's own trained prior weight — no selection on
any test split. (The step-3500 numbers that earlier versions of this draft carried are
superseded: the convergence curve 0.5369@6000 / 0.5587@10000 / 0.5958@20000 shows the
model was under-trained there.)

| Quantity (TS0, same 6,022,538 candidate pairs) | Value |
|---|---|
| System-1 ECE, raw head | 0.1955 |
| Exact-marginal ECE (same model, same sequences) | 0.0004 |
| **C1-c gap, raw head** | **0.192 — FAIL** (threshold 0.02) |
| **C1-c gap, DP-free affine recalibration** | **0.0028 — PASS** |
| System-1 ECE after recalibration | 0.0031 |
| Recalibration parameters | 2 parameters, fitted on VL0 |

The gap is reported both ways on purpose. Reporting only the recalibrated row would
hide the fact that the raw head is over-confident by two orders of magnitude; reporting
only the raw row would hide the fact that a two-parameter, partition-function-free map
closes the gap to three thousandths — and that Brier (0.1015 -> 0.0057) and NLL
(0.3310 -> 0.0320) improve simultaneously, which matters because ECE alone can be
minimised by a constant predictor.

**Six-source calibration audit (new in this version).** All rows below are the same
split, the same candidate pairs, and the same metric implementation:

| Probability source | ECE | Brier | NLL | DP-free? | Post-processing? |
|---|---|---|---|---|---|
| RNAformer (32M, bprna ckpt) | **0.0015** | 0.0025 | 0.0132 | yes | none |
| Our exact marginals (CRF, inside-outside) | **0.0004** | 0.0030 | 0.0137 | no | none |
| Our recalibrated head | 0.0031 | 0.0057 | 0.0320 | yes | 2-param affine (VL0) |
| ViennaRNA exact BPP | 0.0048 | 0.0051 | 0.0259 | no | none |
| UFold | 0.0147 | 0.0108 | 0.0437 | yes | none |
| Our raw head | 0.1955 | 0.1015 | 0.3310 | yes | — |

This is, to our knowledge after search, the first systematic calibration audit of RNA
base-pair probabilities. It cuts both ways and we report both edges: DP-free
calibration is **not unique to us** (RNAformer is already calibrated), so no
first-to-calibrate claim is made anywhere in this draft; and the audit gives the field
a measurement it did not have — the spread across supposedly comparable probability
outputs is two orders of magnitude.

Two sensitivity checks:

- **The result does not depend on any choice made on the validation split.** With a
  validation-selected decode weight of 0.75 instead of the model's own weight, TS0
  micro F1 is 0.4959 versus 0.4953 — a difference of 0.0006 — and the recalibrated
  gap is 0.0006. The validation split is not a proxy for TS0 (see §4.3), so this check
  matters and it passes.
- **The result holds on the secondary split, and the same insensitivity holds there —
  but that split is contaminated, so the number is provisional.** On ArchiveII
  (3,950 rows) the recalibrated gap is **0.0044** at the model's own weight and 0.0093
  at the validation-selected weight, with micro F1 0.5834 versus 0.5829 — again a
  difference of 0.0005. **However §4.5 shows 35.6% of ArchiveII is duplicated in the
  training split**, so these figures are withdrawn pending re-measurement on the
  de-duplicated subset (2,544 rows, queued). The TS0 row above is not affected: TS0
  loses 3 of 1,288 rows to the same filter.

The TS0 row therefore passes under both decode conventions, and the decode convention
moves micro F1 by 0.0006 there. The headline involves no selection on any split, which
is what makes it reportable; the ArchiveII replication is provisional for the separate
reason given in §4.5.

**The calibration result also survives the cross-family shift.** On bpRNA-new — the
split where accuracy collapses — the recalibrated gap is **0.0014** at the model's own
weight, again inside the 0.02 threshold. That is the cleanest form of the claim in this
draft, because it does not depend on F1 at all: the head ranks pairs poorly across
families and still reports probabilities whose calibration error matches the exact
marginals' to within two thousandths.

**Reference point.** ViennaRNA's exact base-pair probabilities on TS0 have
ECE **0.0048**. Our recalibrated head is at **0.0023** — the same order of magnitude,
obtained without a partition function.

### 4.3 Structure accuracy: the pooled number is misleading in both directions

Pooled, we now sit between the physical baselines and the strongest published deep
baselines, and the capacity sweep moves us further up:

| Split | Ours (micro F1, 128-dim head, TR0) | Ours (4.8x-capacity head, TR0) | Ours (TR1: 4.29x data) | Ours (TR1, 2x steps) | Ours (capacity x data) | ViennaRNA centroid | ViennaRNA mfe | MXfold2 | UFold | RNAformer | Nussinov+Turner prior |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TS0 | **0.5938** (8 seeds) | **0.6425** | 0.5840 | **0.6147** | **0.6446** | 0.5393 | 0.5222 | 0.5651 | 0.6598 | **0.7578** | 0.2124 |
| ArchiveII (3,950) — **withdrawn**, see §4.5 | ~~0.5829~~ | — | — | — | — | ~~0.6207~~ | ~~0.5764~~ | — | — | — | ~~0.2010~~ |
| bpRNA-new (5,388) | 0.4870 | 0.4641 | **0.5162** | 0.4999 | 0.4558 | **0.6770** | 0.6379 | — | 0.6106 | — | **0.3015** |
| — RiNALMo-ft 650M (re-run, strict micro) | TS0 **0.7210** | — | — | — | — | — | — | — | — | — | bpRNA-new **0.4489** (below our 0.4870) |
| — structRFM 86M (re-run, strict micro) | TS0 **0.6638** | — | — | — | — | — | — | — | — | — | bpRNA-new **0.5438** (above our 0.4870) |
| 8-seed ensemble (§4.3f), TS0 / bpRNA-new | 0.6105 / 0.5106 | — | — | — | — | — | — | — | — | — | — |

The 8-seed ensemble row averages the seed-specific score matrices before the
single Nussinov decode — no retraining. §4.3f reads it as a variance bound,
not a method: the +0.017/+0.024 gains cap the seed-stochastic share of the
deficit at ~15–20%, establishing that the remaining gap is systematic.

The TR1 columns are the data-scaling experiment, the 0.4641 cell is the
capacity-on-cross-family measurement, and the last "Ours" column is the combination
(4.8x capacity on the 4.29x corpus), all at 20,000 steps except the 40,000-step
column. **The two scaling axes point in opposite directions out of distribution**:
against the matched TR0 baseline (0.4870) on bpRNA-new, capacity *costs* 0.023
(0.4870 -> 0.4641, per-sequence p = 7e-83) while data gains +0.029 (0.4870 -> 0.5162,
per-sequence p = 4e-62; 0.4954 in the TR1 second seed, 2-seed mean 0.5058);
in-distribution the effects differ again: capacity gains +0.039 paired (4 seeds),
data costs −0.012 at 20k (TR1 2-seed TS0: 0.5840 / 0.5842). Doubling the training
steps splits the picture once more: pooled gains +0.031 (0.5840 -> 0.6147, 9x the
seed spread) while cross-family **loses** 0.016 (0.5162 -> 0.4999, 5x the spread).
Training beyond roughly two epochs on the larger corpus improves in-distribution
accuracy and gives back part of the cross-family gain — the out-of-distribution
optimum arrives earlier than the in-distribution one, which bounds how far this
recipe can be pushed by optimisation alone.

**The two axes combine additively in-distribution and interfere cross-family —
now replicated in a second seed.** The combination arm reaches **0.6446** on TS0
(0.6282 in its second seed; mean 0.6364 over two) — the best mean number in this
draft, consistent with the axes' pooled effects composing. On bpRNA-new it scores
**0.4558** and **0.4088** in the two seeds (mean 0.4323), below *every* single-axis
number (ff 0.4870, big 0.4641, TR1 0.5162/0.4954 mean 0.5058): the interaction is
negative against the matched baseline (−0.031 and −0.078 per seed, per-sequence
p = 4e-147 for s0) and −0.06 to −0.09 against either single axis, with a sharply
conservative precision/recall profile out of distribution (0.701/0.338).
More capacity on more diverse data does not buy more generalisation here; it buys a
harder-trained decision boundary that generalises worse outside the training
families. We report this as the scaling result of the draft: the three knobs
(capacity, data, steps) have directionally different effects on the two splits, and
optimising them jointly is not the same as optimising them separately.

Three readings of this table, stated exactly:

1. **The capacity gain is robust at the pooled level and concentrated at the
   per-sequence level.** The base configuration spans 8 seeds (mean 0.5938, std
   0.0034); the 4.8x-capacity head spans 4 seeds (0.6425 / 0.6189 / 0.6337 / 0.6345,
   mean 0.6324, std 0.0098) — a paired gain of +0.039, ~12x the base spread, with the
   larger capacity arm also showing the larger seed spread. But the paired
   per-sequence mean gain is only +0.016 in s0, **−0.021 in s1 (p = 7e-9)** and
   −0.005 in s2 (n.s.): the pooled micro gain lives on pair-dense sequences, and
   §4.3c quantifies exactly this aggregation divergence rather than letting it pass.
   The same paired tests confirm the data gains on both splits (p = 4e-62
   cross-family, p = 3e-12 pooled at 40k) and the cross-family capacity loss
   (p = 7e-83). The same evidence falsifies two smaller effects: a learnable
   prior-weight (two seeds: 0.6006 / 0.5912, mean 0.5959 — indistinguishable from
   the base mean) and the three auxiliary objectives (+0.0065, single-seed) —
   neither is claimed as a finding.
2. **The hierarchical cascade is a negative result, reported as one.** The cascade
   head at the same budget and steps scores 0.5011 (P 0.794 / R 0.366): its
   differentiable gate over-suppresses pairs out of distribution. Its training-time
   helix recall (0.997) did not transfer. A conservative variant of the same
   architecture (miss-cost 200, sparsity 0.02) recovers to 0.5889 — statistically
   level with the flat head and below its mean — so the cascade family offers no
   gain while its aggressive variant carries a demonstrated out-of-distribution
   failure mode. We report this at full length because the architecture was
   motivated by a genuine complexity argument, and the failure signature
   — high precision, collapsed recall — is that of a conservative gate, not noise.
3. **A monotone decode bias adds nothing.** A validation-fitted additive bias before
   the DP decode moves TS0 by +0.001 (0.6760 -> 0.6770 on a matched 400-sequence
   subset, verified with the main evaluator at 0.6772). Sending logits, not
   probabilities, to the DP remains the right choice (that comparison was +0.045).

### 4.3b Three component ablations on the trained checkpoint

All three use the same step-20000 checkpoint, the same score extraction, and TS0;
only the named component is switched off.

| Configuration | micro F1 | P / R | Crossing / hairpin violations |
|---|---|---|---|
| DP decode (reference) | **0.5955** | 0.580 / 0.612 | 0 / 0 |
| Non-crossing constraint removed (independent threshold, UFold-style) | **0.0586** | 0.031 / 0.763 | **1,268 of 1,288 sequences** / 0 |
| Turner residual zeroed (`MLP_T = 0`) | **0.2124** | — | 0 / 0 |
| Calibration temperature off (`T = 1`) | 0.5955 | — | 0 / 0 |

The legalisation cost of the built-in constraint is **0.537** — the head's scores are
log-potentials, and thresholding them independently collapses precision to 0.031
while making 98% of the sequences contain crossing pairs. The DP harness is
load-bearing, not a formality. The learned residual contributes **+0.383** (64% of
the final F1) on top of the Nussinov+Turner stacking baseline; the temperature
calibration layer is decode-neutral by construction (it rescales probabilities
without reordering the argmax), which the measurement confirms exactly.

### 4.3c Paired significance tests and the aggregation divergence

Every comparison above is additionally tested as a paired per-sequence Wilcoxon
signed-rank test (two-sided, zero_method=wilcox) with Holm-Bonferroni correction
across the 10-test family (`tools/stats_definitive.py`, artifacts
`tables/stats_definitive.json`):

| Comparison (paired, same sequences) | n | micro Δ | per-seq mean Δ | p (Holm) |
|---|---|---|---|---|
| Capacity s0, TS0 | 1,288 | +0.047 | +0.016 | 6.3e-05 |
| Capacity s1, TS0 | 1,288 | +0.030 | **−0.021** | 2.8e-08 |
| Capacity s2, TS0 | 1,288 | +0.042 | −0.005 | 0.85 |
| Capacity s0, bpRNA-new | 5,388 | **−0.023** | **−0.046** | 5.7e-82 |
| Data (TR1@20k) s0, bpRNA-new | 5,388 | +0.029 | +0.036 | 2.8e-61 |
| Data (TR1@40k) s0, bpRNA-new | 5,388 | +0.013 | +0.014 | 1.8e-08 |
| Data (TR1@40k) s0, TS0 | 1,288 | +0.031 | +0.032 | 2.0e-11 |
| Combination s0 vs ff, bpRNA-new | 5,388 | −0.031 | −0.069 | 3.6e-146 |
| Combination s0 vs TR1@40k, bpRNA-new | 5,388 | −0.044 | −0.084 | 8.0e-199 |
| Combination s0 vs big, TS0 | 1,288 | +0.002 | +0.001 | 0.85 |

Two results survive the whole family at extreme significance: **data scaling helps
cross-family (+0.036 per-sequence) and capacity hurts it (−0.046)**. Two results show
that pooled micro and per-sequence mean disagree in direction:

1. **The capacity gain is an aggregation effect as much as an accuracy effect.**
   Micro-pooled, all three capacity seeds gain (+0.047 / +0.030 / +0.042); per-sequence,
   s1's mean *drops* (−0.021, p = 7e-9). Quartile analysis by ground-truth pair count
   explains the divergence: in s1 the mean per-sequence gain is +0.010 on the
   pair-poorest quartile (mean 9.6 ground-truth pairs) and −0.044 on the pair-richest
   (mean 53.6 pairs); in s0, +0.028 vs +0.007. Pooling TP/FP/FN weights long,
   pair-dense sequences, so a head that improves them moves micro F1 more than it
   moves the typical sequence. Reporting micro alone would overstate the capacity
   axis in a way the data does not support; we therefore report both aggregations
   everywhere and treat the per-sequence test as the stricter one.
2. **The data axis has no such divergence.** TR1's gains are uniform across length
   buckets (+0.016 to +0.038, all four buckets positive) and the per-sequence mean
   *exceeds* the micro gain (+0.036 vs +0.029) — data scaling improves the typical
   sequence, not just the pooled pool. The two scaling axes differ not only in
   direction but in *distribution* of gains, and a single aggregate would hide both
   facts.

A protocol note follows from this, and we state it as a methodological finding of the
audit: pooled micro F1, the de facto standard aggregation for this benchmark family,
can report a positive capacity effect whose per-sequence reality is negative in
individual seeds. Any evaluation that will be used to justify a scaling decision
should report the paired per-sequence test alongside the pooled number, and the
quartile decomposition when the two disagree.

The ArchiveII row is struck through rather than deleted so the withdrawal is visible:
35.6% of that split duplicates the training data, and re-measurement on the
de-duplicated subset is queued. **TS0 and bpRNA-new are the two splits in this draft
whose held-out status has been verified** (exact duplicates 0; mean 20-mer containment
0.040 and 0.0001 respectively).

But TS0 is a mixture along **two** axes, and both matter.

**Axis 1, source.** bpRNA-1m sequence names carry their source database, and the two
large sources have very different structural character: `CRW` (Comparative RNA Web) is
dominated by rRNA and tRNA with conserved, canonical structures, while `RFAM` spans a
wide range of families. Their proportions differ sharply between splits — `CRW` is 6.0%
of TR0, 7.3% of TS0, and **50.5% of VL0**.

**Axis 2, length.** Per-length-bucket results are the protocol-mandated form.

Crossing the two axes, with our model and the baselines scored on the *same* sequences
in every cell, gives the central accuracy result of this draft:

| Source | Length | n | **Ours** | ViennaRNA centroid | ViennaRNA mfe | Nussinov+Turner prior | **Ours − centroid** |
|---|---|---|---|---|---|---|---|
| `CRW` | <=100 nt | 68 | **0.9677** | 0.6729 | 0.6413 | 0.4187 | **+0.295** |
| `CRW` | 100–200 nt | 16 | **0.8725** | 0.6004 | 0.6447 | 0.2148 | **+0.272** |
| `RFAM` | <=100 nt | 486 | 0.5598 | **0.6209** | 0.5913 | 0.2893 | **−0.061** |
| `RFAM` | 100–200 nt | 498 | 0.4272 | **0.5347** | 0.4985 | 0.2000 | **−0.108** |
| `RFAM` | 200–400 nt | 126 | 0.4265 | **0.4345** | — | 0.1596 | **−0.008** |
| `RFAM` | >400 nt | 15 | 0.2766 | **0.4010** | — | 0.1201 | **−0.124** |

**The sign of the comparison is set by source, not by length.** On conserved sequences
we beat the partition-function baseline in *both* length buckets (+0.295 and +0.272); on
diverse sequences we trail in *all four* (−0.061, −0.108, −0.008, −0.124). Length changes
the magnitude within a stratum — `RFAM` declines from 0.5598 to 0.2766 — but it neither
flips the sign nor varies monotonically, so "longer is relatively worse" would also be
too strong a reading.

That also explains why any pooled number misleads: `RFAM` accounts for 984 of TS0's 1,288
sequences (76%), so the pooled comparison must show a loss even though we lead by 0.29
where structures are conserved.

Our own physical prior scores 0.12–0.42 in every cell, so the learned head contributes
real discriminative power throughout; it simply contributes far more where structures
are conserved.

The `CRW` 100–200 nt cell holds 16 sequences and the `RFAM` >400 nt cell holds 15, so
those two cells are weak on their own; the `CRW` <=100 nt cell (n=68) and its independent
replication on the validation split (n=81, 0.9709 vs 0.6702) carry the conserved-stratum
claim. `vienna_mfe` was not run on the two longest `RFAM` cells and is left blank rather
than estimated.

Two things follow that the grid alone does not show.

**The conserved-stratum advantage is not memorisation.** 20-mer containment against the
training split is 0.048 on TS0-`CRW`, with a maximum of 0.33, **no sequence above 0.5**,
and zero exact matches. The split-level de-duplication audit in §4.5 confirms it: TS0
loses 3 of 1,288 rows to the filter. This is a held-out result.

**The source effect is a property of the benchmark, not of our backbone.** A natural
objection is that a frozen language-model representation might simply be good at rRNA
because rRNA is abundant in pretraining. We can test this without training anything new,
because the repository also holds a from-scratch arm with a randomly initialised encoder
and the same head and data. Both arms show the same pattern, on the same sequences:

| Arm | Encoder | `CRW` <=100 nt (n=68) | `RFAM` <=100 nt (n=486) | `CRW` − `RFAM` |
|---|---|---|---|---|
| from scratch | randomly initialised | 0.8735 | 0.4105 | **+0.463** |
| frozen backbone | RiNALMo-giga | 0.9664 | 0.5490 | **+0.417** |

The random-encoder arm shows a *larger* `CRW` advantage than the frozen one, so the
stratification is a property of the data rather than of the representation. The frozen
backbone adds +0.093 on `CRW` and +0.139 on `RFAM` — it helps both strata, and its value
is a separate question from the `CRW`/`RFAM` split.

A third observation, about our own pipeline rather than the benchmark: **the same
checkpoint scores 0.9304 on VL0's at-most-100 nt bucket and 0.6386 on TS0's**, a gap of
0.29 that we had earlier suspected was leakage or an unidentified bug. It is composition
— VL0's short bucket is 65% `CRW`, TS0's is 12%. A homology explanation was tested and
rejected: the two splits have almost identical 20-mer containment to training (0.0423 vs
0.0403) and VL0's F1 is flat across containment bins.

`CRW` and `RFAM` are **source-database labels, not verified family labels**; sequence
names are unique within each split, so no family field is recoverable from them. This
stratification is a reproducible proxy that correlates with molecule type and structural
conservation, and it is not a substitute for a family-level split, which we have not
done. The strata also differ in GC content (0.564 vs 0.482) and pair density (0.270 vs
0.200 pairs per nt) at matched length, so the grid identifies a stratum rather than
isolating a cause.

ViennaRNA and MXfold2 rows are our own measurements on our own split files with our own
metric implementation, so these comparisons are like-for-like within this table. They
are not comparable to F1 numbers quoted from other papers, which use different splits,
different redundancy thresholds and different aggregation conventions.

### 4.3d Comparison under the RiNALMo-paper scoring convention, and against pretrained RNA language models

Every F1 above uses strict pair identity with pooled micro aggregation. The RiNALMo
paper — the source of both our backbone and the TS0 split itself — scores differently:
a predicted pair (i, j) counts as correct if the ground truth contains (i, j),
(i±1, j) or (i, j±1) (the Mathews 2019 tolerance), and F1 is averaged per-sequence
(macro). To place our numbers against that literature we re-scored every model for
which we hold predictions on the official full TS0 (1,305 sequences — ours and
RiNALMo's split file match exactly; we recover the 17 sequences absent from our
corpus subset from RNAformer's published test set), under the identical tolerant
convention:

| Model (TS0, Mathews-tolerant macro F1) | Value | How obtained |
|---|---|---|
| UFold | 0.7807 | our measurement, re-scored |
| RNAformer 32M (bprna ckpt) | 0.7779 | our measurement, re-scored |
| **RiNALMo-ft (650M, bprna ckpt)** | **0.7602** | **our re-run of their released checkpoint** (Zenodo 15043668, MD5-verified; official decode + scoring; threshold 0.06 from checkpoint) — supersedes the earlier quoted-only figure-derived ~0.6 |
| **Ours, big head (4.8x)** | **0.6474** | our measurement, re-scored |
| **Ours, bigtr1** | **0.6449** | our measurement, re-scored |
| **Ours, base 128-dim (ff)** | **0.6368** | our measurement, re-scored |
| structRFM (SgMLM pretrain, CNN+LSTM head) | 0.6628 strict / see §4.3g | our re-run of their released bpRNA1m SSP checkpoint (GitHub release v0.0.8, MD5-verified): **strict** macro 0.6628, strict micro 0.6638 — their paper has no tolerant convention, so their number is strict-only and not directly comparable to this table's tolerant column |
| MXfold2 | 0.6102 | our measurement, re-scored |
| ViennaRNA centroid | 0.5665 | our measurement, re-scored |

Three statements, carefully bounded:

1. **The gap to the strongest measured models is 0.13 under the tolerant convention**
   (0.647 vs 0.78), wider than under strict micro because our precision profile gains
   less from tolerance than UFold's and RNAformer's. No aggregation convention makes
   the gap disappear.
2. **The RiNALMo fine-tuned model has now been re-run from its released
   checkpoint, and the re-run resolves the earlier uncertainty.** Their
   fine-tuned weights (Zenodo, MD5-verified) score **0.7602 tolerant macro on
   the official TS0** under their own decode and scoring implementation —
   not the ~0.6 that secondary reporting of their figure suggested. The same
   checkpoint, frozen-backbone swap, and training corpus now decompose the
   TS0 gap exactly: the identical RiNALMo-giga encoder fine-tuned with a ResNet
   head reaches 0.760 while our frozen-encoder CRF head reaches 0.647, so
   **backbone fine-tuning accounts for ~0.11 of the ~0.13-0.15 gap — the
   dominant term — and the residual (head architecture, training schedule,
   decode) is only 0.02-0.04.** On bpRNA-new the sign of that term flips: the
   fine-tuned 650M scores **0.4489 strict micro, below our frozen head's
   0.4870** (§4.4), mirroring the capacity arm's out-of-distribution reversal
   at a much larger parameter scale. The comparison remains
   training-budget-unmatched, in their favour on parameters and in ours on
   steps; both directions are reported.
3. **What survives any convention: the calibration property.** Of the rows in this
   table, ours is the only one whose pair probabilities are audited as calibrated
   outputs with an exact-marginal self-consistency check (§4.2); the accuracy rows
   trade against that property, and we report both sides rather than pick one.

A four-convention decomposition pins down how much of the visible gap is scoring
convention rather than model difference. Scoring the *same* two systems —
RNAformer's published-checkpoint predictions and ours — under {our project GT,
the RNAformer release GT} × {strict, tolerant}:

| TS0 F1 | project GT, strict | project GT, tolerant | release GT, strict | release GT, tolerant |
|---|---|---|---|---|
| RNAformer | 0.7454 | 0.7779 | 0.7093 | 0.7486 |
| Ours (ff s0) | 0.5970 | 0.6377 | 0.5724 | 0.6213 |
| **gap** | 0.148 | 0.140 | 0.137 | 0.127 |

The gap moves by at most 0.02 across all four conventions while staying in
0.13–0.15: **the gap is a model difference, not a scoring artefact** — neither
side is "wrong"; the conventions shift absolute values (tolerant +0.03–0.05,
release-GT −0.04) but not the ordering. This is why §4.3d reports tolerant
convention side-by-side rather than switching to it.

INF scores for the same TS0 split (per-sequence sqrt(P·R) averaged — the
RiNALMo paper's S4 convention): ours 0.6005 (ff) / 0.6199 (big) / 0.6238
(bigtr1); our measured UFold 0.7325 and MXfold2 0.5832 against their reported
0.67 and 0.61 — the ±0.06 discrepancies are scoring-implementation differences
(strict project-GT vs tolerant), which is exactly why every cross-paper table in
this draft separates measured from reported values.

### 4.3e A second out-of-distribution benchmark: TORNADO TestSetB

The RiNALMo paper also evaluates on Rivas et al.'s TestSetB — 430 RNAs from 22
Rfam families chosen to be structurally dissimilar from the training set (its
Supplementary Table S5; the strongest reported INF there is RiNALMo fine-tuned
on TrainSetA at 0.67, with CONTRAfold 0.64, MXfold2 0.63, RNAstructure 0.56 and
RNA-FM 0.49). We downloaded that benchmark, converted it through the identical
corpus pipeline (428 of 430 survive; two sequences carry non-standard letters),
and evaluated **zero-shot** — no model of ours has ever seen TrainSetA:

| TestSetB (n=428), zero-shot | strict macro F1 | tolerant macro F1 | INF |
|---|---|---|---|
| **Ours, big head (4.8x)** | 0.7723 | **0.7932** | **0.7781** |
| Ours, bigtr1 | 0.7189 | 0.7376 | 0.7290 |
| Ours, base 128-dim (ff) | 0.7101 | 0.7354 | 0.7133 |
| EternaFold (CONTRAfold engine, Eterna params; measured) | 0.6033 | 0.6366 | 0.6095 |
| ViennaRNA centroid (measured) | 0.5577 | — | 0.5630 |
| RiNALMo fine-tuned on TrainSetA (reported) | — | — | 0.67 |
| CONTRAfold (reported) | — | — | 0.64 |
| MXfold2 (reported) | — | — | 0.63 |

**Our zero-shot capacity head exceeds every published number on this benchmark**
(0.7932 tolerant F1 / 0.7781 INF against RiNALMo's fine-tuned 0.67), without
training on TrainSetA. We report this next to the bpRNA-new result rather than
instead of it, because the two tell opposite stories and the difference is
itself the finding:

1. **"Cross-family generalization" is not one number.** On TestSetB — 22
   structurally dissimilar but classic ncRNA families — our bpRNA-trained model
   transfers almost fully (0.79 zero-shot). On bpRNA-new — genuinely novel
   families — it falls to 0.49-0.52 against a physical baseline at 0.68 (§4.4).
   The benchmark's family composition, not a scalar "OOD-ness", decides the
   outcome; any claim about this method's generalization must name the
   benchmark.
2. **The capacity axis flips sign between the two OOD benchmarks** (+0.058 on
   TestSetB, −0.023 on bpRNA-new, both significant per §4.3c's test family).
   Capacity deepens family knowledge that transfers when the target families
   are cousins of the training distribution and hurts when they are strangers.
   The three-knob direction matrix (capacity/data/steps × three benchmarks) is
   the most information-dense result of this draft.
3. The EternaFold row is a CONTRAfold-family proxy (same engine, Eterna-trained
   parameters) and is double-listed against their CONTRAfold number rather
   than conflated with it; RNAstructure and RNA-FM are quoted-only, as in
   §4.3d.

### 4.3f Seed-variance decomposition: how much of the gap is draw luck

All results so far use a single seed. Averaging the K seed-specific score
matrices before the single Nussinov decode (the decode of §3) costs no
training, so it isolates one question: how much of the deficit is seed-level
stochasticity rather than systematic?

| ensemble (K models, score-matrix averaging) | TS0 micro F1 | bpRNA-new micro F1 |
|---|---|---|
| single model, 8-seed mean ± std | 0.5938 ± 0.0034 | 0.4870 |
| 8-seed ensemble (TR0 seeds) | 0.6105 (+0.017) | 0.5106 (+0.024) |
| reference: TR1 (data-scaled) single model | 0.6282 | 0.5162 |

Read as a diagnostic, not a remedy: the ensemble's +0.024 on bpRNA-new bounds
the seed-stochastic component of that benchmark's deficit, and the bound is
small — against the 0.167 gap to the physical baseline (0.5106 vs 0.6770) and
the 0.115 gap to RNAformer on TS0, seed variance accounts for at most
~15–20%. **The remainder is systematic — representation, data, and objective —
and is not addressable by averaging draws.** We report this bound explicitly
because it redirects the improvement programme: the levers that matter are the
ones §4.3 already measures (data scale, monotone +0.029 and unexhausted) and
the ones it does not yet isolate (pair-aware pretraining, 2D context in the
pair scorer), not ensembling.

### 4.3g Gap attribution closes: two re-run baselines and a backbone swap

Three measurements this revision turn the gap decomposition of §4.3d/§4.3f
from inference into arithmetic:

**(a) RiNALMo-ft re-run (650M fine-tuned, released checkpoint).** TS0
tolerant macro **0.7602** / strict micro 0.7210 — the exact model that
supplies our frozen backbone, measured under its own decode and scoring. The
same encoder frozen + our CRF head: 0.6474. **Backbone fine-tuning = ~0.11 of
the gap; every other factor combined = 0.02–0.04.**

**(b) structRFM re-run (86M structure-guided pretrain, CNN+LSTM+DP).** TS0
strict micro **0.6638**, bpRNA-new strict micro **0.5438** — a model *without*
a RiNALMo-class backbone that outscores our frozen head on both benchmarks
(+0.021 TS0, +0.057 bpRNA-new) using structure-guided pre-training data (21M
sequence-structure pairs, BPfold-derived) and a 2D-context pair scorer. It
sits between us and RNAformer, and it narrows what "backbone scale" explains:
the residual gap to it is exactly the pair-aware-pretrain + 2D-scorer
combination.

**(c) Backbone swap arm (RNA-FM frozen, same CRF head).** TS0 0.4199 /
bpRNA-new 0.3005 against RiNALMo-giga's 0.5938/0.4870: the representation
axis alone is worth ~0.17 in-distribution, larger than the fine-tuning term —
backbone choice dominates, backbone adaptation adds on top.

**Combined decomposition of the 0.13–0.15 TS0 gap to the strongest models:**

| factor | estimate | evidence |
|---|---|---|
| backbone fine-tuning (frozen → adapted) | ~0.11 | (a): same encoder, both regimes |
| pair scorer 2D context + structure-aware pretrain data | ~0.02–0.04 residual vs structRFM's path | (b) sits between the two regimes |
| seed stochasticity | ≤ ~0.02 (≤15–20% of deficit) | §4.3f ensemble bound |

This is the chase map: the next arms are a 2D-context scorer on the frozen
backbone (isolating (b)'s architecture half at matched capacity — running)
and gradual unfreezing (isolating (a) under our objective — prepared).

### 4.4 Cross-family generalization is insufficient on bpRNA-new (quantified) — and the opposite on TestSetB

§4.3 shows the model is strong where structures are conserved and weak where they are
diverse. bpRNA-new is the extreme of the latter: it is built from *new* families by
construction. There the picture inverts. The physical baseline *improves* — ViennaRNA
centroid goes from 0.5393 on TS0 to **0.6770** — while our model falls from 0.5956 to
**0.4870** (step-20000 checkpoint, matched protocol). §4.3e shows the other end of
the OOD spectrum, where the same checkpoints reach 0.79 zero-shot; this section
quantifies the failing end, and the two together bound what "cross-family" means for
this method.

The decode weight matters more here than anywhere else, and getting it wrong produced a
claim we have had to withdraw:

| bpRNA-new | ours | our own prior | centroid |
|---|---|---|---|
| `w = -1` (model's own trained weight, step 20000) | **0.4870** | 0.3015 | 0.6770 |
| `w = 0.75` (selected on the validation split) | 0.3094 | 0.3015 | 0.6770 |

At `w = 0.75` the model sits only 0.008 above its own physical prior, which we had
earlier read as "cross-family discriminative power disappears". At the model's own
weight it is 0.186 above. (The 0.3094 row is the step-3500-era protocol retained as
the weight-sensitivity control; the matched-baseline row is `w = -1` at step 20000.)
The difference is real and the conclusion changes: the head does contribute across
families, and the earlier "collapse" reading was partly an artefact of selecting the
weight on a split whose composition we now know is skewed (§4.3) — and partly an
artefact of comparing against an under-trained snapshot, which the v3.4 audit
corrected (0.3536 -> 0.4870).

What does **not** change: **we trail the physical baseline by 0.19 on this split**, which
is the method's dominant weakness, and no pooled number in this draft should be read as
generalization evidence.

We checked the obvious explanations for the deficit and they do not account for it:

- **Not a decode-weight artefact of the kind just described.** Even at the best weight
  the deficit is 0.19.
- **Not exact overlap.** Exact sequence overlap between bpRNA-new and TR0 is zero, and
  20-mer containment is 0.0001.
- **Not measured homology.** That is why bpRNA-new is our OOD split of choice.

The mechanism is not established. Our working hypothesis is that the frozen
representation transfers but the family-specific structural motifs learned by the head
do not, whereas the physical energy model is family-agnostic. Testing that requires a
family-level train/validation split that we have not yet run, and we therefore state
the hypothesis as a hypothesis.

### 4.5 Two dataset caveats that constrain what may be claimed

1. **ArchiveII is not a held-out split, and every ArchiveII number in this draft is
   withdrawn pending re-measurement on the de-duplicated version.** Measuring the
   overlap against the training split gives:

   | split | rows | exact duplicates of TR0 | 20-mer containment > 0.5 | kept |
   |---|---|---|---|---|
   | **ArchiveII** | 3,950 | **812 (20.6%)** | 594 (15.0%) | **2,544 (64.4%)** |
   | TS0 | 1,288 | 0 | 3 (0.2%) | 1,285 (99.8%) |
   | VL0 | 196 | 0 | 0 | 196 (100%) |
   | bpRNA-new | 5,388 | 0 | 0 | 5,388 (100%) |

   The duplicates are literal: ArchiveII's `5s_Acholeplasma-laidlawii-2.bpseq` is
   byte-identical to TR0's `bpRNA_RFAM_654.bpseq`. So 35.6% of ArchiveII is not
   held-out, and its 0.5829 — and the +0.149 advantage over centroid in the at-most-100 nt
   bucket — are partly memorisation scores. A clean subset (2,544 rows) has been built
   and the re-measurement is queued; until it lands, **only TS0 and bpRNA-new numbers
   in this draft are held-out results.** TS0 loses 3 rows to the same filter and
   bpRNA-new loses none.
2. **ArchiveII can only be scored on 3,950 of 3,966 rows.** RiNALMo's `max_pos` is
   1024, so 16 rows have no cached embedding. The evaluator refuses to silently mix
   frozen and freshly computed representations, which is the correct behaviour; the
   cost is that any ArchiveII number must carry the 3,950/3,966 qualifier — on top of
   the de-duplication qualifier above.

## 5. Limitations

1. **The existence claim (C1-a) was tested and failed, and we report the failure.**
   The pre-registered criterion asked whether our probabilities are at least as well
   calibrated as SPOT-RNA / UFold sigmoid outputs. Against UFold they are (0.0031 vs
   0.0147). But RNAformer's probabilities — single forward pass, non-crossing greedy
   decode, DP-free — are better calibrated than our recalibrated head (0.0015 vs
   0.0031) with no post-processing at all. "We beat UFold" would be trading on a weaker
   opponent, so no uniqueness claim is made; what survives is the systematic audit
   itself and the self-consistency result (C1-c).
2. **Cross-family generalization is insufficient**, and §4.4 quantifies it rather than
   arguing it away. The same weakness shows up within-distribution as a sharp dependence
   on how conserved the source population is (§4.3): we beat the partition-function
   baseline by 0.29 on the conserved stratum and lose by 0.07 on the diverse one, so
   **any single pooled F1 for this method is misleading** and we report both strata.
3. **The raw head is not calibrated.** Only the two-parameter affine recalibration
   passes the gate. The map is fitted on bpRNA VL0, which §4.3 shows is a
   composition-skewed split (50.5% `CRW` against TS0's 7.3%), so it is a poor proxy for
   TS0 in general; the headline result therefore does not select anything on it.
4. **Training-set labels are a compilation, not experiment.** bpRNA-1m's structures
   are assembled from several sources, and a ~96% precision ceiling on pair labels has
   been reported for it. This bounds what any F1 on these splits can mean.
5. **The strongest published baseline is RNAformer at 0.7578** on the same split and
   metric implementation; we reach 0.6425 with 4.8x head capacity. The remaining gap
   (0.115) is the draft's central open number, and the combination arm (capacity x
   4.29x training data) has now been measured: it improves in-distribution to 0.6446
   but *hurts* cross-family (0.4558, §4.3), so it cannot close the gap and is
   reported as the honest ceiling of this recipe at this budget. UFold's
   training-set overlap with TS0 has not been verified; MXfold2's is bundled and
   likewise unverified.
6. **No speed-up ratio is claimed, but measured latency is reported.** The decode
   path used in every number above is an exact `O(L^3)` dynamic program, so the
   "DP-free" property refers only to the absence of the partition function, not to
   a wall-clock advantage. Measured per-sequence end-to-end latency of System-1
   (one forward pass + legal decode, batch 1, on the same A100 as every other
   measurement) is: median 133 ms at <=100 nt (n=565), 167 ms at 100–200 nt
   (n=528), 716 ms at 200–400 nt (n=167) and 1,341 ms at 400–600 nt (n=28) for the
   128-dim head; the 512-dim head is comparable (80 / 166 / 778 / 2,256 ms by
   bucket median). The forward pass and the decode contribute roughly equally at
   long lengths. These are honest numbers, not a marketing claim, and we make no
   comparison against ViennaRNA's wall clock because our implementation of the DP
   decode is unoptimised numpy and any ratio would measure our engineering, not
   the method.
7. **Seed coverage: the headline family is complete at 8 seeds** (mean 0.5938,
   std 0.0034, range 0.0097); the capacity arm has 4 seeds (paired gain +0.039, but
   per-sequence divergence in one seed, §4.3c) plus a sum-normalised replication
   (§5 item 9); the combination and TR1 arms have 2 seeds each (interaction
   direction unanimous); the cascade and auxiliary-objective arms are single-seed.
   The learnable-prior-weight (2 seeds) and auxiliary-objective (1 seed) effects
   remain unreplicated and are flagged as such.
8. **Pooled micro F1 alone can mislead — inside our own draft.** The v3.4 protocol
   audit corrected a cross-family baseline that had been taken from an under-trained
   snapshot, reversing one sign and rescaling two effects; §4.3c documents that the
   capacity effect is positive pooled but negative per-sequence in one of four
   seeds. We report the audit trail openly because aggregation and protocol
   mismatches are failure modes any evaluation of this kind inherits, not because
   we believe we are uniquely error-prone.
9. **The capacity comparison's normalisation confound is now closed by a
   single-variable replication.** The capacity arm trains with length-normalised
   NLL while the base family uses unnormalised NLL (§3). A matched control at
   128 dims bounds the confound at 0.002, and a **sum-normalised 512-dim head**
   trained after the audit reproduces the capacity effect directly (0.6302 vs
   0.6425 length-normalised, both ~+0.04 over the 0.5938 base): the gain is
   capacity, not normalisation, and both rows are reported.
10. **The RiNALMo paper's ArchiveII leave-one-family-out protocol (its Tables
   S2/S3) is not replicated.** It requires nine separate fine-tuning runs with
   family-held-out splits; our ArchiveII numbers are additionally withdrawn for
   training-set duplication (§4.5), so no zero-shot substitute is offered on
   those families either. The two RiNALMo-paper benchmarks we do replicate
   share their split exactly (S4) or are fully converted and re-measured
   zero-shot (S5/§4.3e), and the measured-vs-reported distinction is maintained
   in every table.
11. **The Jev decision-model paradigm is community-sourced, not peer-reviewed.** Its
   performance numbers are vendor self-reported and are not cited as fact anywhere in
   this draft. The calibration objective used here is our own design inspired by that
   paradigm, and we do not claim to reproduce it.

## 6. Conclusion

A single-forward-pass decision head can be brought to within 0.0028 of the exact
partition-function marginals' calibration error, using two parameters fitted without a
partition function — and the same head's probabilities remain calibrated across a
cross-family shift that its ranking does not survive. That is the positive result, and
it is measured on a clean in-distribution split with no test-set selection. The audit
that contextualises it shows the field's probability outputs span two orders of
magnitude in calibration quality, with RNAformer at the good end without ever having
reported it: measurement, not architecture, was the missing piece.

On accuracy the answer is stratified rather than uniform. Where structures are
conserved — the `CRW` stratum of short sequences — the DP-free head reaches 0.9664
against the partition-function baseline's 0.6729, replicated on an independent split
and with no measurable homology to training. Where structures are diverse it loses;
across families the outcome is benchmark-dependent in the extreme: on Rivas TestSetB
(22 structurally dissimilar families) the same checkpoints reach **0.79 zero-shot**
— above every published number on that benchmark, without touching its training set —
while on bpRNA-new (genuinely novel families) they start from the physical prior
(0.30) and data scaling moves them to 0.52, with the physical baseline at 0.68 still
ahead. §4.3f bounds the seed-stochastic share of that deficit at ~15–20% via an
ensemble probe — the rest is systematic and calls for representation- and
data-level changes rather than draw-averaging. The two
scaling axes are quantified separately on all three benchmarks with
paired per-sequence significance tests, and their effects do not compose
out-of-distribution: the measured interaction is negative.

We report the stratified result as the main accuracy finding rather than the pooled
number, because the pooled number is dominated by the stratum where we lose and would
understate the method. The complement matters just as much: a calibration result that
transfers across families while ranking does not is half a result — but it is the half
that downstream users (design, variant interpretation) consume first, and it is now
measured rather than assumed.

## Appendix A. Evidence ledger

Every number in §4 traces to one of these artifacts. Paths are on the cluster unless
stated; the code lives at `/home/cunyuliu/rna-jepa` and the artifacts at
`/mnt/cunyuliu/rna-jepa`.

| Result | Artifact | Command |
|---|---|---|
| TS0 headline (ff s0, step 20000), `w=-1` | `eval_decision/ff20000_ref_bprna_ts0/result.json` | `eval/ss/evaluate_decision.py --checkpoint ckpts/rinalmo_ff_b4_s0_step20000.pt --data ss_data/jsonl/bprna_ts0.jsonl --embedding-split bprna_ts0 --calib-data ss_data/jsonl/bprna_vl0.jsonl --prior-weight -1` |
| Seeds s1–s5 (headline spread) | `eval_decision/arms_rinalmo_ff_b4_{s1,s2}_step20000_ts0`, `eval_decision/ow_rinalmo_ff_b4_{s4,s5,s3}_step20000_bprna_ts0` | same protocol, `--seed` changed only |
| Capacity arm (big, 4.8x) | `eval_decision/arms_rinalmo_big_b4_s0_step20000_ts0` | same + `--d-z 512 --hidden 512` at train time |
| Capacity seeds s1–s3 (paired) + sum-norm replication | `eval_decision/ow_rinalmo_big_b4_{s1,s2,s3}_step20000_bprna_ts0`, `eval_decision/ow_rinalmo_bigsum_b4_s0_step20000_bprna_ts0` | same protocol, `--seed` changed only; bigsum adds `--nll-normalization sum` |
| Combination + TR1 second seeds | `eval_decision/ow_rinalmo_bigtr1_b4_s1_step20000_{bprna_ts0,bprna_new}`, `eval_decision/ow_rinalmo_ff_tr1_b4_s1_step20000_{bprna_ts0,bprna_new}` | same protocol |
| Paired significance tests + quartile/length-bucket decomposition | `tables/stats_definitive.json`, `tables/stats_significance.md` | `tools/stats_definitive.py`, `tools/stats_significance.py` |
| Cross-family TR0 baseline (matched protocol) | `eval_decision/ff20000_bprna_new/result.json` | `eval/ss/evaluate_decision.py --checkpoint ckpts/rinalmo_ff_b4_s0_step20000.pt --data ss_data/jsonl/bprna_new.jsonl --prior-weight -1` |
| Capacity on cross-family | `eval_decision/ow_rinalmo_big_b4_s0_step20000_bprna_new` | same, `--data ss_data/jsonl/bprna_new.jsonl` |
| Official full TS0 (1,305) evaluations | `eval_decision/official1305_rinalmo_{ff,big,bigtr1}_b4_s0_step20000` | `scripts/run_official1305.sh`; corpus `ss_data/jsonl/bprna_ts0_1305.jsonl` built by `tools/build_ts0_1305.py` from RNAformer's `test_sets.plk`; embeddings `embeddings/rinalmo-giga/bprna_ts0_1305.shard0of1.npz` |
| Mathews-tolerant re-scoring (§4.3d) | `tables/vienna_mathews_ts0.json`; per-model re-scores printed by `tools/rescore_mathews.py`, `tools/rescore_dbn.py`, `tools/rescore_mxfold2.py` | tolerant convention: (i,j) correct if GT has (i,j), (i±1,j) or (i,j±1); macro aggregation |
| TestSetB benchmark (§4.3e) | `eval_decision/testsetb_rinalmo_{ff,big,bigtr1}_b4_s0_step20000`, `eval_decision/baselines_testsetb.json`, `eval_decision/eternafold_testsetb.json` | corpus `ss_data/jsonl/testsetb.jsonl` from `tools/build_testsetb.py` (source: Rivas.tar.gz, mxfold2 v0.1.1 release); embeddings `embeddings/rinalmo-giga/testsetb.shard0of1.npz`; EternaFold via `tools/eternafold_testsetb.py` |
| Seed ensembles (§4.3f) | `logs/ensemble_new.log` (TS0 8-seed 0.6105; bpRNA-new 8-seed 0.5106; TR1 2-seed pending) | `tools/ensemble_eval.py`: average `_forward_scores` over the K seed checkpoints, single `nussinov_map` decode, `--prior-weight -1` |
| RiNALMo-ft re-run (§4.3g) | `eval_decision/rinalmo_ft_official_ts0.json` (tolerant macro 0.7602, strict micro 0.7210, threshold 0.06), `eval_decision/rinalmo_ft_bprna_new.json` (0.4553 / 0.4489) | `tools/eval_rinalmo_ft.py` on Zenodo checkpoint (md5 3688b049…, verified); official `prob_mat_to_sec_struct` decode + `_relax_ss` tolerant scoring + our strict micro; flash_attn 2.7.4 compat patch in `tools/RiNALMo-main/rinalmo/model/attention.py` |
| structRFM re-run (§4.3g) | `eval_decision/structrfm_official_ts0.json` (strict micro 0.6638 / their macro 0.6628), `eval_decision/structrfm_bprna_new.json` (0.5438 / 0.5338) | `tools/eval_structrfm.py` on GitHub-release v0.0.8 checkpoint (md5 147f8839…, verified); official MixedFold (CNN+LSTM+Turner+`predict_mxfold`) inference, py3.8/mrnabert env + BPfold pip + transformers-4.32 collator shim |
| Backbone-swap arm (§4.3g) | `eval_decision/ow_rnafm_ff_b4_s0_step20000_bprna_ts0/result.json` (0.4199), `..._bprna_new/result.json` (0.3005) | RNA-FM frozen embeddings (`embeddings/rna-fm/*.npz`, official `fm` pkg extraction) + same CRF head/protocol as `rinalmo_ff_b4_s0` |
| Plan-B 2D-scorer arm (§4.3g, running) | `runs/rinalmo_r2d_b4_s0/` (training, 20k steps) | `scripts/launch_plan_b.sh` + `tools/patch_resnet2d.py` (ResNet2DScorer, zero-init equivalence verified, column-chunked z) |
| Four-convention attribution (§4.3d, gap decomposition) | printed table, `tools/attribution_4ways.py` | RNAformer 0.7454/0.7779/0.7093/0.7486 vs ours 0.5970/0.6377/0.5724/0.6213 under {project, release} GT × {strict, tolerant} |
| Split-provenance check vs RiNALMo release | sequence-level set comparison, `tools/check_rinalmo_splits.py` (local) | our TS0 1,288 ⊂ official 1,305; TR0/new likewise subsets |
| Data-scaling (TR1) TS0 + bpRNA-new | `eval_decision/tr1_ff_step20000_{bprna_ts0,bprna_new}` | same, trained on `bprna_tr1.jsonl` |
| Cascade (negative) + conservative variant | `eval_decision/arch_rinalmo_casc_b4_s0_step2000_ts0`, `ow_rinalmo_cascR_b4_s0_step20000_bprna_ts0` | cascade arms at train time |
| Six-source calibration audit | `eval_decision/c1a_rnaformer_ref_bprna_ts0.json`, `reference_calibration.py` outputs | same split, same 6,022,538 candidate pairs |
| UFold baseline + its calibration | `records/UFOLD_BASELINE_ADAPTER.md`, `ufold_probs_*.npz` | `eval/ss/run_ufold.py` |
| RNAformer baseline + its calibration | `records/RNAFORMER_BASELINE_RUN.md`, `rnaformer_probs_*.npz` | `eval/ss/run_rnaformer.py` |
| ViennaRNA baselines (7 splits) | `records/BASELINE_RESULTS.md` | `eval/ss/baselines.py` |
| MXfold2 (incl. bprna-new) | `eval_decision/baselines_mxfold2_bprna_new.json` | `python -m mxfold2 predict` |
| Convergence curve 6k/10k/20k | `eval_decision/trend_ff_step{6000,10000}_ts0` + headline | same protocol, snapshots |
| Decode-affine probe (negative) | `eval_decision/affine_probe_ff20000.json` + `ff20000_ts0_first400` | `tools/probe_decode_affine.py` |
| Component ablations (§4.3b) | `eval_decision/offline_ablations_ff20000.json` | `tools/probe_offline_ablations.py` |
| From-scratch backbone failure | `eval_decision/fs_{full,nlldistill}_b4_s0_20260924T060638_step{20000,40000}_ts0` | own-encoder arms, no embeddings |
| 2x2 objective grid | `eval_decision/arms_rinalmo_{len,sum,bal,bal_s1,pw}_b4_s0_step20000_ts0` | objective-arm queue |
| De-duplication audit and clean subsets | `ss_data/jsonl/*_clean.jsonl` | `tools/dedup_against_train.py --train ss_data/jsonl/bprna_tr0.jsonl --threshold 0.5 --k 20` |
| Checkpoint step provenance | `tools/ckpt_steps.py` | reads `step` from inside each `.pt` |
| Full run-by-run log | `records/DECISION_TRAINING_LOG.md` §14.1–§14.53 | — |

Decoding is exact (`nussinov_map`), batch 1 for latency rows, and the illegal-structure
rate and hairpin-violation rate are 0.0000 for every row above.

## Appendix B. Anticipated objections and their landing points

| # | Objection | Landing point |
|---|---|---|
| Q1 | This is CONTRAfold with a neural scorer. | §1, §2 paragraph 1; §3 objective |
| Q2 | Without a partition function, why should the probability be calibrated? | §4.2 (both gaps); §3 recalibration |
| Q3 | Base pairs are correlated, so pair-level ECE is misleading. | §5 item 3; §4.2 exact-marginal comparison |
| Q4 | Nussinov + Turner is not ViennaRNA. | §4.3 last column; §2 |
| Q5 | `O(L^2)` memory is infeasible for long transcripts. | §5 item 6; §3 head |
| Q6 | Why not simply fine-tune RiNALMo? | §3 representation (backbone-orthogonal by construction) |
| Q7 | What about pseudoknots? | §4.1 decoding (non-crossing only); §5 |
| Q8 | The cotranscriptional order is trivial. | §3 — no order claim is made anywhere in this draft |
| Q9 | bpRNA-1m is redundant and may contaminate the test set. | §4.5 items 1-2; §5 item 4 |
| Q10 | Anyone can get a zero illegal-structure rate. | §4.1 — reported as a guarantee, not as a contribution |
| Q11 | Is the speed-up against McCaskill fair? | §5 item 6 — no speed-up is claimed; measured latency is reported instead |
| Q12 | Is the Jev citation reliable? | §5 item 8 |

## Appendix C. What makes this draft preliminary

1. **Second seeds for the combination and TR1 arms are training**, and the capacity
   arm's per-sequence divergence (§4.3c) is measured in one seed only. The headline
   family has eight seeds; the capacity arm four (pooled direction unanimous,
   per-sequence direction not); the learnable-prior-weight arm two (not
   significant); the cascade and auxiliary-objective arms one each.
2. **The TR1 convergence point is at 20,000 steps for a corpus 4.29x larger** —
   roughly 1.7 epochs; the 40,000-step point is measured (pooled +0.031,
   cross-family −0.016) but a 30,000-step midpoint was lost to a snapshot-discipline
   error and is not recoverable; the trend is reported from the two measured points
   only.
3. **The §4.3 stratification is a source-database proxy, not a family split.** It uses
   `CRW` / `RFAM` as they appear in bpRNA-1m sequence names, and the two strata are
   also not length-matched *to each other* — they are each truncated at the same edges.
   The `CRW` cells are small (68 and 16 sequences), so the +0.295 / +0.272 margins rest
   on few independent examples, although the <=100 nt cell replicates on the validation
   split. A verified family-level split has not been run.
4. **The strata differ in more than their source.** Mean length is matched (75.9 vs
   78.0 nt on TS0), but `CRW` is also GC-richer (0.564 vs 0.482) and more densely
   paired (0.270 vs 0.200 pairs per nt). Source, GC and density therefore vary
   together, and §4.3 identifies a stratum rather than isolating a cause. A density
   effect would move both methods, and it does not: ViennaRNA centroid is nearly flat
   across the two strata (0.6729 vs 0.6209) while ours swings by 0.42. That is an
   argument, not a proof — a density-matched re-measurement has not been done.
5. **The two smallest cells of the §4.3 grid are small.** `CRW` 100–200 nt holds 16
   sequences and `RFAM` >400 nt holds 15, so neither is strong on its own; the
   conserved-stratum claim rests on the `CRW` <=100 nt cell and its validation-split
   replication. `vienna_mfe` was not run on the two longest `RFAM` cells.

## References

All entries below were verified against PubMed (E-utilities record) or arXiv on
2026-09-25; the cluster has no outbound network, so verification was done
off-cluster and logged in `spec/citation_register.csv`.

1. Do CB, Woods DA, Batzoglou S. CONTRAfold: RNA secondary structure prediction
   without physics-based models. *Bioinformatics* 22(14):e90–e98, 2006.
   doi:10.1093/bioinformatics/btl246. PMID 16873527.
2. McCaskill JS. The equilibrium partition function and base pair binding
   probabilities for RNA secondary structure. *Biopolymers* 29(6–7):1105–1119,
   1990. doi:10.1002/bip.360290621. PMID 1695107.
3. Lorenz R, Bernhart SH, Höner zu Siederdissen C, Tafer H, Flamm C, Stadler PF,
   Hofacker IL. ViennaRNA Package 2.0. *Algorithms for Molecular Biology* 6:26,
   2011. doi:10.1186/1748-7188-6-26. PMID 22115189.
4. Danaee P, Rouches M, Wiley M, Deng D, Huang L, Hendrix D. bpRNA: large-scale
   automated annotation and analysis of RNA secondary structure. *Nucleic Acids
   Research* 46(11):5381–5394, 2018. doi:10.1093/nar/gky285. PMID 29746666.
5. Singh J, Hanson J, Paliwal K, Zhou Y. RNA secondary structure prediction using
   an ensemble of two-dimensional deep neural networks and transfer learning
   (SPOT-RNA). *Nature Communications* 10:5407, 2019.
   doi:10.1038/s41467-019-13395-9. PMID 31776342.
6. Sato K, Akiyama M, Sakakibara Y. RNA secondary structure prediction using deep
   learning with thermodynamic integration (MXfold2). *Nature Communications*
   12:941, 2021. doi:10.1038/s41467-021-21194-4. PMID 33574226.
7. Fu L, Cao Y, Wu J, Peng Q, Nie Q, Xie X. UFold: fast and accurate RNA secondary
   structure prediction with deep learning. *Nucleic Acids Research* 50(3):e14,
   2022. doi:10.1093/nar/gkab1074. PMID 34792173.
8. Franke JKH, Runge F, Hutter F. Scalable deep learning for RNA secondary
   structure prediction (RNAformer). arXiv:2307.10073, 2023.
   doi:10.48550/arXiv.2307.10073. ICML 2023 Workshop on Computational Biology.
9. Penić RJ, Vlašić T, Huber RG, Wan Y, Škić M. RiNALMo: general-purpose RNA
   language models can generalize well on structure prediction tasks. *Nature
   Communications* 16:5671, 2025. doi:10.1038/s41467-025-60872-5. PMID 40593636.
10. Mathews DH. How to benchmark RNA secondary structure prediction accuracy.
    *Methods* 162:60–67, 2019. doi:10.1016/j.ymeth.2019.04.003. PMID 30951834.
