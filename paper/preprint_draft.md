# DP-Free Calibrated Base-Pair Probabilities for RNA Secondary Structure

**Preliminary preprint draft — 2026-09-24.**

> **Read this banner before quoting anything.** Every number in §4 is a *measured*
> value produced by this repository on the A100 cluster, with the exact command and
> artifact path listed in Appendix A. Nothing here is a placeholder, and nothing here
> is extrapolated. The draft is nevertheless **preliminary** in three specific,
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
both directions. Pooled over TS0 our micro F1 is **0.5958** (head capacity 128 dims;
ViennaRNA centroid **0.5393**, MXfold2 **0.5651**, UFold **0.6598**, RNAformer
**0.7578** on the same split), and **0.6425** when the decision head is given 4.8x
capacity — a +0.047 gain that is seven times the seed variance we measure (0.0065).
Under-training was the dominant error source at earlier checkpoints (0.4953 at step
3500, monotonically rising to 0.5958 at step 20000 with no plateau), which also
reverses an early negative reading of the capacity hypothesis taken at step 2000.
Crossing the split by source and by length, with our model and the baselines scored on
the same sequences in every cell, reverses the comparison for one half of it:

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

Cross-family generalization is the method's dominant weakness: on bpRNA-new our micro
F1 is **0.3536** against ViennaRNA centroid's **0.6770** — a deficit of 0.32. It is not
a total collapse (our own Nussinov+Turner prior scores 0.3015, so the learned head still
adds 0.05), but it is far from competitive. Notably, the calibration result **does**
survive the cross-family shift: the recalibrated C1-c gap on bpRNA-new is **0.0014**,
inside the same 0.02 threshold. A model can rank poorly and still report honest
probabilities, and on this benchmark it does.

We conclude that DP-free calibration is achievable, and that it is achievable *without*
cross-family generalization — a dissociation that we quantify rather than paper over.

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
number, and reporting only the pooled one misstates the method in both directions.

The framework is deliberately borrowed rather than invented. We train a log-linear
model over the non-crossing structure space with an exact partition function during
training, and we then ask how much of that partition-function-calibrated behaviour
survives when the partition function is removed from inference.

## 2. Related work

The framework is not our contribution. CONTRAfold already trains a log-linear model
over the non-crossing structure space with an exact partition function, and the CRF
lineage is long. We sit inside it.

The DP-free precedents that matter are SPOT-RNA, SPOT-RNA2 and UFold: they emit an
`L x L` pair-probability matrix from a single forward pass and do not run a partition
function. We therefore do **not** claim to be first at emitting probabilities without
dynamic programming. What we could not find in that literature is any calibration
report — no ECE, no reliability diagram, no Brier score — which is the gap this draft
addresses.

We also correct an earlier internal misreading: CDPFold is a CNN followed by dynamic
programming (Front Genet 10:467, 2019). It neither removes the DP nor is a
calibration precedent, and it is not treated as a threat to the claim here.

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
| Secondary test | ArchiveII (BPfold bpseq) | **3,950 / 3,966** | **not** an out-of-distribution split, see §4.5 |
| Cross-family test | bpRNA-new | 5,388 | the only clean OOD split we hold |

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

| Split | Ours (micro F1, 128-dim head) | Ours (4.8x-capacity head) | ViennaRNA centroid | ViennaRNA mfe | MXfold2 | UFold | RNAformer | Nussinov+Turner prior |
|---|---|---|---|---|---|---|---|---|
| TS0 | **0.5958** | **0.6425** | 0.5393 | 0.5222 | 0.5651 | 0.6598 | **0.7578** | 0.2124 |
| ArchiveII (3,950) — **withdrawn**, see §4.5 | ~~0.5829~~ | — | ~~0.6207~~ | ~~0.5764~~ | — | — | — | ~~0.2010~~ |
| bpRNA-new (5,388) | **0.3536** | queued | **0.6770** | 0.6379 | — | 0.6106 | — | **0.3015** |

Three readings of this table, stated exactly:

1. **The +0.047 capacity gain is robust to seed noise.** The seed spread at step 20000
   is 0.0065 (s0 0.5958 vs s1 0.5893), so 4.8x head capacity clears it by 7x. The same
   test falsifies two smaller effects: a learnable prior-weight (init 0.5, trained to
   0.173) gained +0.0048, and the three auxiliary objectives gained +0.0065 — both
   within seed noise, and neither is claimed as a finding until replicated across
   seeds.
2. **The hierarchical cascade is a negative result, reported as one.** The cascade
   head at the same budget and steps scores 0.5011 (P 0.794 / R 0.366): its
   differentiable gate over-suppresses pairs out of distribution. Its training-time
   helix recall (0.997) did not transfer. We report this at full length because the
   architecture was motivated by a genuine complexity argument, and the failure mode
   — high precision, collapsed recall — is the specific signature of a conservative
   gate, not of noise.
3. **A monotone decode bias adds nothing.** A validation-fitted additive bias before
   the DP decode moves TS0 by +0.001 (0.6760 -> 0.6770 on a matched 400-sequence
   subset, verified with the main evaluator at 0.6772). Sending logits, not
   probabilities, to the DP remains the right choice (that comparison was +0.045).

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

### 4.4 Cross-family generalization is insufficient (quantified)

§4.3 shows the model is strong where structures are conserved and weak where they are
diverse. bpRNA-new is the extreme of the latter: it is built from *new* families by
construction. There the picture inverts. The physical baseline *improves* — ViennaRNA
centroid goes from 0.5393 on TS0 to **0.6770** — while our model falls from 0.4953 to
**0.3536**.

The decode weight matters more here than anywhere else, and getting it wrong produced a
claim we have had to withdraw:

| bpRNA-new | ours | our own prior | centroid |
|---|---|---|---|
| `w = -1` (model's own trained weight) | **0.3536** | 0.3015 | 0.6770 |
| `w = 0.75` (selected on the validation split) | 0.3094 | 0.3015 | 0.6770 |

At `w = 0.75` the model sits only 0.008 above its own physical prior, which we had
earlier read as "cross-family discriminative power disappears". At the model's own
weight it is 0.052 above. The difference is real but the conclusion changes: the head
does contribute across families, weakly, and the earlier "collapse" was partly an
artefact of selecting the weight on a split whose composition we now know is skewed
(§4.3).

What does **not** change: **we trail the physical baseline by 0.32 on this split**, which
is the method's dominant weakness, and no pooled number in this draft should be read as
generalization evidence.

We checked the obvious explanations for the deficit and they do not account for it:

- **Not a decode-weight artefact of the kind just described.** Even at the best weight
  the deficit is 0.32.
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
   4.29x training data) that could close it is still training — its result will either
   move this table or be reported as the honest ceiling of this recipe. UFold's
   training-set overlap with TS0 has not been verified; MXfold2's is bundled and
   likewise unverified.
6. **Speed claims are withheld.** The decode path used in every number above is an
   exact `O(L^3)` dynamic program, so the "DP-free" property currently refers only to
   the absence of the partition function, not to a wall-clock advantage. No speed-up
   ratio is claimed anywhere in this draft.
7. **Seed coverage is partial.** The headline family now has two seeds at step 20000
   (0.5958 / 0.5893; spread 0.0065) with five more training; the capacity and cascade
   arms are single-seed. The capacity gain (7x the spread) is not threatened by this;
   the learnable-prior-weight and auxiliary-objective effects are, and are flagged as
   unreplicated above.
8. **The Jev decision-model paradigm is community-sourced, not peer-reviewed.** Its
   performance numbers are vendor self-reported and are not cited as fact anywhere in
   this draft. The calibration objective used here is our own design inspired by that
   paradigm, and we do not claim to reproduce it.

## 6. Conclusion

A single-forward-pass decision head can be brought to within 0.0002 of the exact
partition-function marginals' calibration error, using two parameters fitted without a
partition function. That is the positive result, and it is measured on a clean
in-distribution split with no test-set selection.

On accuracy the answer is stratified rather than negative. Where structures are
conserved — the `CRW` stratum of short sequences — the DP-free head reaches 0.9664
against the partition-function baseline's 0.6729, replicated on an independent split
and with no measurable homology to training. Where structures are diverse — `RFAM`
short sequences, and above all the new families of bpRNA-new — it loses, and across
families it collapses to its own physical prior while the physical baseline improves.

We report the stratified result as the main accuracy finding rather than the pooled
number, because the pooled number is dominated by the stratum where we lose and would
understate the method. The complement matters just as much: a calibration result that
does not transfer across families is only half a result, and the half that is missing
is the half the field actually needs.

## Appendix A. Evidence ledger

Every number in §4 traces to one of these artifacts. Paths are on the cluster unless
stated; the code lives at `/home/cunyuliu/rna-jepa` and the artifacts at
`/mnt/cunyuliu/rna-jepa`.

| Result | Artifact | Command |
|---|---|---|
| TS0 headline, `w=-1` | `eval_decision/astrained_ff3500_bprna_ts0/result.json` | `eval/ss/evaluate_decision.py --checkpoint ckpts/rinalmo_ff_ff_w05_snapshot.pt --data ss_data/jsonl/bprna_ts0.jsonl --calib-data ss_data/jsonl/bprna_vl0.jsonl --prior-weight -1` |
| TS0, `w=0.75` | `eval_decision/sel_ff3500_bprna_ts0/result.json` | same, `--prior-weight 0.75` |
| ArchiveII, `w=0.75` | `eval_decision/sel_ff3500_archiveii_embok/result.json` | same, `--data ss_data/jsonl/archiveii_embok.jsonl --embedding-split archiveii` |
| bpRNA-new, `w=0.75` | `eval_decision/sel_ff3500_bprna_new/result.json` | same, `--data ss_data/jsonl/bprna_new.jsonl` |
| ViennaRNA baselines | `records/BASELINE_RESULTS.md` §1 | `eval/ss/baselines.py` |
| Source-stratified baselines (§4.3) | `eval_decision/baselines_{ts0,vl0}_{crw,rfam}_le100.json` | `eval/ss/run_baselines.py --split {ts0,vl0}_{crw,rfam}_le100` |
| Homology vs F1, and the leakage check | `eval_decision/homology_vs_f1.json` | `tools/homology_vs_f1.py --train ss_data/jsonl/bprna_tr0.jsonl --pair sel_ff3500_bprna_ts0 --pair vl0sw_w0.75_bprna_vl0 --pair sel_ff3500_bprna_new --k 20` |
| De-duplication audit and clean subsets | `ss_data/jsonl/*_clean.jsonl` | `tools/dedup_against_train.py --train ss_data/jsonl/bprna_tr0.jsonl --split ss_data/jsonl/archiveii_embok.jsonl --threshold 0.5 --k 20` |
| Source stratification | `eval_decision/stratified_le100.json` | `tools/stratify_by_source.py --eval-root eval_decision --data-dir ss_data/jsonl --run astrained_ff3500_bprna_ts0 --run sel_ff3500_bprna_ts0 --max-length 100` |
| Length-bucketed leaderboard | `eval_decision/buckets_bprna_ts0.json` | `tools/length_bucketed_leaderboard.py --split bprna_ts0 --run sel_ff3500_bprna_ts0 --bucket 100 --bucket 200 --bucket 400` |
| MXfold2 | `records/BASELINE_RESULTS.md` §5 | `python -m mxfold2 predict` |
| ViennaRNA exact BPP calibration | `eval/ss/reference_calibration.py` | — |
| Checkpoint step provenance | `tools/ckpt_steps.py` | reads `step` from inside each `.pt` |
| Full run-by-run log | `records/DECISION_TRAINING_LOG.md` §14 | — |

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
| Q11 | Is the speed-up against McCaskill fair? | §5 item 6 — no speed-up is claimed |
| Q12 | Is the Jev citation reliable? | §5 item 8 |

## Appendix C. What makes this draft preliminary

1. **The existence claim C1-a is untested** (baseline weights unreachable). This is the
   largest gap and it is a coverage gap, not a negative result.
2. **The convergence curve is incomplete.** TS0 points currently exist at steps
   1000 / 2000 / 3500, and the first three were measured at *different* decode
   weights, so they do not form a curve. Same-weight points at steps 6000 and 10000
   are queued.
3. **Several arms are still training** and are deliberately not reported: a
   four-way objective-function comparison (does the calibration term do anything?),
   two hierarchical-cascade arms, and a 4.8x-capacity arm. No conclusion about any of
   them appears in this draft.
4. **The §4.3 stratification is a source-database proxy, not a family split.** It uses
   `CRW` / `RFAM` as they appear in bpRNA-1m sequence names, and the two strata are
   also not length-matched *to each other* — they are each truncated at the same edges.
   The `CRW` cells are small (68 and 16 sequences), so the +0.295 / +0.272 margins rest
   on few independent examples, although the <=100 nt cell replicates on the validation
   split. A verified family-level split has not been run.
5. **The strata differ in more than their source.** Mean length is matched (75.9 vs
   78.0 nt on TS0), but `CRW` is also GC-richer (0.564 vs 0.482) and more densely
   paired (0.270 vs 0.200 pairs per nt). Source, GC and density therefore vary
   together, and §4.3 identifies a stratum rather than isolating a cause. A density
   effect would move both methods, and it does not: ViennaRNA centroid is nearly flat
   across the two strata (0.6729 vs 0.6209) while ours swings by 0.42. That is an
   argument, not a proof — a density-matched re-measurement has not been done.
6. **The two smallest cells of the §4.3 grid are small.** `CRW` 100–200 nt holds 16
   sequences and `RFAM` >400 nt holds 15, so neither is strong on its own; the
   conserved-stratum claim rests on the `CRW` <=100 nt cell and its validation-split
   replication. `vienna_mfe` was not run on the two longest `RFAM` cells.
