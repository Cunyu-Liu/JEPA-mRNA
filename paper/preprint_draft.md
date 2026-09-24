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
marginals of the same model of **0.0002**, against a pre-registered threshold of
**0.02**, after fitting a two-parameter affine map on a disjoint validation split.
The map is applied at evaluation time and involves no partition function. The raw,
unrecalibrated head does **not** pass this gate (gap 0.1348), and we report both.

Structure accuracy is **not** a uniform loss, and the aggregate number is misleading in
both directions. Pooled over TS0 our micro F1 is **0.4953**, against ViennaRNA centroid
**0.5393** and MXfold2 **0.5651** on the same split. The comparison reverses once the
split is stratified, along two axes at once:

- **By source.** bpRNA-1m's two large source populations differ by roughly 0.4 F1 and
  appear in very different proportions in different splits. On the conserved `CRW`
  entries (rRNA/tRNA-derived) of at most 100 nt we reach **0.9664** where centroid
  reaches **0.6729** — a gain of 0.29, replicated on the independent validation split
  (**0.9709** vs 0.6702) with no measurable homology to training. On the diverse `RFAM`
  entries of the same length we reach **0.5490** against centroid's **0.6209**.
- **By length.** On TS0 we lead centroid only in the at-most-100 nt bucket
  (**0.6386** vs 0.6277) and trail in every longer bucket, by a margin that grows with
  length (−0.090 at 100–200 nt, −0.133 above 400 nt).

So the method is competitive on short, structurally conserved sequences and loses on
long, diverse ones. A single pooled number, in either direction, misstates it.

Cross-family generalization is insufficient and is reported as such: on bpRNA-new our
micro F1 is **0.3094**, statistically indistinguishable from our own Nussinov+Turner
prior (**0.3015**), while ViennaRNA centroid reaches **0.6770** on the same split.

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

Checkpoint: `rinalmo_ff` at step 3500 (step read from inside the checkpoint file, not
from its name). Decode weight is the model's own trained prior weight — no selection on
any test split.

| Quantity (TS0, 1,288 sequences) | Value |
|---|---|
| System-1 ECE, raw head | 0.1373 |
| Exact-marginal ECE (same model, same sequences) | 0.0012 |
| **C1-c gap, raw head** | **0.1348 — FAIL** (threshold 0.02) |
| **C1-c gap, DP-free affine recalibration** | **0.0002 — PASS** |
| System-1 ECE after recalibration | 0.0023 |
| Recalibration parameters | `a = 0.02263`, `b = -4.7386` (2 parameters, fitted on VL0) |

The gap is reported both ways on purpose. Reporting only the recalibrated row would
hide the fact that the raw head is over-confident by an order of magnitude; reporting
only the raw row would hide the fact that a two-parameter, partition-function-free map
closes the gap to two ten-thousandths.

Two sensitivity checks:

- **The result does not depend on any choice made on the validation split.** With a
  validation-selected decode weight of 0.75 instead of the model's own weight, TS0
  micro F1 is 0.4959 versus 0.4953 — a difference of 0.0006 — and the recalibrated
  gap is 0.0006. The validation split is not a proxy for TS0 (see §4.3), so this check
  matters and it passes.
- **The result holds on the secondary split, and the same insensitivity holds there.**
  On ArchiveII (3,950 rows) the recalibrated gap is **0.0044** at the model's own
  weight and 0.0093 at the validation-selected weight, with micro F1 0.5834 versus
  0.5829 — again a difference of 0.0005. Both are inside the 0.02 threshold.

Four combinations (two splits x two decode conventions) pass, and the decode
convention moves micro F1 by at most 0.0006 anywhere. The headline therefore involves
no selection on any split, which is what makes it reportable.

**Reference point.** ViennaRNA's exact base-pair probabilities on TS0 have
ECE **0.0048**. Our recalibrated head is at **0.0023** — the same order of magnitude,
obtained without a partition function.

### 4.3 Structure accuracy: the pooled number is misleading in both directions

Pooled, we do not beat the physical baselines:

| Split | Ours (micro F1) | ViennaRNA centroid | ViennaRNA mfe | MXfold2 | Nussinov+Turner prior |
|---|---|---|---|---|---|
| TS0 | **0.4953** | **0.5393** | 0.5222 | **0.5651** | 0.2124 |
| ArchiveII (3,950) | 0.5829 | 0.6207 | 0.5764 | — | 0.2010 |
| bpRNA-new (5,388) | **0.3094** | **0.6770** | 0.6379 | — | **0.3015** |

But TS0 is a mixture along **two** axes, and both matter.

**Axis 1, source.** bpRNA-1m sequence names carry their source database, and the two
large sources have very different structural character: `CRW` (Comparative RNA Web) is
dominated by rRNA and tRNA with conserved, canonical structures, while `RFAM` spans a
wide range of families. Their proportions differ sharply between splits — `CRW` is 6.0%
of TR0, 7.3% of TS0, and **50.5% of VL0**.

**Axis 2, length.** Per-length-bucket results are the protocol-mandated form and they
change the sign of the comparison at the short end:

| TS0 length bucket | n | **Ours** | ViennaRNA centroid | ViennaRNA mfe | Nussinov+Turner prior |
|---|---|---|---|---|---|
| <=100 nt | 580 | **0.6386** | 0.6277 | 0.5999 | 0.3084 |
| 100–200 nt | 516 | 0.4478 | **0.5375** | 0.5045 | 0.2002 |
| 200–400 nt | 164 | 0.4538 | **0.4690** | 0.4305 | 0.1572 |
| >400 nt | 28 | 0.3522 | **0.4854** | 0.4561 | 0.1465 |
| whole split | 1,288 | 0.4959 | **0.5393** | 0.5222 | 0.2124 |

We lead only in the shortest bucket (+0.011), and the deficit grows with length
(−0.090, −0.015, −0.133). Our own physical prior scores 0.15–0.31 in every bucket, so
the learned head contributes real discriminative power throughout — it simply does not
contribute enough as sequences get longer.

Stratifying by source *and* matching on length (at most 100 nt), same checkpoint, same
metric implementation:

| Split | Stratum | n | **Ours** | ViennaRNA centroid | ViennaRNA mfe | Nussinov+Turner prior |
|---|---|---|---|---|---|---|
| **TS0** | `CRW`, <=100 nt | 68 | **0.9664** | 0.6729 | 0.6413 | 0.4187 |
| **TS0** | `RFAM`, <=100 nt | 486 | 0.5490 | **0.6209** | 0.5913 | 0.2893 |
| **VL0** | `CRW`, <=100 nt | 81 | **0.9709** | 0.6702 | 0.6693 | 0.4365 |
| **VL0** | `RFAM`, <=100 nt | 43 | **0.7824** | 0.5393 | 0.5163 | 0.3203 |

Three things follow.

**First, on the conserved stratum we beat the partition-function baseline by a wide
margin: +0.29 on TS0 and +0.30 on VL0.** The two splits are independent and agree to
0.005, and our own physical prior sits at 0.42, so the learned head is supplying real
discriminative power rather than riding on physics. We checked the obvious way this
could be an artefact and it is not: 20-mer containment against the training split is
0.048 on TS0-`CRW` with a maximum of 0.33, **no sequence above 0.5**, and zero exact
matches. This is a genuine held-out result, and it is the strongest positive result in
this draft.

**Second, on the diverse stratum we lose by 0.07.** Pooling the two strata reproduces
the aggregate (TS0 at most 100 nt pools to 0.6386; VL0 to 0.9304), so the pooled
comparison is dominated by the larger `RFAM` stratum.

**Third, this explains an anomaly we could not previously account for.** The same
checkpoint scores 0.9304 on VL0's at-most-100 nt bucket and 0.6386 on TS0's, a gap of
0.29 that we had earlier suspected was leakage or an unidentified bug. It is
composition: VL0's short bucket is 65% `CRW`, TS0's is 12%. A homology explanation was
tested and rejected — the two splits have almost identical 20-mer containment to
training (0.0423 vs 0.0403) and VL0's F1 is flat across containment bins.

**Fourth, the source effect is a property of the benchmark, not of our backbone.** A
natural objection to the `CRW` result is that a frozen language-model representation
might simply be good at rRNA because rRNA is abundant in pretraining. We can test this
without training anything new, because the repository also holds a from-scratch arm
with a randomly initialised encoder and the same head and data. Both arms show the same
pattern, on the same sequences:

| Arm | Encoder | `CRW` <=100 nt (n=68) | `RFAM` <=100 nt (n=486) | `CRW` − `RFAM` |
|---|---|---|---|---|
| from scratch | randomly initialised | 0.8735 | 0.4105 | **+0.463** |
| frozen backbone | RiNALMo-giga | 0.9664 | 0.5490 | **+0.417** |

The random-encoder arm shows a *larger* `CRW` advantage than the frozen one, so the
stratification is a property of the data rather than of the representation. The frozen
backbone adds +0.093 on `CRW` and +0.139 on `RFAM` — it helps both strata, and its value
is a separate question from the `CRW`/`RFAM` split.

`CRW` and `RFAM` are **source-database labels, not verified family labels**; sequence
names are unique within each split, so no family field is recoverable from them. This
stratification is a reproducible proxy that correlates with molecule type and
structural conservation, and it is not a substitute for a family-level split, which we
have not done.

The two axes are also **not separated**: the length buckets above are cut on length
alone, so their source composition is unmatched, and the source strata are cut on
source alone at a fixed length. Isolating a length effect from a source effect requires
a joint length x source grid, which we have not built. Both axes are real; neither is
attributed on its own.

ViennaRNA and MXfold2 rows are our own measurements on our own split files with our own
metric implementation, so these comparisons are like-for-like within this table. They
are not comparable to F1 numbers quoted from other papers, which use different splits,
different redundancy thresholds and different aggregation conventions.

### 4.4 Cross-family generalization is insufficient (quantified)

§4.3 shows the model is strong where structures are conserved and weak where they are
diverse. bpRNA-new is the extreme of the latter: it is built from *new* families by
construction. There the picture inverts. The physical baseline *improves* — ViennaRNA
centroid goes from 0.5393 on TS0 to **0.6770** — while our model falls from 0.4953 to
**0.3094**, and 0.3094 is indistinguishable from our own Nussinov+Turner prior at
**0.3015**. In other words, across families the learned head contributes essentially
nothing beyond the physical prior it started from.

We checked the obvious explanations and they do not account for it:

- **Not a decode-weight artefact.** Every length bucket loses both precision and
  recall; it is not one region of the score matrix being mis-weighted.
- **Not exact overlap.** Exact sequence overlap between bpRNA-new and TR0 is zero.
- **Not measured homology.** 20-mer containment is 0.000 for bpRNA-new against TR0,
  which is why it is our OOD split of choice.

The mechanism is not established. Our working hypothesis is that the frozen
representation transfers but the family-specific structural motifs learned by the head
do not, whereas the physical energy model is family-agnostic. Testing that requires a
family-level train/validation split that we have not yet run, and we therefore state
the hypothesis as a hypothesis.

### 4.5 Two dataset caveats that constrain what may be claimed

1. **ArchiveII is not an out-of-distribution split.** 26.7% of its sequences are
   near-duplicates of the training split under a 20-mer containment criterion
   (>0.9). Its 0.5829 is therefore *not* evidence of generalization and must not be
   placed next to published ArchiveII numbers as if it were.
2. **ArchiveII can only be scored on 3,950 of 3,966 rows.** RiNALMo's `max_pos` is
   1024, so 16 rows have no cached embedding. The evaluator refuses to silently mix
   frozen and freshly computed representations, which is the correct behaviour; the
   cost is that any ArchiveII number we report must carry the 3,950/3,966 qualifier.

## 5. Limitations

1. **The existence claim is untested.** C1-a asks whether our probabilities are at
   least as well calibrated as SPOT-RNA / UFold sigmoid outputs. Their weights are
   hosted on Dropbox, Google Drive and NihaoCloud, none of which is reachable from our
   cluster; Zenodo is also unreachable. The claim is therefore *not made*, not *failed*.
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
5. **The strongest learning-based baseline we could run is MXfold2.** Its training set
   is bundled with the package and we have not verified its overlap with TS0, so its
   0.5651 is quoted with that caveat.
6. **Speed claims are withheld.** The decode path used in every number above is an
   exact `O(L^3)` dynamic program, so the "DP-free" property currently refers only to
   the absence of the partition function, not to a wall-clock advantage. No speed-up
   ratio is claimed anywhere in this draft.
7. **Single seed for the headline.** The headline checkpoint is one seed. Multi-seed
   arms are training; no seed-variance statement is made here.
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
   also not length-matched *to each other* — they are each truncated at 100 nt. The
   `CRW` strata are small (68 and 81 sequences), so the +0.29 / +0.30 margins rest on
   few independent examples even though the two splits agree. A verified family-level
   split has not been run.
5. **The strata differ in more than their source.** Mean length is matched (75.9 vs
   78.0 nt on TS0), but `CRW` is also GC-richer (0.564 vs 0.482) and more densely
   paired (0.270 vs 0.200 pairs per nt). Source, GC and density therefore vary
   together, and §4.3 identifies a stratum rather than isolating a cause. A density
   effect would move both methods, and it does not: ViennaRNA centroid is nearly flat
   across the two strata (0.6729 vs 0.6209) while ours swings by 0.42. That is an
   argument, not a proof — a density-matched re-measurement has not been done.
