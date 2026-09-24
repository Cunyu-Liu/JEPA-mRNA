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

Structure accuracy is **not** a win: our micro F1 on TS0 is **0.4953**, against
ViennaRNA centroid **0.5393** and MXfold2 **0.5651** measured on the same split with
the same metric implementation. Cross-family generalization is insufficient and is
reported as such: on bpRNA-new our micro F1 is **0.3094**, statistically
indistinguishable from our own Nussinov+Turner prior (**0.3015**), while ViennaRNA
centroid reaches **0.6770** on the same split.

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
with a measurement and the second with an honest negative.

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
  gap is 0.0006. The validation split is not a proxy for TS0 (the same checkpoint
  scores 0.9230 on VL0 versus 0.6188 on TS0 within the <=100 nt length bucket, for
  reasons we have not identified), so this check matters and it passes.
- **The result holds on the secondary split.** On ArchiveII (3,950 rows) the
  recalibrated gap is 0.0093, still inside the 0.02 threshold.

**Reference point.** ViennaRNA's exact base-pair probabilities on TS0 have
ECE **0.0048**. Our recalibrated head is at **0.0023** — the same order of magnitude,
obtained without a partition function.

### 4.3 Structure accuracy: we do not beat the physical baselines

| Split | Ours (micro F1) | ViennaRNA centroid | ViennaRNA mfe | MXfold2 | Nussinov+Turner prior |
|---|---|---|---|---|---|
| TS0 | **0.4953** | **0.5393** | 0.5222 | **0.5651** | 0.2124 |
| ArchiveII (3,950) | 0.5829 | 0.6207 | 0.5764 | — | 0.2010 |
| bpRNA-new (5,388) | **0.3094** | **0.6770** | 0.6379 | — | **0.3015** |

ViennaRNA and MXfold2 rows are our own measurements on our own split files with our own
metric implementation, so these comparisons are like-for-like within this table. They
are not comparable to F1 numbers quoted from other papers, which use different splits,
different redundancy thresholds and different aggregation conventions.

### 4.4 Cross-family generalization is insufficient (quantified)

On bpRNA-new the picture inverts. The physical baseline *improves* — ViennaRNA
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
   arguing it away.
3. **The raw head is not calibrated.** Only the two-parameter affine recalibration
   passes the gate, and the recalibration parameters come from a validation split
   whose relationship to the test split we do not fully understand (§4.2).
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

The same model does not beat the physical baselines on accuracy, and across families it
collapses to its own prior while the physical baselines improve. We report that as the
central limitation rather than as a footnote, because a calibration result that does not
transfer across families is only half a result — and the half that is missing is the
half the field actually needs.

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
| MXfold2 | `records/BASELINE_RESULTS.md` §5 | `python -m mxfold2 predict` |
| ViennaRNA exact BPP calibration | `eval/ss/reference_calibration.py` | — |
| Checkpoint step provenance | `ckpt_steps.py` | reads `step` from inside each `.pt` |
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
