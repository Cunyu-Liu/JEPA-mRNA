# DP-Free Calibration and a Hierarchical Decision Cascade for RNA Secondary Structure Prediction

> **Sample manuscript stub.** This file is a *scaffold*, not a draft: it exists so that
> `paper/check_manuscript.py` has a well-formed target to lint, and so the shape of a
> compliant manuscript is visible. All numbers are placeholders (`待核验`) and must be
> replaced from measured runs. Nothing here may be quoted as a result.

## Abstract

RNA secondary structure prediction faces a trust-versus-cost dilemma. Physical models
yield calibrated base-pair probabilities but require an `O(L^3)` partition function,
while deep models are cheap yet emit uncalibrated, unconstrained and cross-family-fragile
probabilities. We show that a single-forward-pass decision head can be trained to be as
calibrated as the exact partition-function marginals, so that inference skips the
`O(L^3)` partition function entirely. The resulting calibrated confidence then acts as a
compute-allocation controller, so exact physics is invoked only where the model is
uncertain. On the cross-family benchmark our method shows **smaller OOD degradation**
than the baselines, and it is **more robust** under distribution shift. All headline
numbers are `待核验` pending baseline reproduction.

## 1. Introduction

Current methods are either accurate but slow (the McCaskill / CONTRAfold lineage needs an
`O(L^3)` partition function before it can report a probability) or fast but untrustworthy
(discriminative contact-map models emit uncalibrated sigmoid outputs, need heuristic
post-processing to become legal structures, and degrade badly across families). Two
questions are therefore unanswered: *can this prediction be trusted?* and *when is it
worth paying for `O(L^3)`?*

## 2. Related work

The closest prior work is CONTRAfold, which already trains a log-linear model on the
non-crossing structure space with an exact partition function. This work sits in the
CONTRAfold / CRF lineage; the log-linear CRF framework and the partition function are not
our contribution. We also compare against CDPFold, which likewise predicts a pair
probability matrix without dynamic programming, and against LinearPartition and E2Efold.

## 3. Contributions

We make two contributions and report two supporting results.

**C1 -- DP-free calibration.** We show that the pair probabilities emitted by a
single-forward-pass decision head can be trained to be as calibrated as the exact
marginals of the partition function, so that inference no longer computes the partition
function at all.

**C2 -- Hierarchical decision cascade.** We introduce a three-level cascade that replaces
the `O(L^2)` base-pair decision space with `O(L)` helix-level decisions plus local
refinement, while keeping long-range pairs representable.

**C3 -- Compute-adaptive folding (supporting).** We turn the calibrated confidence into a
compute-allocation controller that upgrades only low-confidence regions to exact physics.

**C4 -- Calibration-first evaluation protocol (supporting).** We report expected
calibration error, reliability diagrams, marginal calibration, posterior log-likelihood
and the legalisation cost as first-class metrics for this task.

## 4. Limitations

1. Jev has no peer-reviewed paper; its performance numbers are vendor self-reported, and
   we cite only its paradigm claims.
2. Jev's RLCD algorithm is undisclosed. Our calibration objective is an RLCD-inspired,
   self-designed objective, and we do not claim to reproduce or match it.
3. The Gibbs / partition-function framework is not our contribution; it belongs to the
   CONTRAfold / CRF lineage.
4. We do not claim that our algorithmic complexity beats LinearFold's `O(L)`.
5. For the 23 inherited downstream tasks whose dev and test splits are identical, the
   reported scores are development scores and are flagged as such.

## 5. Anticipated reviewer objections (Q1-Q12)

| # | Objection | Response strategy | Evidence | Landing point in manuscript |
|---|---|---|---|---|
| **Q1** | This is CONTRAfold with a neural scorer: incremental. | Concede the lineage; narrow the increment to DP-free calibration and compute-adaptive folding. | Head-to-head vs CONTRAfold; S7 delta <= 0.02; S6 matched-FLOPs curve | §2, paragraph 1; §3 C1; §4 Table 3 |
| **Q2** | Without a partition function, why is the probability calibrated? | Quantify the gap rather than assert equivalence. | S7; reliability diagrams; posterior log-likelihood | §3 C1; §4 Fig. 2 |
| **Q3** | Base pairs are correlated, so pair-level ECE is meaningless. | Report marginal calibration, add structure-level calibration, and state the limitation explicitly. | Structure-level calibration curve; correlation summary | §4.4; §4 Limitations item 3 |
| **Q4** | Nussinov + stacking is not ViennaRNA; the prior is too weak. | Concede; use "Nussinov + Turner stacking" as the baseline and never call it ViennaRNA. | Ablation of the learned residual; gap analysis | §3.2; §4 Table 5 |
| **Q5** | `O(L^2)` memory is infeasible for full-length transcripts. | Chunking, gradient checkpointing and banded sparsification; report measured memory at `L=2048`. | Memory measurement; banded-vs-full ablation | §4.5; §4 Table 6 |
| **Q6** | Why not simply fine-tune RiNALMo / RNA-FM? | The contribution is the head and the objective, which are backbone-orthogonal; ablate the backbone. | Backbone ablation | §4.6 |
| **Q7** | What about pseudoknots? | Scope statement: the framework depends on non-crossingness; only a restricted (H-type) class is covered. | Restricted-class results plus scope statement | §4.7; §4 Limitations |
| **Q8** | The cotranscriptional order is a trivial observation. | Concede outright: reported as an ablation only, with no contribution claim. | H4 negative result | §4.8 |
| **Q9** | bpRNA-1m is redundant and may contaminate the test set. | Dual identity thresholds (80% / 90%) plus family-level splitting and contamination detection; report both raw and de-duplicated versions. | Attrition table; contamination rate | §4.1; §7 |
| **Q10** | Anyone can get a zero illegal-structure rate. | Concede; report the legalisation cost (the F1 gap forced by legality) instead. | Corrected H1 experiment | §4.9 |
| **Q11** | Is the 50x speed-up against McCaskill a fair comparison? | System-1 does not compute the partition function; report the speed-up against McCaskill and against LinearPartition separately. | Both speed-up ratios; aligned mean FLOPs | §4.10; §4 Table 7 |
| **Q12** | Is the Jev citation reliable? | State that Jev has no peer-reviewed paper, that the numbers are vendor self-reported, and that only paradigm claims are cited. | Evidence-level table; limitations | §1; §4 Limitations items 1-2 |
