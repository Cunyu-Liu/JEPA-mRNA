# Phase 0B direct-prior-work audit: Global Ensemble Context

**Date:** 2026-11-24
**Repository state audited:** `ab9bb10d58380d86b6eb5c1051b7209a08d4e3fa` (`foldnar-e0-20261118`)
**Scope:** direct prior work for the proposed second-pass pair update; no model experiment, code change, registry change, or external-test development.
**Verdict status:** bounded primary-source audit, not an exhaustive systematic review or a patentability opinion.

## 1. Decision

**No single audited work is directly isomorphic to the full proposed computation. However, every major component has direct precedent, and a broad claim of “ensemble feedback,” “global context,” “second-pass refinement,” or “stop-gradient iterative prediction” is not defensible as new.**

The closest precedents partition the idea as follows:

- McCaskill/ViennaRNA, CONTRAfold, and the current FoldNar NLL already supply exact noncrossing ensemble marginals and structured training.
- ProbScan already computes higher-order loop/helix posterior quantities from inside/outside tables.
- TurboFold/TurboFold II already reinject base-pair probabilities as pseudo-free-energy terms and recompute partition functions, but use homolog-derived extrinsic information rather than a learned same-sequence residual head.
- SPENs and E2Efold already learn energy/constraint-aware iterative structured predictors, but not exact noncrossing posterior-feature feedback.
- RNAformer already recycles pair latents with stop-gradient, but recycles opaque latents rather than calibrated ensemble sufficient statistics.

The **precise defensible increment** is therefore only this combination:

> Within the audited literature, FoldNar proposes a single-sequence, representation-level self-ensemble feedback layer in which stopped exact noncrossing pair marginals and explicitly defined posterior interval/competition statistics condition a learned residual pair potential, with the resulting second-pass potential trained by the same exact structured NLL.

This is a **candidate architectural combination**, not evidence of improvement or an unqualified first-in-literature claim. Stop-gradient is an experimental separation device, not the contribution itself. The current recurrence is a learned pair-additive Nussinov-style Gibbs model, not the full Turner nearest-neighbor thermodynamic ensemble; call its outputs “model ensemble marginals,” not physical equilibrium probabilities without separate calibration evidence.

## 2. Exact object being compared

The live implementation currently constructs one pair-score matrix from

\[
z_{ij}=W[h_i+h_j;\ h_i\odot h_j;\ |h_i-h_j|],
\]

with an optional local 2D scorer and Turner-stacking residual (`src/rnajepa/decision_head.py:130-177,503-565`). Its exact noncrossing inside/outside implementation computes

\[
Z(S)=\sum_{y\in\mathcal Y_{\rm NC}}\exp\!\left(\sum_{(i,j)\in y}S_{ij}\right),
\qquad P_{ij}=\frac{\partial\log Z(S)}{\partial S_{ij}},
\]

and the existing NLL uses the analytic gradient \(P-y\) (`src/rnajepa/harness.py:474-655,680-785`; `src/rnajepa/rlcd.py:374-436`). `src/rnajepa/structured_marginal.py` adds a supervised marginal loss, but no pair representation currently consumes posterior or interval tensors. The historical hierarchical cascade is not GEC.

The proposal is

\[
(P^0,M^0)=\operatorname{InsideOutside}(S^0),\qquad
S^1_{ij}=S^0_{ij}+\Delta_\theta(H_{ij},\operatorname{sg}P^0_{ij},\operatorname{sg}M^0_{ij},g_{ij}),
\]

followed by the same structured NLL on \(S^1\). Stop-gradient blocks only the posterior-input path: **it does not freeze \(S^0\)** because \(S^0\) also enters \(S^1\) additively and through \(H\). Final-NLL gradients can still update the first-pass head through those paths.

Pre-registered \(M\) channels must have posterior semantics. For example, inside pair mass \(A_{ij}=\sum_{i\le k<l\le j}P^0_{kl}\), shared-endpoint competition, and crossing-alternative mass are **expected counts**, not probabilities of a union. They need not be bounded by one and must not be clipped or calibrated as probabilities. ProbScan-style loop probabilities are specific posterior events and require their own recurrences; arbitrary sums of \(P\) must not be relabeled loop probabilities.

## 3. Work-by-work audit

“Direct control” means the comparison that isolates the transferable object; it does not mandate reproducing every external system in the minimal pilot.

### A. RNA ensembles, structured folding, and posterior feedback

| Work and primary sources | Transferable mathematical object | Difference from current/proposed implementation | Direct control | Non-transferable protocol |
|---|---|---|---|---|
| **McCaskill (1990), “The equilibrium partition function and base pair binding probabilities for RNA secondary structure,” [DOI](https://doi.org/10.1002/bip.360290621); ViennaRNA 2.0 (Lorenz et al., 2011), [DOI](https://doi.org/10.1186/1748-7188-6-26), [official implementation](https://github.com/ViennaRNA/ViennaRNA)** | Boltzmann partition function over noncrossing structures and \(P_{ij}=Z^{-1}\sum_{y\ni(i,j)}e^{-E(y)/RT}\), computed by inside/outside DP. | FoldNar already has the learned-score analogue and uses \(P-y\) for NLL. ViennaRNA computes posteriors/decoding but does not feed them into a learned second-pass potential. | Existing \(S^1=S^0\) structured-NLL control plus a decode-only posterior/MEA arm; any gain must require residual representation feedback, not merely posterior decoding. | Turner energies, temperature calibration, MFE/MEA accuracy, and ViennaRNA runtime cannot be transferred to a learned Nussinov ensemble; no physical-ensemble or long-sequence claim follows. |
| **CONTRAfold (Do, Woods, Batzoglou, 2006), [DOI](https://doi.org/10.1093/bioinformatics/btl246), [official project](https://contra.stanford.edu/contrafold/)** | Conditional log-linear noncrossing grammar; discriminative likelihood training uses posterior expected feature counts from inside/outside. | This is the direct precedent for the existing structured NLL, not for posterior features as inputs to another pair scorer. | Same exact-NLL endpoint with the residual disabled is the essential anti-“renamed CRF” control. | Its grammar, feature counts, maximum-likelihood parameter estimation, and benchmark scores do not establish a new FoldNar contribution; do not substitute marginal BCE for the frozen NLL in the minimal test. |
| **ContextFold (Zakov et al., 2011), “Rich parameterization improves RNA structure prediction,” [DOI](https://doi.org/10.1089/cmb.2011.0184)** | Rich loop-decomposable scoring features, including local structural contexts, still compatible with folding DP. | GEC’s \(g_{ij}\) and local motif features belong to this scoring-side tradition; posterior \(P,M\) are global ensemble expectations rather than fixed loop-context parameters. | A parameter/compute-matched local loop/motif/span context arm with no \(P,M\), keeping the same NLL and decoder. | Rich parameter counts and ContextFold’s sequence-wise results are not evidence of cross-family generalization; do not import its train/test protocol or call fixed loop context “ensemble context.” |
| **Sloma & Mathews (2016), exact loop formation probabilities / ProbScan, [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5113201/), [DOI](https://doi.org/10.1261/rna.053694.115), [RNAstructure](https://rna.urmc.rochester.edu/)** | \(P(i,j)=V_{\rm interior}(i,j)V_{\rm exterior}(i,j)/Q\), and analogous posterior probabilities for hairpins, internal/bulge/multibranch loops, stacks, and helices. | This directly anticipates higher-order interval/loop ensemble statistics. ProbScan reports them for motif analysis; it does not learn a residual pair energy from them. | Factor the main arm into **P-only**, **M-only**, and **P+M**, with matched residual capacity. M-only must beat local and shuffled-M controls to support an interval-specific increment. | Motif-enrichment thresholds are not pair/stem F1 evidence; a loop event is not a sum of independent pair events, and joint loop probabilities are not obtained for free. Do not claim pseudoknot support. |
| **TurboFold (Harmanci et al., 2011), [DOI](https://doi.org/10.1186/1471-2105-12-108); TurboFold II (Tan et al., 2017), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5714223/), [DOI](https://doi.org/10.1093/nar/gkx815), [RNAstructure](https://rna.urmc.rochester.edu/)** | Iteratively map posterior pair probabilities to extrinsic pair proclivities, insert them as pseudo-free-energy terms, and recompute partition functions. TurboFold II also updates alignment posteriors. | **Closest conceptual precedent.** Its new information comes from other homologs and alignment coincidence probabilities; FoldNar would use the same sequence’s stopped \(P,M\) and a learned residual head with structured NLL. TurboFold is not merely MEA decoding. | Add a fixed self-prior feedback arm \(\Delta_{ij}=\alpha\phi(P^0_{ij})\), with predeclared \(\phi,\alpha\) and no learned M, to separate generic probabilistic feedback from learned interval/competition refinement. Cite TurboFold explicitly in any related-work claim. | Homolog/MSA inputs, identity weighting, HMM coincidence probabilities, and family-level evaluation are a different information regime; introducing them would invalidate the single-sequence control and create additional leakage risk. |
| **LinearPartition (Zhang et al., 2020), [DOI](https://doi.org/10.1093/bioinformatics/btaa460), [official implementation](https://github.com/LinearFold/LinearPartition), audited commit [`b450fb3`](https://github.com/LinearFold/LinearPartition/tree/b450fb3e63189073b68d385589035f992080aa3a)** | Beam-pruned left-to-right inside/outside approximation to \(\log Z\) and pair probabilities; official code calls `outside` then `cal_PairProb`. | It changes posterior computation, not pair representation. It is relevant only as a later sparse/approximate GEC execution route. | After exact short-sequence GEC passes, compare exact versus beam context at matched candidate coverage; report omitted-pair recall, probability-mass error, runtime, memory, and final pair/stem metrics. | Beam search is not exact global DP. Never transfer its approximate linear-time claim to exact GEC, truncate 7224/19000-nt targets, or hide coverage loss behind speed. |
| **MXfold2 (Sato, Akiyama, Sakakibara, 2021), [paper](https://www.nature.com/articles/s41467-021-21194-4), [DOI](https://doi.org/10.1038/s41467-021-21194-4), [official implementation](https://github.com/mxfold/mxfold2), audited commit [`7ea8188`](https://github.com/mxfold/mxfold2/tree/7ea8188b35c76175beb60410c63aca8b2938eca5)** | Neural loop scores plus Turner parameters in one decomposable score \(f=f_T+f_W\), Zuker-style inference, structured hinge loss and thermodynamic regularization. Official `ZukerFold.make_param` constructs loop scores; `MixedFold` combines Turner and learned parameters. | MXfold2 integrates priors **before one DP**; GEC integrates the posterior of a first DP into a second learned score field. Neither Turner residuals nor structured training is new in FoldNar. | A Turner/local-loop-score arm and a one-pass structured-model arm, without changing the loss to MXfold2’s SSVM or adding thermodynamic regularization in the minimal pilot. | Do not mix max-margin training, IMLE, chemical-probing fine-tuning, or published TrainSetA/B results into the GEC causal comparison; family-wise evaluation discipline is transferable, its numerical results are not a same-protocol control. |

### B. Structured/energy-based learning and iterative reasoning

| Work and primary sources | Transferable mathematical object | Difference from current/proposed implementation | Direct control | Non-transferable protocol |
|---|---|---|---|---|
| **Structured Prediction Energy Networks (Belanger & McCallum, 2016), [paper and citation](https://proceedings.mlr.press/v48/belanger16.html)** | Learned global energy \(E(F(x),\bar y)\) over relaxed outputs, with iterative mirror-descent inference and loss-augmented SSVM training. | SPEN is the general precedent for output-dependent global energies. GEC instead computes an exact posterior under a tractable noncrossing pair-additive energy and uses selected posterior statistics as features for a new tractable energy. | Generic two-pass energy/context arm using \(\sigma(S^0)\) or an unconstrained global label feature, with the same residual capacity and stop rule; it must not match true \(P,M\) gains. | SPEN’s relaxed multi-label optimization, SSVM, local minima, and label-dependency results are not RNA posterior inference; do not claim its global-energy theorem or use its objective as an extra pilot change. |
| **E2Efold (Chen et al., ICLR 2020), “RNA Secondary Structure Prediction By Learning Unrolled Algorithms,” [paper](https://arxiv.org/abs/2002.05810), [official implementation](https://github.com/ml4bio/e2efold), audited commit [`2aab9e7`](https://github.com/ml4bio/e2efold/tree/2aab9e78326ddaa27d14c49eb34001c0003a1960)** | Unrolled primal-dual updates on a relaxed pairing matrix enforce symmetry, canonical/min-loop, and one-partner constraints; official `models.py` iterates learned update rules. | E2Efold refines a soft structure toward constraint satisfaction with a differentiable F1 loss. It computes neither a noncrossing partition function nor posterior interval features for a residual pair potential. | Constraint/post-processing-only and independent-sigmoid iterative controls, evaluated before and after the same MEA decoder, distinguish generic constraint propagation from ensemble-statistic feedback. | Pseudoknot-permitting relaxed outputs and surrogate F1 training are not interchangeable with exact noncrossing NLL; published cross-dataset results do not establish unseen-family generalization. |
| **Neural Algorithmic Reasoning (Veličković et al., 2021), [paper](https://arxiv.org/abs/2105.02761), [DOI](https://doi.org/10.1016/j.patter.2021.100273); CLRS benchmark, [paper](https://arxiv.org/abs/2205.15659), [official implementation](https://github.com/google-deepmind/clrs)** | Repeated processor state updates with algorithm-trajectory hints and OOD-size evaluation; CLRS explicitly exposes intermediate states and previous hidden state. | NAR motivates executing a known DP and learning a refinement processor, but generic recurrence has no ensemble semantics. FoldNar already executes exact DP; the question is whether its posterior output is useful input. | Matched generic recurrent/latent processor with equal passes and parameters, with and without algorithmic posterior inputs. Report long-length generalization separately from in-distribution gains. | CLRS synthetic trajectories, hints, teacher forcing, and size-OOD results cannot be imported as RNA evidence; more iterations alone do not demonstrate algorithmic reasoning. |
| **RNAformer (Franke et al., 2023), “Scalable Deep Learning for RNA Secondary Structure Prediction,” [paper](https://arxiv.org/abs/2307.10073), [official implementation](https://github.com/automl/RNAformer), audited commit [`196e7de`](https://github.com/automl/RNAformer/tree/196e7dea01e5446cfbcb9e9b594cd10a8b537372)** | Full pair-matrix row/column axial attention plus recycling. Official `RNAformer.py` applies `@torch.no_grad()` and detached pair latents for all but the last recycle. | **Direct precedent for stop-gradient pair-latent refinement and global pair propagation.** Its recycled tensor is learned latent state, not an exact posterior or normalized interval/competition statistic. | A stop-gradient latent-recycle arm with matched passes/approximate FLOPs is required to show that posterior semantics, rather than recycling alone, drive gains. | RNAformer’s extra data, model scaling, MFE-distillation experiment, BCE training, and TS0/Rfam results are separate variables; no stop-gradient or axial-attention novelty claim is permissible. |

### C. Recent RNA foundation/2D models and global propagation

| Work and primary sources | Transferable mathematical object | Difference from current/proposed implementation | Direct control | Non-transferable protocol |
|---|---|---|---|---|
| **RiNALMo (Penić et al., 2025), [paper](https://www.nature.com/articles/s41467-025-60872-5), [DOI](https://doi.org/10.1038/s41467-025-60872-5), [official implementation](https://github.com/lbcb-sci/RiNALMo), inspected tree commit [`2c2c5c1`](https://github.com/lbcb-sci/RiNALMo/tree/2c2c5c14a5ae609d8c560a5d9ca32e51e0288955)** | Long-range sequence self-attention, outer-concatenated pair embeddings, and a bottleneck 2D ResNet pair head. | RiNALMo supplies implicit global sequence context but its published secondary-structure head is trained as pairwise BCE; it has no explicit noncrossing posterior feedback. FoldNar must treat foundation embeddings as a controlled input, not as GEC. | Freeze the same RiNALMo backbone/checkpoint across arms; compare the existing pair head, latent recycle, and posterior feedback under the same structured NLL and unfreeze schedule. | Pretraining corpus, 1022-nt random crops, PDB/other transfer data, and pretraining exposure can confound OOD claims; published BCE/decoder metrics are not same-protocol GEC evidence. |
| **BPfold (Zhu et al., 2025), [paper](https://www.nature.com/articles/s41467-025-60048-1), [DOI](https://doi.org/10.1038/s41467-025-60048-1), [official implementation](https://github.com/heqin-zhu/BPfold), audited prediction code at commit [`d37d6aa`](https://github.com/heqin-zhu/BPfold/tree/d37d6aa10cbca13e590ff83917fc4d63fec2ddbc)** | Local base-pair motif energy maps fused into attention; pairwise BCE followed by hard physical refinement. Official `predict.py` postprocesses and calls `remove_lone_pairs`. | BPfold’s auxiliary energy is fixed local motif evidence and its global propagation is attention, not ensemble posterior feedback. Its isolated-pair deletion is especially relevant because GEC targets isolated pairs. | Local motif-energy/attention control and evaluation both before and after any isolated-pair rule. GEC must improve isolated/long-span pairs without deleting them post hoc. | Motif-library construction, ensemble averaging, positive-weight BCE, hard overlap rules, and lone-pair removal cannot be silently combined with GEC; published gains do not isolate representation feedback. |

## 4. Required minimal control closure

The original four arms (`matched_control`, `ensemble_context`, `destroy_interval_shuffle`, `independent_context`) are necessary but insufficient to separate all close precedents. Add the following **seed-0-only, matched-budget** comparisons before any multi-seed decision:

1. **P-only / M-only / P+M:** isolates pair-marginal feedback from ProbScan-like higher-order interval information.
2. **Stopped latent recycle:** matches RNAformer-style generic iterative computation and is the strongest architectural control.
3. **Fixed self-prior feedback:** a predeclared pseudo-energy transform of \(P^0\), with no learned \(M\), acknowledges TurboFold and tests whether learning interval/competition coordination adds anything.
4. **Local motif/loop context:** matches ContextFold/MXfold2/BPfold-style scoring-side information without a first posterior.
5. **Decode-only posterior:** verifies that the gain is not obtainable by using \(P^0\) in the existing MEA decoder without changing pair scores.

All arms need identical initialization, data order, optimizer steps, final structured NLL, stop-gradient rule where applicable, residual parameter/compute budget, and MEA decoder. Report raw and decoded metrics, but do not use raw AUPRC to rescue a pair/stem failure. No external test set, multi-seed launch, SRD/stacking/capacity route, or decoder-threshold search belongs to this audit’s follow-up.

## 5. Claims allowed and claims blocked

**Allowed now (prior-work statement only):** “The audit did not find a single audited method combining same-sequence stopped noncrossing posterior pair and interval/competition features with a learned residual pair potential trained by structured NLL. TurboFold, ProbScan, SPEN/E2Efold, and RNAformer are the closest component precedents.”

**Blocked until separately proven:**

- “First global/ensemble/context/iterative/stop-gradient RNA model.”
- “First inside–outside neural folder” or “new CRF/structured-NLL objective.”
- “New interval/loop posterior mathematics” or “new pseudo-energy feedback.”
- “Thermodynamically calibrated ensemble,” “exact long-sequence inference,” pseudoknot support, cross-family generalization, isolated-pair improvement, or SOTA.

## 6. Audit limits and implementation evidence

- The negative isomorphism conclusion is bounded to the primary works above and their cited official implementations; it is not proof that no unpublished or uncited method exists. A manuscript submission should run a fresh citation-forward search, especially around TurboFold, ProbScan, RNAformer, and posterior-feedback structured prediction.
- Live px2 inspection found no GEC implementation or GEC result. Current code and hashes: `src/rnajepa/harness.py` (exact inside/outside and NLL), `src/rnajepa/decision_head.py` SHA256 `37330a0f6e2000d17ea2d7dee1a802368b11ad96ae834517098ee66ae786e8e3`, `src/rnajepa/structured_marginal.py` SHA256 `df9033168cea3d285cab941f05634da9b25c92734d08903952809f037607c785`, and `src/rnajepa/rlcd.py` SHA256 `efce6c2df29e8c9ad68564670ff6e8625f5bcd67494f25430f0b3c57e5bb59cc`.
- Official implementation inspection was source-level, not a reproduction of published training. Commit-pinned code was read for LinearPartition, MXfold2, E2Efold, RNAformer, and BPfold; RiNALMo’s official tree and primary method description were inspected. No claim here depends on rerunning an external benchmark.
