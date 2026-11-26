# Pair-Hamming Structured Max-Margin Candidate Contract

- `contract_id`: `foldnar_pair_hamming_structured_margin_candidate_v1`
- `base_head`: `82f6b4ef6f948c40aba0a53f73a31e1515af3efb`
- `branch`: `structured-margin-candidate-20261126`
- `status`: `AUTHORIZED_SEED0_FULL_CONVERGENCE_WITH_RESOURCE_GATE`
- `scientific_training_authorized`: true; seed 0 only, all four arms to their own convergence. Seeds 1/2 and decoder search remain forbidden. Numerical/contract review is accepted; historical NLL commit `ecc40d3` is a `QUALIFIED_STRUCTURE_REFERENCE`, not a margin objective or checkpoint source.

## Exact objective

The legal set is the existing canonical Watson-Crick/wobble mask intersected with strict upper `i<j` and `j-i>3`. Targets must be legal non-crossing partial matchings. Pair-Hamming cost is `Delta(y,y*) = sum_(i<j in mask) 1[y_ij != y*_ij]`. Illegal, lower-triangle, non-canonical and min-loop-violating entries contribute no score, cost or gradient.

Optimize only `L_margin = (max_y [sum_(i,j in y)(s_ij + 1 - 2y*_ij)] + |y*| - sum_(i,j in y*) s_ij) / L`, over exact legal non-crossing matchings. This is not marginal BCE, independent-sigmoid supervision, sampling or a soft approximation. `lambda_nll=0` for margin arms; existing inside-outside NLL code and its historical config are unchanged. DP and target scoring cast model scores to float64. Deterministic max-product traceback gives the Danskin subgradient `(y_hat-y*)/L`; ties use existing branch order, while exhaustive gradient tests require a unique maximizer. `normalization=length` is frozen. No margin scale, decoder temperature/gamma/threshold or other hyperparameter search is allowed.

## Four arms and checkpoint boundary

Reuse unchanged `matched_control`, `ensemble_context`, `destroy_interval_shuffle`, `independent_context`, their seed0 initialization, data/ID hashes, batch stream, B=4, optimizer, clipping, dropout and terminal 20/33 schedule from `gec_four_arm_convergence_v1.json`. The new objective has a separate config and artifact namespace. NLL checkpoints/resumes cannot be reused or relabeled.

Margin validation every 500 steps is the unweighted sequence mean of length-normalized pair-Hamming structured margin. Only post-terminal-unfreeze checkpoints are eligible; min_delta=1e-6, patience=5, restore best eligible. Stop reason: `CONVERGED_VALIDATION_PAIR_HAMMING_STRUCTURED_MARGIN_AFTER_TERMINAL_UNFREEZE`. Margin validation values are not numerically comparable with historical NLL and cannot justify same-step/same-checkpoint selection. Cross-objective comparison requires separate convergence under each objective, then the same frozen evaluation protocol and fixed MEA (`temperature=1`, `gamma=1`, no threshold search). Training authorization does not authorize a scientific conclusion before all four runs have converged and passed independent artifact review.

## Verification and smoke evidence

Before CUDA smoke: exhaustive short-sequence loss/gradient enumeration; canonical/min-loop/mask and invalid-target checks; deterministic ties; CPU/CUDA value/gradient agreement; float16/bfloat16/float32/float64 and autocast boundaries; finite extreme/no-pair behavior; unchanged NLL regressions. Margin DP and gradients stay on the requested device with no NumPy/CPU fallback.

The completed two-step smoke for every context arm used the tiny legal fixture with real interruption/resume at step one. Its frozen verification artifact is `/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_smoke_v1/generation_01/smoke_verification_v1.json`, SHA256 `a59f603f026a52af55104885ec23955d06eb28487373e96b6fbfee1f108ffa77`. Full convergence runs use a separate artifact namespace; no decoder search, external-test selection or seeds 1/2.

## Full-training resource gate

Full-ready requires a clean pushed commit, all numerical/device/AMP tests and the pinned four-arm smoke artifact, explicit contract/config/code hashes, a current authorization artifact, plus the worst-case B=4/L=500/terminal-20 forward+backward memory/runtime measurement bound to the same code. Tiny-smoke memory is not full-training evidence. The scheduler launches at most two arms: two only when current free memory covers `ceil(1.5 * 2 * measured_peak + max(4 GiB,20% GPU total))`, otherwise one when the single-arm threshold passes, otherwise a persistent queue. The peak basis is the larger of PyTorch peak allocated and peak reserved. Never kill existing jobs or lower the gate. Each arm must retain its launch identity, logs, checkpoints and convergence stop reason.
