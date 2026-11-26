# Pair-Hamming Structured Max-Margin Candidate Contract

- `contract_id`: `foldnar_pair_hamming_structured_margin_candidate_v1`
- `base_head`: `82f6b4ef6f948c40aba0a53f73a31e1515af3efb`
- `branch`: `structured-margin-candidate-20261126`
- `status`: `FROZEN_BEFORE_IMPLEMENTATION_FOR_TWO_STEP_ENGINEERING_SMOKE_ONLY`
- `scientific_training_authorized`: false; seed 0 only. Seeds 1/2 and full training are forbidden.

## Exact objective

The legal set is the existing canonical Watson-Crick/wobble mask intersected with strict upper `i<j` and `j-i>3`. Targets must be legal non-crossing partial matchings. Pair-Hamming cost is `Delta(y,y*) = sum_(i<j in mask) 1[y_ij != y*_ij]`. Illegal, lower-triangle, non-canonical and min-loop-violating entries contribute no score, cost or gradient.

Optimize only `L_margin = (max_y [sum_(i,j in y)(s_ij + 1 - 2y*_ij)] + |y*| - sum_(i,j in y*) s_ij) / L`, over exact legal non-crossing matchings. This is not marginal BCE, independent-sigmoid supervision, sampling or a soft approximation. `lambda_nll=0` for margin arms; existing inside-outside NLL code and its historical config are unchanged. DP and target scoring cast model scores to float64. Deterministic max-product traceback gives the Danskin subgradient `(y_hat-y*)/L`; ties use existing branch order, while exhaustive gradient tests require a unique maximizer. `normalization=length` is frozen. No margin scale, decoder temperature/gamma/threshold or other hyperparameter search is allowed.

## Four arms and checkpoint boundary

Reuse unchanged `matched_control`, `ensemble_context`, `destroy_interval_shuffle`, `independent_context`, their seed0 initialization, data/ID hashes, batch stream, B=4, optimizer, clipping, dropout and terminal 20/33 schedule from `gec_four_arm_convergence_v1.json`. The new objective has a separate config and artifact namespace. NLL checkpoints/resumes cannot be reused or relabeled.

Margin validation every 500 steps is the unweighted sequence mean of length-normalized pair-Hamming structured margin. Only post-terminal-unfreeze checkpoints are eligible; min_delta=1e-6, patience=5, restore best eligible. Stop reason: `CONVERGED_VALIDATION_PAIR_HAMMING_STRUCTURED_MARGIN_AFTER_TERMINAL_UNFREEZE`. Margin validation values are not numerically comparable with historical NLL and cannot justify same-step/same-checkpoint selection. Cross-objective comparison requires separate convergence under each objective, then the same frozen evaluation protocol and fixed MEA (`temperature=1`, `gamma=1`, no threshold search). No scientific claim is authorized here.

## Verification and smoke limit

Before CUDA smoke: exhaustive short-sequence loss/gradient enumeration; canonical/min-loop/mask and invalid-target checks; deterministic ties; CPU/CUDA value/gradient agreement; float16/bfloat16/float32/float64 and autocast boundaries; finite extreme/no-pair behavior; unchanged NLL regressions. Margin DP and gradients stay on the requested device with no NumPy/CPU fallback.

Run exactly two optimizer steps per context arm on the existing tiny legal fixture, including real interruption/resume at step one. Artifact root: `/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_smoke_v1/generation_01`. No pilot/convergence run, decoder search, external-test selection or seeds 1/2.

## Full-training resource gate

Full-ready requires a clean pushed commit, all numerical/device/AMP tests and four-arm smoke passing, explicit contract/config/code hashes, plus a later worst-case B=4/L=500/terminal-20 forward+backward memory/runtime measurement. Tiny-smoke memory is not full-training evidence. Launch only when free memory covers measured peak times 1.5 plus max(4 GiB,20% GPU) reserve; otherwise run one arm or wait, never kill existing jobs. Full training remains frozen pending separate approval.
