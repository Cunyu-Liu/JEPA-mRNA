# FoldNAR Structured-Margin Independent Scientific and Contract Review

- `review_branch`: `structured-margin-contract-review-20261126`
- `base_commit`: `92f0bc2e6440bcbfc682d2f8ca3e502ec86a6b48`
- `review_scope`: numerical objective, gradient, legal mask, normalization, model selection, convergence, checkpoint state, four-arm scientific sufficiency
- `non_actions`: no full training, no seeds 1/2, no decoder search, no training-semantic change, no existing-process interruption
- `verdict`: **numerical implementation PASS; two-step integration evidence PASS; FULL CONTRACT FAIL / NOT READY**

## 1. Exact objective and gradient

Let `Y` be all non-crossing partial matchings on the canonical legal upper-triangle set `A = {(i,j): i<j, j-i>3, mask_ij=true}`. For `y,y* in Y`, pair Hamming is

`Delta(y,y*) = sum_(i,j in A) |y_ij-y*_ij| = sum_(i,j in A) (1-2y*_ij)y_ij + |y*|`.

Therefore the implemented loss is exactly

`L = [max_(y in Y) {sum_(i,j in y)s_ij + Delta(y,y*)} - sum_(i,j in y*)s_ij]/L_seq`

`= [max_(y in Y) {sum_(i,j in y)(s_ij+1-2y*_ij)} + |y*| - sum_(i,j in y*)s_ij]/L_seq`.

`src/rnajepa/structured_margin.py:150-158` implements these terms without dropping the target-count constant. The loss is nonnegative and zero at a loss-augmented maximizer equal to the target. Length normalization divides both value and gradient by the same sequence length; the trainer then uses an unweighted sequence mean for both training batches and validation, so no hidden pair-count or batch-length weighting is introduced.

The inside recurrence at `src/rnajepa/structured_margin.py:18-57` considers unpaired `j`, unpaired `i`, an `(i,j)` pair, and every `(k,j)` pair with disjoint left/right intervals. It exactly enumerates the non-crossing matching semiring for max-product. Backtrace at `src/rnajepa/structured_margin.py:68-92` places `grad_output` on every selected pair and the target subtraction places `-grad_output` on target pairs. Autograd applies the outer `1/L_seq`; hence the Danskin subgradient is `(y_hat-y*)/L_seq`. At ties, the deterministic maximizer still gives a valid subgradient; the exhaustive unique-maximizer checks avoid claiming a unique gradient at ties.

The legal set is constructed once by `torch.triu(mask, diagonal=min_loop+1)` at `src/rnajepa/structured_margin.py:107`. Only this set enters score augmentation, target scoring and traceback. Illegal/lower-triangle entries may contain NaN/Inf without changing loss or gradient. Targets are checked for range, legality, uniqueness, matching occupancy and crossing at `src/rnajepa/structured_margin.py:112-133`.

## 2. Independent verification

`tools/verify_structured_margin_contract_review_v1.py` uses a separate oracle: enumerate every subset of legal pairs, reject nucleotide reuse and crossings, then directly compute score plus symmetric-difference Hamming cost. It does not reuse the production inside recurrence or the repository's brute-force recursion. The subset oracle is deliberately bounded to at most 20 legal pairs; longer min-loop regression cases use only short fixtures so the verifier cannot become an accidental exponential job.

The independent run covers all `4^4+4^5+4^6=5376` RNA sequences at lengths 4--6, random non-tied scores, a target rotated through the legal structures, and both loss and gradient. Focused tests additionally cover `min_loop={0,1,3,5}`, empty/middle/last targets and both `length` and `sum` normalization. The original 5376-case oracle and the new subset oracle are independent implementations of the same frozen mathematics.

The verifier also audits converged margin artifacts fail-closed: objective/metric identity, all validation rows, pre-terminal ineligibility, five non-improving eligible evaluations, terminal 20/33 state, distinct stopped/restored-best convergence states, checkpoint phase, best/resume tensor equality and SHA256. It does not authorize or launch training.

The independent report is `/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_contract_review_v1/generation_01/independent_contract_review.json`: 5376 target cases passed, with 10368 legal structures scored by the oracle, maximum absolute loss error `1.6653345369377348e-16` and gradient error `0`. The combined regression run passed 48 tests with one explicitly environment-gated CUDA-resume skip. Existing four-arm smoke identity also passes from its registered candidate repository.

Smoke identity deliberately pins absolute candidate-contract and registry paths. Consequently, running the old smoke verifier with `--repo /home/rna_jev_margin_contract_review` rejects those two path fields even though every tracked production file is byte-identical to `92f0bc2`; the registered-path check passes. This path binding was not relaxed or rewritten.

## 3. Selection and convergence fairness

Margin checkpoints must be selected by the unweighted development mean of length-normalized structured margin, every 500 steps, only after terminal 20/33 unfreeze at step 9600. Improvement is strict `loss < best - 1e-6`; five consecutive eligible non-improvements trigger stopping and restoration of the best eligible checkpoint. Step 9600 is not a stop cap. Divergence, resource interruption, a two-step smoke or a manual cap is not convergence.

This is fair **within each objective**, not a common scalar ranking across objectives. Historical NLL values and margin values cannot select the same checkpoint or be compared numerically. A cross-objective comparison requires independent convergence and best-checkpoint selection under each objective, followed by the same frozen development/test panels, fixed MEA (`temperature=1`, `gamma=1`, no search), paired structural metrics and uncertainty analysis. Same-step 10500 comparisons are diagnostic only.

The candidate correctly separates `convergence` (stopped state) from `best_convergence` (restored-best state). This distinction is required: restored best naturally has `bad_evaluations=0`, whereas the stopped state has `bad_evaluations=5`. Comparing the restored checkpoint's convergence directly with the final stopped validation row is a verifier schema error unless the old checkpoint format is explicitly interpreted as two different states.

## 4. Historical NLL qualification

The four historical runs under `/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3` have NLL validation logs showing stop at step 13000, best at 10500, patience five and terminal 20/33. However, the independent artifact `/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3_independent_reeval_v1/independent_reeval.json` reports:

- `status=FAIL_GEC_SEED0_INDEPENDENT_REEVAL`;
- `qualification_pass=false` for all four arms, specifically `checkpoint_and_restored_convergence=false`;
- `overall_pass=false`, with the preregistered structural and intervention gates also failing.

Thus all four historical NLL arms are **reference-only in this review**. Their logs are evidence of NLL stopping, not contract-qualified support for a GEC advantage or for margin superiority. Old metadata must not be relabeled as passing. A future formal NLL comparator needs an explicit old-format stopped/best requalification from raw logs/checkpoints or a clean rerun, followed by independent fixed-decoder metric recomputation.

## 5. Minimum formal matrix

| Arm | New structured-margin run required for the next H-GEC test | Historical NLL role now |
|---|---:|---|
| `matched_control` | **Required**, seed 0, full convergence, best restore | Reference only; required qualified comparator for any margin-vs-NLL control claim |
| `ensemble_context` | **Required**, seed 0, full convergence, best restore | Reference only; required qualified comparator for any margin-vs-NLL treatment claim |
| `destroy_interval_shuffle` | **Required**, seed 0, full convergence, best restore | Reference only; cannot substitute for the margin destroy intervention |
| `independent_context` | **Required**, seed 0, full convergence, best restore | Reference only; cannot substitute for the margin independent-context intervention |

No historical NLL arm may replace any of the four new margin arms. If the only claim is the preregistered context mechanism **under structured margin**, the minimum new training matrix is exactly the four margin arms; NLL reruns are not part of that minimum. If the claim includes **margin versus NLL**, qualified NLL comparators are additionally necessary: at minimum matched-control and ensemble for a treatment/control objective comparison, and all four if the NLL-side intervention controls are also claimed.

The four interventions are sufficient to test the preregistered H-GEC contrast, not to prove a general context mechanism. The source gate remains unchanged: ensemble must strictly beat both destroy and independent controls on micro/macro pair/stem F1, satisfy the matched-control precision/F1 tolerances, improve the isolated/long-span target with counts, paired CI and family-concentration/leave-one-family-out checks. Seed 0 can only gate a later seeds-1/2 proposal; it cannot establish robustness by itself.

## 6. Full-contract decision

**FAIL_FULL_CONTRACT.** The exact numerical core and two-step engineering smoke pass, but no four-arm structured-margin convergence result exists, no independent fixed-MEA structural evaluation of margin best checkpoints exists, and worst-case `B=4/L=500/terminal-20` memory/runtime evidence is absent. Historical NLL qualification and science gates are failed. The next narrative is therefore a proposed experiment contract only, not a supported result.

Full training remains frozen. Passing this review must not relax resource, authorization, convergence, decoder, multi-seed or scientific gates.
