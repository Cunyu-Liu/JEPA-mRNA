# A3 axis: latent target comparison (region summary vs masked positions vs both)

INTERMEDIATE RESULT — smoke scale: 100 optimizer steps, the 22.7k-sequence sample corpus,
batch 4 x grad_accum 2, max_len 512, alpha 0.5, identical seed and data across arms.
**Not a scientific conclusion**: the downstream head-to-head decides which target wins.
This record exists so the configuration choice for the main run is traceable to evidence.

| step | L_MLM region | L_JEPA region | cos region | L_MLM masked | L_JEPA masked | L_JEPA both |
|---|---|---|---|---|---|---|
| 20 | 1.973 | 0.193 | 0.969 | 1.973 | 0.596 | 0.234 |
| 40 | 1.940 | 0.045 | 0.990 | 1.939 | 0.337 | - |
| 60 | 1.835 | 0.017 | 0.996 | 1.835 | 0.266 | - |
| 80 | 1.884 | 0.030 | 0.992 | 1.884 | 0.231 | 0.096 |
| 100 | 1.861 | 0.046 | 0.993 | 1.862 | 0.151 | - |

Runs: `runs/a3_target_region`, `runs/a3_target_masked`, `runs/a3_target_both`.

## Findings

1. **The pooled-region-summary target saturates almost immediately.** cos reaches 0.993 and
   L_JEPA falls to 0.02-0.05 within 60 optimizer steps. The target is a mean over the whole
   region, and even with half the tokens replaced by a constant the remaining half still
   determine that mean almost exactly, so a near-linear read-out solves it.
2. **Raising the mask rate does not fix it.** A separate sweep at mask_prob 0.30 and 0.50
   (`runs/sweep_maskprob_*`) gave L_JEPA 0.036 and 0.040 at step 60 — no better than 0.15.
   More masking removes information but does not make the *target* harder to guess.
3. **Masked-position latent prediction stays informative.** It is 3.3x larger at step 100
   and still descending, i.e. it contributes gradient signal throughout training rather
   than in the first minute.
4. **The latent target form does not disturb the MLM channel.** L_MLM is 1.8613 (region)
   vs 1.8616 (masked) at step 100.
5. Anti-collapse machinery is healthy in all arms: variance 0.83-0.90, covariance driven
   towards 0, orthogonality exactly 0.

## Consequence for the plan

The A3 four-cell ablation (target-set x loss-form) the proposal promises as the gap
ProteinJEPA left open is precisely the experiment that separates these targets. The risk
described above therefore becomes a deliverable rather than a blocker: the main run uses
`jepa_target=both` (structural region prior plus a non-saturated signal), and `region` /
`masked` / `cos` / `mse` remain the A3 arms.

## One caveat to carry forward

The variance/covariance statistic is computed on a batch of 4 sequences x 768 dimensions,
which is a very noisy estimate of a 768-dimensional covariance (VICReg normally wants
batch size on the order of the feature dimension). Main runs therefore use a larger batch,
and the covariance trace is logged so its noisiness stays visible rather than being hidden
by the loss value.