# Factor probe: the control was vacuous, and the corrected control does not support C3

INTERMEDIATE / DIAGNOSTIC. Smoke-scale checkpoint (100 optimizer steps). **Not a final
scientific conclusion** — but it is strong enough to plan the fallback path.

## Why the first control was worthless

The initial design compared the learned K-block basis against K random orthonormal
projections. Both cases reported byte-identical numbers (balanced accuracy 0.933 vs
0.933). That is not a coincidence and it is not evidence about the factors: the K
learned blocks *tile* the hidden space, so concatenating all K is simply an orthogonal
rotation of the whole vector, and a linear probe is rotation-invariant. The control was
a no-op by construction.

## The corrected control

Fix the **rank** instead of the partition: draw one random `r_dim`-dimensional subspace
and probe it exactly like a single learned factor. That is the comparison that has
content — "is the learned r-dim subspace better than an arbitrary r-dim subspace?"

## Results (pool B: task identity within one region, four CDS tasks, identical codon tokenisation)

| probe | balanced acc | chance |
|---|---|---|
| learned factor 0 | 0.908 | 0.232 |
| learned factor 1 | 0.908 | 0.234 |
| learned factor 2 | 0.917 | 0.240 |
| learned factor 3 | 0.908 | 0.236 |
| all factors | 0.933 | 0.242 |
| **random 192-d subspace, trial 0** | **0.908** | 0.233 |
| **random 192-d subspace, trial 1** | **0.908** | 0.228 |

Pool A (region identity) behaves the same way: 0.867-0.889 learned vs 0.878-0.889 random.

Factor specificity — each factor against the union of the others:

| pool | factor | bal_acc(factor) | bal_acc(others) | delta |
|---|---|---|---|---|
| task | 0 | 0.908 | 0.933 | -0.025 |
| task | 1 | 0.908 | 0.933 | -0.025 |
| task | 2 | 0.917 | 0.925 | -0.008 |
| task | 3 | 0.908 | 0.933 | -0.025 |

## What this means, stated carefully

1. The label is decodable from **essentially any** 192-dimensional subspace of the
   encoder representation. So the probe is measuring the *encoder's* information content,
   not any property of the factorisation.
2. No factor is privileged: every factor is slightly **worse** than the union of the
   others, as expected if the four blocks are interchangeable partitions rather than
   specialised semantic axes.
3. Therefore the probe as designed **cannot** support the claim "orthogonal factors
   spontaneously align with RNA functional semantics". Under the proposal's fallback
   path, C3 is not claimed on this evidence and Fig.5 is planned as a training-dynamics
   panel.

## Caveats that must be respected before writing anything final

- This is a 100-step smoke checkpoint. The contract forbids promoting smoke-scale
  proxies to conclusions, so nothing here enters the paper.
- The random-subspace equivalence is a property of the *representation's decodability*
  and would be expected to persist at larger scale, but that is a prediction, not a
  measurement. It will be re-run on the full V1/V2 checkpoints.
- A genuinely stronger test remains available and is cheap: probe whether factor *k*
  carries information about label A that factor *j* does not, using the full
  factor × label association matrix rather than per-label accuracy. That is the
  direction to try before abandoning C3 entirely.

## Actions taken

- `eval/factor_probe.py` control replaced (rank-matched random subspace) and a
  factor-specificity table added, so the report cannot present a per-factor number
  without the comparison that gives it meaning.
- The report now prints the confound warning for the region pool inline.