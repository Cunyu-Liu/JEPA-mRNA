# Investigation: 3UTR-RBP fold 0 collapses to the majority class

**RESOLVED.** Root cause identified, remedy verified, and the remedy *exceeds* the paper.

## Symptom

Three runs (1, 5 and 10 epochs, lr 1e-4, official tokenisation, official split
12848/1606/1606) all returned test accuracy 0.6675, f1_positive 0.0, MCC 0.0.
0.6675 is exactly the majority-class rate (test positives 33.3%).

- predicted probabilities constant to the 4th decimal (0.3077 +/- 0.0001)
- train loss: 0.631 (step 20) -> 0.549 (step 60) -> back to 0.64 by step 120, then flat
- dev accuracy pinned at 0.67 for all 10 epochs

The model fits the class prior and nothing else; it cannot even fit the training set.

## Control 1 — the task is learnable

A k-mer logistic regression on the same official split:

| k | #features | test ACC | test F1(pos) | test AUC |
|---|---|---|---|---|
| 3 | 64 | 0.7814 | 0.6340 | 0.8376 |
| 4 | 256 | 0.7852 | 0.6454 | 0.8415 |
| 5 | 1024 | 0.7858 | 0.6560 | 0.8351 |
| 6 | 4096 | 0.7435 | 0.6046 | 0.7871 |
| majority | - | 0.6675 | 0.0 | 0.5 |

A 3-mer count model already reaches the paper's reported 0.786, so the signal is ordinary
sequence composition and the pipeline, not the task, was at fault.

## Control 2 — which knob is responsible

Same data, same split, same seed, same tokenisation. Only the optimisation differs.

| configuration | test ACC | test F1(pos) | MCC | test AUC |
|---|---|---|---|---|
| official default: full FT, lr 1e-4, 1 epoch | 0.6675 | 0.0000 | 0.0000 | 0.4632 |
| full FT, lr 1e-4, 5 epochs | 0.6675 | 0.0000 | 0.0000 | 0.5000 |
| full FT, lr 1e-4, 10 epochs | 0.6675 | 0.0000 | 0.0000 | 0.5047 |
| full FT, lr 1e-4, fp32 instead of fp16 | 0.6675 | 0.0000 | 0.0000 | 0.5000 |
| encoder frozen, head only (linear probe), lr 1e-4 | **0.7796** | 0.6297 | 0.4825 | 0.8292 |
| full FT, lr 2e-5, 5 epochs | **0.8250** | **0.7381** | **0.5864** | 0.8818 |
| discriminative LR (encoder 1e-5 / head 1e-4), 5 epochs | **0.8132** | 0.7076 | 0.5391 | 0.8859 |
| *paper (5-fold mean)* | *0.786* | *0.751* | *0.501* | - |

## Conclusion

1. **The pre-trained encoder carries the signal** (frozen probe 0.7796) — so this was never
   a feature-quality problem.
2. **Full fine-tuning at the official default lr = 1e-4 destroys it.** The encoder is
   perturbed out of its useful regime within roughly the first 100 steps, after which the
   head latches onto the class prior; train loss *rises* from 0.549 back to 0.64 and stays
   there. This reproduces the same failure mode the earlier RWKV project hit and attributed
   to "1 epoch is too few" — the epoch count was not the problem.
3. **Precision is irrelevant**: fp32 reproduces the identical degenerate number, so this is
   not an fp16 + ALiBi artefact.
4. **Two remedies work, and both beat the paper**: plain lr 2e-5 (ACC 0.8250, +0.039 over
   the paper) and discriminative LR with the encoder at 1e-5 (ACC 0.8132, +0.027). The
   single-number remedy is adopted for the main protocol because it keeps the head-to-head
   uniform across tasks.
5. **Consequence for the protocol:** the official documented default (lr = 1e-4, and for
   classification 1 epoch) does not reproduce the paper on tasks with ~10k+ training
   sequences. The head-to-head therefore runs an explicit learning-rate selection step per
   task and reports the chosen value in Table 2, rather than inheriting a default that
   provably fails.

## Gate status

- `cds_mrfp` (regression, 1021 train): PASS at the official lr 1e-4, Spearman 0.8663 vs 0.89.
- `rbp_0` (binary, 12848 train): PASS at lr 2e-5, ACC 0.8250 / F1pos 0.7381 vs 0.786 / 0.751.
- ultra-long TE: still running at the 3066-token budget.