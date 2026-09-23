# Investigation: 3UTR-RBP fold 0 collapses to the majority class

INTERMEDIATE / DIAGNOSTIC result. Not a final scientific conclusion.

## Symptom

Three runs (1, 5 and 10 epochs, lr 1e-4, official tokenisation, official split
12848/1606/1606) all returned test accuracy 0.6675, f1_positive 0.0, MCC 0.0.
0.6675 is exactly the majority-class rate (test positives 33.3%).

- predicted probabilities constant to the 4th decimal (0.3077 +/- 0.0001)
- train loss: 0.631 (step 20) -> 0.549 (step 60) -> back to 0.64 by step 120, then flat
- dev accuracy pinned at 0.67 for all 10 epochs

So the model fits the class prior and nothing else; it cannot even fit the training set.

## Control: the task is learnable

A k-mer logistic regression on the same official split:

| k | #features | test ACC | test F1(pos) | test AUC |
|---|---|---|---|---|
| 3 | 64 | 0.7814 | 0.6340 | 0.8376 |
| 4 | 256 | 0.7852 | 0.6454 | 0.8415 |
| 5 | 1024 | 0.7858 | 0.6560 | 0.8351 |
| 6 | 4096 | 0.7435 | 0.6046 | 0.7871 |
| majority | - | 0.6675 | 0.0 | 0.5 |

A 3-mer count model already reaches the paper's reported 0.786, so the signal is ordinary
sequence composition and our pipeline is at fault, not the task.

## Controls launched to separate the causes

- `rbp_0_lr2e5_5ep`      : lower learning rate (2e-5) — tests "lr too high wrecks the encoder"
- `rbp_0_lr1e4_5ep_fp32` : fp32 instead of fp16 — tests "fp16 + ALiBi instability"
- `rbp_0_frozen_5ep`     : encoder frozen, only pooler+classifier trained (linear probe) —
                           separates "features uninformative" from "fine-tuning destroys them"

The outcome decides which protocol the head-to-head uses. Until it is resolved the R3 gate
is **not** passed for RBP-family tasks and no downstream claim about 3'UTR performance is
made.

## Working hypothesis (to be confirmed by the controls, not assumed)

The pattern "brief improvement, then regression to the prior, then flat" is what a
learning rate that is too high for a 12.8k-sample binary task looks like: the pretrained
encoder is perturbed out of its useful regime within the first ~100 steps, after which the
head latches onto the class prior. mRFP is regression with 1k samples and does not show
this. If the frozen-encoder probe succeeds while full fine-tuning fails, the protocol needs
a lower learning rate or a layer-wise/warm-up schedule for classification tasks.