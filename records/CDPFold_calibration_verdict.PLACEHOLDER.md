# CDPFold calibration verdict (gate S7 / contribution C1)

**SYNTHETIC SELF-TEST — NOT a real CDPFold verdict.**

This record was produced from crafted probability matrices to exercise the
verdict logic. It is a PLACEHOLDER: the real verdict must be re-generated once
CDPFold predictions (or a published probability matrix) are available. The
CDPFold citation itself is marked 待核验 in `spec/citation_register.csv`.

Generated: 2026-09-23T17:52:53Z

## Inputs

- synthetic_case: `c1_holds`

## Calibration metrics

| model | ECE | marginal calib. | Brier | NLL | mean pred | mean empirical |
|---|---|---|---|---|---|---|
| CDPFold | 0.3788 | 0.3788 | 0.2500 | 0.6931 | 0.5000 | 0.1212 |
| ours (System-1) | 0.0100 | 0.0076 | 0.0001 | 0.0101 | 0.1288 | 0.1212 |
| exact marginals | 0.0100 | 0.0076 | 0.0001 | 0.0101 | 0.1288 | 0.1212 |

## Verdict (S7 threshold |ΔECE| ≤ 0.02)

- ΔECE (CDPFold − ours) = **0.3688**
- within threshold: **False**
- **C1 holds: True**
- reason: CDPFold 校准显著更差（ΔECE=0.3688 > 0.02）-> C1 成立
- ours vs exact (S7 for our own head): True

## Decision

Paper skeleton stays **C2 + C1**: CDPFold's DP-free probabilities are
meaningfully less calibrated than ours, so "DP-free *and* calibrated" remains
a defensible novelty.

## Caveat

Calibration is data- and length-dependent. A single matrix is not a verdict; the
real decision requires CDPFold evaluated on the frozen benchmarks (ArchiveII
de-redundified + bpRNA-new) with the same binning and mask as our own numbers.

