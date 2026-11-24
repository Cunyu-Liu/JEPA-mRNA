# Thermodynamic baseline recovery v4 (2026-11-24)

## Scope and metric contract

The primary comparison uses the frozen project bpRNA-new DATABASE labels and exactly 5,388 unique named sequences. Pair coordinates are zero-based inclusive `i < j`; scoring is exact pair-set equality with no crossing, span, or minimum-loop filtering. Micro-F1 is computed from pooled TP/FP/FN; macro-F1 is the unweighted mean of per-sequence F1. If truth and prediction both have zero pairs, per-sequence F1 is 1. Recovery is CPU-only.

| Predictor / decoding identity | Scope | N | Coverage | TP | FP | FN | Micro-F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vienna_mfe | raw | 5388 | 1.000000000000 | 99112 | 76430 | 36072 | 0.637938247845369 | 0.658083810492516 |
| vienna_centroid | raw | 5388 | 1.000000000000 | 97991 | 56297 | 37193 | 0.677032666371877 | 0.682072792246076 |
| vienna_mea | raw | 5388 | 1.000000000000 | 100390 | 67724 | 34794 | 0.661989198741831 | 0.676424835269224 |
| rnastructure_fold | raw | 5388 | 1.000000000000 | 97675 | 76320 | 37509 | 0.631834633011945 | 0.649659155581660 |
| rnastructure_maxexpect | raw | 5388 | 1.000000000000 | 99991 | 66506 | 35193 | 0.662892260367739 | 0.674669701714053 |
| contrafold_mea | raw | 5388 | 1.000000000000 | 104254 | 59814 | 30930 | 0.696763931402296 | 0.710887911244847 |

## Separate clean scope

ViennaRNA and RNAstructure are not sequence-fitted and have no distinct clean prediction asset; their clean artifact, metrics, and coverage fields are `null`. Their applicable raw 5,388 result is not duplicated under a clean label. CONTRAfold has a historical `clean_tr1c` artifact, shown separately below. It removes two sequences and is a project TR1c exclusion scope, **not** a verified CONTRAfold own-training-clean scope and not eligible for the same-5,388 comparison.

| Predictor | Clean scope | N | Coverage within clean scope | TP | FP | FN | Micro-F1 | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CONTRAfold MEA | clean_tr1c | 5386 | 1.000000000000 | 104221 | 59802 | 30920 | 0.696748271850891 | 0.710866073495700 |

Excluded from raw by `clean_tr1c`: `FP929046.1_562024-562125.bpseq, URS0000D67ADA_12908_1-55.bpseq`. Raw-scope coverage of this clean asset is 5386/5388; no scores are imputed for the excluded sequences.

## Provenance verification

- All four frozen source JSON hashes and the frozen raw/clean label hashes match their expected values.
- ViennaRNA is version 2.7.2. The Python extension SHA-256 matches its installed distribution `RECORD`; no CLI binary or explicit frozen parameter file was recorded, so those fields remain `null`.
- RNAstructure Fold and MaxExpect both report version 6.6 (April 2, 2026). Both executable hashes match the frozen artifacts, and the official Linux archive matches `SHA256SUMS`. Fold uses `--MFE`; MaxExpect uses `--sequence` and the frozen protocol retains the first CT block.
- CONTRAfold uses source commit `87b9aac55cee14fd562049d08f7b92d3131f10ce`; its binary and all three native parameter hashes match. A pre-existing unrelated `src/score_prediction` working-tree modification is recorded rather than hidden.
- Bounded CPU replay checked 12 selected sequences per decoder; every regenerated pair set exactly matched the frozen prediction. Full-table metrics are from frozen predictions, not partial replay extrapolation.
- Every system has three recorded deterministic inference repeats; all recorded counts, micro/macro F1 values, and prediction identity hashes were independently checked. These repeats are not training seeds or ensembles.

## Verification

The standalone verifier does not import this recovery script or the shared evaluator. It rescores normalized prediction JSONL with exact rational per-sequence arithmetic and verifies all artifact hashes, raw/clean scope boundaries, and unavailable-asset `null` fields.

```bash
/root/miniconda3/envs/codonflow/bin/python3 -m pytest -q tests/test_recover_thermodynamic_baselines_v4_20261124.py
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/codonflow/bin/python3 tools/recover_thermodynamic_baselines_v4_20261124.py
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/codonflow/bin/python3 tools/verify_thermodynamic_recovery_v4_20261124.py
```

No shared registry, evaluator, running configuration, or PPT file is modified by this recovery.
