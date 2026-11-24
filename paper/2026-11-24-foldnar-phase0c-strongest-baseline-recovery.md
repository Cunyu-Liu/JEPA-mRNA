# Phase 0C strongest-baseline recovery: MXfold2 and EternaFold

## Scope and decision

This audit is limited to the frozen raw `bprna_new` evaluation assets registered in `manifests/foldnar_v1/{baseline_registry,data_registry,prediction_registry}.json`. It performs no training, model development, threshold selection, or external-test-set adaptation. Shared registries were not edited; the field-level integration payload is `manifests/foldnar_v1/baseline_recovery_phase0c_correction.json`.

Both systems have complete frozen per-sequence predictions and verified official model assets. EternaFold is stronger on this matched raw evaluation by 0.01653739014072031 strict micro-F1 and 0.017749185112890253 macro-F1. This is not a clean/homology-controlled ranking.

| System | Coverage | TP | FP | FN | Micro precision | Micro recall | Strict micro-F1 | Strict macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MXfold2 | 5388/5388 (1.0) | 100227 | 64290 | 34957 | 0.6092197158956217 | 0.7414117055272813 | 0.6688466171284047 | 0.68194864244077 |
| EternaFold | 5388/5388 (1.0) | 104470 | 65197 | 30714 | 0.6157355290068193 | 0.7727985560421352 | 0.685384007269125 | 0.6996978275536603 |

The scoring reference is the frozen project JSONL, SHA-256 `271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9`. Pairs are zero-based inclusive and matched exactly. No summary-only metric, missing-value zero, or imputed prediction was used.

## Recovered official provenance

### MXfold2

- Official release: `v0.1.2`, commit `51b213676708bebd664f0c40873a46e09353e1ee`.
- Official Linux wheel: `mxfold2-0.1.2-cp310-cp310-manylinux_2_17_x86_64.whl`, SHA-256 `a9ea044a6555903e019fc016d8409ba0907bfb143bbf2eda51b1d27aae763c72`.
- Wheel member `mxfold2/models/TrainSetAB.pth` has SHA-256 `8ba1283e70c1c073033161adc120f67cdc467176476358d26a12cadaf63262ec`, exactly matching both the installed checkpoint and frozen prediction metadata.
- Wheel member `TrainSetAB.conf`, SHA-256 `391385f5777c81cdc15b57827ba91c305525d76e400340e225c0760a8adeec42`, fixes the `MixC` architecture and parameter path.
- The official wheel metadata and bundled license verify version `0.1.2` and MIT licensing. License SHA-256: `6e55a4d5663f8853cea027822325c6cb652f5f9f5826295d8314987ad0327b7e`.
- Frozen prediction file SHA-256: `9cc688e0c57f188213c323f553039c76c896029b50e2446f8eeeceafc2b94e19`.

The `v0.1.2` `bpRNAnew.tar.gz` URL returned HTTP 404. The official `v0.1.1` archive was recovered instead (SHA-256 `0d1959ac93e2b15ac47b56c7dd8155d9a0fcd3a692f75f2d0ede9221bf9d9dfb`). It contains 5,401 reference BPSEQ files and no per-sequence prediction structures. All 5,388 project records match an archive record by name and normalized sequence; 13 archive-only records are outside the frozen project split and were not scored. Only 106 common records have exactly equal pair sets, demonstrating that the project labels include pair filtering. The official archive therefore provides identity context, not replacement labels or predictions.

### EternaFold

- Canonical source: `WaymentSteeleLab/EternaFold` (the `eternagame/EternaFold` remote redirects here), pinned commit `87b9aac55cee14fd562049d08f7b92d3131f10ce`.
- Official `parameters/EternaFoldParams.v1`, SHA-256 `1421e89cc8df24b53a320eff6c72b8acfdb2771faba29ab9c798f7bdfc853a8a`, exactly matches the frozen prediction metadata and recovered local file.
- Protocol: official serial CONTRAfold-derived binary, `predict --params EternaFoldParams.v1`, default MEA decoding, one CPU thread. Frozen binary SHA-256: `5b834c29a9cc1ade835662ca70dfd8ea5f437f78e871e4855b22e9d9e0126caf`.
- Official source and parameter licenses verify the BSD-3-Clause terms. Source license SHA-256: `8431990ae8dd84812802a5ef3d7cfe05d90257c96ed740c6e6f04ff32bc22afc`; parameter license SHA-256: `b545ce6c0c8eb651f7e75ed1e074c3be8d1bb1b81b3b822750b69e5e18944597`.
- Frozen prediction file SHA-256: `aa2ef3577f150e42d90fb8954383d2ccc59169e8164209ef83cdcf48b073f04b`.

The pinned EternaFold tree has no bpRNA-new per-sequence prediction asset. EternaBench repository/scoring-table inspection found benchmark scores rather than frozen bpRNA-new structures; those tables were not substituted.

## Independent verification

The new auditor does not import either original baseline scorer. It verifies registry and file hashes, unique ordered sequence identity, pair validity, full coverage, strict per-sequence and aggregate counts, original three-repeat metadata, official wheel contents, parameter identity, and release/reference identity.

Focused tests and full audit:

```bash
cd /home/rna_jev_xens2_dualview_round56
/root/miniconda3/envs/lucaone/bin/python -m pytest -q tests/test_recover_strongest_baselines_phase0c.py
# 5 passed in 0.04s
/root/miniconda3/bin/python3 tools/recover_strongest_baselines_phase0c.py
# PASS for both systems; 5388/5388 coverage
```

A deterministic replay of the first six frozen evaluation records also passed for both systems, with every pair set equal to the frozen prediction. MXfold2 used the official CLI on CPU with `@TrainSetAB.conf`; EternaFold used the official serial binary and recovered parameter file. The replay evidence is `/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v1/phase0c_replay_verification.json`, SHA-256 `d551a22ff6c9b44ffd36de196bad52ca12e103bcc8a19c590baaaef1b43d41ef`.

Verification JSON: `/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v1/phase0c_verification.json`, SHA-256 `4874a66150c70febafa1ee08ce20bc5887e726aa40087c6cb7dcb97e5731e52a`.

Correction JSON SHA-256: `6d9a2d155cc8fc96b58ea7dffa6466d65d6a3697a0521ed5502f3d04380a1970`.

## Remaining BLOCKED_PROVENANCE

- MXfold2 `training_data.homology_overlap`: TrainSetA/B identity is verified, but this slice did not reconstruct the complete training union and its bpRNA-new homology overlap.
- EternaFold `training_data.exact_union`: official documentation and data lists identify the broad multitask sources, but the exact effective training union and bpRNA-new homology overlap were not reconstructed.

Accordingly, these results remain `PROJECT_RAW_GT_ONLY`; they do not establish clean, decontaminated, or CHANRG-protocol comparability. Weight/parameter, license, version, inference protocol, prediction coverage, and same-sequence metric provenance are recovered. No third comparator was needed to complete the assigned two-system recovery.
