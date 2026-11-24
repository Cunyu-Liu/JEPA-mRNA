# FoldNar baseline recovery v3 — SPOT-RNA and RNAformer

Date: 2026-11-24
Scope: frozen project `bpRNA-new`, 5,388 sequences. Recovery used CPU-only JSON/JSONL parsing, hashing, public metadata/config downloads, and exact set scoring. No model inference, training, calibration, registry edit, evaluator edit, running-config edit, or PPT edit was performed.

## Exact independent results

All arms have 5,388/5,388 predictions (100% coverage), unique names in frozen-data order, and no missing or extra names. Scoring uses zero-based inclusive exact pairs, pooled TP/FP/FN for micro metrics, and the unweighted per-sequence mean for macro-F1. The unified empty-pair convention is F1=1 when truth and prediction are both empty. The scorer adds no crossing, canonicality, or minimum-span filter.

| Prediction identity | Role | TP | FP | FN | Micro P | Micro R | Micro F1 | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SPOT-RNA registry raw | Primary registry-frozen artifact | 0 | 0 | 135184 | N/A | 0.0 | **0.0** | 0.0001855976243504083 |
| SPOT-RNA `use_postprocess=True` | Separate recovered protocol, not a replacement for registry raw | 77326 | 52486 | 57858 | 0.5956768249468463 | 0.5720055627884957 | **0.58360126190584** | 0.5682797332188521 |
| RNAformer 32M bpRNA | Primary registry-frozen artifact | 51372 | 21605 | 83812 | 0.7039478191759048 | 0.38001538643626465 | **0.49357948895326215** | 0.45885354923978383 |

The registry SPOT-RNA artifact has 5,388 empty predictions. Its nonzero macro-F1 is exactly 1/5,388 from the sole zero-truth sequence, for which the prediction is also empty. Micro precision is undefined, not zero. The postprocessed arm has three empty predictions; RNAformer has 156.

RNAformer's recorded macro-F1 is 0.4586679516154334 because its historical macro calculation assigns F1=0 to the empty/empty sequence. The unified recomputation assigns F1=1, increasing macro-F1 by exactly 1/5,388. Its TP/FP/FN and micro-F1 match the recorded values. All three recorded repeats and every recorded SPOT-RNA per-sequence TP/FP/FN/F1 were verified; prediction-identity hashes also match.

## Artifact and prediction identities

- Frozen data: `/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl`, SHA256 `271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9`; 5,388 unique names and 135,184 truth pairs.
- SPOT-RNA registry raw: SHA256 `613ba7a4c8ceae0a8197657a6b96e8a01c3c54cd923cfa81974b9ddbb16af6cc`; prediction identity `39c04083926497fb75ef91d28d5638881af2aca7774bb5f02290d428f6559826`.
- SPOT-RNA postprocessed: SHA256 `c76f473574fa0032213f49bc8e03e70406bf7cb2720d4a78a57d149b71ff5ec8`; prediction identity `b712c66072a7ccc8fcf2d7d5b301c3ffb18a70d48a1db1aaf488f0d7fbbda1af`.
- RNAformer: SHA256 `1559e6c4fbfcbb5e36debfe9c41b9ed0af6575a01100beac4b2c07780397c856`; prediction identity `c8a6587bf359afae1b125ca7127f04580b474d6e0445e5b6f327e06bda54ebaf`.

Normalized predictions and per-sequence counts/P/R/F1 are under `/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v3_spot_rnaformer/predictions/` and `reports/`. No missing predictions were imputed.

## Source, weights, and protocol provenance

### SPOT-RNA

The executed distribution is the **MultiMolecule 0.2.2 conversion**, not the original-author model archive. Original source context is `jaswindersingh2/SPOT-RNA` at local commit `dfd027727b3890d2a80aee189b2a48ca27063163` (source license MPL-2.0). The converted `multimolecule/spotrna` resolve endpoint reports repository commit `ea6e067d459da10f3e9154c90dd822ece47cf522`, size 69,959,644 bytes, and SHA256 identity `b7056fcad23a00d5bbaa21f91b54da779e87fa5e2185fb3ccc19c5440cadf0c5`; the complete local `model.safetensors` matches both. A freshly downloaded config matches SHA256 `c260db054bad4f3954e736ab31c74a5270ab045302e6c5e2ca9e8f225f2b45f9`.

The registry raw artifact records threshold 0.335 but is degenerate. The separately preserved postprocessed artifact uses raw full-length sequence, batch size 1, no cached embeddings, `use_postprocess=True` contact maps, threshold 0.335, and three deterministic repeats. Historical CUDA outputs were not rerun.

### RNAformer

The local official source context is `automl/RNAformer` at `196e7dea01e5446cfbcb9e9b594cd10a8b537372` (Apache-2.0 source). The bpRNA state dict is 63,501,297 bytes with verified local/artifact/checksum SHA256 `b2775613e9addafe30d17f7152dc985e8381826842d5d5056030006adfad6769`. A fresh official config download matches SHA256 `2af84e2caed3a784c5e45101be0a04c37c94f4605d3d3e3ed82a18c4d2948054`. The fresh full-weight download timed out; partial content was discarded and was not used for certification.

The frozen protocol records 31,741,954 parameters, 121/121 strict-loaded keys, cycling=6, LoRA=false, raw full-length sequences, batch size 1, no length filtering or cached embeddings, deterministic settings, three repeats, and greedy non-crossing decoding at P>0.5.

## Training, homology, and label limitations

1. **Raw, not decontaminated:** both prior MMseqs2 audits report zero exact training overlaps but two homology-only `bpRNA-new` hits at identity≥0.8 and query coverage≥0.8 (zero-based dataset indices 481 and 1955). The metrics above deliberately retain all 5,388 sequences; the 5,386-row clean subsets are not substituted.
2. **SPOT-RNA training union:** bpRNA TR0 (10,682) plus transfer TR1 (45,865), total 56,547, union SHA256 `a9daee3489613f1b3a2d6315f7b1fb19da74310f121a52933439bfa2ce14b61a`.
3. **RNAformer training identity:** official `bprna_data.plk` SHA256 `3cc294e79a68df83f5886b3347eb159d2fad80e6f8bca3dfd8d134f1fdf3c456`; train n=38,184, validation n=1,184, TS0 n=1,305; train union SHA256 `f41ce67e2f9ca1f3a84c62654ee442ce31153ee92508267e2c1c06fc8483cee0`. Threshold-based overlap removal is not family-level or pretraining decontamination.
4. **Label scope:** project labels are database-derived canonical nested pairs with unresolved per-record experimental lineage. RNAformer release evaluation can include pseudoknots and non-canonical pairs; these project-label numbers are not interchangeable with paper-table values. SPOT-RNA was designed for non-canonical/non-nested pairs, while the frozen adapter emits a decoded secondary structure and project labels exclude those pair types.
5. **BLOCKED_PROVENANCE — SPOT-RNA:** converted-weight license and exact original-ensemble-to-MultiMolecule conversion lineage were not verified. No original-author per-sequence bpRNA-new prediction artifact was recovered; the results apply only to the identified conversion. Direct Hugging Face access failed; public resolve metadata and a mirror-downloaded config supplied the independent identity checks.
6. **BLOCKED_PROVENANCE — RNAformer source snapshot:** the artifact records code commit `c548a77fd5b4dd2a0af622e42fd6a8a7f29240ab` and code SHA256 `ffa3cf372f54bd6472c75e7e105a345175eaf09c0cff311330b35114096fac56`; that commit is absent locally, the public commit request returned HTTP 422, and the current adapter hash differs. Exact historical source recovery remains blocked even though prediction, weight, config, and protocol identities are verified.

## Reproduction and verification

```bash
cd /home/rna_jev_xens2_dualview_round56
/root/miniconda3/envs/lucaone_cu128/bin/python -m unittest \
  tests/test_recover_spot_rnaformer_v3_20261124.py -v
/root/miniconda3/envs/lucaone_cu128/bin/python \
  tools/recover_spot_rnaformer_v3_20261124.py \
  --output-root /teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v3_spot_rnaformer
```

The standalone scorer does not import the project evaluator. Four unit tests cover zero-based bounds, one-partner constraints, exact TP/FP/FN without span filtering, empty/empty macro handling, undefined micro-F1, and rejection of missing predictions. Machine-readable metrics and detailed provenance are `reports/bprna_new_metrics_v3.json` and `reports/official_provenance_v3.json` in the evidence root.
