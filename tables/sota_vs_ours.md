# SOTA vs ours on the frozen eval splits (15.14)

Clean = trained on bprna_tr1c (zero overlap with every frozen
split, see tables/overlap_audit.md) or the split itself is
zero-overlap with the training corpus (bpRNA-new rows).

| tier | split | our best (clean) | P | R | macro | SOTA reference | ref value | delta |
|---|---|---|---|---|---|---|---|---|
| 1 | TS0 | xens plana_tr1c x r2dtr1c **0.7859** | 0.8040 | 0.7686 | 0.7809 | RNAformer 32M (bprna ckpt, our scorer, project GT) | 0.7578 | ✅ |
| 1 | bpRNA-new | r2d_tr1 2-seed ensemble @10k **0.6132** | 0.6314 | 0.5959 | 0.6125 | UFold (our scorer, project GT) | 0.6106 | ✅ |
| 1 | ArchiveII-clean | xens plana_tr1c x r2dtr1c **0.7796** | 0.7996 | 0.7606 | 0.8129 | vienna centroid bucket-max (our scorer) | 0.7212 | ✅ |
| 1 | ArchiveII600 (Mathews macro, NucleicBERT Tab.1 protocol) | plana_tr1c tolerant macro 0.7991/0.8068 (json, not result.json) | — | — | — | RNAErnie+ (quoted; leaky ref: TR0capArchiveII=732) | 0.875 | pending |
| 2 | TS1 | xens plana_tr1c x r2dtr1c **0.8473** | 0.9110 | 0.7918 | 0.8003 | RNAformer inter-family ckpt (our scorer, project GT) | 0.8150 | ✅ |
| 2 | TS-hard | xens plana_tr1c x r2dtr1c **0.8339** | 0.9000 | 0.7768 | 0.7622 | RNAformer bprna ckpt (our scorer, project GT) | 0.7845 | ✅ |
| 2 | TS2 | xens plana_tr1c x r2dtr1c **0.8365** | 0.9461 | 0.7496 | 0.8189 | RNAformer inter-family ckpt (our scorer, project GT) | 0.9043 | −0.0678 |
| 2 | TS3 | xens plana_tr1c x r2dtr1c **0.8738** | 0.9028 | 0.8465 | 0.8191 | RNAformer bprna ckpt (our scorer, project GT) | 0.9410 | −0.0672 |
| 2 | TestSetB | xens plana_tr1c x r2dtr1c **0.8249** | 0.8873 | 0.7707 | 0.8153 | RiNALMo-ft INF 0.67 (quoted, RiNALMo S5) | 0.6700 | ✅ |
