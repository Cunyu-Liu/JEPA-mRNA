# SOTA vs ours on the frozen eval splits (15.14)

Clean = trained on bprna_tr1c (zero overlap with every frozen
split, see tables/overlap_audit.md) or the split itself is
zero-overlap with the training corpus (bpRNA-new rows).

| tier | split | our best (clean) | P | R | macro | SOTA reference | ref value | delta |
|---|---|---|---|---|---|---|---|---|
| 1 | TS0 | plana_tr1c @20k **0.7866** | 0.8241 | 0.7523 | 0.7839 | RNAformer 32M (bprna ckpt, our scorer, project GT) | 0.7578 | ✅ |
| 1 | bpRNA-new | r2d_tr1 2-seed ensemble @10k **0.6132** | 0.6314 | 0.5959 | 0.6125 | UFold (our scorer, project GT) | 0.6106 | ✅ |
| 1 | ArchiveII-clean | r2dtr1c @20k **0.7403** | 0.7230 | 0.7583 | 0.7974 | vienna centroid bucket-max (our scorer) | 0.7212 | ✅ |
| 1 | ArchiveII600 (Mathews macro, NucleicBERT Tab.1 protocol) | pending 15.14 arch600 eval | — | — | — | RNAErnie+ (quoted) | 0.875 | pending |
| 2 | TS1 | plana_tr1c @20k **0.8356** | 0.8920 | 0.7859 | 0.7926 | RNAformer inter-family ckpt (our scorer, project GT) | 0.8150 | ✅ |
| 2 | TS-hard | plana_tr1c @20k **0.8530** | 0.9135 | 0.8000 | 0.7949 | RNAformer bprna ckpt (our scorer, project GT) | 0.7845 | ✅ |
| 2 | TS2 | plana_tr1c @20k **0.8197** | 0.9251 | 0.7359 | 0.7883 | RNAformer inter-family ckpt (our scorer, project GT) | 0.9043 | −0.0846 |
| 2 | TS3 | plana_tr1c @20k **0.9049** | 0.9206 | 0.8897 | 0.8404 | RNAformer bprna ckpt (our scorer, project GT) | 0.9410 | −0.0361 |
| 2 | TestSetB | r2dtr1c @20k **0.7507** | 0.7617 | 0.7400 | 0.7542 | RiNALMo-ft INF 0.67 (quoted, RiNALMo S5) | 0.6700 | ✅ |
