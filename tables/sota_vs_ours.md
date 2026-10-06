# SOTA vs ours on the frozen eval splits (15.20)

Clean = trained on bprna_tr1c (zero overlap with every frozen
split, see tables/overlap_audit.md) or the split itself is
zero-overlap with the training corpus (bpRNA-new rows).

**Ensemble protocol (15.20): xens2 — plana 2-seed in-family average
(0.5 x plana_tr1c_s0 + 0.5 x plana_tr1c_s1) bucket x r2dtr1c_s0 bucket,
w_plana=0.7 on the plana bucket.** w re-selected on VL0 after the bucket
change (0.5/0.7/0.85 -> 0.8789/0.8789/0.8782, 0.7 retained; selection
split only, never test splits), then applied uniformly to ALL splits,
single run each. Supersedes 15.18 xens (single-seed plana bucket).
bpRNA-new keeps the r2d 2-seed ensemble row (0.6132, still its best;
xens2 0.5532).

| tier | split | our best (clean) | P | R | macro | SOTA reference | ref value | delta |
|---|---|---|---|---|---|---|---|---|
| 1 | TS0 | xens2 plana2seed x r2dtr1c @w0.7 **0.8039** | 0.8215 | 0.7869 | 0.8002 | RNAformer 32M (bprna ckpt, our scorer, project GT) | 0.7578 | ✅ +0.046 |
| 1 | bpRNA-new | r2d_tr1 2-seed ensemble @10k **0.6132** | 0.6314 | 0.5959 | 0.6125 | UFold (our scorer, project GT) | 0.6106 | ✅ +0.003 |
| 1 | ArchiveII-clean | xens2 plana2seed x r2dtr1c @w0.7 **0.7760** | 0.7956 | 0.7575 | 0.8095 | vienna centroid bucket-max (our scorer) | 0.7212 | ✅ +0.055 |
| 1 | ArchiveII600 (Mathews macro, NucleicBERT Tab.1 protocol) | plana_tr1c tolerant macro 0.7991/0.8068 (json, not result.json) | — | — | — | RNAErnie+ (quoted; leaky ref: TR0capArchiveII=732) | 0.875 | pending |
| 2 | TS1 | xens2 plana2seed x r2dtr1c @w0.7 **0.8755** | 0.9212 | 0.8342 | 0.8443 | RNAformer inter-family ckpt (our scorer, project GT) | 0.8150 | ✅ +0.061 |
| 2 | TS-hard | xens2 plana2seed x r2dtr1c @w0.7 **0.8732** | 0.9167 | 0.8337 | 0.8383 | RNAformer bprna ckpt (our scorer, project GT) | 0.7845 | ✅ +0.089 |
| 2 | TS2 | xens2 plana2seed x r2dtr1c @w0.7 **0.8588** | 0.9353 | 0.7939 | 0.8407 | RNAformer inter-family ckpt (our scorer, project GT) | 0.9043 | −0.0455 |
| 2 | TS3 | xens2 plana2seed x r2dtr1c @w0.7 **0.8965** | 0.9109 | 0.8825 | 0.8612 | RNAformer bprna ckpt (our scorer, project GT) | 0.9410 | −0.0445 |
| 2 | TestSetB | xens2 plana2seed x r2dtr1c @w0.7 **0.8448** | 0.9104 | 0.7880 | 0.8372 | RiNALMo-ft INF 0.67 (quoted, RiNALMo S5) | 0.6700 | ✅ +0.175 |
