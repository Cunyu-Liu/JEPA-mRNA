# SOTA vs ours on the frozen eval splits (15.18)

Clean = trained on bprna_tr1c (zero overlap with every frozen
split, see tables/overlap_audit.md) or the split itself is
zero-overlap with the training corpus (bpRNA-new rows).

**Ensemble protocol (15.18): xens rows use w_plana=0.7, selected once
on VL0 (clean weight-selection split, zero overlap with every test
split; sweep 0.3/0.5/0.7/0.8/0.9 -> 0.8727 peak at 0.7/0.9, 0.7
locked) and then applied uniformly to ALL splits, single run each.**
The earlier w=0.5 default rows and the post-lock w=0.9 TS2/TS3 probes
(never quoted) are exploratory; superseded by this table.
bpRNA-new keeps the r2d 2-seed ensemble row (xens is not its best:
plana-weight hurts OOD, w0.7 xens = 0.5188 vs 2-seed 0.6132).

| tier | split | our best (clean) | P | R | macro | SOTA reference | ref value | delta |
|---|---|---|---|---|---|---|---|---|
| 1 | TS0 | xens plana_tr1c x r2dtr1c @w0.7 **0.7904** | 0.8184 | 0.7642 | 0.7857 | RNAformer 32M (bprna ckpt, our scorer, project GT) | 0.7578 | ✅ +0.033 |
| 1 | bpRNA-new | r2d_tr1 2-seed ensemble @10k **0.6132** | 0.6314 | 0.5959 | 0.6125 | UFold (our scorer, project GT) | 0.6106 | ✅ +0.003 |
| 1 | ArchiveII-clean | xens plana_tr1c x r2dtr1c @w0.7 **0.7697** | 0.8002 | 0.7415 | 0.8030 | vienna centroid bucket-max (our scorer) | 0.7212 | ✅ +0.049 |
| 1 | ArchiveII600 (Mathews macro, NucleicBERT Tab.1 protocol) | plana_tr1c tolerant macro 0.7991/0.8068 (json, not result.json) | — | — | — | RNAErnie+ (quoted; leaky ref: TR0capArchiveII=732) | 0.875 | pending |
| 2 | TS1 | xens plana_tr1c x r2dtr1c @w0.7 **0.8381** | 0.8987 | 0.7851 | 0.8003* | RNAformer inter-family ckpt (our scorer, project GT) | 0.8150 | ✅ +0.023 |
| 2 | TS-hard | xens plana_tr1c x r2dtr1c @w0.7 **0.8616** | 0.9251 | 0.8063 | 0.7622* | RNAformer bprna ckpt (our scorer, project GT) | 0.7845 | ✅ +0.077 |
| 2 | TS2 | xens plana_tr1c x r2dtr1c @w0.7 **0.8429** | 0.9433 | 0.7618 | 0.8189* | RNAformer inter-family ckpt (our scorer, project GT) | 0.9043 | −0.0614 |
| 2 | TS3 | xens plana_tr1c x r2dtr1c @w0.7 **0.9055** | 0.9271 | 0.8849 | 0.8191* | RNAformer bprna ckpt (our scorer, project GT) | 0.9410 | −0.0355 |
| 2 | TestSetB | xens plana_tr1c x r2dtr1c @w0.7 **0.8280** | 0.8986 | 0.7677 | 0.8162 | RiNALMo-ft INF 0.67 (quoted, RiNALMo S5) | 0.6700 | ✅ +0.158 |

\* macro values marked * are from the w=0.5 run (macro not re-extracted
for w0.7 micro-only verification runs); micro F1 / P / R are the w0.7
locked-protocol numbers. TS1 macro 0.8003 is from xens_ts1 (w0.5),
TS-hard 0.7622 from xens_ts_hard (w0.5), TS2 0.8189 / TS3 0.8191
likewise — used only as indicative macro, never quoted as best.
