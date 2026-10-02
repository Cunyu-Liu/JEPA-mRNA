# Full metrics matrix — every measured (method, dataset) cell

All values re-derived from result.json / baselines_*.json on disk
(script tools/metrics_matrix.py; no hand-typed numbers). Quoted
rows carry their paper source and are flagged. '—' = the metric is
not produced by that run's protocol (e.g. INF needs per-seq rows).

## Measured cells

| Method | Dataset | micro P | micro R | micro F1 | macro F1 | INF | n | source |
|---|---|---|---|---|---|---|---|---|
| vienna_centroid | ArchiveII-clean 100-200 | 0.7144 | 0.6607 | 0.6865 | 0.6725 | 0.6774 | 1136 | baselines_archiveii_embok_clean_gt100_le200.json |
| vienna_centroid | ArchiveII-clean 200-400 | 0.5629 | 0.6128 | 0.5868 | 0.5790 | 0.5824 | 728 | baselines_archiveii_embok_clean_gt200_le400.json |
| vienna_centroid | ArchiveII-clean <=100 | 0.7305 | 0.7122 | 0.7212 | 0.7028 | 0.7067 | 484 | baselines_archiveii_embok_clean_le100.json |
| vienna_centroid | ArchiveII-clean >400 | 0.4431 | 0.6734 | 0.5345 | 0.5262 | 0.5405 | 196 | baselines_archiveii_embok_clean_gt400.json |
| RNAformer (bprna ckpt) | TS0 | 0.8091 | 0.7127 | 0.7578 | 0.7454 | 0.7512 | 1291 | baselines_rnaformer_ref_bprna_ts0.json |
| Ours Plan-A s2 | TS0 | 0.8001 | 0.6773 | 0.7336 | 0.7129 | — | 1288 | plana_giga_s2_step20000 |
| Ours Plan-A (adapted+2D, s0) | TS0 | 0.7689 | 0.6891 | 0.7268 | 0.7139 | — | 1288 | plana_giga_s0_step20000 |
| Ours Plan-A s1 | TS0 | 0.8300 | 0.6428 | 0.7245 | 0.6989 | — | 1288 | plana_giga_s1_step20000 |
| Ours Plan-A ext40k @30k | TS0 | 0.7818 | 0.6689 | 0.7210 | 0.7044 | — | 1288 | plana_giga_s0_ext40k_step30000 |
| Ours Plan-A ext40k @40k | TS0 | 0.7947 | 0.6544 | 0.7178 | 0.6974 | — | 1288 | plana_giga_s0_ext40k_step40000 |
| Ours r2d (Plan-B, frozen+2D, s0) | TS0 | 0.6865 | 0.6410 | 0.6629 | 0.6436 | 0.6499 | 1288 | ow_rinalmo_r2d_b4_s0_step20000_bprna_ts0 |
| Ours r2d (Plan-B, s1) | TS0 | 0.6487 | 0.6632 | 0.6559 | 0.6470 | 0.6525 | 1288 | ow_rinalmo_r2d_b4_s1_step20000_bprna_ts0 |
| Ours bigtr1 (capacity x data) | TS0 | 0.7419 | 0.5699 | 0.6446 | 0.6128 | 0.6233 | 1288 | ow_rinalmo_bigtr1_b4_s0_step20000_bprna_ts0 |
| Ours TR1 @40k | TS0 | 0.5763 | 0.6586 | 0.6147 | 0.6288 | 0.6335 | 1288 | ow_rinalmo_ff_tr1_b4_s0_step40000_bprna_ts0 |
| MXfold2 | TS0 | 0.4816 | 0.6838 | 0.5651 | 0.5698 | 0.5832 | 1288 | baselines_mxfold2_bprna_ts0.json |
| ViennaRNA centroid | TS0 | 0.4710 | 0.6306 | 0.5393 | 0.5288 | 0.5405 | 1288 | baselines_bprna_ts0.json |
| ViennaRNA mea | TS0 | 0.4413 | 0.6451 | 0.5241 | 0.5218 | 0.5346 | 1288 | baselines_bprna_ts0.json |
| ViennaRNA mfe | TS0 | 0.4181 | 0.6391 | 0.5055 | 0.5086 | 0.5222 | 1288 | baselines_bprna_ts0.json |
| vienna_centroid | TestSetB | 0.5125 | 0.5893 | 0.5482 | 0.5577 | 0.5630 | 428 | baselines_testsetb.json |
| vienna_mfe | TestSetB | 0.4578 | 0.6085 | 0.5225 | 0.5430 | 0.5488 | 428 | baselines_testsetb.json |
| nussinov_turner | TestSetB | 0.1970 | 0.2401 | 0.2164 | 0.2251 | 0.2270 | 428 | baselines_testsetb.json |
| ViennaRNA centroid | bpRNA-new | 0.6351 | 0.7249 | 0.6770 | 0.6821 | 0.6871 | 5388 | baselines_bprna_new.json |
| MXfold2 | bpRNA-new | 0.6092 | 0.7414 | 0.6688 | 0.6819 | 0.6868 | 5388 | baselines_mxfold2_bprna_new.json |
| ViennaRNA mea | bpRNA-new | 0.5972 | 0.7426 | 0.6620 | 0.6764 | 0.6814 | 5388 | baselines_bprna_new.json |
| ViennaRNA mfe | bpRNA-new | 0.5646 | 0.7332 | 0.6379 | 0.6581 | 0.6638 | 5388 | baselines_bprna_new.json |
| UFold | bpRNA-new | 0.5328 | 0.7150 | 0.6106 | 0.6439 | 0.6499 | 5388 | baselines_ufold_bprna_new.json |
| Ours r2d (Plan-B, s0) | bpRNA-new | 0.5836 | 0.4389 | 0.5010 | 0.4714 | 0.4868 | 5388 | ow_rinalmo_r2d_b4_s0_step20000_bprna_new |
| Ours r2d (Plan-B, s1) | bpRNA-new | 0.5368 | 0.4679 | 0.5000 | 0.4900 | 0.5011 | 5388 | ow_rinalmo_r2d_b4_s1_step20000_bprna_new |
| Ours TR1 @40k | bpRNA-new | 0.4927 | 0.5074 | 0.4999 | 0.5083 | 0.5122 | 5388 | ow_rinalmo_ff_tr1_b4_s0_step40000_bprna_new |
| RNAformer (bprna ckpt) | bpRNA-new | 0.7039 | 0.3800 | 0.4936 | 0.4588 | 0.4820 | 5388 | baselines_rnaformer_bprna_new.json |
| Ours big | bpRNA-new | 0.6529 | 0.3600 | 0.4641 | 0.4487 | 0.4691 | 5388 | ow_rinalmo_big_b4_s0_step20000_bprna_new |
| Ours bigtr1 | bpRNA-new | 0.7007 | 0.3378 | 0.4558 | 0.4249 | 0.4555 | 5388 | ow_rinalmo_bigtr1_b4_s0_step20000_bprna_new |
| Ours Plan-A (adapted+2D, s0) | bpRNA-new | 0.5694 | 0.3457 | 0.4302 | 0.4156 | — | 5388 | plana_giga_s0_step20000 |
| Ours Plan-A ext40k @30k | bpRNA-new | 0.5933 | 0.3023 | 0.4005 | 0.3798 | — | 5388 | plana_giga_s0_ext40k_step30000 |
| Ours Plan-A ext40k @40k | bpRNA-new | 0.5949 | 0.2999 | 0.3988 | 0.3817 | — | 5388 | plana_giga_s0_ext40k_step40000 |
| Ours Plan-A s2 | bpRNA-new | 0.5881 | 0.3006 | 0.3979 | 0.3844 | — | 5388 | plana_giga_s2_step20000 |
| Ours Plan-A s1 | bpRNA-new | 0.6498 | 0.2648 | 0.3763 | 0.3569 | — | 5388 | plana_giga_s1_step20000 |
| RNAformer (bprna ckpt) | pdb_ts2 | 0.9401 | 0.7908 | 0.8590 | 0.8550 | 0.8633 | 39 | baselines_rnaformer_ref_pdb_ts2.json |
| RNAformer (bprna ckpt) | pdb_ts3 | 0.9444 | 0.9376 | 0.9410 | 0.9340 | 0.9354 | 19 | baselines_rnaformer_ref_pdb_ts3.json |

## Quoted (paper) rows — not measured here

| Method | Dataset | P | R | F1 | macro F1 | INF | n | source |
|---|---|---|---|---|---|---|---|---|
| NucleicBERT ft (quoted) | TS0 (Mathews macro) | 0.7180 | 0.6100 | — | 0.6490 | — | 1305 | NucleicBERT Table 1 (quoted) |
| RNAErnie+ (quoted) | TS0 (Mathews macro) | 0.5750 | 0.6780 | — | 0.6220 | — | 1305 | NucleicBERT Table 1 (quoted) |
| RNA-FM ft (quoted) | TS0 (Mathews macro) | 0.5180 | 0.6200 | — | 0.5640 | — | 1305 | NucleicBERT Table 1 (quoted) |
| RiNALMo-ft (quoted) | TestSetB | — | — | — | — | 0.6700 | 430 | RiNALMo Supp S5 (quoted) |
| CONTRAfold (quoted) | TestSetB | — | — | — | — | 0.6400 | 430 | RiNALMo Supp S5 (quoted) |
| MXfold2 (quoted) | TestSetB | — | — | — | — | 0.6300 | 430 | RiNALMo Supp S5 (quoted) |
| RNAstructure (quoted) | TestSetB | — | — | — | — | 0.5600 | 430 | RiNALMo Supp S5 (quoted) |
| RNA-FM (quoted) | TestSetB | — | — | — | — | 0.4900 | 430 | RiNALMo Supp S5 (quoted) |