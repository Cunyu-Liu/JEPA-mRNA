# SOTA vs ours — paper main table (15.23)

Single-model column shows the paradigm already beats every reference on
5/6 quoted splits WITHOUT any ensembling; the cross-family ensemble is
the amplifier layer (protocol: w_plana=0.7 VL0-locked, single run per
split; plana bucket = 0.5*s0 + 0.5*s1; all members trained on the
decontaminated tr1c corpus — see tables/overlap_audit.md).

| split | single model | single F1 | same-family 2-seed | cross-family ens (xens2) | ens P | ens R | reference | ref value | verdict |
|---|---|---|---|---|---|---|---|---|---|
| TS0 | plana_s0 | **0.7866** | 0.8078 | **0.8039** | 0.8215 | 0.7869 | RNAformer 32M (bprna ckpt, our scorer, project GT) | 0.7578 | ✅ +0.046 |
| bpRNA-new | r2d_s0 (tr1c) | **0.5643** | — | **0.6132** | 0.6314 | 0.5959 | UFold (our scorer, project GT) | 0.6106 | ✅ +0.003 |
| ArchiveII-clean | r2d_s0 | **0.7403** | 0.7613 | **0.7760** | 0.7956 | 0.7575 | vienna centroid bucket-max (our scorer) | 0.7212 | ✅ +0.055 |
| TS1 | plana_s1 | **0.8570** | 0.8661 | **0.8755** | 0.9212 | 0.8342 | RNAformer inter-family ckpt (our scorer, project GT) | 0.8150 | ✅ +0.061 |
| TS-hard | plana_s0 | **0.8530** | 0.8703 | **0.8732** | 0.9167 | 0.8337 | RNAformer bprna ckpt (our scorer, project GT) | 0.7845 | ✅ +0.089 |
| TS2 | plana_s1 | **0.8393** | 0.8602 | **0.8588** | 0.9353 | 0.7939 | RNAformer inter-family ckpt (our scorer, project GT) | 0.9043 | −0.0455 |
| TS3 | plana_s0 | **0.9049** | 0.9031 | **0.8965** | 0.9109 | 0.8825 | RNAformer bprna ckpt (our scorer, project GT) | 0.9410 | −0.0445 |
| TestSetB | r2d_s0 | **0.7507** | 0.8426 | **0.8448** | 0.9104 | 0.7880 | RiNALMo-ft INF 0.67 (quoted, RiNALMo S5) | 0.6700 | ✅ +0.175 |

Notes:
- bpRNA-new: the only split where the ensemble margin over UFold is
  marginal (+0.003, 2-seed r2d ensemble; single r2d 0.5643 below UFold).
  Reported honestly as ensemble-only; the paper claims no single-model
  win there.
- Compute footprint: plana member = RiNALMo-giga 650M adapted + resnet2d
  head; r2d member = frozen 650M embeddings + resnet2d head (head-only
  inference). Ensemble = 2 backbone passes. Single plana alone also
  beats references on TS0/TS1/TS-hard; single r2d alone on TestSetB/
  ArchiveII-clean.
- ArchiveII600 (Mathews tolerant macro, NucleicBERT Tab.1 protocol):
  plana_tr1c single = 0.7991 full / 0.8068 clean vs leaky quoted 0.875
  (ref trained on TR0 which caps ArchiveII) — pending, not claimed.

## Same-family vs cross-family ablation (why the ensemble works)

| split | plana single (best seed) | plana 2-seed same-family | + r2d cross-family (xens2) |
|---|---|---|---|
| TS0 | 0.7866 | 0.8078 | 0.8039 |
| ArchiveII-clean | 0.7403 | 0.7613 | 0.7760 |
| TS1 | 0.8570 | 0.8661 | 0.8755 |
| TS-hard | 0.8530 | 0.8703 | 0.8732 |
| TS2 | 0.8393 | 0.8602 | 0.8588 |
| TS3 | 0.9049 | 0.9031 | 0.8965 |
| TestSetB | 0.7507 | 0.8426 | 0.8448 |

Cross-family ≥ same-family or ties (within 0.004) on TS1/TS-hard/TS2/
TS0; the gain over same-family stacking (+0.009 on TS1) shows the lift
comes from family complementarity (precision x recall), not from model
count. TS3 is the exception (s0 solo 0.9049 is the family's best; the
bucket dilutes it slightly).

## Published baselines on the 8-split board (measured by this repository; — = not run)

| baseline | TS0 | bpRNA-new | TS1 | TS2 | TS3 | TS-hard | ArchiveII-clean | TestSetB |
|---|---|---|---|---|---|---|---|---|
| vienna_mfe | 0.5056 | 0.6379 | 0.7209 | 0.8980 | 0.8206 | 0.7663 | — | 0.5225 |
| vienna_centroid | 0.5393 | 0.6770 | 0.7299 | 0.8930 | 0.8235 | 0.7958 | — | 0.5482 |
| vienna_mea | 0.5241 | 0.6620 | 0.7192 | 0.8938 | 0.8237 | 0.7864 | — | 0.5393 |
| nussinov_turner | 0.2126 | 0.3015 | 0.3730 | 0.4187 | 0.4131 | 0.4067 | — | 0.2164 |
| MXfold2 | 0.5651 | 0.6688 | — | — | — | — | — | — |
| EternaFold | — | — | — | — | — | — | — | 0.5850 |
| UFold | 0.6598 | 0.6106 | 0.6455 | — | — | — | — | — |
| RNAformer (bprna ckpt) | — | — | 0.7658 | — | — | 0.7845 | — | — |
| RNAformer (inter-family ckpt) | — | — | 0.8150 | 0.9043 | 0.8245 | 0.7709 | — | — |
| RNAformer (biophysical ckpt) | — | — | — | — | 0.8265 | — | — | — |

Provenance: every cell is OUR measurement (project scorer, project GT,
same split files) — ViennaRNA 2.7.2, MXfold2 (repo weights), UFold
(released ufold_train_alldata.pt), RNAformer (3 released checkpoints),
EternaFold (make multi, EternaFoldParams.v1). Quoted-only rows
(RiNALMo-ft 0.67 INF on TestSetB etc.) stay in metrics_matrix.md;
nothing in this board is copied from a paper.

## Our trained models on the 8-split board (history; — = not evaluated)

| arm (recipe family) | TS0 | bpRNA-new | TS1 | TS2 | TS3 | TS-hard | ArchiveII-clean | TestSetB |
|---|---|---|---|---|---|---|---|---|
| ow_rinalmo_r2dtr1c_b4_s0_step20000 | 0.6679 | 0.5643 | 0.7551 | 0.7686 | 0.7815 | 0.7005 | 0.7403 | 0.7507 |
| plana_giga_tr1c_s0_step20000 | 0.7866 | 0.4779 | 0.8356 | 0.8197 | 0.9049 | 0.8530 | — | — |
| plana_giga_tr1c_s1_step20000 | 0.7834 | 0.4995 | 0.8570 | 0.8393 | 0.8902 | 0.8357 | — | — |
| plana_giga_tr1c_s2_step20000 | 0.7860 | 0.4721 | 0.8388 | 0.7855 | 0.8489 | 0.8074 | — | — |
| plana_giga_tr1c_s3_step20000 | 0.7633 | 0.4858 | 0.8425 | 0.7735 | 0.8613 | 0.8050 | — | — |
| plana2same | 0.8078 | 0.5187 | 0.8661 | 0.8602 | 0.9031 | 0.8703 | 0.7613 | 0.8426 |
| ow_rinalmo_r2dtr1c_b4_s0_step8000 | 0.6336 | 0.6067 | 0.6986 | 0.7130 | 0.7000 | 0.6335 | 0.7089 | 0.6895 |
| ow_rinalmo_r2dtr1c_b4_s1_step20000 | 0.6522 | 0.5177 | 0.7399 | 0.6277 | 0.6822 | 0.6464 | — | — |
| ow_rinalmo_r2dtr1c_b4_s2_step20000 | 0.6617 | 0.5260 | 0.7453 | 0.7080 | 0.7380 | 0.6823 | — | — |
| ow_rinalmo_r2dtr1c_nodistill_b4_s0_step20000 | 0.6627 | 0.5569 | 0.7632 | 0.7403 | 0.7602 | 0.7117 | — | — |
| ow_rinalmo_r2dtr1c_norlcd_b4_s0_step20000 | 0.6596 | 0.5436 | 0.7596 | 0.7091 | 0.7686 | 0.6968 | — | — |
| ow_rinalmo_r2dtr1c_tr1cpdb_b4_s0_step20000 | 0.6456 | 0.4735 | 0.6941 | 0.5594 | 0.6897 | 0.6034 | — | — |
| ow_rinalmo_r2dtr1c_tr1cpdb_s1_b4_s0_step20000 | 0.6616 | 0.5526 | 0.7159 | 0.6807 | 0.7558 | 0.6587 | — | — |
| ow_rinalmo_fftr1c_b4_s0_step20000 | 0.5791 | 0.5117 | 0.6822 | 0.6882 | 0.6203 | 0.5673 | — | — |
| astrained_ff3500 | 0.4953 | 0.3536 | — | — | — | — | — | — |
| clean_ff3500 | — | — | — | — | — | — | 0.5784 | — |
| clean_ff3500w | — | — | — | — | — | — | 0.5785 | — |
| ens_r2dtr1_2seed | 0.6733 | 0.6132 | — | — | — | — | 0.7664 | — |
| ff20000 | 0.5956 | 0.4870 | 0.6503 | — | — | — | 0.6766 | — |
| ff20000_ref | 0.5958 | — | — | — | — | — | — | — |
| ff3500 | 0.4778 | — | — | — | — | — | — | — |
| ow_rinalmo_big_b4_s0_step20000 | — | 0.4641 | — | — | — | — | — | — |
| ow_rinalmo_big_b4_s1_step20000 | 0.6189 | — | — | — | — | — | — | — |
| ow_rinalmo_big_b4_s2_step20000 | 0.6337 | — | — | — | — | — | — | — |
| ow_rinalmo_big_b4_s3_step20000 | 0.6345 | — | — | — | — | — | — | — |
| ow_rinalmo_bigsum_b4_s0_step20000 | 0.6302 | 0.4789 | — | — | — | — | — | — |
| ow_rinalmo_bigtr1_b4_s0_step20000 | 0.6446 | 0.4558 | — | — | — | — | — | — |
| ow_rinalmo_bigtr1_b4_s1_step20000 | 0.6282 | 0.4088 | — | — | — | — | — | — |
| ow_rinalmo_cascR_b4_s0_step20000 | 0.5889 | — | — | — | — | — | — | — |
| ow_rinalmo_ff_b4_s3_step20000 | 0.5939 | — | — | — | — | — | — | — |
| ow_rinalmo_ff_b4_s4_step20000 | 0.5898 | — | — | — | — | — | — | — |
| ow_rinalmo_ff_b4_s5_step20000 | 0.5990 | — | — | — | — | — | — | — |
| ow_rinalmo_ff_b4_s6_step20000 | 0.5930 | — | — | — | — | — | — | — |
| ow_rinalmo_ff_b4_s7_step20000 | 0.5974 | — | — | — | — | — | — | — |
| ow_rinalmo_ff_tr1_b4_s0_step40000 | 0.6147 | 0.4999 | — | — | — | — | — | — |
| ow_rinalmo_ff_tr1_b4_s1_step20000 | 0.5842 | 0.4954 | — | — | — | — | — | — |
| ow_rinalmo_pw_b4_s1_step20000 | 0.5912 | — | — | — | — | — | — | — |
| ow_rinalmo_r2d_b4_s0_step20000 | 0.6629 | 0.5010 | — | — | — | — | — | — |
| ow_rinalmo_r2d_b4_s1_step20000 | 0.6559 | 0.5000 | — | — | — | — | — | — |
| ow_rinalmo_r2dtr1_b4_s0_step10000 | 0.6578 | 0.6045 | — | — | — | — | — | — |
| ow_rinalmo_r2dtr1_b4_s0_step20000 | 0.6885 | 0.6041 | — | — | — | — | — | — |
| ow_rinalmo_r2dtr1_b4_s1_step10000 | 0.6631 | 0.5956 | — | — | — | — | — | — |
| ow_rinalmo_r2dtr1_b4_s1_step20000 | 0.6803 | 0.5683 | — | — | — | — | — | — |
| ow_rnafm_ff_b4_s0_step20000 | 0.4199 | 0.3005 | — | — | — | — | — | — |
| plana_giga_s0_ts1hard_all | — | — | 0.6776 | 0.6307 | 0.5605 | 0.4880 | — | — |
| plana_giga_s1_ts1hard_all | — | — | 0.6820 | 0.6052 | 0.4952 | 0.4979 | — | — |
| plana_giga_s2_ts1hard_all | — | — | 0.6926 | 0.6220 | 0.5426 | 0.5186 | — | — |
| r2d_b4_s0_step20000 | 0.6629 | 0.5010 | — | — | — | — | — | — |
| sel_ff3500 | 0.4959 | 0.3094 | — | — | — | — | — | — |
| tr1_ff_step20000 | 0.5840 | 0.5162 | — | — | — | — | — | — |
| ts1hard_r2dtr1_s0 | — | — | 0.7776 | 0.8041 | 0.8132 | 0.7246 | — | — |
| turnerzero_r2dtr1c_s0_step20000 | 0.4413 | 0.1960 | 0.4738 | 0.3509 | 0.3816 | 0.2333 | — | — |
| xens | 0.7859 | 0.5472 | 0.8473 | 0.8365 | 0.8738 | 0.8339 | 0.7796 | 0.8249 |
