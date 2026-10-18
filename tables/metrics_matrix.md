# Full metrics matrix — every measured (method, dataset) cell

All values re-derived from result.json / baselines_*.json on disk
(script tools/metrics_matrix.py; no hand-typed numbers). Quoted
rows carry their paper source and are flagged. '—' = the metric is
not produced by that run's protocol (e.g. INF needs per-seq rows).

## Measured cells

| Method | Dataset | micro P | micro R | micro F1 | macro F1 | INF | n | source |
|---|---|---|---|---|---|---|---|---|
| Ours xens (plana_tr1c x r2dtr1c, w0.5 sweep) | ArchiveII-clean | 0.7996 | 0.7606 | 0.7796 | 0.8129 | — | 2544 | xens_archiveii_embok_clean |
| Ours xens2 (plana 2-seed avg x r2d, w0.7) | ArchiveII-clean | 0.7956 | 0.7575 | 0.7760 | 0.8095 | — | 2544 | xens2_archiveii_embok_clean_w0.7 |
| Ours xens (plana_tr1c x r2dtr1c, VL0-locked w0.7) | ArchiveII-clean | 0.8002 | 0.7415 | 0.7697 | 0.8030 | — | 2544 | xens_archiveii_embok_clean_w0.7 |
| Ours r2d_tr1 2-seed ensemble @10k | ArchiveII-clean | 0.7548 | 0.7784 | 0.7664 | 0.8113 | — | 2544 | ens_r2dtr1_2seed_archiveii_embok_clean |
| Ours r2d_tr1c (clean, s0 @20k) ArchiveII-clean | ArchiveII-clean | 0.7230 | 0.7583 | 0.7403 | 0.7974 | 0.7995 | 2544 | ow_rinalmo_r2dtr1c_b4_s0_step20000_archiveii_embok_clean |
| Ours r2d_tr1c (clean, s0 @8k) ArchiveII-clean | ArchiveII-clean | 0.7218 | 0.6964 | 0.7089 | 0.7684 | 0.7722 | 2544 | ow_rinalmo_r2dtr1c_b4_s0_step8000_archiveii_embok_clean |
| vienna_centroid | ArchiveII-clean 100-200 | 0.7144 | 0.6607 | 0.6865 | 0.6725 | 0.6774 | 1136 | baselines_archiveii_embok_clean_gt100_le200.json |
| vienna_centroid | ArchiveII-clean 200-400 | 0.5629 | 0.6128 | 0.5868 | 0.5790 | 0.5824 | 728 | baselines_archiveii_embok_clean_gt200_le400.json |
| vienna_centroid | ArchiveII-clean <=100 | 0.7305 | 0.7122 | 0.7212 | 0.7028 | 0.7067 | 484 | baselines_archiveii_embok_clean_le100.json |
| vienna_centroid | ArchiveII-clean >400 | 0.4431 | 0.6734 | 0.5345 | 0.5262 | 0.5405 | 196 | baselines_archiveii_embok_clean_gt400.json |
| Ours xens2 (plana 2-seed avg x r2d, w0.7) | TS-hard | 0.9167 | 0.8337 | 0.8732 | 0.8383 | — | 28 | xens2_ref_pdb_ts_hard_w0.7 |
| Ours xens (plana_tr1c x r2dtr1c, VL0-locked w0.7) | TS-hard | 0.9251 | 0.8063 | 0.8616 | 0.8065 | — | 28 | xens_ref_pdb_ts_hard_w0.7 |
| Ours xens TS-hard w0.7 (lock-verify / exploratory) | TS-hard | 0.9251 | 0.8063 | 0.8616 | 0.8065 | — | 28 | xens_ref_pdb_ts_hard_w0.7 |
| Ours plana_tr1c s1 (adapted+2D, clean) | TS-hard | 0.8649 | 0.8084 | 0.8357 | 0.8041 | — | 28 | plana_giga_tr1c_s1_step20000 |
| Ours xens (plana_tr1c x r2dtr1c, w0.5 sweep) | TS-hard | 0.9000 | 0.7768 | 0.8339 | 0.7622 | — | 28 | xens_ref_pdb_ts_hard |
| vienna_centroid | TS-hard | 0.8000 | 0.7916 | 0.7958 | 0.7463 | 0.7506 | 28 | baselines_ref_pdb_ts_hard.json |
| vienna_mea | TS-hard | 0.7675 | 0.8063 | 0.7864 | 0.7641 | 0.7672 | 28 | baselines_ref_pdb_ts_hard.json |
| RNAformer (bprna ckpt) | TS-hard | 0.8557 | 0.7242 | 0.7845 | 0.7108 | 0.7145 | 28 | baselines_rnaformer_bprna_ref_pdb_ts_hard.json |
| RNAformer (inter-family ckpt, paper Tab.4 setting) | TS-hard | 0.7960 | 0.7474 | 0.7709 | 0.7643 | 0.7691 | 28 | baselines_rnaformer_interfam_ref_pdb_ts_hard.json |
| vienna_mfe | TS-hard | 0.7318 | 0.8042 | 0.7663 | 0.7480 | 0.7511 | 28 | baselines_ref_pdb_ts_hard.json |
| Ours r2d_tr1 (RETRACTED, leaky tr1) | TS-hard | 0.7701 | 0.6842 | 0.7246 | 0.6663 | 0.6725 | 28 | ts1hard_r2dtr1_s0_ref_pdb_ts_hard |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @20k) | TS-hard | 0.7735 | 0.6400 | 0.7005 | 0.6372 | 0.6443 | 28 | ow_rinalmo_r2dtr1c_b4_s0_step20000_ref_pdb_ts_hard |
| Ours r2d_tr1c (seed=1, worse basin) | TS-hard | 0.8121 | 0.5368 | 0.6464 | 0.5748 | 0.5849 | 28 | ow_rinalmo_r2dtr1c_b4_s1_step20000_ref_pdb_ts_hard |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @8k) | TS-hard | 0.7026 | 0.5768 | 0.6335 | 0.6589 | 0.6676 | 28 | ow_rinalmo_r2dtr1c_b4_s0_step8000_ref_pdb_ts_hard |
| Ours Plan-A s2 (PDB family) | TS-hard | 0.8117 | 0.3811 | 0.5186 | 0.5369 | — | 28 | plana_giga_s2_ts1hard_all |
| Ours Plan-A s1 (PDB family) | TS-hard | 0.7500 | 0.3726 | 0.4979 | 0.5320 | — | 28 | plana_giga_s1_ts1hard_all |
| Ours Plan-A s0 (PDB family) | TS-hard | 0.6655 | 0.3853 | 0.4880 | 0.4919 | — | 28 | plana_giga_s0_ts1hard_all |
| nussinov_turner | TS-hard | 0.4072 | 0.4063 | 0.4067 | 0.4183 | 0.4208 | 28 | baselines_ref_pdb_ts_hard.json |
| Ours xens2 (plana 2-seed avg x r2d, w0.7) | TS0 | 0.8215 | 0.7869 | 0.8039 | 0.8002 | — | 1288 | xens2_bprna_ts0_w0.7 |
| Ours xens (plana_tr1c x r2dtr1c, VL0-locked w0.7) | TS0 | 0.8184 | 0.7642 | 0.7904 | 0.7857 | — | 1288 | xens_bprna_ts0_w0.7 |
| Ours xens (plana_tr1c x r2dtr1c, w0.5 sweep) | TS0 | 0.8040 | 0.7686 | 0.7859 | 0.7809 | — | 1288 | xens_bprna_ts0 |
| Ours plana_tr1c s1 (adapted+2D, clean) | TS0 | 0.7930 | 0.7741 | 0.7834 | 0.7890 | — | 1288 | plana_giga_tr1c_s1_step20000 |
| RNAformer (bprna ckpt) | TS0 | 0.8091 | 0.7127 | 0.7578 | 0.7454 | 0.7512 | 1291 | baselines_rnaformer_ref_bprna_ts0.json |
| Ours Plan-A s2 | TS0 | 0.8001 | 0.6773 | 0.7336 | 0.7129 | — | 1288 | plana_giga_s2_step20000 |
| Ours Plan-A (adapted+2D, s0) | TS0 | 0.7689 | 0.6891 | 0.7268 | 0.7139 | — | 1288 | plana_giga_s0_step20000 |
| Ours Plan-A s1 | TS0 | 0.8300 | 0.6428 | 0.7245 | 0.6989 | — | 1288 | plana_giga_s1_step20000 |
| Ours Plan-A ext40k @30k | TS0 | 0.7818 | 0.6689 | 0.7210 | 0.7044 | — | 1288 | plana_giga_s0_ext40k_step30000 |
| Ours Plan-A ext40k @40k | TS0 | 0.7947 | 0.6544 | 0.7178 | 0.6974 | — | 1288 | plana_giga_s0_ext40k_step40000 |
| Ours r2d_tr1 (frozen+2D+TR1, s0 @20000k) | TS0 | 0.6500 | 0.7319 | 0.6885 | 0.6920 | 0.6975 | 1288 | ow_rinalmo_r2dtr1_b4_s0_step20000_bprna_ts0 |
| Ours r2d_tr1 2-seed ensemble @10k (leaky corpus; new split clean) | TS0 | 0.6434 | 0.7061 | 0.6733 | 0.6689 | — | 1288 | ens_r2dtr1_2seed_bprna_ts0 |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @20k) | TS0 | 0.6393 | 0.6993 | 0.6679 | 0.6699 | 0.6748 | 1288 | ow_rinalmo_r2dtr1c_b4_s0_step20000_bprna_ts0 |
| Ours r2d (Plan-B, frozen+2D, s0) | TS0 | 0.6865 | 0.6410 | 0.6629 | 0.6436 | 0.6499 | 1288 | ow_rinalmo_r2d_b4_s0_step20000_bprna_ts0 |
| Ours r2d_tr1 (frozen+2D+TR1, s0 @10000k) | TS0 | 0.6234 | 0.6963 | 0.6578 | 0.6597 | 0.6651 | 1288 | ow_rinalmo_r2dtr1_b4_s0_step10000_bprna_ts0 |
| Ours r2d (Plan-B, s1) | TS0 | 0.6487 | 0.6632 | 0.6559 | 0.6470 | 0.6525 | 1288 | ow_rinalmo_r2d_b4_s1_step20000_bprna_ts0 |
| Ours r2d_tr1c (seed=1, worse basin) | TS0 | 0.6701 | 0.6353 | 0.6522 | 0.6397 | 0.6456 | 1288 | ow_rinalmo_r2dtr1c_b4_s1_step20000_bprna_ts0 |
| Ours bigtr1 (capacity x data) | TS0 | 0.7419 | 0.5699 | 0.6446 | 0.6128 | 0.6233 | 1288 | ow_rinalmo_bigtr1_b4_s0_step20000_bprna_ts0 |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @8k) | TS0 | 0.6150 | 0.6533 | 0.6336 | 0.6369 | 0.6426 | 1288 | ow_rinalmo_r2dtr1c_b4_s0_step8000_bprna_ts0 |
| Ours TR1 @40k | TS0 | 0.5763 | 0.6586 | 0.6147 | 0.6288 | 0.6335 | 1288 | ow_rinalmo_ff_tr1_b4_s0_step40000_bprna_ts0 |
| MXfold2 | TS0 | 0.4816 | 0.6838 | 0.5651 | 0.5698 | 0.5832 | 1288 | baselines_mxfold2_bprna_ts0.json |
| ViennaRNA centroid | TS0 | 0.4710 | 0.6306 | 0.5393 | 0.5288 | 0.5405 | 1288 | baselines_bprna_ts0.json |
| ViennaRNA mea | TS0 | 0.4413 | 0.6451 | 0.5241 | 0.5218 | 0.5346 | 1288 | baselines_bprna_ts0.json |
| ViennaRNA mfe | TS0 | 0.4181 | 0.6391 | 0.5055 | 0.5086 | 0.5222 | 1288 | baselines_bprna_ts0.json |
| Ours xens2 (plana 2-seed avg x r2d, w0.7) | TS1 | 0.9212 | 0.8342 | 0.8755 | 0.8443 | — | 63 | xens2_ref_pdb_ts1_w0.7 |
| Ours plana_tr1c s1 (adapted+2D, clean) | TS1 | 0.8922 | 0.8245 | 0.8570 | 0.8214 | — | 63 | plana_giga_tr1c_s1_step20000 |
| Ours xens (plana_tr1c x r2dtr1c, w0.5 sweep) | TS1 | 0.9110 | 0.7918 | 0.8473 | 0.8003 | — | 63 | xens_ref_pdb_ts1 |
| Ours xens (plana_tr1c x r2dtr1c, VL0-locked w0.7) | TS1 | 0.8987 | 0.7851 | 0.8381 | 0.7845 | — | 63 | xens_ref_pdb_ts1_w0.7 |
| Ours xens TS1 w0.7 (lock-verify / exploratory) | TS1 | 0.8987 | 0.7851 | 0.8381 | 0.7845 | — | 63 | xens_ref_pdb_ts1_w0.7 |
| RNAformer (inter-family ckpt, paper Tab.4 setting) | TS1 | 0.8404 | 0.7911 | 0.8150 | 0.7898 | 0.7941 | 63 | baselines_rnaformer_interfam_ref_pdb_ts1.json |
| Ours r2d_tr1 (RETRACTED, leaky tr1) | TS1 | 0.8416 | 0.7227 | 0.7776 | 0.7201 | 0.7281 | 63 | ts1hard_r2dtr1_s0_ref_pdb_ts1 |
| RNAformer (bprna ckpt) | TS1 | 0.8711 | 0.6833 | 0.7658 | 0.7230 | 0.7322 | 63 | baselines_rnaformer_bprna_ref_pdb_ts1.json |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @20k) | TS1 | 0.8305 | 0.6922 | 0.7551 | 0.7130 | 0.7219 | 63 | ow_rinalmo_r2dtr1c_b4_s0_step20000_ref_pdb_ts1 |
| Ours r2d_tr1c (seed=1, worse basin) | TS1 | 0.8792 | 0.6387 | 0.7399 | 0.6556 | 0.6693 | 63 | ow_rinalmo_r2dtr1c_b4_s1_step20000_ref_pdb_ts1 |
| vienna_centroid | TS1 | 0.7452 | 0.7152 | 0.7299 | 0.7168 | 0.7210 | 63 | baselines_ref_pdb_ts1.json |
| vienna_mfe | TS1 | 0.6983 | 0.7450 | 0.7209 | 0.7244 | 0.7270 | 63 | baselines_ref_pdb_ts1.json |
| vienna_mea | TS1 | 0.7066 | 0.7323 | 0.7192 | 0.7232 | 0.7256 | 63 | baselines_ref_pdb_ts1.json |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @8k) | TS1 | 0.8037 | 0.6178 | 0.6986 | 0.6676 | 0.6791 | 63 | ow_rinalmo_r2dtr1c_b4_s0_step8000_ref_pdb_ts1 |
| Ours Plan-A s2 (PDB family) | TS1 | 0.8835 | 0.5695 | 0.6926 | 0.6440 | — | 63 | plana_giga_s2_ts1hard_all |
| Ours Plan-A s1 (PDB family) | TS1 | 0.8670 | 0.5621 | 0.6820 | 0.6357 | — | 63 | plana_giga_s1_ts1hard_all |
| Ours Plan-A s0 (PDB family) | TS1 | 0.8346 | 0.5703 | 0.6776 | 0.6192 | — | 63 | plana_giga_s0_ts1hard_all |
| Ours ff | TS1 | 0.7206 | 0.5926 | 0.6503 | 0.6291 | 0.6366 | 63 | ff20000_ref_pdb_ts1 |
| UFold | TS1 | 0.6087 | 0.6870 | 0.6455 | 0.6518 | 0.6554 | 63 | baselines_ufold_ref_pdb_ts1.json |
| nussinov_turner | TS1 | 0.3872 | 0.3599 | 0.3730 | 0.3972 | 0.3997 | 63 | baselines_ref_pdb_ts1.json |
| RNAformer (bprna ckpt) | TS2 | 0.9401 | 0.7908 | 0.8590 | 0.8550 | 0.8633 | 39 | baselines_rnaformer_ref_pdb_ts2.json |
| Ours xens2 (plana 2-seed avg x r2d, w0.7) | TS2 | 0.9353 | 0.7939 | 0.8588 | 0.8407 | — | 39 | xens2_ref_pdb_ts2_w0.7 |
| Ours xens (plana_tr1c x r2dtr1c, VL0-locked w0.7) | TS2 | 0.9433 | 0.7618 | 0.8429 | 0.8234 | — | 39 | xens_ref_pdb_ts2_w0.7 |
| Ours xens TS2 w0.7 (lock-verify / exploratory) | TS2 | 0.9433 | 0.7618 | 0.8429 | 0.8234 | — | 39 | xens_ref_pdb_ts2_w0.7 |
| Ours plana_tr1c s1 (adapted+2D, clean) | TS2 | 0.9062 | 0.7817 | 0.8393 | 0.8364 | — | 39 | plana_giga_tr1c_s1_step20000 |
| Ours xens (plana_tr1c x r2dtr1c, w0.5 sweep) | TS2 | 0.9461 | 0.7496 | 0.8365 | 0.8189 | — | 39 | xens_ref_pdb_ts2 |
| Ours xens TS2 w0.9 (lock-verify / exploratory) | TS2 | 0.9388 | 0.7496 | 0.8336 | 0.8169 | — | 39 | xens_ref_pdb_ts2_w0.9 |
| Ours r2d_tr1 (RETRACTED, leaky tr1) | TS2 | 0.9194 | 0.7145 | 0.8041 | 0.7929 | 0.8029 | 39 | ts1hard_r2dtr1_s0_ref_pdb_ts2 |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @20k) | TS2 | 0.8980 | 0.6718 | 0.7686 | 0.7363 | 0.7475 | 39 | ow_rinalmo_r2dtr1c_b4_s0_step20000_ref_pdb_ts2 |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @8k) | TS2 | 0.8720 | 0.6031 | 0.7130 | 0.6880 | 0.7045 | 39 | ow_rinalmo_r2dtr1c_b4_s0_step8000_ref_pdb_ts2 |
| Ours Plan-A s0 (PDB family) | TS2 | 0.8560 | 0.4992 | 0.6307 | 0.6134 | — | 39 | plana_giga_s0_ts1hard_all |
| Ours r2d_tr1c (seed=1, worse basin) | TS2 | 0.8930 | 0.4840 | 0.6277 | 0.6027 | 0.6363 | 39 | ow_rinalmo_r2dtr1c_b4_s1_step20000_ref_pdb_ts2 |
| Ours Plan-A s2 (PDB family) | TS2 | 0.8753 | 0.4824 | 0.6220 | 0.6086 | — | 39 | plana_giga_s2_ts1hard_all |
| Ours Plan-A s1 (PDB family) | TS2 | 0.8805 | 0.4611 | 0.6052 | 0.5786 | — | 39 | plana_giga_s1_ts1hard_all |
| RNAformer (bprna ckpt) | TS3 | 0.9444 | 0.9376 | 0.9410 | 0.9340 | 0.9354 | 19 | baselines_rnaformer_ref_pdb_ts3.json |
| Ours xens (plana_tr1c x r2dtr1c, VL0-locked w0.7) | TS3 | 0.9271 | 0.8849 | 0.9055 | 0.8600 | — | 19 | xens_ref_pdb_ts3_w0.7 |
| Ours xens TS3 w0.7 (lock-verify / exploratory) | TS3 | 0.9271 | 0.8849 | 0.9055 | 0.8600 | — | 19 | xens_ref_pdb_ts3_w0.7 |
| Ours xens2 (plana 2-seed avg x r2d, w0.7) | TS3 | 0.9109 | 0.8825 | 0.8965 | 0.8612 | — | 19 | xens2_ref_pdb_ts3_w0.7 |
| Ours xens TS3 w0.9 (lock-verify / exploratory) | TS3 | 0.9104 | 0.8777 | 0.8938 | 0.8345 | — | 19 | xens_ref_pdb_ts3_w0.9 |
| Ours plana_tr1c s1 (adapted+2D, clean) | TS3 | 0.8956 | 0.8849 | 0.8902 | 0.8395 | — | 19 | plana_giga_tr1c_s1_step20000 |
| Ours xens (plana_tr1c x r2dtr1c, w0.5 sweep) | TS3 | 0.9028 | 0.8465 | 0.8738 | 0.8191 | — | 19 | xens_ref_pdb_ts3 |
| Ours r2d_tr1 (RETRACTED, leaky tr1) | TS3 | 0.8284 | 0.7986 | 0.8132 | 0.7579 | 0.7595 | 19 | ts1hard_r2dtr1_s0_ref_pdb_ts3 |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @20k) | TS3 | 0.8151 | 0.7506 | 0.7815 | 0.7299 | 0.7325 | 19 | ow_rinalmo_r2dtr1c_b4_s0_step20000_ref_pdb_ts3 |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @8k) | TS3 | 0.7521 | 0.6547 | 0.7000 | 0.6419 | 0.6447 | 19 | ow_rinalmo_r2dtr1c_b4_s0_step8000_ref_pdb_ts3 |
| Ours r2d_tr1c (seed=1, worse basin) | TS3 | 0.7955 | 0.5971 | 0.6822 | 0.6486 | 0.6606 | 19 | ow_rinalmo_r2dtr1c_b4_s1_step20000_ref_pdb_ts3 |
| Ours Plan-A s0 (PDB family) | TS3 | 0.7280 | 0.4556 | 0.5605 | 0.5593 | — | 19 | plana_giga_s0_ts1hard_all |
| Ours Plan-A s2 (PDB family) | TS3 | 0.7926 | 0.4125 | 0.5426 | 0.5245 | — | 19 | plana_giga_s2_ts1hard_all |
| Ours Plan-A s1 (PDB family) | TS3 | 0.7416 | 0.3717 | 0.4952 | 0.4728 | — | 19 | plana_giga_s1_ts1hard_all |
| Ours xens2 (plana 2-seed avg x r2d, w0.7) | TestSetB | 0.9104 | 0.7880 | 0.8448 | 0.8372 | — | 428 | xens2_testsetb_w0.7 |
| Ours xens (plana_tr1c x r2dtr1c, VL0-locked w0.7) | TestSetB | 0.8986 | 0.7677 | 0.8280 | 0.8162 | — | 428 | xens_testsetb_w0.7 |
| Ours xens (plana_tr1c x r2dtr1c, w0.5 sweep) | TestSetB | 0.8873 | 0.7707 | 0.8249 | 0.8153 | — | 428 | xens_testsetb |
| Ours r2d_tr1c (clean, s0 @20k) TestSetB | TestSetB | 0.7617 | 0.7400 | 0.7507 | 0.7542 | 0.7575 | 428 | ow_rinalmo_r2dtr1c_b4_s0_step20000_testsetb |
| Ours r2d_tr1c (clean, s0 @8k) TestSetB | TestSetB | 0.7412 | 0.6446 | 0.6895 | 0.6980 | 0.7059 | 428 | ow_rinalmo_r2dtr1c_b4_s0_step8000_testsetb |
| vienna_centroid | TestSetB | 0.5125 | 0.5893 | 0.5482 | 0.5577 | 0.5630 | 428 | baselines_testsetb.json |
| vienna_mfe | TestSetB | 0.4578 | 0.6085 | 0.5225 | 0.5430 | 0.5488 | 428 | baselines_testsetb.json |
| nussinov_turner | TestSetB | 0.1970 | 0.2401 | 0.2164 | 0.2251 | 0.2270 | 428 | baselines_testsetb.json |
| Ours xens2 VL0 w0.7 (weight-selection split) | VL0 | 0.8846 | 0.8733 | 0.8789 | 0.8661 | — | 196 | xens2_bprna_vl0_w0.7 |
| Ours xens2 VL0 w0.85 (weight-selection split) | VL0 | 0.8859 | 0.8707 | 0.8782 | 0.8647 | — | 196 | xens2_bprna_vl0_w0.85 |
| Ours xens2 VL0 w0.5 (weight-selection split) | VL0 | 0.8770 | 0.8755 | 0.8762 | 0.8579 | — | 196 | xens2_bprna_vl0_w0.5 |
| Ours xens VL0 w0.85 (weight-selection split) | VL0 | 0.8799 | 0.8616 | 0.8707 | 0.8561 | — | 196 | xens_bprna_vl0_w0.85 |
| ViennaRNA centroid | bpRNA-new | 0.6351 | 0.7249 | 0.6770 | 0.6821 | 0.6871 | 5388 | baselines_bprna_new.json |
| MXfold2 | bpRNA-new | 0.6092 | 0.7414 | 0.6688 | 0.6819 | 0.6868 | 5388 | baselines_mxfold2_bprna_new.json |
| ViennaRNA mea | bpRNA-new | 0.5972 | 0.7426 | 0.6620 | 0.6764 | 0.6814 | 5388 | baselines_bprna_new.json |
| ViennaRNA mfe | bpRNA-new | 0.5646 | 0.7332 | 0.6379 | 0.6581 | 0.6638 | 5388 | baselines_bprna_new.json |
| Ours r2d_tr1 2-seed ensemble @10k | bpRNA-new | 0.6314 | 0.5959 | 0.6132 | 0.6125 | — | 5388 | ens_r2dtr1_2seed_bprna_new |
| UFold | bpRNA-new | 0.5328 | 0.7150 | 0.6106 | 0.6439 | 0.6499 | 5388 | baselines_ufold_bprna_new.json |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @8k) | bpRNA-new | 0.6244 | 0.5900 | 0.6067 | 0.6064 | 0.6139 | 5388 | ow_rinalmo_r2dtr1c_b4_s0_step8000_bprna_new |
| Ours r2d_tr1 (frozen+2D+TR1, s0 @10000k) | bpRNA-new | 0.6073 | 0.6017 | 0.6045 | 0.6034 | 0.6097 | 5388 | ow_rinalmo_r2dtr1_b4_s0_step10000_bprna_new |
| Ours r2d_tr1 (frozen+2D+TR1, s0 @20000k) | bpRNA-new | 0.6104 | 0.5980 | 0.6041 | 0.5955 | 0.6012 | 5388 | ow_rinalmo_r2dtr1_b4_s0_step20000_bprna_new |
| Ours r2d_tr1c (frozen+2D+clean TR1, s0 @20k) | bpRNA-new | 0.5992 | 0.5332 | 0.5643 | 0.5592 | 0.5679 | 5388 | ow_rinalmo_r2dtr1c_b4_s0_step20000_bprna_new |
| Ours xens2 (plana 2-seed avg x r2d, w0.7) | bpRNA-new | 0.6574 | 0.4776 | 0.5532 | 0.5423 | — | 5388 | xens2_bprna_new_w0.7 |
| Ours xens (plana_tr1c x r2dtr1c, w0.5 sweep) | bpRNA-new | 0.6566 | 0.4691 | 0.5472 | 0.5326 | — | 5388 | xens_bprna_new |
| Ours xens (plana_tr1c x r2dtr1c, VL0-locked w0.7) | bpRNA-new | 0.6451 | 0.4339 | 0.5188 | 0.5035 | — | 5388 | xens_bprna_new_w0.7 |
| Ours r2d_tr1c (seed=1, worse basin) | bpRNA-new | 0.5984 | 0.4561 | 0.5177 | 0.5057 | 0.5206 | 5388 | ow_rinalmo_r2dtr1c_b4_s1_step20000_bprna_new |
| Ours r2d (Plan-B, s0) | bpRNA-new | 0.5836 | 0.4389 | 0.5010 | 0.4714 | 0.4868 | 5388 | ow_rinalmo_r2d_b4_s0_step20000_bprna_new |
| Ours r2d (Plan-B, s1) | bpRNA-new | 0.5368 | 0.4679 | 0.5000 | 0.4900 | 0.5011 | 5388 | ow_rinalmo_r2d_b4_s1_step20000_bprna_new |
| Ours TR1 @40k | bpRNA-new | 0.4927 | 0.5074 | 0.4999 | 0.5083 | 0.5122 | 5388 | ow_rinalmo_ff_tr1_b4_s0_step40000_bprna_new |
| Ours plana_tr1c s1 (adapted+2D, clean) | bpRNA-new | 0.5619 | 0.4496 | 0.4995 | 0.4979 | — | 5388 | plana_giga_tr1c_s1_step20000 |
| RNAformer (bprna ckpt) | bpRNA-new | 0.7039 | 0.3800 | 0.4936 | 0.4588 | 0.4820 | 5388 | baselines_rnaformer_bprna_new.json |
| Ours big | bpRNA-new | 0.6529 | 0.3600 | 0.4641 | 0.4487 | 0.4691 | 5388 | ow_rinalmo_big_b4_s0_step20000_bprna_new |
| Ours bigtr1 | bpRNA-new | 0.7007 | 0.3378 | 0.4558 | 0.4249 | 0.4555 | 5388 | ow_rinalmo_bigtr1_b4_s0_step20000_bprna_new |
| Ours Plan-A (adapted+2D, s0) | bpRNA-new | 0.5694 | 0.3457 | 0.4302 | 0.4156 | — | 5388 | plana_giga_s0_step20000 |
| Ours Plan-A ext40k @30k | bpRNA-new | 0.5933 | 0.3023 | 0.4005 | 0.3798 | — | 5388 | plana_giga_s0_ext40k_step30000 |
| Ours Plan-A ext40k @40k | bpRNA-new | 0.5949 | 0.2999 | 0.3988 | 0.3817 | — | 5388 | plana_giga_s0_ext40k_step40000 |
| Ours Plan-A s2 | bpRNA-new | 0.5881 | 0.3006 | 0.3979 | 0.3844 | — | 5388 | plana_giga_s2_step20000 |
| Ours Plan-A s1 | bpRNA-new | 0.6498 | 0.2648 | 0.3763 | 0.3569 | — | 5388 | plana_giga_s1_step20000 |

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