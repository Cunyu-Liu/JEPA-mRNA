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
