# PDB/Rfam label-lineage audit (2026-11-27)

External Test is audit-only. No prediction, metric, checkpoint, or model-selection outcome is used.

| Panel | Rows | Internal projection | TR1c exact overlap | Cross-panel overlap | Independent label qualification |
|---|---:|---|---:|---:|---|
| `ref_pdb_ts1` | 63 | 0 failures | 0 | 22 | NOT_QUALIFIED |
| `ref_pdb_ts2` | 39 | 0 failures | 0 | 1 | NOT_QUALIFIED |
| `ref_pdb_ts3` | 19 | 0 failures | 0 | 11 | NOT_QUALIFIED |
| `ref_pdb_ts_hard` | 28 | 0 failures | 0 | 28 | NOT_QUALIFIED |
| `rfam_fam` | 10790 | 0 failures | 284 | 5 | NOT_QUALIFIED |
| `rfam_temporal` | 436 | 0 failures | 33 | 2 | NOT_QUALIFIED |

## Binding conclusions

- All six panels are filtered canonical/nested projections; their dots are not proven experimental negative labels.
- PDB TS records lack PDB/entity/chain mappings in the frozen JSONL. mmCIF entity/poly-seq-scheme/atom_site and DSSR lineage are therefore `UNCHECKED`; all PDB labels require raw-source rebuilding for experimental claims.
- Rfam directory names and BPfold test-result locations do not establish prediction-derived ground truth, first-release dates, or pair provenance. Rfam labels remain comparative-inference candidates, not experimental labels.
- Exact training or cross-panel overlaps require exclusion/recompute before independent evaluation; TS-hard is not sequence-disjoint from TS1/TS3.
- Rfam temporal records already exactly present in Rfam 14.9 cannot be called new to the later temporal interval. Other records remain `UNCHECKED`, not automatically novel.
- The related post-cutoff PDB collection is also not qualified: collection errors remain and full-polymer evaluation requires an unobserved-residue mask or observed-only rebuild.

See `recompute_sets.json` for exact record sets and `record_audit.jsonl.gz` for per-record evidence.
