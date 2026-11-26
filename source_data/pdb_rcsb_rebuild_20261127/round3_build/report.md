# RNAformer PDB RCSB/mmCIF/DSSR rebuild

External Test is audit-only. No prediction, metric, checkpoint, or model-selection outcome is used.

- Source candidates: 153; rebuilt with a unique exact chain and valid DSSR mapping: 58.
- Unresolved/UNCHECKED records: 95; status counts: {'NO_EXACT_CHAIN_MATCH_UNCHECKED': 26, 'AMBIGUOUS_EXACT_CHAIN_MATCHES_UNCHECKED': 61, 'SOURCE_BAD_ALPHABET_UNCHECKED': 4, 'DSSR_MAPPING_ERROR_UNCHECKED': 4}.
- RNAformer-vs-DSSR raw pair exact-match counts: {True: 56, False: 2}. Differences are preserved and require complete metric recomputation.

## Protocols

- `FULL_POLYMER_OBSERVED_MASK_V1`: full polymer sequence and full-polymer 0-based DSSR pairs; only `observed_mask=1` positions may contribute observed-unpaired negatives. Unobserved positions are masked, never negative.
- `OBSERVED_ONLY_V1`: only atom_site-observed residues remain, with observed-only 0-based DSSR pairs and `observed_to_full_indices` for coordinate recovery.
- RCSB search score and subsequence tests are candidate filters only. Identity requires exactly one exact full-polymer or observed-only `(PDB, entity, auth_chain)` match. One-to-many and zero-match records remain `UNCHECKED`.
- `atom_site` is primary for observation; `pdbx_poly_seq_scheme.auth_seq_num` is cross-checked. DSSR pairs are retained without canonical/nested projection.

See `recompute_sets.json` and `unresolved_records.jsonl` for the complete rebuild and gap sets.
