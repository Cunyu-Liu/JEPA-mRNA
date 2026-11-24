# Exposure, temporal and label-lineage audit

Scope: CPU-only TR1c and CHANRG Train/Validation. No CHANRG Test/GenA/GenC/GenF or external benchmark labels are inputs. This is a known-source audit, not complete historical pretraining reconstruction or a clean-data certificate.

## Exact and homology coverage

| Dataset | View | Rows | EXACT | HOMOLOGY | NOT_DETECTED_UNKNOWN | UNCHECKED |
|---|---|---:|---:|---:|---:|---:|
| chanrg_train | matched_clean | 78859 | 27577 | 0 | 51282 | 0 |
| chanrg_train | own_clean | — | — | — | — | UNCHECKED |
| chanrg_train | raw | 95181 | 33475 | 0 | 61706 | 0 |
| chanrg_validation | matched_clean | 5872 | 2100 | 0 | 3772 | 0 |
| chanrg_validation | own_clean | — | — | — | — | UNCHECKED |
| chanrg_validation | raw | 10540 | 3931 | 0 | 6609 | 0 |
| tr1c | matched_clean | — | — | — | — | UNCHECKED |
| tr1c | own_clean | 42564 | 37870 | 2464 | 2230 | 0 |
| tr1c | raw | 42564 | 37870 | 2464 | 2230 | 0 |

EXACT is normalized full-sequence membership in versioned RNAcentral 22, Rfam 14.9, and/or Ensembl 109/Bacteria 56 assets. HOMOLOGY is the existing TR1c 80% identity/coverage union. CHANRG homology is UNCHECKED. Non-detection is not independence or cleanliness.

## Temporal, label and structure-metadata lineage

| Dataset | Assigned dataset-level class | Rows with source date | Rows with PDB/mmCIF metadata | Modified-parent | Residue observation |
|---|---|---:|---:|---|---|
| chanrg_train | COMPARATIVE_INFERRED | 0 | 0 | UNCHECKED | UNCHECKED |
| chanrg_validation | COMPARATIVE_INFERRED | 0 | 0 | UNCHECKED | UNCHECKED |
| tr1c | DATABASE | 0 | 0 | UNCHECKED | UNCHECKED |

TR1c remains DATABASE because its mixed per-record experimental/comparative lineage is unresolved. CHANRG remains COMPARATIVE_INFERRED from its pinned Rfam-consensus provenance. No row is upgraded to EXPERIMENTAL, THERMODYNAMIC_TEACHER or SIMULATED. Absence of these assigned classes is not evidence about unknown underlying origins.

## Unresolved gaps

- Historical nt is unpinned and unavailable; complete RiNALMo exposure remains UNCHECKED even where another declared source gives a positive exact match.
- Source release timestamps are not per-record label dates. All development-record temporal states remain UNCHECKED.
- Aggregated JSONL lacks PDB/mmCIF method, resolution, chain, modified-parent and atom-observation mappings. Actual modified residues or unobserved residues are neither counted nor inferred away.
- TR1c own-clean repeats the supplied manifest population. CHANRG own-clean and TR1c matched-clean have no established population; their counts stay null/UNCHECKED.
- CHANRG matched-clean is descriptive both-side cluster-exclusion accounting, not a newly authorized training asset. NOT_DETECTED_UNKNOWN is never clean.

Machine-readable counts, source hashes, per-record states and independent verification accompany this report.
