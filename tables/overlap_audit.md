# Global train/test overlap audit (15.14)

Exact-sequence overlap between every training corpus and every frozen evaluation split.

| eval split | n | bprna_tr0 | bprna_tr1 | bprna_tr1c | clean vs all |
|---|---|---|---|---|---|
| bprna_ts0 | 1288 | 0 | 1087 | 0 | NO |
| bprna_new | 5388 | 0 | 0 | 0 | **YES** |
| ref_pdb_ts1 | 63 | 0 | 38 | 0 | NO |
| ref_pdb_ts2 | 39 | 0 | 28 | 0 | NO |
| ref_pdb_ts3 | 19 | 0 | 18 | 0 | NO |
| ref_pdb_ts_hard | 28 | 0 | 21 | 0 | NO |
| testsetb | 428 | 247 | 358 | 0 | NO |
| archiveii_embok_clean | 2214 | 0 | 843 | 0 | NO |
| archiveii | 3451 | 732 | 1852 | 0 | NO |

Known leaks (retractions already issued): TR0∩TestSetB=247 (15.13), TR1∩TS0=1087/PDB-family (15.10), TR1∩ArchiveII-clean=843 (15.13).

The ONLY corpus clean against every split is **bprna_tr1c** (built by make_clean_corpus.py with the 9-split blocklist + archiveii).
