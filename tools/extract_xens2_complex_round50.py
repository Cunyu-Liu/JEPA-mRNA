from pathlib import Path

import extract_rinalmo_embeddings


ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = ROOT / "data/rna_complex_round49/xens2_complex"
extract_rinalmo_embeddings.JSONL_DIR = str(DATA)
extract_rinalmo_embeddings.OUT_DIR = str(DATA / "embeddings")


if __name__ == "__main__":
    raise SystemExit(extract_rinalmo_embeddings.main())
