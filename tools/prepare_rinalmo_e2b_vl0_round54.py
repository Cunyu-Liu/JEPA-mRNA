import hashlib
import json
from pathlib import Path

from rnajepa.thermodynamics import thermodynamic_features

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
OUT = ROOT / "data/e2b_thermo_round54"
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    entries = []
    for index, line in enumerate(DATA.open()):
        sequence = json.loads(line)["seq"].upper().replace("T", "U")
        thermodynamic_features(sequence, OUT / "cache")
        key = hashlib.sha256(sequence.encode()).hexdigest()
        entries.append({"index": index, "sequence_sha256": key, "cache": f"{key}.npz",
                        "cache_sha256": sha256(OUT / "cache" / f"{key}.npz")})
        if (index + 1) % 50 == 0:
            print(f"[E2b VL0] {index + 1}/{196}", flush=True)
    result = {"status": "PASS_E2B_VL0_CACHE", "split": "VL0", "n_sequences": len(entries),
              "data_sha256": sha256(DATA), "entries": entries,
              "normalization_source": "TR1c_only", "test_set_used": False}
    (OUT / "validation_cache_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "n_sequences": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
