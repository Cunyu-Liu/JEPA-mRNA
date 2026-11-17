import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/manifests/v2/data_audit")
EXPECTED = {"train.csv": (2000, 3804), "val.csv": (270, 270), "test.csv": (304, 304)}

tr1c_sequences = set()
with open("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl") as handle:
    for line in handle:
        if line.strip():
            tr1c_sequences.add(json.loads(line)["seq"].upper().replace("T", "U"))

report = json.loads((ROOT / "data_exposure_audit.json").read_text())
assert report["tr1c"]["n_sequences"] == 42564
assert report["tr1c"]["n_80pct_clusters"] == 15521
for filename, (unique_removed, rows_removed) in EXPECTED.items():
    with (ROOT / "ncrna_clean_v2" / filename).open(newline="") as handle:
        kept = list(csv.DictReader(handle))
    assert not any(row["sequence"].upper().replace("T", "U") in tr1c_sequences for row in kept)
    stored = report["ncrna_clean"]["splits"][filename]
    assert stored["removed_exact_tr1c_unique"] == unique_removed
    assert stored["removed_exact_tr1c_rows"] == rows_removed
    assert stored["output_n"] == len(kept)
    assert hashlib.sha256((ROOT / "ncrna_clean_v2" / filename).read_bytes()).hexdigest() == stored["sha256"]
assert report["rinalmo_pretrain_exposure"]["status"].startswith("BLOCKED")
print(json.dumps({
    "status": "PASS_INDEPENDENT_DATA_AUDIT",
    "audit_sha256": hashlib.sha256((ROOT / "data_exposure_audit.json").read_bytes()).hexdigest(),
    "clean_overlap": 0,
    "checks": ["identity", "counts", "hashes", "clean overlap", "explicit pretrain limit"],
}, indent=2))
