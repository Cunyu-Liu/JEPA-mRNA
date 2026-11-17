import gzip
import hashlib
import json
from pathlib import Path

from audit_pretrain_exact_overlap import sha256
from audit_rfam_family_mapping import fasta_records

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
EXPECTED_TARGET_COUNT = 4032030


def main():
    directory = ROOT / "homology_round56"
    manifest_path = directory / "ensembl_target_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "PASS_ENSEMBL_HOMOLOGY_TARGET_PREPARED":
        raise RuntimeError("Ensembl target has not passed its preparation gate")
    if manifest["records_after_original_length_filter"] != EXPECTED_TARGET_COUNT:
        raise AssertionError("unexpected target count")
    if manifest["verification_sha256"] != sha256(ROOT / "ensembl_asset_verification.json"):
        raise AssertionError("source verification identity changed")
    target = Path(manifest["target_fasta"])
    index = Path(manifest["target_index"])
    if sha256(target) != manifest["target_fasta_sha256"] or sha256(index) != manifest["target_index_sha256"]:
        raise AssertionError("target or index hash changed")
    ids = set()
    with target.open("rb") as raw, gzip.open(index, "rt") as metadata:
        records = fasta_records(raw)
        for line in metadata:
            row = json.loads(line)
            key, sequence = next(records)
            if key != row["target_id"] or key in ids:
                raise AssertionError(f"target ID mismatch: {key}")
            if len(sequence) != row["length"] or hashlib.sha256(sequence.encode()).hexdigest() != row["sequence_sha256"]:
                raise AssertionError(f"target sequence mismatch: {key}")
            ids.add(key)
        if next(records, None) is not None:
            raise AssertionError("target has records missing from its index")
    if len(ids) != manifest["records_after_original_length_filter"]:
        raise AssertionError("target/index count mismatch")
    report = {"status": "PASS_ENSEMBL_TARGET_INDEPENDENT_QA", "n_targets": len(ids),
              "target_sha256": manifest["target_fasta_sha256"], "index_sha256": manifest["target_index_sha256"],
              "target_manifest_sha256": sha256(manifest_path),
              "scope": "independent target/index identity and sequence-hash verification; not a homology result"}
    output = directory / "ensembl_target_qa.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
