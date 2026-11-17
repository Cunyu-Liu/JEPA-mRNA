import gzip
import hashlib
import io
import json
from pathlib import Path

from audit_ensembl_exact_overlap import fasta_sequences
from audit_pretrain_exact_overlap import sha256

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
OUTPUT = ROOT / "homology_round56"


def main():
    verification_path = ROOT / "ensembl_asset_verification.json"
    if not verification_path.is_file():
        raise RuntimeError("Ensembl target requires independently verified assets")
    verification = json.loads(verification_path.read_text())
    if verification["status"] != "PASS_ENSEMBL_ASSETS_INDEPENDENTLY_VERIFIED":
        raise RuntimeError("Ensembl target requires independently verified assets")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target_path = OUTPUT / "ensembl_targets.fasta.gz"
    index_path = OUTPUT / "ensembl_target_index.jsonl.gz"
    raw_count = 0
    kept_count = 0
    with target_path.open("wb") as raw_target, index_path.open("wb") as raw_index:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_target, mtime=0) as target_compressed:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_index, mtime=0) as index_compressed:
                with io.TextIOWrapper(target_compressed, encoding="ascii") as target, io.TextIOWrapper(index_compressed, encoding="utf-8") as index:
                    for name, values in sorted(verification["verified_files"].items()):
                        for file_number, item in enumerate(sorted(values, key=lambda row: row["path"])):
                            for record_number, sequence in enumerate(fasta_sequences(item["path"])):
                                raw_count += 1
                                if not 16 <= len(sequence) <= 8192:
                                    continue
                                target_id = f"E{kept_count:09d}"
                                target.write(f">{target_id}\n")
                                for offset in range(0, len(sequence), 80):
                                    target.write(sequence[offset:offset + 80] + "\n")
                                index.write(json.dumps({"target_id": target_id, "source": name, "path": item["path"],
                                                        "source_sha256": item["sha256"], "file_number": file_number,
                                                        "record_number": record_number, "length": len(sequence),
                                                        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()}, sort_keys=True) + "\n")
                                kept_count += 1
                    if raw_count != sum(item["raw_fasta_records"] for item in verification["sources"].values()):
                        raise AssertionError("raw FASTA count mismatch")
                    if kept_count != sum(item["records_after_original_length_filter"] for item in verification["sources"].values()):
                        raise AssertionError("filtered FASTA count mismatch")
    manifest = {"status": "PASS_ENSEMBL_HOMOLOGY_TARGET_PREPARED",
                "verification_manifest": str(verification_path), "verification_sha256": sha256(verification_path),
                "target_fasta": str(target_path), "target_fasta_sha256": sha256(target_path),
                "target_index": str(index_path), "target_index_sha256": sha256(index_path),
                "raw_fasta_records": raw_count, "records_after_original_length_filter": kept_count,
                "scope": "deterministically ordered and source-traceable target; no search performed"}
    output = OUTPUT / "ensembl_target_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "n_targets": kept_count,
                      "manifest_sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
