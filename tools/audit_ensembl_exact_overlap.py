import gzip
import json
from pathlib import Path

from audit_pretrain_exact_overlap import normalize, query_sets, sha256

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")


def fasta_sequences(path):
    sequence = []
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    yield normalize("".join(sequence))
                sequence = []
            elif line:
                sequence.append(line)
    if sequence:
        yield normalize("".join(sequence))


def main():
    verification_path = ROOT / "ensembl_asset_verification.json"
    if not verification_path.is_file():
        raise RuntimeError("Ensembl files have not passed independent asset verification")
    verification = json.loads(verification_path.read_text())
    if verification["status"] != "PASS_ENSEMBL_ASSETS_INDEPENDENTLY_VERIFIED":
        raise RuntimeError("Ensembl files have not passed independent asset verification")
    queries = query_sets()
    unique_queries = {name: {sequence for _, sequence in rows} for name, rows in queries.items()}
    matched = {name: set() for name in queries}
    raw_count = 0
    filtered_count = 0
    files = [item for values in verification["verified_files"].values() for item in values]
    for index, item in enumerate(files, 1):
        for sequence in fasta_sequences(item["path"]):
            raw_count += 1
            if not 16 <= len(sequence) <= 8192:
                continue
            filtered_count += 1
            for name, unique in unique_queries.items():
                if sequence in unique:
                    matched[name].add(sequence)
        if index % 1000 == 0 or index == len(files):
            print(f"audited {index}/{len(files)}", flush=True)
    expected_raw = sum(item["raw_fasta_records"] for item in verification["sources"].values())
    expected_filtered = sum(item["records_after_original_length_filter"] for item in verification["sources"].values())
    if (raw_count, filtered_count) != (expected_raw, expected_filtered):
        raise AssertionError(f"FASTA counts changed: {(raw_count, filtered_count)} != {(expected_raw, expected_filtered)}")
    query_report = {}
    for name, rows in queries.items():
        hits = matched[name]
        query_report[name] = {"input_rows": len(rows), "unique_sequences": len(unique_queries[name]),
                              "exact_overlap_unique": len(hits),
                              "exact_overlap_rows": sum(sequence in hits for _, sequence in rows),
                              "overlap_names": sorted(label for label, sequence in rows if sequence in hits)}
    report = {"status": "PASS_ENSEMBL_EXACT_OVERLAP_WITH_NT_AND_HOMOLOGY_PENDING",
              "scientific_scope": "exact sequence overlap with verified Ensembl ncRNA sources only; not 80% homology, family independence or complete historical pretraining decontamination proof",
              "verification_manifest": str(verification_path), "verification_sha256": sha256(verification_path),
              "sources": {"ensembl_109_and_bacteria_56": {
                  "n_files": len(files), "bytes": sum(item["bytes"] for item in files),
                  "raw_fasta_records": raw_count, "records_after_original_length_filter": filtered_count,
                  "queries": query_report}},
              "preprocessing": "RiNALMo paper filters lengths to 16..8192 nt; input sequences longer than 1022 nt are randomly cropped during training",
              "exposure_boundary": "a long source match does not prove that all of its bases appeared in every training crop; exact overlap does not establish family novelty or non-novelty"}
    output = ROOT / "ensembl_exact_overlap.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": sha256(output),
                      "counts": {name: item["exact_overlap_unique"] for name, item in query_report.items()}}, indent=2))


if __name__ == "__main__":
    main()
