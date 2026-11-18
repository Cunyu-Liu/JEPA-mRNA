import argparse
import hashlib
import json
from pathlib import Path

from audit_pretrain_exact_overlap import fasta_sequences, normalize, sha256
from combine_pretrain_exact_overlap import source_paths

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fasta", required=True)
    args = parser.parse_args()
    metadata_path = Path(args.metadata)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "PASS_TEMPORAL_METADATA_FREEZE_LABEL_AND_INDEPENDENCE_QA_PENDING"
    assert sha256(metadata_path) == "b79ecc8f3501966417738d85f94559da917ca20a26bd3a16e9f07b130c1d289c"
    rows = metadata["records"]
    all_queries = {normalize(row["sequence"]) for row in rows}
    eligible = {normalize(row["sequence"]) for row in rows if 16 <= row["length"] <= 8192}
    assert len(all_queries) == 1391 and len(eligible) == 1131
    sources = {"rnacentral_22_active": ROOT / "exact_overlap.json",
               "rfam_14_9": ROOT / "exact_overlap.json",
               "ensembl_109_and_bacteria_56": ROOT / "ensembl_exact_overlap.json"}
    matched = {source: set() for source in sources}
    source_counts = {}
    for source, report_path in sources.items():
        report = json.loads(report_path.read_text())
        source_counts[source] = []
        for path in source_paths(source, {source: report}):
            source_counts[source].append({"path": str(path), "sha256": sha256(path)})
            for sequence in fasta_sequences(path):
                if sequence in eligible and 16 <= len(sequence) <= 8192:
                    matched[source].add(sequence)
        print(source, len(matched[source]), flush=True)
    union = set().union(*matched.values())
    not_detected = eligible - union
    report = {
        "status": "PASS_TEMPORAL_KNOWN_SOURCE_EXACT_EXPOSURE_HOMOLOGY_PENDING",
        "metadata_sha256": sha256(metadata_path),
        "n_all_unique_sequences": len(all_queries),
        "n_16_to_8192_unique_sequences": len(eligible),
        "n_duplicate_entities_removed": len(rows) - len(all_queries),
        "source_paths": source_counts,
        "by_source": {source: len(values) for source, values in matched.items()},
        "exact_union_detected": len(union),
        "exact_not_detected": len(not_detected),
        "matched_sequence_sha256": sorted(hashlib.sha256(sequence.encode()).hexdigest() for sequence in union),
        "not_detected_sequence_sha256": sorted(hashlib.sha256(sequence.encode()).hexdigest() for sequence in not_detected),
        "limitations": ["Only normalized full-sequence matches in known public sources", "not detected is unknown, not clean or independent",
                        "historical nt snapshot is not pinned", "80% homology and structure-label gates remain pending",
                        "No model outcome was used to choose candidates, dates, filters or thresholds", "No model inference or training has been started"],
    }
    output = Path(args.output)
    fasta = Path(args.fasta)
    for path in (output, fasta):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite: {path}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with fasta.open("w") as handle:
        for index, sequence in enumerate(sorted(eligible)):
            handle.write(f">temporal_unique_{index:04d}|exact_detected={int(sequence in union)}\n{sequence}\n")
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("sequence_sha256")}, ensure_ascii=False, indent=2))
    print("artifact_sha256", sha256(output), "fasta_sha256", sha256(fasta))


if __name__ == "__main__":
    main()
