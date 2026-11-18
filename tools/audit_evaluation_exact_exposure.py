import argparse
import hashlib
import json
from pathlib import Path

from audit_pretrain_exact_overlap import fasta_sequences
from combine_pretrain_exact_overlap import source_paths

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
SOURCES = ["rnacentral_22_active", "rfam_14_9", "ensembl_109_and_bacteria_56"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_sequences(records, source_map, downstream_map):
    sequences = {row["sequence"].upper().replace("T", "U") for row in records}
    matched = {}
    source_paths_report = {}
    for source, paths in source_map.items():
        hits = set()
        source_paths_report[source] = []
        for path in paths:
            path = Path(path)
            source_paths_report[source].append({"path": str(path), "sha256": sha256(path)})
            for sequence in fasta_sequences(path):
                if sequence in sequences and 16 <= len(sequence) <= 8192:
                    hits.add(sequence)
        matched[source] = hits
    downstream = {}
    for name, rows in downstream_map.items():
        reference = {row["seq"].upper().replace("T", "U") for row in rows}
        downstream[name] = len(sequences & reference)
    union = set().union(*matched.values())
    return {"by_source": {source: len(hits) for source, hits in matched.items()},
            "source_paths": source_paths_report,
            "matched_by_source_sha256": {source: sorted(hashlib.sha256(sequence.encode()).hexdigest() for sequence in hits)
                                         for source, hits in matched.items()},
            "exact_union_detected": len(union), "exact_not_detected": len(sequences) - len(union),
            "matched_sequence_sha256": sorted(hashlib.sha256(sequence.encode()).hexdigest() for sequence in union),
            "not_detected_sequence_sha256": sorted(hashlib.sha256(sequence.encode()).hexdigest() for sequence in sequences - union),
            "downstream_exact_overlap": downstream}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--train", default="/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
    parser.add_argument("--dev", default="/root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl")
    args = parser.parse_args()
    evaluation_path = Path(args.evaluation)
    evaluation = json.loads(evaluation_path.read_text())
    assert evaluation["status"] == "PASS_TEMPORAL_EVALUATION_SEQUENCE_IDENTITIES_EXPOSURE_RECHECK_REQUIRED"
    records = evaluation["records"]
    assert len(records) == evaluation["n_unique_evaluation_sequences"]
    assert len({row["sequence"] for row in records}) == len(records)
    source_map = {}
    for source in SOURCES:
        report_path = ROOT / ("ensembl_exact_overlap.json" if source.startswith("ensembl") else "exact_overlap.json")
        report = json.loads(report_path.read_text())
        source_map[source] = source_paths(source, {source: report})
    downstream = {}
    for name, filename in [("TR1c", args.train), ("VL0", args.dev)]:
        downstream[name] = [json.loads(line) for line in Path(filename).open() if line.strip()]
    result = audit_sequences(records, source_map, downstream)
    result.update({"status": "PASS_EVALUATION_EXACT_EXPOSURE_HOMOLOGY_PENDING",
                   "evaluation_sha256": sha256(evaluation_path),
                   "n_unique_evaluation_sequences": len(records),
                   "n_chains": evaluation["n_chains"],
                   "n_sequences_with_metadata_mismatch": evaluation["n_sequences_with_metadata_mismatch"],
                   "limitations": ["Only normalized full-sequence matches in known public sources", "not detected is unknown, not clean",
                                   "Historical nt snapshot is not pinned", "Final-label 80% homology is pending",
                                   "No model inference, training or outcome-based filtering"]})
    output = Path(args.output)
    fasta = Path(args.fasta)
    for path in (output, fasta):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite: {path}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    matched_hashes = set(result["matched_sequence_sha256"])
    with fasta.open("w") as handle:
        for row in records:
            detected = int(hashlib.sha256(row["sequence"].encode()).hexdigest() in matched_hashes)
            handle.write(f">{row['eval_id']}|exact_detected={detected}\n{row['sequence']}\n")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("sequence_sha256") and key != "source_paths"}, ensure_ascii=False, indent=2))
    print("artifact_sha256", sha256(output), "fasta_sha256", sha256(fasta))


if __name__ == "__main__":
    main()
