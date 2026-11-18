import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_pretrain_exact_overlap import fasta_sequences
from combine_pretrain_exact_overlap import source_paths

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--union", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    union_path = Path(args.union)
    union = json.loads(union_path.read_text())
    assert union["status"] == "BLOCKED_EXACT_NOT_HOMOLOGY_DETECTED"
    missed = set(union["exact_detected_not_homology_detected"])
    queries = {}
    for line in Path(args.fasta).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:]
            queries[name] = ""
        else:
            queries[name] += line.strip()
    assert missed <= set(queries)
    target_sequences = {queries[name] for name in missed}
    remaining = set(target_sequences)
    scanned = []
    for source in ["rnacentral_22_active", "rfam_14_9", "ensembl_109_and_bacteria_56"]:
        report_path = ROOT / ("ensembl_exact_overlap.json" if source.startswith("ensembl") else "exact_overlap.json")
        report = json.loads(report_path.read_text())
        for path in source_paths(source, {source: report}):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            scanned.append({"source": source, "path": str(path), "sha256": digest})
            for sequence in fasta_sequences(path):
                remaining.discard(sequence)
        if not remaining:
            break
    assert not remaining, f"exact positives not recovered directly: {sorted(remaining)}"
    details = []
    for name in sorted(missed):
        sequence = queries[name]
        most_common = Counter(sequence).most_common(1)[0]
        details.append({"query": name, "length": len(sequence), "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                        "non_ACGU": sorted(set(sequence) - set("ACGU")),
                        "max_base_fraction": most_common[1] / len(sequence),
                        "most_common_base": most_common[0],
                        "interpretation": "EXACT_POSITIVE_MMSEQS_NOT_DETECTED_SHORT_OR_LOW_COMPLEXITY_CONSISTENT_NOT_PROOF_OF_ABSENCE"})
    union["status"] = "PASS_TEMPORAL_HOMOLOGY_WITH_EXACT_POSITIVE_SEARCH_UNDETECTED"
    union["exact_search_undetected_resolution"] = {
        "status": "PASS_DIRECT_FASTA_EXACT_CONFIRMATION",
        "n_queries": len(details), "details": details, "scanned_sources": scanned,
        "scope": "Direct exact membership overrides a search-only no-hit; it does not turn other no-hit queries into clean sequences",
        "reason_boundary": "Short/low-complexity explanation is consistent with these examples, not a proof of why MMseqs2 omitted each hit",
    }
    union["detected_by_exact_or_homology"] = union["detected_by_any_source"] + len(details)
    union["unknown_not_detected_by_exact_or_homology"] = union["n_queries"] - union["detected_by_exact_or_homology"]
    union["homology_union_sha256_before_resolution"] = hashlib.sha256(union_path.read_bytes()).hexdigest()
    union["limitations"].append("Some exact-positive short/low-complexity queries are not recovered by MMseqs2; exact and homology evidence are reported separately")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite: {output}")
    output.write_text(json.dumps(union, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in union.items() if key not in {"query_records", "sources", "exact_search_undetected_resolution"}}, ensure_ascii=False, indent=2))
    print("artifact_sha256", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
