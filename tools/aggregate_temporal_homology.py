import argparse
import hashlib
import json
from pathlib import Path

SOURCES = ["rnacentral_22_active", "rfam_14_9", "ensembl_109_and_bacteria_56"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_queries(path):
    queries = {}
    name = None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            header = line[1:].split()[0]
            name = header.split("|", 1)[0]
            assert header.rsplit("exact_detected=", 1)[-1] in {"0", "1"}
            assert name not in queries
            queries[name] = ""
        elif line.strip():
            assert name is not None
            queries[name] += line.strip().upper().replace("T", "U")
    assert len(queries) == 1131
    assert all(16 <= len(sequence) <= 8192 for sequence in queries.values())
    return queries


def aggregate(fasta, root):
    queries = read_queries(fasta)
    records = {name: {"query": name, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                      "exact_detected": name.rsplit("exact_detected=", 1)[-1] == "1",
                      "detected_by_source": []} for name, sequence in queries.items()}
    source_reports = {}
    for source in SOURCES:
        directory = Path(root) / source
        manifest_path = directory / "search_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "PASS_SEARCH_COMPLETED_AGGREGATION_PENDING"
        assert manifest["source"] == source and manifest["mmseqs_version"] == "17.b804f"
        command = manifest["command"]
        assert command[command.index("--min-seq-id") + 1] == "0.8"
        assert command[command.index("--cov-mode") + 1] == "0"
        assert command[command.index("-c") + 1] == "0.8"
        assert command[command.index("--max-seqs") + 1] == "300"
        matches = directory / "matches.tsv"
        assert sha256(matches) == manifest["result_sha256"]
        hits = set()
        counts = {name: 0 for name in queries}
        with matches.open() as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.rstrip("\n").split("\t")
                assert len(fields) == 12, (source, line_number)
                query, target, identity, alignment_length, query_length, target_length = fields[:6]
                assert query in queries, (source, query)
                assert len(queries[query]) == int(query_length)
                assert float(identity) >= 80.0 and int(target_length) > 0
                assert int(alignment_length) >= 0.8 * int(query_length)
                assert int(alignment_length) >= 0.8 * int(target_length)
                hits.add(query)
                counts[query] += 1
        assert all(count <= 300 for count in counts.values())
        source_reports[source] = {"detected": len(hits), "not_detected": len(queries) - len(hits),
                                  "matches_sha256": sha256(matches), "search_manifest_sha256": sha256(manifest_path),
                                  "target_sha256": manifest["target_sha256"]}
        for name in hits:
            records[name]["detected_by_source"].append(source)
    union = {name for name, row in records.items() if row["detected_by_source"]}
    for row in records.values():
        row["interpretation"] = "DETECTED_KNOWN_SOURCE" if row["detected_by_source"] else "UNKNOWN_NOT_CLEAN"
    return {"status": "PASS_TEMPORAL_HOMOLOGY_UNION_LABEL_GATE_PENDING",
            "n_queries": len(queries), "query_fasta_sha256": sha256(fasta), "sources": source_reports,
            "detected_by_any_source": len(union), "not_detected_by_any_source": len(queries) - len(union),
            "query_records": [records[name] for name in sorted(records)],
            "protocol": "MMseqs2 17.b804f; >=80% identity and >=80% bidirectional coverage; up to300 targets/query",
            "limitations": ["Not detected is unknown, not clean or independent", "Historical nt snapshot remains unpinned",
                            "Canonical polymer and structure-label gates are separate", "No model inference or training has occurred"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert sha256(args.metadata) == "b79ecc8f3501966417738d85f94559da917ca20a26bd3a16e9f07b130c1d289c"
    report = aggregate(args.fasta, args.root)
    report["metadata_sha256"] = sha256(args.metadata)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "query_records"}, ensure_ascii=False, indent=2))
    print("artifact_sha256", sha256(output))


if __name__ == "__main__":
    main()
