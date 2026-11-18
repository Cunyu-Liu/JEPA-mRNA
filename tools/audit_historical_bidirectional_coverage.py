import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SOURCES = ["rnacentral_22_active", "rfam_14_9", "ensembl_109_and_bacteria_56"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--previous-union", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    queries = set()
    for line in (root / "queries.fasta").read_text().splitlines():
        if line.startswith(">"):
            queries.add(line[1:].split()[0])
    assert len(queries) == 165854
    previous = json.loads(Path(args.previous_union).read_text())
    assert previous["n_queries"] == len(queries)
    raw_union, strict_union = set(), set()
    source_reports = {}
    for source in SOURCES:
        directory = root / source
        manifest_path = directory / "search_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "PASS_SEARCH_COMPLETED_AGGREGATION_PENDING"
        matches = directory / "matches.tsv"
        assert sha256(matches) == manifest["result_sha256"]
        raw, strict = set(), set()
        counts = Counter()
        excluded = 0
        examples = []
        with matches.open() as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.rstrip("\n").split("\t")
                assert len(fields) == 12, (source, line_number)
                query, target, identity, alignment_length, query_length, target_length = fields[:6]
                assert query in queries and float(identity) >= 80.0
                alignment_length, query_length, target_length = int(alignment_length), int(query_length), int(target_length)
                assert alignment_length >= 0.8 * query_length
                raw.add(query)
                counts[query] += 1
                if alignment_length < 0.8 * target_length:
                    excluded += 1
                    if len(examples) < 30:
                        examples.append({"query": query, "target": target, "alnlen": alignment_length,
                                         "qlen": query_length, "tlen": target_length})
                else:
                    strict.add(query)
        source_reports[source] = {
            "raw_detected": len(raw), "strict_detected": len(strict),
            "queries_with_only_sub_target_coverage_hits": len(raw - strict),
            "reported_matches": sum(counts.values()), "excluded_target_coverage": excluded,
            "max_reported_targets_per_query": max(counts.values(), default=0),
            "queries_with_over_300_reported_targets": sum(value > 300 for value in counts.values()),
            "matches_sha256": sha256(matches), "search_manifest_sha256": sha256(manifest_path),
            "excluded_examples": examples,
        }
        raw_union |= raw
        strict_union |= strict
        print(source, source_reports[source], flush=True)
    assert len(raw_union) == previous["detected_by_any_source"], (len(raw_union), previous["detected_by_any_source"])
    report = {
        "status": "PASS_HISTORICAL_BIDIRECTIONAL_POSTFILTER_REVIEW",
        "n_queries": len(queries), "sources": source_reports,
        "raw_detected_by_any_source": len(raw_union), "strict_detected_by_any_source": len(strict_union),
        "queries_lost_from_raw_union": len(raw_union - strict_union),
        "lost_union_query_examples": sorted(raw_union - strict_union)[:100],
        "previous_union_sha256": sha256(args.previous_union),
        "filter": "alnlen >= 0.8*qlen and alnlen >= 0.8*tlen; original TSV and manifests unchanged",
        "limitations": ["No-hit and newly undetected remain unknown, not clean", "This is a coverage correction, not a new model result",
                        "--max-seqs is not treated as a cap on final reported rows", "Historical nt snapshot remains unpinned",
                        "Previous positive matches remain exposure evidence; raw union counts are not silently rewritten"],
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "sources"}, ensure_ascii=False, indent=2))
    print("artifact_sha256", sha256(output))


if __name__ == "__main__":
    main()
