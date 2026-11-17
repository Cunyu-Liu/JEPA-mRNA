import argparse
import gzip
import io
import json
from pathlib import Path

from audit_pretrain_exact_overlap import sha256

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
OUTPUT = ROOT / "homology_round56"


def read_query_index(path):
    rows = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row["query_id"] in rows:
                raise AssertionError(f"duplicate query ID: {row['query_id']}")
            rows[row["query_id"]] = row
    return rows


def summarize_matches(lines, query_rows):
    hits = {}
    n_alignments = 0
    for line_number, line in enumerate(lines, 1):
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 12:
            raise AssertionError(f"unexpected MMseqs2 fields at line {line_number}")
        query, target = fields[:2]
        if query not in query_rows:
            raise AssertionError(f"unknown query ID: {query}")
        pident, alnlen, qlen, tlen = map(float, fields[2:6])
        if pident < 80 or alnlen <= 0 or qlen != query_rows[query]["length"] or tlen <= 0:
            raise AssertionError(f"invalid or below-threshold alignment at line {line_number}")
        evalue, bits = map(float, fields[10:12])
        record = {"target": target, "pident_percent": pident, "alignment_length": alnlen,
                  "query_length": qlen, "target_length": tlen, "query_start": int(fields[6]),
                  "query_end": int(fields[7]), "target_start": int(fields[8]), "target_end": int(fields[9]),
                  "evalue": evalue, "bits": bits}
        hits.setdefault(query, []).append(record)
        n_alignments += 1
    best = {query: sorted(rows, key=lambda row: (-row["pident_percent"], -row["alignment_length"], -row["bits"], row["target"]))[0]
            for query, rows in hits.items()}
    return best, n_alignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["rnacentral_22_active", "rfam_14_9", "ensembl_109_and_bacteria_56"])
    args = parser.parse_args()
    directory = OUTPUT / args.source
    search_path = directory / "search_manifest.json"
    search = json.loads(search_path.read_text())
    if search["status"] != "PASS_SEARCH_COMPLETED_AGGREGATION_PENDING":
        raise RuntimeError("search has not completed")
    matches_path = directory / "matches.tsv"
    if sha256(matches_path) != search["result_sha256"]:
        raise AssertionError("search results changed")
    query_index_path = OUTPUT / "query_index.jsonl.gz"
    query_manifest_path = OUTPUT / "query_manifest.json"
    query_manifest = json.loads(query_manifest_path.read_text())
    if sha256(query_index_path) != query_manifest["query_index_sha256"]:
        raise AssertionError("query index changed")
    query_rows = read_query_index(query_index_path)
    with matches_path.open() as handle:
        best, n_alignments = summarize_matches(handle, query_rows)
    sets = sorted({name for row in query_rows.values() for name in row["source_sets"]})
    per_set = {}
    for name in sets:
        ids = {query for query, row in query_rows.items() if name in row["source_sets"]}
        detected = ids.intersection(best)
        per_set[name] = {"n_queries": len(ids), "detected_homology_unique": len(detected),
                         "not_detected_by_this_search": len(ids - detected),
                         "detected_fraction": len(detected) / len(ids)}
    per_query_path = directory / "query_homology.jsonl.gz"
    with per_query_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as output:
                for query, row in query_rows.items():
                    output.write(json.dumps({"query_id": query, "source_sets": row["source_sets"],
                                             "detected": query in best, "best_alignment": best.get(query),
                                             "negative_interpretation": None if query in best else "NOT_DETECTED_NOT_PROOF_OF_NO_HOMOLOGY"}, sort_keys=True) + "\n")
    report = {"status": "PASS_HOMOLOGY_SEARCH_AGGREGATED_NOT_FAMILY_INDEPENDENCE",
              "source": args.source, "n_queries": len(query_rows), "n_alignments": n_alignments,
              "detected_homology_unique": len(best), "not_detected_by_this_search": len(query_rows) - len(best),
              "detected_fraction": len(best) / len(query_rows), "queries": per_set,
              "search_manifest": str(search_path), "search_manifest_sha256": sha256(search_path),
              "matches_sha256": search["result_sha256"], "per_query_path": str(per_query_path),
              "per_query_sha256": sha256(per_query_path), "threshold": "MMseqs2 min-seq-id=0.8, cov-mode=0, c=0.8",
              "scope": "heuristic detected homology at the frozen threshold; no-hit is not proof of absence; no independent-family or complete pretraining exposure conclusion",
              "target_limit": "at most 300 reported targets per query; not a complete count of homologs",
              "input_limit": "RiNALMo filters 16..8192 nt and randomly crops training inputs above 1022 nt"}
    output = directory / "homology_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "detected_homology_unique": len(best),
                      "summary_sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
