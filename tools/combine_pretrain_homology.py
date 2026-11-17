import gzip
import io
import json
from pathlib import Path

from audit_pretrain_exact_overlap import sha256

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
OUTPUT = ROOT / "homology_round56"
SOURCES = ["rnacentral_22_active", "rfam_14_9", "ensembl_109_and_bacteria_56"]


def main():
    query_manifest_path = OUTPUT / "query_manifest.json"
    query_manifest = json.loads(query_manifest_path.read_text())
    merged = {}
    memberships = {}
    source_reports = {}
    for source in SOURCES:
        summary_path = OUTPUT / source / "homology_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"source homology summary is incomplete: {source}")
        summary = json.loads(summary_path.read_text())
        if summary["status"] != "PASS_HOMOLOGY_SEARCH_AGGREGATED_NOT_FAMILY_INDEPENDENCE":
            raise RuntimeError(f"source has not passed aggregation: {source}")
        per_query_path = Path(summary["per_query_path"])
        if sha256(per_query_path) != summary["per_query_sha256"]:
            raise AssertionError(f"per-query results changed: {source}")
        ids = set()
        with gzip.open(per_query_path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                query = row["query_id"]
                if query in ids or (row["detected"] != (row["best_alignment"] is not None)):
                    raise AssertionError(f"invalid per-query record: {source}/{query}")
                ids.add(query)
                if memberships.setdefault(query, row["source_sets"]) != row["source_sets"]:
                    raise AssertionError(f"query-set membership mismatch: {query}")
                merged.setdefault(query, {})[source] = row["best_alignment"]
        if len(ids) != query_manifest["n_unique_queries"] or set(merged) != ids:
            raise AssertionError(f"query coverage mismatch: {source}")
        source_reports[source] = {"summary_path": str(summary_path), "summary_sha256": sha256(summary_path),
                                  "detected_homology_unique": summary["detected_homology_unique"],
                                  "detected_by_query_set": summary["queries"]}
    union = {query for query, values in merged.items() if any(value is not None for value in values.values())}
    per_set = {}
    for name, query_counts in query_manifest["queries"].items():
        ids = {query for query, source_sets in memberships.items() if name in source_sets}
        if len(ids) != query_counts["homology_queries"]:
            raise AssertionError(f"query-set identity mismatch: {name}")
        per_set[name] = {"n_queries": len(ids), "detected_by_any_source": len(ids & union),
                         "not_detected_by_any_source": len(ids - union),
                         "detected_by_source": {source: sum(values[source] is not None for query, values in merged.items() if query in ids)
                                                for source in SOURCES}}
    union_path = OUTPUT / "query_homology_union.jsonl.gz"
    with union_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for query, values in merged.items():
                    handle.write(json.dumps({"query_id": query,
                                             "detected_sources": [source for source in SOURCES if values[source] is not None],
                                             "not_detected_sources": [source for source in SOURCES if values[source] is None],
                                             "best_by_source": values,
                                             "interpretation": "DETECTED_SOURCE_EXPOSURE" if query in union else "NOT_DETECTED_LIMITED_SOURCES_NT_HISTORY_UNRESOLVED"}, sort_keys=True) + "\n")
    report = {"status": "PASS_KNOWN_SOURCE_HOMOLOGY_UNION_NT_AND_FAMILY_LIMITS_REMAIN",
              "scientific_scope": "union of detected 80% identity/bidirectional-coverage hits in three verified public sources; no independent-family or complete historical pretraining conclusion",
              "n_queries": len(merged), "detected_by_any_source": len(union),
              "not_detected_by_any_source": len(merged) - len(union), "sources": source_reports,
              "queries": per_set, "query_manifest_sha256": sha256(query_manifest_path),
              "union_per_query": str(union_path), "union_per_query_sha256": sha256(union_path),
              "limitations": ["no-hit is not proof of no homology", "heuristic search and at most 300 reported targets per query",
                              "nt historical snapshot is not pinned or downloaded", "source membership does not establish family identity",
                              "RiNALMo randomly crops training sequences above 1022 nt"]}
    output = OUTPUT / "homology_union_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "detected_by_any_source": len(union),
                      "summary_sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
