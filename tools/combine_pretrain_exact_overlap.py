import json
from pathlib import Path

from audit_pretrain_exact_overlap import fasta_sequences, query_sets, sha256

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")


def ambiguous_sequences(rows, overlap_names):
    by_name = {}
    for name, sequence in rows:
        by_name.setdefault(name, set()).add(sequence)
    return {sequence for name in set(overlap_names) for sequence in by_name.get(name, set()) if len(by_name[name]) > 1}


def matched_sequences(rows, overlap_names, resolved_conflicts=None):
    by_name = {}
    for name, sequence in rows:
        by_name.setdefault(name, set()).add(sequence)
    names = set(overlap_names)
    if not names.issubset(by_name):
        raise AssertionError("overlap contains a name absent from query rows")
    if any(len(sequences) > 1 for sequences in by_name.values()) and resolved_conflicts is None:
        raise AssertionError("name collision requires sequence-level resolution")
    resolved_conflicts = resolved_conflicts or set()
    matched = set()
    n_rows = 0
    for name, sequence in rows:
        hit = name in names if len(by_name[name]) == 1 else sequence in resolved_conflicts
        if hit:
            matched.add(sequence)
            n_rows += 1
    return matched, n_rows


def source_paths(source, reports):
    if source != "ensembl_109_and_bacteria_56":
        path = Path(reports[source]["sources"][source]["path"])
        if sha256(path) != reports[source]["sources"][source]["sha256"]:
            raise AssertionError(f"source FASTA changed: {source}")
        return [path]
    verification_path = ROOT / "ensembl_asset_verification.json"
    verification = json.loads(verification_path.read_text())
    if verification["status"] != "PASS_ENSEMBL_ASSETS_INDEPENDENTLY_VERIFIED":
        raise RuntimeError("Ensembl verification is incomplete")
    if sha256(verification_path) != reports[source]["verification_sha256"]:
        raise AssertionError("Ensembl verification identity changed")
    paths = []
    for values in verification["verified_files"].values():
        for item in values:
            path = Path(item["path"])
            if sha256(path) != item["sha256"]:
                raise AssertionError(f"Ensembl source changed: {path}")
            paths.append(path)
    return paths


def resolve_conflicts(source, reports, queries):
    candidates = set()
    for name, rows in queries.items():
        report = reports[source]["sources"][source]["queries"][name]
        candidates.update(ambiguous_sequences(rows, report["overlap_names"]))
    if not candidates:
        return set(), {"n_candidate_sequences": 0, "n_exact_matched_sequences": 0}
    resolved = set()
    for path in source_paths(source, reports):
        for sequence in fasta_sequences(path):
            if sequence in candidates and 16 <= len(sequence) <= 8192:
                resolved.add(sequence)
    if not resolved.issubset(candidates):
        raise AssertionError("unexpected conflict resolution")
    return resolved, {"n_candidate_sequences": len(candidates), "n_exact_matched_sequences": len(resolved)}


def main():
    sources = {"rnacentral_22_active": ROOT / "exact_overlap.json",
               "rfam_14_9": ROOT / "exact_overlap.json",
               "ensembl_109_and_bacteria_56": ROOT / "ensembl_exact_overlap.json"}
    reports = {name: json.loads(path.read_text()) for name, path in sources.items()}
    queries = query_sets()
    resolved = {source: resolve_conflicts(source, reports, queries) for source in sources}
    output = {"status": "PASS_EXACT_UNION_WITH_NT_AND_HOMOLOGY_PENDING",
              "scientific_scope": "deduplicated exact matches in versioned RNAcentral, Rfam and Ensembl public sources; not complete historical pretraining exposure or family independence",
              "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()},
              "name_conflict_resolution": {source: value[1] for source, value in resolved.items()},
              "queries": {}}
    for query_name, rows in queries.items():
        matched = {}
        source_counts = {}
        for source, path in sources.items():
            report = reports[source]["sources"][source]["queries"][query_name]
            hits, n_rows = matched_sequences(rows, report["overlap_names"], resolved[source][0])
            if len(hits) != report["exact_overlap_unique"] or n_rows != report["exact_overlap_rows"]:
                raise AssertionError(f"exact count mismatch: {source}/{query_name}")
            matched[source] = hits
            source_counts[source] = {"exact_overlap_unique": len(hits), "exact_overlap_rows": n_rows}
        union = set().union(*matched.values())
        output["queries"][query_name] = {"input_rows": len(rows),
                                         "unique_sequences": len({sequence for _, sequence in rows}),
                                         "exact_overlap_unique_any_known_source": len(union),
                                         "exact_overlap_rows_any_known_source": sum(sequence in union for _, sequence in rows),
                                         "by_source": source_counts,
                                         "not_exact_detected_interpretation": "UNKNOWN_NOT_CLEAN_OR_INDEPENDENT"}
    output["limitations"] = ["sources are deduplicated by normalized full sequence, not summed",
                             "same-name/different-sequence conflicts are resolved by direct source FASTA membership",
                             "nt historical snapshot is not pinned or downloaded",
                             "homology and family analyses are separate pending tasks",
                             "no exact match in these sources is not proof of no pretraining exposure",
                             "long training sequences are randomly cropped above 1022 nt"]
    destination = ROOT / "exact_union_summary.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"],
                      "queries": {name: value["exact_overlap_unique_any_known_source"] for name, value in output["queries"].items()},
                      "sha256": sha256(destination)}, indent=2))


if __name__ == "__main__":
    main()
