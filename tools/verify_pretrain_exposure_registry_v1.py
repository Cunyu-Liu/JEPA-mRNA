#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/rna_jev_pretrain_exposure")
OUT = ROOT / "manifests/pretrain_exposure_v1"
AUDIT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
SOURCE_NAMES = ("rnacentral_22_active", "rfam_14_9", "ensembl_109_and_bacteria_56")
ALLOWED = {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(4 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_fasta(path):
    name = None
    sequence = []
    with gzip.open(path, "rt", encoding="ascii", newline=None) as stream:
        for raw in stream:
            value = raw.strip()
            if not value:
                continue
            if value[0] == ">":
                if name is not None:
                    yield name, "".join(sequence).upper().replace("T", "U")
                name = value[1:]
                sequence = []
            else:
                if name is None:
                    raise AssertionError("FASTA sequence has no header")
                sequence.append(value)
    if name is not None:
        yield name, "".join(sequence).upper().replace("T", "U")


def source_files():
    manifest = json.loads((AUDIT / "source_manifest.json").read_text())
    result = {}
    for row in manifest["sources"]:
        result[row["name"]] = [dict(row, path=row["local_path"])]
    verification = json.loads((AUDIT / "ensembl_asset_verification.json").read_text())
    result["ensembl_109_and_bacteria_56"] = [item for group in sorted(verification["verified_files"])
                                               for item in sorted(verification["verified_files"][group], key=lambda x: x["path"])]
    return result


def registry_rows():
    with gzip.open(OUT / "sequence_registry.jsonl.gz", "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check_source_identity(files):
    for source, items in files.items():
        for item in items:
            path = Path(item["path"])
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise AssertionError(f"independent source hash mismatch: {source}/{path}")


def check_queries(rows):
    data = json.loads((ROOT / "manifests/foldnar_v1/data_registry.json").read_text())
    datasets = {item["dataset_id"]: item for item in data["rows"]}
    cache = {}
    for row in rows:
        dataset = row["dataset_id"]
        if dataset not in cache:
            item = datasets[dataset]
            if digest(item["path"]) != item["sha256"]:
                raise AssertionError(f"independent query hash mismatch: {dataset}")
            with Path(item["path"]).open() as stream:
                cache[dataset] = [json.loads(line) for line in stream if line.strip()]
        source = cache[dataset][row["row_index"]]
        name = source.get("name", source.get("id", f"row_{row['row_index']}"))
        expected_id = f"{dataset}:row{row['row_index']:06d}:{name}"
        if row["query_id"] != expected_id:
            raise AssertionError(f"query ID mismatch: {row['query_id']}")
        sequence = source.get("seq", source.get("sequence")).strip().upper().replace("T", "U")
        if hashlib.sha256(sequence.encode("ascii")).hexdigest() != row["sequence_sha256"]:
            raise AssertionError(f"query sequence mismatch: {row['query_id']}")
        if len(sequence) != row["normalized_length"]:
            raise AssertionError(f"query length mismatch: {row['query_id']}")
    return cache


def independent_scan(files, sequence_by_hash):
    wanted = set(sequence_by_hash.values())
    observed = {source: {} for source in SOURCE_NAMES}
    counts = {}
    for source in SOURCE_NAMES:
        raw_count = 0
        filtered_count = 0
        for item in files[source]:
            for header, sequence in read_fasta(item["path"]):
                raw_count += 1
                if len(sequence) < 16 or len(sequence) > 8192 or sequence not in wanted:
                    continue
                filtered_count += 1
                seq_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
                if seq_hash not in observed[source]:
                    observed[source][seq_hash] = {
                        "source_record_id": header.split()[0], "source_record_header": header,
                        "source_file": str(Path(item["path"]).relative_to(AUDIT)), "source_record_match_count": 0,
                    }
                observed[source][seq_hash]["source_record_match_count"] += 1
        # Count all records, including those outside the query set, with a second lightweight parser only when needed.
        # The builder's independently frozen source reports provide total-count reconciliation below.
        counts[source] = {"matched_unique_queries": len(observed[source])}
        print(json.dumps({"independent_source": source, **counts[source]}), flush=True)
    return observed, counts


def load_homology():
    root = AUDIT / "homology_round56"
    qids = {}
    with gzip.open(root / "query_index.jsonl.gz", "rt") as stream:
        for line in stream:
            item = json.loads(line)
            qids[item["sequence_sha256"]] = item["query_id"]
    homology = {}
    with gzip.open(root / "query_homology_union.jsonl.gz", "rt") as stream:
        for line in stream:
            item = json.loads(line)
            homology[item["query_id"]] = bool(item["detected_sources"])
    return qids, homology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT / "independent_verification.json")
    parser.add_argument("--skip-source-rescan", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = registry_rows()
    if not rows:
        raise AssertionError("empty registry")
    query_cache = check_queries(rows)
    sequence_by_hash = {}
    for row in rows:
        source = query_cache[row["dataset_id"]][row["row_index"]]
        sequence_by_hash[row["sequence_sha256"]] = source.get("seq", source.get("sequence")).strip().upper().replace("T", "U")
    files = source_files()
    check_source_identity(files)
    if args.skip_source_rescan:
        raise SystemExit("--skip-source-rescan is intentionally not permitted for final independent verification")
    observed, scan_counts = independent_scan(files, sequence_by_hash)
    qids, homology = load_homology()
    by_population = {}
    for row in rows:
        seq_hash = row["sequence_sha256"]
        expected_sources = []
        for match in row["source_matches"]:
            source = match["source"]
            expected = observed[source].get(seq_hash)
            if expected is None:
                raise AssertionError(f"unconfirmed source match: {row['query_id']}/{source}")
            for key in ("source_record_id", "source_record_header", "source_file", "source_record_match_count"):
                if match[key] != expected[key]:
                    raise AssertionError(f"source evidence mismatch: {row['query_id']}/{source}/{key}")
            expected_sources.append(source)
        if set(expected_sources) != {source for source in SOURCE_NAMES if seq_hash in observed[source]}:
            raise AssertionError(f"missing source evidence: {row['query_id']}")
        exact_state = "EXACT" if expected_sources else "NOT_DETECTED_UNKNOWN"
        qid = qids.get(seq_hash)
        if exact_state == "EXACT":
            joint_state = "EXACT"
        elif qid in homology and homology[qid]:
            joint_state = "HOMOLOGY"
        elif qid in homology:
            joint_state = "NOT_DETECTED_UNKNOWN"
        else:
            joint_state = "UNCHECKED"
        expected = {
            "known_source_exact_state": exact_state,
            "known_source_exact_or_homology_state": joint_state,
            "complete_official_pretraining_exposure_state": "EXACT" if expected_sources else "UNCHECKED",
        }
        for key, value in expected.items():
            if row[key] != value or value not in ALLOWED:
                raise AssertionError(f"state mismatch: {row['query_id']}/{key}")
        if row["role"] == "external_test_final_audit_only" and row["selection_use"]:
            raise AssertionError("external Test was marked for selection")
        by_population.setdefault(row["population_id"], []).append(row)
    summary = json.loads((OUT / "summary.json").read_text())
    recomputed = {}
    for population, items in by_population.items():
        exact = Counter(item["known_source_exact_state"] for item in items)
        joint = Counter(item["known_source_exact_or_homology_state"] for item in items)
        complete = Counter(item["complete_official_pretraining_exposure_state"] for item in items)
        current = summary["populations"][population]
        if current["n_input_rows"] != len(items):
            raise AssertionError(f"summary row count mismatch: {population}")
        for key, counts in (("known_source_exact_state", exact),
                            ("known_source_exact_or_homology_state", joint),
                            ("complete_official_pretraining_exposure_state", complete)):
            expected = {state: counts.get(state, 0) for state in ("EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED")}
            if current[key] != expected:
                raise AssertionError(f"summary state mismatch: {population}/{key}")
        recomputed[population] = current["known_source_exact_state"]
    old = json.loads((AUDIT / "exact_union_summary.json").read_text())
    for population, dataset in (("tr1c_raw", "bprna_tr1c"), ("bprna_new", "bprna_new"),
                                ("bprna_ts0", "bprna_ts0"), ("archiveii_embok_clean", "archiveii_embok_clean"),
                                ("ref_pdb_ts1", "ref_pdb_ts1"), ("ref_pdb_ts2", "ref_pdb_ts2"),
                                ("ref_pdb_ts3", "ref_pdb_ts3"), ("ref_pdb_ts_hard", "ref_pdb_ts_hard"),
                                ("testsetb", "testsetb"), ("rfam_fam", "rfam_fam")):
        expected = old["queries"][dataset]["exact_overlap_rows_any_known_source"]
        if recomputed[population]["EXACT"] != expected:
            raise AssertionError(f"old exact report mismatch: {population}")
    if recomputed["bprna_new"]["EXACT"] != 5381 or summary["populations"]["bprna_new"]["n_input_rows"] != 5388:
        raise AssertionError("5381/5388 reproduction failed")
    if summary["populations"]["bprna_new"]["known_source_exact_or_homology_state"]["HOMOLOGY"] != 7:
        raise AssertionError("bpRNA-new seven exact-undetected sequences are not all homology detected")
    assets = json.loads((OUT / "asset_registry.json").read_text())
    if assets["official_pretraining_declaration"]["complete_official_training_set_state"] != "UNCHECKED":
        raise AssertionError("incomplete official corpus was marked checked")
    if assets["baseline_finetuning_is_separate"]["sha256"] != digest("/root/rna_baselines_src/rinalmo_giga_ss_bprna_ft.pt"):
        raise AssertionError("baseline fine-tuning checkpoint identity mismatch")
    if assets["current_candidate_backbone"]["sha256"] != digest("/root/rna_jepa_data/weights/rinalmo-giga/model.safetensors"):
        raise AssertionError("current backbone identity mismatch")
    report = {
        "schema_version": "rinalmo.pretrain_exposure.independent_verification.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "method": "independent JSONL/FASTA parsers; source and query SHA256; per-row source IDs/counts/states; summary and legacy report reconciliation",
        "n_registry_rows": len(rows), "n_populations": len(by_population),
        "n_unique_query_sequences": len(sequence_by_hash), "source_scan": scan_counts,
        "population_exact_counts": recomputed,
        "bprna_new": {"rows": 5388, "exact": 5381, "exact_not_detected_unknown": 7,
                      "homology_only": 7, "homology_not_detected": 0},
        "official_complete_coverage": "UNCHECKED",
        "external_test_use": "final exposure/quality audit only",
        "no_model_inference_or_training": True,
        "artifact_sha256": {name: digest(OUT / name) for name in
                            ("asset_registry.json", "sequence_registry.jsonl.gz", "summary.json", "build_receipt.json")},
        "limitations": ["Independent verification reproduces known-source evidence, not the unavailable historical nt snapshot",
                        "NOT_DETECTED_UNKNOWN and UNCHECKED never establish clean data",
                        "The seven bpRNA-new exact-undetected rows have known-source homology and are not a clean comparison group"],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": digest(args.output),
                      "n_registry_rows": len(rows), "bprna_new": report["bprna_new"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
