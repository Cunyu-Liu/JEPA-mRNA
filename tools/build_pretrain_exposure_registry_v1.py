#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
REPO = Path("/home/rna_jev_pretrain_exposure")
DATA_REGISTRY = REPO / "manifests/foldnar_v1/data_registry.json"
MODEL_REGISTRY = REPO / "manifests/foldnar_v2/model_dev_registry.json"
SOURCES = ("rnacentral_22_active", "rfam_14_9", "ensembl_109_and_bacteria_56")
STATES = {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}
OLD_EXACT_DATASETS = {
    "bprna_tr1c", "bprna_new", "bprna_ts0", "archiveii_embok_clean",
    "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard",
    "testsetb", "rfam_fam",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(sequence):
    return sequence.strip().upper().replace("T", "U")


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def iter_fasta(path):
    header = None
    chunks = []
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, normalize("".join(chunks))
                header = line[1:]
                chunks = []
            elif line:
                if header is None:
                    raise ValueError(f"sequence before FASTA header: {path}")
                chunks.append(line)
    if header is not None:
        yield header, normalize("".join(chunks))


def stable_id(dataset, index, row):
    name = row.get("name", row.get("id", f"row_{index}"))
    return f"{dataset}:row{index:06d}:{name}"


def load_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl_gz(path, rows):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_assets():
    initial = json.loads((AUDIT / "source_manifest.json").read_text())
    by_name = {row["name"]: row for row in initial["sources"]}
    assets = {}
    for name in SOURCES[:2]:
        row = by_name[name]
        path = Path(row["local_path"])
        if sha256(path) != row["sha256"]:
            raise AssertionError(f"source SHA256 mismatch: {name}")
        assets[name] = {
            "release": row["release"], "files": [{"path": str(path), "sha256": row["sha256"], "bytes": row["bytes"]}],
            "url": row["url"], "license_evidence": row["license"],
        }
    verification_path = AUDIT / "ensembl_asset_verification.json"
    verification = json.loads(verification_path.read_text())
    if verification["status"] != "PASS_ENSEMBL_ASSETS_INDEPENDENTLY_VERIFIED":
        raise AssertionError("Ensembl assets are not independently verified")
    files = []
    for group in sorted(verification["verified_files"]):
        for item in sorted(verification["verified_files"][group], key=lambda value: value["path"]):
            path = Path(item["path"])
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise AssertionError(f"Ensembl source identity mismatch: {path}")
            files.append({"path": str(path), "sha256": item["sha256"], "bytes": item["bytes"], "url": item["url"]})
    assets["ensembl_109_and_bacteria_56"] = {
        "release": "Ensembl 109 plus Ensembl Bacteria 56", "files": files,
        "manifest": str(verification_path), "manifest_sha256": sha256(verification_path),
        "license_evidence": "https://www.ensembl.org/info/about/legal/disclaimer.html",
    }
    return assets, initial, verification


def load_populations():
    registry = json.loads(DATA_REGISTRY.read_text())
    registered = {row["dataset_id"]: row for row in registry["rows"]}
    model = json.loads(MODEL_REGISTRY.read_text())
    populations = []
    train_path = Path(model["route_panels"][0]["train_ids"]["path"])
    dev_path = Path(model["route_panels"][0]["validation_ids"]["path"])
    tr1c = registered["bprna_tr1c"]
    rows = load_jsonl(tr1c["path"])
    all_rows = []
    for index, row in enumerate(rows):
        query_id = stable_id("bprna_tr1c", index, row)
        all_rows.append({"dataset_id": "bprna_tr1c", "role": "train", "row_index": index,
                         "query_id": query_id, "sequence": normalize(row.get("seq", row.get("sequence")))})
    by_id = {row["query_id"]: row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise AssertionError("TR1c stable IDs are not unique")
    populations.append({"population_id": "tr1c_raw", "dataset_id": "bprna_tr1c", "role": "train",
                        "selection_use": False, "input": tr1c, "rows": all_rows})
    for population_id, role, path, expected in (
        ("tr1c_cluster80_train", "train", train_path, 33421),
        ("tr1c_cluster80_dev", "development", dev_path, 9143),
    ):
        ids = [line.strip() for line in path.open() if line.strip()]
        if len(ids) != expected or len(set(ids)) != expected or not set(ids).issubset(by_id):
            raise AssertionError(f"invalid registered population: {population_id}")
        selected = [dict(by_id[value]) for value in ids]
        for row in selected:
            row["role"] = role
        populations.append({"population_id": population_id, "dataset_id": "bprna_tr1c", "role": role,
                            "selection_use": role == "development",
                            "id_file": {"path": str(path), "sha256": sha256(path), "n_ids": len(ids)},
                            "input": tr1c, "rows": selected})
    for dataset_id, item in registered.items():
        if item.get("role") != "test" and dataset_id != "ncrna_rfamnovel_test":
            continue
        rows = []
        for index, row in enumerate(load_jsonl(item["path"])):
            rows.append({"dataset_id": dataset_id, "role": "external_test_final_audit_only", "row_index": index,
                         "query_id": stable_id(dataset_id, index, row),
                         "sequence": normalize(row.get("seq", row.get("sequence")))})
        populations.append({"population_id": dataset_id, "dataset_id": dataset_id,
                            "role": "external_test_final_audit_only", "selection_use": False,
                            "input": item, "rows": rows})
    for population in populations:
        item = population["input"]
        if sha256(item["path"]) != item["sha256"] or len(population["rows"]) == 0:
            raise AssertionError(f"query asset identity mismatch: {population['population_id']}")
        if population["population_id"] in {"tr1c_raw", "tr1c_cluster80_train", "tr1c_cluster80_dev"}:
            expected = len(load_jsonl(item["path"]))
        else:
            expected = item["n_sequences"]
        if population["population_id"] == "tr1c_raw" or population["role"] == "external_test_final_audit_only":
            if len(population["rows"]) != expected:
                raise AssertionError(f"query count mismatch: {population['population_id']}")
    return populations


def load_homology():
    root = AUDIT / "homology_round56"
    query_by_hash = {}
    with gzip.open(root / "query_index.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            query_by_hash[row["sequence_sha256"]] = row["query_id"]
    detected = {}
    searched = set()
    with gzip.open(root / "query_homology_union.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            searched.add(row["query_id"])
            detected[row["query_id"]] = bool(row["detected_sources"])
    return query_by_hash, searched, detected, {
        "query_index": str(root / "query_index.jsonl.gz"),
        "query_index_sha256": sha256(root / "query_index.jsonl.gz"),
        "union": str(root / "query_homology_union.jsonl.gz"),
        "union_sha256": sha256(root / "query_homology_union.jsonl.gz"),
        "summary": str(root / "homology_union_summary.json"),
        "summary_sha256": sha256(root / "homology_union_summary.json"),
        "protocol": "existing MMseqs2 80% identity, cov-mode 0, c 0.8; no new search",
    }


def scan_sources(assets, query_sequences):
    matches = {name: {} for name in SOURCES}
    scan_counts = {}
    for name in SOURCES:
        raw_count = 0
        filtered_count = 0
        for item in assets[name]["files"]:
            path = Path(item["path"])
            relative_path = str(path.relative_to(AUDIT))
            for header, sequence in iter_fasta(path):
                raw_count += 1
                if not 16 <= len(sequence) <= 8192:
                    continue
                filtered_count += 1
                if sequence not in query_sequences:
                    continue
                evidence = matches[name].setdefault(sequence, {
                    "source_record_id": header.split()[0], "source_record_header": header,
                    "source_file": relative_path, "source_record_match_count": 0,
                })
                evidence["source_record_match_count"] += 1
        scan_counts[name] = {"raw_fasta_records": raw_count, "records_after_original_length_filter": filtered_count}
        print(json.dumps({"source": name, **scan_counts[name], "matched_unique_queries": len(matches[name])}), flush=True)
    return matches, scan_counts


def state_counts(rows, key):
    counts = Counter(row[key] for row in rows)
    return {state: counts.get(state, 0) for state in ("EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED")}


def build(output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    assets, initial, ensembl_verification = source_assets()
    populations = load_populations()
    query_sequences = {row["sequence"] for population in populations for row in population["rows"]}
    matches, scan_counts = scan_sources(assets, query_sequences)
    query_by_hash, searched, detected, homology_evidence = load_homology()
    sequence_rows = []
    summaries = {}
    for population in populations:
        output_rows = []
        source_sequence_sets = {name: set() for name in SOURCES}
        for row in population["rows"]:
            sequence = row["sequence"]
            seq_hash = sequence_sha256(sequence)
            source_matches = []
            exact_sources = []
            for name in SOURCES:
                if sequence in matches[name]:
                    exact_sources.append(name)
                    source_sequence_sets[name].add(sequence)
                    evidence = matches[name][sequence]
                    source_matches.append({"source": name, **evidence})
            exact_state = "EXACT" if exact_sources else "NOT_DETECTED_UNKNOWN"
            query_id = query_by_hash.get(seq_hash)
            if exact_state == "EXACT":
                joint_state = "EXACT"
            elif query_id in searched and detected[query_id]:
                joint_state = "HOMOLOGY"
            elif query_id in searched:
                joint_state = "NOT_DETECTED_UNKNOWN"
            else:
                joint_state = "UNCHECKED"
            output_row = {
                "population_id": population["population_id"], "dataset_id": row["dataset_id"],
                "role": row["role"], "row_index": row["row_index"], "query_id": row["query_id"],
                "sequence_sha256": seq_hash, "normalized_length": len(sequence),
                "known_source_exact_state": exact_state,
                "known_source_exact_or_homology_state": joint_state,
                "complete_official_pretraining_exposure_state": "EXACT" if exact_sources else "UNCHECKED",
                "source_matches": source_matches,
                "homology_query_id": query_id if query_id in searched else None,
                "selection_use": population["selection_use"],
            }
            if set(value for key, value in output_row.items() if key.endswith("_state")) - STATES:
                raise AssertionError("invalid exposure state")
            output_rows.append(output_row)
            sequence_rows.append(output_row)
        exact_union = {name for name in SOURCES for sequence in source_sequence_sets[name]}
        old = None
        dataset_id = population["dataset_id"]
        if dataset_id in OLD_EXACT_DATASETS and population["population_id"] == dataset_id or population["population_id"] == "tr1c_raw":
            old_report = json.loads((AUDIT / "exact_union_summary.json").read_text())
            old = old_report["queries"][dataset_id]["exact_overlap_rows_any_known_source"]
        summaries[population["population_id"]] = {
            "dataset_id": dataset_id, "role": population["role"], "selection_use": population["selection_use"],
            "n_input_rows": len(output_rows), "n_unique_sequences": len({row["sequence"] for row in population["rows"]}),
            "known_source_exact_state": state_counts(output_rows, "known_source_exact_state"),
            "known_source_exact_or_homology_state": state_counts(output_rows, "known_source_exact_or_homology_state"),
            "complete_official_pretraining_exposure_state": state_counts(output_rows, "complete_official_pretraining_exposure_state"),
            "by_source": {name: {"exact_rows": sum(row["sequence"] in matches[name] for row in population["rows"]),
                                 "exact_unique_sequences": len(source_sequence_sets[name])} for name in SOURCES},
            "previous_exact_union_count": old,
        }
        if old is not None and summaries[population["population_id"]]["known_source_exact_state"]["EXACT"] != old:
            raise AssertionError(f"previous exact count mismatch: {population['population_id']}")
        if population["population_id"] == "bprna_new":
            if summaries[population["population_id"]]["known_source_exact_state"]["EXACT"] != 5381:
                raise AssertionError("bpRNA-new 5381/5388 evidence was not reproduced")
            if summaries[population["population_id"]]["known_source_exact_or_homology_state"]["HOMOLOGY"] != 7:
                raise AssertionError("bpRNA-new known-source homology evidence was not reproduced")
    asset_registry = {
        "schema_version": "rinalmo.pretrain_exposure.assets.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": "existing legal assets; static normalized full-sequence audit; no inference, training, downloads or model selection",
        "current_candidate_backbone": {
            "identity": "RiNALMo-giga", "loader": "tools/rinalmo_preflight.py",
            "path": "/root/rna_jepa_data/weights/rinalmo-giga/model.safetensors",
            "sha256": sha256("/root/rna_jepa_data/weights/rinalmo-giga/model.safetensors"),
            "config_path": "/root/rna_jepa_data/weights/rinalmo-giga/config.json",
            "config_sha256": sha256("/root/rna_jepa_data/weights/rinalmo-giga/config.json"),
            "architecture": {"hidden_size": 1280, "num_hidden_layers": 33, "max_position_embeddings": 1024},
            "upstream_distribution": "multimolecule/rinalmo-giga; direct byte identity to Zenodo rinalmo_giga_pretrained.pt is UNCHECKED",
        },
        "official_pretraining_declaration": {
            "repository": "https://github.com/lbcb-sci/RiNALMo", "commit": "2c2c5c14a5ae609d8c560a5d9ca32e51e0288955",
            "declared_sources": ["RNAcentral", "Rfam", "nt", "Ensembl 109", "Ensembl Bacteria 56"],
            "paper_preprocessing": "36M unique sequences; 17M clusters at 70% identity and 0.8 coverage; lengths 16..8192 nt; training crops above 1022 nt",
            "complete_official_training_set_state": "UNCHECKED",
            "reason": "No complete official sequence manifest; historical nt snapshot has no pinned release/date/SHA; local public snapshots must not be renamed as the exact historical training union",
            "nt": initial["nt"], "nt_assessment": str(AUDIT / "nt_snapshot_assessment.json"),
            "nt_assessment_sha256": sha256(AUDIT / "nt_snapshot_assessment.json"),
        },
        "known_public_source_snapshots": {name: {key: value for key, value in asset.items() if key != "files"} | {
            "n_files": len(asset["files"]), "bytes": sum(item["bytes"] for item in asset["files"]),
            "file_sha256": [item["sha256"] for item in asset["files"]] if len(asset["files"]) == 1 else None,
            "scan_counts": scan_counts[name],
        } for name, asset in assets.items()},
        "baseline_finetuning_is_separate": {
            "baseline_id": "rinalmo_ft", "checkpoint": "rinalmo_giga_ss_bprna_ft.pt",
            "path": "/root/rna_baselines_src/rinalmo_giga_ss_bprna_ft.pt",
            "sha256": sha256("/root/rna_baselines_src/rinalmo_giga_ss_bprna_ft.pt"),
            "fine_tuning_corpus": "bpRNA TR0 (registered downstream fine-tuning corpus)",
            "interpretation": "This baseline's supervised fine-tuning exposure is additional to and not evidence about the current candidate backbone's self-supervised pretraining exposure",
        },
        "homology_evidence": homology_evidence,
        "state_semantics": {
            "EXACT": "normalized full sequence is present in at least one hash-bound known public source",
            "HOMOLOGY": "no exact hit, but an existing 80% identity/bidirectional-coverage search detected a known-source hit",
            "NOT_DETECTED_UNKNOWN": "the declared known-source search did not detect a match; this is not clean or independent",
            "UNCHECKED": "coverage is unavailable or the complete historical corpus cannot be assessed",
        },
    }
    summary = {
        "schema_version": "rinalmo.pretrain_exposure.summary.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete_official_pretraining_coverage_state": "UNCHECKED",
        "normalization": "uppercase and T->U; preserve all other symbols; full sequence; source length filter 16..8192 nt",
        "populations": summaries,
        "external_test_policy": "final exposure/quality audit only; no model, arm, threshold, checkpoint or hyperparameter selection",
        "sota_limitations": [
            "Known-source EXACT or HOMOLOGY exposure invalidates clean-OOD, pretraining-unseen and uncontaminated-SOTA wording for the affected population",
            "NOT_DETECTED_UNKNOWN is not clean; UNCHECKED is not a negative exposure result",
            "Historical nt is unpinned, so no complete exposure rate, uncontaminated remainder or absolute generalization claim is identifiable",
            "Raw benchmark comparisons remain exposure-aware descriptive results, not proof of superiority on unseen RNA families",
            "RiNALMo-ft bpRNA TR0 fine-tuning must not be attributed to the current candidate backbone pretraining",
        ],
    }
    write_json(output_dir / "asset_registry.json", asset_registry)
    write_jsonl_gz(output_dir / "sequence_registry.jsonl.gz", sequence_rows)
    write_json(output_dir / "summary.json", summary)
    fingerprints = {name: sha256(output_dir / name) for name in ("asset_registry.json", "sequence_registry.jsonl.gz", "summary.json")}
    write_json(output_dir / "build_receipt.json", {"schema_version": "rinalmo.pretrain_exposure.build_receipt.v1",
                                                    "n_sequence_rows": len(sequence_rows), "sha256": fingerprints})
    print(json.dumps({"output": str(output_dir), "n_sequence_rows": len(sequence_rows),
                      "bprna_new": summaries["bprna_new"], "sha256": fingerprints}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO / "manifests/pretrain_exposure_v1")
    args = parser.parse_args()
    build(args.output_dir)
