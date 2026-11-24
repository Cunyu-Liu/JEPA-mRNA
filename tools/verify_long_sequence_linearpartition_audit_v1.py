#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_UPSTREAM_COMMIT = "b450fb3e63189073b68d385589035f992080aa3a"
CANONICAL_BASE_PAIRS = {"G-C", "C-G", "A-U", "U-A", "G-U", "U-G"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256_text(payload)


def normalise_sequence(sequence: str) -> str:
    if not sequence:
        raise ValueError("empty sequence")
    normalised = sequence.upper().replace("T", "U")
    if set(normalised) - set("ACGUN"):
        raise ValueError("unsupported nucleotide alphabet")
    return normalised


def source_to_frozen_row(source_row: dict) -> dict:
    converted = dict(source_row)
    converted["seq"] = normalise_sequence(source_row["seq"])
    pairs = [[left - 1, right - 1] for left, right in source_row["pairs"]]
    converted["pairs"] = pairs
    converted["canonical_nested_pairs"] = pairs
    return converted


def validate_zero_based_pairs(pairs, length: int) -> list[tuple[int, int]]:
    result = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("pair must have two coordinates")
        left, right = pair
        if isinstance(left, bool) or isinstance(right, bool):
            raise ValueError("Boolean coordinate")
        if not isinstance(left, int) or not isinstance(right, int):
            raise ValueError("non-integer coordinate")
        if not 0 <= left < right < length:
            raise ValueError("pair outside full sequence")
        result.append((left, right))
    if len(set(result)) != len(result):
        raise ValueError("duplicate pair")
    positions = [position for pair in result for position in pair]
    if len(set(positions)) != len(positions):
        raise ValueError("partner conflict")
    for index, (left, right) in enumerate(result):
        for other_left, other_right in result[index + 1:]:
            if left < other_left < right < other_right or other_left < left < other_right < right:
                raise ValueError("crossing pair in canonical nested labels")
    return result


def reference_helices(pairs, minimum_length: int = 2) -> tuple[list[dict], int]:
    pair_set = set(pairs)
    helices = []
    isolated = 0
    for left, right in sorted(pair_set):
        if (left - 1, right + 1) in pair_set:
            continue
        stack = []
        while (left, right) in pair_set:
            stack.append((left, right))
            left += 1
            right -= 1
        if len(stack) < minimum_length:
            isolated += 1
            continue
        helices.append({
            "outer_pair": list(stack[0]),
            "inner_pair": list(stack[-1]),
            "length": len(stack),
            "pairs": [list(pair) for pair in stack],
        })
    return helices, isolated


def candidate_and_helix_recall(reference_pairs, candidate_pairs, minimum_helix_length: int = 2) -> dict:
    reference = set(reference_pairs)
    candidate = set(candidate_pairs)
    helices, isolated = reference_helices(reference, minimum_helix_length)
    true_positive = len(reference & candidate)
    complete = sum(all(tuple(pair) in candidate for pair in helix["pairs"]) for helix in helices)
    return {
        "coordinate_convention": "zero-based inclusive",
        "reference_pair_count": len(reference),
        "candidate_pair_count": len(candidate),
        "true_positive_pair_count": true_positive,
        "false_negative_pair_count": len(reference - candidate),
        "candidate_pair_recall": true_positive / len(reference) if reference else None,
        "reference_helix_count": len(helices),
        "completely_recovered_helix_count": complete,
        "complete_helix_recall": complete / len(helices) if helices else None,
        "isolated_reference_pair_count_excluded_from_helix_metric": isolated,
        "helix_definition": f"maximal consecutive (i,j),(i+1,j-1) stacks with length >= {minimum_helix_length}; complete recovery requires every pair",
    }


def parse_linearpartition_bpp(path: Path, sequence_length: int) -> list[dict]:
    pairs = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"invalid BPP row {line_number}")
            left, right = int(fields[0]), int(fields[1])
            probability = float(fields[2])
            if not 1 <= left < right <= sequence_length:
                raise ValueError(f"BPP coordinate outside 1-based full sequence at row {line_number}")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"invalid BPP probability at row {line_number}")
            pair = (left - 1, right - 1)
            if pair in pairs:
                raise ValueError(f"duplicate BPP pair at row {line_number}")
            pairs[pair] = probability
    return [{"pair": list(pair), "probability": pairs[pair]} for pair in sorted(pairs)]


def validate_full_coverage(records: list[dict], expected: dict[str, dict]) -> dict:
    names = [record.get("name") for record in records]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("prediction records do not exactly cover expected chains")
    for record in records:
        target = expected[record["name"]]
        if record.get("input_length") != target["length"]:
            raise ValueError("input length changed")
        if record.get("input_sequence_sha256") != target["sequence_sha256"]:
            raise ValueError("input sequence identity changed")
        status = record.get("status")
        if status == "COMPLETED":
            if record.get("output_length") != target["length"]:
                raise ValueError("completed output is not full length")
            validate_zero_based_pairs(record.get("pairs", []), target["length"])
        elif status != "FAILED_EXPLICIT":
            raise ValueError("each target must be COMPLETED or FAILED_EXPLICIT")
    return {"status": "PASS_FULL_COVERAGE", "n_targets": len(expected)}


def approximation_label(exact_global_dp: bool, approximate_or_sparse: bool) -> str:
    if exact_global_dp and not approximate_or_sparse:
        return "EXACT_GLOBAL_DP"
    return "APPROXIMATE_SPARSE"


def unmeasured_resources(reason: str) -> dict:
    return {
        "memory": {
            "status": "NOT_MEASURED", "peak_cpu_rss_bytes": None,
            "peak_gpu_allocated_bytes": None, "peak_gpu_reserved_bytes": None,
            "reason": reason,
        },
        "flops": {
            "status": "NOT_MEASURED", "value": None, "unit": "FLOPs",
            "reason": reason, "required_method": "profile the exact full-length execution; do not infer from short chains",
        },
        "latency": {
            "status": "NOT_MEASURED", "wall_seconds": None,
            "reason": reason, "required_method": "timed full-length execution after protocol freeze",
        },
    }


def scan_jsonl(path: Path, target_length: int) -> dict:
    count = 0
    maximum = 0
    over_1024 = 0
    over_3819 = 0
    targets = []
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            length = len(row["seq"])
            count += 1
            maximum = max(maximum, length)
            over_1024 += length > 1024
            over_3819 += length > 3819
            if length == target_length:
                targets.append({"row_index": row_index, "row": row})
    return {
        "n_rows": count, "max_length": maximum, "n_over_1024": over_1024,
        "n_over_3819": over_3819, "targets": targets,
    }


def select_named_rows(path: Path, names: set[str]) -> dict[str, dict]:
    selected = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            name = row.get("name")
            if name in names:
                if name in selected:
                    raise AssertionError(f"duplicate source identity: {name}")
                selected[name] = row
    if set(selected) != names:
        raise AssertionError(f"missing source identities: {sorted(names - set(selected))}")
    return selected


def file_record(path: Path, expected_sha256: str | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise AssertionError(f"SHA256 mismatch: {path}")
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": actual, "expected_sha256": expected_sha256,
        "hash_match": True if expected_sha256 is not None else None,
        "hash_scope": "full_content",
    }


def verify_original_label(row: dict) -> dict:
    source = row["source"]
    label_path = Path(source["source_json"])
    label_hash = sha256_file(label_path)
    if label_hash != source["source_json_sha256"]:
        raise AssertionError(f"label JSON identity mismatch: {label_path}")
    document = json.loads(label_path.read_text())
    matches = [chain for chain in document["chains"] if chain["id"] == row["name"]]
    if len(matches) != 1:
        raise AssertionError(f"original chain not unique: {row['name']}")
    chain = matches[0]
    sequence = normalise_sequence(row["seq"])
    if normalise_sequence(chain["sequence"]) != sequence or source["sequence"] != sequence:
        raise AssertionError(f"sequence identity mismatch: {row['name']}")
    if len(sequence) != chain["length"] or len(sequence) != source["length"]:
        raise AssertionError(f"nucleotide count mismatch: {row['name']}")
    if chain["raw_pairs"] != row["raw_pairs"] or source["raw_pairs"] != row["raw_pairs"]:
        raise AssertionError(f"raw pair identity mismatch: {row['name']}")
    if chain["canonical_nested_pairs"] != source["canonical_nested_pairs"]:
        raise AssertionError(f"source canonical pair mismatch: {row['name']}")
    if len(chain["raw_pair_records"]) != len(chain["raw_pairs"]):
        raise AssertionError(f"raw DSSR record count mismatch: {row['name']}")
    if chain["n_pseudoknot_or_noncanonical"] != len(chain["raw_pairs"]) - len(chain["canonical_nested_pairs"]):
        raise AssertionError(f"label filtering count mismatch: {row['name']}")
    related = {}
    for key, filename in (("cif_sha256", f"{chain['pdb_id']}.cif"),
                          ("dssr_json_sha256", "dssr.json"),
                          ("parent_map_sha256", "parents.json")):
        related[key] = file_record(label_path.with_name(filename), document[key])
    validate_zero_based_pairs(row["pairs"], len(sequence))
    return {
        "source_json": file_record(label_path, source["source_json_sha256"]),
        "related_source_files": related,
        "pdb_id": chain["pdb_id"], "auth_chain": chain["auth_chain"],
        "asym_ids": chain["asym_ids"], "entity_ids": chain["entity_ids"],
        "experimental_method": document["experimental_method"],
        "resolution_angstrom": document["resolution_angstrom"],
        "deposition_date": document["deposition_date"],
        "dssr_program": document["dssr_program"],
        "dssr_seq_id_mode": document["dssr_seq_id_mode"],
        "observed_alignment": document["observed_alignment"],
        "modified_parent_map_verified_through_parent_map_hash": True,
        "raw_dssr_pair_count": len(chain["raw_pairs"]),
        "canonical_nested_pair_count_1based": len(chain["canonical_nested_pairs"]),
        "raw_pair_record_count": len(chain["raw_pair_records"]),
        "n_pseudoknot_or_noncanonical": chain["n_pseudoknot_or_noncanonical"],
        "sequence_exact_after_uppercase_T_to_U_only": True,
        "nucleotide_count_preserved": True,
        "raw_pairs_preserved_1based": True,
        "canonical_nested_pairs_match_source": True,
        "dssr_binary_sha256_declared": document["dssr_binary_sha256"],
        "dssr_binary_reexecution": "NOT_RUN_CPU_AUDIT_DOES_NOT_REANNOTATE_PDB",
    }


def active_gec_cuda_processes() -> int:
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"tools/train_plan_a_gec_convergence_v1.py" in command and b"--device\x00cuda" in command:
            count += 1
    return count


def git_output(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True).strip()


def upstream_record(repository: Path) -> dict:
    remote = git_output(repository, "remote", "get-url", "origin")
    commit = git_output(repository, "rev-parse", "HEAD")
    if remote != "https://github.com/LinearFold/LinearPartition.git" or commit != EXPECTED_UPSTREAM_COMMIT:
        raise AssertionError("unexpected LinearPartition upstream identity")
    if git_output(repository, "status", "--porcelain", "--untracked-files=no"):
        raise AssertionError("tracked upstream source has local modifications")
    files = ["README.md", "LICENSE", "Makefile", "linearpartition", "src/LinearPartition.cpp",
             "src/LinearPartition.h", "src/bpp.cpp", "src/Utils/feature_weight.h",
             "src/Utils/energy_parameter.h"]
    checkpoint_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}
    checkpoints = [str(path) for path in repository.rglob("*")
                   if path.is_file() and path.suffix.lower() in checkpoint_suffixes and ".git" not in path.parts]
    common = {
        "beam_size": 100, "sharpturn": False, "dangles": 2, "bpp_cutoff": 0.0,
        "mea_gamma_if_structure_requested": 3.0, "input": "full-length FASTA with --fasta; no truncation",
        "execution_label": approximation_label(False, True),
        "execution_label_reason": "finite beam pruning and sparse emitted BPP support are not exact global cubic DP",
    }
    return {
        "official_repository": remote, "commit": commit,
        "tree": git_output(repository, "rev-parse", "HEAD^{tree}"),
        "commit_utc": git_output(repository, "show", "-s", "--format=%cI", "HEAD"),
        "tag_v1_0_commit": git_output(repository, "rev-parse", "v1.0^{}"),
        "version_selection": "official master pinned by full commit; v1.0 differs and must not be substituted silently",
        "source_files": [file_record(repository / name) for name in files],
        "external_model_checkpoints": checkpoints,
        "weights_status": "NO_EXTERNAL_MODEL_CHECKPOINT_FOUND; embedded variant parameters are hashed source assets",
        "parameter_provenance": {
            "LinearPartition-C": "default; embedded feature_weight.h generated from feature weights; do not describe as parameter-free",
            "LinearPartition-V": "enabled by -V; embedded ViennaRNA energy_parameter.h",
        },
        "license": {
            "status": "VERIFIED_CUSTOM_LICENSE_NOT_SPDX_ASSIGNED",
            "path": str(repository / "LICENSE"), "sha256": sha256_file(repository / "LICENSE"),
        },
        "protocols": {
            "LinearPartition-C": dict(common, variant_flag=None, command_template="./linearpartition --fasta -b 100 -c 0.0 --rewrite FULL_PATH.bpp < FULL_PATH.fasta"),
            "LinearPartition-V": dict(common, variant_flag="-V", command_template="./linearpartition -V --fasta -b 100 -c 0.0 --rewrite FULL_PATH.bpp < FULL_PATH.fasta"),
        },
        "protocol_freeze_requirement": "register C or V explicitly before scientific comparison; do not merge with LinearFold or LinearPartition-E",
        "tracked_source_clean": True,
    }


def write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def run_audit(args) -> dict:
    repo = args.repo.resolve()
    evidence = args.evidence_dir.resolve()
    manifest_path = repo / "source_data/round53/rinalmo_full_d1_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "PASS_RINALMO_FULL_D1_DATA":
        raise AssertionError("D1 manifest is not a passed frozen asset manifest")
    train_path = Path(manifest["train_d1"])
    holdout_path = Path(manifest["holdout"])
    train_file = file_record(train_path, manifest["train_d1_sha256"])
    holdout_file = file_record(holdout_path, manifest["holdout_sha256"])
    train = scan_jsonl(train_path, 7224)
    holdout = scan_jsonl(holdout_path, 19000)
    if (train["n_rows"], train["max_length"], train["n_over_1024"], train["n_over_3819"]) != (
            manifest["n_total_train"], 7224, manifest["train_over_1024"], manifest["train_over_3819"]):
        raise AssertionError("training full-file counts do not match manifest")
    if (holdout["n_rows"], holdout["max_length"], holdout["n_over_1024"], holdout["n_over_3819"]) != (
            manifest["n_complex_holdout"], 19000, manifest["holdout_over_1024"], manifest["holdout_over_3819"]):
        raise AssertionError("holdout full-file counts do not match manifest")
    if not train["targets"] or not holdout["targets"]:
        raise AssertionError("registered long-chain lengths not recovered")

    decon_dir = holdout_path.parent.parent / "decontamination"
    verification = json.loads((decon_dir / "verification.json").read_text())
    if verification["status"] != "PASS_COMPLEX_DECONTAMINATION_INDEPENDENT_AUDIT":
        raise AssertionError("decontamination verification did not pass")
    source_files = {
        "tr1c": file_record(Path(manifest["tr1c"]), manifest["tr1c_sha256"]),
        "decontamination_verification": file_record(decon_dir / "verification.json", manifest["source_hashes"]["verification"]),
        "decontamination_manifest": file_record(decon_dir / "decontamination.json", manifest["source_hashes"]["decontamination"]),
        "complex_train_1based": file_record(decon_dir / "complex_train.jsonl", manifest["source_hashes"]["complex_train"]),
        "complex_holdout_1based": file_record(decon_dir / "complex_holdout.jsonl", manifest["source_hashes"]["complex_holdout"]),
    }
    train_sources = select_named_rows(decon_dir / "complex_train.jsonl", {item["row"]["name"] for item in train["targets"]})
    holdout_sources = select_named_rows(decon_dir / "complex_holdout.jsonl", {item["row"]["name"] for item in holdout["targets"]})
    targets = []
    complete_rows = []
    for split, items, source_rows, frozen_file in (
            ("train", train["targets"], train_sources, train_file),
            ("holdout", holdout["targets"], holdout_sources, holdout_file)):
        for item in items:
            row = item["row"]
            name = row["name"]
            source_row = source_rows[name]
            if row != source_to_frozen_row(source_row):
                raise AssertionError(f"frozen/source conversion mismatch: {name}")
            if row["raw_pairs"] != source_row["raw_pairs"]:
                raise AssertionError(f"raw pairs changed during conversion: {name}")
            sequence = normalise_sequence(row["seq"])
            validate_zero_based_pairs(row["pairs"], len(sequence))
            original = verify_original_label(row)
            helices, isolated = reference_helices(validate_zero_based_pairs(row["pairs"], len(sequence)))
            resource_reason = "CPU-only provenance audit; no long-chain inference; seed0 CUDA training was not interrupted"
            target = {
                "name": name, "split": split, "row_index_zero_based": item["row_index"],
                "length": len(sequence), "sequence_sha256": sha256_text(sequence),
                "frozen_file": frozen_file,
                "pair_counts": {
                    "raw_dssr_1based": len(row["raw_pairs"]),
                    "canonical_nested_frozen_0based": len(row["pairs"]),
                },
                "pairs_sha256": sha256_json(row["pairs"]),
                "raw_pairs_sha256": sha256_json(row["raw_pairs"]),
                "source_conversion": {
                    "status": "PASS_EXACT_ROW_IDENTITY",
                    "sequence": "uppercase and T->U only; length and nucleotide order preserved",
                    "pairs": "source 1-based inclusive -> frozen 0-based inclusive by subtracting one from both endpoints",
                    "raw_pairs": "unchanged 1-based inclusive DSSR-derived pairs",
                    "source_embedded_canonical_pairs": "unchanged 1-based inclusive provenance copy",
                    "label_scope": "evaluation uses frozen canonical nested pairs, not all raw DSSR pairs",
                },
                "original_label_verification": original,
                "evaluation_plan": {
                    "status": "BLOCKED_NO_LONG_CHAIN_PREDICTIONS",
                    "reference": "frozen canonical_nested_pairs == pairs, zero-based inclusive",
                    "candidate_universe": "all pairs emitted by the preregistered protocol at its frozen BPP cutoff; report cutoff and do not treat absent approximate pairs as exact zeros",
                    "pair_metric": "candidate_pair_recall = |reference intersect candidate| / |reference|",
                    "helix_metric": "complete reference helix recall for maximal stacked helices of length >=2; also retain per-pair recall",
                    "reference_helix_count": len(helices),
                    "isolated_reference_pair_count_excluded_from_helix_metric": isolated,
                    "candidate_pair_recall": None, "complete_helix_recall": None,
                    "precision_f1": None,
                    "reason": "no verified LinearPartition-class or GEC predictions for this chain",
                },
                "full_coverage_output_requirements": {
                    "one_explicit_record_per_target": True,
                    "input_length_and_sequence_sha256_must_match": True,
                    "no_truncation_windowing_or_nucleotide_deletion": True,
                    "completed_bpp_pairs_must_be_in_bounds_for_full_length": True,
                    "completed_dot_bracket_or_bpseq_must_cover_full_length": True,
                    "failed_targets_must_be_FAILED_EXPLICIT_not_omitted_or_zero_scored": True,
                    "sparse_bpp_is_allowed_but_must_be_labeled_APPROXIMATE_SPARSE_when_beam_pruned_or_approximate": True,
                },
                "resource_measurements": {
                    "LinearPartition-class": unmeasured_resources(resource_reason),
                    "GEC": unmeasured_resources(resource_reason + "; exact O(L^3) support is not established"),
                },
            }
            targets.append(target)
            complete_rows.append(row)

    data_registry = json.loads((repo / "manifests/foldnar_v1/data_registry.json").read_text())
    registry_rows = {row["dataset_id"]: row for row in data_registry["rows"]
                     if row.get("dataset_id") in {"longchain_7224", "longchain_19000"}}
    if set(registry_rows) != {"longchain_7224", "longchain_19000"}:
        raise AssertionError("registered long-chain contract rows absent")
    baseline_registry = json.loads((repo / "manifests/foldnar_v1/baseline_registry.json").read_text())
    baseline_rows = [row for row in baseline_registry["rows"] if row.get("baseline_id") == "linearpartition"]
    if len(baseline_rows) != 1:
        raise AssertionError("LinearPartition baseline row not unique")
    prediction_registry_path = repo / "manifests/foldnar_v1/prediction_registry.json"
    prediction_registry = json.loads(prediction_registry_path.read_text())
    predictions = [row for row in prediction_registry["rows"]
                   if "linearpartition" in json.dumps(row).lower()
                   or row.get("dataset_id") in {"longchain_7224", "longchain_19000"}]
    upstream = upstream_record(args.upstream_dir.resolve())
    gec_cuda_count = active_gec_cuda_processes()
    report = {
        "schema_version": "long_sequence_audit_v1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_CPU_ASSET_AND_PROVENANCE_VERIFICATION_WITH_REPRODUCIBILITY_BLOCKERS",
        "scientific_completion": False,
        "scope": "read-only audit; no registry/config/evaluator/PPT modification; no long-chain inference or new model",
        "manifest": file_record(manifest_path),
        "frozen_files": {"train": train_file, "holdout": holdout_file},
        "full_file_counts": {
            "train": {key: value for key, value in train.items() if key != "targets"},
            "holdout": {key: value for key, value in holdout.items() if key != "targets"},
            "n_7224_chains": len(train["targets"]), "n_19000_chains": len(holdout["targets"]),
        },
        "source_files": source_files,
        "targets": targets,
        "registry_reconciliation": {
            "status": "BLOCKED_STALE_REGISTRY_LINKAGE_NOT_MODIFIED",
            "rows": registry_rows,
            "finding": "registered rows have path=null/exists=false/BLOCKED although the frozen D1 assets and exact long rows are present and verified",
            "required_followup": "a separately authorized registry update should link both 7224-nt chains and the 19000-nt chain without changing their frozen hashes",
        },
        "linearpartition": {
            "upstream": upstream,
            "registered_baseline_row": baseline_rows[0],
            "existing_prediction_count": len(predictions),
            "existing_predictions": predictions,
            "prediction_registry": file_record(prediction_registry_path),
            "related_artifacts": [
                {"path": "/home/rna_jev/third_party/EternaFold/LinearPartition-E.patch", "classification": "DIFFERENT_VARIANT_NOT_A_C_OR_V_PREDICTION"},
                {"path": "/teams/HoulabData_1788502323/liucunyu/codonflow1031/rk05/ensembledesign_build_cd9324_v1/linearpartition.stdout.log", "classification": "BUILD_LOG_ONLY_NO_SEQUENCE_PREDICTION"},
            ],
            "status": "OFFICIAL_SOURCE_AND_PROTOCOL_RECOVERED__LONG_CHAIN_PREDICTIONS_UNAVAILABLE",
        },
        "execution_policy": {
            "gec_exact_cubic_long_chain_support": "NOT_ESTABLISHED",
            "gec_claim_prohibition": "do not claim exact O(L^3) GEC supports 7224 or 19000 nt without full-length exact execution and output verification",
            "approximation_label": "APPROXIMATE_SPARSE",
            "approximation_label_required_for": ["beam pruning", "top-k or cutoff candidate pruning", "local windows", "sparse candidate decoding presented as approximate", "any non-exact global DP"],
            "short_sequence_gate": "no new long-sequence model before short-sequence GEC passes",
            "gpu_inference_executed": False,
            "active_gec_cuda_training_processes_observed": gec_cuda_count,
        },
        "blockers": [
            "REGISTRY_LONGCHAIN_LINKAGE_STALE_AND_NOT_MODIFIED",
            "LINEARPARTITION_REGISTERED_VARIANT_AND_PROTOCOL_NOT_YET_UPDATED",
            "NO_VERIFIED_LONG_CHAIN_LINEARPARTITION_OR_GEC_PREDICTIONS",
            "CANDIDATE_PAIR_AND_COMPLETE_HELIX_RECALL_NOT_COMPUTABLE",
            "FULL_COVERAGE_OUTPUT_NOT_YET_AVAILABLE",
            "MEMORY_FLOPS_LATENCY_NOT_MEASURED",
            "EXACT_CUBIC_GEC_LONG_CHAIN_SUPPORT_NOT_ESTABLISHED",
            "SHORT_SEQUENCE_GEC_GATE_STILL_BLOCKS_NEW_LONG_SEQUENCE_MODEL_WORK",
        ],
    }
    target_json = {target["name"]: {"length": target["length"], "sequence_sha256": target["sequence_sha256"]} for target in targets}
    report["coverage_validator_contract"] = {
        "expected_targets": target_json,
        "validator": "validate_full_coverage in tools/verify_long_sequence_linearpartition_audit_v1.py",
    }
    rows_path = evidence / "long_chain_records_complete.jsonl"
    write_new_text(rows_path, "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in complete_rows))
    report["complete_long_chain_records"] = file_record(rows_path)
    output = evidence / "audit_report.json"
    write_new_text(output, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"status": report["status"], "output": str(output), "complete_rows": str(rows_path),
            "targets": [{"name": target["name"], "length": target["length"],
                         "pairs": target["pair_counts"]["canonical_nested_frozen_0based"],
                         "helices": target["evaluation_plan"]["reference_helix_count"]} for target in targets],
            "existing_long_chain_predictions": len(predictions),
            "active_gec_cuda_training_processes_observed": gec_cuda_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/long_sequence_audit_v1"))
    parser.add_argument("--upstream-dir", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/long_sequence_audit_v1/upstream_linearpartition"))
    args = parser.parse_args()
    result = run_audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
