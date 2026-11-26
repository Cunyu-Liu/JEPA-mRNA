#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

SPAN_BUCKETS = ("0-49", "50-199", "200-999", ">=1000")
EXPECTED_IDENTITIES = {
    7224: {"9MR4_5", "9Q7Q_5"},
    19000: {"7AM2_1"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def file_record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "hash_scope": "full_content"}


def normalise_sequence(sequence: str) -> str:
    result = sequence.upper().replace("T", "U")
    if not result or set(result) - set("ACGUN"):
        raise ValueError("invalid or empty RNA sequence")
    return result


def span_bucket(left: int, right: int) -> str:
    span = right - left
    if span < 50:
        return "0-49"
    if span < 200:
        return "50-199"
    if span < 1000:
        return "200-999"
    return ">=1000"


def validate_candidate_pairs(pairs, length: int) -> list[dict]:
    result = []
    seen = set()
    for item in pairs:
        if not isinstance(item, dict) or set(item) - {"pair", "probability"}:
            raise ValueError("each emitted pair requires pair and optional probability only")
        pair = item.get("pair")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("invalid pair coordinates")
        left, right = pair
        if isinstance(left, bool) or isinstance(right, bool) or not isinstance(left, int) or not isinstance(right, int):
            raise ValueError("pair coordinates must be integers")
        if not 0 <= left < right < length:
            raise ValueError("pair exceeds the untruncated input")
        if (left, right) in seen:
            raise ValueError("duplicate emitted pair")
        seen.add((left, right))
        probability = item.get("probability")
        if probability is not None and (isinstance(probability, bool) or not isinstance(probability, (int, float))
                                        or not math.isfinite(probability) or not 0.0 <= probability <= 1.0):
            raise ValueError("invalid pair probability")
        result.append({"pair": [left, right], "probability": probability})
    return sorted(result, key=lambda item: item["pair"])


def parse_bpp_artifact(path: Path, length: int) -> list[dict]:
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(">"):
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"invalid BPP row {row_number}")
            left, right, probability = int(fields[0]), int(fields[1]), float(fields[2])
            pairs.append({"pair": [left - 1, right - 1], "probability": probability})
    return validate_candidate_pairs(pairs, length)


def classify_execution(protocol: dict) -> str:
    exact = protocol.get("exact_global_dp") is True
    pruning = any(protocol.get(key) is True for key in
                  ("beam_pruning", "windowing", "top_k_pruning", "sparse_decoding"))
    cutoff = protocol.get("bpp_cutoff")
    if exact and not pruning and cutoff == 0.0:
        return "EXACT_GLOBAL_DP"
    if not exact:
        return "APPROXIMATE_SPARSE"
    raise ValueError("pruned execution cannot enter the exact global DP column")


def validate_resources(resources: dict, device: str) -> None:
    latency = resources.get("latency", {})
    if latency.get("status") != "MEASURED" or not isinstance(latency.get("wall_seconds"), (int, float)) \
            or not math.isfinite(latency["wall_seconds"]) or latency["wall_seconds"] <= 0:
        raise ValueError("completed target lacks measured latency")
    memory = resources.get("memory", {})
    if memory.get("status") != "MEASURED" or not isinstance(memory.get("peak_cpu_rss_bytes"), int) \
            or memory["peak_cpu_rss_bytes"] <= 0:
        raise ValueError("completed target lacks measured peak CPU RSS")
    if device == "cuda":
        for key in ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"):
            if not isinstance(memory.get(key), int) or memory[key] < 0:
                raise ValueError("CUDA execution lacks measured GPU memory")
    elif device == "cpu":
        if memory.get("peak_gpu_allocated_bytes") is not None or memory.get("peak_gpu_reserved_bytes") is not None:
            raise ValueError("CPU execution must use null, not zero, for inapplicable GPU memory")
    else:
        raise ValueError("unsupported execution device")
    flops = resources.get("flops", {})
    if flops.get("status") != "MEASURED" or not isinstance(flops.get("value"), (int, float)) \
            or not math.isfinite(flops["value"]) or flops["value"] < 0 \
            or flops.get("unit") != "FLOPs" or not flops.get("method") \
            or flops.get("estimated") is not False:
        raise ValueError("completed target lacks measured, non-estimated FLOPs")


def pair_recall(reference_pairs, candidate_pairs) -> dict:
    reference = {tuple(pair) for pair in reference_pairs}
    candidate = {tuple(item["pair"]) for item in candidate_pairs}
    result = {"overall": {"reference": len(reference), "true_positive": len(reference & candidate)}}
    result["overall"]["recall"] = result["overall"]["true_positive"] / len(reference) if reference else None
    for bucket in SPAN_BUCKETS:
        selected = {pair for pair in reference if span_bucket(*pair) == bucket}
        true_positive = len(selected & candidate)
        result[bucket] = {"reference": len(selected), "true_positive": true_positive,
                          "recall": true_positive / len(selected) if selected else None}
    return result


def named_pair_recall(reference, candidate) -> dict:
    reference, candidate = set(reference), set(candidate)
    result = {"overall": {"reference": len(reference), "true_positive": len(reference & candidate)}}
    result["overall"]["recall"] = result["overall"]["true_positive"] / len(reference) if reference else None
    for bucket in SPAN_BUCKETS:
        selected = {item for item in reference if span_bucket(item[1], item[2]) == bucket}
        true_positive = len(selected & candidate)
        result[bucket] = {"reference": len(selected), "true_positive": true_positive,
                          "recall": true_positive / len(selected) if selected else None}
    return result


def validate_predictions(records: list[dict], targets: list[dict], protocol: dict) -> dict:
    column = classify_execution(protocol)
    expected = {target["name"]: target for target in targets}
    names = [record.get("name") for record in records]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("prediction records must cover every target exactly once")
    per_target = []
    all_reference, all_candidate = [], []
    completed_count = 0
    completed_nucleotides = 0
    for record in records:
        target = expected[record["name"]]
        length = target["length"]
        if record.get("input_length") != length or record.get("input_sequence_sha256") != target["sequence_sha256"]:
            raise ValueError("input identity or untruncated length mismatch")
        if record.get("summary_only") is True:
            raise ValueError("summary-only rows are not predictions")
        status = record.get("status")
        if status == "FAILED_EXPLICIT":
            if not record.get("failure_reason"):
                raise ValueError("explicit failure requires a reason")
            per_target.append({"name": record["name"], "status": status, "metrics": None})
            continue
        if status != "COMPLETED" or record.get("output_length") != length:
            raise ValueError("completed target must have full-length output")
        pairs = validate_candidate_pairs(record.get("pairs"), length)
        artifact = record.get("pair_artifact")
        if not artifact or artifact.get("format") != "bpp_1based_tsv":
            raise ValueError("completed target lacks raw BPP pair artifact metadata")
        path = Path(artifact["path"])
        if sha256_file(path) != artifact.get("sha256") or artifact.get("row_count") != len(pairs):
            raise ValueError("pair artifact identity/count mismatch")
        if parse_bpp_artifact(path, length) != pairs:
            raise ValueError("in-memory pairs are not the complete raw artifact rows")
        validate_resources(record.get("resources", {}), protocol.get("device"))
        reference = [tuple(pair) for pair in target["pairs"]]
        recall = pair_recall(reference, pairs)
        all_reference.extend((record["name"], *pair) for pair in reference)
        all_candidate.extend((record["name"], *item["pair"]) for item in pairs)
        completed_count += 1
        completed_nucleotides += length
        per_target.append({"name": record["name"], "status": status, "pair_rows": len(pairs),
                           "recall": recall, "resources": record["resources"]})
    aggregate = named_pair_recall(all_reference, all_candidate)
    complete = completed_count == len(targets)
    return {
        "status": "PASS_FULL_ACCEPTANCE" if complete else "INCOMPLETE_EXPLICIT_FAILURES",
        "execution_column": column,
        "protocol": protocol,
        "coverage": {"expected_targets": len(targets), "records": len(records),
                     "record_coverage": 1.0, "completed_targets": completed_count,
                     "completed_target_coverage": completed_count / len(targets),
                     "completed_nucleotide_coverage": completed_nucleotides / sum(t["length"] for t in targets)},
        "micro_pair_recall": aggregate,
        "per_target": per_target,
        "long_range_definition": "span=j-i on zero-based inclusive coordinates; micro counts pool reference pairs; empty buckets are null",
    }


def freeze_targets(repo: Path) -> dict:
    manifest_path = repo / "source_data/round53/rinalmo_full_d1_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    targets = []
    for split, key, hash_key in (("train", "train_d1", "train_d1_sha256"),
                                  ("holdout", "holdout", "holdout_sha256")):
        path = Path(manifest[key])
        if sha256_file(path) != manifest[hash_key]:
            raise AssertionError("frozen JSONL hash mismatch")
        found = {}
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                row = json.loads(line)
                length = len(row["seq"])
                if length not in EXPECTED_IDENTITIES:
                    continue
                name = row["name"]
                if name in found:
                    raise AssertionError("duplicate long target")
                sequence = normalise_sequence(row["seq"])
                if len(sequence) != length:
                    raise AssertionError("normalisation changed length")
                source = row["source"]
                label_path = Path(source["source_json"])
                if sha256_file(label_path) != source["source_json_sha256"]:
                    raise AssertionError("original label JSON hash mismatch")
                document = json.loads(label_path.read_text())
                chains = [chain for chain in document["chains"] if chain["id"] == name]
                if len(chains) != 1:
                    raise AssertionError("original chain identity is not unique")
                chain = chains[0]
                converted_pairs = [[left - 1, right - 1] for left, right in source["canonical_nested_pairs"]]
                if normalise_sequence(chain["sequence"]) != sequence or source["sequence"] != sequence \
                        or chain["raw_pairs"] != row["raw_pairs"] or converted_pairs != row["pairs"]:
                    raise AssertionError("original chain to frozen row identity mismatch")
                found[name] = {
                    "name": name, "split": split, "row_index_zero_based": row_index,
                    "length": length, "sequence_sha256": sha256_text(sequence),
                    "pairs": row["pairs"], "pairs_sha256": sha256_json(row["pairs"]),
                    "raw_pairs_sha256": sha256_json(row["raw_pairs"]),
                    "canonical_pair_count": len(row["pairs"]), "raw_pair_count": len(row["raw_pairs"]),
                    "frozen_jsonl": file_record(path),
                    "scientific_role": "HOLDOUT_EVALUATION" if split == "holdout" else "TRAIN_ONLY_NOT_UNBIASED_EVALUATION",
                    "pdb_id": chain["pdb_id"], "source_label_json": file_record(label_path),
                    "original_chain_identity_verified": True,
                }
        expected_names = EXPECTED_IDENTITIES[7224 if split == "train" else 19000]
        if set(found) != expected_names:
            raise AssertionError(f"long target identity mismatch: {sorted(found)}")
        targets.extend(found[name] for name in sorted(found))
    return {
        "schema_version": "longseq_target_identity_v1.0",
        "status": "PASS_EXACT_FROZEN_INPUT_IDENTITY",
        "source_manifest": file_record(manifest_path),
        "coordinate_convention": "reference pairs zero-based inclusive; raw DSSR pairs remain one-based inclusive",
        "sequence_policy": "uppercase and T->U only; no truncation/windowing/deletion",
        "targets": targets,
    }


def static_audit(repo: Path, recovery_report_path: Path) -> dict:
    frozen = freeze_targets(repo)
    recovery = json.loads(recovery_report_path.read_text())
    if recovery["status"] != "PASS_OFFICIAL_SOURCE_BUILD_AND_SHORT_CPU_SMOKE" \
            or recovery["long_chain_inference_executed"] is not False:
        raise AssertionError("official recovery gate did not pass")
    public_targets = [{key: value for key, value in target.items() if key != "pairs"}
                      for target in frozen["targets"]]
    return {
        "schema_version": "longseq_static_acceptance_v1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_STATIC_CONTRACT__LONG_SEQUENCE_EVALUATION_NOT_RUN",
        "full_ready": False,
        "ready": ["official source pinned and hashed", "C/V CPU binaries compiled", "official 76nt C/V smoke with full BPP and MEA",
                  "exact 7224/19000nt input identities", "strict raw-pair/coverage/long-range/resource validator"],
        "not_ready": ["no 7224/19000nt LinearPartition prediction", "coverage/recall not measured on long targets",
                      "memory/latency/FLOPs not measured on long targets", "no exact global DP long-sequence implementation validated"],
        "targets": public_targets,
        "recovery_report": file_record(recovery_report_path),
        "execution_columns": {
            "EXACT_GLOBAL_DP": "requires exact_global_dp=true, zero cutoff and no beam/window/top-k/sparse pruning; no implementation qualified here",
            "APPROXIMATE_SPARSE": "LinearPartition-C/V beam=100, cutoff=0.0; all emitted rows retained; absent pairs are not exact zeros",
        },
        "metrics": {"coverage": "NOT_MEASURED", "pair_recall": "NOT_MEASURED",
                    "long_range_recall_buckets": {bucket: "NOT_MEASURED" for bucket in SPAN_BUCKETS},
                    "memory": "NOT_MEASURED", "latency": "NOT_MEASURED", "FLOPs": "NOT_MEASURED"},
        "resource_protocol": {
            "latency": "wall seconds for each complete full-length execution; warmup and repeats preregistered before timing",
            "memory": "peak CPU RSS from /usr/bin/time -v or equivalent; GPU allocated/reserved only for actual CUDA execution, null for CPU",
            "FLOPs": "hardware performance counters or instrumented operation counts for the exact execution; never scale short-chain estimates",
            "device": "LinearPartition acceptance can run serially on CPU; no GPU task is needed or authorized here",
        },
        "gec": {"long_sequence_support": "NOT_ESTABLISHED", "exact_global_cubic_support": "NOT_ESTABLISHED",
                "claim": "GEC is not claimed to support 7224 or 19000nt"},
        "long_chain_inference_executed": False,
        "gpu_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--recovery-report", type=Path, required=True)
    parser.add_argument("--target-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    frozen = freeze_targets(args.repo)
    args.target_output.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    audit = static_audit(args.repo, args.recovery_report)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": audit["status"], "full_ready": audit["full_ready"],
                      "targets": [{key: target[key] for key in
                                   ("name", "split", "length", "sequence_sha256", "canonical_pair_count", "raw_pair_count")}
                                  for target in audit["targets"]]}, indent=2))


if __name__ == "__main__":
    main()
