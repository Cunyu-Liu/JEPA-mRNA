#!/usr/bin/env python3
"""Independently recompute frozen context-crossover metrics from raw TR1c truth.

This verifier does not import the official evaluator, its aggregation routines,
or the crossover implementation. It consumes raw dev truth and sequence-level
S0/S1 predictions and validates native predictions against historical exports.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
SPAN_NAMES = ("le3", "4_7", "8_15", "16_31", "32_63", "64_127", "ge128")
SUBGROUPS = ("isolated_pair:isolated",) + tuple(f"pair_span:{name}" for name in SPAN_NAMES)
SCHEMA = "foldnar.gec_frozen_context_crossover.v1"
VERIFICATION_SCHEMA = "foldnar.gec_frozen_context_crossover.independent_verification.v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_identity(document: dict, label: str) -> dict:
    path = Path(document["path"])
    if not path.is_file() or path.stat().st_size != document["bytes"] or file_sha256(path) != document["sha256"]:
        raise ValueError(f"artifact identity mismatch: {label}")
    return document


def jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(value: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checked_pairs(raw, length: int) -> frozenset[tuple[int, int]]:
    result = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("malformed pair")
        i, j = pair
        if (type(i) is not int or type(j) is not int or not 0 <= i < j < length):
            raise ValueError("invalid pair coordinates")
        result.append((i, j))
    flat = [position for pair in result for position in pair]
    if len(set(result)) != len(result) or len(set(flat)) != len(flat):
        raise ValueError("duplicate pair or paired endpoint")
    return frozenset(result)


def load_raw_panel(config_path: Path, repo: Path) -> tuple[list[dict], dict]:
    config = json.loads(config_path.read_text())
    registry_path = Path(config["assets"]["registry"])
    if not registry_path.is_absolute():
        registry_path = repo / registry_path
    registry = json.loads(registry_path.read_text())
    panel = next(row for row in registry["panels"] if row["panel_id"] == "tr1c_cluster80_dev_v1")
    dev_path = Path(panel["dev_ids"]["path"])
    if file_sha256(dev_path) != panel["dev_ids"]["sha256"]:
        raise ValueError("dev ID hash mismatch")
    dev_ids = dev_path.read_text().splitlines()
    if len(dev_ids) != 9143 or len(set(dev_ids)) != len(dev_ids):
        raise ValueError("full dev IDs are not 9143 unique rows")
    panel_path = Path(registry["environment_contracts"]["panel_records"]["path"])
    panel_rows = {row["sequence_id"]: row for row in jsonl(panel_path)}
    raw_path = Path(config["assets"]["data"])
    if file_sha256(raw_path) != config["assets"]["data_sha256"]:
        raise ValueError("raw TR1c hash mismatch")
    selected = set(dev_ids)
    records = []
    with raw_path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            sequence_id = f"bprna_tr1c:row{row_index:06d}:{raw['name']}"
            if sequence_id not in selected:
                continue
            sequence = str(raw["seq"]).upper().replace("T", "U")
            truth = checked_pairs(raw.get("pairs", []), len(sequence))
            metadata = panel_rows[sequence_id]
            digest = hashlib.sha256(sequence.encode()).hexdigest()
            if (metadata["sequence_sha256"] != digest or metadata["length"] != len(sequence)
                    or metadata["n_pairs"] != len(truth) or metadata["cluster_assignment"] != "dev"):
                raise ValueError(f"raw/panel mismatch: {sequence_id}")
            records.append({"id": sequence_id, "sequence": sequence, "sequence_sha256": digest,
                            "truth": truth, "family_id": metadata.get("family_id")})
    if len(records) != len(selected) or {row["id"] for row in records} != selected:
        raise ValueError("raw records do not exactly cover dev IDs")
    records.sort(key=lambda row: row["id"])
    identities = {"config": config_path, "registry": registry_path, "raw_tr1c": raw_path,
                  "dev_ids": dev_path, "panel_records": panel_path}
    return records, identities


def family_list(records: list[dict]) -> list[str]:
    sequences = defaultdict(int)
    pairs = defaultdict(int)
    for row in records:
        family = row["family_id"]
        if family is not None:
            sequences[family] += 1
            pairs[family] += len(row["truth"])
    answer = sorted(name for name in sequences if sequences[name] >= 30 and pairs[name] >= 500)
    if not answer:
        raise ValueError("no eligible macro family")
    return answer


def pair_span(pair: tuple[int, int]) -> str:
    distance = pair[1] - pair[0]
    if distance <= 3:
        return "le3"
    if distance <= 7:
        return "4_7"
    if distance <= 15:
        return "8_15"
    if distance <= 31:
        return "16_31"
    if distance <= 63:
        return "32_63"
    if distance <= 127:
        return "64_127"
    return "ge128"


def stem_set(pairs) -> set[frozenset[tuple[int, int]]]:
    unseen = set(pairs)
    components = set()
    while unseen:
        pending = [unseen.pop()]
        component = set(pending)
        while pending:
            i, j = pending.pop()
            for adjacent in ((i - 1, j + 1), (i + 1, j - 1)):
                if adjacent in unseen:
                    unseen.remove(adjacent)
                    component.add(adjacent)
                    pending.append(adjacent)
        components.add(frozenset(component))
    return components


def lone_pairs(pairs) -> set[tuple[int, int]]:
    return {(i, j) for i, j in pairs
            if (i - 1, j + 1) not in pairs and (i + 1, j - 1) not in pairs}


def tp_fp_fn(truth, predicted) -> tuple[int, int, int]:
    return len(truth & predicted), len(predicted - truth), len(truth - predicted)


def scores(counts, empty=0.0) -> tuple[float, float, float]:
    tp, fp, fn = counts
    p = tp / (tp + fp) if tp + fp else empty
    r = tp / (tp + fn) if tp + fn else empty
    f = 2 * p * r / (p + r) if p + r else empty
    return float(p), float(r), float(f)


def recompute(records: list[dict], predictions: dict[str, frozenset], eligible: list[str],
              macro_enabled: bool = True) -> dict:
    pair_total = np.zeros(3, dtype=np.int64)
    stem_total = np.zeros(3, dtype=np.int64)
    pair_by_sequence, stem_by_sequence = [], []
    subgroup_counts = {name: np.zeros(3, dtype=np.int64) for name in SUBGROUPS}
    for row in records:
        truth = row["truth"]
        predicted = predictions[row["id"]]
        pair_counts = tp_fp_fn(truth, predicted)
        stem_counts = tp_fp_fn(stem_set(truth), stem_set(predicted))
        pair_total += pair_counts
        stem_total += stem_counts
        pair_by_sequence.append(scores(pair_counts, empty=1.0))
        stem_by_sequence.append(scores(stem_counts, empty=1.0))
        selected = {"isolated_pair:isolated": (lone_pairs(truth), lone_pairs(predicted))}
        for name in SPAN_NAMES:
            selected[f"pair_span:{name}"] = (
                {pair for pair in truth if pair_span(pair) == name},
                {pair for pair in predicted if pair_span(pair) == name},
            )
        for name, (truth_subset, predicted_subset) in selected.items():
            subgroup_counts[name] += tp_fp_fn(truth & truth_subset, predicted & predicted_subset)

    def macro(values):
        family_means = []
        for family in eligible:
            indexes = [index for index, row in enumerate(records) if row["family_id"] == family]
            family_means.append(np.mean(np.asarray(values)[indexes], axis=0))
        return tuple(float(value) for value in np.mean(family_means, axis=0))

    result = {}
    for kind, total, values, macro_values in (
        ("pair", pair_total, pair_by_sequence, macro(pair_by_sequence) if macro_enabled else None),
        ("stem", stem_total, stem_by_sequence, macro(stem_by_sequence) if macro_enabled else None),
    ):
        for scope, observed in (("micro", scores(total)),
                                ("macro", macro_values if macro_enabled else (None, None, None))):
            for index, statistic in enumerate(("precision", "recall", "f1")):
                result[f"{scope}_{kind}_{statistic}"] = observed[index]
        result[f"{kind}_confusion_tp_fp_fn"] = total.tolist()
    for name, counts in subgroup_counts.items():
        observed = scores(counts)
        result[name] = {"micro_precision": observed[0], "micro_recall": observed[1],
                        "micro_f1": observed[2], "confusion_tp_fp_fn": counts.tolist()}
    return result


def subtract(left: dict, right: dict) -> dict:
    answer = {}
    for scope in ("micro", "macro"):
        for kind in ("pair", "stem"):
            for statistic in ("precision", "recall", "f1"):
                name = f"{scope}_{kind}_{statistic}"
                answer[name] = left[name] - right[name]
    answer["subgroups"] = {
        name: {key: left[name][key] - right[name][key]
               for key in ("micro_precision", "micro_recall", "micro_f1")}
        for name in SUBGROUPS
    }
    return answer


def compare_values(actual, expected, label: str, errors: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            errors.append(f"{label}: key mismatch")
            return
        for key in expected:
            compare_values(actual[key], expected[key], f"{label}.{key}", errors)
    elif isinstance(expected, list):
        if actual != expected:
            errors.append(f"{label}: list mismatch")
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
            errors.append(f"{label}: {actual} != {expected}")
    elif actual != expected:
        errors.append(f"{label}: {actual} != {expected}")


def verify(run_dir: Path, source_root: Path, config_path: Path) -> dict:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary.get("schema_version") != SCHEMA or summary.get("status") != "COMPLETE"
            or summary.get("gec_pass_claim") is not False or summary.get("sota_claim") is not False
            or summary.get("seeds_1_2_authorized") is not False
            or summary["environment"].get("device") != "cuda:0"
            or summary["environment"].get("cpu_fallback") is not False
            or summary["environment"].get("processes") != 1):
        raise ValueError("invalid or ineligible crossover summary")
    repo = Path(summary["repository"])
    all_records, data_paths = load_raw_panel(config_path, repo)
    eligible = family_list(all_records)
    records = (sorted((row for row in all_records if row["truth"] and len(row["sequence"]) >= 8),
                      key=lambda row: (len(row["sequence"]), row["id"]))[:2]
               if summary["mode"] == "smoke" else all_records)
    if len(records) != summary["record_count"]:
        raise ValueError("diagnostic subset size mismatch")
    if eligible != summary["macro_eligible_families"]:
        raise ValueError("macro eligible family mismatch")
    by_id = {row["id"]: row for row in records}
    all_by_id = {row["id"]: row for row in all_records}
    expected_ids = set(by_id)
    errors = []
    recomputed = {}
    s0_by_model = {}
    native_validation = {}
    for model in ARMS:
        for name in ("best_checkpoint", "prediction_export", "native_predictions"):
            check_identity(summary["input_identities"][model][name], f"{model}.{name}")
        export = json.loads((source_root / model / "prediction_export.json").read_text())
        if export["best_checkpoint_sha256"] != summary["input_identities"][model]["best_checkpoint"]["sha256"]:
            raise ValueError(f"native export checkpoint mismatch: {model}")
        native_rows = jsonl(source_root / model / "dev_predictions.jsonl")
        native_predictions = {}
        for row in native_rows:
            sequence_id = row["sequence_id"]
            authoritative = all_by_id.get(sequence_id)
            if authoritative is None or row["sequence_sha256"] != authoritative["sequence_sha256"]:
                raise ValueError(f"historical sequence identity mismatch: {model}")
            if checked_pairs(row["true_pairs"], len(authoritative["sequence"])) != authoritative["truth"]:
                raise ValueError(f"historical truth mismatch: {model} {sequence_id}")
            native_predictions[sequence_id] = checked_pairs(row["predicted_pairs"], len(authoritative["sequence"]))
        if set(native_predictions) != set(all_by_id):
            raise ValueError(f"historical native coverage mismatch: {model}")
        native_count = 0
        for context in ARMS:
            key = f"{model}__{context}"
            path = Path(summary["artifacts"][key]["path"])
            check_identity(summary["artifacts"][key], key)
            rows = jsonl(path)
            if len(rows) != len(records) or {row["sequence_id"] for row in rows} != expected_ids:
                raise ValueError(f"prediction coverage mismatch: {key}")
            s1_predictions = {}
            current_s0 = {}
            seen = set()
            for row in rows:
                sequence_id = row["sequence_id"]
                if sequence_id in seen:
                    raise ValueError(f"duplicate prediction: {key} {sequence_id}")
                seen.add(sequence_id)
                authoritative = by_id[sequence_id]
                length = len(authoritative["sequence"])
                if (row["model_arm"] != model or row["context_arm"] != context
                        or row["sequence_sha256"] != authoritative["sequence_sha256"]
                        or row["sequence_length"] != length
                        or row["checkpoint_sha256"] != export["best_checkpoint_sha256"]
                        or checked_pairs(row["true_pairs"], length) != authoritative["truth"]):
                    raise ValueError(f"row identity mismatch: {key} {sequence_id}")
                s0 = checked_pairs(row["s0_predicted_pairs"], length)
                s1 = checked_pairs(row["s1_predicted_pairs"], length)
                current_s0[sequence_id] = s0
                s1_predictions[sequence_id] = s1
                if context == model:
                    if row["native_context"] is not True or row["native_prediction_exact_match"] is not True:
                        errors.append(f"{key}.{sequence_id}: native flag mismatch")
                    if s1 == native_predictions[sequence_id]:
                        native_count += 1
                    else:
                        errors.append(f"{key}.{sequence_id}: native decoded pairs differ")
            if model in s0_by_model and current_s0 != s0_by_model[model]:
                errors.append(f"{model}.{context}: shared S0 differs across contexts")
            s0_by_model[model] = current_s0
            recomputed[key] = recompute(records, s1_predictions, eligible, summary["mode"] == "full")
            compare_values(recomputed[key], summary["metrics"][key], f"metrics.{key}", errors)
        s0_key = f"{model}__matched_control_s0"
        recomputed[s0_key] = recompute(records, s0_by_model[model], eligible, summary["mode"] == "full")
        compare_values(recomputed[s0_key], summary["metrics"][s0_key], f"metrics.{s0_key}", errors)
        native_validation[model] = {"matched_records": native_count, "record_count": len(records),
                                    "pass": native_count == len(records)}
        compare_values(native_validation[model], summary["native_prediction_validation"][model],
                       f"native.{model}", errors)
        evidence = summary["model_evidence"][model]
        if (evidence["model_tensor_sha256_before"] != evidence["model_tensor_sha256_after"]
                or evidence["all_parameters_requires_grad_false"] is not True
                or evidence["all_modules_eval"] is not True):
            errors.append(f"{model}: frozen parameter evidence failed")
    comparisons = {}
    for model in (ARMS if summary["mode"] == "full" else ()):
        native = recomputed[f"{model}__{model}"]
        s0 = recomputed[f"{model}__matched_control_s0"]
        matched = recomputed[f"{model}__matched_control"]
        comparisons[model] = {
            "native_s1_minus_shared_s0": subtract(native, s0),
            "each_context_s1_minus_shared_s0": {
                context: subtract(recomputed[f"{model}__{context}"], s0) for context in ARMS
            },
            "each_context_s1_minus_matched_s1": {
                context: subtract(recomputed[f"{model}__{context}"], matched) for context in ARMS
            },
        }
    compare_values(comparisons, summary["comparisons"], "comparisons", errors)
    for name, document in summary["code_manifest"].items():
        check_identity(document, f"code.{name}")
    for name, document in summary["data_identities"].items():
        check_identity(document, f"data.{name}")
    result = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "run_dir": str(run_dir), "mode": summary["mode"], "record_count": len(records),
        "independent_raw_truth_recomputation": True,
        "official_evaluator_imported": False, "official_aggregate_read": False,
        "native_prediction_validation": native_validation,
        "metrics": recomputed, "comparisons": comparisons, "errors": errors,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": "single_seed0_cluster_dev_frozen_causal_diagnostic_only",
        "gec_pass_claim": False, "sota_claim": False, "seeds_1_2_authorized": False,
    }
    if errors:
        raise ValueError("independent verification failed: " + "; ".join(errors[:8]))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path,
                        default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3"))
    parser.add_argument("--gec-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(args.run_dir, args.source_root, args.gec_config)
    atomic_json(result, args.output)
    print(json.dumps({"status": result["status"], "record_count": result["record_count"],
                      "native_prediction_validation": result["native_prediction_validation"],
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
