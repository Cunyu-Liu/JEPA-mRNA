#!/usr/bin/env python3
"""Hash-bound, preregistered exposure sensitivity for the four-arm margin evaluation."""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tools import gec_seed0_independent_reeval_v1 as independent

SCHEMA = "foldnar.structured_margin_exposure_sensitivity.v1"
SOURCE_COMMIT = "4247d1e4a971ceb8c63e3f8338e026a23c18941b"
DEFAULT_EXPOSURE_ROOT = Path("/home/rna_jev_pretrain_exposure/manifests/pretrain_exposure_v1")
STATES = ("EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN")
MIN_SEQUENCES = 30
MIN_TRUTH_OBJECTS = 30
EXPOSURE_CONTRACT = {
    "source_commit": SOURCE_COMMIT,
    "artifact_sha256": {
        "sequence_registry.jsonl.gz": "5c0d95a31abe849a76975f06216835a3707129d139beb92569d112b645503301",
        "summary.json": "af39cd32a42344340f7d083b606e9d79208414d15b0e47d0690c40bfb28f4866",
        "asset_registry.json": "37cdabc5b5132fc29224b1a3890fd25e642999f5fc87e9c7b96613a509006e4a",
        "independent_verification.json": "a8f43054123e9ce1a02cf436f223267d4d1f7167e7227d78088c20f8786e5484",
    },
    "dev_counts": {"EXACT": 7978, "HOMOLOGY": 539, "NOT_DETECTED_UNKNOWN": 626},
    "bprna_new": {"rows": 5388, "exact": 5381, "homology": 7},
}
STATE_SEMANTICS = {
    "EXACT": "normalized full sequence detected in a hash-bound known public source",
    "HOMOLOGY": "no exact hit; existing MMseqs2 80% identity and bidirectional-coverage union detected a known-source hit",
    "NOT_DETECTED_UNKNOWN": "no match in the declared known-source search; not clean, independent or pretraining-unseen",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_artifacts(root: Path, contract: dict) -> dict:
    artifacts = {}
    for name, expected_sha256 in contract["artifact_sha256"].items():
        path = root / name
        actual = independent.identity(path)
        if actual["sha256"] != expected_sha256:
            raise ValueError(f"exposure artifact hash mismatch: {name}")
        artifacts[name] = actual
    return artifacts


def bind_exposure_registry(frozen: list, root: Path = DEFAULT_EXPOSURE_ROOT,
                           contract: dict = EXPOSURE_CONTRACT) -> tuple[dict[str, str], dict]:
    root = Path(root)
    artifacts = _check_artifacts(root, contract)
    summary = _read_json(root / "summary.json")
    assets = _read_json(root / "asset_registry.json")
    verification = _read_json(root / "independent_verification.json")
    if summary.get("schema_version") != "rinalmo.pretrain_exposure.summary.v1":
        raise ValueError("unexpected exposure summary schema")
    if assets.get("schema_version") != "rinalmo.pretrain_exposure.assets.v1":
        raise ValueError("unexpected exposure asset schema")
    if verification.get("schema_version") != "rinalmo.pretrain_exposure.independent_verification.v1":
        raise ValueError("unexpected exposure verification schema")
    if (summary.get("complete_official_pretraining_coverage_state") != "UNCHECKED"
            or assets["official_pretraining_declaration"].get("complete_official_training_set_state") != "UNCHECKED"
            or verification.get("official_complete_coverage") != "UNCHECKED"):
        raise ValueError("complete official pretraining corpus must remain UNCHECKED")
    if verification.get("status") != "PASS" or verification.get("no_model_inference_or_training") is not True:
        raise ValueError("exposure registry independent verification is not eligible")
    for name, expected in contract["artifact_sha256"].items():
        if name == "independent_verification.json":
            continue
        if verification["artifact_sha256"].get(name) != expected:
            raise ValueError(f"independent verification artifact hash mismatch: {name}")
    dev_summary = summary["populations"].get("tr1c_cluster80_dev")
    if not isinstance(dev_summary, dict):
        raise ValueError("cluster dev exposure summary is missing")
    expected_counts = contract["dev_counts"]
    if (dev_summary.get("n_input_rows") != sum(expected_counts.values())
            or dev_summary.get("known_source_exact_or_homology_state") != {
                **{state: 0 for state in (*STATES, "UNCHECKED")}, **expected_counts,
            }):
        raise ValueError("cluster dev exposure counts differ from the pinned contract")
    bprna_new = summary["populations"].get("bprna_new")
    expected_new = contract["bprna_new"]
    if (not isinstance(bprna_new, dict)
            or bprna_new.get("n_input_rows") != expected_new["rows"]
            or bprna_new["known_source_exact_state"].get("EXACT") != expected_new["exact"]
            or bprna_new["known_source_exact_or_homology_state"].get("HOMOLOGY") != expected_new["homology"]):
        raise ValueError("bpRNA-new exposure counts differ from the pinned contract")

    expected = {record.sequence_id: record for record in frozen}
    assignments = {}
    observed_counts = Counter()
    with gzip.open(root / "sequence_registry.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("population_id") != "tr1c_cluster80_dev":
                continue
            sequence_id = row.get("query_id")
            if sequence_id in assignments or sequence_id not in expected:
                raise ValueError(f"duplicate or out-of-panel exposure row: {sequence_id}")
            record = expected[sequence_id]
            if (row.get("sequence_sha256") != record.sequence_sha256
                    or row.get("normalized_length") != record.length
                    or row.get("dataset_id") != "bprna_tr1c"
                    or row.get("role") != "development" or row.get("selection_use") is not True):
                raise ValueError(f"exposure sequence identity mismatch: {sequence_id}")
            state = row.get("known_source_exact_or_homology_state")
            if state not in STATES:
                raise ValueError(f"unsupported or unchecked exposure state: {sequence_id}")
            exact_state = row.get("known_source_exact_state")
            complete_state = row.get("complete_official_pretraining_exposure_state")
            matches = row.get("source_matches")
            if state == "EXACT":
                valid = exact_state == "EXACT" and complete_state == "EXACT" and bool(matches)
            else:
                expected_exact = "NOT_DETECTED_UNKNOWN"
                valid = exact_state == expected_exact and complete_state == "UNCHECKED" and not matches
            if not valid or not isinstance(row.get("homology_query_id"), str) or not row["homology_query_id"]:
                raise ValueError(f"inconsistent exposure evidence: {sequence_id}")
            assignments[sequence_id] = state
            observed_counts[state] += 1
    if set(assignments) != set(expected):
        raise ValueError("exposure registry does not exactly cover the frozen dev panel")
    if dict(observed_counts) != expected_counts:
        raise ValueError("recomputed cluster dev exposure counts differ from the pinned contract")
    provenance = {
        "source_commit": contract["source_commit"],
        "registry_schema": "rinalmo.pretrain_exposure.sequence_registry.v1",
        "state_field": "known_source_exact_or_homology_state",
        "state_semantics": STATE_SEMANTICS,
        "complete_official_pretraining_coverage_state": "UNCHECKED",
        "official_pretraining_clean_claim_allowed": False,
        "not_detected_unknown_is_clean": False,
        "artifacts": artifacts,
        "assignment_sha256": independent.canonical_hash(assignments),
        "cluster_dev_counts": dict(observed_counts),
        "bprna_new_anchor": expected_new,
    }
    return assignments, provenance


def _support(records: list, truth_objects: list, label: str, target: bool) -> dict:
    n_with_truth = sum(bool(objects) for objects in truth_objects)
    n_truth = sum(len(objects) for objects in truth_objects)
    n_sequences = n_with_truth if target else len(records)
    if n_sequences < MIN_SEQUENCES or n_truth < MIN_TRUTH_OBJECTS:
        raise ValueError(
            f"insufficient support for {label}: sequences={n_sequences}, truth_objects={n_truth}; "
            f"required={MIN_SEQUENCES}/{MIN_TRUTH_OBJECTS}"
        )
    return {
        "n_sequences": n_sequences,
        "n_sequences_with_truth": n_with_truth,
        "n_truth_objects": n_truth,
        "min_sequences": MIN_SEQUENCES,
        "min_truth_objects": MIN_TRUTH_OBJECTS,
        "support_pass": True,
    }


def _observed_metrics(counts: np.ndarray) -> dict:
    totals = counts.sum(axis=0)
    precision, recall, f1 = independent.prf(totals)
    return {
        "tp": int(totals[0]), "fp": int(totals[1]), "fn": int(totals[2]),
        "precision": precision, "recall": recall, "f1": f1,
    }


def _comparison(distributions: dict, control: str) -> dict:
    result = {}
    for statistic in ("precision", "recall", "f1"):
        ensemble = distributions["ensemble_context"][statistic]
        reference = distributions[control][statistic]
        result[statistic] = {
            "ensemble_context": ensemble["observed"],
            control: reference["observed"],
            "delta_ensemble_minus_control": ensemble["observed"] - reference["observed"],
            "paired_ci95": independent.ci95(ensemble["sequence"] - reference["sequence"]),
            "family_cluster_sensitivity_ci95": independent.ci95(ensemble["family"] - reference["family"]),
        }
    return result


def family_concentration(records: list, truth_objects: list) -> dict:
    family_weights = Counter()
    family_sequences = Counter()
    unmapped = 0
    total = 0
    n_sequences = 0
    for record, objects in zip(records, truth_objects):
        weight = len(objects)
        if not weight:
            continue
        n_sequences += 1
        total += weight
        if record.family_id is None:
            unmapped += weight
        else:
            family_weights[record.family_id] += weight
            family_sequences[record.family_id] += 1
    known_total = sum(family_weights.values())
    top_family, top_weight = family_weights.most_common(1)[0] if family_weights else (None, 0)
    return {
        "role": "DESCRIPTIVE_FAMILY_CONCENTRATION_ONLY_NEVER_GATE",
        "status": "OK" if known_total else "NO_KNOWN_FAMILY_MAPPING",
        "n_sequences_with_truth": n_sequences,
        "n_truth_objects": total,
        "n_known_families": len(family_weights),
        "top_known_family": top_family,
        "top_known_family_fraction_of_all": top_weight / total if known_total else None,
        "known_family_hhi": (
            sum((weight / known_total) ** 2 for weight in family_weights.values()) if known_total else None
        ),
        "unmapped_or_ambiguous_family_fraction": unmapped / total if total else None,
        "known_family_sequence_counts": dict(sorted(family_sequences.items())),
        "known_family_truth_weights": dict(sorted(family_weights.items())),
    }


def evaluate_exposure_sensitivity(frozen: list, arms: dict, assignments: dict[str, str],
                                  provenance: dict, resamples: int = independent.BOOTSTRAP_RESAMPLES,
                                  seed: int = independent.BOOTSTRAP_SEED) -> dict:
    if set(arms) != set(independent.ARMS):
        raise ValueError("exposure sensitivity requires exact four-arm coverage")
    if set(assignments) != {record.sequence_id for record in frozen}:
        raise ValueError("exposure assignments do not exactly cover frozen records")
    if resamples < 20:
        raise ValueError("insufficient paired bootstrap resamples")
    strata = {}
    for state in STATES:
        indices = [index for index, record in enumerate(frozen)
                   if assignments.get(record.sequence_id) == state]
        records = [frozen[index] for index in indices]
        state_arms = {arm: [arms[arm][index] for index in indices] for arm in independent.ARMS}
        for arm, rows in state_arms.items():
            if any(row.sequence_id != record.sequence_id for row, record in zip(rows, records)):
                raise ValueError(f"{state}/{arm}: unpaired prediction ordering")
        arrays = {arm: independent.make_arm_arrays(rows, state_arms["matched_control"])
                  for arm, rows in state_arms.items()}
        truth = {
            "pair": [record.true_pairs for record in records],
            "stem": [independent.stems(record.true_pairs) for record in records],
        }
        support = {kind: _support(records, objects, f"{state}/{kind}", False)
                   for kind, objects in truth.items()}
        target_truth = {target: [independent.selected_pairs(record.true_pairs, target) for record in records]
                        for target in independent.TARGETS}
        target_support = {target: _support(records, objects, f"{state}/{target}", True)
                          for target, objects in target_truth.items()}
        bootstrap = independent.PairedBootstrap(records, resamples=resamples, seed=seed)
        per_arm = {arm: {"pair": _observed_metrics(arrays[arm]["pair_counts"]),
                         "stem": _observed_metrics(arrays[arm]["stem_counts"]),
                         "targets": {target: _observed_metrics(arrays[arm]["subgroup_counts"][target])
                                     for target in independent.TARGETS}}
                   for arm in independent.ARMS}
        distributions = {}
        for arm in independent.ARMS:
            distributions[arm] = {}
            for kind in ("pair", "stem"):
                distributions[arm][kind] = {
                    statistic: independent.metric_distribution(
                        arrays[arm], bootstrap, f"micro_{kind}_{statistic}"
                    ) for statistic in ("precision", "recall", "f1")
                }
            distributions[arm]["targets"] = {
                target: {statistic: independent.metric_distribution(
                    arrays[arm], bootstrap, f"micro_pair_{statistic}", target
                ) for statistic in ("precision", "recall", "f1")}
                for target in independent.TARGETS
            }
        comparisons = {}
        for control in independent.CONTROLS:
            comparisons[control] = {
                "pair": _comparison({arm: distributions[arm]["pair"] for arm in independent.ARMS}, control),
                "stem": _comparison({arm: distributions[arm]["stem"] for arm in independent.ARMS}, control),
                "targets": {
                    target: _comparison(
                        {arm: distributions[arm]["targets"][target] for arm in independent.ARMS}, control
                    ) for target in independent.TARGETS
                },
            }
        strata[state] = {
            "support": {**support, "targets": target_support},
            "per_arm": per_arm,
            "ensemble_vs_controls": comparisons,
            "family_concentration": {
                "pair": family_concentration(records, truth["pair"]),
                "stem": family_concentration(records, truth["stem"]),
                "targets": {target: family_concentration(records, objects)
                            for target, objects in target_truth.items()},
            },
        }
    return {
        "schema_version": SCHEMA,
        "status": "COMPLETE_PREREGISTERED_SENSITIVITY",
        "role": "PREREGISTERED_SENSITIVITY_ONLY",
        "gate_use": "FORBIDDEN",
        "checkpoint_selection_use": "FORBIDDEN",
        "checkpoint_or_arm_selection_performed": False,
        "registry": provenance,
        "support_contract": {
            "min_sequences_per_exposure_metric": MIN_SEQUENCES,
            "min_truth_objects_per_exposure_metric": MIN_TRUTH_OBJECTS,
            "target_sequence_rule": "at least one selected truth pair",
            "failure_policy": "FAIL_CLOSED_NO_METRIC_OR_CI_EMITTED",
        },
        "ci_contract": {
            "unit": "paired sequences on identical frozen IDs within each exposure stratum",
            "resamples": resamples, "seed": seed, "confidence": 0.95,
            "family_cluster_sensitivity": "registered family; unmapped sequences are singleton clusters; sensitivity only",
        },
        "strata": strata,
    }
