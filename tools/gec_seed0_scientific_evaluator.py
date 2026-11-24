#!/usr/bin/env python3
"""Read-only four-arm GEC seed0 evaluation against the frozen model-dev contract.

Each ARM.jsonl row must contain sequence_id, true_pairs and predicted_pairs.
Pairs are zero-based [i, j], i < j. Optional raw_scores are [i, j, score]
triples on a candidate set shared by all arms. Inputs are never modified.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ARMS = (
    "matched_control",
    "ensemble_context",
    "destroy_interval_shuffle",
    "independent_context",
)
TARGETS = ("isolated_pair:isolated",) + tuple(
    f"pair_span:{name}" for name in ("16_31", "32_63", "64_127", "ge128")
)
SPANS = ("le3", "4_7", "8_15", "16_31", "32_63", "64_127", "ge128")
GLOBAL_METRICS = (
    "micro_pair_precision", "micro_pair_recall", "micro_pair_f1",
    "macro_pair_precision", "macro_pair_recall", "macro_pair_f1",
    "micro_stem_precision", "micro_stem_recall", "micro_stem_f1",
    "macro_stem_f1",
)
LOSS_TOLERANCE = 0.005


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pairs(raw: Iterable[Iterable[int]]) -> tuple[tuple[int, int], ...]:
    answer = []
    for value in raw:
        if len(value) != 2:
            raise ValueError("pair must have exactly two coordinates")
        i, j = value
        if isinstance(i, bool) or isinstance(j, bool) or not isinstance(i, (int, np.integer)):
            raise ValueError("pair coordinates must be integers")
        if not isinstance(j, (int, np.integer)) or not 0 <= i < j:
            raise ValueError("pair coordinates must satisfy 0 <= i < j")
        answer.append((int(i), int(j)))
    if len(set(answer)) != len(answer):
        raise ValueError("duplicate pair")
    endpoints = [position for pair in answer for position in pair]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("a nucleotide occurs in more than one pair")
    return tuple(sorted(answer))


def span_bin(pair: tuple[int, int]) -> str:
    span = pair[1] - pair[0]
    for upper, name in zip((3, 7, 15, 31, 63, 127), SPANS[:-1]):
        if span <= upper:
            return name
    return "ge128"


def isolated_pairs(structure: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    members = set(structure)
    return {
        (i, j) for i, j in structure
        if (i + 1, j - 1) not in members and (i - 1, j + 1) not in members
    }


def conflicts(a: tuple[int, int], b: tuple[int, int]) -> bool:
    i, j = a
    k, l = b
    return bool({i, j} & {k, l}) or i < k < j < l or k < i < l < j


def environment_pairs(
    structure: tuple[tuple[int, int], ...], environment: str, stratum: str,
    reference: tuple[tuple[int, int], ...] | None = None,
) -> set[tuple[int, int]]:
    if environment == "isolated_pair":
        answer = isolated_pairs(structure)
        return answer if stratum == "isolated" else set(structure) - answer
    if environment == "pair_span":
        return {pair for pair in structure if span_bin(pair) == stratum}
    if environment == "competition":
        if reference is None:
            raise ValueError("competition environment requires a fixed reference")
        return {pair for pair in structure if any(conflicts(pair, other) for other in reference)}
    raise ValueError(f"unknown environment: {environment}")


def stems(structure: tuple[tuple[int, int], ...]) -> set[frozenset[tuple[int, int]]]:
    remaining = set(structure)
    answer = set()
    while remaining:
        stack = [remaining.pop()]
        component = set(stack)
        while stack:
            i, j = stack.pop()
            for neighbor in ((i + 1, j - 1), (i - 1, j + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        answer.add(frozenset(component))
    return answer


def confusion(
    truth: Iterable, predicted: Iterable, selected_truth: set | None = None,
    selected_predicted: set | None = None,
) -> tuple[int, int, int]:
    truth_all, predicted_all = set(truth), set(predicted)
    if selected_truth is None:
        truth_set, predicted_set = truth_all, predicted_all
        return len(truth_set & predicted_set), len(predicted_set - truth_set), len(truth_set - predicted_set)
    truth_set = truth_all & selected_truth
    predicted_set = predicted_all & selected_predicted
    return (
        len(truth_set & predicted_all),
        len(predicted_set - truth_all),
        len(truth_set - predicted_all),
    )


def prf(counts: np.ndarray | Iterable[float], empty: float = 0.0) -> tuple[float, float, float]:
    tp, fp, fn = (float(x) for x in counts)
    precision = tp / (tp + fp) if tp + fp else empty
    recall = tp / (tp + fn) if tp + fn else empty
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else empty
    return precision, recall, f1


def sequence_prf(counts: np.ndarray) -> np.ndarray:
    return np.asarray([prf(row, empty=1.0) for row in counts], dtype=np.float64)


@dataclass
class Record:
    sequence_id: str
    family_id: str | None
    true_pairs: tuple[tuple[int, int], ...]
    predicted_pairs: tuple[tuple[int, int], ...] = ()
    raw_scores: list[tuple[int, int, float]] | None = None


@dataclass
class ArmArrays:
    pair_counts: np.ndarray
    stem_counts: np.ndarray
    pair_sequence_prf: np.ndarray
    stem_sequence_f1: np.ndarray
    subgroup_counts: dict[str, np.ndarray]


class BootstrapSpace:
    def __init__(self, records: list[Record], resamples: int, seed: int):
        self.records = records
        self.resamples = resamples
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.indices = rng.integers(0, len(records), size=(resamples, len(records)), dtype=np.int32)
        grouped: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            if record.family_id is not None:
                grouped.setdefault(record.family_id, []).append(index)
        self.family_ids = sorted(grouped)
        self.family_rows = [np.asarray(grouped[name], dtype=np.int32) for name in self.family_ids]
        totals = [int(sum(len(r.true_pairs) for r in (records[i] for i in rows))) for rows in self.family_rows]
        sizes = [len(rows) for rows in self.family_rows]
        self.eligible = np.asarray([
            index for index, (size, total) in enumerate(zip(sizes, totals))
            if size >= 30 and total >= 500
        ], dtype=np.int32)
        counts = np.zeros((resamples, len(records)), dtype=np.float64)
        for row, sample in enumerate(self.indices):
            counts[row] = np.bincount(sample, minlength=len(records))
        self.family_multiplicity = [counts[:, rows] for rows in self.family_rows]
        self.family_totals = [multiplicity.sum(axis=1) for multiplicity in self.family_multiplicity]

    def macro(self, values: np.ndarray) -> tuple[float, np.ndarray]:
        observed_by_family = []
        sampled_by_family = []
        eligible = set(self.eligible.tolist())
        for index, rows in enumerate(self.family_rows):
            if index not in eligible:
                continue
            observed_by_family.append(float(values[rows].mean()))
            sampled_by_family.append(
                (self.family_multiplicity[index] * values[rows]).sum(axis=1)
                / self.family_totals[index]
            )
        if not observed_by_family:
            raise ValueError("no macro-component-eligible mapped family")
        return float(np.mean(observed_by_family)), np.mean(sampled_by_family, axis=0)


def load_panel(registry_path: Path, panel_id: str) -> tuple[dict, list[Record], dict]:
    registry = json.loads(registry_path.read_text())
    if registry.get("schema_version") != "foldnar.model_dev_registry.v1":
        raise ValueError("unexpected registry schema")
    panel = next((p for p in registry["panels"] if p["panel_id"] == panel_id), None)
    if panel is None or panel_id == "legacy_vl0_control" or "dev_ids" not in panel:
        raise ValueError("route evaluation requires a registered non-legacy dev panel")
    used = {
        "panel_records": registry["environment_contracts"]["panel_records"],
        "environment_summary": registry["artifacts"]["environment_summary"],
        "dev_ids": panel["dev_ids"],
    }
    identities = {}
    for name, artifact in used.items():
        path = Path(artifact["path"])
        if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise ValueError(f"frozen artifact identity mismatch: {name}")
        identities[name] = {"path": str(path), "sha256": artifact["sha256"]}
    allowed = set(Path(panel["dev_ids"]["path"]).read_text().splitlines())
    rows = read_jsonl(used["panel_records"]["path"])
    records = [
        Record(row["sequence_id"], row.get("family_id"), pairs(row.get("true_pairs", ())))
        for row in rows if row["sequence_id"] in allowed
    ]
    if len(records) != len(allowed) or {r.sequence_id for r in records} != allowed:
        raise ValueError("panel records do not exactly cover frozen dev IDs")
    by_id = {row["sequence_id"]: row for row in rows}
    return registry, records, {"identities": identities, "panel_rows": by_id, "panel": panel}


def bind_predictions(
    panel_records: list[Record], arm_rows: dict[str, list[dict]], panel_rows: dict[str, dict],
) -> dict[str, list[Record]]:
    expected_ids = {record.sequence_id for record in panel_records}
    reference_truth: dict[str, tuple[tuple[int, int], ...]] = {}
    bound: dict[str, dict[str, Record]] = {}
    for arm, rows in arm_rows.items():
        if len(rows) != len(expected_ids):
            raise ValueError(f"{arm}: prediction count does not match panel")
        by_id = {}
        for row in rows:
            sequence_id = row["sequence_id"]
            if sequence_id in by_id or sequence_id not in expected_ids:
                raise ValueError(f"{arm}: duplicate or out-of-panel sequence_id")
            truth = pairs(row["true_pairs"])
            pred = pairs(row["predicted_pairs"])
            metadata = panel_rows[sequence_id]
            if len(truth) != metadata["n_pairs"]:
                raise ValueError(f"{arm}: true pair count mismatch for {sequence_id}")
            span_counts = Counter(span_bin(pair) for pair in truth)
            if any(span_counts.get(name, 0) != metadata["pair_span_counts"].get(name, 0) for name in SPANS):
                raise ValueError(f"{arm}: pair-span identity mismatch for {sequence_id}")
            isolated = len(isolated_pairs(truth))
            if isolated != metadata["isolated_pair_counts"].get("isolated", 0):
                raise ValueError(f"{arm}: isolated-pair identity mismatch for {sequence_id}")
            if sequence_id in reference_truth and reference_truth[sequence_id] != truth:
                raise ValueError("true pairs differ across arms")
            reference_truth[sequence_id] = truth
            if "sequence_sha256" in row and row["sequence_sha256"] != metadata["sequence_sha256"]:
                raise ValueError(f"{arm}: sequence identity mismatch for {sequence_id}")
            raw = None
            if "raw_scores" in row:
                raw = [(int(i), int(j), float(score)) for i, j, score in row["raw_scores"]]
                if not all(0 <= i < j and math.isfinite(score) for i, j, score in raw):
                    raise ValueError(f"{arm}: invalid raw score")
                if len({(i, j) for i, j, _ in raw}) != len(raw):
                    raise ValueError(f"{arm}: duplicate raw-score candidate")
            by_id[sequence_id] = Record(sequence_id, "", truth, pred, raw)
        bound[arm] = by_id
    ordered = {}
    metadata_by_id = {record.sequence_id: record for record in panel_records}
    for arm, by_id in bound.items():
        ordered[arm] = [
            Record(sid, metadata_by_id[sid].family_id, reference_truth[sid],
                   by_id[sid].predicted_pairs, by_id[sid].raw_scores)
            for sid in sorted(expected_ids)
        ]
    return ordered


def make_arrays(records: list[Record], reference: list[Record]) -> ArmArrays:
    subgroup_names = list(TARGETS) + ["competition:reference_conflicts"]
    pair_counts, stem_counts, subgroup_counts = [], [], {name: [] for name in subgroup_names}
    for record, ref in zip(records, reference):
        truth, pred = record.true_pairs, record.predicted_pairs
        pair_counts.append(confusion(truth, pred))
        true_stems, pred_stems = stems(truth), stems(pred)
        stem_counts.append(confusion(true_stems, pred_stems))
        for name in subgroup_names:
            environment, stratum = name.split(":", 1)
            reference_pairs = ref.predicted_pairs if environment == "competition" else None
            true_selected = environment_pairs(truth, environment, stratum, reference_pairs)
            pred_selected = environment_pairs(pred, environment, stratum, reference_pairs)
            subgroup_counts[name].append(confusion(truth, pred, true_selected, pred_selected))
    pair_array = np.asarray(pair_counts, dtype=np.float64)
    stem_array = np.asarray(stem_counts, dtype=np.float64)
    return ArmArrays(
        pair_array, stem_array, sequence_prf(pair_array),
        sequence_prf(stem_array)[:, 2],
        {name: np.asarray(rows, dtype=np.float64) for name, rows in subgroup_counts.items()},
    )


def concentration(records: list[Record], counts: np.ndarray, contract: dict) -> dict:
    true_counts = counts[:, 0] + counts[:, 2]
    family_weights: Counter[str] = Counter()
    family_sequences: Counter[str] = Counter()
    unmapped = 0.0
    for index, record in enumerate(records):
        weight = float(true_counts[index])
        if not weight:
            continue
        if record.family_id is None:
            unmapped += weight
        else:
            family_weights[record.family_id] += weight
            family_sequences[record.family_id] += 1
    total = float(true_counts.sum())
    known_total = float(sum(family_weights.values()))
    top_family, top_weight = (family_weights.most_common(1)[0] if family_weights else (None, 0.0))
    top_fraction = top_weight / total if total else 1.0
    hhi = sum((weight / known_total) ** 2 for weight in family_weights.values()) if known_total else 1.0
    unmapped_fraction = unmapped / total if total else 1.0
    values = {
        "n_sequences": int(np.count_nonzero(true_counts)),
        "n_pairs": int(total),
        "n_known_families": len(family_weights),
        "top_known_family": top_family,
        "top_known_family_fraction_of_all": top_fraction,
        "known_family_hhi": hhi,
        "unmapped_or_ambiguous_family_fraction": unmapped_fraction,
        "family_sequence_counts": dict(family_sequences),
        "family_pair_weights": dict(family_weights),
    }
    failures = []
    if values["n_sequences"] < contract["min_sequences"]:
        failures.append("min_sequences")
    if values["n_pairs"] < contract["min_pairs"]:
        failures.append("min_pairs")
    if values["n_known_families"] < contract["min_known_families"]:
        failures.append("min_known_families")
    if top_fraction > contract["max_top_family_fraction"]:
        failures.append("top_family_fraction")
    if hhi > contract["max_family_hhi"]:
        failures.append("family_hhi")
    if unmapped_fraction > contract["max_unmapped_family_fraction"]:
        failures.append("unmapped_family_fraction")
    values.update(route_eligible=not failures, route_failure_reasons=failures,
                  route_role="ROUTE_STRATUM" if not failures else "DIAGNOSTIC_ONLY")
    return values


def recall_deltas(ensemble: np.ndarray, control: np.ndarray) -> float:
    return prf(ensemble)[1] - prf(control)[1]


def diversity(records: list[Record], ensemble: np.ndarray, control: np.ndarray) -> dict:
    total_delta = recall_deltas(ensemble.sum(axis=0), control.sum(axis=0))
    per_sequence = np.asarray([recall_deltas(e, c) for e, c in zip(ensemble, control)])
    leave_sequence = []
    for index in range(len(records)):
        leave_sequence.append(recall_deltas(
            ensemble.sum(axis=0) - ensemble[index], control.sum(axis=0) - control[index]
        ))
    families = sorted({r.family_id for r in records if r.family_id is not None})
    family_deltas = {}
    leave_family = []
    for family in families:
        rows = np.asarray([i for i, record in enumerate(records) if record.family_id == family])
        if not rows.size or not (ensemble[rows, 0] + ensemble[rows, 2]).sum():
            continue
        e_sum, c_sum = ensemble[rows].sum(axis=0), control[rows].sum(axis=0)
        family_deltas[family] = recall_deltas(e_sum, c_sum)
        leave_family.append(recall_deltas(ensemble.sum(axis=0) - e_sum, control.sum(axis=0) - c_sum))
    return {
        "recall_delta": total_delta,
        "n_improved_sequences": int(np.count_nonzero(per_sequence > 0)),
        "n_improved_families": int(sum(delta > 0 for delta in family_deltas.values())),
        "min_leave_one_sequence_out_recall_delta": min(leave_sequence, default=total_delta),
        "min_leave_one_family_out_recall_delta": min(leave_family, default=total_delta),
        "family_recall_deltas": family_deltas,
        "non_single_sequence": bool(leave_sequence and min(leave_sequence) > 0),
        "non_single_family": bool(leave_family and min(leave_family) > 0),
    }


def evaluate_arrays(
    records: list[Record], arrays: dict[str, ArmArrays], resamples: int = 2000,
    seed: int = 20261118,
) -> dict:
    ci_contract_pass = resamples == 2000 and seed == 20261118
    boot = BootstrapSpace(records, resamples, seed)

    def series(arm: str, metric: str, subgroup: str | None = None):
        data = arrays[arm]
        counts = data.subgroup_counts[subgroup] if subgroup else (
            data.stem_counts if metric.startswith("micro_stem") else data.pair_counts
        )
        sampled = counts[boot.indices].sum(axis=1)
        observed_counts = counts.sum(axis=0)
        if metric.startswith("micro_"):
            index = {"precision": 0, "recall": 1, "f1": 2}[metric.rsplit("_", 1)[1]]
            return prf(observed_counts)[index], np.asarray([prf(row)[index] for row in sampled])
        if subgroup:
            raise ValueError("subgroup bootstrap is micro-only")
        if metric == "macro_stem_f1":
            return boot.macro(data.stem_sequence_f1)
        index = {"precision": 0, "recall": 1, "f1": 2}[metric.rsplit("_", 1)[1]]
        return boot.macro(data.pair_sequence_prf[:, index])

    comparisons = {}
    for control in ("matched_control", "destroy_interval_shuffle", "independent_context"):
        metrics = {}
        for metric in GLOBAL_METRICS:
            e_observed, e_sample = series("ensemble_context", metric)
            c_observed, c_sample = series(control, metric)
            delta = e_sample - c_sample
            metrics[metric] = {
                "ensemble_context": e_observed, control: c_observed,
                "delta": e_observed - c_observed,
                "ci95": [float(x) for x in np.quantile(delta, (0.025, 0.975))],
            }
        subgroups = {}
        for subgroup in list(TARGETS) + ["competition:reference_conflicts"]:
            subgroup_metrics = {}
            for metric in ("precision", "recall", "f1"):
                e_observed, e_sample = series("ensemble_context", "micro_pair_" + metric, subgroup)
                c_observed, c_sample = series(control, "micro_pair_" + metric, subgroup)
                delta = e_sample - c_sample
                subgroup_metrics[metric] = {
                    "ensemble_context": e_observed, control: c_observed,
                    "delta": e_observed - c_observed,
                    "ci95": [float(x) for x in np.quantile(delta, (0.025, 0.975))],
                }
            e_fn = arrays["ensemble_context"].subgroup_counts[subgroup][:, 2]
            c_fn = arrays[control].subgroup_counts[subgroup][:, 2]
            fn_delta = e_fn[boot.indices].sum(axis=1) - c_fn[boot.indices].sum(axis=1)
            subgroup_metrics["fn_count"] = {
                "ensemble_context": float(e_fn.sum()), control: float(c_fn.sum()),
                "delta": float(e_fn.sum() - c_fn.sum()),
                "ci95": [float(x) for x in np.quantile(fn_delta, (0.025, 0.975))],
            }
            subgroups[subgroup] = subgroup_metrics
        comparisons[control] = {"metrics": metrics, "subgroups": subgroups}

    targets = {}
    route_contract = {
        "min_sequences": 100, "min_pairs": 1000, "min_known_families": 10,
        "max_family_hhi": 0.25, "max_top_family_fraction": 0.5,
        "max_unmapped_family_fraction": 0.5,
    }
    for target in TARGETS:
        e_counts = arrays["ensemble_context"].subgroup_counts[target]
        c_counts = arrays["matched_control"].subgroup_counts[target]
        summary = concentration(records, c_counts, route_contract)
        summary["diversity"] = diversity(records, e_counts, c_counts)
        summary["recall_delta"] = comparisons["matched_control"]["subgroups"][target]["recall"]["delta"]
        summary["precision_delta"] = comparisons["matched_control"]["subgroups"][target]["precision"]["delta"]
        summary["f1_delta"] = comparisons["matched_control"]["subgroups"][target]["f1"]["delta"]
        summary["strict_improvement"] = summary["recall_delta"] > 0 and (
            not target.startswith("isolated_pair:") or summary["precision_delta"] > 0
        )
        summary["target_gate_pass"] = bool(
            summary["route_eligible"] and summary["strict_improvement"]
            and summary["diversity"]["non_single_family"]
            and summary["diversity"]["non_single_sequence"]
        )
        targets[target] = summary
    return {"comparisons": comparisons, "targets": targets,
            "ci_contract_pass": ci_contract_pass,
            "bootstrap": {"unit": "paired_sequence", "resamples": resamples,
                          "seed": seed, "confidence": 0.95,
                          "n_sequences": len(records),
                          "n_macro_eligible_families": len(boot.eligible)}}


def verify_gates(evaluation: dict) -> dict:
    matched = evaluation["comparisons"]["matched_control"]["metrics"]
    gates = {
        "paired_ci_contract": evaluation["ci_contract_pass"],
        "micro_pair_f1_nonnegative": matched["micro_pair_f1"]["delta"] >= 0,
        "macro_pair_f1_strictly_positive": matched["macro_pair_f1"]["delta"] > 0,
        "micro_precision_loss_at_most_0.005": matched["micro_pair_precision"]["delta"] >= -LOSS_TOLERANCE,
        "macro_precision_loss_at_most_0.005": matched["macro_pair_precision"]["delta"] >= -LOSS_TOLERANCE,
        "micro_stem_f1_loss_at_most_0.005": matched["micro_stem_f1"]["delta"] >= -LOSS_TOLERANCE,
        "macro_stem_f1_loss_at_most_0.005": matched["macro_stem_f1"]["delta"] >= -LOSS_TOLERANCE,
        "strictly_improved_diverse_target": any(t["target_gate_pass"] for t in evaluation["targets"].values()),
    }
    for control in ("destroy_interval_shuffle", "independent_context"):
        metrics = evaluation["comparisons"][control]["metrics"]
        for metric in ("micro_pair_f1", "macro_pair_f1", "micro_stem_f1", "macro_stem_f1"):
            gates[f"{control}:{metric}_strict_advantage"] = metrics[metric]["delta"] > 0
    return {"pass": all(gates.values()), "gates": {name: bool(value) for name, value in gates.items()},
            "failed_gates": [name for name, value in gates.items() if not value]}


def raw_auxiliary(arms: dict[str, list[Record]]) -> dict:
    if any(any(record.raw_scores is None for record in records) for records in arms.values()):
        return {"role": "AUXILIARY_ONLY_NEVER_GATE", "status": "NOT_PROVIDED_COMPLETE_FOUR_ARM_COVERAGE"}
    candidate_reference = None
    output = {"role": "AUXILIARY_ONLY_NEVER_GATE", "status": "OK", "arms": {}}
    for arm, records in arms.items():
        labels, scores, candidates = [], [], []
        for record in records:
            truth = set(record.true_pairs)
            for i, j, score in record.raw_scores or ():
                labels.append(int((i, j) in truth))
                scores.append(score)
                candidates.append((record.sequence_id, i, j))
        if candidate_reference is None:
            candidate_reference = set(candidates)
        elif set(candidates) != candidate_reference:
            raise ValueError("raw-score candidate coverage differs across arms")
        order = np.argsort(-np.asarray(scores), kind="mergesort")
        sorted_scores = np.asarray(scores)[order]
        sorted_labels = np.asarray(labels, dtype=np.int64)[order]
        tp = fp = 0
        previous_recall = 0.0
        auprc = 0.0
        recall_at_p08 = 0.0
        positives = int(sorted_labels.sum())
        start = 0
        while start < len(sorted_labels):
            end = start + 1
            while end < len(sorted_labels) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            group_tp = int(sorted_labels[start:end].sum())
            tp += group_tp
            fp += end - start - group_tp
            recall = tp / positives if positives else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            auprc += (recall - previous_recall) * precision
            if precision >= 0.8:
                recall_at_p08 = recall
            previous_recall = recall
            start = end
        output["arms"][arm] = {"auprc": auprc, "recall_at_precision_0.8": recall_at_p08,
                               "n_candidates": len(labels), "n_positive": positives}
    return output


def evaluate(
    registry_path: Path, panel_id: str, prediction_paths: dict[str, Path],
    resamples: int = 2000, seed: int = 20261118,
) -> dict:
    if set(prediction_paths) != set(ARMS):
        raise ValueError(f"exactly four arms required: {ARMS}")
    registry, panel_records, panel_info = load_panel(registry_path, panel_id)
    arm_rows = {arm: read_jsonl(path) for arm, path in prediction_paths.items()}
    arms = bind_predictions(panel_records, arm_rows, panel_info["panel_rows"])
    reference = arms["matched_control"]
    arrays = {arm: make_arrays(records, reference) for arm, records in arms.items()}
    result = evaluate_arrays(reference, arrays, resamples=resamples, seed=seed)
    result["gates"] = verify_gates(result)
    result["status"] = "PASS_GEC_SEED0" if result["gates"]["pass"] else "FAIL_GEC_SEED0"
    result["auxiliary_raw_ranking"] = raw_auxiliary(arms)
    result["panel"] = {
        "panel_id": panel_id, "n_sequences": len(reference),
        "n_pairs": sum(len(record.true_pairs) for record in reference),
        "registry": str(registry_path), "registry_sha256": sha256(registry_path),
        "frozen_utc": registry["frozen_utc"], "artifacts": panel_info["identities"],
        "prediction_sha256": {arm: sha256(path) for arm, path in prediction_paths.items()},
        "route_scope": "model development only; no external test data",
    }
    result["metric_definitions"] = {
        "macro": "equal-weight mapped families with >=30 sequences and >=500 true pairs; within-family mean of per-sequence metrics",
        "stem": "maximal inward/outward stacking connected components; exact whole-stem matches",
        "isolated_pair": registry["environment_contracts"]["pair_level"]["isolated_pair"],
        "competition": "pairs crossing or sharing an endpoint with a matched_control predicted pair; fixed for all arms",
        "target": "long-span recall >0; isolated recall and precision >0; route eligible and positive after removing any one family or sequence",
        "control_advantage": "strictly positive micro/macro pair-F1 and micro/macro stem-F1 delta",
        "raw_ranking": "auxiliary only; never included in gates",
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--panel-id", default="tr1c_rfam_family_dev_v1")
    parser.add_argument("--prediction", action="append", required=True,
                        help="ARM=/path/to/arm.jsonl; repeat for all four arms")
    parser.add_argument("--output", type=Path, help="new report path; inputs are read-only")
    args = parser.parse_args(argv)
    paths = {}
    for item in args.prediction:
        arm, separator, path = item.partition("=")
        if not separator or arm in paths:
            parser.error("each --prediction must be a unique ARM=PATH")
        paths[arm] = Path(path)
    try:
        result = evaluate(args.registry, args.panel_id, paths)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            if args.output.exists():
                raise ValueError(f"refusing to overwrite existing report: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
        return 0 if result["gates"]["pass"] else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"GEC evaluation input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
