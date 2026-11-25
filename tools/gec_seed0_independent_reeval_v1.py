#!/usr/bin/env python3
"""Independent GEC seed0 re-evaluation from logs, checkpoints and predictions.

The official evaluator, run_meta.json and aggregate reports are never imported or
read. Checkpoint deserialization and metric recomputation are deliberately
CPU-only; no model inference or GPU validation is performed by this script.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ARMS = (
    "matched_control",
    "ensemble_context",
    "destroy_interval_shuffle",
    "independent_context",
)
CONTROLS = ("matched_control", "destroy_interval_shuffle", "independent_context")
SPANS = ("le3", "4_7", "8_15", "16_31", "32_63", "64_127", "ge128")
TARGETS = ("isolated_pair:isolated",) + tuple(
    f"pair_span:{name}" for name in ("16_31", "32_63", "64_127", "ge128")
)
SUBGROUPS = TARGETS + ("competition:reference_conflicts",)
GLOBAL_METRICS = (
    "micro_pair_precision", "micro_pair_recall", "micro_pair_f1",
    "macro_pair_precision", "macro_pair_recall", "macro_pair_f1",
    "micro_stem_precision", "micro_stem_recall", "micro_stem_f1",
    "macro_stem_precision", "macro_stem_recall", "macro_stem_f1",
)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261118
LOSS_TOLERANCE = 0.005


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: str | Path) -> dict:
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_jsonl(path: str | Path) -> list[dict]:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_pairs(raw: Iterable[Iterable[int]], length: int, label: str) -> tuple[tuple[int, int], ...]:
    answer = []
    for value in raw:
        if len(value) != 2:
            raise ValueError(f"{label}: pair must have two coordinates")
        i, j = value
        if (isinstance(i, bool) or isinstance(j, bool)
                or not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer))
                or not 0 <= i < j < length):
            raise ValueError(f"{label}: invalid pair coordinate {value}")
        answer.append((int(i), int(j)))
    if len(set(answer)) != len(answer):
        raise ValueError(f"{label}: duplicate pair")
    endpoints = [position for pair in answer for position in pair]
    if len(endpoints) != len(set(endpoints)):
        raise ValueError(f"{label}: nucleotide has multiple partners")
    return tuple(sorted(answer))


def span_bin(pair: tuple[int, int]) -> str:
    span = pair[1] - pair[0]
    for upper, name in zip((3, 7, 15, 31, 63, 127), SPANS[:-1]):
        if span <= upper:
            return name
    return "ge128"


def isolated(structure: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    members = set(structure)
    return {(i, j) for i, j in members
            if (i + 1, j - 1) not in members and (i - 1, j + 1) not in members}


def stems(structure: Iterable[tuple[int, int]]) -> set[frozenset[tuple[int, int]]]:
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


def conflicts(a: tuple[int, int], b: tuple[int, int]) -> bool:
    i, j = a
    k, l = b
    return bool({i, j} & {k, l}) or i < k < j < l or k < i < l < j


def selected_pairs(structure: Iterable[tuple[int, int]], subgroup: str,
                   reference: Iterable[tuple[int, int]] = ()) -> set[tuple[int, int]]:
    structure = set(structure)
    environment, stratum = subgroup.split(":", 1)
    if environment == "isolated_pair":
        answer = isolated(structure)
        return answer if stratum == "isolated" else structure - answer
    if environment == "pair_span":
        return {pair for pair in structure if span_bin(pair) == stratum}
    if environment == "competition":
        return {pair for pair in structure if any(conflicts(pair, other) for other in reference)}
    raise ValueError(f"unknown subgroup {subgroup}")


def confusion(truth: Iterable, predicted: Iterable, truth_selection: set | None = None,
              predicted_selection: set | None = None) -> tuple[int, int, int]:
    truth, predicted = set(truth), set(predicted)
    if truth_selection is None:
        return len(truth & predicted), len(predicted - truth), len(truth - predicted)
    selected_truth = truth & truth_selection
    selected_predicted = predicted & predicted_selection
    return (len(selected_truth & selected_predicted), len(selected_predicted - selected_truth),
            len(selected_truth - selected_predicted))


def prf_array(counts: np.ndarray, empty: float = 0.0) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    precision = np.divide(tp, tp + fp, out=np.full_like(tp, empty), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.full_like(tp, empty), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.full_like(tp, empty), where=(precision + recall) != 0)
    return np.stack((precision, recall, f1), axis=-1)


def prf(counts: Iterable[float], empty: float = 0.0) -> tuple[float, float, float]:
    return tuple(float(x) for x in prf_array(np.asarray(counts), empty=empty))


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def reconstruct_validation(rows: list[dict], terminal_step: int = 9600,
                           patience: int = 5, min_delta: float = 1e-6) -> dict:
    expected = []
    mismatches = []
    best_loss = math.inf
    best_step = None
    bad = 0
    stopped = False
    for index, row in enumerate(rows):
        step = row.get("step")
        if step != (index + 1) * 500:
            mismatches.append({"step": step, "field": "step", "expected": (index + 1) * 500})
        if not isinstance(step, int) or isinstance(step, bool):
            raise ValueError("validation step must be an integer")
        expected_unfrozen = min(20, max(0, ((step - 1600) // 800) * 2))
        eligible = step >= terminal_step and row.get("n_unfrozen") == 20
        loss = row.get("validation_loss")
        if not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isfinite(loss):
            raise ValueError(f"nonfinite validation loss at step {step}")
        if stopped:
            mismatches.append({"step": step, "field": "row", "expected": "no evaluations after stop"})
        if eligible:
            improved = loss < best_loss - min_delta
            if improved:
                best_loss, best_step, bad = loss, step, 0
            else:
                bad += 1
            save_best = improved
            stop = bad >= patience
        else:
            improved = save_best = stop = False
        values = {
            "bad_evaluations": bad, "best_step": best_step,
            "best_validation_loss": None if best_step is None else best_loss,
            "eligible": eligible, "improved": improved, "n_unfrozen": expected_unfrozen,
            "save_best": save_best, "step": step, "stop": stop, "validation_loss": loss,
        }
        for field, value in values.items():
            if row.get(field) != value:
                mismatches.append({"step": step, "field": field,
                                   "expected": value, "observed": row.get(field)})
        expected.append(values)
        stopped = stopped or stop
    final = expected[-1] if expected else {}
    return {
        "rows": expected, "mismatches": mismatches, "stop_step": final.get("step"),
        "best_step": final.get("best_step"), "best_validation_loss": final.get("best_validation_loss"),
        "bad_evaluations": final.get("bad_evaluations"), "stop": final.get("stop", False),
        "terminal_unfrozen_blocks": final.get("n_unfrozen"),
        "final_convergence": {
            "terminal_step": terminal_step, "terminal_unfrozen_blocks": 20,
            "patience": patience, "min_delta": min_delta, "best_step": final.get("best_step"),
            "best_validation_loss": final.get("best_validation_loss"),
            "bad_evaluations": final.get("bad_evaluations"), "eligible": True,
        },
    }


def audit_training_log(path: Path, stop_step: int) -> dict:
    rows = read_jsonl(path)
    errors = []
    for expected_step, row in enumerate(rows, 1):
        if row.get("step") != expected_step or row.get("optimizer_steps") != expected_step:
            errors.append(f"noncontiguous optimizer ledger at {expected_step}")
        if row.get("optimizer_step") is not True or row.get("divergence_reason"):
            errors.append(f"invalid optimizer row at {expected_step}")
    if len(rows) != stop_step:
        errors.append(f"training rows {len(rows)} != validation stop step {stop_step}")
    return {"path": str(path), "sha256": sha256(path), "rows": len(rows),
            "errors": errors, "pass": not errors}


def load_checkpoint(path: Path, label: str) -> tuple[dict | None, list[str]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        return None, [f"{label}: checkpoint deserialization failed: {type(error).__name__}"]
    if not isinstance(checkpoint, dict):
        return None, [f"{label}: checkpoint is not a dictionary"]
    return checkpoint, []


def checkpoint_content(checkpoint: dict, label: str) -> tuple[dict, list[str]]:
    errors = []
    hashes = {}
    for component in ("encoder", "head", "residual"):
        state = checkpoint.get(component)
        if not isinstance(state, dict) or not state:
            errors.append(f"{label}: missing {component} tensors")
            continue
        try:
            hashes[component] = tensor_state_sha256(state)
        except Exception as error:
            errors.append(f"{label}: invalid {component} tensors: {type(error).__name__}")
    stored = checkpoint.get("model_tensor_sha256")
    if stored != hashes:
        errors.append(f"{label}: stored tensor hashes differ from independent content hashes")
    return hashes, errors


def audit_checkpoint_pair(arm_dir: Path, arm: str, reconstruction: dict,
                          config_sha256: str, source_sha256: str) -> dict:
    best_path, resume_path = arm_dir / "best.pt", arm_dir / "resume.pt"
    best_identity, resume_identity = identity(best_path), identity(resume_path)
    errors = []
    best, load_errors = load_checkpoint(best_path, f"{arm} best")
    errors.extend(load_errors)
    best_hashes = {}
    run_identity = None
    if best is not None:
        best_hashes, content_errors = checkpoint_content(best, f"{arm} best")
        errors.extend(content_errors)
        run_identity = best.get("run_identity")
        expected_best_convergence = dict(reconstruction["final_convergence"])
        expected_best_convergence.update(
            best_step=reconstruction["best_step"], best_validation_loss=reconstruction["best_validation_loss"],
            bad_evaluations=0, eligible=True,
        )
        checks = {
            "format": best.get("format") == "foldnar.gec_checkpoint.v2",
            "phase": best.get("checkpoint_phase") == "best_eligible",
            "step": best.get("step") == reconstruction["best_step"],
            "optimizer_steps": best.get("optimizer_steps") == reconstruction["best_step"],
            "unfrozen": best.get("n_unfrozen") == 20,
            "convergence": best.get("convergence") == expected_best_convergence,
            "validation_loss": best.get("validation_loss") == reconstruction["best_validation_loss"],
        }
        errors.extend(f"{arm} best: {name} mismatch" for name, passed in checks.items() if not passed)
        selected = best.get("model")
        if not isinstance(selected, dict) or set(selected) != {"encoder", "head", "residual"}:
            errors.append(f"{arm} best: missing selected model state")
        else:
            for component in ("encoder", "head", "residual"):
                if tensor_state_sha256(selected[component]) != best_hashes.get(component):
                    errors.append(f"{arm} best: selected {component} differs from checkpoint state")
        if not isinstance(run_identity, dict):
            errors.append(f"{arm} best: missing run identity")
        else:
            identity_checks = {
                "arm": run_identity.get("arm") == arm,
                "config": run_identity.get("gec_config_sha256") == config_sha256,
                "source_contract": run_identity.get("source_contract_sha256") == source_sha256,
                "not_smoke": run_identity.get("engineering_smoke") is False,
                "terminal_unfrozen": run_identity.get("terminal_unfrozen_blocks") == 20,
            }
            errors.extend(f"{arm} best run identity: {name} mismatch"
                          for name, passed in identity_checks.items() if not passed)
    del best
    resume, load_errors = load_checkpoint(resume_path, f"{arm} resume")
    errors.extend(load_errors)
    resume_hashes = {}
    if resume is not None:
        resume_hashes, content_errors = checkpoint_content(resume, f"{arm} resume")
        errors.extend(content_errors)
        checks = {
            "format": resume.get("format") == "foldnar.gec_checkpoint.v2",
            "phase": resume.get("checkpoint_phase") == "restored_best_eligible",
            "terminal_status": resume.get("terminal_status") == "converged",
            "restored_step": resume.get("step") == reconstruction["best_step"],
            "restored_optimizer_steps": resume.get("optimizer_steps") == reconstruction["best_step"],
            "unfrozen": resume.get("n_unfrozen") == 20,
            "final_convergence_from_validation_log": (
                resume.get("convergence") == reconstruction["final_convergence"]
            ),
            "run_identity_matches_best": resume.get("run_identity") == run_identity,
            "selected_tensors_match_best": resume_hashes == best_hashes,
        }
        errors.extend(f"{arm} resume: {name} mismatch" for name, passed in checks.items() if not passed)
        observed_convergence = resume.get("convergence")
    else:
        observed_convergence = None
    del resume
    return {
        "best_checkpoint": best_identity, "resume_checkpoint": resume_identity,
        "best_step_from_log": reconstruction["best_step"],
        "expected_final_convergence": reconstruction["final_convergence"],
        "observed_resume_convergence": observed_convergence,
        "independent_best_tensor_sha256": best_hashes,
        "independent_resume_tensor_sha256": resume_hashes,
        "run_identity_sha256": canonical_hash(run_identity) if run_identity is not None else None,
        "common_run_identity_sha256": (
            canonical_hash({key: value for key, value in run_identity.items() if key != "arm"})
            if run_identity is not None else None
        ),
        "errors": errors, "pass": not errors,
    }


@dataclass(frozen=True)
class Record:
    sequence_id: str
    family_id: str | None
    cluster_id: int
    length: int
    sequence_sha256: str
    true_pairs: tuple[tuple[int, int], ...]
    predicted_pairs: tuple[tuple[int, int], ...] = ()


def checked_frozen_identity(name: str, artifact: dict) -> dict:
    actual = identity(artifact["path"])
    if actual["bytes"] != artifact["bytes"] or actual["sha256"] != artifact["sha256"]:
        raise ValueError(f"frozen identity mismatch: {name}")
    return actual


def read_ids(artifact: dict) -> set[str]:
    rows = Path(artifact["path"]).read_text().splitlines()
    if not rows or len(rows) != len(set(rows)) or any(not row or row.strip() != row for row in rows):
        raise ValueError("empty, duplicate or malformed frozen IDs")
    return set(rows)


def load_frozen_panel(repo: Path, config_path: Path) -> tuple[list[Record], dict, dict, dict]:
    config = json.loads(config_path.read_text())
    source_path = Path(config["source_contract"])
    registry_path = Path(config["assets"]["registry"])
    if not registry_path.is_absolute():
        registry_path = repo / registry_path
    registry = json.loads(registry_path.read_text())
    panel = next(row for row in registry["panels"] if row["panel_id"] == "tr1c_cluster80_dev_v1")
    panel_identity = checked_frozen_identity("panel_records", registry["environment_contracts"]["panel_records"])
    dev_identity = checked_frozen_identity("dev_ids", panel["dev_ids"])
    train_identity = checked_frozen_identity("train_ids", panel["train_ids"])
    raw_asset = next(asset for asset in registry["assets"]
                     if asset.get("dataset_id") == "bprna_tr1c"
                     and asset.get("role") == "legal_train_dev_source")
    raw_path = Path(raw_asset["source"])
    raw_identity = identity(raw_path)
    if raw_identity["sha256"] != raw_asset["sha256"] or config["assets"]["data_sha256"] != raw_asset["sha256"]:
        raise ValueError("frozen raw TR1c hash mismatch")
    if Path(config["assets"]["data"]).resolve() != raw_path.resolve():
        raise ValueError("GEC config raw source differs from registry")
    dev_ids, train_ids = read_ids(panel["dev_ids"]), read_ids(panel["train_ids"])
    if dev_ids & train_ids:
        raise ValueError("exact train/dev ID leakage")
    panel_rows_list = read_jsonl(panel_identity["path"])
    panel_rows = {row["sequence_id"]: row for row in panel_rows_list}
    if len(panel_rows) != len(panel_rows_list):
        raise ValueError("duplicate panel sequence ID")
    train_clusters = {int(panel_rows[row]["cluster_id"]) for row in train_ids}
    if any(int(panel_rows[row]["cluster_id"]) in train_clusters for row in dev_ids):
        raise ValueError("train/dev cluster leakage")
    records = []
    seen_raw = set()
    for row_index, raw in enumerate(read_jsonl(raw_path)):
        sequence_id = f"bprna_tr1c:row{row_index:06d}:" + raw["name"]
        seen_raw.add(sequence_id)
        if sequence_id not in dev_ids:
            continue
        sequence = str(raw["seq"]).upper().replace("T", "U")
        metadata = panel_rows[sequence_id]
        truth = validate_pairs(raw["pairs"], len(sequence), sequence_id)
        sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
        if (metadata["source_row_index"] != row_index or metadata["name"] != raw["name"]
                or metadata["length"] != len(sequence) or metadata["sequence_sha256"] != sequence_hash
                or metadata["n_pairs"] != len(truth) or metadata["cluster_assignment"] != "dev"):
            raise ValueError(f"raw/panel identity mismatch: {sequence_id}")
        span_counts = Counter(span_bin(pair) for pair in truth)
        if any(metadata["pair_span_counts"].get(name, 0) != span_counts.get(name, 0) for name in SPANS):
            raise ValueError(f"panel span mismatch: {sequence_id}")
        if metadata["isolated_pair_counts"].get("isolated", 0) != len(isolated(truth)):
            raise ValueError(f"panel isolated-pair mismatch: {sequence_id}")
        records.append(Record(sequence_id, metadata.get("family_id"), int(metadata["cluster_id"]),
                              len(sequence), sequence_hash, truth))
    if seen_raw != set(panel_rows):
        raise ValueError("raw source and panel rows do not exactly correspond")
    if len(records) != len(dev_ids) or {row.sequence_id for row in records} != dev_ids:
        raise ValueError("frozen dev coverage mismatch")
    records.sort(key=lambda row: row.sequence_id)
    inputs = {
        "config": identity(config_path), "source_contract": identity(source_path),
        "registry": identity(registry_path), "raw_tr1c": raw_identity,
        "panel_records": panel_identity, "dev_ids": dev_identity, "train_ids": train_identity,
    }
    return records, registry, config, inputs


def bind_predictions(frozen: list[Record], prediction_paths: dict[str, Path],
                     checkpoint_identities: dict[str, dict]) -> tuple[dict[str, list[Record]], dict]:
    expected = {row.sequence_id: row for row in frozen}
    bound = {}
    prediction_identities = {}
    for arm in ARMS:
        path = prediction_paths[arm]
        prediction_identities[arm] = identity(path)
        rows = read_jsonl(path)
        by_id = {}
        checkpoint_sha256 = checkpoint_identities[arm]["best_checkpoint"]["sha256"]
        for row in rows:
            sequence_id = row.get("sequence_id")
            if sequence_id in by_id or sequence_id not in expected:
                raise ValueError(f"{arm}: duplicate or out-of-panel prediction {sequence_id}")
            authoritative = expected[sequence_id]
            if (row.get("sequence_sha256") != authoritative.sequence_sha256
                    or row.get("sequence_length") != authoritative.length):
                raise ValueError(f"{arm}: sequence identity mismatch {sequence_id}")
            if row.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError(f"{arm}: prediction checkpoint mismatch {sequence_id}")
            truth = validate_pairs(row["true_pairs"], authoritative.length, f"{arm} truth {sequence_id}")
            if truth != authoritative.true_pairs:
                raise ValueError(f"{arm}: frozen truth mismatch {sequence_id}")
            predicted = validate_pairs(row["predicted_pairs"], authoritative.length,
                                       f"{arm} prediction {sequence_id}")
            by_id[sequence_id] = Record(
                authoritative.sequence_id, authoritative.family_id, authoritative.cluster_id,
                authoritative.length, authoritative.sequence_sha256, authoritative.true_pairs, predicted,
            )
        if len(rows) != len(expected) or set(by_id) != set(expected):
            raise ValueError(f"{arm}: predictions do not exactly cover the frozen dev panel")
        bound[arm] = [by_id[row.sequence_id] for row in frozen]
    return bound, prediction_identities


def make_arm_arrays(records: list[Record], matched_reference: list[Record]) -> dict:
    pair_counts, stem_counts = [], []
    subgroup_counts = {name: [] for name in SUBGROUPS}
    for record, reference in zip(records, matched_reference):
        truth, predicted = record.true_pairs, record.predicted_pairs
        pair_counts.append(confusion(truth, predicted))
        stem_counts.append(confusion(stems(truth), stems(predicted)))
        for subgroup in SUBGROUPS:
            reference_pairs = reference.predicted_pairs if subgroup.startswith("competition:") else ()
            subgroup_counts[subgroup].append(confusion(
                truth, predicted, selected_pairs(truth, subgroup, reference_pairs),
                selected_pairs(predicted, subgroup, reference_pairs),
            ))
    pair_array = np.asarray(pair_counts, dtype=np.float64)
    stem_array = np.asarray(stem_counts, dtype=np.float64)
    return {
        "pair_counts": pair_array, "stem_counts": stem_array,
        "pair_sequence_prf": prf_array(pair_array, empty=1.0),
        "stem_sequence_prf": prf_array(stem_array, empty=1.0),
        "subgroup_counts": {name: np.asarray(value, dtype=np.float64)
                            for name, value in subgroup_counts.items()},
    }


class PairedBootstrap:
    def __init__(self, records: list[Record], resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED):
        self.records = records
        self.resamples = resamples
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.indices = rng.integers(0, len(records), size=(resamples, len(records)), dtype=np.int32)
        self.multiplicity = np.zeros((resamples, len(records)), dtype=np.int16)
        for row in range(resamples):
            self.multiplicity[row] = np.bincount(self.indices[row], minlength=len(records))
        families = {}
        self.group_of_record = []
        for index, record in enumerate(records):
            group = f"family:{record.family_id}" if record.family_id is not None else f"singleton:{record.sequence_id}"
            families.setdefault(group, len(families))
            self.group_of_record.append(families[group])
        self.group_ids = sorted(families, key=families.get)
        self.group_of_record = np.asarray(self.group_of_record, dtype=np.int32)
        self.group_rows = [np.flatnonzero(self.group_of_record == index)
                           for index in range(len(self.group_ids))]
        family_rng = np.random.default_rng(seed)
        self.group_indices = family_rng.integers(
            0, len(self.group_ids), size=(resamples, len(self.group_ids)), dtype=np.int32,
        )
        self.record_group_multiplicity = self.group_indices[:, self.group_of_record]
        self.eligible_families = []
        self.eligible_groups = []
        for group_index, rows in enumerate(self.group_rows):
            family_id = records[rows[0]].family_id
            if (family_id is not None and len(rows) >= 30
                    and sum(len(records[row].true_pairs) for row in rows) >= 500):
                self.eligible_families.append(family_id)
                self.eligible_groups.append(group_index)

    def sample_counts(self, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sequence = np.einsum("bi,ij->bj", self.multiplicity, counts, optimize=True)
        family = np.einsum("bi,ij->bj", self.record_group_multiplicity, counts, optimize=True)
        return sequence, family

    def sample_macro(self, values: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        sequence_parts = []
        all_group_means = np.asarray([float(values[rows].mean()) for rows in self.group_rows])
        for group_index in self.eligible_groups:
            rows = self.group_rows[group_index]
            multiplicity = self.multiplicity[:, rows]
            sequence_parts.append((multiplicity @ values[rows]) / multiplicity.sum(axis=1))
        if not sequence_parts:
            raise ValueError("no macro-eligible mapped family")
        observed = float(all_group_means[self.eligible_groups].mean())
        sequence = np.mean(sequence_parts, axis=0)
        family = all_group_means[self.group_indices[:, self.eligible_groups]].mean(axis=1)
        return observed, sequence, family


def observed_arm_metrics(records: list[Record], arrays: dict, bootstrap: PairedBootstrap) -> dict:
    result = {}
    for prefix in ("pair", "stem"):
        counts = arrays[f"{prefix}_counts"]
        micro = prf(counts.sum(axis=0))
        sequence_prf = arrays[f"{prefix}_sequence_prf"]
        family_values = []
        for group_index in bootstrap.eligible_groups:
            family_values.append(sequence_prf[bootstrap.group_rows[group_index]].mean(axis=0))
        macro = np.mean(family_values, axis=0)
        for index, name in enumerate(("precision", "recall", "f1")):
            result[f"micro_{prefix}_{name}"] = micro[index]
            result[f"macro_{prefix}_{name}"] = float(macro[index])
    return result


def metric_distribution(arrays: dict, bootstrap: PairedBootstrap, metric: str,
                        subgroup: str | None = None) -> dict:
    prefix = "stem" if "_stem_" in metric else "pair"
    statistic = metric.rsplit("_", 1)[1]
    statistic_index = {"precision": 0, "recall": 1, "f1": 2}[statistic]
    if metric.startswith("micro_"):
        counts = (arrays["subgroup_counts"][subgroup] if subgroup is not None
                  else arrays[f"{prefix}_counts"])
        observed = prf(counts.sum(axis=0))[statistic_index]
        sequence_counts, family_counts = bootstrap.sample_counts(counts)
        return {
            "observed": observed,
            "sequence": prf_array(sequence_counts)[..., statistic_index],
            "family": prf_array(family_counts)[..., statistic_index],
        }
    if subgroup is not None:
        raise ValueError("macro subgroup metrics are not defined")
    observed, sequence, family = bootstrap.sample_macro(arrays[f"{prefix}_sequence_prf"][..., statistic_index])
    return {"observed": observed, "sequence": sequence, "family": family}


def ci95(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def comparison(distribution_a: dict, distribution_b: dict, control: str) -> dict:
    return {
        "ensemble_context": distribution_a["observed"], control: distribution_b["observed"],
        "delta": distribution_a["observed"] - distribution_b["observed"],
        "ci95": ci95(distribution_a["sequence"] - distribution_b["sequence"]),
        "family_cluster_sensitivity_ci95": ci95(distribution_a["family"] - distribution_b["family"]),
    }


def concentration(records: list[Record], counts: np.ndarray, contract: dict) -> dict:
    true_counts = counts[:, 0] + counts[:, 2]
    family_weights, family_sequences = Counter(), Counter()
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
    top_family, top_weight = family_weights.most_common(1)[0] if family_weights else (None, 0.0)
    values = {
        "n_sequences": int(np.count_nonzero(true_counts)), "n_pairs": int(total),
        "n_known_families": len(family_weights), "top_known_family": top_family,
        "top_known_family_fraction_of_all": top_weight / total if total else 1.0,
        "known_family_hhi": (sum((weight / known_total) ** 2 for weight in family_weights.values())
                             if known_total else 1.0),
        "unmapped_or_ambiguous_family_fraction": unmapped / total if total else 1.0,
        "family_sequence_counts": dict(family_sequences), "family_pair_weights": dict(family_weights),
    }
    failures = []
    for name, contract_name, observed in (
        ("min_sequences", "min_sequences", values["n_sequences"]),
        ("min_pairs", "min_pairs", values["n_pairs"]),
        ("min_known_families", "min_known_families", values["n_known_families"]),
    ):
        if observed < contract[contract_name]:
            failures.append(name)
    if values["top_known_family_fraction_of_all"] > contract["max_top_family_fraction"]:
        failures.append("top_family_fraction")
    if values["known_family_hhi"] > contract["max_family_hhi"]:
        failures.append("family_hhi")
    if values["unmapped_or_ambiguous_family_fraction"] > contract["max_unmapped_family_fraction"]:
        failures.append("unmapped_family_fraction")
    values.update(route_eligible=not failures, route_failure_reasons=failures,
                  route_role="ROUTE_STRATUM" if not failures else "DIAGNOSTIC_ONLY")
    return values


def diversity(records: list[Record], ensemble: np.ndarray, control: np.ndarray) -> dict:
    ensemble_total, control_total = ensemble.sum(axis=0), control.sum(axis=0)
    delta = prf(ensemble_total)[1] - prf(control_total)[1]
    per_sequence = np.asarray([prf(e)[1] - prf(c)[1] for e, c in zip(ensemble, control)])
    leave_sequence = [prf(ensemble_total - ensemble[index])[1] - prf(control_total - control[index])[1]
                      for index in range(len(records))]
    family_deltas = {}
    leave_family = []
    for family in sorted({record.family_id for record in records if record.family_id is not None}):
        rows = np.asarray([index for index, record in enumerate(records) if record.family_id == family])
        if not rows.size or not (ensemble[rows, 0] + ensemble[rows, 2]).sum():
            continue
        ensemble_sum, control_sum = ensemble[rows].sum(axis=0), control[rows].sum(axis=0)
        family_deltas[family] = prf(ensemble_sum)[1] - prf(control_sum)[1]
        leave_family.append(prf(ensemble_total - ensemble_sum)[1] - prf(control_total - control_sum)[1])
    return {
        "recall_delta": delta, "n_improved_sequences": int(np.count_nonzero(per_sequence > 0)),
        "n_improved_families": int(sum(value > 0 for value in family_deltas.values())),
        "min_leave_one_sequence_out_recall_delta": min(leave_sequence, default=delta),
        "min_leave_one_family_out_recall_delta": min(leave_family, default=delta),
        "family_recall_deltas": family_deltas,
        "non_single_sequence": bool(leave_sequence and min(leave_sequence) > 0),
        "non_single_family": bool(leave_family and min(leave_family) > 0),
    }


def evaluate_metrics(frozen: list[Record], arms: dict[str, list[Record]],
                     resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    if resamples != BOOTSTRAP_RESAMPLES or seed != BOOTSTRAP_SEED:
        raise ValueError("frozen paired CI contract requires 2000 resamples and seed 20261118")
    arrays = {arm: make_arm_arrays(arms[arm], arms["matched_control"]) for arm in ARMS}
    bootstrap = PairedBootstrap(frozen, resamples, seed)
    arm_metrics = {arm: observed_arm_metrics(frozen, arrays[arm], bootstrap) for arm in ARMS}
    distributions = {
        arm: {metric: metric_distribution(arrays[arm], bootstrap, metric) for metric in GLOBAL_METRICS}
        for arm in ARMS
    }
    subgroup_distributions = {
        arm: {subgroup: {name: metric_distribution(arrays[arm], bootstrap, f"micro_pair_{name}", subgroup)
                         for name in ("precision", "recall", "f1")}
              for subgroup in SUBGROUPS}
        for arm in ARMS
    }
    comparisons = {}
    for control in CONTROLS:
        subgroup_results = {}
        for subgroup in SUBGROUPS:
            values = {name: comparison(subgroup_distributions["ensemble_context"][subgroup][name],
                                       subgroup_distributions[control][subgroup][name], control)
                      for name in ("precision", "recall", "f1")}
            ensemble_fn = arrays["ensemble_context"]["subgroup_counts"][subgroup][:, 2:3]
            control_fn = arrays[control]["subgroup_counts"][subgroup][:, 2:3]
            ensemble_seq, ensemble_family = bootstrap.sample_counts(ensemble_fn)
            control_seq, control_family = bootstrap.sample_counts(control_fn)
            values["fn_count"] = {
                "ensemble_context": float(ensemble_fn.sum()), control: float(control_fn.sum()),
                "delta": float(ensemble_fn.sum() - control_fn.sum()),
                "ci95": ci95(ensemble_seq[:, 0] - control_seq[:, 0]),
                "family_cluster_sensitivity_ci95": ci95(ensemble_family[:, 0] - control_family[:, 0]),
            }
            subgroup_results[subgroup] = values
        comparisons[control] = {
            "metrics": {metric: comparison(distributions["ensemble_context"][metric],
                                           distributions[control][metric], control)
                        for metric in GLOBAL_METRICS},
            "subgroups": subgroup_results,
        }
    route_contract = {
        "min_sequences": 100, "min_pairs": 1000, "min_known_families": 10,
        "max_family_hhi": 0.25, "max_top_family_fraction": 0.5,
        "max_unmapped_family_fraction": 0.5,
    }
    targets = {}
    for target in TARGETS:
        ensemble_counts = arrays["ensemble_context"]["subgroup_counts"][target]
        control_counts = arrays["matched_control"]["subgroup_counts"][target]
        summary = concentration(frozen, control_counts, route_contract)
        summary["diversity"] = diversity(frozen, ensemble_counts, control_counts)
        target_comparison = comparisons["matched_control"]["subgroups"][target]
        summary["recall_delta"] = target_comparison["recall"]["delta"]
        summary["precision_delta"] = target_comparison["precision"]["delta"]
        summary["f1_delta"] = target_comparison["f1"]["delta"]
        summary["paired_ci95"] = {name: target_comparison[name]["ci95"]
                                  for name in ("precision", "recall", "f1", "fn_count")}
        summary["strict_improvement"] = summary["recall_delta"] > 0 and (
            not target.startswith("isolated_pair:") or summary["precision_delta"] > 0
        )
        summary["target_gate_pass"] = bool(
            summary["route_eligible"] and summary["strict_improvement"]
            and summary["diversity"]["non_single_family"]
            and summary["diversity"]["non_single_sequence"]
        )
        targets[target] = summary
    return {
        "arm_metrics": arm_metrics, "comparisons": comparisons, "targets": targets,
        "ci_contract_pass": True,
        "bootstrap": {
            "unit": "paired sequence on identical frozen IDs", "resamples": resamples,
            "seed": seed, "confidence": 0.95, "n_sequences": len(frozen),
            "n_macro_eligible_families": len(bootstrap.eligible_families),
            "macro_eligible_families": bootstrap.eligible_families,
            "family_cluster_sensitivity": {
                "unit": "registered family; unmapped sequences are singleton clusters",
                "role": "SENSITIVITY_ONLY_NEVER_REPLACES_PAIRED_SEQUENCE_CI_OR_GATES",
                "resamples": resamples, "seed": seed, "confidence": 0.95,
                "n_clusters": len(bootstrap.group_ids),
            },
        },
        "auxiliary_raw_ranking": {
            "role": "AUXILIARY_ONLY_NEVER_GATE", "status": "NOT_PROVIDED_COMPLETE_FOUR_ARM_COVERAGE",
        },
    }


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
        "strictly_improved_diverse_target": any(row["target_gate_pass"] for row in evaluation["targets"].values()),
    }
    for control in ("destroy_interval_shuffle", "independent_context"):
        for metric in ("micro_pair_f1", "macro_pair_f1", "micro_stem_f1", "macro_stem_f1"):
            gates[f"{control}:{metric}_strict_advantage"] = (
                evaluation["comparisons"][control]["metrics"][metric]["delta"] > 0
            )
    return {"pass": all(gates.values()), "gates": {name: bool(value) for name, value in gates.items()},
            "failed_gates": [name for name, value in gates.items() if not value]}


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(repo: Path, artifact_root: Path, output_root: Path) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output directory must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = repo / "configs/gec_four_arm_convergence_v1.json"
    config_sha256 = sha256(config_path)
    config = json.loads(config_path.read_text())
    source_sha256 = sha256(config["source_contract"])
    frozen, registry, config, inputs = load_frozen_panel(repo, config_path)
    validation_arms = {}
    checkpoint_arms = {}
    for arm in ARMS:
        print(f"[independent] reconstructing {arm} validation", flush=True)
        validation_path = artifact_root / arm / "validation_log.jsonl"
        reconstruction = reconstruct_validation(read_jsonl(validation_path))
        log_errors = list(reconstruction["mismatches"])
        if (reconstruction["stop_step"] != 13000 or reconstruction["best_step"] != 10500
                or reconstruction["bad_evaluations"] != 5 or reconstruction["stop"] is not True):
            log_errors.append("run did not stop at 13000 with best 10500 and five bad evaluations")
        training = audit_training_log(artifact_root / arm / "train_log.jsonl", reconstruction["stop_step"])
        validation_arms[arm] = {
            "validation_log": identity(validation_path), "training_log": training,
            "reconstruction": {key: value for key, value in reconstruction.items() if key != "rows"},
            "errors": log_errors + training["errors"],
            "pass": not log_errors and training["pass"],
        }
        print(f"[independent] auditing {arm} checkpoints on explicit CPU deserialization", flush=True)
        checkpoint_arms[arm] = audit_checkpoint_pair(
            artifact_root / arm, arm, reconstruction, config_sha256, source_sha256,
        )
    common_identity_hashes = {row["common_run_identity_sha256"] for row in checkpoint_arms.values()}
    if len(common_identity_hashes) != 1:
        for arm in ARMS:
            checkpoint_arms[arm]["errors"].append("four-arm checkpoint run identity differs")
            checkpoint_arms[arm]["pass"] = False
    prediction_paths = {arm: artifact_root / arm / "dev_predictions.jsonl" for arm in ARMS}
    print("[independent] binding frozen truth and complete prediction coverage", flush=True)
    arms, prediction_identities = bind_predictions(frozen, prediction_paths, checkpoint_arms)
    inputs.update({f"{arm}:validation_log": validation_arms[arm]["validation_log"] for arm in ARMS})
    inputs.update({f"{arm}:train_log": identity(artifact_root / arm / "train_log.jsonl") for arm in ARMS})
    inputs.update({f"{arm}:best_checkpoint": checkpoint_arms[arm]["best_checkpoint"] for arm in ARMS})
    inputs.update({f"{arm}:resume_checkpoint": checkpoint_arms[arm]["resume_checkpoint"] for arm in ARMS})
    inputs.update({f"{arm}:dev_predictions": prediction_identities[arm] for arm in ARMS})
    shared_init = identity(config["assets"]["shared_init"])
    if shared_init["sha256"] != config["assets"]["shared_init_sha256"]:
        raise ValueError("shared initialization hash mismatch")
    inputs["shared_init"] = shared_init
    inputs["gec_shared_residual_init"] = identity(artifact_root / "gec_shared_residual_init_seed0.pt")
    test_path = repo / "tests/test_gec_seed0_independent_reeval_v1.py"
    code_identity = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "script": identity(Path(__file__).resolve()),
    }
    if test_path.is_file():
        code_identity["tests"] = identity(test_path)
    print("[independent] recomputing all decoded metrics and 2000 paired bootstrap resamples", flush=True)
    evaluation = evaluate_metrics(frozen, arms)
    evaluation["metric_gates"] = verify_gates(evaluation)
    qualification_gates = {
        f"{arm}:validation_and_stop": validation_arms[arm]["pass"] for arm in ARMS
    }
    qualification_gates.update({
        f"{arm}:checkpoint_and_restored_convergence": checkpoint_arms[arm]["pass"] for arm in ARMS
    })
    qualification_gates.update({f"{arm}:exact_prediction_coverage": True for arm in ARMS})
    qualification_pass = all(qualification_gates.values())
    overall_pass = qualification_pass and evaluation["metric_gates"]["pass"]
    report = {
        "schema_version": "foldnar.gec_seed0_independent_reeval.v1",
        "status": "PASS_GEC_SEED0_INDEPENDENT_REEVAL" if overall_pass else "FAIL_GEC_SEED0_INDEPENDENT_REEVAL",
        "overall_pass": overall_pass, "qualification_pass": qualification_pass,
        "qualification_gates": qualification_gates,
        "failed_qualification_gates": [name for name, passed in qualification_gates.items() if not passed],
        "independence": {
            "official_evaluator_imported": False, "run_meta_read": False,
            "aggregate_summary_read": False,
            "inputs": "validation_log.jsonl, train_log.jsonl, best.pt, resume.pt, dev_predictions.jsonl and frozen truth",
        },
        "code_identity": code_identity,
        "execution": {
            "metric_device": "cpu", "checkpoint_deserialization": "map_location=cpu",
            "model_inference_performed": False, "gpu_validation_required": False,
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        },
        "panel": {
            "panel_id": "tr1c_cluster80_dev_v1", "n_sequences": len(frozen),
            "n_pairs": sum(len(row.true_pairs) for row in frozen),
            "exact_train_id_overlap": False, "train_cluster_overlap": False,
            "route_scope": "model development only; no external test data",
        },
        "validation_audit": validation_arms, "checkpoint_audit": checkpoint_arms,
        **evaluation,
    }
    gate_summary = {
        "status": report["status"], "overall_pass": overall_pass,
        "qualification_pass": qualification_pass,
        "failed_qualification_gates": report["failed_qualification_gates"],
        "metric_gates": evaluation["metric_gates"], "arm_metrics": evaluation["arm_metrics"],
        "matched_control_deltas": evaluation["comparisons"]["matched_control"]["metrics"],
        "destroy_deltas": evaluation["comparisons"]["destroy_interval_shuffle"]["metrics"],
        "independent_deltas": evaluation["comparisons"]["independent_context"]["metrics"],
        "targets": evaluation["targets"],
    }
    write_json(output_root / "input_manifest.json", {"inputs": inputs})
    write_json(output_root / "validation_checkpoint_audit.json", {
        "validation_audit": validation_arms, "checkpoint_audit": checkpoint_arms,
        "qualification_gates": qualification_gates, "qualification_pass": qualification_pass,
    })
    write_json(output_root / "independent_reeval.json", report)
    write_json(output_root / "gate_summary.json", gate_summary)
    manifest = {name: identity(output_root / name) for name in (
        "input_manifest.json", "validation_checkpoint_audit.json", "independent_reeval.json", "gate_summary.json",
    )}
    write_json(output_root / "artifact_manifest.json", {"artifacts": manifest})
    return gate_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = run(args.repo.resolve(), args.artifact_root, args.output_root)
        print(json.dumps({
            "status": summary["status"], "qualification_pass": summary["qualification_pass"],
            "metric_gates_pass": summary["metric_gates"]["pass"],
            "failed_qualification_gates": summary["failed_qualification_gates"],
            "failed_metric_gates": summary["metric_gates"]["failed_gates"],
        }, indent=2, sort_keys=True))
        return 0 if summary["overall_pass"] else 1
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError) as error:
        print(f"independent re-evaluation input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
