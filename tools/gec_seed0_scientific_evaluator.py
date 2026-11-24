#!/usr/bin/env python3
"""Read-only four-arm GEC seed0 evaluation against the frozen model-dev contract.

Each ARM.jsonl row must contain sequence_id, true_pairs, predicted_pairs,
sequence_length, sequence_sha256 and checkpoint_sha256. Truth is accepted only
when it exactly matches the frozen raw TR1c source. Pairs are zero-based [i, j],
i < j. Optional raw_scores are [i, j, score] triples on a candidate set shared
by all arms. Inputs are never modified.
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
        len(truth_set & predicted_set),
        len(predicted_set - truth_set),
        len(truth_set - predicted_set),
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
    sequence_length: int = 0
    sequence_sha256: str = ""
    cluster_id: int | None = None


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


class FamilyClusterBootstrapSpace:
    def __init__(self, records: list[Record], resamples: int, seed: int):
        grouped: dict[str, list[int]] = {}
        self.mapped_family_by_group: dict[str, str | None] = {}
        for index, record in enumerate(records):
            key = (f"family:{record.family_id}" if record.family_id is not None
                   else f"unmapped_singleton:{record.sequence_id}")
            grouped.setdefault(key, []).append(index)
            self.mapped_family_by_group[key] = record.family_id
        self.group_ids = sorted(grouped)
        self.rows = [np.asarray(grouped[name], dtype=np.int32) for name in self.group_ids]
        rng = np.random.default_rng(seed)
        self.indices = rng.integers(
            0, len(self.rows), size=(resamples, len(self.rows)), dtype=np.int32,
        )
        self.multiplicity = np.zeros((resamples, len(records)), dtype=np.int32)
        for group_index, rows in enumerate(self.rows):
            for record_index in rows:
                self.multiplicity[:, record_index] = (self.indices == group_index).sum(axis=1)
        self.n_mapped_families = sum(
            self.mapped_family_by_group[name] is not None for name in self.group_ids
        )
        self.n_unmapped_singletons = len(self.rows) - self.n_mapped_families

    def sample_counts(self, counts: np.ndarray) -> np.ndarray:
        return np.einsum("bi,ij->bj", self.multiplicity, counts, optimize=True)

    def macro(self, values: np.ndarray, eligible_families: set[str]) -> tuple[float, np.ndarray]:
        eligible = [
            index for index, name in enumerate(self.group_ids)
            if self.mapped_family_by_group[name] in eligible_families
        ]
        if not eligible:
            raise ValueError("no macro-component-eligible mapped family")
        family_means = np.asarray([
            float(values[rows].mean()) if rows.size else np.nan for rows in self.rows
        ])
        observed = float(np.mean(family_means[eligible]))
        sampled = family_means[self.indices[:, eligible]].mean(axis=1)
        return observed, sampled


def _validate_bounds(
    values: Iterable[tuple[int, int]], length: int, label: str,
) -> None:
    if any(j >= length for _, j in values):
        raise ValueError(f"{label}: pair coordinate outside frozen sequence length")


def _frozen_identity(name: str, artifact: dict) -> dict:
    path = Path(artifact["path"])
    if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
        raise ValueError(f"frozen artifact identity mismatch: {name}")
    return {"path": str(path), "sha256": artifact["sha256"], "bytes": artifact["bytes"]}


def _read_frozen_ids(name: str, artifact: dict) -> set[str]:
    values = Path(artifact["path"]).read_text().splitlines()
    if not values or any(not value or value.strip() != value for value in values):
        raise ValueError(f"{name}: empty or malformed ID file")
    if len(set(values)) != len(values):
        raise ValueError(f"{name}: duplicate IDs")
    return set(values)


def load_panel(registry_path: Path, panel_id: str) -> tuple[dict, list[Record], dict]:
    registry = json.loads(registry_path.read_text())
    if registry.get("schema_version") != "foldnar.model_dev_registry.v1":
        raise ValueError("unexpected registry schema")
    if panel_id != "tr1c_cluster80_dev_v1":
        raise ValueError("scientific evaluation requires the registered cluster-dev panel")
    panel = next((p for p in registry["panels"] if p["panel_id"] == panel_id), None)
    if panel is None or panel.get("split_unit") != "existing 80% cluster":
        raise ValueError("missing registered cluster-dev panel")
    if "train_ids" not in panel or "dev_ids" not in panel:
        raise ValueError("cluster-dev panel lacks train/dev ID artifacts")
    used = {
        "panel_records": registry["environment_contracts"]["panel_records"],
        "environment_summary": registry["artifacts"]["environment_summary"],
        "dev_ids": panel["dev_ids"],
        "train_ids": panel["train_ids"],
    }
    identities = {name: _frozen_identity(name, artifact) for name, artifact in used.items()}
    raw_asset = next((
        asset for asset in registry["assets"]
        if asset.get("dataset_id") == "bprna_tr1c"
        and asset.get("role") == "legal_train_dev_source"
    ), None)
    if raw_asset is None:
        raise ValueError("registry lacks the frozen raw TR1c source")
    raw_path = Path(raw_asset["source"])
    if sha256(raw_path) != raw_asset["sha256"]:
        raise ValueError("frozen artifact identity mismatch: raw_tr1c")
    identities["raw_tr1c"] = {
        "path": str(raw_path), "sha256": raw_asset["sha256"], "bytes": raw_path.stat().st_size,
    }
    allowed = _read_frozen_ids("dev_ids", used["dev_ids"])
    planned_train_ids = _read_frozen_ids("train_ids", used["train_ids"])
    rows = read_jsonl(used["panel_records"]["path"])
    panel_rows = {row["sequence_id"]: row for row in rows}
    if len(panel_rows) != len(rows):
        raise ValueError("duplicate panel sequence_id")
    raw_records = {}
    for row_index, raw in enumerate(read_jsonl(raw_path)):
        sequence_id = f"bprna_tr1c:row{row_index:06d}:" + raw["name"]
        sequence = str(raw["seq"]).upper().replace("T", "U")
        truth = pairs(raw["pairs"])
        _validate_bounds(truth, len(sequence), f"raw TR1c {sequence_id}")
        raw_records[sequence_id] = {
            "sequence_id": sequence_id, "source_row_index": row_index, "name": raw["name"],
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "length": len(sequence), "true_pairs": truth,
        }
    if set(panel_rows) != set(raw_records):
        raise ValueError("panel rows do not exactly cover frozen raw TR1c rows")
    for sequence_id, raw in raw_records.items():
        metadata = panel_rows[sequence_id]
        if metadata["source_row_index"] != raw["source_row_index"] or metadata["name"] != raw["name"]:
            raise ValueError(f"raw TR1c row identity mismatch for {sequence_id}")
        if metadata["sequence_sha256"] != raw["sequence_sha256"] or metadata["length"] != raw["length"]:
            raise ValueError(f"raw TR1c sequence identity mismatch for {sequence_id}")
        truth = raw["true_pairs"]
        span_counts = Counter(span_bin(pair) for pair in truth)
        if (metadata["n_pairs"] != len(truth)
                or any(span_counts.get(name, 0) != metadata["pair_span_counts"].get(name, 0)
                       for name in SPANS)
                or len(isolated_pairs(truth)) != metadata["isolated_pair_counts"].get("isolated", 0)):
            raise ValueError(f"panel truth metadata mismatch for {sequence_id}")
    exact_overlap = allowed & planned_train_ids
    if exact_overlap:
        raise ValueError(f"panel rows occur in planned train IDs: {sorted(exact_overlap)[:3]}")
    train_clusters = {
        int(panel_rows[sequence_id]["cluster_id"]) for sequence_id in planned_train_ids
    }
    cluster_overlap = {
        sequence_id for sequence_id in allowed
        if int(panel_rows[sequence_id]["cluster_id"]) in train_clusters
    }
    if cluster_overlap:
        raise ValueError(f"panel rows share a registered train cluster: {sorted(cluster_overlap)[:3]}")
    records = []
    for sequence_id in sorted(allowed):
        metadata = panel_rows[sequence_id]
        if metadata.get("cluster_assignment") != "dev":
            raise ValueError(f"panel row is not registered cluster-dev: {sequence_id}")
        raw = raw_records[sequence_id]
        records.append(Record(
            sequence_id, metadata.get("family_id"), raw["true_pairs"],
            sequence_length=raw["length"], sequence_sha256=raw["sequence_sha256"],
            cluster_id=int(metadata["cluster_id"]),
        ))
    if len(records) != len(allowed):
        raise ValueError("panel records do not exactly cover frozen dev IDs")
    return registry, records, {
        "identities": identities, "panel_rows": panel_rows, "panel": panel,
        "allowed_ids": allowed, "planned_train_ids": planned_train_ids,
        "train_clusters": train_clusters,
    }


def validate_run_metadata(
    registry_path: Path, gec_config_path: Path, metadata_paths: dict[str, Path], panel_info: dict,
) -> dict:
    if set(metadata_paths) != set(ARMS):
        raise ValueError(f"exactly four run metadata files required: {ARMS}")
    config_path = gec_config_path.resolve()
    config = json.loads(config_path.read_text())
    config_sha256 = sha256(config_path)
    if config.get("config_id") != "gec_four_arm_implementation_v1" or tuple(config["arms"]) != ARMS:
        raise ValueError("unexpected GEC implementation config")
    training = config["training"]
    if (training["seed"] != 0 or training["n_blocks"] != 33
            or training["max_unfrozen_blocks"] != 20 or training["terminal_step"] != 9600
            or training["fixed_step_cap"] is not None):
        raise ValueError("GEC config is not the planned converged seed0 20/33 run")
    raw_identity = panel_info["identities"]["raw_tr1c"]
    if (Path(config["assets"]["data"]).resolve() != Path(raw_identity["path"]).resolve()
            or config["assets"]["data_sha256"] != raw_identity["sha256"]):
        raise ValueError("GEC config raw TR1c identity mismatch")
    configured_registry = Path(config["assets"]["registry"])
    if not configured_registry.is_absolute():
        configured_registry = Path(__file__).resolve().parents[1] / configured_registry
    if configured_registry.resolve() != registry_path.resolve():
        raise ValueError("GEC config registry identity mismatch")
    for key, name in (("train_ids", "train_ids"), ("validation_ids", "dev_ids")):
        artifact = panel_info["identities"][name]
        if Path(config["assets"][key]).resolve() != Path(artifact["path"]).resolve():
            raise ValueError(f"GEC config {key} differs from the frozen cluster panel")
    source_contract = Path(config["source_contract"])
    source_contract_sha256 = sha256(source_contract)
    convergence_contract = config["convergence"]
    if convergence_contract["stop_reason"] != "CONVERGED_VALIDATION_NLL_AFTER_TERMINAL_UNFREEZE":
        raise ValueError("unexpected GEC convergence stop reason")
    expected_schedule = {
        "seed": training["seed"], "batch_size": training["batch_size"],
        "head_lr": training["head_lr"], "backbone_lr": training["backbone_lr"],
        "warmup_head_steps": training["warmup_head_steps"],
        "unfreeze_every": training["unfreeze_every"],
        "unfreeze_per_step": training["unfreeze_per_step"],
        "max_unfrozen_blocks": 20, "warmup_steps": training["optimizer_warmup_steps"],
        "grad_clip": training["grad_clip"],
        "validation_every": convergence_contract["validation_every"],
        "validation_patience": convergence_contract["patience"],
        "validation_min_delta": convergence_contract["min_delta"],
    }
    common_fields = (
        "gec_config_sha256", "source_contract_sha256", "shared_init_sha256", "gec_init_sha256",
        "initial_residual_tensor_sha256", "residual_parameter_count",
        "trainable_head_and_residual_parameter_count", "train_record_ids", "validation_record_ids",
        "objective", "decoder",
    )
    reference = None
    summaries = {}
    for arm in ARMS:
        metadata_path = metadata_paths[arm].resolve()
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema_version") != "foldnar.gec_training_run.v1" or metadata.get("arm") != arm:
            raise ValueError(f"{arm}: unexpected run metadata identity")
        if Path(metadata["gec_config"]).resolve() != config_path or metadata["gec_config_sha256"] != config_sha256:
            raise ValueError(f"{arm}: run does not use the exact common GEC config")
        if (Path(metadata["source_contract"]).resolve() != source_contract.resolve()
                or metadata["source_contract_sha256"] != source_contract_sha256):
            raise ValueError(f"{arm}: source contract identity mismatch")
        if (Path(metadata["shared_init"]).resolve() != Path(config["assets"]["shared_init"]).resolve()
                or metadata["shared_init_sha256"] != config["assets"]["shared_init_sha256"]):
            raise ValueError(f"{arm}: shared initialization identity mismatch")
        train_ids = metadata["train_record_ids"]
        validation_ids = metadata["validation_record_ids"]
        if not isinstance(train_ids, list) or len(set(train_ids)) != len(train_ids):
            raise ValueError(f"{arm}: duplicate or malformed train IDs")
        if set(train_ids) != panel_info["planned_train_ids"]:
            raise ValueError(f"{arm}: training IDs differ from planned cluster train IDs")
        if (not isinstance(validation_ids, list) or len(set(validation_ids)) != len(validation_ids)
                or set(validation_ids) != panel_info["allowed_ids"]):
            raise ValueError(f"{arm}: validation IDs differ from full frozen cluster-dev panel")
        panel_train_overlap = panel_info["allowed_ids"] & set(train_ids)
        if panel_train_overlap:
            raise ValueError(f"{arm}: panel rows occur in run train IDs")
        run_train_clusters = {
            int(panel_info["panel_rows"][sequence_id]["cluster_id"]) for sequence_id in train_ids
        }
        if any(record_cluster in run_train_clusters for record_cluster in (
            int(panel_info["panel_rows"][sequence_id]["cluster_id"])
            for sequence_id in panel_info["allowed_ids"]
        )):
            raise ValueError(f"{arm}: panel rows share a run train cluster")
        schedule = metadata["schedule"]
        if (metadata["engineering_smoke"] is not False or schedule["engineering_smoke"] is not False
                or schedule["steps"] != 0 or schedule["train_record_limit"] != 0
                or schedule["validation_record_limit"] != 0):
            raise ValueError(f"{arm}: smoke or subsampled run is scientifically ineligible")
        for name, expected in expected_schedule.items():
            if schedule[name] != expected:
                raise ValueError(f"{arm}: common training config mismatch for {name}")
        if metadata["decoder"] != config["model"]["decoder"]:
            raise ValueError(f"{arm}: decoder differs from common config")
        if (metadata["status"] != "converged" or metadata["converged"] is not True
                or metadata["stop_reason"] != "CONVERGED_VALIDATION_NLL_AFTER_TERMINAL_UNFREEZE"
                or metadata["divergence_reason"]):
            raise ValueError(f"{arm}: run is not converged after terminal unfreeze")
        if (metadata["final_unfrozen_blocks"] != 20
                or metadata["backbone_report"]["n_layers"] != 33):
            raise ValueError(f"{arm}: final checkpoint is not 20/33 unfrozen blocks")
        convergence = metadata["convergence"]
        best_step = convergence["best_step"]
        if (convergence["terminal_step"] != 9600 or convergence["terminal_unfrozen_blocks"] != 20
                or convergence["eligible"] is not True or isinstance(best_step, bool)
                or not isinstance(best_step, int) or best_step < 9600
                or metadata["steps_completed"] < best_step
                or convergence["bad_evaluations"] < convergence_contract["patience"]
                or not math.isfinite(float(convergence["best_validation_loss"]))):
            raise ValueError(f"{arm}: best checkpoint is not eligible post-terminal convergence")
        environment = metadata["environment"]
        if (not str(environment["requested_device"]).startswith("cuda:")
                or environment["resolved_device"] != environment["requested_device"]
                or schedule["device"] != environment["requested_device"]):
            raise ValueError(f"{arm}: scientific run lacks explicit CUDA execution")
        best_path = metadata_path.parent / "best.pt"
        best_sha256 = metadata["best_checkpoint_sha256"]
        if not best_path.is_file() or sha256(best_path) != best_sha256:
            raise ValueError(f"{arm}: eligible best-checkpoint identity mismatch")
        current = {name: metadata[name] for name in common_fields}
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError(f"{arm}: common four-arm run metadata mismatch")
        summaries[arm] = {
            "path": str(metadata_path), "run_meta_sha256": sha256(metadata_path),
            "best_checkpoint_path": str(best_path), "best_checkpoint_sha256": best_sha256,
            "best_step": best_step, "resolved_device": environment["resolved_device"],
        }
    return {
        "config_path": str(config_path), "config_sha256": config_sha256,
        "source_contract_path": str(source_contract),
        "source_contract_sha256": source_contract_sha256,
        "planned_train_ids_sha256": panel_info["identities"]["train_ids"]["sha256"],
        "arms": summaries,
    }


def bind_predictions(
    panel_records: list[Record], arm_rows: dict[str, list[dict]], run_validation: dict,
) -> dict[str, list[Record]]:
    expected_ids = {record.sequence_id for record in panel_records}
    metadata_by_id = {record.sequence_id: record for record in panel_records}
    bound = {}
    for arm, rows in arm_rows.items():
        if len(rows) != len(expected_ids):
            raise ValueError(f"{arm}: prediction count does not match panel")
        checkpoint_sha256 = run_validation["arms"][arm]["best_checkpoint_sha256"]
        by_id = {}
        for row in rows:
            sequence_id = row["sequence_id"]
            if sequence_id in by_id or sequence_id not in expected_ids:
                raise ValueError(f"{arm}: duplicate or out-of-panel sequence_id")
            authoritative = metadata_by_id[sequence_id]
            if row.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError(f"{arm}: prediction checkpoint mismatch for {sequence_id}")
            if row.get("sequence_sha256") != authoritative.sequence_sha256:
                raise ValueError(f"{arm}: sequence identity mismatch for {sequence_id}")
            length = row.get("sequence_length")
            if isinstance(length, bool) or not isinstance(length, int) or length != authoritative.sequence_length:
                raise ValueError(f"{arm}: sequence length mismatch for {sequence_id}")
            truth = pairs(row["true_pairs"])
            if truth != authoritative.true_pairs:
                raise ValueError(f"{arm}: authoritative true-pair mismatch for {sequence_id}")
            pred = pairs(row["predicted_pairs"])
            _validate_bounds(pred, authoritative.sequence_length, f"{arm} {sequence_id}")
            raw = None
            if "raw_scores" in row:
                raw = [(int(i), int(j), float(score)) for i, j, score in row["raw_scores"]]
                if not all(0 <= i < j and math.isfinite(score) for i, j, score in raw):
                    raise ValueError(f"{arm}: invalid raw score")
                if len({(i, j) for i, j, _ in raw}) != len(raw):
                    raise ValueError(f"{arm}: duplicate raw-score candidate")
                _validate_bounds(((i, j) for i, j, _ in raw), authoritative.sequence_length,
                                 f"{arm} raw scores {sequence_id}")
            by_id[sequence_id] = Record(
                sequence_id, authoritative.family_id, authoritative.true_pairs, pred, raw,
                authoritative.sequence_length, authoritative.sequence_sha256, authoritative.cluster_id,
            )
        if set(by_id) != expected_ids:
            raise ValueError(f"{arm}: predictions do not exactly cover frozen dev IDs")
        bound[arm] = [by_id[sequence_id] for sequence_id in sorted(expected_ids)]
    return bound


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
    family_boot = FamilyClusterBootstrapSpace(records, resamples, seed)
    eligible_families = {boot.family_ids[index] for index in boot.eligible}

    def series(arm: str, metric: str, subgroup: str | None = None):
        data = arrays[arm]
        counts = data.subgroup_counts[subgroup] if subgroup else (
            data.stem_counts if metric.startswith("micro_stem") else data.pair_counts
        )
        sampled = counts[boot.indices].sum(axis=1)
        family_sampled = family_boot.sample_counts(counts)
        observed_counts = counts.sum(axis=0)
        if metric.startswith("micro_"):
            index = {"precision": 0, "recall": 1, "f1": 2}[metric.rsplit("_", 1)[1]]
            return (
                prf(observed_counts)[index],
                np.asarray([prf(row)[index] for row in sampled]),
                np.asarray([prf(row)[index] for row in family_sampled]),
            )
        if subgroup:
            raise ValueError("subgroup bootstrap is micro-only")
        if metric == "macro_stem_f1":
            observed, paired_sample = boot.macro(data.stem_sequence_f1)
            family_observed, family_sample = family_boot.macro(
                data.stem_sequence_f1, eligible_families,
            )
        else:
            index = {"precision": 0, "recall": 1, "f1": 2}[metric.rsplit("_", 1)[1]]
            observed, paired_sample = boot.macro(data.pair_sequence_prf[:, index])
            family_observed, family_sample = family_boot.macro(
                data.pair_sequence_prf[:, index], eligible_families,
            )
        if observed != family_observed:
            raise AssertionError("paired and family-cluster macro observations differ")
        return observed, paired_sample, family_sample

    comparisons = {}
    for control in ("matched_control", "destroy_interval_shuffle", "independent_context"):
        metrics = {}
        for metric in GLOBAL_METRICS:
            e_observed, e_sample, e_family_sample = series("ensemble_context", metric)
            c_observed, c_sample, c_family_sample = series(control, metric)
            delta = e_sample - c_sample
            family_delta = e_family_sample - c_family_sample
            metrics[metric] = {
                "ensemble_context": e_observed, control: c_observed,
                "delta": e_observed - c_observed,
                "ci95": [float(x) for x in np.quantile(delta, (0.025, 0.975))],
                "family_cluster_sensitivity_ci95": [
                    float(x) for x in np.quantile(family_delta, (0.025, 0.975))
                ],
            }
        subgroups = {}
        for subgroup in list(TARGETS) + ["competition:reference_conflicts"]:
            subgroup_metrics = {}
            for metric in ("precision", "recall", "f1"):
                e_observed, e_sample, e_family_sample = series(
                    "ensemble_context", "micro_pair_" + metric, subgroup,
                )
                c_observed, c_sample, c_family_sample = series(
                    control, "micro_pair_" + metric, subgroup,
                )
                delta = e_sample - c_sample
                family_delta = e_family_sample - c_family_sample
                subgroup_metrics[metric] = {
                    "ensemble_context": e_observed, control: c_observed,
                    "delta": e_observed - c_observed,
                    "ci95": [float(x) for x in np.quantile(delta, (0.025, 0.975))],
                    "family_cluster_sensitivity_ci95": [
                        float(x) for x in np.quantile(family_delta, (0.025, 0.975))
                    ],
                }
            e_fn = arrays["ensemble_context"].subgroup_counts[subgroup][:, 2]
            c_fn = arrays[control].subgroup_counts[subgroup][:, 2]
            fn_delta = e_fn[boot.indices].sum(axis=1) - c_fn[boot.indices].sum(axis=1)
            family_fn_delta = (
                family_boot.sample_counts(e_fn[:, None])[:, 0]
                - family_boot.sample_counts(c_fn[:, None])[:, 0]
            )
            subgroup_metrics["fn_count"] = {
                "ensemble_context": float(e_fn.sum()), control: float(c_fn.sum()),
                "delta": float(e_fn.sum() - c_fn.sum()),
                "ci95": [float(x) for x in np.quantile(fn_delta, (0.025, 0.975))],
                "family_cluster_sensitivity_ci95": [
                    float(x) for x in np.quantile(family_fn_delta, (0.025, 0.975))
                ],
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
                          "n_macro_eligible_families": len(boot.eligible),
                          "family_cluster_sensitivity": {
                              "unit": "registered_family_cluster; unmapped rows retained as labeled singletons",
                              "role": "SENSITIVITY_ONLY_NEVER_REPLACES_PAIRED_SEQUENCE_CI_OR_GATES",
                              "resamples": resamples, "seed": seed, "confidence": 0.95,
                              "n_clusters": len(family_boot.rows),
                              "n_mapped_families": family_boot.n_mapped_families,
                              "n_unmapped_singleton_clusters": family_boot.n_unmapped_singletons,
                          }}}


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
    run_metadata_paths: dict[str, Path], gec_config_path: Path | None = None,
    resamples: int = 2000, seed: int = 20261118,
) -> dict:
    if set(prediction_paths) != set(ARMS):
        raise ValueError(f"exactly four arms required: {ARMS}")
    if gec_config_path is None:
        gec_config_path = Path(__file__).resolve().parents[1] / "configs/gec_four_arm_convergence_v1.json"
    registry, panel_records, panel_info = load_panel(registry_path, panel_id)
    run_validation = validate_run_metadata(
        registry_path, gec_config_path, run_metadata_paths, panel_info,
    )
    arm_rows = {arm: read_jsonl(path) for arm, path in prediction_paths.items()}
    arms = bind_predictions(panel_records, arm_rows, run_validation)
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
        "leakage_eligibility": {
            "exact_train_id_overlap": False, "train_cluster_overlap": False,
            "planned_train_ids_sha256": run_validation["planned_train_ids_sha256"],
        },
    }
    result["run_validation"] = run_validation
    result["metric_definitions"] = {
        "macro": "equal-weight mapped families with >=30 sequences and >=500 true pairs; within-family mean of per-sequence metrics",
        "stem": "maximal inward/outward stacking connected components; exact whole-stem matches",
        "isolated_pair": registry["environment_contracts"]["pair_level"]["isolated_pair"],
        "competition": "pairs crossing or sharing an endpoint with a matched_control predicted pair; fixed for all arms",
        "target": "long-span recall >0; isolated recall and precision >0; route eligible and positive after removing any one family or sequence",
        "control_advantage": "strictly positive micro/macro pair-F1 and micro/macro stem-F1 delta",
        "raw_ranking": "auxiliary only; never included in gates",
        "subgroup_confusion": "TP/FP/FN require truth and prediction to occupy the same selected environment",
        "family_cluster_sensitivity": "additional sensitivity interval only; paired-sequence ci95 and gates are unchanged",
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--panel-id", default="tr1c_cluster80_dev_v1")
    parser.add_argument("--gec-config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/gec_four_arm_convergence_v1.json")
    parser.add_argument("--prediction", action="append", required=True,
                        help="ARM=/path/to/arm.jsonl; repeat for all four arms")
    parser.add_argument("--run-metadata", action="append", required=True,
                        help="ARM=/path/to/run_meta.json; repeat for all four arms")
    parser.add_argument("--output", type=Path, help="new report path; inputs are read-only")
    args = parser.parse_args(argv)
    paths = {}
    for item in args.prediction:
        arm, separator, path = item.partition("=")
        if not separator or arm in paths:
            parser.error("each --prediction must be a unique ARM=PATH")
        paths[arm] = Path(path)
    metadata_paths = {}
    for item in args.run_metadata:
        arm, separator, path = item.partition("=")
        if not separator or arm in metadata_paths:
            parser.error("each --run-metadata must be a unique ARM=PATH")
        metadata_paths[arm] = Path(path)
    try:
        result = evaluate(
            args.registry, args.panel_id, paths, metadata_paths, args.gec_config,
        )
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
