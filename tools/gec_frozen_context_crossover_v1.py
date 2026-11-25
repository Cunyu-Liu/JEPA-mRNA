#!/usr/bin/env python3
"""Frozen seed0 four-model by four-context causal crossover on cluster-dev."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from pair_decode_common import decode_mea
from rinalmo_preflight import VOCAB, load_encoder
from rnajepa import harness
from rnajepa.decision_head import FlatDecisionHead, pair_mask_from_ids
from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
from rnajepa.gec_context import GECResidualV1, gec_destroy_key_v1, gec_geometry_v1, require_explicit_cuda
from rnajepa.gec_numeric_v1 import compute_gec_context_v1
from tools.train_plan_a_gec_convergence_v1 import NEG_BIG, load_registered_records

ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
SPANS = ("le3", "4_7", "8_15", "16_31", "32_63", "64_127", "ge128")
SUBGROUPS = ("isolated_pair:isolated",) + tuple(f"pair_span:{name}" for name in SPANS)
SCHEMA = "foldnar.gec_frozen_context_crossover.v1"
SOURCE_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3")
OUTPUT_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_frozen_context_crossover_v1")
CONFIG = REPO / "configs/gec_four_arm_convergence_v1.json"
DECODER = {"name": "MEA", "temperature": 1.0, "gamma": 1.0,
           "score_variant": "train_off/plana_off", "search_allowed": False}
CODE_FILES = (
    "tools/gec_frozen_context_crossover_v1.py",
    "tools/train_plan_a_gec_convergence_v1.py",
    "tools/run_gec_seed0_production_v1.py",
    "tools/rinalmo_preflight.py",
    "tools/pair_decode_common.py",
    "src/rnajepa/gec_numeric_v1.py",
    "src/rnajepa/gec_context.py",
    "src/rnajepa/harness.py",
    "src/rnajepa/harness_cuda.py",
    "src/rnajepa/decision_head.py",
    "src/rnajepa/encoder.py",
    "src/rnajepa/prior_alignment.py",
    "src/rnajepa/raw_score_matrix.py",
    "configs/gec_four_arm_convergence_v1.json",
    "manifests/foldnar_v1/model_dev_registry.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def identity(path: str | Path) -> dict:
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_hashes(encoder, head, residual) -> dict[str, str]:
    return {name: tensor_state_sha256(module.state_dict())
            for name, module in (("encoder", encoder), ("head", head), ("residual", residual))}


def require_frozen(modules) -> None:
    for name, module in modules:
        if module.training:
            raise ValueError(f"{name} is not in eval mode")
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError(f"{name} has a trainable parameter")
        if any(parameter.device.type != "cuda" for parameter in module.parameters()):
            raise ValueError(f"{name} is not entirely on CUDA")


def atomic_json(value: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl_gz(rows: list[dict], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_config(config: dict) -> None:
    if (tuple(config.get("arms", ())) != ARMS or config["training"]["seed"] != 0
            or config["model"]["d_z"] != 128 or config["model"]["residual_hidden"] != 64
            or config["model"]["decoder"] != DECODER):
        raise ValueError("frozen four-arm seed0/decoder identity mismatch")


def historical_predictions(source_root: Path, config: dict) -> tuple[dict, dict]:
    predictions, identities = {}, {}
    for arm in ARMS:
        best = source_root / arm / "best.pt"
        export_path = source_root / arm / "prediction_export.json"
        export = json.loads(export_path.read_text())
        best_hash = sha256(best)
        prediction_path = source_root / arm / "dev_predictions.jsonl"
        if (export.get("arm") != arm or export.get("record_count") != 9143
                or export.get("panel_id") != "tr1c_cluster80_dev_v1"
                or export.get("device") != "cuda:0" or export.get("cpu_fallback") is not False
                or export.get("decoder") != config["model"]["decoder"]
                or export.get("best_checkpoint_sha256") != best_hash
                or export.get("prediction_sha256") != sha256(prediction_path)):
            raise ValueError(f"invalid historical export identity for {arm}")
        by_id = {}
        for row in read_jsonl(prediction_path):
            sequence_id = row["sequence_id"]
            if sequence_id in by_id or row["checkpoint_sha256"] != best_hash:
                raise ValueError(f"invalid historical prediction binding for {arm}")
            by_id[sequence_id] = row
        if len(by_id) != 9143:
            raise ValueError(f"historical {arm} does not cover full cluster-dev")
        predictions[arm] = by_id
        identities[arm] = {"best_checkpoint": identity(best), "prediction_export": identity(export_path),
                           "native_predictions": identity(prediction_path)}
    return predictions, identities


def family_eligibility(records: list[dict], registry_path: Path) -> tuple[dict, list[str]]:
    registry = json.loads(registry_path.read_text())
    panel_path = Path(registry["environment_contracts"]["panel_records"]["path"])
    families = {row["sequence_id"]: row.get("family_id") for row in read_jsonl(panel_path)}
    counts = Counter()
    pair_counts = Counter()
    for record in records:
        family = families[record["id"]]
        if family is not None:
            counts[family] += 1
            pair_counts[family] += len(record["pairs"])
    eligible = sorted(family for family in counts if counts[family] >= 30 and pair_counts[family] >= 500)
    if not eligible:
        raise ValueError("no macro-eligible family")
    return families, eligible


def valid_pairs(raw, length: int) -> tuple[tuple[int, int], ...]:
    answer = []
    for value in raw:
        if len(value) != 2:
            raise ValueError("pair must contain two coordinates")
        i, j = value
        if (isinstance(i, bool) or isinstance(j, bool) or not isinstance(i, int)
                or not isinstance(j, int) or not 0 <= i < j < length):
            raise ValueError("invalid pair coordinate")
        answer.append((i, j))
    endpoints = [position for pair in answer for position in pair]
    if len(set(answer)) != len(answer) or len(set(endpoints)) != len(endpoints):
        raise ValueError("duplicate pair or endpoint")
    return tuple(sorted(answer))


def span_bin(pair: tuple[int, int]) -> str:
    span = pair[1] - pair[0]
    for upper, name in zip((3, 7, 15, 31, 63, 127), SPANS):
        if span <= upper:
            return name
    return "ge128"


def isolated_pairs(structure) -> set[tuple[int, int]]:
    members = set(structure)
    return {(i, j) for i, j in members
            if (i + 1, j - 1) not in members and (i - 1, j + 1) not in members}


def stems(structure) -> set[frozenset[tuple[int, int]]]:
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


def confusion(truth, predicted) -> tuple[int, int, int]:
    truth, predicted = set(truth), set(predicted)
    return len(truth & predicted), len(predicted - truth), len(truth - predicted)


def prf(counts, empty=0.0) -> tuple[float, float, float]:
    tp, fp, fn = (float(value) for value in counts)
    precision = tp / (tp + fp) if tp + fp else empty
    recall = tp / (tp + fn) if tp + fn else empty
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else empty
    return precision, recall, f1


class MetricAccumulator:
    def __init__(self, records: list[dict], families: dict, eligible_families: list[str],
                 macro_enabled: bool = True):
        self.records = records
        self.families = families
        self.eligible_families = eligible_families
        self.macro_enabled = macro_enabled
        self.pair_counts = np.zeros(3, dtype=np.int64)
        self.stem_counts = np.zeros(3, dtype=np.int64)
        self.pair_sequence = []
        self.stem_sequence = []
        self.subgroups = {name: np.zeros(3, dtype=np.int64) for name in SUBGROUPS}

    def update(self, record: dict, predicted) -> None:
        truth = set(valid_pairs(record["pairs"], len(record["sequence"])))
        predicted = set(valid_pairs(predicted, len(record["sequence"])))
        pair_counts = confusion(truth, predicted)
        self.pair_counts += pair_counts
        self.stem_counts += confusion(stems(truth), stems(predicted))
        self.pair_sequence.append(prf(pair_counts, empty=1.0))
        self.stem_sequence.append(prf(confusion(stems(truth), stems(predicted)), empty=1.0))
        selections = {"isolated_pair:isolated": (isolated_pairs(truth), isolated_pairs(predicted))}
        for name in SPANS:
            selections[f"pair_span:{name}"] = (
                {pair for pair in truth if span_bin(pair) == name},
                {pair for pair in predicted if span_bin(pair) == name},
            )
        for name, (selected_truth, selected_predicted) in selections.items():
            self.subgroups[name] += confusion(truth & selected_truth, predicted & selected_predicted)

    def _macro(self, values: np.ndarray) -> tuple[float, float, float]:
        grouped = defaultdict(list)
        for index, record in enumerate(self.records):
            family = self.families[record["id"]]
            if family in self.eligible_families:
                grouped[family].append(values[index])
        if set(grouped) != set(self.eligible_families):
            raise ValueError("macro family coverage mismatch")
        return tuple(float(value) for value in np.mean(
            [np.mean(grouped[family], axis=0) for family in self.eligible_families], axis=0,
        ))

    def export(self) -> dict:
        result = {"micro_pair_prf": prf(self.pair_counts), "micro_stem_prf": prf(self.stem_counts),
                  "macro_pair_prf": self._macro(np.asarray(self.pair_sequence)) if self.macro_enabled else None,
                  "macro_stem_prf": self._macro(np.asarray(self.stem_sequence)) if self.macro_enabled else None,
                  "pair_confusion_tp_fp_fn": self.pair_counts.tolist(),
                  "stem_confusion_tp_fp_fn": self.stem_counts.tolist(), "subgroups": {}}
        for name, counts in self.subgroups.items():
            result["subgroups"][name] = {"micro_prf": prf(counts), "confusion_tp_fp_fn": counts.tolist()}
        return flatten_metrics(result)


def flatten_metrics(result: dict) -> dict:
    answer = {}
    for kind in ("pair", "stem"):
        for scope in ("micro", "macro"):
            for index, statistic in enumerate(("precision", "recall", "f1")):
                prf_values = result[f"{scope}_{kind}_prf"]
                answer[f"{scope}_{kind}_{statistic}"] = None if prf_values is None else prf_values[index]
    for name, values in result["subgroups"].items():
        answer[name] = {"micro_precision": values["micro_prf"][0], "micro_recall": values["micro_prf"][1],
                        "micro_f1": values["micro_prf"][2], "confusion_tp_fp_fn": values["confusion_tp_fp_fn"]}
    answer["pair_confusion_tp_fp_fn"] = result["pair_confusion_tp_fp_fn"]
    answer["stem_confusion_tp_fp_fn"] = result["stem_confusion_tp_fp_fn"]
    return answer


def metric_delta(left: dict, right: dict) -> dict:
    names = [f"{scope}_{kind}_{statistic}" for scope in ("micro", "macro")
             for kind in ("pair", "stem") for statistic in ("precision", "recall", "f1")]
    answer = {name: left[name] - right[name] for name in names}
    answer["subgroups"] = {name: {key: left[name][key] - right[name][key]
                                  for key in ("micro_precision", "micro_recall", "micro_f1")}
                           for name in SUBGROUPS}
    return answer


def native_pairs(row: dict, length: int) -> tuple[tuple[int, int], ...]:
    return valid_pairs(row["predicted_pairs"], length)


def decode_scores(scores: torch.Tensor, mask: torch.Tensor) -> tuple[tuple[int, int], ...]:
    mask_numpy = mask.detach().cpu().numpy()
    _, posterior = harness.inside_outside(
        scores.detach().cpu().numpy().astype(np.float64) / 1.0, mask_numpy,
    )
    if not np.isfinite(posterior).all():
        raise FloatingPointError("non-finite posterior")
    return tuple(sorted(decode_mea(posterior, mask_numpy, 1.0)))


def encode_base(record: dict, encoder, head, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence = record["sequence"]
    length = len(sequence)
    tokens = [VOCAB.get(char, VOCAB["<unk>"]) for char in sequence]
    framed = [VOCAB["<cls>"]] + tokens + [VOCAB["<eos>"]]
    encoder_ids = torch.tensor([framed], dtype=torch.long, device=device)
    attention = torch.ones_like(encoder_ids)
    head_ids = torch.tensor([[BASE_TO_INDEX.get(char, UNK_INDEX) for char in sequence]],
                            dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        encoded = encoder(input_ids=encoder_ids, attention_mask=attention)
    hidden = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded[0]
    hidden = hidden[:, 1:length + 1, :].float().to(torch.float16).to(torch.float32)
    chunks = [head.pair_repr.cross(hidden, hidden[:, start:start + 64]) for start in range(0, length, 64)]
    pair_representation = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
    raw_scores = head.score_pair_matrix(pair_representation, head_ids, False, None, None)
    lengths_tensor = torch.tensor([length], device=device)
    base_scores = head.calibration(raw_scores, lengths_tensor)
    full_mask = pair_mask_from_ids(head_ids, head.min_loop).to(device)
    base_scores = torch.where(full_mask, base_scores, torch.full_like(base_scores, NEG_BIG))
    mask = torch.as_tensor(harness.valid_pair_mask(sequence), device=device)
    return base_scores[0], pair_representation[0], mask


def crossover_record(record: dict, encoder, head, residual, device) -> dict:
    first_pass, pair_representation, mask = encode_base(record, encoder, head, device)
    s0_pairs = decode_scores(first_pass, mask)
    key = gec_destroy_key_v1(seed=0, epoch=0, step=0, sequence=record["sequence"])
    ensemble = compute_gec_context_v1(first_pass, mask, mode="ensemble_context", shuffle_key=key)
    geometry = gec_geometry_v1(mask, head.min_loop)
    rows = {}
    for context_arm in ARMS:
        numerics = compute_gec_context_v1(
            first_pass, mask, mode=context_arm, marginals=ensemble.marginals,
            log_z=ensemble.log_z, shuffle_key=key,
        )
        delta = residual(pair_representation, numerics.context, geometry, mask)
        scores = torch.where(mask, first_pass + delta, torch.full_like(first_pass, NEG_BIG))
        s1_pairs = decode_scores(scores, mask)
        rows[context_arm] = {
            "schema_version": SCHEMA + ".sequence", "sequence_id": record["id"],
            "sequence_length": len(record["sequence"]), "sequence_sha256": record["sequence_sha256"],
            "true_pairs": [list(pair) for pair in sorted(record["pairs"])],
            "s0_predicted_pairs": [list(pair) for pair in s0_pairs],
            "s1_predicted_pairs": [list(pair) for pair in s1_pairs],
        }
    return rows


def load_model(arm: str, source_root: Path, config: dict, device):
    best = source_root / arm / "best.pt"
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    if (checkpoint.get("checkpoint_phase") != "best_eligible" or checkpoint.get("n_unfrozen") != 20
            or checkpoint.get("run_identity", {}).get("arm") != arm):
        raise ValueError(f"ineligible frozen checkpoint for {arm}")
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    head = FlatDecisionHead(d_model=int(report["d_model"]), d_z=128, hidden=64,
                            scorer="resnet2d", chunk_size=0, resnet_blocks=4,
                            resnet_hidden=64, use_turner_prior=True)
    residual = GECResidualV1(128, 64)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    residual.load_state_dict(checkpoint["residual"], strict=True)
    head.use_turner_prior = False
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    residual = residual.to(device).eval()
    modules = (("encoder", encoder), ("head", head), ("residual", residual))
    for _, module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    require_frozen(modules)
    before = model_hashes(encoder, head, residual)
    if before != checkpoint.get("model_tensor_sha256"):
        raise ValueError(f"loaded tensor hashes differ from checkpoint for {arm}")
    del checkpoint
    return encoder, head, residual, before, report


def comparison_report(metrics: dict) -> dict:
    result = {}
    for model in ARMS:
        native = metrics[f"{model}__{model}"]
        matched = metrics[f"{model}__matched_control"]
        result[model] = {
            "native_s1_minus_shared_s0": metric_delta(native, metrics[f"{model}__matched_control_s0"]),
            "each_context_s1_minus_shared_s0": {
                context: metric_delta(metrics[f"{model}__{context}"], metrics[f"{model}__matched_control_s0"])
                for context in ARMS
            },
            "each_context_s1_minus_matched_s1": {
                context: metric_delta(metrics[f"{model}__{context}"], matched) for context in ARMS
            },
        }
    return result


def run(mode: str, output_root: Path, source_root: Path, config_path: Path) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError(mode)
    output = output_root / mode
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix or overwrite artifacts: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if os.environ.get("RN_CUDA_INSIDE") != "1":
        raise ValueError("crossover requires RN_CUDA_INSIDE=1")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = require_explicit_cuda("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    started_utc = utc_now()
    config = json.loads(config_path.read_text())
    validate_config(config)
    all_records = load_registered_records(config, "validation", 0)
    if len(all_records) != 9143:
        raise ValueError("unexpected diagnostic panel size")
    records = ([row for row in all_records if row["pairs"] and len(row["sequence"]) >= 8][:2]
               if mode == "smoke" else all_records)
    if len(records) != (2 if mode == "smoke" else 9143):
        raise ValueError("unexpected diagnostic subset size")
    registry_path = Path(config["assets"]["registry"])
    if not registry_path.is_absolute():
        registry_path = REPO / registry_path
    families, eligible_families = family_eligibility(all_records, registry_path)
    historical, input_identities = historical_predictions(source_root, config)
    metrics, native, artifacts, model_evidence = {}, {}, {}, {}
    s0_accumulators = {}
    for model in ARMS:
        model_started = time.monotonic()
        encoder, head, residual, hashes_before, report = load_model(model, source_root, config, device)
        rows_by_context = {context: [] for context in ARMS}
        accumulators = {context: MetricAccumulator(records, families, eligible_families, mode == "full")
                        for context in ARMS}
        s0_accumulator = MetricAccumulator(records, families, eligible_families, mode == "full")
        native_matches = 0
        with torch.inference_mode():
            for index, record in enumerate(records, 1):
                rows = crossover_record(record, encoder, head, residual, device)
                historical_row = historical[model][record["id"]]
                if (historical_row["sequence_sha256"] != record["sequence_sha256"]
                        or valid_pairs(historical_row["true_pairs"], len(record["sequence"])) != tuple(record["pairs"])):
                    raise ValueError(f"historical truth/sequence mismatch: {model} {record['id']}")
                native_observed = valid_pairs(rows[model]["s1_predicted_pairs"], len(record["sequence"]))
                native_expected = native_pairs(historical_row, len(record["sequence"]))
                if native_observed != native_expected:
                    raise RuntimeError(f"native prediction mismatch before completion: {model} {record['id']}")
                native_matches += 1
                for context in ARMS:
                    row = rows[context]
                    row.update(model_arm=model, context_arm=context,
                               checkpoint_sha256=input_identities[model]["best_checkpoint"]["sha256"],
                               native_context=context == model,
                               native_prediction_exact_match=native_observed == native_expected)
                    rows_by_context[context].append(row)
                    accumulators[context].update(record, row["s1_predicted_pairs"])
                s0_accumulator.update(record, rows[model]["s0_predicted_pairs"])
                if index % 100 == 0 or index == len(records):
                    print(f"[gec-crossover] mode={mode} model={model} {index}/{len(records)}", flush=True)
        hashes_after = model_hashes(encoder, head, residual)
        require_frozen((("encoder", encoder), ("head", head), ("residual", residual)))
        if hashes_after != hashes_before:
            raise RuntimeError(f"frozen parameters changed during inference: {model}")
        for context in ARMS:
            key = f"{model}__{context}"
            path = output / f"{key}.jsonl.gz"
            atomic_jsonl_gz(rows_by_context[context], path)
            artifacts[key] = identity(path)
            metrics[key] = accumulators[context].export()
        metrics[f"{model}__matched_control_s0"] = s0_accumulator.export()
        s0_accumulators[model] = metrics[f"{model}__matched_control_s0"]
        native[model] = {"matched_records": native_matches, "record_count": len(records), "pass": True}
        model_evidence[model] = {
            "model_tensor_sha256_before": hashes_before, "model_tensor_sha256_after": hashes_after,
            "all_parameters_requires_grad_false": True, "all_modules_eval": True,
            "elapsed_seconds": time.monotonic() - model_started,
            "cuda_max_memory_allocated_bytes_after_model": torch.cuda.max_memory_allocated(device),
            "cuda_max_memory_reserved_bytes_after_model": torch.cuda.max_memory_reserved(device),
            "backbone_report": report,
        }
        del encoder, head, residual
        torch.cuda.empty_cache()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    summary = {
        "schema_version": SCHEMA, "status": "COMPLETE", "mode": mode,
        "scientific_scope": "single_seed0_cluster_dev_frozen_causal_diagnostic_only",
        "gec_pass_claim": False, "sota_claim": False, "seeds_1_2_authorized": False,
        "panel_id": "tr1c_cluster80_dev_v1", "record_count": len(records),
        "record_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in records).encode()).hexdigest(),
        "models": list(ARMS), "input_contexts": list(ARMS), "decoder": DECODER,
        "inference_semantics": {"singleton_eval": True, "encoder_autocast": "bfloat16",
                                "hidden_roundtrip": "float32_to_float16_to_float32", "zchunk": 64,
                                "inside_outside": "exact_float64", "turner_prior": False,
                                "validation_destroy_seed_epoch_step": [0, 0, 0]},
        "native_prediction_validation": native, "metrics": metrics,
        "comparisons": comparison_report(metrics) if mode == "full" else {},
        "artifacts": artifacts, "source_root": str(source_root), "input_identities": input_identities,
        "data_identities": {"config": identity(config_path), "registry": identity(registry_path),
                            "raw_tr1c": identity(config["assets"]["data"]),
                            "dev_ids": identity(config["assets"]["validation_ids"]),
                            "source_contract": identity(config["source_contract"])},
        "code_manifest": {name: identity(REPO / name) for name in CODE_FILES},
        "repository": str(REPO), "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "git_status_porcelain_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "started_utc": started_utc, "completed_utc": utc_now(), "elapsed_seconds": time.monotonic() - started,
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "numpy": np.__version__,
                        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                        "RN_CUDA_INSIDE": os.environ["RN_CUDA_INSIDE"], "cpu_fallback": False,
                        "deterministic_algorithms": True, "tf32": False, "processes": 1},
        "memory_evidence": {"scope": "measured_inference_only_not_training_peak",
                            "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)},
        "model_evidence": model_evidence, "macro_eligible_families": eligible_families,
    }
    path = output / "summary.json"
    atomic_json(summary, path)
    print(json.dumps({"status": "COMPLETE", "mode": mode, "summary": str(path),
                      "native_prediction_validation": native}, indent=2), flush=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--gec-config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    run(args.mode, args.output_root, args.source_root, args.gec_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
