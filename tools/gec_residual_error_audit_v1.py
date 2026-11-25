#!/usr/bin/env python3
"""Frozen-checkpoint GEC residual and MEA error decomposition on registered dev only."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from rnajepa import harness
from rnajepa.harness import _max_dp, _traceback_max
from tools.gec_seed0_independent_reeval_v1 import conflicts, isolated, span_bin
from tools.pair_decode_common import mea_pair_weights

ARMS = ("ensemble_context", "destroy_interval_shuffle")
SPANS = ("le3", "4_7", "8_15", "16_31", "32_63", "64_127", "ge128")
SCHEMA = "foldnar.gec_residual_error_audit.v1"
NEG_BIG = -1.0e9
STOPGRAD_PROBE_COUNT = 64
STOPGRAD_EPSILON = 1.0e-3


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: str | Path) -> dict:
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def interval_value(table: np.ndarray, start: int, end: int) -> float:
    return 0.0 if start > end else float(table[start, end])


def interval_table(table: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    valid = start <= end
    safe_start = np.clip(start, 0, table.shape[0] - 1)
    safe_end = np.clip(end, 0, table.shape[1] - 1)
    return np.where(valid, table[safe_start, safe_end], 0.0)


def pair_outside_values(weights: np.ndarray, mask: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Maximum weight outside each fixed pair, including enclosing and disjoint pairs."""
    length = weights.shape[0]
    outside = np.zeros((length, length), dtype=np.float64)
    parent_left = np.full((length, length), -math.inf, dtype=np.float64)
    for distance in range(length - 1, -1, -1):
        rows = np.arange(length - distance)
        cols = rows + distance
        for i in rows.tolist():
            k = np.arange(i)
            t = np.arange(i + 1, length)
            if not k.size or not t.size:
                continue
            values = (
                outside[k[:, None], t[None, :]] + weights[k[:, None], t[None, :]]
                + interval_table(inside, k[:, None] + 1, np.full((k.size, 1), i - 1))
            )
            parent_left[i, t] = np.where(mask[k[:, None], t[None, :]], values, -math.inf).max(axis=0)
        best = interval_table(inside, np.zeros_like(rows), rows - 1) + interval_table(
            inside, cols + 1, np.full_like(cols, length - 1)
        )
        t = np.arange(length)[None, :]
        candidates = parent_left[rows] + interval_table(inside, cols[:, None] + 1, t - 1)
        nested = np.where(t > cols[:, None], candidates, -math.inf).max(axis=1)
        outside[rows, cols] = np.maximum(best, nested)
    return outside


def forced_pair_values(weights: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """Return optimal value, forced-pair values and the tied native traceback."""
    table, decisions, partners = _max_dp(np.asarray(weights, dtype=np.float64), mask, band=None)
    length = weights.shape[0]
    best = interval_value(table, 0, length - 1)
    outside = pair_outside_values(weights, mask, table)
    forced = weights + interval_table(
        table, np.arange(length)[:, None] + 1, np.arange(length)[None, :] - 1
    ) + outside
    forced = np.where(np.triu(mask, k=1), forced, np.nan)
    predicted = _traceback_max(decisions, partners, length)
    return np.full((length, length), best, dtype=np.float64), forced, predicted


def classify_pair(selected: bool, legal: bool, weight: float) -> str:
    if selected:
        return "selected"
    if not legal:
        return "illegal_or_noncanonical"
    return "positive_weight_mea_competition" if weight > 0.0 else "nonpositive_score_margin"


def central_difference(plus: float, minus: float, epsilon: float) -> float:
    if not epsilon > 0:
        raise ValueError("epsilon must be positive")
    return (plus - minus) / (2.0 * epsilon)


@dataclass
class OnlineStats:
    n: int = 0
    total: float = 0.0
    total_abs: float = 0.0
    total_sq: float = 0.0
    positive: int = 0
    negative: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        array = array[np.isfinite(array)]
        if not array.size:
            return
        self.n += int(array.size)
        self.total += float(array.sum(dtype=np.float64))
        self.total_abs += float(np.abs(array).sum(dtype=np.float64))
        self.total_sq += float(np.square(array).sum(dtype=np.float64))
        self.positive += int((array > 0).sum())
        self.negative += int((array < 0).sum())
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def export(self) -> dict:
        if not self.n:
            return {"n": 0}
        mean = self.total / self.n
        return {
            "n": self.n,
            "mean": mean,
            "mean_abs": self.total_abs / self.n,
            "rms": math.sqrt(max(0.0, self.total_sq / self.n)),
            "std": math.sqrt(max(0.0, self.total_sq / self.n - mean * mean)),
            "positive_fraction": self.positive / self.n,
            "negative_fraction": self.negative / self.n,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class GroupStats:
    values: dict[str, OnlineStats] = field(default_factory=lambda: defaultdict(OnlineStats))
    selected: int = 0
    nonpositive_score_margin: int = 0
    positive_weight_mea_competition: int = 0
    illegal_or_noncanonical: int = 0
    competition_gap: OnlineStats = field(default_factory=OnlineStats)

    def update_counts(self, classes: list[str], gaps: list[float]) -> None:
        for label in classes:
            if label == "selected":
                self.selected += 1
            elif label == "nonpositive_score_margin":
                self.nonpositive_score_margin += 1
            elif label == "positive_weight_mea_competition":
                self.positive_weight_mea_competition += 1
            elif label == "illegal_or_noncanonical":
                self.illegal_or_noncanonical += 1
            else:
                raise ValueError(label)
        self.competition_gap.update(gaps)

    def export(self) -> dict:
        truth = self.selected + self.nonpositive_score_margin + self.positive_weight_mea_competition + self.illegal_or_noncanonical
        return {
            "truth_pairs": truth,
            "recall": self.selected / truth if truth else None,
            "selected": self.selected,
            "nonpositive_score_margin": self.nonpositive_score_margin,
            "positive_weight_mea_competition": self.positive_weight_mea_competition,
            "illegal_or_noncanonical": self.illegal_or_noncanonical,
            "competition_gap": self.competition_gap.export(),
            "values": {name: stat.export() for name, stat in sorted(self.values.items())},
        }


def metric_arrays(s0, delta, alternate_delta, posterior, weights, context, alternate):
    q = np.clip(1.0 - posterior.sum(axis=1), 0.0, 1.0)
    result = {
        "s0": s0,
        "delta_native": delta,
        "delta_counterfactual": alternate_delta,
        "counterfactual_response": alternate_delta - delta,
        "posterior": posterior,
        "unpaired_i": np.broadcast_to(q[:, None], s0.shape),
        "unpaired_j": np.broadcast_to(q[None, :], s0.shape),
        "mea_weight": weights,
    }
    for channel in range(context.shape[-1]):
        result[f"context_{channel}_native"] = context[..., channel]
        result[f"context_{channel}_counterfactual"] = alternate[..., channel]
    return result


def subgroup_names(pair: tuple[int, int], isolated_truth: set) -> list[str]:
    names = ["true_all", f"span:{span_bin(pair)}", "isolated:true" if pair in isolated_truth else "isolated:false"]
    return names


def nll_from_logz(log_z: float, scores: np.ndarray, truth: list[tuple[int, int]]) -> float:
    length = scores.shape[0]
    return float((log_z - sum(float(scores[i, j]) for i, j in truth)) / length)


def load_family_map(registry_path: Path) -> dict[str, str | None]:
    registry = json.loads(registry_path.read_text())
    path = Path(registry["artifacts"]["panel_records"]["path"])
    answer = {}
    for row in read_jsonl_gz(path):
        answer[row["sequence_id"]] = row.get("family_id")
    return answer


def validation_and_training_summary(arm_root: Path) -> dict:
    validation = [json.loads(line) for line in (arm_root / "validation_log.jsonl").read_text().splitlines() if line.strip()]
    training = [json.loads(line) for line in (arm_root / "train_log.jsonl").read_text().splitlines() if line.strip()]
    best = min((row for row in validation if row["eligible"]), key=lambda row: row["validation_loss"])
    final = validation[-1]
    windows = {}
    for name, low, high in (("pre_best", 10001, 10500), ("post_best", 10501, 11000), ("last_500", 12501, 13000)):
        selected = [row for row in training if low <= int(row["step"]) <= high]
        losses = np.asarray([float(row["loss"]) for row in selected])
        total_gradient = np.asarray([float(row["gradient_norm"]) for row in selected])
        residual_gradient = np.asarray([float(row["residual_gradient_norm"]) for row in selected])
        windows[name] = {
            "n": len(losses), "mean": float(np.mean(losses)), "median": float(np.median(losses)),
            "residual_gradient_norm_median": float(np.median(residual_gradient)),
            "residual_to_total_gradient_ratio_median": float(np.median(
                residual_gradient / np.maximum(total_gradient, 1e-30)
            )),
        }
    return {
        "best_step": best["step"],
        "best_validation_nll": best["validation_loss"],
        "stop_step": final["step"],
        "stop_validation_nll": final["validation_loss"],
        "post_best_validation_nll_change": final["validation_loss"] - best["validation_loss"],
        "training_loss_windows": windows,
        "stop_reason_satisfied": bool(final["stop"] and final["bad_evaluations"] == 5),
    }


def run_arm(args) -> None:
    from rinalmo_preflight import VOCAB, load_encoder
    from rnajepa.decision_head import FlatDecisionHead, pair_mask_from_ids
    from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
    from rnajepa.gec_context import GECResidualV1, gec_destroy_key_v1, gec_geometry_v1, require_explicit_cuda
    from rnajepa.gec_numeric_v1 import compute_gec_context_v1
    from tools.train_plan_a_gec_convergence_v1 import load_registered_records

    if os.environ.get("RN_CUDA_INSIDE") != "1":
        raise ValueError("RN_CUDA_INSIDE=1 is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = require_explicit_cuda(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f"{args.arm}.jsonl.gz"
    summary_path = output / f"{args.arm}_summary.json"
    if raw_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {raw_path} or {summary_path}")
    source_root = Path(args.source_root)
    arm_root = source_root / args.arm
    config_path = REPO / "configs/gec_four_arm_convergence_v1.json"
    config = json.loads(config_path.read_text())
    records = load_registered_records(config, "validation", 0)
    if args.limit:
        records = records[:args.limit]
    family_map = load_family_map(REPO / config["assets"]["registry"])
    best_path = arm_root / "best.pt"
    initial_path = source_root / "gec_shared_residual_init_seed0.pt"
    started = time.time()
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_phase") != "best_eligible" or checkpoint.get("n_unfrozen") != 20:
        raise ValueError("audit requires frozen terminal best-eligible checkpoint")
    parameter_updates = {}
    for name, final_tensor in checkpoint["residual"].items():
        difference = final_tensor - initial["model"][name]
        parameter_updates[name] = {
            "l2": float(difference.norm()), "max_abs": float(difference.abs().max()),
            "final_bias": float(final_tensor.reshape(-1)[0]) if name == "net.4.bias" else None,
        }
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    head = FlatDecisionHead(
        d_model=int(report["d_model"]), d_z=config["model"]["d_z"], hidden=64,
        scorer="resnet2d", chunk_size=0, resnet_blocks=4, resnet_hidden=64,
        use_turner_prior=True,
    )
    residual = GECResidualV1(config["model"]["d_z"], config["model"]["residual_hidden"])
    zero_residual = GECResidualV1(config["model"]["d_z"], config["model"]["residual_hidden"])
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    residual.load_state_dict(checkpoint["residual"], strict=True)
    zero_residual.load_state_dict(initial["model"], strict=True)
    head.use_turner_prior = False
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    residual = residual.to(device).eval()
    zero_residual = zero_residual.to(device).eval()
    del checkpoint, initial
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    historical = {}
    historical_path = arm_root / "dev_predictions.jsonl"
    if not args.limit:
        with historical_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                historical[row["sequence_id"]] = row
        if set(historical) != {row["id"] for row in records}:
            raise ValueError("historical prediction coverage mismatch")

    groups = defaultdict(GroupStats)
    family_groups = defaultdict(GroupStats)
    per_sequence = []
    zero_max_abs = 0.0
    identity_max_abs = 0.0
    prediction_mismatches = []
    native_nlls = []
    stopgrad_rows = []
    probe_indices = set(np.linspace(0, len(records) - 1, min(STOPGRAD_PROBE_COUNT, len(records)), dtype=int).tolist())
    temporary = raw_path.with_name(raw_path.name + f".tmp.{os.getpid()}")
    with torch.no_grad(), gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            sequence = record["sequence"]
            truth = sorted(tuple(pair) for pair in record["pairs"])
            truth_set = set(truth)
            isolated_truth = isolated(truth)
            length = len(sequence)
            tokens = [VOCAB.get(char, VOCAB["<unk>"]) for char in sequence]
            framed = [VOCAB["<cls>"]] + tokens + [VOCAB["<eos>"]]
            encoder_ids = torch.tensor([framed], dtype=torch.long, device=device)
            attention = torch.ones_like(encoder_ids)
            head_ids = torch.tensor([[BASE_TO_INDEX.get(char, UNK_INDEX) for char in sequence]], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = encoder(input_ids=encoder_ids, attention_mask=attention)
            hidden = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded[0]
            hidden = hidden[:, 1:length + 1, :].float().to(torch.float16).to(torch.float32)
            chunks = [head.pair_repr.cross(hidden, hidden[:, start:start + 64]) for start in range(0, length, 64)]
            pair_representation = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
            raw_scores = head.score_pair_matrix(pair_representation, head_ids, False, None, None)
            base_scores = head.calibration(raw_scores, torch.tensor([length], device=device))
            full_mask = pair_mask_from_ids(head_ids, head.min_loop).to(device)
            base_scores = torch.where(full_mask, base_scores, torch.full_like(base_scores, NEG_BIG))
            mask = torch.as_tensor(harness.valid_pair_mask(sequence), device=device)
            s0 = base_scores[0]
            native = compute_gec_context_v1(
                s0, mask, mode=args.arm,
                shuffle_key=gec_destroy_key_v1(0, 0, 0, sequence),
            )
            counterfactual_mode = "destroy_interval_shuffle" if args.arm == "ensemble_context" else "ensemble_context"
            destroyed = compute_gec_context_v1(
                s0, mask, mode=counterfactual_mode, marginals=native.marginals, log_z=native.log_z,
                shuffle_key=gec_destroy_key_v1(0, 0, 0, sequence),
            )
            geometry = gec_geometry_v1(mask, head.min_loop)
            delta = residual(pair_representation[0], native.context, geometry, mask)
            alternate_delta = residual(pair_representation[0], destroyed.context, geometry, mask)
            zero_delta = zero_residual(pair_representation[0], native.context, geometry, mask)
            zero_max_abs = max(zero_max_abs, float(zero_delta.abs().max()))
            scores = torch.where(mask, s0 + delta, torch.full_like(s0, NEG_BIG))
            if bool(mask.any()):
                identity_max_abs = max(identity_max_abs, float((scores[mask] - (s0[mask] + delta[mask])).abs().max()))
            mask_np = mask.cpu().numpy()
            log_z, posterior = harness.inside_outside(scores.cpu().numpy().astype(np.float64), mask_np)
            weights = mea_pair_weights(posterior, 1.0)
            best_values, forced_values, predicted = forced_pair_values(weights, mask_np)
            predicted_set = set(predicted)
            if not args.limit:
                expected = {tuple(pair) for pair in historical[record["id"]]["predicted_pairs"]}
                if predicted_set != expected:
                    prediction_mismatches.append(record["id"])
            native_nll = nll_from_logz(log_z, scores.cpu().numpy(), truth)
            native_nlls.append(native_nll)
            s0_np = s0.cpu().numpy().astype(np.float64)
            delta_np = delta.cpu().numpy().astype(np.float64)
            alternate_np = alternate_delta.cpu().numpy().astype(np.float64)
            context_np = native.context.cpu().numpy().astype(np.float64)
            alternate_context_np = destroyed.context.cpu().numpy().astype(np.float64)
            arrays = metric_arrays(s0_np, delta_np, alternate_np, posterior, weights, context_np, alternate_context_np)
            upper = np.triu(mask_np, k=1)
            rows, cols = np.where(upper)
            for name, values in arrays.items():
                groups["all_legal"].values[name].update(values[rows, cols])
            true_rows = []
            family = family_map[record["id"]]
            for pair in truth:
                i, j = pair
                legal = bool(mask_np[i, j])
                selected = pair in predicted_set
                weight = float(weights[i, j]) if legal else float("nan")
                gap = float(best_values[0, 0] - forced_values[i, j]) if legal else float("nan")
                label = classify_pair(selected, legal, weight)
                if legal and selected and abs(gap) > 1e-7:
                    raise AssertionError(f"selected pair has nonzero forced gap: {record['id']} {pair} {gap}")
                names = subgroup_names(pair, isolated_truth)
                for group_name in names + [f"family:{family}"]:
                    target = family_groups[family] if group_name.startswith("family:") else groups[group_name]
                    target.update_counts([label], [gap] if legal else [])
                    for metric_name, values in arrays.items():
                        target.values[metric_name].update([values[i, j]])
                true_rows.append({
                    "pair": [i, j], "span": span_bin(pair), "isolated": pair in isolated_truth,
                    "family": family, "selected": selected, "class": label,
                    "s0": float(s0_np[i, j]), "delta_native": float(delta_np[i, j]),
                    "delta_counterfactual": float(alternate_np[i, j]),
                    "counterfactual_response": float(alternate_np[i, j] - delta_np[i, j]),
                    "posterior": float(posterior[i, j]), "mea_weight": weight,
                    "mea_forced_gap": gap,
                    "context_native": context_np[i, j].tolist(),
                    "context_counterfactual": alternate_context_np[i, j].tolist(),
                })
            stopgrad = None
            if index in probe_indices:
                legal_delta = delta[mask]
                rms = float(legal_delta.square().mean().sqrt())
                if rms > 0:
                    direction = torch.where(mask, delta / rms, torch.zeros_like(delta))
                    contexts = []
                    for sign in (1.0, -1.0):
                        shifted = s0 + sign * STOPGRAD_EPSILON * direction
                        shifted_numerics = compute_gec_context_v1(
                            shifted, mask, mode=args.arm,
                            shuffle_key=gec_destroy_key_v1(0, 0, 0, sequence),
                        )
                        shifted_delta = residual(pair_representation[0], shifted_numerics.context, geometry, mask)
                        shifted_scores = torch.where(mask, shifted + shifted_delta, torch.full_like(s0, NEG_BIG))
                        stopped_scores = torch.where(mask, shifted + delta, torch.full_like(s0, NEG_BIG))
                        full_np = shifted_scores.cpu().numpy().astype(np.float64)
                        stopped_np = stopped_scores.cpu().numpy().astype(np.float64)
                        full_log_z, _ = harness.inside_outside(full_np, mask_np)
                        stopped_log_z, _ = harness.inside_outside(stopped_np, mask_np)
                        contexts.append({
                            "full_nll": nll_from_logz(full_log_z, full_np, truth),
                            "stopgrad_nll": nll_from_logz(stopped_log_z, stopped_np, truth),
                            "shifted_delta": shifted_delta.cpu().numpy().astype(np.float64),
                        })
                    full_derivative = central_difference(contexts[0]["full_nll"], contexts[1]["full_nll"], STOPGRAD_EPSILON)
                    stopped_derivative = central_difference(contexts[0]["stopgrad_nll"], contexts[1]["stopgrad_nll"], STOPGRAD_EPSILON)
                    response = (contexts[0]["shifted_delta"] - contexts[1]["shifted_delta"])[upper] / (2 * STOPGRAD_EPSILON)
                    stopgrad = {
                        "delta_rms": rms,
                        "full_directional_nll_derivative": full_derivative,
                        "stopgrad_directional_nll_derivative": stopped_derivative,
                        "omitted_feedback_directional_nll_derivative": full_derivative - stopped_derivative,
                        "delta_feedback_rms": float(np.sqrt(np.mean(np.square(response)))),
                    }
                    stopgrad_rows.append(stopgrad)
            row = {
                "sequence_id": record["id"], "sequence_sha256": record["sequence_sha256"],
                "length": length, "family": family, "native_nll": native_nll,
                "predicted_pairs": [list(pair) for pair in predicted],
                "true_pairs": true_rows, "stopgrad_probe": stopgrad,
            }
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            if (index + 1) % 100 == 0 or index + 1 == len(records):
                print(f"[residual-audit] arm={args.arm} {index + 1}/{len(records)}", flush=True)
    os.replace(temporary, raw_path)
    expected_validation = next(
        row["validation_loss"] for row in (json.loads(line) for line in (arm_root / "validation_log.jsonl").read_text().splitlines())
        if row["step"] == 10500
    )
    replay_nll = float(np.mean(native_nlls)) if native_nlls else None
    if not args.limit and abs(replay_nll - expected_validation) > 1e-6:
        raise AssertionError(f"native validation NLL replay mismatch: {replay_nll} vs {expected_validation}")
    if prediction_mismatches:
        raise AssertionError(f"native MEA replay mismatches: {prediction_mismatches[:3]}")
    summary = {
        "schema_version": SCHEMA, "arm": args.arm, "status": "COMPLETE",
        "scope": {"panel_id": "tr1c_cluster80_dev_v1", "records": len(records), "external_test_sets": False, "limited_smoke": bool(args.limit)},
        "execution": {
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
            "cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
            "elapsed_seconds": time.time() - started, "torch": torch.__version__, "cuda": torch.version.cuda,
        },
        "checkpoint": identity(best_path), "initial_residual": identity(initial_path),
        "historical_predictions": None if args.limit else identity(historical_path),
        "raw_diagnostics": identity(raw_path),
        "invariants": {
            "s1_equals_s0_plus_delta_max_abs": identity_max_abs,
            "zero_init_delta_max_abs": zero_max_abs,
            "historical_mea_prediction_mismatches": len(prediction_mismatches),
            "native_mean_nll": replay_nll, "checkpoint_validation_nll": expected_validation,
            "native_nll_abs_difference": abs(replay_nll - expected_validation),
        },
        "parameter_updates_from_shared_zero_init": parameter_updates,
        "groups": {name: group.export() for name, group in sorted(groups.items())},
        "families": {str(name): group.export() for name, group in sorted(family_groups.items(), key=lambda item: str(item[0]))},
        "stopgrad_forward_sensitivity": {
            "probe_count": len(stopgrad_rows), "epsilon": STOPGRAD_EPSILON,
            "probe_selection": f"np.linspace over length-sorted registered dev indices, at most {STOPGRAD_PROBE_COUNT}",
            "direction": "native legal-pair delta normalized to unit RMS; forward sensitivity only, not a training-gradient causal estimate",
            "aggregate": {name: OnlineStats(**{}).export() for name in []},
            "rows_sha256": hashlib.sha256(json.dumps(stopgrad_rows, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        },
        "optimization_log": validation_and_training_summary(arm_root),
    }
    for name in ("delta_rms", "full_directional_nll_derivative", "stopgrad_directional_nll_derivative", "omitted_feedback_directional_nll_derivative", "delta_feedback_rms"):
        stat = OnlineStats()
        stat.update([row[name] for row in stopgrad_rows])
        summary["stopgrad_forward_sensitivity"]["aggregate"][name] = stat.export()
    atomic_json(summary, summary_path)
    print(json.dumps({"arm": args.arm, "status": "COMPLETE", "summary": str(summary_path), "peak_gib": summary["execution"]["cuda_allocated_peak_bytes"] / 2**30}, indent=2), flush=True)


def load_true_rows(path: Path) -> dict[tuple[str, int, int], dict]:
    answer = {}
    for sequence in read_jsonl_gz(path):
        for row in sequence["true_pairs"]:
            i, j = row["pair"]
            answer[(sequence["sequence_id"], i, j)] = {**row, "sequence_id": sequence["sequence_id"]}
    return answer


def transition_summary(ensemble: dict, destroy: dict) -> dict:
    if set(ensemble) != set(destroy):
        raise ValueError("arm truth coverage mismatch")
    groups = defaultdict(lambda: defaultdict(int))
    score_deltas = defaultdict(OnlineStats)
    gap_values = defaultdict(OnlineStats)
    for key, ensemble_row in ensemble.items():
        destroy_row = destroy[key]
        family = ensemble_row["family"]
        memberships = ["all", f"family:{family}", f"span:{ensemble_row['span']}", "isolated:true" if ensemble_row["isolated"] else "isolated:false"]
        transition = f"{destroy_row['class']}->{ensemble_row['class']}"
        for membership in memberships:
            groups[membership][transition] += 1
        if destroy_row["selected"] and not ensemble_row["selected"]:
            lost = "recall_lost"
        elif ensemble_row["selected"] and not destroy_row["selected"]:
            lost = "recall_gained"
        else:
            continue
        score_deltas[lost].update([ensemble_row["mea_weight"] - destroy_row["mea_weight"]])
        score_deltas[lost + "_native_delta_change"].update([ensemble_row["delta_native"] - destroy_row["delta_native"]])
        gap_values[lost].update([ensemble_row["mea_forced_gap"]])
    return {
        "transitions": {group: dict(sorted(rows.items())) for group, rows in sorted(groups.items())},
        "mea_weight_change": {name: stat.export() for name, stat in score_deltas.items()},
        "ensemble_forced_gap_on_changed_recall": {name: stat.export() for name, stat in gap_values.items()},
        "classification_contract": "selected; otherwise nonpositive_score_margin iff legal MEA weight <=0; positive_weight_mea_competition iff legal MEA weight >0; illegal labels separate",
    }


def summarize(args) -> None:
    output = Path(args.output)
    source_root = Path(args.source_root)
    ensemble_path = output / "ensemble_context.jsonl.gz"
    destroy_path = output / "destroy_interval_shuffle.jsonl.gz"
    ensemble_summary = json.loads((output / "ensemble_context_summary.json").read_text())
    destroy_summary = json.loads((output / "destroy_interval_shuffle_summary.json").read_text())
    for summary in (ensemble_summary, destroy_summary):
        if summary["scope"]["limited_smoke"] or summary["scope"]["records"] != 9143:
            raise ValueError("comparison requires complete registered dev diagnostics")
    ensemble = load_true_rows(ensemble_path)
    destroy = load_true_rows(destroy_path)
    comparison = {
        "schema_version": SCHEMA, "status": "COMPLETE", "scope": "registered TR1c cluster80 dev only; no external test",
        "independent_seed0_failure_remains": True,
        "causal_inference_prohibited": "Frozen forward associations and counterfactual decoder interventions do not identify training causation.",
        "recall_transitions_destroy_to_ensemble": transition_summary(ensemble, destroy),
        "native_vs_counterfactual_context": {},
        "validation_logs": {},
    }
    for name, summary in (("ensemble_context", ensemble_summary), ("destroy_interval_shuffle", destroy_summary)):
        comparison["native_vs_counterfactual_context"][name] = {
            group: summary["groups"][group]["values"] for group in ("true_all", "isolated:true", "span:ge128")
        }
        comparison["validation_logs"][name] = validation_and_training_summary(source_root / name)
    atomic_json(comparison, output / "comparison.json")
    inputs = {}
    for arm in ARMS:
        root = source_root / arm
        for name in ("best.pt", "dev_predictions.jsonl", "validation_log.jsonl", "train_log.jsonl"):
            inputs[f"{arm}:{name}"] = identity(root / name)
    inputs["gec_shared_residual_init_seed0.pt"] = identity(source_root / "gec_shared_residual_init_seed0.pt")
    outputs = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json" and not path.name.startswith("."):
            outputs[path.name] = identity(path)
    manifest = {
        "schema_version": SCHEMA, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "script": identity(Path(__file__).resolve()), "inputs": inputs, "outputs": outputs,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "created_utc": datetime.now(timezone.utc).isoformat(), "external_test_sets": False,
    }
    atomic_json(manifest, output / "artifact_manifest.json")
    print(json.dumps({"status": "COMPLETE", "comparison": str(output / 'comparison.json')}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", default="/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summarize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        summarize(args)
    else:
        if args.arm is None:
            raise ValueError("--arm is required unless --summarize")
        run_arm(args)


if __name__ == "__main__":
    main()
