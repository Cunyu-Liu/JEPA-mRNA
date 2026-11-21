#!/usr/bin/env python
"""Independent identity and numerical verification for the RAW_SCORE matrix.

This tool deliberately does not import the evaluator. It reloads every exported
score matrix, decodes again with the frozen C2 identity, independently aggregates
pair/stem/raw endpoints, and replays training order, unfreeze and checkpoint
identity before accepting a pilot verdict.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from pair_decode_common import decode_mea  # noqa: E402
from rnajepa.harness import inside_outside, valid_pair_mask  # noqa: E402
from rnajepa.prior_alignment import tensor_state_sha256  # noqa: E402
from rnajepa.prior_structure_metrics import pair_and_stem_counts, prf  # noqa: E402
from rnajepa.raw_score_matrix import (ALL_ARMS, NEW_TRAINING_ARMS,  # noqa: E402
                                      capped_unfreeze_blocks, load_raw_score_contract)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def same_number(actual, expected, tolerance=1e-11) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if isinstance(expected, (bool, int, str)):
        return actual == expected
    return math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=tolerance)


def assert_nested_close(actual, expected, prefix="") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"metric keys differ at {prefix}")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{prefix}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"list length differs at {prefix}")
        for index, value in enumerate(expected):
            assert_nested_close(actual[index], value, f"{prefix}[{index}]")
    elif not same_number(actual, expected):
        raise ValueError(f"metric mismatch at {prefix}: {actual!r} != {expected!r}")


def load_records(path: str | Path, limit: int = 0):
    records = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            seq = str(row["seq"]).upper().replace("T", "U")
            records.append((hashlib.sha256(seq.encode()).hexdigest(), seq,
                            {tuple(sorted(map(int, pair))) for pair in row.get("pairs", [])}))
    return records[:limit] if limit else records


def control_path(contract: dict, key: str, seed: int) -> Path:
    value = Path(contract["arms"]["full_control"][key])
    if seed != 0:
        value = Path(str(value).replace("seed0_backbone_lr_1e-6",
                                        f"seed{seed}_backbone_lr_1e-6"))
    return value


def export_dir(contract: dict, arm: str, seed: int, stage: Path) -> Path:
    if arm == "full_control":
        return control_path(contract, "export", seed)
    return stage / arm / "export_train_off"


def checkpoint_path(contract: dict, arm: str, seed: int, stage: Path, steps: int) -> Path:
    if arm == "full_control":
        return control_path(contract, "checkpoint", seed)
    return stage / arm / "snapshots" / f"{arm}_step{steps}.pt"


def independent_raw_metrics(score_matrix: np.ndarray, mask: np.ndarray, truth: set[tuple[int, int]],
                            targets=(0.8, 0.9)) -> dict:
    """Group exact score ties, then integrate the right-continuous PR staircase."""
    rows, cols = np.where(np.triu(np.asarray(mask, dtype=bool), k=1))
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    scores = np.asarray(score_matrix, dtype=np.float64)[rows, cols]
    labels = np.fromiter(((int(i), int(j)) in truth for i, j in zip(rows, cols)),
                         dtype=bool, count=len(rows))
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("legal raw ranking must be non-empty and finite")
    rank = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[rank], labels[rank]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    tp = np.cumsum(sorted_labels)[ends].astype(np.int64)
    fp = ends + 1 - tp
    positives = int(sorted_labels.sum())
    recall = tp.astype(np.float64) / positives if positives else np.zeros(len(tp), dtype=np.float64)
    precision = tp.astype(np.float64) / (tp + fp)
    result = {
        "auprc": float(np.sum(np.diff(np.r_[0.0, recall]) * precision)),
        "n_positives": positives,
        "n_candidates": int(len(scores)),
    }
    for target in targets:
        feasible = precision >= target
        result[f"recall_at_precision_{target:.1f}"] = (
            float(np.max(recall[feasible])) if np.any(feasible) else None)
    return result


def micro_raw(score_parts: list[np.ndarray], label_parts: list[np.ndarray]) -> dict:
    scores = np.concatenate(score_parts)
    labels = np.concatenate(label_parts)
    positives = int(labels.sum())
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    tp = np.cumsum(labels)[ends].astype(np.int64)
    fp = ends + 1 - tp
    recall = tp.astype(np.float64) / positives if positives else np.zeros(len(tp), dtype=np.float64)
    precision = tp.astype(np.float64) / (tp + fp)
    result = {"micro_auprc": float(np.sum(np.diff(np.r_[0.0, recall]) * precision)),
              "micro_fixed_precision_feasible": {}}
    for target in (0.8, 0.9):
        feasible = precision >= target
        value = float(np.max(recall[feasible])) if np.any(feasible) else None
        result[f"micro_recall_at_precision_{target:.1f}"] = value
        result["micro_fixed_precision_feasible"][f"{target:.1f}"] = value is not None
    return result


def aggregate(rows: list[dict], score_parts: list[np.ndarray], label_parts: list[np.ndarray],
              checkpoint_sha256: str) -> dict:
    output = {}
    for level in ("pair", "stem"):
        totals = {key: sum(int(row[level][key]) for row in rows) for key in ("tp", "fp", "fn")}
        for key, value in prf(**totals).items():
            output[f"micro_{level}_{key}"] = value
        for metric in ("precision", "recall", "f1"):
            output[f"macro_{level}_{metric}"] = float(np.mean([
                prf(row[level]["tp"], row[level]["fp"], row[level]["fn"])[metric] for row in rows]))
    output.update(micro_raw(score_parts, label_parts))
    output["macro_auprc"] = float(np.mean([row["raw"]["auprc"] for row in rows]))
    for target in (0.8, 0.9):
        key = f"recall_at_precision_{target:.1f}"
        values = [row["raw"][key] for row in rows]
        output[f"macro_{key}"] = float(np.mean([0.0 if value is None else value for value in values]))
        output[f"macro_feasible_fraction_precision_{target:.1f}"] = float(
            np.mean([value is not None for value in values]))
    output["n_sequences"] = len(rows)
    output["export_checkpoint_sha256"] = checkpoint_sha256
    return output


def expected_arm_contract_identity(contract, contract_sha256, arm):
    """Allow only the explicitly declared v1->v2 unchanged-arm inheritance."""
    migration = contract.get("contract_migration", {})
    inherited = migration.get("inherited_run_contract_sha256", {})
    if arm in inherited:
        if (migration.get("previous_contract_id") != "raw_score_mechanism_4arm_v1"
                or not migration.get("previous_contract_sha256")
                or arm not in migration.get("unchanged_arms", [])
                or migration.get("formal_pilot_vl0_evaluation_before_migration") is not False):
            raise ValueError(f"invalid contract migration identity for {arm}")
        return migration["previous_contract_id"], inherited[arm]
    return contract["contract_id"], contract_sha256


def verify_run_and_checkpoint(contract: dict, contract_sha256: str, arm: str, seed: int,
                              stage: Path, steps: int, mode: str, initial_encoder=None) -> dict:
    run_dir = stage / arm
    meta = read_json(run_dir / "run_meta.json")
    prior = meta["prior_alignment"]
    arm_contract = contract["arms"][arm]
    expected_contract_id, expected_contract_sha256 = expected_arm_contract_identity(
        contract, contract_sha256, arm)
    schedule = meta["schedule"]
    terminal_unfrozen = capped_unfreeze_blocks(
        steps, schedule["warmup_head_steps"], schedule["unfreeze_every"],
        schedule["unfreeze_per_step"], schedule["n_blocks"], schedule["max_unfrozen_blocks"])
    objective = meta["objective"]
    method = arm_contract.get("marginal_vjp_method", "central_difference")
    objective_ok = (
        objective["lambda_marginal_bce"] == arm_contract["lambda_marginal_bce"]
        and objective.get("marginal_vjp_method", "central_difference") == method)
    if method == "central_difference":
        objective_ok = objective_ok and objective["marginal_vjp_epsilon"] == arm_contract["marginal_vjp_epsilon"]
    elif method == "exact_tangent":
        objective_ok = (objective_ok and objective["marginal_vjp_epsilon"] is None
                        and objective.get("marginal_backward") == "exact_forward_mode_tangent_float64")
    else:
        raise ValueError(f"unknown contract VJP method for {arm}")
    checks = {
        "completed": meta["status"] == "completed" and meta["stop_reason"] == "step_cap",
        "steps": meta["steps_completed"] == steps and not meta["early_stopped"] and not meta["converged"],
        "no_divergence": not meta["divergence_reason"],
        "arm": prior["raw_score_arm"] == arm and prior["mode"] == mode,
        "contract": (prior["contract_id"] == expected_contract_id
                     and prior["contract_sha256"] == expected_contract_sha256),
        "trajectory": prior["training_trajectory"] == "train_off" and prior["train_turner"] == "off",
        "schedule": schedule["steps"] == steps and schedule["max_unfrozen_blocks"] == arm_contract["max_unfrozen_blocks"],
        "objective": objective_ok,
        "terminal_unfrozen": str(meta["trainable_parameter_scope"]).startswith(f"{terminal_unfrozen}/33 "),
        "external_closed": not prior["old_external_panels_used"],
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"run identity failed for {arm}: {failed}")

    batches = read_jsonl(run_dir / "batch_order.jsonl")
    ledger = read_jsonl(run_dir / "compute_ledger.jsonl")
    losses = read_jsonl(run_dir / "train_log.jsonl")
    losses_by_step = {row["step"]: row for row in losses}
    expected_loss_steps = {1, *range(25, steps + 1, 25)}
    if (len(batches) != steps or len(ledger) != steps
            or len(losses_by_step) != len(losses) or set(losses_by_step) != expected_loss_steps):
        raise ValueError(f"incomplete or unexpected training logs for {arm}")
    for step, (batch, row) in enumerate(zip(batches, ledger), 1):
        loss = losses_by_step.get(step)
        expected = capped_unfreeze_blocks(
            step, schedule["warmup_head_steps"], schedule["unfreeze_every"],
            schedule["unfreeze_per_step"], schedule["n_blocks"], schedule["max_unfrozen_blocks"])
        if (batch["step"] != step or row["step"] != step
                or row["sequence_sha256"] != batch["sequences"]
                or row["unfrozen"] != expected
                or not row["optimizer_step"] or row["divergence_reason"]):
            raise ValueError(f"training replay failed for {arm} at step {step}")
        if loss is not None and (
                loss["unfrozen"] != expected or loss["divergence_reason"]
                or not math.isfinite(loss["loss"]) or not math.isfinite(loss["gnorm"])
                or not math.isfinite(loss["terms"]["nll"])):
            raise ValueError(f"logged loss replay failed for {arm} at step {step}")
        if arm == "marginal" and loss is not None and not math.isfinite(loss["terms"]["marginal_bce"]):
            raise ValueError(f"non-finite marginal objective for {arm} at step {step}")

    checkpoint = checkpoint_path(contract, arm, seed, stage, steps)
    checkpoint_sha256 = sha256_file(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_identity = state["prior_alignment_checkpoint"]
    if (state["step"] != steps or checkpoint_identity["step"] != steps
            or checkpoint_identity["actual_unfrozen_blocks"] != terminal_unfrozen
            or state["meta"]["prior_alignment"]["raw_score_arm"] != arm
            or state["meta"]["prior_alignment"]["contract_sha256"] != expected_contract_sha256):
        raise ValueError(f"checkpoint identity failed for {arm}")
    encoder = state["model"]["encoder"]
    if tensor_state_sha256(encoder) != checkpoint_identity["backbone_update_sha256"]:
        raise ValueError(f"checkpoint tensor hash failed for {arm}")
    changed_outside = []
    if initial_encoder is not None:
        allowed_layers = set(range(33 - terminal_unfrozen, 33))
        for name, tensor in initial_encoder.items():
            match = re.search(r"encoder\.layer\.(\d+)\.", name)
            allowed = match is not None and int(match.group(1)) in allowed_layers
            if not allowed and not torch.equal(tensor.cpu(), encoder[name].cpu()):
                changed_outside.append(name)
        if changed_outside:
            raise ValueError(f"unauthorized backbone updates for {arm}: {changed_outside[:8]}")
    return {"run_meta": meta, "batches": batches, "checkpoint_sha256": checkpoint_sha256,
            "terminal_unfrozen": terminal_unfrozen, "changed_outside_allowed_scope": changed_outside}


def direction_gate(candidate: dict, control: dict, rules: dict):
    keys = ("micro_auprc", "micro_pair_f1", "macro_pair_f1", "micro_pair_precision",
            "macro_pair_precision", "micro_stem_f1", "macro_stem_f1")
    deltas = {key: candidate[key] - control[key] for key in keys}
    key = "micro_recall_at_precision_0.8"
    deltas[key] = None if candidate[key] is None or control[key] is None else candidate[key] - control[key]
    checks = {
        "micro_auprc_strictly_positive": deltas["micro_auprc"] > rules["micro_auprc_delta_min_exclusive"],
        "micro_recall_at_precision_0.8_strictly_positive": (
            deltas[key] is not None and deltas[key] > rules["micro_recall_at_precision_0.8_delta_min_exclusive"]),
        "micro_pair_f1_noninferior": deltas["micro_pair_f1"] >= rules["micro_pair_f1_delta_min_inclusive"],
        "macro_pair_f1_noninferior": deltas["macro_pair_f1"] >= rules["macro_pair_f1_delta_min_inclusive"],
        "micro_pair_precision_preserved": deltas["micro_pair_precision"] >= rules["micro_pair_precision_delta_min_inclusive"],
        "macro_pair_precision_preserved": deltas["macro_pair_precision"] >= rules["macro_pair_precision_delta_min_inclusive"],
        "micro_stem_f1_preserved": deltas["micro_stem_f1"] >= rules["micro_stem_f1_delta_min_inclusive"],
        "macro_stem_f1_preserved": deltas["macro_stem_f1"] >= rules["macro_stem_f1_delta_min_inclusive"],
        "at_least_one_pair_f1_strictly_positive": deltas["micro_pair_f1"] > 0 or deltas["macro_pair_f1"] > 0,
    }
    return deltas, checks, all(checks.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=9600)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--skip-training-replay", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite verification: {args.out}")
    contract = load_raw_score_contract(args.contract)
    contract_sha256 = sha256_file(args.contract)
    verdict = read_json(args.evaluation / "pilot_verdict.json")
    if (verdict["contract_sha256"] != contract_sha256 or verdict["seed"] != args.seed
            or verdict["n_sequences"] != (args.limit or 196)):
        raise ValueError("verdict contract/seed/evaluation identity mismatch")
    if verdict["external_evaluation_allowed"] or verdict["candidate_selected"] or verdict["sota_claim"]:
        raise ValueError("scientific boundary was opened by the verdict")
    rows_expected = {(row["arm"], row["sequence_sha256"]): row
                     for row in read_jsonl(args.evaluation / "per_sequence.jsonl")}
    records = load_records(contract["training"]["evaluation_data"], args.limit)
    arms = ("full_control",) if args.control_only else ALL_ARMS
    mode = "smoke" if args.limit else ("pilot" if args.seed == 0 else "full")
    shared_init = Path(contract["training"]["shared_init_template"].format(seed=args.seed))
    initial_encoder = None
    if not args.skip_training_replay and not args.control_only:
        initial = torch.load(shared_init, map_location="cpu", weights_only=False)
        initial_encoder = initial["model"]["encoder"]
    run_identities = {}
    if not args.skip_training_replay:
        for arm in NEW_TRAINING_ARMS if not args.control_only else ():
            run_identities[arm] = verify_run_and_checkpoint(
                contract, contract_sha256, arm, args.seed, args.stage, args.steps, mode, initial_encoder)
        if not args.control_only:
            reference_batches = run_identities["frozen"]["batches"]
            for arm in ("limited4", "marginal"):
                if run_identities[arm]["batches"] != reference_batches:
                    raise ValueError(f"batch order differs between frozen and {arm}")
            control_meta_path = control_path(contract, "run_meta", args.seed)
            control_batches = read_jsonl(control_meta_path.parent / "batch_order.jsonl")[:args.steps]
            if control_batches != reference_batches:
                raise ValueError("new-arm batch order differs from the reused full control")
    del initial_encoder

    recomputed = {}
    for arm in arms:
        export = export_dir(contract, arm, args.seed, args.stage)
        metadata = read_json(export / "export_metadata.json")
        checkpoint = checkpoint_path(contract, arm, args.seed, args.stage,
                                     9600 if arm == "full_control" else args.steps)
        checkpoint_sha256 = sha256_file(checkpoint)
        if (metadata["checkpoint"]["sha256"] != checkpoint_sha256
                or metadata["seed"] != args.seed
                or metadata["checkpoint_step"] != (9600 if arm == "full_control" else args.steps)
                or metadata["train_trajectory"] != "train_off" or metadata["old_external_panels_used"]):
            raise ValueError(f"export/checkpoint identity mismatch for {arm}")
        if arm != "full_control":
            _expected_id, expected_export_contract_sha256 = expected_arm_contract_identity(
                contract, contract_sha256, arm)
            if metadata["contract_sha256"] != expected_export_contract_sha256:
                raise ValueError(f"new-arm export contract mismatch for {arm}")
        arm_rows, score_parts, label_parts = [], [], []
        for sequence_hash, seq, truth in records:
            path = export / "bprna_vl0" / f"{sequence_hash}.npz"
            with np.load(path, allow_pickle=False) as data:
                if str(data["seq"]) != seq or str(data["sequence_sha256"]) != sequence_hash:
                    raise ValueError(f"score/record mismatch for {arm} {sequence_hash}")
                if data["checkpoint_sha256"].item() != checkpoint_sha256:
                    raise ValueError(f"per-sequence checkpoint mismatch for {arm} {sequence_hash}")
                scores = np.asarray(data["plana_off"], dtype=np.float64)
                np_truth = {tuple(sorted(map(int, pair))) for pair in data["gt_pairs"].tolist()}
            if scores.shape != (len(seq), len(seq)) or np_truth != truth:
                raise ValueError(f"score/truth shape mismatch for {arm} {sequence_hash}")
            mask = valid_pair_mask(seq)
            if not np.isfinite(scores[mask]).all():
                raise ValueError(f"non-finite legal raw score for {arm} {sequence_hash}")
            log_z, posterior = inside_outside(scores, mask)
            if not math.isfinite(log_z) or not np.isfinite(posterior).all():
                raise ValueError(f"non-finite posterior for {arm} {sequence_hash}")
            predicted = set(decode_mea(posterior, mask, 1.0))
            row = {"seed": args.seed, "arm": arm, "sequence_sha256": sequence_hash,
                   "decoder": "MEA", "temperature": 1.0, "gamma": 1.0,
                   "n_truth_pairs": len(truth), "n_predicted_pairs": len(predicted)}
            row.update(pair_and_stem_counts(predicted, truth))
            row["raw"] = independent_raw_metrics(scores, mask, truth)
            expected = rows_expected.get((arm, sequence_hash))
            if expected is None:
                raise ValueError(f"missing per-sequence verdict row for {arm} {sequence_hash}")
            assert_nested_close(expected, row, f"{arm}.{sequence_hash}")
            arm_rows.append(row)
            rows, cols = np.where(np.triu(mask, k=1))
            order = np.lexsort((cols, rows))
            rows, cols = rows[order], cols[order]
            score_parts.append(scores[rows, cols])
            label_parts.append(np.fromiter(((int(i), int(j)) in truth for i, j in zip(rows, cols)),
                                           dtype=bool, count=len(rows)))
        recomputed[arm] = aggregate(arm_rows, score_parts, label_parts, checkpoint_sha256)
        assert_nested_close(verdict["arms"][arm], recomputed[arm], arm)
        print(f"[verified-metrics] {arm}: {len(arm_rows)} sequences", flush=True)

    if not args.control_only:
        if args.limit:
            if verdict["decision"] != "SMOKE_ONLY_NO_SCIENTIFIC_VERDICT" or verdict["promoted_arms"]:
                raise ValueError("smoke was incorrectly assigned a scientific verdict")
            comparisons = {}
        else:
            comparisons = {}
            for arm in NEW_TRAINING_ARMS:
                deltas, checks, passed = direction_gate(
                    recomputed[arm], recomputed["full_control"],
                    contract["pilot"]["per_arm_direction_gate"])
                comparisons[arm] = {"delta_vs_full_control": deltas, "checks": checks,
                                    "direction_passed": passed}
            assert_nested_close(verdict["comparisons"], comparisons, "comparisons")
            promoted = [arm for arm in NEW_TRAINING_ARMS if comparisons[arm]["direction_passed"]]
            decision = "PILOT_DIRECTION_CONFIRMED" if promoted else "STOP_RAW_SCORE_MATRIX_NO_DIRECTION"
            if verdict["promoted_arms"] != promoted or verdict["decision"] != decision:
                raise ValueError("independently recomputed pilot decision differs")
        provenance_path = args.provenance or (args.stage / "provenance.json")
        if not args.limit:
            provenance = read_json(provenance_path)
            if (provenance["CONTRACT_SHA256"] != contract_sha256
                    or provenance["decision"] != verdict["decision"]
                    or provenance["promoted_arms"] != verdict["promoted_arms"]
                    or provenance["external_evaluation_allowed"]):
                raise ValueError("provenance does not match the verified verdict")
        comparisons_out = comparisons
    else:
        comparisons_out = {}
    output = {
        "schema_version": 1, "status": "PASS",
        "verification": "independent_full_metric_and_identity_replay",
        "contract_id": contract["contract_id"], "contract_sha256": contract_sha256,
        "seed": args.seed, "steps": args.steps, "n_sequences": len(records),
        "arms": list(arms), "training_replay_performed": not args.skip_training_replay and not args.control_only,
        "metrics": recomputed, "comparisons": comparisons_out,
        "decision": verdict["decision"], "promoted_arms": verdict["promoted_arms"],
        "external_evaluation_allowed": False, "sota_claim": False,
        "verdict_sha256": sha256_file(args.evaluation / "pilot_verdict.json"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "decision": output["decision"],
                      "promoted_arms": output["promoted_arms"],
                      "verdict_sha256": output["verdict_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
