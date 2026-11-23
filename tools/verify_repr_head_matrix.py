#!/usr/bin/env python
"""Independent identity and numerical verification for the representation/head matrix.

Deliberately does not import the evaluator's aggregation. Reloads every exported
score matrix, decodes again with the frozen MEA identity, independently aggregates
pair/stem/raw endpoints, and replays training order, unfreeze, architecture and
checkpoint identity for all four retrained arms before accepting a pilot verdict.
Reuses the audited RAW_SCORE numerical primitives.
"""
from __future__ import annotations

import argparse
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
from rnajepa.prior_structure_metrics import pair_and_stem_counts  # noqa: E402
from rnajepa.raw_score_matrix import capped_unfreeze_blocks  # noqa: E402
from rnajepa.repr_head_matrix import (CONTROL_ARM, STOP_DECISION, CONFIRM_DECISION,  # noqa: E402
                                      active_repr_head_arms, load_repr_head_contract)
from verify_raw_score_matrix import (sha256_file, read_json, read_jsonl,  # noqa: E402
                                     assert_nested_close, load_records,
                                     independent_raw_metrics, aggregate, direction_gate)


def verify_run_and_checkpoint_rh(contract, contract_sha256, arm, seed, stage, steps, mode,
                                 initial_encoder=None):
    run_dir = stage / arm
    meta = read_json(run_dir / "run_meta.json")
    prior = meta["prior_alignment"]
    arm_contract = contract["arms"][arm]
    schedule = meta["schedule"]
    terminal_unfrozen = capped_unfreeze_blocks(
        steps, schedule["warmup_head_steps"], schedule["unfreeze_every"],
        schedule["unfreeze_per_step"], schedule["n_blocks"], schedule["max_unfrozen_blocks"])
    objective = meta["objective"]
    checks = {
        "completed": meta["status"] == "completed" and meta["stop_reason"] == "step_cap",
        "steps": meta["steps_completed"] == steps and not meta["early_stopped"] and not meta["converged"],
        "no_divergence": not meta["divergence_reason"],
        "arm": prior["raw_score_arm"] == arm and prior["mode"] == mode,
        "contract": (prior["contract_id"] == contract["contract_id"]
                     and prior["contract_sha256"] == contract_sha256),
        "trajectory": prior["training_trajectory"] == "train_off" and prior["train_turner"] == "off",
        "schedule": schedule["steps"] == steps and schedule["max_unfrozen_blocks"] == arm_contract["max_unfrozen_blocks"],
        "objective_nll_only": objective["lambda_marginal_bce"] == arm_contract["lambda_marginal_bce"] == 0.0,
        "terminal_unfrozen": str(meta["trainable_parameter_scope"]).startswith(f"{terminal_unfrozen}/33 "),
        "external_closed": not prior["old_external_panels_used"],
        "head_init_fresh": meta.get("head_init") == "fresh" and arm_contract["head_init"] == "fresh",
        "architecture": (meta.get("d_z") == arm_contract["d_z"]
                         and meta.get("resnet_blocks") == arm_contract["resnet_blocks"]
                         and meta.get("resnet_hidden") == arm_contract["resnet_hidden"]),
        "compute_policy": meta.get("resnet_per_block_checkpoint") == contract["training"].get("resnet_per_block_checkpoint"),
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

    checkpoint = stage / arm / "snapshots" / f"{arm}_step{steps}.pt"
    checkpoint_sha256 = sha256_file(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ci = state["prior_alignment_checkpoint"]
    if (state["step"] != steps or ci["step"] != steps
            or ci["actual_unfrozen_blocks"] != terminal_unfrozen
            or state["meta"]["prior_alignment"]["raw_score_arm"] != arm
            or state["meta"]["prior_alignment"]["contract_sha256"] != contract_sha256):
        raise ValueError(f"checkpoint identity failed for {arm}")
    encoder = state["model"]["encoder"]
    if tensor_state_sha256(encoder) != ci["backbone_update_sha256"]:
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
    parser.add_argument("--skip-training-replay", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite verification: {args.out}")
    contract = load_repr_head_contract(args.contract)
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
    mode = "smoke" if args.limit else ("pilot" if args.seed == 0 else "full")
    shared_init = Path(contract["training"]["shared_init_template"].format(seed=args.seed))
    initial_encoder = None
    if not args.skip_training_replay:
        initial = torch.load(shared_init, map_location="cpu", weights_only=False)
        initial_encoder = initial["model"]["encoder"]
    arms = active_repr_head_arms(contract, args.seed)
    variant_arms = arms[1:]
    run_identities = {}
    if not args.skip_training_replay:
        for arm in arms:
            run_identities[arm] = verify_run_and_checkpoint_rh(
                contract, contract_sha256, arm, args.seed, args.stage, args.steps, mode, initial_encoder)
        reference_batches = run_identities[CONTROL_ARM]["batches"]
        for arm in variant_arms:
            if run_identities[arm]["batches"] != reference_batches:
                raise ValueError(f"batch order differs between {CONTROL_ARM} and {arm}")
    del initial_encoder

    recomputed = {}
    for arm in arms:
        export = args.stage / arm / "export_train_off"
        metadata = read_json(export / "export_metadata.json")
        checkpoint = args.stage / arm / "snapshots" / f"{arm}_step{args.steps}.pt"
        checkpoint_sha256 = sha256_file(checkpoint)
        if (metadata["checkpoint"]["sha256"] != checkpoint_sha256
                or metadata["seed"] != args.seed
                or metadata["checkpoint_step"] != args.steps
                or metadata["train_trajectory"] != "train_off" or metadata["old_external_panels_used"]):
            raise ValueError(f"export/checkpoint identity mismatch for {arm}")
        if metadata["contract_sha256"] != contract_sha256:
            raise ValueError(f"export contract mismatch for {arm}")
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

    if args.limit:
        if verdict["decision"] != "SMOKE_ONLY_NO_SCIENTIFIC_VERDICT" or verdict["promoted_arms"]:
            raise ValueError("smoke was incorrectly assigned a scientific verdict")
        comparisons = {}
    else:
        comparisons = {}
        for arm in variant_arms:
            deltas, checks, passed = direction_gate(
                recomputed[arm], recomputed[CONTROL_ARM],
                contract["pilot"]["per_arm_direction_gate"])
            comparisons[arm] = {"delta_vs_control_matched": deltas, "checks": checks,
                                "direction_passed": passed}
        assert_nested_close(verdict["comparisons"], comparisons, "comparisons")
        promoted = [arm for arm in variant_arms if comparisons[arm]["direction_passed"]]
        decision = CONFIRM_DECISION if promoted else STOP_DECISION
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
    output = {
        "schema_version": 1, "status": "PASS",
        "verification": "independent_full_metric_and_identity_replay",
        "contract_id": contract["contract_id"], "contract_sha256": contract_sha256,
        "seed": args.seed, "steps": args.steps, "n_sequences": len(records),
        "arms": list(arms), "training_replay_performed": not args.skip_training_replay,
        "metrics": recomputed, "comparisons": comparisons,
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