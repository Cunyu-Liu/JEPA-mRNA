#!/usr/bin/env python3
"""Decode and score the frozen RAW_SCORE four-arm mechanism matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from pair_decode_common import decode_mea  # noqa: E402
from rnajepa.harness import inside_outside, valid_pair_mask  # noqa: E402
from rnajepa.prior_alignment import sha256_file  # noqa: E402
from rnajepa.prior_structure_metrics import pair_and_stem_counts, prf  # noqa: E402
from rnajepa.raw_score_matrix import ALL_ARMS, NEW_TRAINING_ARMS, load_raw_score_contract  # noqa: E402
from rnajepa.raw_score_metrics import (PRECISION_TARGETS, legal_pair_coordinates,  # noqa: E402
                                       raw_ranking_metrics, sequence_raw_metrics)


def _macro(rows, level, metric):
    return float(np.mean([prf(row[level]["tp"], row[level]["fp"], row[level]["fn"])[metric]
                          for row in rows]))


def _summarize_decoded(rows):
    output = {}
    for level in ("pair", "stem"):
        counts = {key: int(sum(row[level][key] for row in rows)) for key in ("tp", "fp", "fn")}
        for key, value in prf(**counts).items():
            output[f"micro_{level}_{key}"] = value
        for key in ("precision", "recall", "f1"):
            output[f"macro_{level}_{key}"] = _macro(rows, level, key)
    return output


def _macro_raw(rows):
    output = {"macro_auprc": float(np.mean([row["raw"]["auprc"] for row in rows]))}
    for target in PRECISION_TARGETS:
        key = f"recall_at_precision_{target:.1f}"
        values = [row["raw"][key] for row in rows]
        output[f"macro_{key}"] = float(np.mean([0.0 if value is None else value for value in values]))
        output[f"macro_feasible_fraction_precision_{target:.1f}"] = float(
            np.mean([value is not None for value in values]))
    return output


def _load_records(path, limit=0):
    records = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            seq = str(row["seq"]).upper().replace("T", "U")
            records.append((hashlib.sha256(seq.encode()).hexdigest(), seq,
                            {tuple(sorted(map(int, pair))) for pair in row.get("pairs", [])}))
    return records[:limit] if limit else records


def _export_dir(contract, arm, seed, stage):
    if arm == "full_control":
        if seed == 0:
            return Path(contract["arms"][arm]["export"])
        return Path(str(Path(contract["arms"][arm]["export"])).replace(
            "seed0_backbone_lr_1e-6", f"seed{seed}_backbone_lr_1e-6"))
    return stage / arm / "export_train_off"


def _verify_export_metadata(export_dir, seed, new_arm, expected_step, expected_count):
    metadata = json.loads((export_dir / "export_metadata.json").read_text())
    if metadata.get("seed") != seed or metadata.get("checkpoint_step") != expected_step:
        raise ValueError(f"export identity mismatch: {export_dir}")
    if metadata.get("train_trajectory") != "train_off" or metadata.get("old_external_panels_used"):
        raise ValueError(f"export trajectory mismatch: {export_dir}")
    if metadata.get("n_selected") != expected_count or metadata.get("n_skipped_length") != 0:
        raise ValueError(f"incomplete VL0 export: {export_dir}")
    if new_arm and metadata.get("n_exported", 0) + metadata.get("n_resumed", 0) != expected_count:
        raise ValueError(f"incomplete new-arm export: {export_dir}")
    return metadata


def _verify_completion(stage, arm):
    meta_path = stage / arm / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    expected_unfrozen = {"frozen": 0, "limited4": 4, "marginal": 20}[arm]
    if (meta.get("status") != "completed" or meta.get("stop_reason") != "step_cap"
            or meta.get("steps_completed") != 9600 or meta.get("early_stopped")
            or meta.get("divergence_reason") or meta.get("converged")):
        raise ValueError(f"incomplete or diverged run: {arm}")
    if not str(meta.get("trainable_parameter_scope", "")).startswith(f"{expected_unfrozen}/33 "):
        raise ValueError(f"terminal unfreeze mismatch: {arm}")
    return meta


def _direction_gate(arm_metrics, control, rules):
    deltas = {key: arm_metrics[key] - control[key] for key in (
        "micro_auprc", "micro_pair_f1", "macro_pair_f1", "micro_pair_precision",
        "macro_pair_precision", "micro_stem_f1", "macro_stem_f1")}
    if arm_metrics["micro_recall_at_precision_0.8"] is None or control["micro_recall_at_precision_0.8"] is None:
        deltas["micro_recall_at_precision_0.8"] = None
    else:
        deltas["micro_recall_at_precision_0.8"] = (
            arm_metrics["micro_recall_at_precision_0.8"] - control["micro_recall_at_precision_0.8"])
    checks = {
        "micro_auprc_strictly_positive": deltas["micro_auprc"] > rules["micro_auprc_delta_min_exclusive"],
        "micro_recall_at_precision_0.8_strictly_positive": (
            deltas["micro_recall_at_precision_0.8"] is not None
            and deltas["micro_recall_at_precision_0.8"] > rules["micro_recall_at_precision_0.8_delta_min_exclusive"]),
        "micro_pair_f1_noninferior": deltas["micro_pair_f1"] >= rules["micro_pair_f1_delta_min_inclusive"],
        "macro_pair_f1_noninferior": deltas["macro_pair_f1"] >= rules["macro_pair_f1_delta_min_inclusive"],
        "micro_pair_precision_preserved": deltas["micro_pair_precision"] >= rules["micro_pair_precision_delta_min_inclusive"],
        "macro_pair_precision_preserved": deltas["macro_pair_precision"] >= rules["macro_pair_precision_delta_min_inclusive"],
        "micro_stem_f1_preserved": deltas["micro_stem_f1"] >= rules["micro_stem_f1_delta_min_inclusive"],
        "macro_stem_f1_preserved": deltas["macro_stem_f1"] >= rules["macro_stem_f1_delta_min_inclusive"],
        "at_least_one_pair_f1_strictly_positive": (
            deltas["micro_pair_f1"] > 0.0 or deltas["macro_pair_f1"] > 0.0),
    }
    return deltas, checks, all(checks.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=9600)
    args = parser.parse_args()
    contract = load_raw_score_contract(args.contract)
    records = _load_records(contract["training"]["evaluation_data"], args.limit)
    if not records:
        raise ValueError("empty evaluation set")
    args.out.mkdir(parents=True, exist_ok=False)
    all_rows, per_arm, export_metadata = [], {}, {}
    for arm in ALL_ARMS:
        if not args.limit and arm in NEW_TRAINING_ARMS:
            _verify_completion(args.stage, arm)
        export_dir = _export_dir(contract, arm, args.seed, args.stage)
        metadata = _verify_export_metadata(
            export_dir, args.seed, arm in NEW_TRAINING_ARMS and not args.limit,
            9600 if arm == "full_control" else args.steps,
            196 if arm == "full_control" else len(records))
        export_metadata[arm] = metadata
        arm_rows, score_parts, label_parts = [], [], []
        for index, (sequence_hash, seq, truth) in enumerate(records):
            path = export_dir / "bprna_vl0" / f"{sequence_hash}.npz"
            with np.load(path, allow_pickle=False) as data:
                if str(data["seq"]) != seq or str(data["sequence_sha256"]) != sequence_hash:
                    raise ValueError(f"score/record mismatch: {arm} {sequence_hash}")
                scores = np.asarray(data["plana_off"], dtype=np.float64)
                np_truth = {tuple(sorted(map(int, pair))) for pair in data["gt_pairs"].tolist()}
            if np_truth != truth or scores.shape != (len(seq), len(seq)):
                raise ValueError(f"invalid raw score matrix: {arm} {sequence_hash}")
            mask = valid_pair_mask(seq)
            if not np.isfinite(scores[mask]).all():
                raise ValueError(f"non-finite legal raw score: {arm} {sequence_hash}")
            log_z, posterior = inside_outside(scores, mask)
            if not np.isfinite(log_z) or not np.isfinite(posterior).all():
                raise ValueError(f"non-finite inside-outside: {arm} {sequence_hash}")
            predicted = set(decode_mea(posterior, mask, contract["evaluation"]["decoder"]["gamma"]))
            row = {"seed": args.seed, "arm": arm, "sequence_sha256": sequence_hash,
                   "decoder": "MEA", "temperature": 1.0, "gamma": 1.0,
                   "n_truth_pairs": len(truth), "n_predicted_pairs": len(predicted)}
            row.update(pair_and_stem_counts(predicted, truth))
            row["raw"] = sequence_raw_metrics(scores, mask, truth)
            arm_rows.append(row)
            rows, cols = legal_pair_coordinates(mask)
            labels = np.asarray([(int(i), int(j)) in truth for i, j in zip(rows, cols)], dtype=bool)
            order = np.lexsort((cols, rows))
            score_parts.append(scores[rows, cols][order])
            label_parts.append(labels[order])
            if index % 25 == 0:
                print(f"[raw-matrix] seed={args.seed} arm={arm} {index}/{len(records)}", flush=True)
        metrics = _summarize_decoded(arm_rows)
        micro_raw = raw_ranking_metrics(np.concatenate(score_parts), np.concatenate(label_parts))
        metrics["micro_auprc"] = micro_raw["auprc"]
        for target in PRECISION_TARGETS:
            key = f"recall_at_precision_{target:.1f}"
            metrics[f"micro_{key}"] = micro_raw[key]
            metrics["micro_fixed_precision_feasible"] = metrics.get("micro_fixed_precision_feasible", {})
            metrics["micro_fixed_precision_feasible"][f"{target:.1f}"] = micro_raw[key] is not None
        metrics.update(_macro_raw(arm_rows))
        metrics["n_sequences"] = len(arm_rows)
        metrics["export_checkpoint_sha256"] = metadata["checkpoint"]["sha256"]
        per_arm[arm] = metrics
        all_rows.extend(arm_rows)
    with (args.out / "per_sequence.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    control = per_arm["full_control"]
    comparisons = {}
    if not args.limit:
        for arm in NEW_TRAINING_ARMS:
            deltas, checks, passed = _direction_gate(
                per_arm[arm], control, contract["pilot"]["per_arm_direction_gate"])
            comparisons[arm] = {"delta_vs_full_control": deltas, "checks": checks,
                                "direction_passed": passed}
    promoted = [arm for arm in NEW_TRAINING_ARMS if comparisons.get(arm, {}).get("direction_passed")]
    decision = ("PILOT_DIRECTION_CONFIRMED" if promoted else "STOP_RAW_SCORE_MATRIX_NO_DIRECTION")
    if args.limit:
        decision = "SMOKE_ONLY_NO_SCIENTIFIC_VERDICT"
    verdict = {
        "schema_version": 1, "status": "PASS", "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(args.contract), "seed": args.seed,
        "n_sequences": len(records), "arms": per_arm, "comparisons": comparisons,
        "promoted_arms": promoted, "decision": decision,
        "external_evaluation_allowed": False, "candidate_selected": False,
        "sota_claim": False, "decoder": contract["evaluation"]["decoder"],
        "decoder_temperature_or_gamma_searched": False,
        "raw_metrics_are_auxiliary_diagnostics": True,
        "full_control_reused_not_retrained": True,
        "exports": {arm: {"path": str(_export_dir(contract, arm, args.seed, args.stage)),
                          "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                          "contract_sha256": metadata["contract_sha256"]}
                    for arm, metadata in export_metadata.items()},
    }
    (args.out / "pilot_verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "promoted_arms": promoted,
                      "arms": per_arm, "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
