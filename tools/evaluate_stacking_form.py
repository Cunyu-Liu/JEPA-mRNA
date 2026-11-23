#!/usr/bin/env python
"""Evaluate the seed0 stacking-form matrix and fixed-control target subgroup."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "tools"))
from pair_decode_common import decode_mea
from rnajepa.harness import inside_outside, valid_pair_mask
from rnajepa.prior_alignment import sha256_file
from rnajepa.prior_structure_metrics import pair_and_stem_counts
from rnajepa.raw_score_metrics import (PRECISION_TARGETS, legal_pair_coordinates,
                                       raw_ranking_metrics, sequence_raw_metrics)
from rnajepa.stacking_form_matrix import ARMS, CONTROL_ARM, load_stacking_form_contract
from evaluate_raw_score_matrix import (_summarize_decoded, _macro_raw, _load_records,
                                       _verify_export_metadata, _direction_gate)
from analyze_pair_stacking_fn import pair_environment


def target_pairs(truth, control_predicted):
    output = set()
    for pair in truth:
        if pair_environment(pair, truth) != "helix_boundary":
            continue
        i, j = pair
        if {(i - 1, j + 1), (i + 1, j - 1)} & truth & control_predicted:
            output.add(pair)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=9600)
    args = parser.parse_args()
    contract = load_stacking_form_contract(args.contract)
    records = _load_records(contract["training"]["evaluation_data"], args.limit)
    args.out.mkdir(parents=True, exist_ok=False)
    per_arm, all_rows, targets = {}, {}, {}
    for arm in ARMS:
        export_dir = args.stage / arm / "export_train_off"
        metadata = _verify_export_metadata(export_dir, 0, not args.limit, args.steps, len(records))
        arm_rows, score_parts, label_parts = [], [], []
        for index, (sequence_hash, seq, truth) in enumerate(records):
            with np.load(export_dir / "bprna_vl0" / f"{sequence_hash}.npz", allow_pickle=False) as data:
                if str(data["seq"]) != seq:
                    raise ValueError("score/record mismatch")
                scores = np.asarray(data["plana_off"], dtype=np.float64)
                np_truth = {tuple(sorted(map(int, pair))) for pair in data["gt_pairs"].tolist()}
            if np_truth != truth:
                raise ValueError("truth mismatch")
            mask = valid_pair_mask(seq)
            log_z, posterior = inside_outside(scores, mask)
            if not math_isfinite(log_z, posterior):
                raise ValueError("non-finite inside-outside")
            predicted = set(decode_mea(posterior, mask, 1.0))
            if arm == CONTROL_ARM:
                targets[sequence_hash] = target_pairs(truth, predicted)
            target = targets[sequence_hash]
            row = {"seed": 0, "arm": arm, "sequence_sha256": sequence_hash,
                   "decoder": "MEA", "temperature": 1.0, "gamma": 1.0,
                   "n_truth_pairs": len(truth), "n_predicted_pairs": len(predicted),
                   "target_supported_boundary_n": len(target),
                   "target_supported_boundary_tp": len(target & predicted)}
            row.update(pair_and_stem_counts(predicted, truth))
            row["raw"] = sequence_raw_metrics(scores, mask, truth)
            arm_rows.append(row)
            rows, cols = legal_pair_coordinates(mask)
            labels = np.asarray([(int(i), int(j)) in truth for i, j in zip(rows, cols)], dtype=bool)
            order = np.lexsort((cols, rows))
            score_parts.append(scores[rows, cols][order]); label_parts.append(labels[order])
        metrics = _summarize_decoded(arm_rows)
        micro_raw = raw_ranking_metrics(np.concatenate(score_parts), np.concatenate(label_parts))
        metrics["micro_auprc"] = micro_raw["auprc"]
        for target_value in PRECISION_TARGETS:
            key = f"recall_at_precision_{target_value:.1f}"
            metrics[f"micro_{key}"] = micro_raw[key]
            metrics["micro_fixed_precision_feasible"] = metrics.get("micro_fixed_precision_feasible", {})
            metrics["micro_fixed_precision_feasible"][f"{target_value:.1f}"] = micro_raw[key] is not None
        metrics.update(_macro_raw(arm_rows))
        target_n = sum(row["target_supported_boundary_n"] for row in arm_rows)
        target_tp = sum(row["target_supported_boundary_tp"] for row in arm_rows)
        metrics["target_supported_boundary_n"] = target_n
        metrics["target_supported_boundary_recall"] = target_tp / target_n if target_n else None
        metrics["n_sequences"] = len(arm_rows)
        metrics["export_checkpoint_sha256"] = metadata["checkpoint"]["sha256"]
        per_arm[arm] = metrics
        all_rows.extend(arm_rows)
    with (args.out / "per_sequence.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    control = per_arm[CONTROL_ARM]
    deltas, checks, direction_passed = _direction_gate(
        per_arm["stacking_closure"], control, contract["pilot"]["per_arm_direction_gate"])
    target_ok = (per_arm["stacking_closure"]["target_supported_boundary_recall"]
                 > control["target_supported_boundary_recall"])
    destroy_checks = {metric: all(per_arm["stacking_closure"][metric] > per_arm[arm][metric]
                                  for arm in ("destroy_offset", "destroy_channel_shuffle"))
                      for metric in contract["pilot"]["must_beat_both_destroy_controls_on"]}
    all_checks = {**checks, "target_supported_boundary_recall_strictly_positive": target_ok,
                  **{f"beats_destroy_{key}": value for key, value in destroy_checks.items()}}
    passed = direction_passed and target_ok and all(destroy_checks.values()) and not args.limit
    decision = ("PILOT_FORM_DIRECTION_CONFIRMED" if passed else
                "SMOKE_ONLY_NO_SCIENTIFIC_VERDICT" if args.limit else "STOP_STACKING_CLOSURE_FORM_SEED0")
    verdict = {
        "schema_version": 1, "status": "PASS", "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(args.contract), "seed": 0, "n_sequences": len(records),
        "arms": per_arm, "delta_vs_local_control": deltas, "checks": all_checks,
        "direction_passed": direction_passed, "destroy_checks": destroy_checks,
        "decision": decision, "seed1_seed2_allowed": passed,
        "external_evaluation_allowed": False, "strict_ood_allowed": False,
        "candidate_selected": False, "sota_claim": False,
        "target_definition": "helix-boundary truth pairs with a true stacking neighbor predicted by local_control; fixed denominator for every arm",
        "interpretation_limit": "Seed0 VL0 model-dev only. Pass only authorizes the preregistered three-seed confirmation, never external evaluation.",
    }
    (args.out / "pilot_verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "checks": all_checks,
                      "stacking_closure": per_arm["stacking_closure"], "local_control": control}, indent=2))


def math_isfinite(log_z, posterior):
    return bool(np.isfinite(log_z) and np.isfinite(posterior).all())


if __name__ == "__main__":
    main()
