#!/usr/bin/env python3
"""Decode and score the frozen representation/head capacity four-arm matrix.

Reuses the audited RAW_SCORE metric primitives (no reimplementation). The gate
baseline is the RETRAINED control_matched arm under the identical fresh-head-init
policy, not a reused full_control. control_matched vs the prior full_control
reference is reported only as an integrity sanity check, never as a promotion gate.
"""
from __future__ import annotations

import argparse
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
from rnajepa.prior_structure_metrics import pair_and_stem_counts  # noqa: E402
from rnajepa.raw_score_metrics import (PRECISION_TARGETS, legal_pair_coordinates,  # noqa: E402
                                       raw_ranking_metrics, sequence_raw_metrics)
from rnajepa.repr_head_matrix import (ALL_ARMS, CONTROL_ARM, VARIANT_ARMS,  # noqa: E402
                                      STOP_DECISION, CONFIRM_DECISION, load_repr_head_contract)
from evaluate_raw_score_matrix import (_summarize_decoded, _macro_raw, _load_records,  # noqa: E402
                                       _verify_export_metadata, _direction_gate)

# Prior-round reused full_control VL0 seed0 micro pair-F1 (raw_score_mechanism_4arm_v3
# pilot_verdict, independently rechecked). Integrity reference only.
REF_FULL_CONTROL_MICRO_PAIR_F1 = 0.827895


def _verify_completion(stage, arm):
    meta = json.loads((stage / arm / "run_meta.json").read_text())
    if (meta.get("status") != "completed" or meta.get("stop_reason") != "step_cap"
            or meta.get("steps_completed") != 9600 or meta.get("early_stopped")
            or meta.get("divergence_reason") or meta.get("converged")):
        raise ValueError(f"incomplete or diverged run: {arm}")
    if not str(meta.get("trainable_parameter_scope", "")).startswith("20/33 "):
        raise ValueError(f"terminal unfreeze mismatch: {arm}")
    if meta.get("head_init") != "fresh":
        raise ValueError(f"fresh-head-init policy violated: {arm}")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=9600)
    args = parser.parse_args()
    contract = load_repr_head_contract(args.contract)
    records = _load_records(contract["training"]["evaluation_data"], args.limit)
    if not records:
        raise ValueError("empty evaluation set")
    args.out.mkdir(parents=True, exist_ok=False)
    all_rows, per_arm, export_metadata = [], {}, {}
    for arm in ALL_ARMS:
        if not args.limit:
            _verify_completion(args.stage, arm)
        export_dir = args.stage / arm / "export_train_off"
        metadata = _verify_export_metadata(export_dir, args.seed, not args.limit,
                                           args.steps, len(records))
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
                print(f"[repr-head] seed={args.seed} arm={arm} {index}/{len(records)}", flush=True)
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
    control = per_arm[CONTROL_ARM]
    comparisons = {}
    if not args.limit:
        for arm in VARIANT_ARMS:
            deltas, checks, passed = _direction_gate(
                per_arm[arm], control, contract["pilot"]["per_arm_direction_gate"])
            comparisons[arm] = {"delta_vs_control_matched": deltas, "checks": checks,
                                "direction_passed": passed}
    promoted = [arm for arm in VARIANT_ARMS if comparisons.get(arm, {}).get("direction_passed")]
    decision = CONFIRM_DECISION if promoted else STOP_DECISION
    if args.limit:
        decision = "SMOKE_ONLY_NO_SCIENTIFIC_VERDICT"
    control_sanity = {"reference_full_control_micro_pair_f1": REF_FULL_CONTROL_MICRO_PAIR_F1,
                      "control_matched_micro_pair_f1": control["micro_pair_f1"],
                      "delta": control["micro_pair_f1"] - REF_FULL_CONTROL_MICRO_PAIR_F1,
                      "role": "integrity_sanity_not_promotion_gate"}
    verdict = {
        "schema_version": 1, "status": "PASS", "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(args.contract), "seed": args.seed,
        "n_sequences": len(records), "arms": per_arm, "comparisons": comparisons,
        "promoted_arms": promoted, "decision": decision, "control_arm": CONTROL_ARM,
        "control_sanity": control_sanity,
        "external_evaluation_allowed": False, "candidate_selected": False, "sota_claim": False,
        "decoder": contract["evaluation"]["decoder"], "decoder_temperature_or_gamma_searched": False,
        "raw_metrics_are_auxiliary_diagnostics": True, "full_control_reused_not_retrained": False,
        "exports": {arm: {"path": str(args.stage / arm / "export_train_off"),
                          "checkpoint_sha256": md["checkpoint"]["sha256"],
                          "contract_sha256": md["contract_sha256"]}
                    for arm, md in export_metadata.items()},
    }
    (args.out / "pilot_verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "promoted_arms": promoted, "control_sanity": control_sanity,
                      "arms": per_arm, "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()