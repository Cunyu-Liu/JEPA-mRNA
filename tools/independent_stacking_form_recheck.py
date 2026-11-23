#!/usr/bin/env python
"""Independent raw-score recheck for the seed0 stacking-form pilot."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "tools"))
import verify_raw_score_matrix as V
from analyze_pair_stacking_fn import pair_environment
from rnajepa.stacking_closure import MODES
from rnajepa.stacking_form_matrix import CONTROL_ARM, load_stacking_form_contract

KEYS = ("micro_pair_f1", "macro_pair_f1", "micro_auprc", "micro_pair_precision",
        "macro_pair_precision", "micro_stem_f1", "macro_stem_f1", "micro_recall_at_precision_0.8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=9600)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite independent recheck")
    contract = load_stacking_form_contract(args.contract)
    records = V.load_records(contract["training"]["evaluation_data"])
    recomputed, targets = {}, {}
    for arm in MODES:
        export = args.stage / arm / "export_train_off"
        metadata = json.loads((export / "export_metadata.json").read_text())
        checkpoint = args.stage / arm / "snapshots" / f"{arm}_step{args.steps}.pt"
        checkpoint_sha = V.sha256_file(checkpoint)
        assert metadata["checkpoint"]["sha256"] == checkpoint_sha
        rows, score_parts, label_parts = [], [], []
        for sequence_hash, seq, truth in records:
            with np.load(export / "bprna_vl0" / f"{sequence_hash}.npz", allow_pickle=False) as data:
                assert str(data["seq"]) == seq and str(data["checkpoint_sha256"]) == checkpoint_sha
                scores = np.asarray(data["plana_off"], dtype=np.float64)
                assert {tuple(sorted(map(int, pair))) for pair in data["gt_pairs"].tolist()} == truth
            mask = V.valid_pair_mask(seq)
            log_z, posterior = V.inside_outside(scores, mask)
            assert math.isfinite(log_z) and np.isfinite(posterior).all()
            predicted = set(V.decode_mea(posterior, mask, 1.0))
            if arm == CONTROL_ARM:
                target = set()
                for pair in truth:
                    if pair_environment(pair, truth) == "helix_boundary":
                        i, j = pair
                        if {(i - 1, j + 1), (i + 1, j - 1)} & truth & predicted:
                            target.add(pair)
                targets[sequence_hash] = target
            target = targets[sequence_hash]
            row = {"arm": arm, "sequence_sha256": sequence_hash, "n_truth_pairs": len(truth),
                   "n_predicted_pairs": len(predicted), "target_supported_boundary_n": len(target),
                   "target_supported_boundary_tp": len(target & predicted)}
            row.update(V.pair_and_stem_counts(predicted, truth))
            row["raw"] = V.independent_raw_metrics(scores, mask, truth)
            rows.append(row)
            rr, cc = np.where(np.triu(mask, k=1)); order = np.lexsort((cc, rr)); rr, cc = rr[order], cc[order]
            score_parts.append(scores[rr, cc])
            label_parts.append(np.fromiter(((int(i), int(j)) in truth for i, j in zip(rr, cc)),
                                           dtype=bool, count=len(rr)))
        metrics = V.aggregate(rows, score_parts, label_parts, checkpoint_sha)
        target_n = sum(row["target_supported_boundary_n"] for row in rows)
        target_tp = sum(row["target_supported_boundary_tp"] for row in rows)
        metrics["target_supported_boundary_n"] = target_n
        metrics["target_supported_boundary_recall"] = target_tp / target_n if target_n else None
        recomputed[arm] = metrics
    deltas, checks, direction = V.direction_gate(recomputed["stacking_closure"], recomputed[CONTROL_ARM],
                                                  contract["pilot"]["per_arm_direction_gate"])
    target_ok = (recomputed["stacking_closure"]["target_supported_boundary_recall"]
                 > recomputed[CONTROL_ARM]["target_supported_boundary_recall"])
    destroy = {metric: all(recomputed["stacking_closure"][metric] > recomputed[arm][metric]
                           for arm in ("destroy_offset", "destroy_channel_shuffle"))
               for metric in contract["pilot"]["must_beat_both_destroy_controls_on"]}
    passed = direction and target_ok and all(destroy.values())
    expected_decision = "PILOT_FORM_DIRECTION_CONFIRMED" if passed else "STOP_STACKING_CLOSURE_FORM_SEED0"
    official = json.loads((args.evaluation / "pilot_verdict.json").read_text())
    differences = {}
    for arm in MODES:
        differences[arm] = {}
        for key in (*KEYS, "target_supported_boundary_recall"):
            actual, expected = recomputed[arm][key], official["arms"][arm][key]
            differences[arm][key] = 0.0 if actual is None and expected is None else abs(actual - expected)
    max_diff = max(value for arm in differences.values() for value in arm.values())
    overall = max_diff <= 1e-6 and expected_decision == official["decision"]
    result = {"status": "PASS" if overall else "FAIL", "scope": "INDEPENDENT_STACKING_FORM_RAW_RECHECK",
              "contract_sha256": V.sha256_file(args.contract), "n_sequences": len(records),
              "recomputed_arms": recomputed, "delta_vs_local_control": deltas,
              "direction_checks": checks, "target_passed": target_ok, "destroy_checks": destroy,
              "recomputed_decision": expected_decision, "official_decision": official["decision"],
              "max_abs_diff": max_diff, "differences": differences,
              "external_evaluation_allowed": False, "strict_ood_allowed": False, "sota_claim": False}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "max_abs_diff": max_diff,
                      "decision": expected_decision}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
