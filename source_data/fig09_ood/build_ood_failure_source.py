#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "status": "descriptive_offline_reanalysis_not_new_model_evidence",
        "protocol": "Original xens2 predictions and CHANRG JSONL aligned in file order; micro-F1 recomputed from pair sets",
        "splits": {},
        "sources": {},
    }
    for split in ["genc", "genf"]:
        data_path = args.input_dir / f"chanrg_{split}.jsonl"
        result_path = args.input_dir / f"{split}_xens2_result.json"
        data = [json.loads(line) for line in data_path.open()]
        result = json.load(result_path.open())
        predictions = result["per_sequence"]
        assert len(data) == len(predictions) == result["n_sequences"]
        groups = defaultdict(lambda: {"n": 0, "tp": 0, "fp": 0, "fn": 0, "zero_prediction": 0,
                                      "zero_correct_prediction": 0, "n_pred": 0, "n_gt": 0})
        for row, prediction in zip(data, predictions):
            pred = {tuple(pair) for pair in prediction["pred_pairs"]}
            true = {tuple(pair) for pair in row["pairs"]}
            assert len(pred) == prediction["n_pred"] and len(true) == prediction["n_gt"]
            family = groups[row["family"]]
            family["n"] += 1
            family["tp"] += len(pred & true)
            family["fp"] += len(pred - true)
            family["fn"] += len(true - pred)
            family["zero_prediction"] += int(not pred)
            family["zero_correct_prediction"] += int(not (pred & true))
            family["n_pred"] += len(pred)
            family["n_gt"] += len(true)
        total = {key: sum(family[key] for family in groups.values())
                 for key in ["n", "tp", "fp", "fn", "zero_prediction", "zero_correct_prediction", "n_pred", "n_gt"]}
        f1 = 2 * total["tp"] / (2 * total["tp"] + total["fp"] + total["fn"])
        assert abs(f1 - result["pair_level"]["micro"]["f1"]) < 1e-12
        families = []
        for name, family in groups.items():
            tp, fp, fn = family["tp"], family["fp"], family["fn"]
            families.append({
                "family": name, **family,
                "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
                "zero_prediction_fraction": family["zero_prediction"] / family["n"],
                "zero_correct_prediction_fraction": family["zero_correct_prediction"] / family["n"],
                "predicted_pairs_per_sequence": family["n_pred"] / family["n"],
                "true_pairs_per_sequence": family["n_gt"] / family["n"],
                "fn_fraction_of_split": fn / total["fn"] if total["fn"] else 0.0,
            })
        families.sort(key=lambda row: (-row["fn"], row["family"]))
        payload["splits"][split] = {
            **total, "micro_f1": f1,
            "zero_prediction_fraction": total["zero_prediction"] / total["n"],
            "zero_correct_prediction_fraction": total["zero_correct_prediction"] / total["n"],
            "predicted_pairs_per_sequence": total["n_pred"] / total["n"],
            "true_pairs_per_sequence": total["n_gt"] / total["n"],
            "families": families,
        }
        payload["sources"][split] = {"data_sha256": sha256(data_path), "result_sha256": sha256(result_path)}
    current = [0.644258, payload["splits"]["genc"]["micro_f1"], payload["splits"]["genf"]["micro_f1"]]
    payload["ood_context"] = {
        "gena_micro_f1": current[0],
        "current_ood_mean": sum(current) / 3,
        "ofoldna_ss": {"GenA": 0.5695, "GenC": 0.3956, "GenF": 0.4035, "ood_mean": 0.4410},
        "arithmetic_if_genc_matched_ofoldna_not_a_prediction": (current[0] + 0.3956 + current[2]) / 3,
        "conclusion": "Both splits underpredict pairs. Empty outputs are rare, but 20.8% of GenC and 39.2% of GenF have no correct predicted pair. No improvement is claimed by this reanalysis.",
        "log_correction": "The older 20.8%/39.2% values are zero-correct-pair fractions, not zero-output fractions; the terms must not be interchanged.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
