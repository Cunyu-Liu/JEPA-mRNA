#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from analyze_structrfm_confidence_round48 import SPLITS
from confidence_metrics_round49 import evaluate_structure_confidence


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = args.root / "eval_decision" / args.experiment
    threshold_path = base / "confidence_vl0_thresholds.json"
    threshold_record = json.loads(threshold_path.read_text())
    metas = [json.loads((args.root / f"runs/{args.run_prefix}{seed}/run_meta.json").read_text())
             for seed in range(3)]
    reference_seed = min(range(3), key=lambda seed: (metas[seed]["best_validation_loss"], seed))
    assert threshold_record["reference_seed"] == reference_seed
    thresholds = threshold_record["thresholds"]
    previous = json.loads((base / "confidence_summary.json").read_text())
    assert previous["thresholds"]["thresholds"] == thresholds
    sources = {str(threshold_path.relative_to(args.root)): sha256(threshold_path),
               str((base / "confidence_summary.json").relative_to(args.root)): sha256(base / "confidence_summary.json")}
    output = {"status": "PASS_STRUCTURE_CONFIDENCE_INDEPENDENT_ONLY",
              "experiment": args.experiment, "reference_seed": reference_seed,
              "selection_rule": "thresholds remain frozen from minimum best VL0 seed; no test fitting",
              "best_confidence_claim": False,
              "missing_for_best_claim": "same-candidate-pair/GT calibration and structure-confidence comparison against probability baselines",
              "vl0": {}, "source_hashes": sources}
    for scope in ["vl0"] + SPLITS:
        output[scope] = {}
        for seed in range(3):
            path = base / (f"vl0_s{seed}.json" if scope == "vl0" else f"s{seed}/{scope}.json")
            result = json.loads(path.read_text())
            assert result["device"] == "cuda" and result["repeats"] == 3
            assert result["n"] == len(result["per_sequence"])
            sources[str(path.relative_to(args.root))] = sha256(path)
            row = evaluate_structure_confidence(result["per_sequence"], thresholds)
            old = (previous["vl0"] if scope == "vl0" else previous["splits"][scope])[f"s{seed}"]
            for key in ["spearman", "risk", "aurc", "vl0_thresholds"]:
                if key == "spearman":
                    if row[key]["value"] is None:
                        assert old[key] is None
                    else:
                        assert abs(old[key] - row[key]["value"]) < 1e-12
                elif key == "aurc":
                    assert old[key] == row[key] or (old[key] is None and row[key] is None)
                elif key == "risk":
                    assert str(old[key]) == str(row[key])
                else:
                    assert old[key] == row[key]
            output[scope][f"s{seed}"] = row
    atomic_json(args.out, output)
    assert json.dumps(json.loads(args.out.read_text()), sort_keys=True) == json.dumps(output, sort_keys=True)
    print(json.dumps({"status": output["status"], "experiment": args.experiment,
                      "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
