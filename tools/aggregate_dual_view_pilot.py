import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(a, b):
    return float(a / b) if b else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.artifact_root) / "artifacts/p2/pilot_v1"
    results = {}
    for arm in ("H0", "H1", "H2", "H3", "H4", "H5", "H6"):
        meta = load(root / arm / "run_meta.json")
        if meta["status"] != "completed":
            raise AssertionError(f"{arm} is not complete")
        if meta["seed"] != 42 or meta["schedule"]["steps"] != 500:
            raise AssertionError(f"pilot identity mismatch for {arm}")
        if meta["objective"]["lambda_jepa"] != 0 or meta["motif"] is not None or meta["thermodynamics"] is not None:
            raise AssertionError(f"forbidden arm contamination for {arm}")
        evaluation = load(root / arm / "model_dev_eval.json")
        if evaluation["data_role"] != "model-dev" or evaluation["data_sha256"] != "1920d906de5ea84403aa83a97bfab5538a780bd0457d211d6de44b36b60a7f4d":
            raise AssertionError(f"wrong evaluation role or data for {arm}")
        if any(evaluation["training_only_call_counts"].values()) or evaluation["structure_labels_used_for_inference"]:
            raise AssertionError(f"structure label leakage for {arm}")
        per_sequence = evaluation["per_sequence"]
        tp = sum(record["tp"] for record in per_sequence)
        fp = sum(record["fp"] for record in per_sequence)
        fn = sum(record["fn"] for record in per_sequence)
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        micro = ratio(2 * precision * recall, precision + recall)
        macro = float(np.mean([record["f1"] for record in per_sequence]))
        stored = evaluation["pair_level"]
        if abs(micro - stored["micro"]["f1"]) > 1e-12 or abs(macro - stored["macro"]["f1"]) > 1e-12:
            raise AssertionError(f"independent metric mismatch for {arm}")
        if (tp, fp, fn) != tuple(stored["micro"][key] for key in ("tp", "fp", "fn")):
            raise AssertionError(f"independent count mismatch for {arm}")
        diagnostics = [json.loads(line) for line in (root / arm / "dual_view_diagnostics.jsonl").read_text().splitlines()] if arm != "H0" else []
        if arm != "H0" and not diagnostics:
            raise AssertionError(f"missing dual-view diagnostics for {arm}")
        results[arm] = {
            "macro_f1": macro,
            "micro_f1": micro,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "checkpoint_step": evaluation["checkpoint_step"],
            "checkpoint_sha256": evaluation["checkpoint_sha256"],
            "converged": meta["converged"],
            "stop_reason": meta["stop_reason"],
            "steps_completed": meta["steps_completed"],
            "n_diagnostic_records": len(diagnostics),
        }
    controls = ("H0", "H3", "H4", "H5", "H6")
    strongest = max(results[arm]["macro_f1"] for arm in controls)
    passed = results["H2"]["macro_f1"] > strongest
    output = {
        "status": "PASS_PILOT_MECHANISM_DEV_ONLY" if passed else "STOP_PILOT_NEGATIVE_DEV_ONLY",
        "pilot_passed": passed,
        "p3_started": False,
        "scientific_scope": "single-seed model-dev mechanism pilot only; no unseen-family accuracy or convergence claim",
        "seed": 42,
        "decision_rule": "H2 model-dev macro-F1 must exceed H0, H3, H4, H5 and H6",
        "strongest_control_macro_f1": strongest,
        "h2_minus_strongest_control": results["H2"]["macro_f1"] - strongest,
        "results": results,
        "source_sha256": sha256(args.artifact_root + "/manifests/p2_pilot_v1_config.json"),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
