import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_structrfm_confidence_round48 import COVERAGE
from confidence_metrics_round49 import evaluate_structure_confidence


SPLITS = ("bprna_ts0", "bprna_new", "archiveii_embok_clean", "ref_pdb_ts1",
          "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "rfam_fam")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--threshold-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    training_path = root / "audits/round52/rinalmo_jmlp_training_qa.json"
    benchmark_path = root / "audits/round52/rinalmo_jmlp_benchmark_qa.json"
    training = json.loads(training_path.read_text())
    benchmark = json.loads(benchmark_path.read_text())
    assert training["status"] == "PASS_JMLP_THREE_SEED_TRAINING_QA"
    assert benchmark["status"] == "PASS_JMLP_MAIN8_RFAM_INDEPENDENT_QA" and benchmark["complete"]
    best = {int(seed): row["best_vl0"] for seed, row in training["runs"].items()}
    reference_seed = min(best, key=lambda seed: (best[seed], seed))
    reference_path = root / "evals/round52" / f"rinalmo_jmlp_length_s{reference_seed}_vl0_nussinov.json"
    reference = json.loads(reference_path.read_text())
    assert reference["n"] == 196 and reference["repeats"] == 3
    thresholds = np.quantile([row["structure_conf"] for row in reference["per_sequence"]], 1 - COVERAGE)
    threshold_record = {
        "status": "PASS_JMLP_VL0_FROZEN_CONFIDENCE_THRESHOLDS",
        "reference_seed": reference_seed, "reference_best_vl0": best[reference_seed],
        "coverage": COVERAGE.tolist(), "thresholds": thresholds.tolist(),
        "reference_result_sha256": sha256_file(reference_path),
        "protocol": "minimum best VL0 seed; thresholds use only its 196 VL0 structures; no test fitting",
    }
    threshold_path = Path(args.threshold_output)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = threshold_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(threshold_record, indent=2, allow_nan=False) + "\n")
    temporary.replace(threshold_path)
    result = {
        "status": "PASS_JMLP_CONFIDENCE_VL0_FROZEN_EVALUATION",
        "best_confidence_claim": False, "thresholds": threshold_record,
        "vl0": {}, "splits": {}, "source_hashes": {
            "training": sha256_file(training_path), "benchmark": sha256_file(benchmark_path),
            "thresholds": sha256_file(threshold_path), "analysis_code": sha256_file(__file__)},
        "limitations": [
            "Thresholds are frozen from VL0 only; main eight and Rfam are never used to fit confidence thresholds.",
            "structure_conf is a decoded-pair geometric-mean score, not a calibrated probability of structure correctness.",
            "No same-candidate probability-baseline comparison is included; best-confidence claims are not licensed.",
        ],
    }
    for seed in (42, 2026, 3407):
        path = root / "evals/round52" / f"rinalmo_jmlp_length_s{seed}_vl0_nussinov.json"
        cell = json.loads(path.read_text())
        assert cell["n"] == 196 and cell["repeats"] == 3
        result["vl0"][str(seed)] = evaluate_structure_confidence(cell["per_sequence"], thresholds)
        result["source_hashes"][f"vl0_{seed}"] = sha256_file(path)
    for split in SPLITS:
        result["splits"][split] = {}
        for seed in (42, 2026, 3407):
            path = root / "evals/round52/jmlp_benchmark" / f"rinalmo_jmlp_length_s{seed}_{split}.json"
            cell = json.loads(path.read_text())
            assert cell["status"] == "PASS" and cell["raw_n"] == cell["n"] and cell["repeats"] == 3
            result["splits"][split][str(seed)] = evaluate_structure_confidence(cell["per_sequence"], thresholds)
            result["source_hashes"][f"{split}_{seed}"] = sha256_file(path)
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "reference_seed": reference_seed,
                      "n_splits": len(result["splits"]), "sha256": sha256_file(path)}, indent=2))


if __name__ == "__main__":
    main()
