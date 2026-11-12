import argparse
import json
import statistics
from pathlib import Path

from qa_rinalmo_b0_training_round54 import sha256_file

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")


def aggregate(records, key):
    result = {}
    for metric in ("all", "span30", "span100"):
        tp = sum(row[key][metric]["tp"] for row in records)
        fp = sum(row[key][metric]["fp"] for row in records)
        fn = sum(row[key][metric]["fn"] for row in records)
        result[metric] = {"micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                          "macro_f1": statistics.mean(row[key][metric]["f1"] for row in records),
                          "tp": tp, "fp": fp, "fn": fn}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = {}
    for seed in (42, 2026, 3407):
        path = ROOT / f"audits/round54/e2c_s{seed}_vl0.json"
        result = json.loads(path.read_text())
        assert result["status"] == "PASS_E2C_VL0" and result["n_sequences"] == 196
        assert result["seed"] == seed and not result["test_set_used"]
        assert len(result["records"]) == 196
        runs[str(seed)] = {"original": aggregate(result["records"], "original"),
                           "physics": aggregate(result["records"], "physics"),
                           "result_sha256": sha256_file(path),
                           "checkpoint_sha256": result["checkpoint_sha256"]}
    delta = {seed: runs[seed]["physics"]["all"]["micro_f1"] - runs[seed]["original"]["all"]["micro_f1"] for seed in runs}
    mean_delta = statistics.mean(delta.values())
    report = {"status": "PASS_E2C_INDEPENDENT_RECOMPUTE", "runs": runs, "delta_micro_f1": delta,
              "mean_delta_micro_f1": mean_delta, "retain_physics_dp": mean_delta > 0,
              "test_set_used": False, "checkpoint_training": False,
              "physics_dp_binary_sha256": sha256_file(ROOT / "bin/physics_dp.so"),
              "physics_dp_source_sha256": sha256_file(Path("/home/rna_jev/tools/physics_dp.cpp"))}
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    decision = {"status": "E2C_VL0_DECISION", "mean_delta_micro_f1": mean_delta,
                "retain_physics_dp": mean_delta > 0, "test_set_used": False,
                "summary_sha256": sha256_file(ROOT / "audits/round54/e2c_summary.json")}
    (ROOT / "audits/round54/e2c_vl0_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "delta_micro_f1": delta,
                      "mean_delta_micro_f1": mean_delta, "retain": mean_delta > 0}, indent=2))


if __name__ == "__main__":
    main()
