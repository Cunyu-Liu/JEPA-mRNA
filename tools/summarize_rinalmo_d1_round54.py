import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
SEEDS = (42, 2026, 3407)


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_sd(values):
    return {"mean": statistics.mean(values), "population_sd": statistics.pstdev(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = ROOT / "audits/round54"
    baseline = load(audit / "b0_training_qa.json")
    assert baseline["status"] == "PASS_B0_TRAINING_THREE_CHECK_QA"
    training = {"b0": baseline["runs"], "d1": {}}
    holdout = {"b0": {}, "d1": {}}
    inputs = {"b0_training": sha256(audit / "b0_training_qa.json")}
    for arm in ("b0", "d1"):
        for seed in SEEDS:
            path = audit / f"{arm}_s{seed}_holdout_qa.json"
            holdout[arm][str(seed)] = load(path)
            assert holdout[arm][str(seed)]["status"] == f"PASS_{arm.upper()}_COMPLEX_HOLDOUT_1022_QA"
            inputs[f"{arm}_s{seed}_holdout"] = sha256(path)
            if arm == "d1":
                path = audit / f"d1_s{seed}_training_qa.json"
                training[arm][str(seed)] = load(path)
                assert training[arm][str(seed)]["status"] == "PASS_D1_SINGLE_SEED_TRAINING_QA"
                inputs[f"d1_s{seed}_training"] = sha256(path)
    report = {"status": "PASS_D0_D1_THREE_SEED_DESCRIPTIVE_QA", "seeds": list(SEEDS),
              "source_sha256": inputs, "arms": {}, "paired_d1_minus_b0": {},
              "test_set_used": False, "scientific_conclusion": False}
    for arm in ("b0", "d1"):
        values = [training[arm][str(seed)]["best_vl0"] for seed in SEEDS]
        arm_report = {"best_vl0": mean_sd(values), "holdout": {}}
        for bucket in ("all", "le400", "401_1022"):
            arm_report["holdout"][bucket] = {
                metric: mean_sd([holdout[arm][str(seed)]["buckets"][bucket][metric] for seed in SEEDS])
                for metric in ("micro_f1", "macro_f1")}
        if arm == "d1":
            arm_report["new_chain_coverage"] = {
                str(seed): {key: training[arm][str(seed)][key] for key in
                            ("new_chains", "new_chains_seen_at_least_once",
                             "new_chain_batch_appearances", "new_chain_coverage_complete")}
                for seed in SEEDS}
        report["arms"][arm] = arm_report
    report["paired_d1_minus_b0"]["best_vl0"] = {
        str(seed): training["d1"][str(seed)]["best_vl0"] - training["b0"][str(seed)]["best_vl0"]
        for seed in SEEDS}
    report["paired_d1_minus_b0"]["holdout_micro_f1"] = {
        bucket: {str(seed): holdout["d1"][str(seed)]["buckets"][bucket]["micro_f1"]
                 - holdout["b0"][str(seed)]["buckets"][bucket]["micro_f1"] for seed in SEEDS}
        for bucket in ("all", "le400", "401_1022")}
    report["directional_checks"] = {
        "mean_vl0_improved": report["arms"]["d1"]["best_vl0"]["mean"] < report["arms"]["b0"]["best_vl0"]["mean"],
        "mean_holdout_micro_f1_improved": {
            bucket: report["arms"]["d1"]["holdout"][bucket]["micro_f1"]["mean"]
            > report["arms"]["b0"]["holdout"][bucket]["micro_f1"]["mean"]
            for bucket in ("all", "le400", "401_1022")}}
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "directional_checks": report["directional_checks"]}, indent=2))


if __name__ == "__main__":
    main()
