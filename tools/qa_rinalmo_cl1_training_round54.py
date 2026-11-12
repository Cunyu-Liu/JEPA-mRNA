import argparse
import json
import statistics
from pathlib import Path

from qa_rinalmo_b0_training_round54 import read_jsonl, sha256_file, validate_run

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")


def write(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({key: value for key, value in report.items() if key not in {"runs", "paired_deltas"}}))


def training(seed):
    run = ROOT / "runs" / f"round54_cl1_s{seed}"
    report = validate_run(run, seed, sha256_file(DATA), arm="cl1")
    meta = json.loads((run / "run_meta.json").read_text())
    baseline = json.loads((ROOT / "runs" / f"round54_b0_s{seed}/run_meta.json").read_text())
    assert meta["schedule"] == baseline["schedule"]
    assert meta["validation"] == baseline["validation"]
    assert meta["data"] == baseline["data"] == str(DATA)
    assert meta["objective"]["cl1_token_mask"] == 0.15 and meta["objective"]["cl1_views"] == 2
    assert meta["objective"]["lambda_jepa"] == baseline["objective"]["lambda_jepa"] == 0.0
    logs = read_jsonl(run / "train_log.jsonl")
    assert all(row["cl1_views"] == 2 and len(row["cl1_masked_tokens"]) == 2 for row in logs)
    assert all(set(row["terms"]) == {"nll", "distill", "rlcd", "cal"} for row in logs)
    report.update({"status": "PASS_CL1_SINGLE_SEED_TRAINING_QA",
                   "masked_tokens_total": sum(sum(row["cl1_masked_tokens"]) for row in logs),
                   "logged_steps": len(logs), "contrastive_loss": False,
                   "deployment_change": False, "scientific_conclusion": False})
    return report


def summary():
    runs = {}
    for seed in (42, 2026, 3407):
        report = json.loads((ROOT / f"audits/round54/cl1_s{seed}_training_qa.json").read_text())
        assert report["status"] == "PASS_CL1_SINGLE_SEED_TRAINING_QA"
        runs[str(seed)] = report
    baseline = json.loads((ROOT / "audits/round54/b0_training_qa.json").read_text())["runs"]
    values = [row["best_vl0"] for row in runs.values()]
    return {"status": "PASS_CL1_THREE_SEED_DESCRIPTIVE_QA", "runs": runs,
            "mean_best_vl0": statistics.mean(values), "population_sd_best_vl0": statistics.pstdev(values),
            "paired_deltas": {seed: runs[seed]["best_vl0"] - baseline[seed]["best_vl0"] for seed in runs},
            "test_set_used": False, "scientific_conclusion": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[42, 2026, 3407])
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summary() if args.summary else training(args.seed)
    write(Path(args.output), report)


if __name__ == "__main__":
    main()
