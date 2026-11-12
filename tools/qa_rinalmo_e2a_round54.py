import argparse
import json
import statistics
from pathlib import Path

from qa_rinalmo_b0_training_round54 import read_jsonl, sha256_file, stable_load, validate_run

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
ASSETS = ROOT / "data/e2a_motif_round54"


def training(seed, arm):
    run = ROOT / "runs" / f"round54_{arm}_s{seed}"
    report = validate_run(run, seed, sha256_file(DATA), arm=arm)
    meta = json.loads((run / "run_meta.json").read_text())
    baseline = json.loads((ROOT / f"runs/round54_b0_s{seed}/run_meta.json").read_text())
    expected = "shuffled" if arm == "e2a_shuffled" else "motif"
    assert meta["schedule"] == baseline["schedule"] and meta["validation"] == baseline["validation"]
    assert meta["data"] == baseline["data"] == str(DATA)
    assert meta["motif"]["mode"] == expected and meta["motif"]["features"] == 6
    assert meta["motif"]["asset_dir"] == str(ASSETS)
    assert meta["objective"]["cl1_token_mask"] == meta["objective"]["lambda_jepa"] == 0
    logs = read_jsonl(run / "train_log.jsonl")
    assert all(row["motif_projection_grad"] > 0 for row in logs)
    assert all(set(row["terms"]) == {"nll", "distill", "rlcd", "cal"} for row in logs)
    state = stable_load(run / "best.pt")
    assert any(key.startswith("motif_input.projection.") for key in state["model"]["head"])
    assert not any(key.startswith("pair_jepa") for key in state["model"]["head"])
    report.update({"status": f"PASS_{arm.upper()}_SINGLE_SEED_TRAINING_QA",
                   "motif_projection_gradient": "NONZERO", "logged_steps": len(logs),
                   "normalization_sha256": sha256_file(ASSETS / "normalization.json"),
                   "parameters_sha256": sha256_file(ASSETS / "parameters.npz"),
                   "test_set_used": False, "scientific_conclusion": False})
    return report


def summary(arm):
    runs = {str(seed): training(seed, arm) for seed in (42, 2026, 3407)}
    values = [row["best_vl0"] for row in runs.values()]
    b0 = json.loads((ROOT / "audits/round54/b0_training_qa.json").read_text())["runs"]
    return {"status": f"PASS_{arm.upper()}_THREE_SEED_DESCRIPTIVE_QA", "runs": runs,
            "mean_best_vl0": statistics.mean(values), "population_sd_best_vl0": statistics.pstdev(values),
            "paired_delta_b0": {seed: runs[seed]["best_vl0"] - b0[seed]["best_vl0"] for seed in runs},
            "test_set_used": False, "scientific_conclusion": False}


def failure():
    assert not (ASSETS / "reference_qa.json").exists()
    assert not list((ROOT / "runs").glob("round54_e2a_*_s*/run_meta.json"))
    return {"status": "STOP_E2A_PARAMETER_GATE",
            "reason": "Vienna reference mismatch persists after one root-cause repair; no approximation or training is licensed.",
            "parameter_sha256": sha256_file(ASSETS / "rna_turner2004.par"),
            "special_parameter_sha256": sha256_file(ASSETS / "special_hairpins.par"),
            "prepare_log_sha256": sha256_file(ROOT / "logs/round54/e2a_prepare.log"),
            "motif_training_runs": 0, "shuffled_training_runs": 0,
            "e2b_reason": "E2a cannot answer whether local motif energy is sufficient; independent E2b is required by the preregistered condition.",
            "scientific_conclusion": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[42, 2026, 3407])
    parser.add_argument("--arm", choices=["e2a_motif", "e2a_shuffled"], default="e2a_motif")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--parameter-gate-failure", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = failure() if args.parameter_gate_failure else (summary(args.arm) if args.summary else training(args.seed, args.arm))
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
