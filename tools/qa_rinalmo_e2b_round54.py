import argparse
import json
import statistics
from pathlib import Path

from qa_rinalmo_b0_training_round54 import read_jsonl, sha256_file, stable_load, validate_run

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
ASSETS = ROOT / "data/e2b_thermo_round54"


def training(seed, arm):
    run = ROOT / "runs" / f"round54_{arm}_s{seed}"
    report = validate_run(run, seed, sha256_file(DATA), arm=arm)
    meta = json.loads((run / "run_meta.json").read_text())
    baseline = json.loads((ROOT / f"runs/round54_b0_s{seed}/run_meta.json").read_text())
    expected = "shuffled" if arm == "e2b_shuffled" else "online"
    assert meta["schedule"] == baseline["schedule"] and meta["validation"] == baseline["validation"]
    assert meta["data"] == baseline["data"] == str(DATA)
    assert meta["thermodynamics"]["mode"] == expected
    assert meta["thermodynamics"]["features"] == 8
    assert meta["thermodynamics"]["cache_dir"] == str(ASSETS / "cache")
    assert meta["motif"] is None and meta["objective"]["lambda_jepa"] == 0
    assert meta["objective"]["cl1_token_mask"] == meta["objective"]["cl2_infonce_weight"] == 0
    logs = read_jsonl(run / "train_log.jsonl")
    assert all(row["motif_projection_grad"] > 0 for row in logs)
    assert all(set(row["terms"]) == {"nll", "distill", "rlcd", "cal"} for row in logs)
    state = stable_load(run / "best.pt")
    assert any(key.startswith("motif_input.projection.") for key in state["model"]["head"])
    report.update({"status": f"PASS_{arm.upper()}_SINGLE_SEED_TRAINING_QA",
                   "projection_gradient": "NONZERO", "logged_steps": len(logs),
                   "normalization_sha256": sha256_file(ASSETS / "normalization.json"),
                   "cache_manifest_sha256": sha256_file(ASSETS / "train_cache_manifest.json"),
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


def smoke(mode):
    run = ROOT / f"runs/round54_e2b_{mode}_smoke_s42"
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["seed"] == 42 and meta["status"] == "completed" and meta["steps_completed"] == 2
    assert meta["environment"]["resolved_device"].startswith("cuda")
    assert meta["thermodynamics"]["mode"] == mode and meta["schedule"]["steps"] == 2
    assert meta["objective"]["nll_normalization"] == "sum"
    assert all(meta["objective"][name] == 1.0 for name in ("lambda_nll", "lambda_distill", "lambda_rlcd", "lambda_cal"))
    logs = read_jsonl(run / "train_log.jsonl")
    assert all(row["motif_projection_grad"] > 0 for row in logs)
    best = stable_load(run / "best.pt")
    resume = stable_load(run / "resume.pt")
    assert best["step"] == resume["step"] == 2
    for part in ("encoder", "head"):
        assert list(best["model"][part]) == list(resume[part])
        assert all(__import__("torch").equal(best["model"][part][key], resume[part][key]) for key in best["model"][part])
    return {"status": "PASS_E2B_CUDA_SMOKE", "run": meta["arm"], "train_log": logs,
            "best_vl0": meta["best_validation_loss"], "best_resume_weights_equal": True,
            "run_meta_sha256": sha256_file(run / "run_meta.json"),
            "best_sha256": sha256_file(run / "best.pt"), "resume_sha256": sha256_file(run / "resume.pt"),
            "scientific_conclusion": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[42, 2026, 3407])
    parser.add_argument("--arm", choices=["e2b", "e2b_shuffled"], default="e2b")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--smoke", choices=["online", "shuffled"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = smoke(args.smoke) if args.smoke else (summary(args.arm) if args.summary else training(args.seed, args.arm))
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
