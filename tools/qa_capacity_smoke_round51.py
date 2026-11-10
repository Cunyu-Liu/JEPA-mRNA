import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate(root, latent):
    root = Path(root)
    meta = json.loads((root / "run_meta.json").read_text())
    config = meta["config"]
    assert config["steps"] == 2 and config["batch_size"] == 2 and config["seed"] == 42
    assert config["scorer"] == "pairmlp" and config["pair_latent_dim"] == latent
    assert config["hidden"] == 64 and config["d_z"] == 128
    assert config["nll_normalization"] == "length" and config["lambda_jepa"] == 0
    assert config["allow_cpu"] is False and config["device"] == "cuda"
    assert meta["status"] == "completed" and meta["steps_completed"] == 2
    assert meta["stop_reason"] == "step_cap" and not meta["converged"]
    assert meta["environment"]["resolved_device"].startswith("cuda")
    assert meta["environment"]["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert not meta["gradient_coverage"]["missing"]
    assert not meta["gradient_coverage"]["nonfinite"]
    assert len(meta["gradient_coverage"]["covered"]) == 12
    assert sum(name.startswith("head.pair_jepa.context_predictor")
               for name in meta["gradient_coverage"]["excluded"]) == 6
    assert (root / "stale_pre_exclusion").is_dir() or latent == 490
    best = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    resume = torch.load(root / "resume.pt", map_location="cpu", weights_only=False)
    assert best["step"] == resume["step"] == 2
    assert "validation_loss" not in best
    assert list(best["model"]) == list(resume["model"])
    assert all(torch.equal(best["model"][key], resume["model"][key])
               for key in best["model"])
    scorer_keys = {key: value for key, value in best["model"].items()
                   if key.startswith("head.pair_jepa.scorer.")}
    input_weight = scorer_keys["head.pair_jepa.scorer.input_projection.weight"]
    hidden_weight = scorer_keys["head.pair_jepa.scorer.hidden.1.weight"]
    logit_weight = scorer_keys["head.pair_jepa.scorer.logit.weight"]
    assert input_weight.shape == (latent, 128)
    assert hidden_weight.shape == (latent, latent)
    assert logit_weight.shape == (1, latent)
    assert not any(key.startswith("head.pair_jepa.context_predictor")
                   for key in meta["gradient_coverage"]["covered"])
    validation = read_jsonl(root / "validation_log.jsonl")
    assert len(validation) == 1 and validation[0]["step"] == 2
    assert validation[0]["validation_loss"] == meta["best_validation_loss"]
    assert math.isfinite(meta["best_validation_loss"])
    ledger = read_jsonl(root / "ledger.jsonl")
    completed = [row for row in ledger if row["status"] == "completed"]
    assert len(completed) == 1
    assert completed[0]["step"] == 2
    assert completed[0]["stop_reason"] == "step_cap"
    assert math.isfinite(completed[0]["final_loss"])
    starts = [row for row in ledger if row["status"] == "start"]
    assert all(row["resume_from"] == 0 and row["steps"] == 2 for row in starts)
    assert not (root / "train_log.jsonl").exists()
    return {
        "latent": latent,
        "n_params": meta["model"]["n_params"],
        "n_learnable_params": meta["model"]["n_learnable_params"],
        "scorer_n_params": sum(value.numel() for value in scorer_keys.values()),
        "best_step": meta["best_step"],
        "best_validation_loss": meta["best_validation_loss"],
        "final_loss": meta["final_loss"],
        "ledger": ledger,
        "validation": validation,
        "environment": meta["environment"],
        "git_commit": meta["git_commit"],
        "data_sha256": meta["data"]["version"],
        "validation_data_sha256": meta["validation"]["data"]["version"],
        "best_resume_weights_equal": True,
        "stale_optimizer_excluded_from_live_run": latent == 490 or (
            (root / "stale_pre_exclusion").is_dir()
            and not (root / "stale_pre_exclusion.json").is_symlink()),
        "hashes": {name: sha256_file(root / name) for name in
                   ("run_meta.json", "best.pt", "resume.pt", "ledger.jsonl",
                    "validation_log.jsonl")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    reports = {str(latent): validate(root / "runs" / f"capacity_probe_latent{latent}_s42", latent)
               for latent in (490, 1024)}
    output = {
        "status": "PASS_CAPACITY_SMOKE_THREE_CHECK_QA",
        "scientific_conclusion": False,
        "evidence_level": "two_step_engineering_smoke_only",
        "reports": reports,
        "limitations": [
            "Two steps cannot diagnose underfitting, convergence, VL0 improvement, or test performance.",
            "The old hidden64/hidden1024 PairMLP control is not a capacity comparison.",
            "The stale pre-exclusion run and its optimizer are preserved but excluded from this QA.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": output["status"], "reports": {
        key: {field: value[field] for field in
              ("latent", "n_params", "scorer_n_params", "best_validation_loss")}
        for key, value in reports.items()}}, indent=2))


if __name__ == "__main__":
    main()
