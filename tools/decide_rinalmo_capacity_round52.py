import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adjudicate(training, diagnostics, replication, error_reports, representation_reports):
    assert training["status"] == "PASS_L_O_TRAINING_THREE_CHECK_QA"
    assert training["allow_capacity_diagnostic"] and len(training["runs"]) == 6
    assert diagnostics["status"] == "PASS_ALL_L_O_ERROR_AND_REPRESENTATION_PROBES"
    assert replication["status"] == "PASS_RINALMO_CAPACITY_INDEPENDENT_CUDA_REPLICATION"
    assert replication["equivalent"]
    assert set(replication["arms"]) == {"length_s42", "olr_s42"}
    arms = {}
    for arm in ("length", "olr"):
        errors = [error_reports[f"{arm}_s{seed}"] for seed in (42, 2026, 3407)]
        representations = [representation_reports[f"{arm}_s{seed}"] for seed in (42, 2026, 3407)]
        error_underfit = all(report["train_mean_losses"]["nll"] > report["validation_mean_losses"]["nll"]
                             and report["train_mean_losses"]["total"] > report["validation_mean_losses"]["total"]
                             for report in errors)
        representation_underfit = all(report["train"]["weighted_nll"] > report["validation"]["weighted_nll"]
                                      for report in representations)
        run_reports = [training["runs"][f"{arm}_s{seed}"] for seed in (42, 2026, 3407)]
        relative_spikes = [math.log1p((report["max_vl0"] - report["best_vl0"])
                                      / max(abs(report["best_vl0"]), 1e-12)) for report in run_reports]
        arms[arm] = {
            "error_underfit_all_three": error_underfit,
            "representation_underfit_all_three": representation_underfit,
            "median_log1p_relative_late_spike": statistics.median(relative_spikes),
            "population_sd_best_vl0": statistics.pstdev(report["best_vl0"] for report in run_reports),
            "best_vl0_values": [report["best_vl0"] for report in run_reports],
            "max_vl0_values": [report["max_vl0"] for report in run_reports],
        }
    capacity_supported = all(arm["error_underfit_all_three"] and arm["representation_underfit_all_three"]
                             for arm in arms.values())
    next_base = min(arms, key=lambda arm: (arms[arm]["median_log1p_relative_late_spike"],
                                           arms[arm]["population_sd_best_vl0"], arm))
    return {
        "status": "PASS_RINALMO_CAPACITY_DIAGNOSTIC_THREE_CHECK_QA",
        "scientific_conclusion": False,
        "capacity_insufficiency_supported": capacity_supported,
        "allow_c_wide": capacity_supported, "allow_c_local": capacity_supported,
        "allow_j_gate": not capacity_supported, "allow_pairjepa_training": False,
        "next_base": next_base, "next_base_rule": (
            "Lowest median log1p((max_vl0-best_vl0)/abs(best_vl0)); best-VL0 SD then arm name tie-break; no test metric"),
        "arms": arms,
        "rationale": [
            "C-wide/C-local require both error-objective and frozen-representation underfitting in every seed; no expansion otherwise.",
            "The base choice uses only complete VL0 curves and is not a claim of late-curve stability or structure F1.",
            "NLL-length and NLL-sum absolute losses are never compared directly.",
            "J-mlp remains the necessary independently trained control before J1/fixed/random.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    audits = root / "audits/round52"
    training = json.loads((audits / "rinalmo_loss_optimizer_training_qa.json").read_text())
    diagnostics = json.loads((audits / "rinalmo_capacity_diagnostics_complete.json").read_text())
    replication = json.loads((audits / "rinalmo_capacity_replication.json").read_text())
    error_reports = {}
    representation_reports = {}
    for arm in ("length", "olr"):
        for seed in (42, 2026, 3407):
            error_reports[f"{arm}_s{seed}"] = json.loads(
                (audits / f"rinalmo_capacity_error_{arm}_s{seed}_qa.json").read_text())
            representation_reports[f"{arm}_s{seed}"] = json.loads(
                (audits / f"rinalmo_capacity_representation_{arm}_s{seed}_qa.json").read_text())
    result = adjudicate(training, diagnostics, replication, error_reports, representation_reports)
    result["evidence_sha256"] = {
        "training": sha256_file(audits / "rinalmo_loss_optimizer_training_qa.json"),
        "diagnostics": sha256_file(audits / "rinalmo_capacity_diagnostics_complete.json"),
        "replication": sha256_file(audits / "rinalmo_capacity_replication.json"),
        "error_qa": {key: sha256_file(audits / f"rinalmo_capacity_error_{key}_qa.json")
                     for key in error_reports},
        "representation_qa": {key: sha256_file(audits / f"rinalmo_capacity_representation_{key}_qa.json")
                              for key in representation_reports},
        "code": sha256_file(__file__),
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
