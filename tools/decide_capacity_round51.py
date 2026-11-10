import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    audits = root / "audits/round51"
    smoke = json.loads((audits / "capacity_smoke_qa.json").read_text())
    error = json.loads((audits / "capacity_error_qa.json").read_text())
    repeat = json.loads((audits / "capacity_error_v3_qa.json").read_text())
    equivalence = json.loads((audits / "capacity_error_repeat_equivalence.json").read_text())
    assert smoke["status"] == "PASS_CAPACITY_SMOKE_THREE_CHECK_QA"
    assert error["status"] == repeat["status"] == "PASS_CAPACITY_ERROR_INDEPENDENT_QA"
    assert equivalence["status"] == "PASS_INDEPENDENT_CUDA_REPLICATION"
    assert error["three_seed_mean"] == repeat["three_seed_mean"]
    assert all(report["train_mean_losses"]["nll"]
               > report["validation_mean_losses"]["nll"]
               for report in error["reports"].values())
    assert all(report["train_mean_losses"]["total"]
               > report["validation_mean_losses"]["total"]
               for report in error["reports"].values())
    output = {
        "status": "PASS_CAPACITY_DIAGNOSTIC_THREE_CHECK_QA",
        "capacity_insufficiency_supported": False,
        "allow_c_wide": False,
        "allow_c_local": False,
        "allow_j_gate": True,
        "allow_pairjepa_training": False,
        "decision_scope": "Capacity diagnostic only; no test-set selection or architecture training.",
        "evidence": {
            "smoke_sha256": sha256_file(audits / "capacity_smoke_qa.json"),
            "error_qa_sha256": sha256_file(audits / "capacity_error_qa.json"),
            "repeat_qa_sha256": sha256_file(audits / "capacity_error_v3_qa.json"),
            "repeat_equivalence_sha256": sha256_file(
                audits / "capacity_error_repeat_equivalence.json"),
            "training_qa_sha256": sha256_file(audits / "loss_optimizer_training_qa.json"),
            "error_three_seed_mean": error["three_seed_mean"],
            "per_seed": {seed: {
                "train_nll": report["train_mean_losses"]["nll"],
                "validation_nll": report["validation_mean_losses"]["nll"],
                "train_total": report["train_mean_losses"]["total"],
                "validation_total": report["validation_mean_losses"]["total"],
            } for seed, report in error["reports"].items()},
            "scorer_parameters": {latent: report["scorer_n_params"]
                                  for latent, report in smoke["reports"].items()},
        },
        "rationale": [
            "All three stable L-length checkpoints have higher sampled TR1c NLL than VL0 NLL; the anticipated training-underfitting gap is absent.",
            "The two-step width smoke validates only CUDA and parameter plumbing, not capacity efficacy.",
            "Existing near-parameter PairMLP/ResNet comparisons did not establish that parameter count is the bottleneck.",
            "Therefore C-wide and C-local are not licensed. This does not prove that the current capacity is theoretically sufficient.",
        ],
        "next_gate": "Implement and independently validate J-gate before any Pair-JEPA training.",
        "unchanged_boundaries": [
            "L-length remains the stable base but has not met the good-performance threshold for CHANRG.",
            "No SOTA, final confidence, long-sequence, or raw-sequence speed claim is made.",
            "PPT remains unchanged until independently reviewed stage results are eligible.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
