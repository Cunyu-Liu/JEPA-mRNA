import argparse
import json
import statistics
from pathlib import Path

from qa_loss_optimizer_training_round51 import sha256_file
from qa_rinalmo_r0_training_round52 import validate_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    data_sha256 = sha256_file("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
    runs = {}
    for arm, normalization, lr in (("length", "length", 1e-5), ("olr", "sum", 1e-6)):
        for seed in (42, 2026, 3407):
            report = validate_run(root / "runs" / f"round52_rinalmo_{arm}_s{seed}",
                                  seed, data_sha256, arm, normalization, lr)
            runs[f"{arm}_s{seed}"] = report
    aggregates = {}
    for arm in ("length", "olr"):
        values = [runs[f"{arm}_s{seed}"]["best_vl0"] for seed in (42, 2026, 3407)]
        aggregates[arm] = {
            "mean_best_vl0": statistics.mean(values),
            "population_sd_best_vl0": statistics.pstdev(values),
            "best_vl0_values": values,
        }
    output = {
        "status": "PASS_L_O_TRAINING_THREE_CHECK_QA",
        "allow_capacity_diagnostic": True,
        "scientific_conclusion": False,
        "runs": runs,
        "aggregates": aggregates,
        "data_sha256": data_sha256,
        "teacher_manifest_sha256": sha256_file(
            "/root/rna_jepa_data/ss_data/teacher/bprna_tr1c/manifest.json"),
        "limitations": [
            "Natural early stopping and complete late curves are verified; this is not a test-set result.",
            "VL0 values under different NLL normalizations are not direct structure-performance comparisons.",
            "No interaction, capacity, Pair-JEPA, contrastive, D0/D1, or final-speed claim is allowed yet.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": output["status"], "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
